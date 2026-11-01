"""Host-only genome helpers: host dicts, ``repr``, hashing, sympy codegen and drawing.

Every method here is host-only: it converts with ``Tensor.numpy()``/``to_numpy`` and walks
rows in Python, or imports ``networkx``/``matplotlib``/``sympy`` inside the body, so nothing
in this module runs inside an ``evoxc`` trace and ``import tensorneat`` never pulls in a
drawing or code-generation dependency.
The compute contract (:class:`.base.BaseGenome`) mixes this class in, so the public surface
stays on ``DefaultGenome``/``RecurrentGenome`` unchanged; the split only keeps the traced
contract and the host presentation apart.
"""

from typing import Any

import evoxc
import evoxc.numpy as enp
import numpy as np

from ..common import I_INF, State, to_numpy
from ..common.errors import ConfigError
from .utils import GenomeTensors, re_cound_idx

#: Scale applied to a float attribute before it is rounded to an ``int32`` hash word.
#: ``evoxc`` has no float/int bitcast, so :meth:`BaseGenome.hash` quantises instead; the
#: scale is coarse enough that two attributes that differ visibly hash differently and fine
#: enough that gene values stay far inside the ``int32`` range (they are clipped to +-5).
HASH_QUANT_SCALE: int = 1_000_000


def _row_slice(table: evoxc.SymbolicTensor, index: int) -> evoxc.SymbolicTensor:
    """Read one static column of a ``(P, M)`` traced table as a ``(P,)`` vector.

    A slice-then-reshape instead of ``table[:, index]``: ``evoxc``'s reverse-mode rule for an
    axis-dropping integer index has no usable cotangent shape, and the same idiom is used by
    the gene attribute readers.

    Args:
        table: ``(P, M)`` traced tensor.
        index: Static column index in ``[0, M)``.

    Returns:
        ``(P,)``-shaped view of column ``index``.
    """
    pop_size = int(table.shape[0])
    sliced = table[(slice(None), slice(index, index + 1))]
    return evoxc.reshape(sliced, (pop_size,))


class BaseGenomeHost:
    """Host-side surface mixed into :class:`.base.BaseGenome`.

    Nothing here is traced: the methods convert one individual's payload to NumPy/Python
    containers, hash it through evoxc ops for de-duplication, or generate sympy/networkx
    output. They read ``self._split_payload``, the gene objects and the row/key invariant from
    :class:`.base.BaseGenome`, which is also where the abstract ``sympy_func``/``visualize``
    hooks of a genome family live.
    """

    @staticmethod
    def _individual(payload: GenomeTensors) -> GenomeTensors:
        """Host numpy views of one individual's payload (never fed back into a graph).

        Args:
            payload: ``(P, ...)`` payload batch, or already unbatched arrays.

        Returns:
            The four arrays as ``numpy`` arrays without the population axis.

        Raises:
            ConfigError: If the payload holds more than one individual — these helpers convert
                one genome to host dicts, and a batch would be walked through the wrong axis.
        """
        arrays = tuple(np.asarray(to_numpy(value)) for value in payload)
        node_keys = arrays[0]
        if node_keys.ndim > 1:
            if node_keys.shape[0] != 1:
                raise ConfigError(
                    f"this helper takes one individual, got a batch of "
                    f"{node_keys.shape[0]}: slice it first, e.g. "
                    "GenomeTensors(*(array[p : p + 1] for array in payload))"
                )
            arrays = tuple(array[0] for array in arrays)
        return GenomeTensors(*arrays)

    def get_input_idx(self) -> list[int]:
        """Keys and row positions of the input nodes.

        Returns:
            ``[0, 1, ..., num_inputs-1]``; the row/key invariant makes both coincide.
        """
        return list(self.input_idx)

    def get_output_idx(self) -> list[int]:
        """Keys and row positions of the output nodes.

        Returns:
            ``[max_nodes-num_outputs, ..., max_nodes-1]``; the row/key invariant makes both
            coincide.
        """
        return list(self.output_idx)

    def hash(self, genome: GenomeTensors | Any) -> evoxc.SymbolicTensor:
        """Hash every individual's payload for fast de-duplication (in-graph).

        An xor-mix over the ``int32`` key columns.  evoxc has no float/int bitcast, so float
        attributes are quantised (scaled by :data:`HASH_QUANT_SCALE`, rounded, cast to
        ``int32``) before mixing instead of being hashed by bit pattern; NaN padding quantises
        to one sentinel (``0``), so identical genomes hash identically and padding never
        distinguishes two payloads.

        Args:
            genome: ``(P, ...)`` payload batch, a ``GenomeTensors`` or its four arrays.

        Returns:
            ``(P,) uint32`` hashes, one per individual.
        """
        payload, _ = self._split_payload(genome)
        pop_size = int(payload.node_keys.shape[0])
        node_rows = self._row_hashes(
            self.node_gene, payload.node_keys, payload.node_attrs
        )
        conn_rows = self._row_hashes(
            self.conn_gene, payload.conn_keys, payload.conn_attrs
        )
        total = enp.zeros((pop_size,), dtype=evoxc.uint32)
        for index in range(int(node_rows.shape[1])):
            total = evoxc.bitwise_xor(total, _row_slice(node_rows, index))
        for index in range(int(conn_rows.shape[1])):
            total = evoxc.bitwise_xor(total, _row_slice(conn_rows, index))
        return total

    def _row_hashes(
        self,
        gene: Any,
        keys: evoxc.SymbolicTensor,
        attrs: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Hash every gene row of one family: key columns xor quantised attribute columns.

        Args:
            gene: The gene family owning ``hash`` (``node_gene`` or ``conn_gene``).
            keys: ``(P, R)`` or ``(P, R, CK)`` int32 key rows, ``I_INF`` padded.
            attrs: ``(P, R, CA)`` float32 attribute rows, NaN padded.

        Returns:
            ``(P, R) uint32`` row hashes.
        """
        key_rows = (
            keys if len(keys.shape) == 3 else evoxc.reshape(keys, (*keys.shape, 1))
        )
        quantised = enp.where(
            evoxc.isnan(attrs),
            enp.zeros((), dtype=evoxc.float32),
            evoxc.round(
                evoxc.multiply(
                    attrs, enp.full((), float(HASH_QUANT_SCALE), dtype=evoxc.float32)
                )
            ),
        )
        return evoxc.bitwise_xor(
            gene.hash(key_rows), gene.hash(evoxc.cast(quantised, evoxc.int32))
        )

    def network_dict(
        self, state: State, genome: GenomeTensors, whether_re_cound_idx: bool = True
    ) -> dict[str, dict]:
        """Convert one individual to host-side dicts (never traced).

        Host-only: converts with ``Tensor.numpy()`` and walks rows in Python, so the result must
        never be fed back into a graph.  ``genome`` is one individual (unbatched arrays, or a
        batch with ``P == 1``).

        Args:
            state: Host configuration container.
            genome: Payload of one individual.
            whether_re_cound_idx: Renumber hidden keys to be contiguous above
                ``max(input/output keys)`` via ``.utils.re_cound_idx``; input and output keys
                are untouched.

        Returns:
            ``{"nodes": {key: node_dict}, "conns": {(in_key, out_key): conn_dict}}`` from
            ``node_gene.to_dict``/``conn_gene.to_dict``; ``default.py`` adds
            ``topo_order``/``topo_layers``/``useful_nodes``.

        Raises:
            ConfigError: If ``genome`` holds more than one individual.
        """
        payload, _ = self._split_payload(genome)
        individual = self._individual(payload)
        node_keys, node_attrs, conn_keys, conn_attrs = individual
        if whether_re_cound_idx:
            node_keys, node_attrs, conn_keys = re_cound_idx(
                node_keys,
                node_attrs,
                conn_keys,
                self.get_input_idx(),
                self.get_output_idx(),
            )
        return {
            "nodes": self._get_node_dict(state, node_keys, node_attrs),
            "conns": self._get_conn_dict(state, conn_keys, conn_attrs),
        }

    def _get_node_dict(
        self,
        state: State,
        node_keys: Any,
        node_attrs: Any,
    ) -> dict:
        """Map node key -> node dict for one individual (host-only).

        Args:
            state: Host configuration container.
            node_keys: ``(N,)`` node key row of one individual, ``I_INF`` padded.
            node_attrs: ``(N, NA)`` attribute rows of the same individual, NaN padded.

        Returns:
            ``{node_key: node_gene.to_dict(...)}``, padded rows skipped.
        """
        node_dict = {}
        for row in range(node_keys.shape[0]):
            key = int(node_keys[row])
            if key == I_INF:
                continue
            node_dict[key] = self.node_gene.to_dict(
                state, node_keys[row : row + 1], node_attrs[row]
            )
        return node_dict

    def _get_conn_dict(
        self,
        state: State,
        conn_keys: Any,
        conn_attrs: Any,
    ) -> dict:
        """Map connection ``(in_key, out_key)`` -> conn dict for one individual (host-only).

        Args:
            state: Host configuration container.
            conn_keys: ``(E, CK)`` connection key rows of one individual, ``I_INF`` padded.
            conn_attrs: ``(E, CA)`` attribute rows of the same individual, NaN padded.

        Returns:
            ``{(in_key, out_key): conn_gene.to_dict(...)}``, padded rows skipped.
        """
        conn_dict = {}
        for row in range(conn_keys.shape[0]):
            if int(conn_keys[row, 0]) == I_INF:
                continue
            converted = self.conn_gene.to_dict(
                state, conn_keys[row], conn_attrs[row]
            )
            conn_dict[(converted["in"], converted["out"])] = converted
        return conn_dict

    def repr(self, state: State, genome: GenomeTensors, precision: int = 2) -> str:
        """Render one individual as a multi-line string (host-only).

        Args:
            state: Host configuration container.
            genome: Payload of one individual.
            precision: Decimal places forwarded to the gene ``repr`` hooks.

        Returns:
            ``"<ClassName>(nodes=<count>, conns=<count>):"`` then one line per node (annotated
            ``(input)``/``(output)``) and one line per connection; padded rows contribute no
            line at all.

        Raises:
            ConfigError: If ``genome`` holds more than one individual.
        """
        payload, _ = self._split_payload(genome)
        individual = self._individual(payload)
        node_keys, node_attrs, conn_keys, conn_attrs = individual
        present_nodes = [
            row for row in range(node_keys.shape[0]) if int(node_keys[row]) != I_INF
        ]
        present_conns = [
            row for row in range(conn_keys.shape[0]) if int(conn_keys[row, 0]) != I_INF
        ]
        inputs, outputs = set(self.get_input_idx()), set(self.get_output_idx())
        lines = [
            f"{type(self).__name__}(nodes={len(present_nodes)}, "
            f"conns={len(present_conns)}):"
        ]
        for row in present_nodes:
            text = self.node_gene.repr(
                state, node_keys[row : row + 1], node_attrs[row], precision
            )
            key = int(node_keys[row])
            if key in inputs:
                text += " (input)"
            elif key in outputs:
                text += " (output)"
            lines.append(text)
        for row in present_conns:
            lines.append(
                self.conn_gene.repr(state, conn_keys[row], conn_attrs[row], precision)
            )
        return "\n".join(lines) + "\n"

    def sympy_func(
        self,
        state: State,
        network: dict,
        sympy_input_transform: Any = None,
        sympy_output_transform: Any = None,
        backend: str = "numpy",
    ) -> tuple:
        """Build the sympy expression of a network and its NumPy evaluator (host-only).

        Declared here, implemented by the concrete genomes; the tuple is the contract with
        ``..common.sympy_tools``.

        Args:
            state: Host configuration container.
            network: Output of :meth:`network_dict`.
            sympy_input_transform: Sympy counterpart of ``input_transform``.
            sympy_output_transform: Sympy counterpart of ``output_transform``.
            backend: Codegen backend; ``"numpy"`` only in this rewrite (no JAX).

        Returns:
            ``(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func,
            topo_order, useful_nodes)`` — the legacy 8-tuple.

        Raises:
            NotImplementedError: Always — concrete genomes implement it.
        """
        raise NotImplementedError(
            "concrete genomes return the 8-tuple consumed by common.sympy_tools"
        )

    def visualize(self, network: dict, **kwargs: Any) -> None:
        """Draw a network (host-only; networkx/matplotlib imported inside the body).

        Declared here, implemented by the concrete genomes, so ``import tensorneat`` never
        imports ``networkx``/``matplotlib``.

        Args:
            network: Output of :meth:`network_dict`.
            **kwargs: Per-genome drawing options.

        Returns:
            ``None``; the figure goes to the path chosen by the concrete genome.

        Raises:
            NotImplementedError: Always — concrete genomes implement it.
        """
        raise NotImplementedError("concrete genomes draw the network and save the figure")
