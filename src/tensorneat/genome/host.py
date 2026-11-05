"""Host-only genome helpers: host dicts, ``repr``, hashing, gradients and drawing.

Every method here is host-only: it converts with ``Tensor.numpy()``/``to_numpy`` and walks
rows in Python, or imports ``networkx``/``matplotlib``/``sympy`` inside the body, so nothing
in this module runs inside an ``evoxc`` trace and ``import tensorneat`` never pulls in a
drawing or code-generation dependency.
``hash`` is host-only too: it folds numpy host rows with
:func:`..common.tools.hash_rows` (a ``uint32`` mix over ``int32`` words), which has no
in-graph form because ``evoxc`` has no bitcast -- a traced payload is therefore refused
loudly instead of being silently reinterpreted.
The compute contract (:class:`.base.BaseGenome`) mixes this class in, so the public surface
stays on ``DefaultGenome``/``RecurrentGenome`` unchanged; the split only keeps the traced
contract and the host presentation apart.

## Gradient descent through evolved topologies

Exact gradients through an evolved topology are two steps: derive the execution schedule,
then differentiate the pass built on it.  :meth:`BaseGenome.grad` takes an **already-derived**
``Transformed`` because the schedule derivation (``argsort``/``sort``) has no AD rule and must
stay outside the differentiated graph.
:meth:`BaseGenomeHost.grad_from_payload` is the convenience wrapper for that recipe: it
derives the schedule on the host with a one-off executable around ``transform``, builds the
gradient graph, compiles it and runs it, returning ``(node_attr_grads, conn_attr_grads)``::

    node_grads, conn_grads = genome.grad_from_payload(state, payload, inputs, loss_fn)

A caller that already holds a schedule (a cached one-off executable, a repeated call on the
same payload) derives it once and calls ``genome.grad(state, transformed, *arrays, inputs,
loss_fn)`` directly, then lowers/compiles/loads the returned ``evoxc.Graph`` itself -- a
fresh graph is traced per ``grad`` call, so repeated use must cache it.
Only the float attribute tensors are differentiated; the key tensors and the schedule are
non-differentiated inputs.  The loss is scalarized as ``sum(loss_fn(outputs))`` and the
population axis is independent of that sum, so per-individual gradients are unchanged.
``evoxc.grad`` returns gradients only, so the loss *value* is never part of either call --
read it from ``forward`` with the same ``loss_fn``.
"""

from typing import Any

import evoxc
import numpy as np

from ..common import I_INF, State, to_numpy
from ..common.errors import ConfigError, TensorNEATError
from ..common.tools import hash_rows
from .utils import GenomeTensors, re_cound_idx

#: Scale applied to a float attribute before it is rounded to an ``int32`` hash word.
#: ``evoxc`` has no float/int bitcast, so :meth:`BaseGenomeHost.hash` quantises instead.
#: A quantised magnitude reaches the ``int32`` range at ``|attr| ~= 2147.48`` and wraps
#: beyond it, so the hash is a **heuristic de-duplication key**: it is not injective for
#: attributes of magnitude above ~2.1e3, and it is never an identity test.
HASH_QUANT_SCALE: int = 1_000_000

#: Payload field names in ``GenomeTensors`` order, for spelling/rank error messages.
_PAYLOAD_FIELDS: tuple[str, ...] = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: Accepted ranks per payload field, for each of the two spellings.
_BATCHED_RANKS = {"node_keys": 2, "node_attrs": 3, "conn_keys": 3, "conn_attrs": 3}
_BARE_RANKS = {"node_keys": 1, "node_attrs": 2, "conn_keys": 2, "conn_attrs": 2}
_BATCHED_SHAPES = {
    "node_keys": "(1, max_nodes)",
    "node_attrs": "(1, max_nodes, attr_width)",
    "conn_keys": "(1, max_conns, key_width)",
    "conn_attrs": "(1, max_conns, attr_width)",
}
_BARE_SHAPES = {
    "node_keys": "(max_nodes,)",
    "node_attrs": "(max_nodes, attr_width)",
    "conn_keys": "(max_conns, key_width)",
    "conn_attrs": "(max_conns, attr_width)",
}
#: How each spelling is named in an error message.
_SPELLINGS = (
    "a one-individual batch -- node_keys (1, max_nodes), node_attrs (1, max_nodes, "
    "attr_width), conn_keys (1, max_conns, key_width), conn_attrs (1, max_conns, "
    "attr_width), i.e. state.pop/best_genome at P = 1 -- or the bare per-individual "
    "rows -- node_keys (max_nodes,), node_attrs (max_nodes, attr_width), conn_keys "
    "(max_conns, key_width), conn_attrs (max_conns, attr_width)"
)


def _stage_options(
    backend: str,
    device: Any,
    compile_options: Any,
    *,
    with_device: bool,
) -> dict[str, Any]:
    """The keyword arguments one ``evoxc`` stage of :meth:`grad_from_payload` receives.

    Nothing is defaulted or rewritten: ``backend`` always, ``device`` when it is not
    ``None`` and the stage takes one, and every ``compile_options`` entry expanded into
    ``**options`` (a backend that cannot accept one raises its own error instead of
    falling back).

    Args:
        backend: ``evoxc`` backend name.
        device: Backend device, or ``None`` for the backend default.
        compile_options: Mapping expanded into the stage's options, or ``None``.
        with_device: ``False`` for ``lower``/``compile``, which take no ``device``.

    Returns:
        The options for that stage.
    """
    options: dict[str, Any] = {"backend": backend}
    if with_device and device is not None:
        options["device"] = device
    if compile_options:
        options.update(dict(compile_options))
    if not with_device:
        options.pop("device", None)
    return options


class BaseGenomeHost:
    """Host-side surface mixed into :class:`.base.BaseGenome`.

    Nothing here is traced: the methods convert a payload to NumPy/Python containers, fold
    it through numpy bit views for de-duplication, derive and run a gradient graph on the
    host, or generate sympy/networkx output. They read ``self._split_payload``, the gene
    objects and the row/key invariant from :class:`.base.BaseGenome`, which is also where the
    abstract ``sympy_func``/``visualize`` hooks of a genome family live.
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

    def _one_individual_payload(self, genome: Any) -> GenomeTensors:
        """Normalise a payload argument to one individual's four host rows.

        Both documented spellings are accepted, so a hand-built row and a one-individual
        batch (the spelling ``Pipeline.auto_run`` returns and a checkpoint stores) reach the
        host helpers the same way:

        * **batch-first** with a leading population axis of 1 — ``node_keys (1, max_nodes)``,
          ``node_attrs (1, max_nodes, attr_width)``, ``conn_keys (1, max_conns, key_width)``,
          ``conn_attrs (1, max_conns, attr_width)``; or
        * the **bare per-individual rows** — ``node_keys (max_nodes,)``,
          ``node_attrs (max_nodes, attr_width)``, ``conn_keys (max_conns, key_width)``,
          ``conn_attrs (max_conns, attr_width)``.

        Args:
            genome: A ``GenomeTensors`` (or any four-element sequence of the four payload
                arrays) in either spelling; leaves may be ``evoxc.Tensor``.

        Returns:
            ``GenomeTensors`` of the four bare rows, converted to numpy, with no population
            axis.

        Raises:
            ConfigError: If the payload is missing (``None``), is not four non-scalar
                arrays, holds a batch of more than one individual, or if the four fields
                disagree about the spelling or about the population axis.
        """
        if genome is None:
            raise ConfigError(
                "no genome payload was given (got None); pass " + _SPELLINGS
            )
        fields = list(genome) if isinstance(genome, (tuple, list)) else [genome]
        if len(fields) != 4:
            raise ConfigError(
                "a genome payload is the four arrays (node_keys, node_attrs, conn_keys, "
                f"conn_attrs), got {len(fields)} array(s) of {type(genome).__name__}; "
                "pass " + _SPELLINGS
            )
        arrays = tuple(np.asarray(to_numpy(field)) for field in fields)
        node_keys, node_attrs, conn_keys, conn_attrs = arrays
        if any(array.ndim == 0 for array in arrays):
            raise ConfigError(
                "a genome payload field is rank-0; pass " + _SPELLINGS
            )
        if node_keys.ndim > 2:
            raise ConfigError(
                f"payload field node_keys is rank-{node_keys.ndim} with shape "
                f"{node_keys.shape}; pass " + _SPELLINGS
            )
        if conn_keys.ndim not in (2, 3):
            raise ConfigError(
                f"payload field conn_keys is rank-{conn_keys.ndim} with shape "
                f"{conn_keys.shape}; pass " + _SPELLINGS
            )

        # ``node_keys`` decides the spelling: a bare row is rank 1, a batch-first payload
        # is rank 2.  Every other field must agree with it.
        batched = node_keys.ndim == 2
        ranks = _BATCHED_RANKS if batched else _BARE_RANKS
        shapes = _BATCHED_SHAPES if batched else _BARE_SHAPES
        for name, array in zip(_PAYLOAD_FIELDS, arrays):
            if array.ndim != ranks[name]:
                raise ConfigError(
                    f"payload field {name} is rank-{array.ndim} with shape {array.shape}; "
                    f"expected {shapes[name]} (the four fields are either all batch-first "
                    "or all bare per-individual rows), so pass " + _SPELLINGS
                )
        if not batched:
            return GenomeTensors(*arrays)

        population = int(node_keys.shape[0])
        if population != 1:
            raise ConfigError(
                f"the payload holds {population} individuals (node_keys shape "
                f"{node_keys.shape}); this helper takes one individual, so slice it first, "
                "e.g. GenomeTensors(*(array[p : p + 1] for array in payload))"
            )
        leaders = {
            name: int(array.shape[0]) for name, array in zip(_PAYLOAD_FIELDS, arrays)
        }
        if len(set(leaders.values())) != 1:
            raise ConfigError(
                "the four payload arrays disagree about the population axis: "
                f"node_keys {node_keys.shape}, node_attrs {node_attrs.shape}, "
                f"conn_keys {conn_keys.shape}, conn_attrs {conn_attrs.shape}; expected "
                "(P, max_nodes), (P, max_nodes, attr_width), (P, max_conns, key_width), "
                "(P, max_conns, attr_width) with one shared P"
            )
        return GenomeTensors(*(array[0] for array in arrays))

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

    def hash(self, genome: GenomeTensors | Any) -> np.ndarray:
        """Hash every individual's payload for fast de-duplication (host-only).

        A numpy fold over the host rows.  The key columns and the quantised attribute
        columns of every gene row are hashed with :func:`..common.tools.hash_rows` (the
        legacy ``0x9E3779B9`` left-to-right ``uint32`` fold over ``int32`` words), the two
        row hashes of a gene row are xored, the row hashes are xor-folded over the row axis
        -- so a payload whose padding sits on different rows hashes equal -- and the node
        and connection family totals are xored.
        ``evoxc`` has no float/int bitcast, so float attributes are quantised (scaled by
        :data:`HASH_QUANT_SCALE`, rounded, cast to ``int32``) instead of being hashed by bit
        pattern; NaN padding quantises to one sentinel (``0``), so identical genomes hash
        identically and padding never distinguishes two payloads.
        The result is a **heuristic de-duplication key**, not an identity test: it is not
        injective for attribute magnitudes above ~2.1e3 (see :data:`HASH_QUANT_SCALE`).

        Args:
            genome: ``(P, ...)`` batch-first payload, a ``GenomeTensors`` or its four arrays.

        Returns:
            ``(P,) numpy.uint32`` hashes, one per individual.

        Raises:
            TensorNEATError: If the payload is not four batch-first arrays (a bare
                per-individual row has no population axis), or if a leaf is a traced
                ``evoxc.SymbolicTensor``.
        """
        payload, _ = self._split_payload(genome)
        for name, value in zip(payload.field_names, payload):
            if isinstance(value, evoxc.SymbolicTensor):
                raise TensorNEATError(
                    "hash is host-only: it mixes with numpy bit views and has no in-graph "
                    f"form (payload field {name} is a traced SymbolicTensor); call it on "
                    "host numpy arrays, and compare gene identity inside a graph with the "
                    "key columns plus evoxc.equal instead"
                )
        node_hashes = self._family_hashes(payload.node_keys, payload.node_attrs)
        conn_hashes = self._family_hashes(payload.conn_keys, payload.conn_attrs)
        return np.bitwise_xor(node_hashes, conn_hashes)

    @staticmethod
    def _family_hashes(keys: Any, attrs: Any) -> np.ndarray:
        """Xor-fold one gene family's row hashes into one hash per individual.

        Args:
            keys: ``(P, R)`` or ``(P, R, key_width)`` host key rows, ``I_INF`` padded.
            attrs: ``(P, R, attr_width)`` host attribute rows, NaN padded; an
                ``attr_width`` of 0 contributes no attribute word.

        Returns:
            ``(P,) numpy.uint32`` family hashes.
        """
        key_rows = np.asarray(to_numpy(keys))
        if key_rows.ndim == 2:
            key_rows = key_rows[:, :, np.newaxis]
        pop_size, row_count = int(key_rows.shape[0]), int(key_rows.shape[1])
        flat_keys = np.ascontiguousarray(
            key_rows.reshape(pop_size * row_count, int(key_rows.shape[2]))
        )
        row_hashes = hash_rows(flat_keys).reshape(pop_size, row_count)

        attr_rows = np.asarray(to_numpy(attrs))
        width = int(attr_rows.shape[2])
        if width:
            quantised = np.where(
                np.isnan(attr_rows),
                0.0,
                np.rint(attr_rows * float(HASH_QUANT_SCALE)),
            ).astype(np.int32)
            flat_attrs = np.ascontiguousarray(
                quantised.reshape(pop_size * row_count, width)
            )
            row_hashes = row_hashes ^ hash_rows(flat_attrs).reshape(pop_size, row_count)
        return np.bitwise_xor.reduce(row_hashes, axis=1)

    def network_dict(
        self, state: State, genome: GenomeTensors, whether_re_cound_idx: bool = True
    ) -> dict[str, dict]:
        """Convert one individual to host-side dicts (never traced).

        Host-only: converts with ``Tensor.numpy()`` and walks rows in Python, so the result must
        never be fed back into a graph.  ``genome`` is one individual in either spelling: a
        one-individual batch (``node_keys (1, max_nodes)``, ``node_attrs (1, max_nodes,
        attr_width)``, ``conn_keys (1, max_conns, key_width)``, ``conn_attrs (1, max_conns,
        attr_width)``) or the bare per-individual rows (``node_keys (max_nodes,)``, ...) that
        a checkpoint stores; a payload of ``P > 1`` individuals is refused.

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
            ConfigError: If ``genome`` is missing, is not four field arrays, holds more than
                one individual, or its fields disagree about the spelling/population axis.
        """
        individual = self._individual(self._one_individual_payload(genome))
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

        ``genome`` is one individual in either spelling: a one-individual batch
        (``node_keys (1, max_nodes)``, ``node_attrs (1, max_nodes, attr_width)``, ``conn_keys
        (1, max_conns, key_width)``, ``conn_attrs (1, max_conns, attr_width)``) or the bare
        per-individual rows (``node_keys (max_nodes,)``, ...) that a checkpoint stores; a
        payload of ``P > 1`` individuals is refused.

        Args:
            state: Host configuration container.
            genome: Payload of one individual.
            precision: Decimal places forwarded to the gene ``repr`` hooks.

        Returns:
            ``"<ClassName>(nodes=<count>, conns=<count>):"`` then one line per node (annotated
            ``(input)``/``(output)``) and one line per connection; padded rows contribute no
            line at all.

        Raises:
            ConfigError: If ``genome`` is missing, is not four field arrays, holds more than
                one individual, or its fields disagree about the spelling/population axis.
        """
        individual = self._individual(self._one_individual_payload(genome))
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

    def grad_from_payload(
        self,
        state: State,
        payload: GenomeTensors,
        inputs: Any,
        loss_fn: Any,
        *,
        backend: str = "numpy",
        device: Any = None,
        compile_options: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Differentiate one batch-first payload's network through itself (host-only).

        The convenience wrapper around the two-step recipe of this module: derive the
        schedule on the host with a one-off ``evoxc`` executable around :meth:`transform`
        (the schedule's ``argsort`` has no AD rule, so it may not live inside the
        differentiated graph), build the gradient graph with :meth:`BaseGenome.grad`, then
        lower/compile/load/run it and return the two gradient arrays.

        Args:
            state: Host configuration container from :meth:`setup`.
            payload: The **batch-first** payload of the population — a ``GenomeTensors`` or
                its four arrays, ``node_keys (P, max_nodes)``, ``node_attrs (P, max_nodes,
                attr_width)``, ``conn_keys (P, max_conns, key_width)``, ``conn_attrs (P,
                max_conns, attr_width)``.  Bare per-individual rows are refused: a gradient
                is a population-shaped quantity.
            inputs: ``(P, num_inputs)`` float32 network inputs.
            loss_fn: Callable ``(P, num_outputs) float32 -> (*)`` returning the
                per-individual loss; it is scalarized as ``sum(loss_fn(outputs))``, and the
                population axis is independent of the sum, so per-individual gradients are
                unaffected.  Must be written with a ``multiply`` for a square (evoxc's
                ``power`` pullback emits a ``log`` unconditionally).
            backend: ``evoxc`` backend name for both builds; ``"numpy"`` by default (the
                reference interpreter, no optional dependency).
            device: Backend device forwarded to ``build``/``load``; ``None`` (the backend
                default) is not passed at all.
            compile_options: Mapping expanded into every stage's ``**options``, exactly as
                ``Pipeline.compile_options`` is into ``evoxc.build``'s.

        Returns:
            ``(node_attr_grads, conn_attr_grads)`` as numpy arrays whose shapes and dtypes
            mirror ``node_attrs``/``conn_attrs`` (``float32``).  Padding and absent rows
            gradient to exactly ``0``, because the attribute tensors are NaN-cleaned and
            masked at the graph input.  The loss value is **not** returned (an
            ``evoxc.grad`` graph carries gradients only): call :meth:`BaseGenome.forward`
            with the same ``loss_fn`` if it is needed.

        Raises:
            TensorNEATError: If the payload is not four batch-first arrays, if a payload leaf
                is a traced ``evoxc.SymbolicTensor`` (this helper is host-only), or if
                ``inputs`` does not carry one row per individual.
            ConfigError: If the genome's topology or genes cannot be expressed by the dense
                gradient pass (see :meth:`BaseGenome.grad`).
            ShapeBoundError: If an individual's in-degree exceeds the schedule's static
                ``max_in_degree``.

        Notes:
            A **fresh** graph is traced and compiled per call (like :meth:`BaseGenome.grad`
            itself), so a caller that repeats the call on the same shapes should cache the
            executable itself; only the schedule derivation cost is paid once here.
        """
        fields = payload if isinstance(payload, (tuple, list)) else (payload,)
        payload, _ = self._split_payload(*fields)
        for name, value in zip(payload.field_names, payload):
            if isinstance(value, evoxc.SymbolicTensor):
                raise TensorNEATError(
                    "grad_from_payload is host-only: it derives the schedule and runs the "
                    f"gradient graph on host arrays, but payload field {name} is a traced "
                    "SymbolicTensor; call it outside any trace"
                )
        arrays = tuple(
            np.ascontiguousarray(to_numpy(field)) for field in payload
        )
        inputs = np.ascontiguousarray(to_numpy(inputs))
        population = int(arrays[0].shape[0])
        if inputs.shape[0] != population:
            raise TensorNEATError(
                f"inputs has {int(inputs.shape[0])} row(s) but the payload holds "
                f"{population} individual(s); one input row per individual is required"
            )

        def kernel(*leaves: Any) -> Any:
            """Rebuild the payload from graph data and derive its schedule."""
            return self.transform(state, GenomeTensors(*leaves))

        specs = tuple(
            evoxc.TensorSpec(tuple(int(size) for size in array.shape), array.dtype)
            for array in arrays
        )
        schedule_executable = evoxc.build(
            kernel,
            *specs,
            **_stage_options(backend, device, compile_options, with_device=True),
        )
        transformed = evoxc.run(
            schedule_executable,
            *(evoxc.from_numpy(array) for array in arrays),
        )

        graph = self.grad(state, transformed, *arrays, inputs, loss_fn)
        graph_options = _stage_options(
            backend, device, compile_options, with_device=False
        )
        compiled = evoxc.compile(
            evoxc.lower(graph, **graph_options), **graph_options
        )
        executable = evoxc.load(
            compiled,
            **_stage_options(backend, device, compile_options, with_device=True),
        )
        gradients = evoxc.run(
            executable,
            *(
                evoxc.from_numpy(array)
                for array in (*arrays, np.ascontiguousarray(inputs))
            ),
        )
        return tuple(np.asarray(to_numpy(gradient)) for gradient in gradients)

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
