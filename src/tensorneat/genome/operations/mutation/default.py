"""NEAT-python style mutation: at most one structural change plus value mutation.

All four structural operations are batch-wide kernels: each gate is an elementwise mask on
the population axis, and an individual whose gate is off — or whose candidate is invalid —
comes out unchanged.  Capacity invariants are hard: ``max_nodes``, ``max_conns`` and the
genome's ``max_in_degree`` are never exceeded, a refused add is a no-op for that individual
(never a silent repair), and new rows go to ``first_free_row`` in the same layout for keys
and attributes.

The gate draw, the operation ordering and the value mutation are shared and live in
:class:`~.base.BaseMutation`; this module supplies the four probabilities and the four
per-operation helpers.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import I_INF, ConfigError, State, masked_flat_gather
from ...utils import (
    GenomeTensors,
    add_conn,
    add_node,
    delete_conn_by_pos,
    delete_conns_by_key,
    delete_node_by_pos,
    fetch_random_index,
    first_free_row,
    valid_count,
)
from .base import BaseMutation, _broadcast_row, _gated, _int32

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome


def _edge_keys(
    src_keys: evoxc.SymbolicTensor,
    dst_keys: evoxc.SymbolicTensor,
    new_conn_keys: evoxc.SymbolicTensor,
    marker_column: int,
    genome: "BaseGenome",
) -> evoxc.SymbolicTensor:
    """Build the ``(P, CK)`` key row of one new connection.

    The key row is ``(in_key, out_key)`` plus the ``historical_marker`` column when — and
    only when — the connection gene declares one.  The marker is not a node reference but
    an identity, so it comes from the caller-allocated innovation column of
    ``new_conn_keys``; the branch is a static Python ``if`` on the gene's specification,
    because a ``CK`` of ``2`` and one of ``3`` are different static shapes.

    Args:
        src_keys: ``(P,) int32`` source node keys.
        dst_keys: ``(P,) int32`` destination node keys.
        new_conn_keys: ``(P, 3) int32`` allocated connection keys.
        marker_column: Column of ``new_conn_keys`` used as the ``historical_marker``.
        genome ("BaseGenome"): Genome whose ``conn_gene`` decides whether the column exists.

    Returns:
        ``(P, CK) int32`` key rows, ``CK`` being ``conn_gene.key_width``.
    """
    columns = [src_keys, dst_keys]
    if genome.conn_gene.uses_historical_marker:
        columns.append(
            evoxc.reshape(
                new_conn_keys[:, marker_column : marker_column + 1],
                (new_conn_keys.shape[0],),
            )
        )
    return enp.stack(columns, axis=-1)


def _attribute_row(
    attrs: evoxc.SymbolicTensor, rows: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Gather one attribute row per individual out of a ``(P, M, W)`` table.

    Used by the node split, which has to carry the attributes of the connection it deletes
    over to its copy edge.  Rows are resolved through flat indices because
    ``evoxc.gather`` replaces the indexed axis instead of indexing it per row, and the
    ``I_INF`` "no row" sentinel is redirected to ``0`` *after* the mask is applied, so the
    backend never sees it.

    Args:
        attrs: ``(P, M, W) float32`` attribute table.
        rows: ``(P,) int32`` row per individual, ``I_INF`` for "absent" (yielding NaN).

    Returns:
        ``(P, W) float32`` attribute rows, NaN where ``rows`` is ``I_INF``.
    """
    pop_size, row_count, width = attrs.shape
    present = evoxc.not_equal(rows, _int32(I_INF))
    safe = evoxc.select(present, rows, _int32(0))
    columns = enp.reshape(safe, (pop_size, 1)) * _int32(width) + enp.arange(
        0, width, dtype=evoxc.int32
    )
    return masked_flat_gather(
        enp.reshape(attrs, (pop_size, row_count * width)),
        columns,
        mask=enp.reshape(present, (pop_size, 1)),
        fill=float("nan"),
    )


class DefaultMutation(BaseMutation):
    """Structural and value mutation with per-operation probabilities.

    The four probabilities are independent per individual and per generation, but at most
    one structural mutation is applied per individual: one shared
    ``evoxc.random.uniform(key, (4, P))`` draw gates all of them and the first operation
    that hits wins.  A probability of ``0`` therefore removes its operation from the
    draw's reach entirely (an all-``False`` row).

    Args:
        conn_add (float): Probability per individual of attempting to add one connection.
        conn_delete (float): Probability per individual of attempting to delete a connection.
        node_add (float): Probability per individual of attempting a node split.
        node_delete (float): Probability per individual of attempting to delete a hidden node.

    Raises:
        TensorNEATError: If any probability is not a real number in ``[0, 1]``.
    """

    def __init__(
        self,
        conn_add: float = 0.2,
        conn_delete: float = 0.2,
        node_add: float = 0.1,
        node_delete: float = 0.1,
    ):
        """Store the four probabilities (their meaning is documented on the class).

        Validating here means a bad configuration fails at construction time, not at the
        first ``evoxc.build``.

        Args:
            conn_add (float): Probability of attempting to add a connection.
            conn_delete (float): Probability of attempting to delete a connection.
            node_add (float): Probability of attempting a node split.
            node_delete (float): Probability of attempting to delete a hidden node.

        Raises:
            TensorNEATError: If any probability is not a real number in ``[0, 1]``.
        """
        self.conn_add = self._probability(conn_add, "conn_add")
        self.conn_delete = self._probability(conn_delete, "conn_delete")
        self.node_add = self._probability(node_add, "node_add")
        self.node_delete = self._probability(node_delete, "node_delete")

    @staticmethod
    def _probability(value: float, name: str) -> float:
        """Validate one structural probability.

        Args:
            value (float): The configured probability.
            name (str): Attribute name, used in the error message.

        Returns:
            float: ``value`` as a Python ``float``.

        Raises:
            ConfigError: If ``value`` is not a real number in ``[0, 1]`` (a ``bool`` is
                rejected too: it is a configuration mistake, not a probability).
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"{name} must be a real number in [0, 1], got {value!r}"
            )
        probability = float(value)
        if not 0.0 <= probability <= 1.0:
            raise ConfigError(
                f"{name} must be a real number in [0, 1], got {probability!r}"
            )
        return probability

    def _mutate_add_node(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        new_node_keys: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
        gate: evoxc.SymbolicTensor,
    ) -> GenomeTensors:
        """Split one connection per individual with a new node, preserving network output.

        The chosen connection is deleted, the new node is written at ``first_free_row`` with
        the node gene's identity attributes, then two connections are added: an identity one
        from the old source (``conn_gene.new_identity_attrs``) and a copy of the original one
        into the old destination (with the recorded original attributes).

        No-op for an individual when ``gate`` is false, when :meth:`choose_connection_key`
        finds no connection (``I_INF``), when ``first_free_row`` finds no free node row, or
        when fewer than two connection rows are free.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing the identity attribute factories.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this helper.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs (NaN padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs (NaN padded).
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated conn keys;
                ``[:, 0]``/``[:, 1]`` become the ``historical_marker`` of the identity/copy
                edge when the conn gene declares one.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected the
                add-node operation" mask from the shared gate draw.

        Returns:
            GenomeTensors: the four payload arrays; an individual that cannot split is unchanged.
        """
        pop_size = node_keys.shape[0]
        edge_count = conn_keys.shape[1]
        in_keys, out_keys, rows = self.choose_connection_key(key, conn_keys)
        original_attrs = _attribute_row(conn_attrs, rows)
        node_row = first_free_row(node_keys)
        # Two free connection rows are needed: the split deletes one and writes two.
        free_conns = _int32(edge_count) - valid_count(conn_keys)
        splittable = evoxc.logical_and(
            evoxc.logical_and(
                gate, evoxc.not_equal(rows, _int32(I_INF))
            ),
            evoxc.logical_and(
                evoxc.not_equal(node_row, _int32(I_INF)),
                evoxc.greater_equal(free_conns, _int32(2)),
            ),
        )
        split_keys, split_attrs = delete_conn_by_pos(conn_keys, conn_attrs, rows)
        split_node_keys, split_node_attrs = add_node(
            node_keys,
            node_attrs,
            node_row,
            new_node_keys,
            _broadcast_row(genome.node_gene.new_identity_attrs(state), pop_size),
        )
        identity_row = first_free_row(split_keys)
        split_keys, split_attrs = add_conn(
            split_keys,
            split_attrs,
            identity_row,
            _edge_keys(in_keys, new_node_keys, new_conn_keys, 0, genome),
            _broadcast_row(genome.conn_gene.new_identity_attrs(state), pop_size),
        )
        # The second row comes from the post-first-write key array, so the two new
        # connection rows can never be the same one.
        copy_row = first_free_row(split_keys)
        split_keys, split_attrs = add_conn(
            split_keys,
            split_attrs,
            copy_row,
            _edge_keys(new_node_keys, out_keys, new_conn_keys, 1, genome),
            original_attrs,
        )
        return GenomeTensors(
            _gated(splittable, split_node_keys, node_keys),
            _gated(splittable, split_node_attrs, node_attrs),
            _gated(splittable, split_keys, conn_keys),
            _gated(splittable, split_attrs, conn_attrs),
        )

    def _mutate_delete_node(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        gate: evoxc.SymbolicTensor,
    ) -> GenomeTensors:
        """Delete one hidden node per individual together with all its connections.

        Input rows (``0..num_inputs-1``) and output rows (``N-num_outputs..N-1``) are never
        candidates, so :meth:`choose_node_key` is called with both ``allow_*_keys=False``;
        the node row is cleared with ``delete_node_by_pos`` and every connection touching the
        deleted key is cleared with ``delete_conns_by_key``.

        No-op when ``gate`` is false or when no hidden node exists
        (:meth:`choose_node_key` returns ``I_INF``).

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing the input/output row ranges.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this helper.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs (NaN padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs (NaN padded).
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected the
                delete-node operation" mask from the shared gate draw.

        Returns:
            GenomeTensors: the four payload arrays, with the deleted node's row and every touching
            connection row sent back to their padding sentinels.
        """
        deleted_keys, rows = self.choose_node_key(
            key,
            node_keys,
            genome.num_inputs,
            genome.num_outputs,
            allow_input_keys=False,
            allow_output_keys=False,
        )
        deletable = evoxc.logical_and(gate, evoxc.not_equal(rows, _int32(I_INF)))
        deleted_node_keys, deleted_node_attrs = delete_node_by_pos(
            node_keys, node_attrs, rows
        )
        deleted_conn_keys, deleted_conn_attrs = delete_conns_by_key(
            conn_keys, conn_attrs, deleted_keys
        )
        return GenomeTensors(
            _gated(deletable, deleted_node_keys, node_keys),
            _gated(deletable, deleted_node_attrs, node_attrs),
            _gated(deletable, deleted_conn_keys, conn_keys),
            _gated(deletable, deleted_conn_attrs, conn_attrs),
        )

    def _mutate_add_conn(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
        gate: evoxc.SymbolicTensor,
    ) -> GenomeTensors:
        """Add one connection per individual with the connection gene's zero attributes.

        Two independent :meth:`choose_node_key` draws pick the endpoints: the source may be
        any existing node (``allow_input_keys=True, allow_output_keys=True``), the
        destination any existing node that is not an input (``allow_input_keys=False``).

        No-op for an individual when ``gate`` is false, when an endpoint is ``I_INF``, when
        ``first_free_row`` finds no free connection row, when the ``(in_key, out_key)`` pair
        already exists, when the edge would close a cycle and ``genome.network_type ==
        "feedforward"`` (the genome's ``check_cycles`` predicate, equivalent to the legacy
        ``common.graph.check_cycles`` reachability test), or when it would push the
        destination's in-degree past ``genome.max_in_degree``.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing ``max_in_degree``, ``network_type`` and
                ``conn_gene.new_zero_attrs``.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this helper.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs (NaN padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs (NaN padded).
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated conn keys;
                ``[:, 2]`` becomes the ``historical_marker`` of the new edge when declared.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected the
                add-connection operation" mask from the shared gate draw.

        Returns:
            GenomeTensors: the four payload arrays; a refused add leaves the individual unchanged.
        """
        pop_size = node_keys.shape[0]
        source_key, destination_key = evoxc.random.split_n(key, 2)
        in_keys, _ = self.choose_node_key(
            source_key,
            node_keys,
            genome.num_inputs,
            genome.num_outputs,
            allow_input_keys=True,
            allow_output_keys=True,
        )
        out_keys, _ = self.choose_node_key(
            destination_key,
            node_keys,
            genome.num_inputs,
            genome.num_outputs,
            allow_input_keys=False,
            allow_output_keys=True,
        )
        conn_row = first_free_row(conn_keys)
        present = evoxc.not_equal(conn_keys[:, :, 0], _int32(I_INF))
        in_column = enp.reshape(in_keys, (pop_size, 1))
        out_column = enp.reshape(out_keys, (pop_size, 1))
        duplicate = evoxc.reduce_max(
            evoxc.logical_and(
                present,
                evoxc.logical_and(
                    evoxc.equal(conn_keys[:, :, 0], in_column),
                    evoxc.equal(conn_keys[:, :, 1], out_column),
                ),
            ),
            axes=1,
        )
        in_degree = evoxc.cast(
            evoxc.reduce_sum(
                evoxc.cast(
                    evoxc.logical_and(
                        present, evoxc.equal(conn_keys[:, :, 1], out_column)
                    ),
                    evoxc.int32,
                ),
                axes=1,
            ),
            evoxc.int32,
        )
        allowed = evoxc.logical_and(
            evoxc.logical_and(
                gate,
                evoxc.logical_and(
                    evoxc.not_equal(in_keys, _int32(I_INF)),
                    evoxc.not_equal(out_keys, _int32(I_INF)),
                ),
            ),
            evoxc.logical_and(
                evoxc.not_equal(conn_row, _int32(I_INF)),
                evoxc.logical_not(duplicate),
            ),
        )
        allowed = evoxc.logical_and(
            allowed, evoxc.less(in_degree, _int32(genome.max_in_degree))
        )
        if genome.network_type == "feedforward":
            allowed = evoxc.logical_and(
                allowed,
                evoxc.logical_not(
                    genome.check_cycles(node_keys, conn_keys, in_keys, out_keys)
                ),
            )
        added_keys, added_attrs = add_conn(
            conn_keys,
            conn_attrs,
            conn_row,
            _edge_keys(in_keys, out_keys, new_conn_keys, 2, genome),
            _broadcast_row(genome.conn_gene.new_zero_attrs(state), pop_size),
        )
        return GenomeTensors(
            node_keys,
            node_attrs,
            _gated(allowed, added_keys, conn_keys),
            _gated(allowed, added_attrs, conn_attrs),
        )

    def _mutate_delete_conn(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        gate: evoxc.SymbolicTensor,
    ) -> GenomeTensors:
        """Delete one connection per individual.

        The candidate row comes from :meth:`choose_connection_key` and is cleared with
        ``delete_conn_by_pos``, which restores both padding sentinels (``I_INF`` keys, NaN
        attributes) while leaving every other row bit-identical.

        No-op when ``gate`` is false or when there is no connection to delete (``I_INF``
        row).

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning the connection gene.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this helper.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs (NaN padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs (NaN padded).
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected the
                delete-connection operation" mask from the shared gate draw.

        Returns:
            GenomeTensors: the four payload arrays, with the chosen connection row cleared.
        """
        _, _, rows = self.choose_connection_key(key, conn_keys)
        deletable = evoxc.logical_and(gate, evoxc.not_equal(rows, _int32(I_INF)))
        deleted_keys, deleted_attrs = delete_conn_by_pos(conn_keys, conn_attrs, rows)
        return GenomeTensors(
            node_keys,
            node_attrs,
            _gated(deletable, deleted_keys, conn_keys),
            _gated(deletable, deleted_attrs, conn_attrs),
        )

    def choose_node_key(
        self,
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        num_inputs: int,
        num_outputs: int,
        allow_input_keys: bool = False,
        allow_output_keys: bool = False,
    ) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
        """Pick one candidate node per individual, out of the eligible rows.

        Eligibility is an elementwise mask over ``node_keys``: absent rows (``I_INF``) are
        dropped, and input rows (``0..num_inputs-1``) / output rows
        (``N-num_outputs..N-1``) are dropped unless the matching flag is ``True``.  The draw
        is a single batched ``fetch_random_index`` over the ``(P, N)`` mask — never a Python
        loop over individuals.

        The key and the row come from that one draw, so they always agree: the returned key
        is the key *at* the returned row, never a second, independent sample.

        Args:
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this draw.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            num_inputs (int): ``I``, the number of input rows ``0..I-1`` (static).
            num_outputs (int): ``O``, the number of output rows ``N-O..N-1`` (static).
            allow_input_keys (bool): Whether input rows may be chosen.
            allow_output_keys (bool): Whether output rows may be chosen.

        Returns:
            tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]: ``(keys, rows)``, both
            ``(P,) int32``, with ``I_INF`` in both entries where no candidate exists.
        """
        _, node_count = node_keys.shape
        present = evoxc.not_equal(node_keys, _int32(I_INF))
        rows = enp.reshape(enp.arange(0, node_count, dtype=evoxc.int32), (1, node_count))
        if not allow_input_keys:
            present = evoxc.logical_and(present, evoxc.greater_equal(rows, _int32(num_inputs)))
        if not allow_output_keys:
            present = evoxc.logical_and(
                present, evoxc.less(rows, _int32(node_count - num_outputs))
            )
        chosen = fetch_random_index(key, present)
        keys = masked_flat_gather(
            node_keys,
            chosen,
            mask=evoxc.not_equal(chosen, _int32(I_INF)),
            fill=I_INF,
        )
        return keys, chosen

    def choose_connection_key(
        self,
        key: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
    ) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
        """Pick one candidate connection per individual, out of the existing rows.

        One ``fetch_random_index`` draw over the ``(P, E)`` presence mask resolves the row,
        and the ``(in_key, out_key)`` pair is gathered from that same row — so the caller
        gets a consistent ``(in_key, out_key, row)`` triple for the price of one draw, and
        absent rows (``I_INF``) are never candidates.

        Args:
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this draw.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).

        Returns:
            tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
            ``(in_keys, out_keys, rows)``, all ``(P,) int32``, with ``I_INF`` in all three
            entries for individuals that have no connection.
        """
        rows = fetch_random_index(key, evoxc.not_equal(conn_keys[:, :, 0], _int32(I_INF)))
        present = evoxc.not_equal(rows, _int32(I_INF))
        return (
            masked_flat_gather(conn_keys[:, :, 0], rows, mask=present, fill=I_INF),
            masked_flat_gather(conn_keys[:, :, 1], rows, mask=present, fill=I_INF),
            rows,
        )


__all__ = ["DefaultMutation"]
