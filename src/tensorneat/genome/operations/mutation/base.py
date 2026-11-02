"""Mutation contract for the batched NEAT gene payload.

A mutation operator is a trace-time kernel over a whole population: the four payload
arrays enter and leave with identical static shapes, the population lives on axis 0 and
every gate is an elementwise mask — no Python loop over individuals, no ``vmap``, no
``scan``/``while_loop``/``cond``.
Structural keys are caller-allocated ``int32`` tensors, and the only randomness consumed
here is one rank-0 ``evoxc`` key, so a ``(seed, config, shape)`` triple reproduces a run.

The kernel is factored into two stages — :meth:`BaseMutation.mutate_structure` (at most
one structural change per individual) and :meth:`BaseMutation.mutate_values` (attribute
mutation of every existing gene).  Both are implemented here, because both are the same
for every operator family; a subclass only supplies the per-operation helpers
(``_mutate_add_node``, ``_mutate_delete_node``, ``_mutate_add_conn``, ``_mutate_delete_conn``)
and the four probabilities they gate on (:attr:`BaseMutation.structural_probabilities`).
"""

from typing import TYPE_CHECKING

import numpy as np
import evoxc
import evoxc.numpy as enp

from ....common import I_INF, ConfigError, State, StatefulBaseClass
from ...utils import GenomeTensors
from ...window import fill
from .arrays import (
    column_base,
    first_row,
    flatten,
    graph_constant,
    i32 as _int32,
    row_offsets,
)
from .plan import (
    AddConnPlan,
    AddNodePlan,
    DeleteConnPlan,
    DeleteNodePlan,
    Edit,
    EditContext,
    apply,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps genome -> operations acyclic
    from ...base import BaseGenome

#: Number of structural operations the shared gate draw decides between, in the order
#: ``(node_add, node_delete, conn_add, conn_delete)``.  It doubles as the "no operation
#: won" index, which is why it is one past the last row.
_OPERATION_COUNT = 4


def _broadcast_row(row: evoxc.SymbolicTensor, pop_size: int) -> evoxc.SymbolicTensor:
    """Spread one gene attribute row over the population axis.

    The gene factories (``new_identity_attrs``, ``new_zero_attrs``) build a single
    ``(W,)`` row because the values are static, while the payload writers take a
    ``(P, W)`` block; ``broadcast_to`` expresses that in one traced op, where an
    elementwise add against a ``(1, W)`` view would cost a full-size ``zeros`` constant
    plus an ``add``.  ``broadcast_to`` also normalizes a ``(1, W)`` row, so a gene factory
    returning either rank costs the same single op — no intermediate ``reshape``.

    Args:
        row: ``(W,)`` or ``(1, W)`` attribute row of one gene.
        pop_size: Population size ``P`` (static).

    Returns:
        ``(P, W)`` tensor whose every row equals ``row``.
    """
    return enp.broadcast_to(row, (pop_size,) + tuple(row.shape[-1:]))


def _edge_key_row(
    src_keys: evoxc.SymbolicTensor,
    dst_keys: evoxc.SymbolicTensor,
    new_conn_keys: evoxc.SymbolicTensor,
    marker_column: int,
    key_width: int,
    genome: "BaseGenome",
) -> evoxc.SymbolicTensor:
    """Build the ``(P, 1, CK)`` key row of one new connection.

    The row is ``(in_key, out_key)`` plus the ``historical_marker`` column when — and only
    when — the connection gene declares one.  The marker is an identity, not a node
    reference, so it comes from the caller-allocated innovation column of
    ``new_conn_keys``; the branch is a static Python ``if`` on the gene's specification,
    because a ``CK`` of ``2`` and one of ``3`` are different static shapes.

    Args:
        src_keys: ``(P,) int32`` source node keys.
        dst_keys: ``(P,) int32`` destination node keys.
        new_conn_keys: ``(P, 3) int32`` allocated connection keys.
        marker_column: Column of ``new_conn_keys`` used as the ``historical_marker``.
        key_width: ``CK``, the connection gene's key width (static).
        genome ("BaseGenome"): Genome whose ``conn_gene`` decides whether the column exists.

    Returns:
        ``(P, 1, CK) int32`` block holding the single key row.  The row is stacked from the
        ``(P,)`` columns and then treated as a one-row block, which is one ``stack`` plus one
        ``reshape`` instead of one reshape per column plus a concatenate.
    """
    pop_size = src_keys.shape[0]
    columns = [src_keys, dst_keys]
    if genome.conn_gene.uses_historical_marker:
        columns.append(
            enp.reshape(
                evoxc.slice(new_conn_keys, (0, marker_column), (pop_size, 1)),
                (pop_size,),
            )
        )
    return enp.reshape(enp.stack(columns, axis=1), (pop_size, 1, key_width))


def _gated(
    gate: evoxc.SymbolicTensor,
    new: evoxc.SymbolicTensor,
    old: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Keep ``new`` where ``gate`` is true and ``old`` where it is false.

    The writers of ``genome.utils`` are unconditional (a scatter can never be a per-row
    no-op), so every structural helper writes garbage into a redirected row and this
    ``select`` is what turns "refused" into a bit-identical row — ``-0.0`` and ``NaN``
    included, because nothing is recomputed for a refused individual.

    Args:
        gate: ``(P,)`` bool eligibility mask.
        new: Written payload of rank 1, 2 or 3.
        old: Original payload of the same shape as ``new``.

    Returns:
        ``old`` where ``gate`` is false, ``new`` where it is true.
    """
    # ``select`` broadcasts the operands but not a lower-rank predicate, so the ``(P,)``
    # gate is expanded to the payload's rank (``(P, 1)``, ``(P, 1, 1)``, ...).
    mask = enp.reshape(gate, (gate.shape[0],) + (1,) * (len(old.shape) - 1))
    return evoxc.select(mask, new, old)


class BaseMutation(StatefulBaseClass):
    """Batched mutation operator: one structural change plus value mutation.

    Subclasses supply the two things that differ between operator families: the four
    probabilities :attr:`structural_probabilities` (validated by the subclass
    constructor) and the four per-operation helpers ``_mutate_add_node``,
    ``_mutate_delete_node``, ``_mutate_add_conn`` and ``_mutate_delete_conn``, each of
    which takes a trailing ``(P,)`` bool ``gate`` and is a batch-wide no-op where that
    gate is false.  Everything else — the key split, the shared gate draw, the operation
    ordering and the value mutation — lives here.

    Padding conventions are preserved throughout: absent ``int32`` key rows are ``I_INF``
    and absent float attribute rows are NaN, so no operator ever leaks a NaN into a key
    column or an ``I_INF`` into an attribute column.
    """

    def __call__(
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
    ) -> GenomeTensors:
        """Mutate a whole population in one kernel call.

        The key is split internally into one key per stage; :meth:`mutate_structure` runs
        first and :meth:`mutate_values` sees the arrays it produced.  Nothing is applied
        per individual: gates are elementwise masks on axis 0 and genes whose candidate is
        invalid are simply carried through.

        Args:
            state (State): Immutable host configuration container, read for gene metadata
                (attribute bounds, mutation rates); never mutated here.
            genome ("BaseGenome"): Genome owning the node/conn gene objects and the
                static invariants (``max_nodes``, ``max_conns``, ``max_in_degree``,
                ``network_type``).
            key (evoxc.SymbolicTensor): Single rank-0 ``evoxc`` RNG key for this call.
                ``evoxc`` keys are rank-0 only, so independence comes from batched draw
                shapes, not from splitting one key per individual.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node key rows; ``I_INF``
                marks an absent row.  Input keys are rows ``0..num_inputs-1``, output keys
                are rows ``N-num_outputs..N-1`` and are never moved.
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attribute rows;
                absent rows are NaN.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys —
                ``(in_key, out_key)`` plus ``historical_marker`` when the conn gene
                declares it; ``I_INF`` marks an absent row.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection
                attributes (e.g. ``weight``); absent rows are NaN.
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` caller-allocated,
                population-unique keys for the node an add-node mutation may insert.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` caller-allocated,
                population-unique keys for the connections this generation may insert;
                column 0 feeds the add-node identity edge, column 1 the add-node copy edge,
                column 2 the add-conn edge.

        Returns:
            GenomeTensors: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` with the shapes and
            dtypes of the inputs — the mutated population.

        Raises:
            ConfigError: If ``new_node_keys`` is not ``(P,)`` or ``new_conn_keys`` is
                not ``(P, 3)`` for the population size ``P`` of the payload arrays.
        """
        pop_size = node_keys.shape[0]
        if tuple(new_node_keys.shape) != (pop_size,):
            raise ConfigError(
                "new_node_keys must be (P,) for the payload's population size "
                f"{pop_size}, got {tuple(new_node_keys.shape)}"
            )
        if tuple(new_conn_keys.shape) != (pop_size, 3):
            raise ConfigError(
                "new_conn_keys must be (P, 3) for the payload's population size "
                f"{pop_size}, got {tuple(new_conn_keys.shape)}"
            )
        structure_key, values_key = evoxc.random.split_n(key, 2)
        node_keys, node_attrs, conn_keys, conn_attrs = self.mutate_structure(
            state,
            genome,
            structure_key,
            node_keys,
            node_attrs,
            conn_keys,
            conn_attrs,
            new_node_keys,
            new_conn_keys,
        )
        return self.mutate_values(
            state, genome, values_key, node_keys, node_attrs, conn_keys, conn_attrs
        )

    @property
    def structural_probabilities(self) -> tuple[float, float, float, float]:
        """Per-operation probabilities, in gate-draw order.

        The order is ``(node_add, node_delete, conn_add, conn_delete)`` and matches both
        the rows of the shared ``(4, P)`` draw and the order the helpers run in.  A
        subclass owns the four values as instance attributes — it is also the one that
        validates them, so a bad configuration fails at construction time.

        Returns:
            tuple[float, float, float, float]: The four probabilities, each in ``[0, 1]``.
        """
        return (self.node_add, self.node_delete, self.conn_add, self.conn_delete)

    def mutate_structure(
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
    ) -> GenomeTensors:
        """Apply at most one structural change per individual.

        One batched ``evoxc.random.uniform(key, (4, P))`` draw is compared against the
        four :attr:`structural_probabilities`; the *first* row that hit wins
        (``argmax`` plus a hit test, so row ``4`` means "nothing"), which is what keeps
        the change count at one per individual and per generation.  The four helpers then
        run in the order add-node, delete-node, add-conn, delete-conn, each with its own
        key (from one ``split_n``) and its own ``(P,)`` gate, so three of them are
        batch-wide no-ops by construction.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing gene objects and static capacity
                invariants.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key; split into one key per
                operation plus one for the gate draw.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes,
                NaN padded.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys,
                ``I_INF`` padded.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection
                attributes, NaN padded.
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated connection
                keys.

        Returns:
            GenomeTensors: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` after the structural
            stage; every individual whose gate is off is returned unchanged.

        Raises:
            ConfigError: If ``genome.network_type`` is neither ``"feedforward"`` nor
                ``"recurrent"`` — the add-connection helper would otherwise have to guess
                whether a cycle-closing edge is legal.
        """
        if genome.network_type not in ("feedforward", "recurrent"):
            raise ConfigError(
                "mutation needs genome.network_type to be 'feedforward' or 'recurrent', "
                f"got {genome.network_type!r}"
            )
        pop_size = node_keys.shape[0]
        # Four operation keys plus the gate key, in the order the operations run.
        keys = evoxc.random.split_n(key, _OPERATION_COUNT + 1)
        gates = self._structural_gates(keys[_OPERATION_COUNT], pop_size)
        context = self._edit_context(
            genome, node_keys, node_attrs, conn_keys, conn_attrs
        )
        plans = self._plan_operations(
            state,
            genome,
            keys,
            gates,
            context,
            node_keys,
            conn_keys,
            conn_attrs,
            new_node_keys,
            new_conn_keys,
            pop_size,
            conn_attrs.shape[2],
        )
        edit = self._merge_plan(
            state,
            genome,
            plans,
            context,
            new_node_keys,
            new_conn_keys,
            pop_size,
            node_attrs.shape[2],
            conn_keys.shape[2],
            conn_attrs.shape[2],
        )
        return apply(edit, context, node_keys, node_attrs, conn_keys, conn_attrs)

    def mutate_values(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
    ) -> GenomeTensors:
        """Mutate the attributes of every existing gene in the population.

        One batched ``genome.node_gene.mutate`` and one ``genome.conn_gene.mutate`` call
        cover the whole ``(P, N, NA)`` / ``(P, E, CA)`` tables — the genes are batched over
        their leading axes, so no Python loop or per-individual key split is needed.  The
        result is re-masked with the padding pattern read *before* the mutation, so a
        padded row can never become finite.

        Only ``node_attrs`` and ``conn_attrs`` change; the key arrays are returned
        untouched, and padded rows keep their sentinel values (``I_INF`` keys, NaN attrs).

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome whose ``node_gene``/``conn_gene`` perform the
                batched attribute mutation.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this stage.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes to
                mutate, NaN padded.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys,
                ``I_INF`` padded.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes
                to mutate, NaN padded.

        Returns:
            GenomeTensors: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` with mutated
            attribute arrays and unchanged key arrays.
        """
        node_key, conn_key = evoxc.random.split_n(key, 2)
        node_padding = evoxc.isnan(node_attrs)
        conn_padding = evoxc.isnan(conn_attrs)
        mutated_nodes = genome.node_gene.mutate(state, node_key, node_attrs)
        mutated_conns = genome.conn_gene.mutate(state, conn_key, conn_attrs)
        return GenomeTensors(
            node_keys,
            evoxc.select(node_padding, node_attrs, mutated_nodes),
            conn_keys,
            evoxc.select(conn_padding, conn_attrs, mutated_conns),
        )

    def _structural_gates(
        self, key: evoxc.SymbolicTensor, pop_size: int
    ) -> tuple[
        evoxc.SymbolicTensor,
        evoxc.SymbolicTensor,
        evoxc.SymbolicTensor,
        evoxc.SymbolicTensor,
    ]:
        """Resolve which of the four structural operations each individual performs.

        The four probabilities are compared against one ``(4, P)`` uniform draw; the first
        row that hit wins, and an individual for which no row hit gets no structural change
        at all.  Both outcomes are ordinary elementwise results, so the kernel stays
        branch-free (``cond`` is a region op and cannot be batched).

        Args:
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this draw.
            pop_size (int): Population size ``P`` (static).

        Returns:
            tuple: four ``(P,)`` bool gates, in the order add-node, delete-node, add-conn,
            delete-conn; at most one of them is true per individual.

        Notes:
            The comparison runs against a ``(4, 1)`` constant column of static thresholds,
            so a probability of ``0`` produces an all-``False`` row (the operation can never
            win) and a probability of ``1`` an all-``True`` one, without a Python branch.
            The four gates are recovered from the winning row index with four ``equal`` ops
            rather than by slicing a one-hot block, which is cheaper under the reference
            interpreter (one op per gate, no reshape).
        """
        drawn = evoxc.random.uniform(
            key, (_OPERATION_COUNT, pop_size), 0.0, 1.0, dtype=evoxc.float32
        )
        thresholds = graph_constant(
            np.asarray(self.structural_probabilities, dtype=np.float32).reshape(
                _OPERATION_COUNT, 1
            ),
            np.float32,
        )
        hit = evoxc.less(drawn, thresholds)
        # First hitting row, or the "no operation" index when no row hit (`argmax` of an
        # all-False column would otherwise name row 0, a real operation).
        winner = evoxc.select(
            evoxc.reduce_max(hit, axes=0),
            evoxc.cast(evoxc.argmax(hit, axis=0), evoxc.int32),
            _int32(_OPERATION_COUNT),
        )
        return tuple(
            evoxc.equal(winner, _int32(index)) for index in range(_OPERATION_COUNT)
        )

    def _edit_context(
        self,
        genome: "BaseGenome",
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
    ) -> EditContext:
        """Build the row bookkeeping shared by the four operations of one stage.

        Every operation needs the same presence masks and the same flat index blocks; the
        context builds each of them once, which is what keeps the stage's traced-op count
        proportional to the number of *distinct* accesses rather than to the number of
        operations that happen to read the same array.

        Args:
            genome ("BaseGenome"): Genome supplying the row ranges of the input and output
                genes (the rows no structural operation may move).
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node key rows.
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attribute rows.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection key rows.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attribute
                rows.

        Returns:
            EditContext: masks, the shared rank-0 constants and the flattened index blocks.
        """
        pop_size, node_count = node_keys.shape
        edge_count = conn_keys.shape[1]
        key_width = conn_keys.shape[2]
        attr_width = conn_attrs.shape[2]
        sentinel = _int32(I_INF)
        present_node = evoxc.not_equal(node_keys, sentinel)
        row_index = graph_constant(
            np.arange(node_count, dtype=np.int32).reshape(1, node_count)
        )
        conn_row_index = graph_constant(
            np.arange(edge_count, dtype=np.int32).reshape(1, edge_count)
        )
        conn_src = conn_keys[:, :, 0]
        conn_dst = conn_keys[:, :, 1]
        present_conn = evoxc.not_equal(conn_src, sentinel)
        # The row ranges no structural operation may move: inputs first, outputs last.
        hidden_start = _int32(genome.num_inputs)
        after_inputs = evoxc.greater_equal(row_index, hidden_start)
        present_conn_float = evoxc.cast(present_conn, evoxc.float32)
        free_conn_row, free_conn_any = first_row(
            evoxc.logical_not(present_conn)
        )
        return EditContext(
            node_row_index=row_index,
            node_row_column=graph_constant(
                np.arange(node_count, dtype=np.int32).reshape(1, node_count, 1)
            ),
            conn_row_index=conn_row_index,
            conn_row_column=graph_constant(
                np.arange(edge_count, dtype=np.int32).reshape(1, edge_count, 1)
            ),
            conn_src=conn_src,
            conn_dst=conn_dst,
            present_node=present_node,
            free_node=evoxc.equal(node_keys, sentinel),
            deletable_node=evoxc.logical_and(
                after_inputs,
                evoxc.logical_and(
                    present_node,
                    evoxc.less(row_index, _int32(node_count - genome.num_outputs)),
                ),
            ),
            targetable_node=evoxc.logical_and(present_node, after_inputs),
            present_conn=present_conn,
            free_conn=evoxc.logical_not(present_conn),
            present_conn_float=present_conn_float,
            present_conn_any=evoxc.reduce_max(present_conn, axes=1),
            free_conn_row=free_conn_row,
            free_conn_any=free_conn_any,
            sentinel=sentinel,
            zero=_int32(0),
            nan_fill=enp.full((), float("nan"), dtype=evoxc.float32),
            in_degree_limit=_int32(genome.max_in_degree),
            node_row_limit=_int32(node_count),
            conn_row_limit=_int32(edge_count),
            node_flat=flatten(node_keys),
            node_offsets=row_offsets(pop_size, node_count),
            conn_key_flat=flatten(conn_keys),
            conn_key_base=graph_constant(
                column_base(pop_size, edge_count, key_width)
            ),
            conn_attr_flat=flatten(conn_attrs),
            conn_attr_base=graph_constant(
                column_base(pop_size, edge_count, attr_width)
            ),
            key_width=_int32(key_width),
            attr_width=_int32(attr_width),
        )

    def _plan_operations(
        self,
        state: State,
        genome: "BaseGenome",
        keys: evoxc.SymbolicTensor,
        gates: tuple[evoxc.SymbolicTensor, ...],
        context: EditContext,
        node_keys: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        new_node_keys: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
        pop_size: int,
        attr_width: int,
    ):
        """Resolve the four operations' candidates, eliding those that cannot fire.

        A probability of ``0`` makes its gate row all-``False``, so the operation can never
        apply; skipping its plan then keeps a whole operation's worth of traced ops (the
        cycle predicate included) out of the graph instead of computing a result that the
        merge immediately discards.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning the genes and the static invariants.
            keys (evoxc.SymbolicTensor): ``(5,)`` per-operation keys from one ``split_n``.
            gates (tuple): four ``(P,)`` bool gates from :meth:`_structural_gates`.
            context (EditContext): Shared masks and flat index blocks.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes.
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated connection keys.
            pop_size (int): Population size ``P`` (static).
            attr_width (int): ``CA``, used for the elided plan's placeholder value.

        Returns:
            tuple: ``(add_node, delete_node, add_conn, delete_conn)`` plans.
        """
        # One element each, broadcast by the plans' ``select``s: a population-wide
        # ``enp.full`` here is a graph constant proportional to ``P`` (see
        # ``..window.fill``) for a value that has no per-individual content.
        ghost = fill((pop_size,), I_INF, dtype=evoxc.int32)
        never = fill((pop_size,), False, dtype=evoxc.bool_)
        add_node = (
            self.plan_add_node(
                state,
                genome,
                keys[0],
                context,
                node_keys,
                conn_keys,
                conn_attrs,
                new_node_keys,
                gates[0],
            )
            if self.node_add > 0.0
            else AddNodePlan(
                never,
                ghost,
                ghost,
                ghost,
                ghost,
                ghost,
                ghost,
                fill((pop_size, 1, attr_width), 0.0, dtype=evoxc.float32),
            )
        )
        delete_node = (
            self.plan_delete_node(
                state, genome, keys[1], context, node_keys, conn_keys, gates[1]
            )
            if self.node_delete > 0.0
            else DeleteNodePlan(
                never,
                ghost,
                fill(tuple(context.present_conn.shape), False, dtype=evoxc.bool_),
            )
        )
        add_conn = (
            self.plan_add_conn(
                state, genome, keys[2], context, node_keys, conn_keys, new_conn_keys, gates[2]
            )
            if self.conn_add > 0.0
            else AddConnPlan(never, ghost, ghost, ghost)
        )
        delete_conn = (
            self.plan_delete_conn(state, genome, keys[3], context, conn_keys, gates[3])
            if self.conn_delete > 0.0
            else DeleteConnPlan(never, ghost)
        )
        return add_node, delete_node, add_conn, delete_conn

    def _merge_plan(
        self,
        state: State,
        genome: "BaseGenome",
        plans: tuple,
        context: EditContext,
        new_node_keys: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
        pop_size: int,
        node_attr_width: int,
        key_width: int,
        conn_attr_width: int,
    ) -> Edit:
        """Merge the four mutually exclusive plans into one write per payload-array slot.

        Each array slot is filled by a small ``select`` chain in the order the operations
        run — node deletion after node split, delete-connection after add-connection — with
        the array's own length as the "no write" row, which matches no row in
        :func:`~.plan.apply`.  Because only one gate is true per individual, the chain is a
        priority encoder, not a blend.

        A slot that only ever writes the "cleared" value (connection slot 0, whose row is the
        consumed or the deleted connection) carries the shared rank-0 ``I_INF``/NaN constant
        instead of a full-size block, and a slot whose row means "no write" whenever its
        value is meaningless (connection slot 2) carries the block unguarded — the row
        comparison already suppresses the write.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning the gene attribute factories.
            plans (tuple): the four plans from :meth:`_plan_operations`.
            context (EditContext): Shared constants, masks and flat index blocks.
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated connection keys.
            pop_size (int): Population size ``P`` (static).
            node_attr_width (int): ``NA``.
            key_width (int): ``CK``, the connection gene's key width.
            conn_attr_width (int): ``CA``.

        Returns:
            Edit: the plan to hand to :func:`~.plan.apply`.
        """
        add_node, delete_node, add_conn, delete_conn = plans
        if self.node_add > 0.0:
            node_identity = _broadcast_row(
                genome.node_gene.new_identity_attrs(state), pop_size
            )
        else:
            node_identity = fill(
                (pop_size, node_attr_width), 0.0, dtype=evoxc.float32
            )
        node_rows = evoxc.select(
            add_node.ok,
            add_node.node_row,
            evoxc.select(delete_node.ok, delete_node.node_row, context.node_row_limit),
        )
        node_key_values = evoxc.select(add_node.ok, new_node_keys, context.sentinel)
        node_attr_values = evoxc.select(
            enp.reshape(add_node.ok, (pop_size, 1)), node_identity, context.nan_fill
        )
        conn_rows = (
            evoxc.select(
                add_node.ok,
                add_node.delete_row,
                evoxc.select(
                    delete_conn.ok, delete_conn.conn_row, context.conn_row_limit
                ),
            ),
            evoxc.select(
                add_node.ok,
                add_node.identity_row,
                evoxc.select(
                    add_conn.ok, add_conn.conn_row, context.conn_row_limit
                ),
            ),
            evoxc.select(add_node.ok, add_node.copy_row, context.conn_row_limit),
        )
        add_node_column = enp.reshape(add_node.ok, (pop_size, 1, 1))
        identity_keys = _edge_key_row(
            add_node.in_key, new_node_keys, new_conn_keys, 0, key_width, genome
        )
        copy_keys = _edge_key_row(
            new_node_keys, add_node.out_key, new_conn_keys, 1, key_width, genome
        )
        add_keys = _edge_key_row(
            add_conn.in_key, add_conn.out_key, new_conn_keys, 2, key_width, genome
        )
        if self.node_add > 0.0:
            conn_identity = _broadcast_row(
                genome.conn_gene.new_identity_attrs(state), pop_size
            )
        else:
            conn_identity = fill(
                (pop_size, conn_attr_width), 0.0, dtype=evoxc.float32
            )
        if self.conn_add > 0.0:
            conn_zero = _broadcast_row(
                genome.conn_gene.new_zero_attrs(state), pop_size
            )
        else:
            conn_zero = fill(
                (pop_size, conn_attr_width), 0.0, dtype=evoxc.float32
            )
        conn_attr_slot_1 = evoxc.select(
            add_node_column,
            enp.reshape(conn_identity, (pop_size, 1, conn_attr_width)),
            enp.reshape(conn_zero, (pop_size, 1, conn_attr_width)),
        )
        return Edit(
            node_rows,
            node_key_values,
            node_attr_values,
            conn_rows,
            (context.sentinel, evoxc.select(add_node_column, identity_keys, add_keys), copy_keys),
            (context.nan_fill, conn_attr_slot_1, add_node.copy_attrs),
            delete_node.drop,
        )

__all__ = ["BaseMutation"]
