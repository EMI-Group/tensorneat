"""Mutation contract for the batched NEAT gene payload.

A mutation operator is a trace-time kernel over a whole population: the four payload
arrays enter and leave with identical static shapes, the population lives on axis 0 and
every gate is an elementwise mask — no Python loop over individuals, no ``vmap``, no
``scan``/``while_loop``/``cond``.
Structural keys are caller-allocated ``int32`` tensors, and the only randomness consumed
here is one rank-0 ``evoxc`` key, so a ``(seed, config, shape)`` triple reproduces a run.

The kernel is factored into two stages — :meth:`BaseMutation.mutate_structure` (the four
structural operations, each with its own independent per-individual gate) and
:meth:`BaseMutation.mutate_values` (attribute mutation of every existing gene).  Both are
implemented here, because both are the same for every operator family; a subclass only
supplies the per-operation plan helpers (``plan_add_node``, ``plan_delete_node``,
``plan_add_conn``, ``plan_delete_conn``) and the four probabilities they gate on
(:attr:`BaseMutation.structural_probabilities`).

Composition rule (shared by every family): the four structural stages run **sequentially**
in the fixed order add-node → delete-node → add-conn → delete-conn, so a single call may
apply up to four structural changes to one individual.  Each stage plans against the
payload the previous stage left behind, so delete-node may delete the node add-node just
created, delete-conn may consume an edge add-node or add-conn just created, and add-conn's
duplicate / in-degree / cycle tests all see the post-add-node and post-delete-node graph.
Every refusal (duplicate edge, self-connection, cycle, capacity, ``max_in_degree``, a
colliding caller allocation) is a bit-exact no-op for that individual and never a partially
applied write; a probability of ``0`` still removes that operation's whole chain — the
cycle predicate included — from the traced graph.

Heritability rule: the value stage runs over the tables the structural stages produced, and
the rows those stages wrote in this call are then put back verbatim.  A gene a structural
operation creates therefore enters selection with the attributes the operation gave it —
an add-node's identity edge still carries ``weight = 1.0``, its copy edge the consumed
edge's weight and the new node its ``new_identity_attrs`` — instead of being re-drawn by the
same call's attribute mutation.
"""

from typing import TYPE_CHECKING

import numpy as np
import evoxc
import evoxc.numpy as enp

from ....common import I_INF, ConfigError, State, StatefulBaseClass
from ...utils import GenomeTensors
from .arrays import (
    column_base,
    first_row,
    flatten,
    graph_constant,
    i32 as _int32,
    row_offsets,
    uniform_bounds,
)
from .plan import (
    Edit,
    EditContext,
    apply,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps genome -> operations acyclic
    from ...base import BaseGenome

#: Number of structural operations, in the order ``(node_add, node_delete, conn_add,
#: conn_delete)``.  Every row of the shared ``(4, P)`` gate draw is an *independent*
#: Bernoulli decision, and the four stages run one after another, so the constant is the
#: number of gate rows and of stages — not an index past the last row.
_OPERATION_COUNT = 4

#: Per-stage static write flags ``(write_nodes, clear_conn)`` handed to
#: :func:`~.plan.apply`, in stage order.  A stage that statically cannot need a payload
#: array passes it through untouched instead of paying for a full-size ``select`` that
#: would provably write nothing: add-node writes a node row *and* clears the consumed edge
#: (slot 0); delete-node clears a node row and lets its detach mask do the connection
#: clearing; add-conn touches no node row and no cleared row (its slot 0 is the "no write"
#: row); delete-conn clears exactly one connection row and no node row.
_STAGE_WRITES = (
    (True, True),
    (True, True),
    (False, False),
    (False, True),
)


def _broadcast_row(
    row: evoxc.SymbolicTensor, pop_size: int, width: int
) -> evoxc.SymbolicTensor:
    """Spread one gene attribute row over the population axis.

    The gene factories (``new_identity_attrs``, ``new_zero_attrs``) build a single
    ``(W,)`` row because the values are static, while the payload writers take a
    ``(P, W)`` block; ``broadcast_to`` expresses that in one traced op, where an
    elementwise add against a ``(1, W)`` view would cost a full-size ``zeros`` constant
    plus an ``add``.  ``broadcast_to`` also normalizes a ``(1, W)`` row, so a gene factory
    returning either rank costs the same single op — no intermediate ``reshape``.

    The width is passed explicitly rather than read off ``row.shape[-1:]`` so a
    zero-width family (``attr_specs == ()``) yields a ``(P, 0)`` block instead of a
    rank-2 broadcast of a rank-0/one-dimensional row, which the caller's
    ``(P, 1, 0)`` ``reshape`` could not consume.

    Args:
        row: ``(W,)`` or ``(1, W)`` attribute row of one gene (a rank-0 row for
            ``width == 0``).
        pop_size: Population size ``P`` (static).
        width: Attribute width ``W`` of the gene (static).

    Returns:
        ``(P, W)`` tensor whose every row equals ``row``.
    """
    return enp.broadcast_to(row, (pop_size, width))


def _written_rows(
    before: evoxc.SymbolicTensor, after: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Mark the rows one call's structural stages wrote, from a key-table difference.

    Only the structural stages change a key array (the value stage touches attributes
    alone), so one difference between the payload's key tables and the ones
    :meth:`BaseMutation.mutate_structure` returned is the exact set of rows whose genes
    were created in this call.  A written row carries a different key; a row the
    structural stage *cleared* (a deleted node, a consumed or deleted connection) turns
    ``I_INF`` and is already NaN in the attribute tables, so the value stage's own padding
    mask keeps it and it needs no separate term.  No structural operation ever rewrites an
    attribute row while leaving its key row unchanged, so a row whose keys match is a row
    whose attributes must be left alone.

    Args:
        before (evoxc.SymbolicTensor): Key array as the call received it — ``(P, max_nodes)``
            for the node table or ``(P, max_conns, CK)`` for the connection table.
        after (evoxc.SymbolicTensor): The same key array as ``mutate_structure`` returned.

    Returns:
        ``(P, max_nodes)`` / ``(P, max_conns)`` bool: True where this call wrote the row.
        Three-dimensional key tables fold their key-width axis, so a connection row counts
        as written when *any* of its key columns changed (the ``historical_marker`` one
        included).
    """
    differs = evoxc.not_equal(before, after)
    if len(differs.shape) == 2:
        return differs
    # One ``logical_or`` per key column over kept-axis views, never a trailing-axis
    # ``reduce_max``: an innermost-axis reduce over a ``(P, E, CK)`` block costs an order of
    # magnitude more than the same predicate folded from ``CK`` slices.
    written = differs[:, :, 0]
    for column in range(1, differs.shape[2]):
        written = evoxc.logical_or(written, differs[:, :, column])
    return written


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


class _StageContext:
    """One structural stage's shared row bookkeeping plus the uniform-bound pair.

    The presence masks, row-index constants and flat index blocks live in the
    :class:`~.plan.EditContext`; the stage also needs the two rank-0 ``float32``
    bounds every ``evoxc.random.uniform`` draw shares (:func:`~.arrays.uniform_bounds`),
    because a Python float bound costs one ``float64`` ``constant`` *per draw* under the
    reference interpreter.  That pair is a draw detail of this stage rather than row
    bookkeeping, so it rides here instead of widening ``EditContext`` — whose field set
    is the frozen contract with the write plan — while every other attribute is read
    through to the wrapped context, so the plan methods keep seeing the same object
    shape.

    Attributes:
        context (EditContext): The stage's row bookkeeping (every other attribute of
            this object reads through to it).
        uniform_low: Rank-0 ``float32`` ``0.0``.
        uniform_high: Rank-0 ``float32`` ``1.0``.
    """

    __slots__ = ("context", "uniform_low", "uniform_high")

    def __init__(
        self,
        context: EditContext,
        low: evoxc.SymbolicTensor,
        high: evoxc.SymbolicTensor,
    ):
        self.context = context
        self.uniform_low = low
        self.uniform_high = high

    @property
    def uniform_bounds(
        self,
    ) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
        """The shared ``(low, high)`` pair, in the order ``evoxc.random.uniform`` takes.

        Returns:
            tuple: the rank-0 ``float32`` ``0.0`` and ``1.0`` tensors.
        """
        return self.uniform_low, self.uniform_high

    def __getattr__(self, name: str):
        # Only reached for names this wrapper does not declare, i.e. the EditContext
        # fields; the delegation is what keeps every plan's ``context.<field>`` read
        # unchanged.
        return getattr(self.context, name)


class BaseMutation(StatefulBaseClass):
    """Batched mutation operator: four structural operations plus value mutation.

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

        The key is split internally into one key per stage.  :meth:`mutate_structure` runs
        first and :meth:`mutate_values` sees the tables it produced, after which the rows
        the structural stages wrote are restored verbatim: the value stage mutates the
        *pre-existing* genes only, so a gene a structural operation creates is heritable
        (an add-node's identity edge keeps ``weight = 1.0``, its copy edge the consumed
        edge's weight, the new node its ``new_identity_attrs``) rather than re-drawn before
        selection sees it.  Nothing is applied per individual: gates are elementwise masks
        on axis 0 and genes whose candidate is invalid are simply carried through.

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
        node_count = node_keys.shape[1]
        edge_count = conn_keys.shape[1]
        node_attr_width = node_attrs.shape[2]
        conn_attr_width = conn_attrs.shape[2]
        structured = self.mutate_structure(
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
        mutated = self.mutate_values(state, genome, values_key, *structured)
        # The value stage ran over the same tables, so the rows the structural stage wrote
        # in this call are put back afterwards: a gene a structural operation creates is
        # heritable — the value stage must not re-draw it before selection sees it.
        node_attrs = structured.node_attrs
        if node_attr_width:
            written = _written_rows(node_keys, structured.node_keys)
            node_attrs = evoxc.select(
                enp.reshape(written, (pop_size, node_count, 1)),
                structured.node_attrs,
                mutated.node_attrs,
            )
        conn_attrs = structured.conn_attrs
        if conn_attr_width:
            written = _written_rows(conn_keys, structured.conn_keys)
            conn_attrs = evoxc.select(
                enp.reshape(written, (pop_size, edge_count, 1)),
                structured.conn_attrs,
                mutated.conn_attrs,
            )
        return GenomeTensors(
            structured.node_keys, node_attrs, structured.conn_keys, conn_attrs
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
        """Apply the four structural operations in sequence, each independently gated.

        One batched ``evoxc.random.uniform(key, (4, P))`` draw is compared against the
        four :attr:`structural_probabilities`, and **every** row is its own independent
        Bernoulli decision — one individual may fire any subset of the four, so one call
        can apply up to four structural changes to one genome (the frozen pre-rewrite
        library behaved the same way).

        The stages then run in the fixed order add-node → delete-node → add-conn →
        delete-conn, each with its own key (from one ``split_n``) and its own ``(P,)``
        gate, and each stage plans against the payload the previous stage produced.  The
        consequences are deliberate: delete-node may delete the node add-node just
        created, delete-conn may consume an edge add-node or add-conn just created, and
        add-conn's duplicate / in-degree / cycle tests see the post-add-node and
        post-delete-node graph.  A probability of ``0`` elides its stage completely (the
        cycle predicate included); a stage that fires but whose candidate is refused
        (duplicate edge, self-connection, cycle, capacity, ``max_in_degree``, a colliding
        caller allocation) is a bit-exact no-op for that individual.

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
                keys; column 0 is the add-node identity edge, column 1 the add-node copy
                edge and column 2 the add-conn edge.

        Returns:
            GenomeTensors: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` after every
            stage; an individual whose gates are all off — or whose candidates were all
            refused — is returned unchanged.

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
        # Both rank-0 bounds are hoisted once for the whole call and shared by every
        # stage's draws; a Python float bound would embed one ``float64`` constant per
        # draw.  ``0.0``/``1.0`` are exact in ``float32``, so the draws are bit-identical.
        uniform = uniform_bounds()
        gates = self._structural_gates(keys[_OPERATION_COUNT], pop_size, uniform)
        probabilities = self.structural_probabilities
        key_width = conn_keys.shape[2]
        node_attr_width = node_attrs.shape[2]
        conn_attr_width = conn_attrs.shape[2]
        payload = (node_keys, node_attrs, conn_keys, conn_attrs)
        # Static Python loop over the four stages: ``index`` is a Python int, so both the
        # elision below and the plan/write dispatch inside the helpers are ordinary
        # Python control flow, never a traced one.
        for index in range(_OPERATION_COUNT):
            if probabilities[index] == 0.0:
                continue
            node_keys, node_attrs, conn_keys, conn_attrs = payload
            # Rebuilt per stage: every mask and index block describes the payload this
            # stage is about to edit, not the one it started from.
            context = self._edit_context(
                genome, node_keys, node_attrs, conn_keys, conn_attrs
            )
            stage = self._stage_context(context, uniform)
            plan = self._plan_operation(
                index,
                state,
                genome,
                keys[index],
                gates[index],
                stage,
                node_keys,
                conn_keys,
                conn_attrs,
                new_node_keys,
                new_conn_keys,
            )
            edit = self._stage_edit(
                index,
                state,
                genome,
                plan,
                context,
                new_node_keys,
                new_conn_keys,
                pop_size,
                node_attr_width,
                key_width,
                conn_attr_width,
            )
            write_nodes, clear_conn = _STAGE_WRITES[index]
            payload = tuple(
                apply(
                    edit,
                    context,
                    node_keys,
                    node_attrs,
                    conn_keys,
                    conn_attrs,
                    write_nodes=write_nodes,
                    clear_conn=clear_conn,
                )
            )
        return GenomeTensors(*payload)

    def _stage_context(
        self,
        context: EditContext,
        uniform: tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor],
    ) -> _StageContext:
        """Wrap one stage's row bookkeeping together with the shared uniform bounds.

        The two rank-0 ``float32`` bounds are built **once** per
        :meth:`mutate_structure` call and shared by every stage's draws; a Python float
        bound would embed one ``float64`` ``constant`` *per draw* instead.  ``0.0``/``1.0``
        are exact in ``float32``, so the drawn values are bit-identical (see
        :func:`~.arrays.uniform_bounds`).

        Args:
            context (EditContext): The stage's masks, constants and flat index blocks,
                derived from the payload this stage starts from.
            uniform (tuple): The call's shared ``(low, high)`` rank-0 ``float32`` pair.

        Returns:
            _StageContext: ``context`` plus the shared ``(low, high)`` bounds.
        """
        low, high = uniform
        return _StageContext(context, low, high)

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
        """Mutate the attributes of every gene in the population that already exists.

        One batched ``genome.node_gene.mutate`` and one ``genome.conn_gene.mutate`` call
        cover the whole ``(P, N, NA)`` / ``(P, E, CA)`` tables — the genes are batched over
        their leading axes, so no Python loop or per-individual key split is needed.  The
        result is re-masked with the padding pattern read *before* the mutation, so a
        padded row can never become finite.

        Called from :meth:`__call__` on the tables :meth:`mutate_structure` returned, so
        the rows a structural operation created in this call are drawn over too;
        :meth:`__call__` then restores them from the structural write, which is what makes
        a newly created gene heritable within the same call.  Run standalone (as the
        two-stage API allows) this method makes no such distinction.

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
        self,
        key: evoxc.SymbolicTensor,
        pop_size: int,
        uniform: tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor],
    ) -> tuple[
        evoxc.SymbolicTensor,
        evoxc.SymbolicTensor,
        evoxc.SymbolicTensor,
        evoxc.SymbolicTensor,
    ]:
        """Resolve the four independent per-individual structural gates.

        The four probabilities are compared against one ``(4, P)`` uniform draw and each
        row is read as its own Bernoulli decision — ``gate[i][p] = drawn[i, p] <
        probabilities[i]`` — so an individual may fire any subset of the four operations
        (up to all four).  Both outcomes are ordinary elementwise results, so the kernel
        stays branch-free (``cond`` is a region op and cannot be batched).

        Args:
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this draw.
            pop_size (int): Population size ``P`` (static).
            uniform (tuple): The call's shared ``(low, high)`` rank-0 ``float32`` bounds
                (:func:`~.arrays.uniform_bounds`); ``0.0``/``1.0`` as tensors cost two
                constants per call instead of two per draw, and draw bit-identically.

        Returns:
            tuple: four ``(P,)`` bool gates, in the order add-node, delete-node, add-conn,
            delete-conn; each is independent of the other three.

        Notes:
            The comparison runs against a ``(4, 1)`` constant column of static thresholds,
            so a probability of ``0`` produces an all-``False`` row (the operation can never
            fire) and a probability of ``1`` an all-``True`` one, without a Python branch.
            Each gate is the static row selection ``hit[index]`` (one op: the axis-dropping
            gather ``getitem`` lowers to), so the four gates cost four ops and no reshape.
        """
        low, high = uniform
        drawn = evoxc.random.uniform(
            key, (_OPERATION_COUNT, pop_size), low, high, dtype=evoxc.float32
        )
        thresholds = graph_constant(
            np.asarray(self.structural_probabilities, dtype=np.float32).reshape(
                _OPERATION_COUNT, 1
            ),
            np.float32,
        )
        hit = evoxc.less(drawn, thresholds)
        return tuple(hit[index] for index in range(_OPERATION_COUNT))

    def _edit_context(
        self,
        genome: "BaseGenome",
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
    ) -> EditContext:
        """Build the row bookkeeping one structural stage needs from its input payload.

        Built once per stage — against the payload that stage is about to edit, because a
        previous stage may have moved a row — so every operation of the stage shares one
        set of presence masks and flat index blocks.  That is what keeps the stage's
        traced-op count proportional to the number of *distinct* accesses rather than to
        the number of operations that happen to read the same array.

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

    def _plan_operation(
        self,
        index: int,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        gate: evoxc.SymbolicTensor,
        context: _StageContext,
        node_keys: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        new_node_keys: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
    ):
        """Resolve the candidate of **one** structural stage.

        The dispatch on ``index`` is a static Python ``if`` chain over the four stage
        numbers, so a stage traces only its own plan.  The elided "never" placeholder
        plans the merged formulation carried for the three operations it was *not*
        running are gone: a stage whose probability is ``0`` is skipped wholesale by
        :meth:`mutate_structure` (keeping even the cycle predicate out of the graph) and a
        stage that fires always plans for real, so there is nothing left to elide.

        Args:
            index (int): Stage number in ``0..3``, in the order add-node, delete-node,
                add-conn, delete-conn.
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning the genes and the static invariants.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this stage's draws.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` independent gate of this stage.
            context (_StageContext): The stage's masks, constants and flat index blocks —
                derived from the payload this stage *starts from* — plus the shared
                ``(low, high)`` uniform bounds; the plans read it as an ``EditContext``
                because every other attribute is delegated.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys of this stage's
                input payload.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes
                (the stage's flattened copy in ``context`` is what the add-node read
                consumes).
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated connection keys.

        Returns:
            AddNodePlan | DeleteNodePlan | AddConnPlan | DeleteConnPlan: the single plan of
            stage ``index``.  The four ``plan_*`` methods keep their names and parameter
            lists: they are the subclass extension points.
        """
        if index == 0:
            return self.plan_add_node(
                state,
                genome,
                key,
                context,
                node_keys,
                conn_keys,
                conn_attrs,
                new_node_keys,
                gate,
            )
        if index == 1:
            return self.plan_delete_node(
                state, genome, key, context, node_keys, conn_keys, gate
            )
        if index == 2:
            return self.plan_add_conn(
                state, genome, key, context, node_keys, conn_keys, new_conn_keys, gate
            )
        return self.plan_delete_conn(state, genome, key, context, conn_keys, gate)

    def _stage_edit(
        self,
        index: int,
        state: State,
        genome: "BaseGenome",
        plan,
        context: EditContext,
        new_node_keys: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
        pop_size: int,
        node_attr_width: int,
        key_width: int,
        conn_attr_width: int,
    ) -> Edit:
        """Turn one stage's plan into the writes :func:`~.plan.apply` performs.

        This is the single-operation form of the merged write plan: each stage names only
        the rows it can actually write, and the array it cannot touch is left as ``None``
        and skipped statically by the caller's
        ``_STAGE_WRITES``-driven ``write_nodes`` / ``clear_conn`` flags.

        A slot whose every value is the "cleared" one carries the shared rank-0
        ``I_INF``/NaN constant instead of a full-size block — the write's ``select``
        broadcasts it — and a slot that names no row when its value is meaningless
        carries the block unguarded, because the row comparison already suppresses the
        write.

        Args:
            index (int): Stage number in ``0..3``.
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning the gene attribute factories.
            plan: The plan of this stage from :meth:`_plan_operation`.
            context (EditContext): Shared constants, masks and flat index blocks of this
                stage's input payload.
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated connection
                keys; column 0 is the add-node identity edge, column 1 the add-node copy
                edge and column 2 the add-conn edge.
            pop_size (int): Population size ``P`` (static).
            node_attr_width (int): ``NA``.
            key_width (int): ``CK``, the connection gene's key width.
            conn_attr_width (int): ``CA``.

        Returns:
            Edit: the writes of this stage, to hand to :func:`~.plan.apply`.
        """
        if index == 0:
            # Add-node: one node row, the consumed connection row (slot 0) and the two new
            # edge rows (slots 1 = identity, 2 = copy, which inherits the recorded
            # attributes of the consumed edge).
            return Edit(
                evoxc.select(plan.ok, plan.node_row, context.node_row_limit),
                evoxc.select(plan.ok, new_node_keys, context.sentinel),
                evoxc.select(
                    enp.reshape(plan.ok, (pop_size, 1)),
                    _broadcast_row(
                        genome.node_gene.new_identity_attrs(state),
                        pop_size,
                        node_attr_width,
                    ),
                    context.nan_fill,
                ),
                (
                    evoxc.select(plan.ok, plan.delete_row, context.conn_row_limit),
                    evoxc.select(plan.ok, plan.identity_row, context.conn_row_limit),
                    evoxc.select(plan.ok, plan.copy_row, context.conn_row_limit),
                ),
                (
                    context.sentinel,
                    _edge_key_row(
                        plan.in_key, new_node_keys, new_conn_keys, 0, key_width, genome
                    ),
                    _edge_key_row(
                        new_node_keys, plan.out_key, new_conn_keys, 1, key_width, genome
                    ),
                ),
                (
                    context.nan_fill,
                    enp.reshape(
                        _broadcast_row(
                            genome.conn_gene.new_identity_attrs(state),
                            pop_size,
                            conn_attr_width,
                        ),
                        (pop_size, 1, conn_attr_width),
                    ),
                    plan.copy_attrs,
                ),
                None,
            )
        if index == 1:
            # Delete-node: clear the node row; every connection row touching the deleted
            # key detaches through ``drop`` in apply's single clearing pass, so slot 0
            # names no row of its own and the scatter writes nothing.
            return Edit(
                evoxc.select(plan.ok, plan.node_row, context.node_row_limit),
                context.sentinel,
                context.nan_fill,
                (context.conn_row_limit,),
                (context.sentinel,),
                (context.nan_fill,),
                plan.drop,
            )
        if index == 2:
            # Add-conn: apply skips the clearing pass altogether (``clear_conn=False``), so
            # slot 0 exists only to keep the slot numbering and slot 1 is the one write.
            return Edit(
                None,
                None,
                None,
                (
                    context.conn_row_limit,
                    evoxc.select(plan.ok, plan.conn_row, context.conn_row_limit),
                ),
                (
                    context.sentinel,
                    _edge_key_row(
                        plan.in_key, plan.out_key, new_conn_keys, 2, key_width, genome
                    ),
                ),
                (
                    context.nan_fill,
                    enp.reshape(
                        _broadcast_row(
                            genome.conn_gene.new_zero_attrs(state),
                            pop_size,
                            conn_attr_width,
                        ),
                        (pop_size, 1, conn_attr_width),
                    ),
                ),
                None,
            )
        # Delete-conn: exactly one cleared connection row, no node write.
        return Edit(
            None,
            None,
            None,
            (evoxc.select(plan.ok, plan.conn_row, context.conn_row_limit),),
            (context.sentinel,),
            (context.nan_fill,),
            None,
        )



__all__ = ["BaseMutation"]
