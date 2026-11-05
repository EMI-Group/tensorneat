"""NEAT-python style mutation: independent structural changes plus value mutation.

All four structural operations are batch-wide kernels: each gate is an elementwise mask on
the population axis, and an individual whose gate is off — or whose candidate is invalid —
comes out unchanged.  Capacity invariants are hard: ``max_nodes``, ``max_conns`` and the
genome's ``max_in_degree`` are never exceeded, a refused add is a no-op for that individual
(never a silent repair), and new rows go to the first free row of each array —
``arrays.first_row`` over the stage's ``context.free_node`` / ``context.free_conn`` masks —
in the same layout for keys and attributes.

The four operations have **independent** per-individual gates and run in the fixed order
add-node → delete-node → add-conn → delete-conn, each stage planning against the payload the
previous stage produced.  This module therefore **plans** each stage instead of applying it
directly: a helper resolves one operation's candidate rows and values, and
:meth:`.base.BaseMutation.mutate_structure` turns that plan into the stage's writes (see
:mod:`~.plan`).  Planning keeps a stage's traced-op count proportional to the number of
distinct accesses rather than to the number of intermediate whole-array rewrites, while the
sequential composition keeps the legacy semantics — a single call may apply several
structural changes to one individual.  The gate draw, the stage composition, the writes and
the value mutation are shared and live in :class:`~.base.BaseMutation`; this module supplies
the four probabilities and the four per-operation plans.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import ConfigError, State
from .arrays import (
    first_row,
    masked_random_row,
    random_row,
    take_block_at,
)
from .base import BaseMutation
from .cycle import closes_cycle
from .plan import AddConnPlan, AddNodePlan, DeleteConnPlan, DeleteNodePlan

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome


def _fetch_two(
    key: evoxc.SymbolicTensor,
    masks: evoxc.SymbolicTensor,
    low: "evoxc.SymbolicTensor | None" = None,
    high: "evoxc.SymbolicTensor | None" = None,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Draw one uniformly chosen ``True`` entry per row of a ``(P, k, L)`` mask block.

    The ``k`` choices (the source and the destination of a new edge) come from *one* batched
    draw and *one* ``argmax``: the maximiser of i.i.d. uniforms restricted to the ``True``
    set is uniform over that set, and the ``k`` slices are drawn from independent uniforms,
    so the choices stay independent of each other.

    Args:
        key: Rank-0 ``evoxc.random`` key; the whole batch and every choice come from it.
        masks: ``(P, k, L) bool`` eligibility masks.
        low: Optional rank-0 ``float32`` lower bound (``0.0`` when omitted) — the stage's
            shared bound tensor, which draws bit-identically to the Python float.
        high: Optional rank-0 ``float32`` upper bound (``1.0`` when omitted).

    Returns:
        ``(chosen, all_hit)``: ``(P, k) int32`` chosen column indices (``0`` where a slice has
        no ``True`` entry) and ``(P,) bool`` "every slice of this individual has an eligible
        entry" — ``any`` over each slice's candidates, folded with ``all`` over the slices.
    """
    draw = evoxc.random.uniform(
        key,
        tuple(masks.shape),
        0.0 if low is None else low,
        1.0 if high is None else high,
        dtype=evoxc.float32,
    )
    weights = evoxc.multiply(draw, evoxc.cast(masks, evoxc.float32))
    chosen = evoxc.cast(evoxc.argmax(weights, axis=2), evoxc.int32)
    # ``any`` over each slice's candidates, then ``all`` over the slices: the two reduces
    # compose, because a single reduce over both axes would ask "is every node eligible",
    # which is false for any genome with a node outside a slice.
    slice_hit = evoxc.reduce_max(masks, axes=2)
    return chosen, evoxc.reduce_min(slice_hit, axes=1)


class DefaultMutation(BaseMutation):
    """Structural and value mutation with per-operation probabilities.

    The four probabilities are independent per individual and per generation: one shared
    ``evoxc.random.uniform(key, (4, P))`` draw is compared against them and **each row is
    its own Bernoulli decision**, so an individual may fire any subset of the four
    operations and one call can apply up to four structural changes to the same genome (the
    frozen pre-rewrite library behaved the same way).

    The stages run in the fixed order add-node → delete-node → add-conn → delete-conn, and
    each plans against the payload the previous stage left behind: delete-node may delete
    the node add-node just created, delete-conn may consume an edge add-node or add-conn
    just created, and add-conn's duplicate / ``max_in_degree`` / cycle tests see the
    post-add-node and post-delete-node graph.  A refused candidate (duplicate edge,
    self-connection, cycle, capacity, ``max_in_degree``, a colliding caller allocation) is a
    bit-exact no-op for that individual, never a repair.  A probability of ``0`` still
    removes its operation — the cycle predicate included — from the traced graph entirely.

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

    def plan_add_node(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        context: object,
        node_keys: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        new_node_keys: evoxc.SymbolicTensor,
        gate: evoxc.SymbolicTensor,
    ) -> AddNodePlan:
        """Resolve one node split per individual, preserving the network's output.

        The chosen connection is consumed, the new node goes to the first free node row
        (``first_row(context.free_node)``) with the node gene's identity attributes, and two
        connections are created: an identity one from the old source
        (``conn_gene.new_identity_attrs``) and a copy of the consumed one into the old
        destination (with the recorded original attributes).  The two connection rows are the
        *first* and *second* free rows once the consumed row counts as free, which is exactly
        what the sequential formulation produced.

        No candidate when ``gate`` is false, when there is no connection to consume, when the
        node table is full, or when fewer than two connection rows are free.

        Structural node keys are **caller-allocated** (``algorithm/`` draws them from a
        population-side maximum), so this plan also verifies the allocation it is handed:
        a split is refused when ``new_node_keys`` already appears in ``node_keys`` (a
        duplicate key would fold two node rows onto one schedule slot) or when the consumed
        edge's destination is already at ``genome.max_in_degree`` incoming edges
        (``max_in_degree`` counts the destination's pre-write rows, which the split leaves
        unchanged).  Refusing is a bit-exact no-op — nothing in the payload is rewritten —
        so an over-eager allocation costs one skipped mutation, never an invalid genome.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing the identity attribute factory.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this operation's draw.
            context (EditContext): Shared masks, constants and flat index blocks of the
                stage; the base's stage wrapper also carries the shared ``uniform_low``
                / ``uniform_high`` rank-0 bounds every draw in this stage passes to
                ``evoxc.random.uniform``.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes;
                the stage's flattened copy in ``context`` is what the read consumes.
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected add-node".

        Returns:
            AddNodePlan: the candidate rows and values; ``ok`` is false for a refused split.
        """
        pop_size = node_keys.shape[0]
        delete_row, has_connection = masked_random_row(
            key,
            context.present_conn_float,
            context.present_conn_any,
            context.uniform_low,
            context.uniform_high,
        )
        delete_column = enp.reshape(delete_row, (pop_size, 1, 1))
        key_row = take_block_at(
            context.conn_key_flat, context.conn_key_base, delete_column, context.key_width
        )
        copy_attrs = take_block_at(
            context.conn_attr_flat,
            context.conn_attr_base,
            delete_column,
            context.attr_width,
        )
        node_row, has_node = first_row(context.free_node)
        # The two new edges take the first two rows that are free *before* the split, which
        # is exactly the documented "refused unless two connection rows are free" rule: the
        # second ``first_row`` has no hit when only one row was free.
        identity_row = context.free_conn_row
        remaining = evoxc.logical_and(
            context.free_conn,
            evoxc.greater(
                context.conn_row_index, enp.reshape(identity_row, (pop_size, 1))
            ),
        )
        copy_row, has_copy = first_row(remaining)
        in_key = evoxc.reshape(
            evoxc.slice(key_row, (0, 0, 0), (pop_size, 1, 1)), (pop_size,)
        )
        out_key = evoxc.reshape(
            evoxc.slice(key_row, (0, 0, 1), (pop_size, 1, 1)), (pop_size,)
        )
        # The caller allocates structural keys and may hand out one the payload already
        # uses (e.g. a crossover child can inherit a larger key from a parent other than
        # the one the allocation was based on).  Writing it would put two node rows on one
        # schedule slot, and the extra identity edge can then push a destination past
        # ``max_in_degree``; a colliding allocation therefore refuses the split instead of
        # duplicating a key.  The comparison is one ``(P, N)`` equality after broadcasting
        # the ``(P,)`` keys to a column, folded by a reduce over the node axis.
        aliased = evoxc.reduce_max(
            evoxc.equal(node_keys, enp.reshape(new_node_keys, (pop_size, 1))), axes=1
        )
        # The consumed row (``in_key -> out_key``) is replaced by the copy edge
        # (``new -> out_key``), so the destination's post-write in-degree equals its
        # pre-write count; refusing a destination already over the limit is therefore a
        # no-op on every legal payload and closes the case the alias guard cannot.
        out_degree = evoxc.reduce_sum(
            evoxc.cast(
                evoxc.equal(context.conn_dst, enp.reshape(out_key, (pop_size, 1))),
                evoxc.int32,
            ),
            axes=1,
        )
        ok = evoxc.logical_and(
            evoxc.logical_and(gate, has_connection),
            evoxc.logical_and(has_node, has_copy),
        )
        ok = evoxc.logical_and(ok, evoxc.logical_not(aliased))
        ok = evoxc.logical_and(
            ok, evoxc.less_equal(out_degree, context.in_degree_limit)
        )
        return AddNodePlan(
            ok,
            node_row,
            delete_row,
            identity_row,
            copy_row,
            in_key,
            out_key,
            copy_attrs,
        )

    def plan_delete_node(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        context: object,
        node_keys: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        gate: evoxc.SymbolicTensor,
    ) -> DeleteNodePlan:
        """Resolve one hidden-node deletion per individual.

        Input rows (``0..num_inputs-1``) and output rows (``N-num_outputs..N-1``) are never
        candidates.  Every connection touching the deleted key is detached in the same
        elementwise pass, so the operation costs one mask and never a per-connection delete.

        No candidate when ``gate`` is false or when the genome has no hidden node.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing the input/output row ranges.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this operation's draw.
            context (EditContext): Shared masks, constants and flat index blocks of the
                stage; the base's stage wrapper also carries the shared ``uniform_low``
                / ``uniform_high`` rank-0 bounds every draw in this stage passes to
                ``evoxc.random.uniform``.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected
                delete-node".

        Returns:
            DeleteNodePlan: the row to clear and the (already gated) detach mask.
        """
        pop_size = node_keys.shape[0]
        node_row, has_node = random_row(
            key, context.deletable_node, context.uniform_low, context.uniform_high
        )
        target = evoxc.gather(
            context.node_flat,
            evoxc.add(context.node_offsets, enp.reshape(node_row, (pop_size, 1))),
            axis=0,
        )
        # A live row is never ``I_INF``, so an absent connection row (both key columns
        # ``I_INF``) can never compare equal and needs no present mask here.
        touches = evoxc.logical_or(
            evoxc.equal(context.conn_src, target),
            evoxc.equal(context.conn_dst, target),
        )
        ok = evoxc.logical_and(gate, has_node)
        return DeleteNodePlan(
            ok, node_row, evoxc.logical_and(touches, enp.reshape(ok, (pop_size, 1)))
        )

    def plan_add_conn(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        context: object,
        node_keys: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        new_conn_keys: evoxc.SymbolicTensor,
        gate: evoxc.SymbolicTensor,
    ) -> AddConnPlan:
        """Resolve one added connection per individual.

        The source may be any existing node, the destination any existing node that is not an
        input.  The pair is refused when an endpoint is absent, when the ``(in_key, out_key)``
        pair already exists, when the destination would exceed ``genome.max_in_degree``, and —
        for a feedforward genome — when the edge would close a cycle
        (:func:`~.cycle.closes_cycle`, the bounded reachability predicate).

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing ``max_in_degree`` and ``network_type``.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this operation's draws.
            context (EditContext): Shared masks, constants and flat index blocks of the
                stage; the base's stage wrapper also carries the shared ``uniform_low``
                / ``uniform_high`` rank-0 bounds every draw in this stage passes to
                ``evoxc.random.uniform``.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated connection keys.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected add-conn".

        Returns:
            AddConnPlan: the candidate row and endpoints; ``ok`` is false for a refused add.
        """
        pop_size = node_keys.shape[0]
        candidates = enp.stack([context.present_node, context.targetable_node], axis=1)
        chosen, all_hit = _fetch_two(
            key,
            candidates,
            context.uniform_low,
            context.uniform_high,
        )
        # Both endpoints in one gather: the chosen columns index the node table per
        # individual, so the two keys and their ``(P, 1)`` comparison columns come out of a
        # single flat gather of the stacked row indices.
        endpoints = evoxc.gather(
            context.node_flat, evoxc.add(context.node_offsets, chosen), axis=0
        )
        in_column = evoxc.slice(endpoints, (0, 0), (pop_size, 1))
        out_column = evoxc.slice(endpoints, (0, 1), (pop_size, 1))
        in_keys = enp.reshape(in_column, (pop_size,))
        out_keys = enp.reshape(out_column, (pop_size,))
        # Endpoint keys are never ``I_INF`` (both masks only ever name a live row), so an
        # absent connection row cannot match and the present mask is not needed either time.
        duplicate = evoxc.reduce_max(
            evoxc.logical_and(
                evoxc.equal(context.conn_src, in_column),
                evoxc.equal(context.conn_dst, out_column),
            ),
            axes=1,
        )
        in_degree = evoxc.reduce_sum(
            evoxc.cast(evoxc.equal(context.conn_dst, out_column), evoxc.int32), axes=1
        )
        allowed = evoxc.logical_and(
            evoxc.logical_and(
                evoxc.logical_and(gate, context.free_conn_any),
                evoxc.logical_and(all_hit, evoxc.logical_not(duplicate)),
            ),
            evoxc.less(in_degree, context.in_degree_limit),
        )
        if genome.network_type == "feedforward":
            allowed = evoxc.logical_and(
                allowed,
                evoxc.logical_not(
                    closes_cycle(
                        node_keys, conn_keys, in_keys, out_keys, genome.key_span
                    )
                ),
            )
        return AddConnPlan(allowed, context.free_conn_row, in_keys, out_keys)

    def plan_delete_conn(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        context: object,
        conn_keys: evoxc.SymbolicTensor,
        gate: evoxc.SymbolicTensor,
    ) -> DeleteConnPlan:
        """Resolve one connection deletion per individual.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning the connection gene.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this operation's draw.
            context (EditContext): Shared masks, constants and flat index blocks of the
                stage; the base's stage wrapper also carries the shared ``uniform_low``
                / ``uniform_high`` rank-0 bounds every draw in this stage passes to
                ``evoxc.random.uniform``.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected
                delete-conn".

        Returns:
            DeleteConnPlan: the row to clear; ``ok`` is false when there is none.
        """
        conn_row, has_connection = masked_random_row(
            key,
            context.present_conn_float,
            context.present_conn_any,
            context.uniform_low,
            context.uniform_high,
        )
        return DeleteConnPlan(evoxc.logical_and(gate, has_connection), conn_row)


#: Connection-gate probabilities of the mutation operator a genome builds when its
#: ``mutation`` argument is ``None``.  add-conn is accepted about half as often as
#: delete-conn at equal rates (duplicate, self-connection, cycle and ``max_in_degree``
#: refusals), so equal gates leave the population under-wired; these values keep the
#: operator's structural steady state positive (XOR3d pop 100, 150 generations: 13/20
#: seeds escape the constant-0.5 basin against 4/20 at 0.2/0.2).
#: ``DefaultMutation``'s own constructor defaults stay NEAT-python's ``0.2``/``0.2``, so an
#: explicitly constructed operator is unchanged.
DEFAULT_GENOME_CONN_ADD = 0.4
DEFAULT_GENOME_CONN_DELETE = 0.05


__all__ = ["DefaultMutation"]
