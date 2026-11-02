"""NEAT-python style mutation: at most one structural change plus value mutation.

All four structural operations are batch-wide kernels: each gate is an elementwise mask on
the population axis, and an individual whose gate is off — or whose candidate is invalid —
comes out unchanged.  Capacity invariants are hard: ``max_nodes``, ``max_conns`` and the
genome's ``max_in_degree`` are never exceeded, a refused add is a no-op for that individual
(never a silent repair), and new rows go to ``first_free_row`` in the same layout for keys
and attributes.

Because the four operations are mutually exclusive, this module **plans** them instead of
applying them one after another: each helper resolves one operation's candidate rows and
values, and :meth:`.base.BaseMutation.mutate_structure` merges the four plans into a single
per-array write (see :mod:`~.plan`).  That keeps the traced-op count of the stage
proportional to the number of distinct accesses rather than to the number of intermediate
whole-array rewrites.  The gate draw, the merge and the value mutation are shared and live
in :class:`~.base.BaseMutation`; this module supplies the four probabilities and the four
per-operation plans.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import ConfigError, State
from .arrays import first_row, i32, random_row, take_block, take_rows
from .base import BaseMutation
from .cycle import closes_cycle
from .plan import AddConnPlan, AddNodePlan, DeleteConnPlan, DeleteNodePlan

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome


def _fetch_two(
    key: evoxc.SymbolicTensor, masks: evoxc.SymbolicTensor
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Draw one uniformly chosen ``True`` entry per row of a ``(P, k, L)`` mask block.

    The ``k`` choices (the source and the destination of a new edge) come from *one* batched
    draw and *one* ``argmax``: the maximiser of i.i.d. uniforms restricted to the ``True``
    set is uniform over that set, and the ``k`` slices are drawn from independent uniforms,
    so the choices stay independent of each other.

    Args:
        key: Rank-0 ``evoxc.random`` key; the whole batch and every choice come from it.
        masks: ``(P, k, L) bool`` eligibility masks.

    Returns:
        ``(chosen, hit)``, both ``(P, k)``: the chosen column index (``0`` where a slice has
        no ``True`` entry) and the per-slice eligibility flag.
    """
    draw = evoxc.random.uniform(
        key, tuple(masks.shape), 0.0, 1.0, dtype=evoxc.float32
    )
    weights = evoxc.multiply(draw, evoxc.cast(masks, evoxc.float32))
    chosen = evoxc.cast(evoxc.argmax(weights, axis=2), evoxc.int32)
    return chosen, evoxc.reduce_max(masks, axes=2)


class DefaultMutation(BaseMutation):
    """Structural and value mutation with per-operation probabilities.

    The four probabilities are independent per individual and per generation, but at most
    one structural mutation is applied per individual: one shared
    ``evoxc.random.uniform(key, (4, P))`` draw gates all of them and the first operation
    that hits wins.  A probability of ``0`` therefore removes its operation from the draw's
    reach entirely (an all-``False`` row) — and, because the plan is built per operation,
    from the traced graph as well.

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

        The chosen connection is consumed, the new node goes to ``first_free_row`` of the node
        table with the node gene's identity attributes, and two connections are created: an
        identity one from the old source (``conn_gene.new_identity_attrs``) and a copy of the
        consumed one into the old destination (with the recorded original attributes).  The
        two connection rows are the *first* and *second* free rows once the consumed row
        counts as free, which is exactly what the sequential formulation produced.

        No candidate when ``gate`` is false, when there is no connection to consume, when the
        node table is full, or when fewer than two connection rows are free.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing the identity attribute factory.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this operation's draw.
            context (EditContext): Shared masks and flat index blocks of the stage.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes.
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected add-node".

        Returns:
            AddNodePlan: the candidate rows and values; ``ok`` is false for a refused split.
        """
        pop_size = node_keys.shape[0]
        key_width = conn_keys.shape[2]
        attr_width = conn_attrs.shape[2]
        delete_row, has_connection = random_row(key, context.present_conn)
        key_row = take_block(
            context.conn_key_flat,
            context.conn_key_offsets,
            delete_row,
            context.conn_key_columns,
            key_width,
        )
        copy_attrs = take_block(
            context.conn_attr_flat,
            context.conn_attr_offsets,
            delete_row,
            context.conn_attr_columns,
            attr_width,
        )
        node_row, has_node = first_row(context.free_node)
        # The two new edges take the first two rows that are free *before* the split, which
        # is exactly the documented "refused unless two connection rows are free" rule: the
        # second ``first_row`` has no hit when only one row was free.
        identity_row, _ = first_row(context.free_conn)
        remaining = evoxc.logical_and(
            context.free_conn,
            evoxc.logical_not(
                evoxc.equal(
                    context.conn_row_index, enp.reshape(identity_row, (pop_size, 1))
                )
            ),
        )
        copy_row, has_copy = first_row(remaining)
        ok = evoxc.logical_and(
            evoxc.logical_and(gate, has_connection),
            evoxc.logical_and(has_node, has_copy),
        )
        return AddNodePlan(
            ok,
            node_row,
            delete_row,
            identity_row,
            copy_row,
            evoxc.reshape(evoxc.slice(key_row, (0, 0), (pop_size, 1)), (pop_size,)),
            evoxc.reshape(evoxc.slice(key_row, (0, 1), (pop_size, 1)), (pop_size,)),
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
            context (EditContext): Shared masks and flat index blocks of the stage.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected
                delete-node".

        Returns:
            DeleteNodePlan: the row to clear and the (already gated) detach mask.
        """
        pop_size = node_keys.shape[0]
        node_row, has_node = random_row(key, context.deletable_node)
        deleted_key = take_rows(context.node_flat, context.node_offsets, node_row)
        target = enp.reshape(deleted_key, (pop_size, 1))
        touches = evoxc.logical_and(
            evoxc.logical_or(
                evoxc.equal(conn_keys[:, :, 0], target),
                evoxc.equal(conn_keys[:, :, 1], target),
            ),
            context.present_conn,
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
            context (EditContext): Shared masks and flat index blocks of the stage.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated connection keys.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected add-conn".

        Returns:
            AddConnPlan: the candidate row and endpoints; ``ok`` is false for a refused add.
        """
        pop_size, node_count = node_keys.shape
        candidates = enp.concatenate(
            [
                enp.reshape(context.present_node, (pop_size, 1, node_count)),
                enp.reshape(context.targetable_node, (pop_size, 1, node_count)),
            ],
            axis=1,
        )
        rows, hits = _fetch_two(key, candidates)
        endpoints = take_rows(context.node_flat, context.node_offsets, rows)
        in_keys = evoxc.reshape(evoxc.slice(endpoints, (0, 0), (pop_size, 1)), (pop_size,))
        out_keys = evoxc.reshape(evoxc.slice(endpoints, (0, 1), (pop_size, 1)), (pop_size,))
        conn_row, has_room = first_row(context.free_conn)
        in_column = enp.reshape(in_keys, (pop_size, 1))
        out_column = enp.reshape(out_keys, (pop_size, 1))
        duplicate = evoxc.reduce_max(
            evoxc.logical_and(
                context.present_conn,
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
                        context.present_conn, evoxc.equal(conn_keys[:, :, 1], out_column)
                    ),
                    evoxc.int32,
                ),
                axes=1,
            ),
            evoxc.int32,
        )
        allowed = evoxc.logical_and(
            evoxc.logical_and(gate, has_room),
            evoxc.logical_and(
                evoxc.reshape(evoxc.slice(hits, (0, 0), (pop_size, 1)), (pop_size,)),
                evoxc.reshape(evoxc.slice(hits, (0, 1), (pop_size, 1)), (pop_size,)),
            ),
        )
        allowed = evoxc.logical_and(
            evoxc.logical_and(allowed, evoxc.logical_not(duplicate)),
            evoxc.less(in_degree, context.in_degree_limit),
        )
        if genome.network_type == "feedforward":
            allowed = evoxc.logical_and(
                allowed,
                evoxc.logical_not(closes_cycle(node_keys, conn_keys, in_keys, out_keys)),
            )
        return AddConnPlan(allowed, conn_row, in_keys, out_keys)

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
            context (EditContext): Shared masks and flat index blocks of the stage.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected
                delete-conn".

        Returns:
            DeleteConnPlan: the row to clear; ``ok`` is false when there is none.
        """
        conn_row, has_connection = random_row(key, context.present_conn)
        return DeleteConnPlan(evoxc.logical_and(gate, has_connection), conn_row)


__all__ = ["DefaultMutation"]
