"""Mutation variant whose add-connection step is directed at recurrent (cyclic) edges.

Inherits the structural/value mutation of :class:`~.default.DefaultMutation` unchanged and
replaces only the add-connection helper: candidate endpoint pairs are drawn as one batched
``randint`` tableau and the first accepted row is selected with a reduce (never a ``cond``
region), so the whole population is still resolved by one batch-wide kernel.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import ConfigError, State
from .arrays import first_row, i32, row_offsets, take_rows
from .cycle import closes_cycle
from .default import DefaultMutation
from .plan import AddConnPlan

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome


def _pick_cell(
    table: evoxc.SymbolicTensor,
    index: evoxc.SymbolicTensor,
    pop_size: int,
    width: int,
) -> evoxc.SymbolicTensor:
    """Gather one cell per individual out of a ``(P, width)`` table with a flat index.

    The index block is built once by the caller (``row_offsets`` of the same width plus the
    chosen column) and shared by every picked table, so a second and third pick pay only the
    flatten, the gather and the reshape — never another index computation.

    Args:
        table (evoxc.SymbolicTensor): ``(P, width)`` source table.
        index (evoxc.SymbolicTensor): ``(P,) int32`` flat index into the flattened table,
            already in ``[0, pop_size * width)`` by construction.
        pop_size (int): Population size ``P`` (static).
        width (int): Row width of ``table`` (static).

    Returns:
        evoxc.SymbolicTensor: ``(P,)`` tensor of ``table``'s dtype holding the picked cells.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.
    """
    return enp.reshape(
        evoxc.gather(enp.reshape(table, (pop_size * width,)), index, axis=0),
        (pop_size,),
    )


class RecurrentMutation(DefaultMutation):
    """Add-connection mutation that *forces* a cycle with probability ``p_recur``.

    Node add/delete and connection delete behave exactly as in
    :class:`~.default.DefaultMutation`; only the add-connection helper changes.  Instead of
    one candidate pair per individual, it samples a bounded tableau of candidate
    ``(src, dst)`` rows and keeps the first acceptable one, which lets the operator insist
    on a cycle-closing edge rather than merely bias the draw toward one.

    Args:
        p_recur (float): Probability that an accepted add-connection *must* close a cycle.
            When the flag fires, a candidate edge that does not close a cycle is discarded —
            the requirement is hard, not a bias.  A value of ``0`` degenerates to the
            :class:`~.default.DefaultMutation` behaviour.
        max_conn_tries (int): Number of candidate ``(src, dst)`` rows drawn per individual and
            generation; bounds the work of the add-connection step independently of the
            population size.
        **kwargs: Forwarded to :class:`~.default.DefaultMutation`
            (``conn_add``, ``conn_delete``, ``node_add``, ``node_delete``).

    Raises:
        TensorNEATError: If ``p_recur`` is not a real number in ``[0, 1]``, if
            ``max_conn_tries`` is not an ``int`` of at least ``1``, or if an inherited
            probability is outside ``[0, 1]``.
    """

    def __init__(
        self,
        *,
        p_recur: float = 0.1,
        max_conn_tries: int = 20,
        **kwargs,
    ):
        """Store ``p_recur``/``max_conn_tries`` and forward the inherited probabilities.

        Args:
            p_recur (float): Probability that an accepted edge must close a cycle.
            max_conn_tries (int): Size of the candidate tableau per individual.
            **kwargs: ``conn_add``/``conn_delete``/``node_add``/``node_delete``, validated
                by :class:`~.default.DefaultMutation`.

        Raises:
            TensorNEATError: If ``p_recur`` is not a real number in ``[0, 1]``, if
                ``max_conn_tries`` is not an ``int`` of at least ``1``, or if an inherited
                probability is outside ``[0, 1]``.
        """
        super().__init__(**kwargs)
        if isinstance(p_recur, bool) or not isinstance(p_recur, (int, float)):
            raise ConfigError(
                f"p_recur must be a real number in [0, 1], got {p_recur!r}"
            )
        if not 0.0 <= float(p_recur) <= 1.0:
            raise ConfigError(
                f"p_recur must be a real number in [0, 1], got {p_recur!r}"
            )
        if isinstance(max_conn_tries, bool) or not isinstance(max_conn_tries, int):
            raise ConfigError(
                f"max_conn_tries must be an int of at least 1, got {max_conn_tries!r}"
            )
        if max_conn_tries < 1:
            raise ConfigError(
                f"max_conn_tries must be an int of at least 1, got {max_conn_tries!r}"
            )
        self.p_recur = float(p_recur)
        self.max_conn_tries = int(max_conn_tries)

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
        """Resolve the first acceptable candidate edge per individual, with a cycle bias.

        Every candidate is resolved at once: one
        ``evoxc.random.randint(key, (P, max_conn_tries, 2), 0, N)`` draw supplies the
        endpoint *rows* of all attempts of all individuals (no per-attempt Python loop over
        individuals and no per-individual key split), the rows are turned into node keys with
        **one** flat gather of the interleaved ``(src, dst)`` pair, and one
        ``uniform(key, (P, max_conn_tries))`` draw decides which attempts *must* close a
        cycle.

        An attempt is accepted when both endpoints exist, the destination is not an input
        row, the ``(in_key, out_key)`` pair does not already exist, the destination's
        in-degree stays below ``genome.max_in_degree``, and — when its ``p_recur`` decision
        fired — the edge closes a cycle.  The *first* accepted attempt wins, selected with a
        reduce over the attempt axis rather than a ``cond`` region.

        The cycle predicate is the only loop region here, and it is consulted by exactly two
        branches: the ``network_type == "feedforward"`` veto (always required) and the
        ``p_recur`` requirement, which exists only for a recurrent genome with
        ``p_recur > 0``.  Both are trace-time conditions (a Python string and a Python
        float), so a **recurrent** genome with ``p_recur == 0`` — the degenerate
        :class:`~.default.DefaultMutation` add-connection — skips the whole
        :func:`~.cycle.closes_cycle` region instead of tracing a dead result.

        Padding never enters the two connection-table slabs: a padded row is ``I_INF`` in
        *every* key column, while an accepted candidate endpoint is finite by construction
        (``exists``), so key equality alone excludes padded rows and the ``present_conn``
        factor the :class:`~.default.DefaultMutation` plan needs is redundant here.  An
        attempt whose endpoint *is* ``I_INF`` may match a padded row in the slabs, but every
        decision that reads them is already gated by ``exists``.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing ``max_in_degree`` and ``network_type``.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this operation's draws.
            context (EditContext): Shared masks and flat index blocks of the stage.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated connection keys;
                ``[:, 2]`` becomes the ``historical_marker`` of the new edge.
            gate (evoxc.SymbolicTensor): ``(P,) bool`` "this individual selected add-conn".

        Returns:
            AddConnPlan: the winning candidate row and endpoints; ``ok`` is false when no
            attempt was accepted or the connection table is full.
        """
        pop_size, node_count = node_keys.shape
        edge_count = conn_keys.shape[1]
        tries = self.max_conn_tries
        attempt_key, recur_key = evoxc.random.split_n(key, 2)
        rows = evoxc.random.randint(
            attempt_key, (pop_size, tries, 2), 0, node_count, dtype=evoxc.int32
        )
        # One flat gather of the whole `(P, tries, 2)` endpoint block: the drawn rows are
        # node rows in `[0, N)`, so they index the stage's already-flattened node keys
        # without a range check, and both key columns come out of a single gather.
        endpoints = take_rows(context.node_flat, context.node_offsets, rows)
        in_keys = enp.reshape(
            evoxc.slice(endpoints, (0, 0, 0), (pop_size, tries, 1)), (pop_size, tries)
        )
        out_keys = enp.reshape(
            evoxc.slice(endpoints, (0, 0, 1), (pop_size, tries, 1)), (pop_size, tries)
        )
        # Duplicate and in-degree slabs, both against the same `(P, tries, E)` output-key
        # comparison so the second `2 M`-element pass of the previous formulation is gone.
        out_column = enp.reshape(out_keys, (pop_size, tries, 1))
        same_out = evoxc.equal(
            enp.reshape(conn_keys[:, :, 1], (pop_size, 1, edge_count)), out_column
        )
        duplicate = evoxc.reduce_max(
            evoxc.logical_and(
                same_out,
                evoxc.equal(
                    enp.reshape(conn_keys[:, :, 0], (pop_size, 1, edge_count)),
                    enp.reshape(in_keys, (pop_size, tries, 1)),
                ),
            ),
            axes=2,
        )
        in_degree = evoxc.cast(evoxc.reduce_sum(same_out, axes=2), evoxc.int32)
        destination = enp.reshape(
            evoxc.slice(rows, (0, 0, 1), (pop_size, tries, 1)), (pop_size, tries)
        )
        accepted = evoxc.logical_and(
            evoxc.logical_and(
                evoxc.not_equal(in_keys, context.sentinel),
                evoxc.not_equal(out_keys, context.sentinel),
            ),
            evoxc.logical_and(
                # Input rows are the first `num_inputs` rows and never take an in-edge.
                evoxc.greater_equal(destination, i32(genome.num_inputs)),
                evoxc.logical_and(
                    evoxc.logical_not(duplicate),
                    evoxc.less(in_degree, context.in_degree_limit),
                ),
            ),
        )
        required = None
        if genome.network_type == "recurrent" and self.p_recur > 0.0:
            required = evoxc.less(
                evoxc.random.uniform(
                    recur_key, (pop_size, tries), 0.0, 1.0, dtype=evoxc.float32
                ),
                enp.full((), self.p_recur, dtype=evoxc.float32),
            )
        # The first accepted attempt: `argmax` of a boolean row is exactly its first `True`
        # (and `0` for an empty row, which `chosen` gates), so no `I_INF` sentinel and no
        # index sanitisation are needed — the index is always in `[0, tries)`.
        attempt = evoxc.cast(evoxc.argmax(accepted, axis=1), evoxc.int32)
        chosen = evoxc.reduce_max(accepted, axes=1)
        conn_row, has_room = first_row(context.free_conn)
        allowed = evoxc.logical_and(evoxc.logical_and(gate, has_room), chosen)
        # One index block, shared by every picked cell: `p * tries + attempt`.
        pick = enp.reshape(
            evoxc.add(enp.reshape(attempt, (pop_size, 1)), row_offsets(pop_size, tries)),
            (pop_size,),
        )
        in_key = _pick_cell(in_keys, pick, pop_size, tries)
        out_key = _pick_cell(out_keys, pick, pop_size, tries)
        # One cycle query for the *chosen* edge instead of one per attempt: the reachability
        # relaxation is the most expensive predicate here (a loop region), and a rejected
        # attempt costs a later one nothing but a re-draw.
        #
        # A static gate decides whether the query is needed at all: only the feedforward veto
        # below and the `p_recur` requirement consume `closes`, and `required` exists exactly
        # when `network_type == "recurrent"` and `p_recur > 0`.  For a recurrent genome with
        # `p_recur == 0` the add-connection degenerates to the plain draw, so the whole
        # region (and its ~106 traced ops) is skipped rather than traced and discarded.
        if genome.network_type == "feedforward" or self.p_recur > 0.0:
            closes = closes_cycle(
                node_keys, conn_keys, in_key, out_key, genome.key_span
            )
        else:
            closes = None
        if genome.network_type == "feedforward":
            # A feedforward genome must stay acyclic, which also makes the `p_recur`
            # requirement unsatisfiable — no cycle-closing edge is ever accepted.
            allowed = evoxc.logical_and(allowed, evoxc.logical_not(closes))
        elif required is not None:
            allowed = evoxc.logical_and(
                allowed,
                evoxc.logical_or(
                    evoxc.logical_not(_pick_cell(required, pick, pop_size, tries)),
                    closes,
                ),
            )
        return AddConnPlan(allowed, conn_row, in_key, out_key)


__all__ = ["RecurrentMutation"]
