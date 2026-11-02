"""Mutation variant whose add-connection step is directed at recurrent (cyclic) edges.

Inherits the structural/value mutation of :class:`~.default.DefaultMutation` unchanged and
replaces only the add-connection helper: candidate endpoint pairs are drawn as one batched
``randint`` tableau and the first accepted row is selected with a reduce (never a ``cond``
region), so the whole population is still resolved by one batch-wide kernel.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import I_INF, ConfigError, State, fetch_first, flat_gather, masked_flat_gather
from .arrays import i32
from .cycle import closes_cycle
from .default import DefaultMutation
from .plan import AddConnPlan

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome


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
        one flat gather each, and one ``uniform(key, (P, max_conn_tries))`` draw decides
        which attempts *must* close a cycle.

        An attempt is accepted when both endpoints exist, the destination is not an input
        row, the ``(in_key, out_key)`` pair does not already exist, the destination's
        in-degree stays below ``genome.max_in_degree``, and — when its ``p_recur`` decision
        fired — the edge closes a cycle.  The *first* accepted attempt wins, selected with a
        reduce over the attempt axis rather than a ``cond`` region.

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
        source_rows = evoxc.slice(rows, (0, 0, 0), (pop_size, tries, 1))
        destination_rows = evoxc.slice(rows, (0, 0, 1), (pop_size, tries, 1))
        source = enp.reshape(source_rows, (pop_size, tries))
        destination = enp.reshape(destination_rows, (pop_size, tries))
        in_keys = flat_gather(node_keys, source)
        out_keys = flat_gather(node_keys, destination)
        present_edges = enp.reshape(
            context.present_conn, (pop_size, 1, edge_count)
        )
        existing_in = enp.reshape(conn_keys[:, :, 0], (pop_size, 1, edge_count))
        existing_out = enp.reshape(conn_keys[:, :, 1], (pop_size, 1, edge_count))
        in_column = enp.reshape(in_keys, (pop_size, tries, 1))
        out_column = enp.reshape(out_keys, (pop_size, tries, 1))
        duplicate = evoxc.reduce_max(
            evoxc.logical_and(
                present_edges,
                evoxc.logical_and(
                    evoxc.equal(existing_in, in_column),
                    evoxc.equal(existing_out, out_column),
                ),
            ),
            axes=2,
        )
        in_degree = evoxc.cast(
            evoxc.reduce_sum(
                evoxc.cast(
                    evoxc.logical_and(
                        present_edges, evoxc.equal(existing_out, out_column)
                    ),
                    evoxc.int32,
                ),
                axes=2,
            ),
            evoxc.int32,
        )
        accepted = evoxc.logical_and(
            evoxc.logical_and(
                evoxc.not_equal(in_keys, i32(I_INF)),
                evoxc.not_equal(out_keys, i32(I_INF)),
            ),
            evoxc.logical_and(
                # Input rows are the first `num_inputs` rows and never take an in-edge.
                evoxc.greater_equal(destination, i32(genome.num_inputs)),
                evoxc.logical_not(duplicate),
            ),
        )
        accepted = evoxc.logical_and(
            accepted, evoxc.less(in_degree, i32(genome.max_in_degree))
        )
        required = None
        if genome.network_type == "recurrent" and self.p_recur > 0.0:
            required = evoxc.less(
                evoxc.random.uniform(
                    recur_key, (pop_size, tries), 0.0, 1.0, dtype=evoxc.float32
                ),
                enp.full((), self.p_recur, dtype=evoxc.float32),
            )
        attempt = evoxc.reduce_min(
            evoxc.select(
                accepted,
                enp.reshape(enp.arange(0, tries, dtype=evoxc.int32), (1, tries)),
                i32(I_INF),
            ),
            axes=1,
        )
        chosen = evoxc.not_equal(attempt, i32(I_INF))
        conn_row = fetch_first(context.free_conn)
        allowed = evoxc.logical_and(
            gate,
            evoxc.logical_and(chosen, evoxc.not_equal(conn_row, i32(I_INF))),
        )
        in_key = masked_flat_gather(in_keys, attempt, mask=chosen, fill=I_INF)
        out_key = masked_flat_gather(out_keys, attempt, mask=chosen, fill=I_INF)
        # One cycle query for the *chosen* edge instead of one per attempt: the reachability
        # relaxation is the most expensive predicate here (a loop region), and a rejected
        # attempt costs a later one nothing but a re-draw.
        closes = closes_cycle(node_keys, conn_keys, in_key, out_key)
        if genome.network_type == "feedforward":
            # A feedforward genome must stay acyclic, which also makes the `p_recur`
            # requirement unsatisfiable — no cycle-closing edge is ever accepted.
            allowed = evoxc.logical_and(allowed, evoxc.logical_not(closes))
        elif required is not None:
            allowed = evoxc.logical_and(
                allowed,
                evoxc.logical_or(
                    evoxc.logical_not(
                        masked_flat_gather(
                            required, attempt, mask=chosen, fill=False
                        )
                    ),
                    closes,
                ),
            )
        return AddConnPlan(allowed, conn_row, in_key, out_key)
__all__ = ["RecurrentMutation"]
