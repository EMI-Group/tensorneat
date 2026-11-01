"""Mutation variant whose add-connection step is directed at recurrent (cyclic) edges.

Inherits the structural/value mutation of :class:`~.default.DefaultMutation` unchanged and
replaces only the add-connection helper: candidate endpoint pairs are drawn as one batched
``randint`` tableau and the first accepted row is selected with a reduce (never a ``cond``
region), so the whole population is still resolved by one batch-wide kernel.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import I_INF, ConfigError, State, flat_gather, masked_flat_gather
from ...utils import (
    GenomeTensors,
    add_conn,
    first_free_row,
)
from .base import _broadcast_row, _gated, _int32
from .default import DefaultMutation, _edge_keys

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
        """Add the first acceptable candidate edge per individual, with a cycle bias.

        Every candidate is resolved at once: one
        ``evoxc.random.randint(key, (P, max_conn_tries, 2), 0, N)`` draw supplies the
        endpoint *rows* of all attempts of all individuals (no per-attempt Python loop over
        individuals and no per-individual key split), the rows are turned into node keys with
        one flat gather each, and one ``uniform(key, (P, max_conn_tries))`` draw decides
        which attempts *must* close a cycle.

        An attempt is accepted when both endpoints exist, the destination is not an input
        row (inputs never take an incoming edge), the ``(in_key, out_key)`` pair does not
        already exist, the destination's in-degree stays below ``genome.max_in_degree``, and
        — when its ``p_recur`` decision fired — ``genome.check_cycles`` reports the edge as
        cycle-closing.  The *first* accepted attempt wins, selected with
        ``where(accept, attempt_index, I_INF)`` reduced to its minimum rather than a ``cond``
        region, and the edge is written at ``first_free_row`` with
        ``conn_gene.new_zero_attrs``.  No-op for an individual when its ``conn_add`` gate is
        off, when no attempt is accepted, or when no connection row is free.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing ``max_in_degree``, ``network_type``,
                ``conn_gene.new_zero_attrs`` and the cycle predicate ``check_cycles``.
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
            GenomeTensors: the four payload arrays; an individual without an accepted
            candidate is unchanged.

        Raises:
            evoxc.ShapeError / evoxc.BackendError: Propagated unchanged from
                ``genome.check_cycles`` when the cycle predicate is needed and the backend
                refuses the traversal; never caught or re-wrapped here.
        """
        pop_size, node_count = node_keys.shape
        edge_count = conn_keys.shape[1]
        tries = self.max_conn_tries
        attempt_key, recur_key = evoxc.random.split_n(key, 2)
        rows = evoxc.random.randint(
            attempt_key, (pop_size, tries, 2), 0, node_count, dtype=evoxc.int32
        )
        source_rows = rows[:, :, 0]
        destination_rows = rows[:, :, 1]
        # A node key of I_INF means "no node on that row", so an absent endpoint is a key.
        in_keys = flat_gather(node_keys, source_rows)
        out_keys = flat_gather(node_keys, destination_rows)
        present_edges = enp.reshape(
            evoxc.not_equal(conn_keys[:, :, 0], _int32(I_INF)),
            (pop_size, 1, edge_count),
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
                evoxc.not_equal(in_keys, _int32(I_INF)),
                evoxc.not_equal(out_keys, _int32(I_INF)),
            ),
            evoxc.logical_and(
                # Input rows are the first `num_inputs` rows and never take an in-edge.
                evoxc.greater_equal(destination_rows, _int32(genome.num_inputs)),
                evoxc.logical_not(duplicate),
            ),
        )
        accepted = evoxc.logical_and(
            accepted, evoxc.less(in_degree, _int32(genome.max_in_degree))
        )
        if genome.network_type == "feedforward":
            # A feedforward genome must stay acyclic, which also makes the `p_recur`
            # requirement unsatisfiable — no cycle-closing edge is ever accepted.
            accepted = evoxc.logical_and(
                accepted,
                evoxc.logical_not(
                    self._closes_cycle(genome, node_keys, conn_keys, in_keys, out_keys)
                ),
            )
        elif self.p_recur > 0.0:
            required = evoxc.less(
                evoxc.random.uniform(
                    recur_key, (pop_size, tries), 0.0, 1.0, dtype=evoxc.float32
                ),
                enp.full((), self.p_recur, dtype=evoxc.float32),
            )
            accepted = evoxc.logical_and(
                accepted,
                evoxc.logical_or(
                    evoxc.logical_not(required),
                    self._closes_cycle(genome, node_keys, conn_keys, in_keys, out_keys),
                ),
            )
        attempt = evoxc.reduce_min(
            evoxc.select(
                accepted,
                enp.reshape(enp.arange(0, tries, dtype=evoxc.int32), (1, tries)),
                _int32(I_INF),
            ),
            axes=1,
        )
        chosen = evoxc.not_equal(attempt, _int32(I_INF))
        conn_row = first_free_row(conn_keys)
        allowed = evoxc.logical_and(
            gate,
            evoxc.logical_and(chosen, evoxc.not_equal(conn_row, _int32(I_INF))),
        )
        added_keys, added_attrs = add_conn(
            conn_keys,
            conn_attrs,
            conn_row,
            _edge_keys(
                masked_flat_gather(in_keys, attempt, mask=chosen, fill=I_INF),
                masked_flat_gather(out_keys, attempt, mask=chosen, fill=I_INF),
                new_conn_keys,
                2,
                genome,
            ),
            _broadcast_row(genome.conn_gene.new_zero_attrs(state), pop_size),
        )
        return GenomeTensors(
            node_keys,
            node_attrs,
            _gated(allowed, added_keys, conn_keys),
            _gated(allowed, added_attrs, conn_attrs),
        )

    @staticmethod
    def _closes_cycle(
        genome: "BaseGenome",
        node_keys: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        in_keys: evoxc.SymbolicTensor,
        out_keys: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Ask the genome's cycle predicate once per candidate column.

        ``check_cycles`` takes one ``(P,)`` source/destination pair per individual, so the
        ``(P, max_conn_tries)`` candidate tableau is walked column by column at *trace*
        time (a static Python loop over at most ``max_conn_tries`` columns, never a loop
        over individuals) and the answers are stacked back into ``(P, max_conn_tries)``.
        Each column costs one reachability relaxation, so ``max_conn_tries`` bounds this
        step's cost — it is skipped entirely when no requirement depends on it.

        Args:
            genome ("BaseGenome"): Genome owning the ``check_cycles`` predicate.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            in_keys (evoxc.SymbolicTensor): ``(P, T) int32`` candidate source keys.
            out_keys (evoxc.SymbolicTensor): ``(P, T) int32`` candidate destination keys.

        Returns:
            ``(P, T) bool``: True where adding that candidate edge would close a cycle.
        """
        pop_size, tries = in_keys.shape
        return enp.stack(
            [
                genome.check_cycles(
                    node_keys,
                    conn_keys,
                    evoxc.reshape(in_keys[:, column : column + 1], (pop_size,)),
                    evoxc.reshape(out_keys[:, column : column + 1], (pop_size,)),
                )
                for column in range(tries)
            ],
            axis=-1,
        )


__all__ = ["RecurrentMutation"]
