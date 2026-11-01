"""Mutation variant whose add-connection step is directed at recurrent (cyclic) edges.

Inherits the structural/value mutation of :class:`~.default.DefaultMutation` unchanged and
overrides only the add-connection strategy: candidate endpoint pairs are drawn as one batched
``randint`` tableau and the first accepted row is selected with a reduce (never a ``cond``
region), so the whole population is still resolved by one batch-wide kernel.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import I_INF, State, TensorNEATError
from ...utils import add_conn, fetch_random_index, first_free_row, valid_count
from .base import Payload
from .default import DefaultMutation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome


class RecurrentMutation(DefaultMutation):
    """Add-connection mutation that *forces* a cycle with probability ``p_recur``.

    Node add/delete and connection delete behave exactly as in
    :class:`~.default.DefaultMutation`; only the add-connection step changes.  Instead of one
    candidate pair per individual, the helper samples a bounded tableau of candidate
    ``(src, dst)`` rows and keeps the first acceptable one, which lets the operator insist on
    a cycle-closing edge rather than merely bias the draw toward one.

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
        TensorNEATError: If ``p_recur`` is outside ``[0, 1]``, if ``max_conn_tries < 1``, or if
            an inherited probability is outside ``[0, 1]``.
    """

    def __init__(
        self,
        *,
        p_recur: float = 0.1,
        max_conn_tries: int = 20,
        **kwargs,
    ):
        """Store ``p_recur``/``max_conn_tries`` and forward the inherited probabilities.

        Raises:
            TensorNEATError: If ``p_recur`` is outside ``[0, 1]``, if ``max_conn_tries < 1``,
                or if an inherited probability is outside ``[0, 1]``.
        """
        raise NotImplementedError(
            "validate p_recur/max_conn_tries, store them and forward **kwargs to DefaultMutation"
        )

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
    ) -> Payload:
        """Apply at most one gated structural mutation, with the recurrent add-connection.

        The gate draw, the add-node / delete-node / delete-conn helpers and their no-op
        conditions are those of :class:`~.default.DefaultMutation`; only add-conn is replaced.

        The replacement draws every candidate at once with one
        ``evoxc.random.randint(key, (max_conn_tries, 2), 0, N)`` batch — one row per attempt
        per individual, no per-attempt Python loop and no per-individual key split.  Per
        attempt, a ``p_recur`` draw decides whether that attempt *must* close a cycle; an
        attempt is accepted when the edge is not a duplicate, respects the ``max_in_degree``
        cap, and satisfies its cycle requirement.  The *first* accepted row wins, selected
        with a reduce over the attempt axis (``where(accept, attempt_row, I_INF)`` reduced to
        its minimum) rather than a ``cond`` region, and the edge is written at
        ``first_free_row`` with ``conn_gene.new_zero_attrs``.  No-op for an individual when
        its ``conn_add`` gate is off, when no attempt is accepted, or when no connection row
        is free.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing ``max_in_degree``, ``network_type``,
                ``conn_gene.new_zero_attrs`` and the cycle predicate ``check_cycles``.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this stage.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs (NaN padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs (NaN padded).
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys at most.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated conn keys:
                ``[:, 0]``/``[:, 1]`` for the add-node edges, ``[:, 2]`` for add-conn (written
                as the ``historical_marker`` when the conn gene declares one).

        Returns:
            Payload: the four payload arrays after the structural stage.

        Raises:
            TensorNEATError: If ``genome.network_type`` is neither ``"feedforward"`` nor
                ``"recurrent"``.
        """
        raise NotImplementedError(
            "draw the (max_conn_tries, 2) candidate tableau and keep the first accepted row"
        )


__all__ = ["RecurrentMutation"]
