"""NEAT-python style mutation: at most one structural change plus value mutation.

All four structural operations are batch-wide kernels: each gate is an elementwise mask on
the population axis, and an individual whose gate is off — or whose candidate is invalid —
comes out unchanged.  Capacity invariants are hard: ``max_nodes``, ``max_conns`` and the
genome's ``max_in_degree`` are never exceeded, a refused add is a no-op for that individual
(never a silent repair), and new rows go to ``first_free_row`` in the same layout for keys
and attributes.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import I_INF, State, TensorNEATError
from ...utils import (
    add_conn,
    add_node,
    delete_conn_by_pos,
    delete_node_by_pos,
    fetch_random_index,
    first_free_row,
    valid_count,
)
from .base import BaseMutation, Payload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome


class DefaultMutation(BaseMutation):
    """Structural and value mutation with per-operation probabilities.

    The four probabilities are independent per individual and per generation: one
    ``evoxc.random.uniform(key, (4, P))`` draw gates all of them, so at most one structural
    mutation per individual is applied while value mutation always runs afterwards.
    A probability of ``0`` removes its operation from the graph entirely.

    Args:
        conn_add (float): Probability per individual of attempting to add one connection.
        conn_delete (float): Probability per individual of attempting to delete a connection.
        node_add (float): Probability per individual of attempting a node split.
        node_delete (float): Probability per individual of attempting to delete a hidden node.

    Raises:
        TensorNEATError: If any probability is outside ``[0, 1]``.
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

        Raises:
            TensorNEATError: If any probability is outside ``[0, 1]``.
        """
        raise NotImplementedError(
            "store the four probabilities and reject values outside [0, 1]"
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
        """Apply at most one gated structural mutation per individual.

        Draws ``evoxc.random.uniform(key, (4, P))`` once; its rows gate add-node,
        delete-node, add-conn and delete-conn, each helper below being a batch-wide op that
        is a no-op for the individuals whose gate is off.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing genes, ``max_in_degree``, ``network_type``.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this stage.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs (NaN padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs (NaN padded).
            new_node_keys (evoxc.SymbolicTensor): ``(P,) int32`` allocated node keys at most.
            new_conn_keys (evoxc.SymbolicTensor): ``(P, 3) int32`` allocated conn keys:
                ``[:, 0]``/``[:, 1]`` for the add-node edges, ``[:, 2]`` for add-conn.

        Returns:
            Payload: the four payload arrays after the structural stage.

        Raises:
            TensorNEATError: If ``genome.network_type`` is neither ``"feedforward"`` nor
                ``"recurrent"``.
        """
        raise NotImplementedError(
            "draw the (4, P) gate matrix and run the four gated helpers in order"
        )

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
    ) -> Payload:
        """Split one connection per individual with a new node, preserving network output.

        The chosen connection is deleted, the new node is written at ``first_free_row`` with
        the node gene's identity attributes, then two connections are added: an identity one
        from the old source (``conn_gene.new_identity_attrs``) and a copy of the original one
        into the old destination (with the recorded original attributes).

        No-op for an individual when the ``node_add`` gate is off, when
        :meth:`choose_connection_key` finds no connection (``I_INF``), when
        ``first_free_row`` finds no free node row, or when fewer than two connection rows
        are free.

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

        Returns:
            Payload: the four payload arrays; an individual that cannot split is unchanged.
        """
        raise NotImplementedError(
            "choose a connection, delete it, insert the node and add the two replacement edges"
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
    ) -> Payload:
        """Delete one hidden node per individual together with all its connections.

        Input rows (``0..num_inputs-1``) and output rows (``N-num_outputs..N-1``) are never
        candidates, so :meth:`choose_node_key` is called with both ``allow_*_keys=False``;
        the node row is cleared with ``delete_node_by_pos`` and every connection touching the
        deleted key is cleared with ``delete_conn_by_pos``.

        No-op when the ``node_delete`` gate is off or when no hidden node exists
        (:meth:`choose_node_key` returns ``I_INF``).

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing the input/output row ranges.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this helper.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs (NaN padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs (NaN padded).

        Returns:
            Payload: the four payload arrays, with the deleted node's row and every touching
            connection row sent back to their padding sentinels.
        """
        raise NotImplementedError(
            "choose a hidden node, clear its row and every connection touching its key"
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
    ) -> Payload:
        """Add one connection per individual with the connection gene's zero attributes.

        Two independent :meth:`choose_node_key` draws pick the endpoints: the source may be
        any existing node (``allow_input_keys=True, allow_output_keys=True``), the
        destination any existing node that is not an input (``allow_input_keys=False``).

        No-op for an individual when the ``conn_add`` gate is off, when an endpoint is
        ``I_INF``, when ``first_free_row`` finds no free connection row, when the
        ``(in_key, out_key)`` pair already exists, when the edge would close a cycle and
        ``genome.network_type == "feedforward"`` (the genome's ``check_cycles`` predicate,
        equivalent to the legacy ``common.graph.check_cycles`` reachability test), or when it
        would push the destination's in-degree past ``genome.max_in_degree``.

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

        Returns:
            Payload: the four payload arrays; a refused add leaves the individual unchanged.
        """
        raise NotImplementedError(
            "draw both endpoints, reject duplicates/cycles/in-degree overflow and add the edge"
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
    ) -> Payload:
        """Delete one connection per individual.

        The candidate row comes from :meth:`choose_connection_key` and is cleared with
        ``delete_conn_by_pos``.  No-op when the ``conn_delete`` gate is off or when there is
        no connection to delete (``I_INF`` row).

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning the connection gene.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this helper.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs (NaN padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs (NaN padded).

        Returns:
            Payload: the four payload arrays, with the chosen connection row cleared.
        """
        raise NotImplementedError(
            "choose a connection row and clear it with delete_conn_by_pos"
        )

    def mutate_values(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
    ) -> Payload:
        """Mutate every existing gene's attributes with one batched draw per array.

        ``genome.node_gene.mutate`` and ``genome.conn_gene.mutate`` are called once each on
        the full ``(P, N, NA)`` / ``(P, E, CA)`` arrays (genes are batched over their leading
        axes); the result is re-masked so padded rows keep NaN, and key arrays are returned
        untouched.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome whose gene objects perform the attribute draw.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this stage.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            node_attrs (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs (NaN padded).
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).
            conn_attrs (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs (NaN padded).

        Returns:
            Payload: mutated attributes, unchanged keys, padding sentinels restored.
        """
        raise NotImplementedError(
            "call each gene's batched mutate once and re-mask padded rows to NaN"
        )

    def choose_node_key(
        self,
        key: evoxc.SymbolicTensor,
        node_keys: evoxc.SymbolicTensor,
        allow_input_keys: bool = False,
        allow_output_keys: bool = False,
    ) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
        """Pick one candidate node per individual, out of the eligible rows.

        Eligibility is an elementwise mask over ``node_keys``: absent rows (``I_INF``) are
        dropped, and input rows (``0..num_inputs-1``) / output rows
        (``N-num_outputs..N-1``) are dropped unless the matching flag is ``True``.  The draw
        is a single batched ``fetch_random_index`` over the ``(P, N)`` mask — never a Python
        loop over individuals.

        Args:
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this draw.
            node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node keys (``I_INF`` padded).
            allow_input_keys (bool): Whether input rows may be chosen.
            allow_output_keys (bool): Whether output rows may be chosen.

        Returns:
            tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]: ``(keys, rows)``, both
            ``(P,) int32``, with ``I_INF`` in both entries where no candidate exists.
        """
        raise NotImplementedError(
            "build the eligibility mask and draw one row per individual"
        )

    def choose_connection_key(
        self,
        key: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
    ) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
        """Pick one candidate connection per individual, out of the existing rows.

        Args:
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this draw.
            conn_keys (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys (``I_INF`` padded).

        Returns:
            tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
            ``(in_keys, out_keys, rows)``, all ``(P,) int32``, with ``I_INF`` in all three
            entries for individuals that have no connection.
        """
        raise NotImplementedError(
            "draw one existing connection row per individual and gather its in/out keys"
        )


__all__ = ["DefaultMutation"]
