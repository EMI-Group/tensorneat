"""Mutation contract for the batched NEAT gene payload.

A mutation operator is a trace-time kernel over a whole population: the four payload
arrays enter and leave with identical static shapes, the population lives on axis 0 and
every gate is an elementwise mask — no Python loop over individuals, no ``vmap``, no
``scan``/``while_loop``/``cond``.
Structural keys are caller-allocated ``int32`` tensors, and the only randomness consumed
here is one rank-0 ``evoxc`` key, so a ``(seed, config, shape)`` triple reproduces a run.
"""

from typing import TYPE_CHECKING

import evoxc

from ....common import State, StatefulBaseClass, TensorNEATError
from ...utils import GenomeTensors

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps genome -> operations acyclic
    from ...base import BaseGenome


class BaseMutation(StatefulBaseClass):
    """Batched mutation operator: one structural change plus value mutation.

    Subclasses implement the two stages the kernel is factored into:
    :meth:`mutate_structure` (at most one structural change per individual, gated by the
    operator's probabilities) and :meth:`mutate_values` (attribute mutation of every
    existing gene).  Both stages are batch-wide: individuals whose gate is off, or whose
    candidate is invalid, come out of the kernel unchanged rather than being skipped by a
    Python branch.

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
            TensorNEATError: If ``new_node_keys`` is not ``(P,)`` or ``new_conn_keys`` is
                not ``(P, 3)`` for the population size ``P`` of the payload arrays.
        """
        raise NotImplementedError(
            "split the key, apply mutate_structure then mutate_values, return the payload"
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
    ) -> GenomeTensors:
        """Apply at most one structural change per individual.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome providing gene objects and static capacity
                invariants.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key; subclasses draw one batched
                gate vector plus one key per structural operation from it.
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
            NotImplementedError: Always — this method is abstract.
        """
        raise NotImplementedError(
            "run one gated structural mutation per individual and return the payload"
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
    ) -> GenomeTensors:
        """Mutate the attributes of every existing gene in the population.

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

        Raises:
            NotImplementedError: Always — this method is abstract.
        """
        raise NotImplementedError(
            "mutate every gene attribute with one batched draw and restore padding sentinels"
        )


__all__ = ["BaseMutation"]
