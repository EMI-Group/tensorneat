"""Compatibility distance contract: one scalar distance per parent pair, batch-first.

The algorithm forms the pairs, so the kernel sees a population of parent pairs on axis 0 and
returns ``(P,)`` distances.  Alignment of homologous genes is done with one sort per gene
family (never an ``O(M^2)`` scan) and every count is a ``reduce_*`` over the gene axis; no
operator loops over individuals and nothing here is differentiable.
"""

from typing import TYPE_CHECKING

import evoxc

from ....common import State, StatefulBaseClass, TensorNEATError

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps genome -> operations acyclic
    from ...base import BaseGenome


class BaseDistance(StatefulBaseClass):
    """Pairwise compatibility distance between two batches of genomes.

    The result is additive over gene families — node distance plus connection distance — and
    normalized per pair, so pairs with different gene counts stay comparable and speciation
    thresholds are meaningful across a population.  Empty pairs contribute ``0`` rather than
    ``NaN``.
    """

    def __call__(
        self,
        state: State,
        genome: "BaseGenome",
        node_keys1: evoxc.SymbolicTensor,
        node_attrs1: evoxc.SymbolicTensor,
        conn_keys1: evoxc.SymbolicTensor,
        conn_attrs1: evoxc.SymbolicTensor,
        node_keys2: evoxc.SymbolicTensor,
        node_attrs2: evoxc.SymbolicTensor,
        conn_keys2: evoxc.SymbolicTensor,
        conn_attrs2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Compute one compatibility distance per parent pair.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning ``node_gene``/``conn_gene`` (their key
                widths and ``distance`` hooks).
            node_keys1 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys of parent 1,
                ``I_INF`` padded.
            node_attrs1 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs of parent
                1, NaN padded.
            conn_keys1 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys of parent 1,
                ``I_INF`` padded.
            conn_attrs1 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs of parent
                1, NaN padded.
            node_keys2 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys of parent 2,
                ``I_INF`` padded.
            node_attrs2 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs of parent
                2, NaN padded.
            conn_keys2 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys of parent 2,
                ``I_INF`` padded.
            conn_attrs2 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs of parent
                2, NaN padded.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` compatibility distances for the batch of
            parent pairs.

        Raises:
            TensorNEATError: If the two parents' payload shapes differ.
        """
        raise NotImplementedError(
            "sum the node-gene and conn-gene distances over the population axis"
        )


__all__ = ["BaseDistance"]
