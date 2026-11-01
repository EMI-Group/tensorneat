"""Compatibility distance with disjoint/excess and attribute terms.

``distance = (non_homologous_count * compatibility_disjoint
+ homologous_attribute_distance * compatibility_weight) / max(cnt1, cnt2)``

Alignment of homologous genes is one sorted comparison per gene family (``O(M log M)``),
all counts are ``reduce_*`` over the gene axis, and the whole population of parent pairs is
resolved in a single batched kernel.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import State, TensorNEATError
from .base import BaseDistance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome
    from ...gene import BaseGene


class DefaultDistance(BaseDistance):
    """Sum of node-gene and connection-gene compatibility distance.

    The two terms are additive: each gene family aligns its homologous keys, counts the genes
    without a homologue (disjoint + excess) and sums the per-gene attribute distance over the
    homologous pairs, then normalizes by the larger gene count so that pairs of structurally
    different sizes stay comparable.

    Args:
        compatibility_disjoint (float): Weight of the non-homologous gene count.
        compatibility_weight (float): Weight of the homologous attribute distance.

    Raises:
        TensorNEATError: If either weight is negative.
    """

    def __init__(
        self,
        compatibility_disjoint: float = 1.0,
        compatibility_weight: float = 0.4,
    ):
        """Store the two distance weights.

        Raises:
            TensorNEATError: If either weight is negative.
        """
        raise NotImplementedError(
            "store the two weights and reject negative values with TensorNEATError"
        )

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
        """Return ``node_distance + conn_distance`` for each parent pair.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning ``node_gene``/``conn_gene``.
            node_keys1 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys, parent 1,
                ``I_INF`` padded.
            node_attrs1 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs, parent 1,
                NaN padded.
            conn_keys1 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys, parent 1,
                ``I_INF`` padded.
            conn_attrs1 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs, parent 1,
                NaN padded.
            node_keys2 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys, parent 2,
                ``I_INF`` padded.
            node_attrs2 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs, parent 2,
                NaN padded.
            conn_keys2 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys, parent 2,
                ``I_INF`` padded.
            conn_attrs2 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs, parent 2,
                NaN padded.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` distances, one per parent pair.
        """
        raise NotImplementedError(
            "add gene_distance over the node gene and the conn gene"
        )

    def gene_distance(
        self,
        state: State,
        gene: "BaseGene",
        keys1: evoxc.SymbolicTensor,
        attrs1: evoxc.SymbolicTensor,
        keys2: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Compatibility distance of one gene family between two batches of genomes.

        Aligns homologous genes with a single sorted comparison of the concatenated key
        columns (``O(M log M)``, the batched equivalent of ``np.intersect1d`` in higher
        dimension), so a key present in both parents becomes one adjacent pair and every
        unmatched gene is counted as non-homologous.

        Args:
            state (State): Immutable host configuration container.
            gene ("BaseGene"): Node or connection gene supplying key width and
                ``distance``.
            keys1 (evoxc.SymbolicTensor): ``(P, M, K) int32`` key columns, parent 1, ``I_INF``
                padded.
            attrs1 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes, parent 1, NaN
                padded.
            keys2 (evoxc.SymbolicTensor): ``(P, M, K) int32`` key columns, parent 2, ``I_INF``
                padded.
            attrs2 (evoxc.SymbolicTensor): ``(P, M, A) float32`` attributes, parent 2, NaN
                padded.

        Returns:
            evoxc.SymbolicTensor: ``(P,) float32`` where each entry is
            ``(non_homologous_count * compatibility_disjoint
            + homologous_attribute_distance * compatibility_weight) / max(cnt1, cnt2)``,
            guarded to ``0`` for pairs where both genomes are empty in this gene family.
        """
        raise NotImplementedError(
            "align both key sets by one sorted comparison and combine the two terms"
        )


__all__ = ["DefaultDistance"]
