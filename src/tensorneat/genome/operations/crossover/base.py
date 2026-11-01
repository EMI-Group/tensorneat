"""Crossover contract: key-aligned recombination of two batches of parent genomes.

Crossover is a trace-time kernel: both parents are ``(P, ...)`` payload batches, parent 1
being the fitter one, so a gene of the offspring is taken from parent 1 unless it has a
homologue in parent 2.
Alignment is by exact key-column match, resolved with one sort per array — never an
``O(M^2)`` scan — and no operator loops over individuals or allocates keys.
"""

from typing import TYPE_CHECKING

import evoxc

from ....common import State, StatefulBaseClass, TensorNEATError

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps genome -> operations acyclic
    from ...base import BaseGenome

#: The offspring payload: ``node_keys``, ``node_attrs``, ``conn_keys``, ``conn_attrs``.
Offspring = tuple[
    evoxc.SymbolicTensor,  # node_keys  (P, N) int32
    evoxc.SymbolicTensor,  # node_attrs (P, N, NA) float32
    evoxc.SymbolicTensor,  # conn_keys  (P, E, CK) int32
    evoxc.SymbolicTensor,  # conn_attrs (P, E, CA) float32
]


class BaseCrossover(StatefulBaseClass):
    """Recombine two parent genomes into one offspring genome, batch-first.

    The offspring takes parent 1's structure (``node_keys``/``conn_keys``) verbatim and only
    its float attributes may come from parent 2: a gene present in parent 2 under the same
    key columns is recombined attribute-by-attribute by the gene's own ``crossover`` hook, a
    gene absent there keeps parent 1's attributes unchanged.  Padding rows stay padded
    (``I_INF`` keys, NaN attributes), so an offspring never gains or loses rows.
    """

    def __call__(
        self,
        state: State,
        genome: "BaseGenome",
        key: evoxc.SymbolicTensor,
        node_keys1: evoxc.SymbolicTensor,
        node_attrs1: evoxc.SymbolicTensor,
        conn_keys1: evoxc.SymbolicTensor,
        conn_attrs1: evoxc.SymbolicTensor,
        node_keys2: evoxc.SymbolicTensor,
        node_attrs2: evoxc.SymbolicTensor,
        conn_keys2: evoxc.SymbolicTensor,
        conn_attrs2: evoxc.SymbolicTensor,
    ) -> Offspring:
        """Cross a batch of parent pairs into a batch of offspring.

        Parent 1 is the **fitter** parent: its keys define the offspring's structure and its
        attributes survive wherever parent 2 has no homologue.  The single rank-0 key is
        split internally into one key per gene family (nodes, connections).

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning the gene objects, their ``key_width``s and
                the ``crossover`` hooks.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this call.
            node_keys1 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys of parent 1
                (fitter parent), ``I_INF`` padded.
            node_attrs1 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes of
                parent 1, NaN padded.
            conn_keys1 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys of
                parent 1, ``I_INF`` padded.
            conn_attrs1 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes
                of parent 1, NaN padded.
            node_keys2 (evoxc.SymbolicTensor): ``(P, N, NK) int32`` node keys of parent 2,
                ``I_INF`` padded.
            node_attrs2 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes of
                parent 2, NaN padded.
            conn_keys2 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys of
                parent 2, ``I_INF`` padded.
            conn_attrs2 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes
                of parent 2, NaN padded.

        Returns:
            Offspring: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` — parent 1's keys
            with the recombined attributes.

        Raises:
            TensorNEATError: If the two parents' payload shapes differ (crossover does not
                resize genomes; ``max_nodes``/``max_conns`` are build-time constants).
        """
        raise NotImplementedError(
            "assert matching parent shapes and recombine each gene family against parent 1"
        )


__all__ = ["BaseCrossover", "Offspring"]
