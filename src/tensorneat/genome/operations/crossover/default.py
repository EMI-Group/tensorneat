"""Key-aligned crossover: parent 1 keeps its structure, attributes may come from parent 2.

Alignment is exact on the whole key column tuple of a gene (``(in_key, out_key)`` plus
``historical_marker`` for connections), found through a hash plus an ``argsort`` merge plus
an exact key-row confirmation of every candidate pair: ``O(M log M)`` per array, never an
``O(M^2)`` scan, and still one batched kernel for the whole population.
The merge itself lives in :mod:`..alignment`, shared with the compatibility distance, and
this module only turns its homologue table into recombined attribute rows.
"""

from typing import TYPE_CHECKING

import evoxc
import evoxc.numpy as enp

from ....common import I_INF, State, masked_flat_gather
from ...utils import GenomeTensors
from ..alignment import find_homologous
from .base import BaseCrossover

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...base import BaseGenome
    from ...gene import BaseGene


class DefaultCrossover(BaseCrossover):
    """Cross two parents per individual by exact key alignment.

    Parent 1 (the fitter parent) supplies ``node_keys``/``conn_keys`` verbatim; for each of
    its genes the homologous row in parent 2 is located and the float attributes are
    recombined by the gene's ``crossover`` hook (per-attribute 50/50 by default).  Genes with
    no homologue keep parent 1's attributes, and padded rows stay padded.
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
    ) -> GenomeTensors:
        """Cross node and connection genes of the two parents, batch-first.

        Splits the key into one key per gene family, computes the homologue table with
        :meth:`find_homologous` (once for nodes, once for connections) and recombines the
        attributes with :meth:`crossover_gene_attrs`; the key arrays of parent 1 are returned
        unchanged.

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning ``node_gene``/``conn_gene``.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this call.
            node_keys1 (evoxc.SymbolicTensor): ``(P, N) int32`` node keys, parent 1
                (fitter), ``I_INF`` padded.
            node_attrs1 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs, parent 1,
                NaN padded.
            conn_keys1 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys, parent 1,
                ``I_INF`` padded.
            conn_attrs1 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs, parent 1,
                NaN padded.
            node_keys2 (evoxc.SymbolicTensor): ``(P, N) int32`` node keys, parent 2,
                ``I_INF`` padded.
            node_attrs2 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attrs, parent 2,
                NaN padded.
            conn_keys2 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` conn keys, parent 2,
                ``I_INF`` padded.
            conn_attrs2 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` conn attrs, parent 2,
                NaN padded.

        Returns:
            GenomeTensors: ``(node_keys1, recombined_node_attrs, conn_keys1,
            recombined_conn_attrs)``.
        """
        return super().__call__(
            state,
            genome,
            key,
            node_keys1,
            node_attrs1,
            conn_keys1,
            conn_attrs1,
            node_keys2,
            node_attrs2,
            conn_keys2,
            conn_attrs2,
        )

    def find_homologous(
        self,
        keys1: evoxc.SymbolicTensor,
        keys2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Locate, for every gene of parent 1, its homologous row in parent 2.

        Exact match on the whole key-column tuple: one hash per gene (the gene's ``hash``
        over its ``int32`` key columns), one ``argsort`` of the two parents' tagged hashes,
        and a comparison of the full key rows of every ``(parent 1, parent 2)`` adjacency
        that shares a hash — so a hash collision cannot produce a false homologue.  That is
        ``O(M log M)`` in the gene count ``M`` — never an ``O(M^2)`` scan — and all of it is
        one batched kernel over the population axis.

        Args:
            keys1 (evoxc.SymbolicTensor): ``(P, M, K) int32`` key columns of parent 1,
                ``I_INF`` padded.
            keys2 (evoxc.SymbolicTensor): ``(P, M, K) int32`` key columns of parent 2,
                ``I_INF`` padded.

        Returns:
            evoxc.SymbolicTensor: ``(P, M) int32`` row indices into parent 2, with ``I_INF``
            for genes whose homologue is absent there (and for parent 1's padded rows).

        Notes:
            Delegates to the module-level
            :func:`tensorneat.genome.operations.alignment.find_homologous`, which owns the
            sorted-merge alignment; nothing family-specific is left to do here.
        """
        return find_homologous(keys1, keys2)

    def crossover_gene_attrs(
        self,
        state: State,
        gene: "BaseGene",
        key: evoxc.SymbolicTensor,
        attrs1: evoxc.SymbolicTensor,
        homologous_rows: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Recombine one gene family's attributes through the homologue table.

        Homologous rows are gathered from ``attrs2`` and crossed with
        ``gene.crossover(state, key, attrs1, gathered_attrs2)`` in a single batched call; a
        gene whose ``homologous_rows`` entry is ``I_INF`` keeps ``attrs1`` unchanged (no
        gather is allowed to read a bogus row), and padded rows stay NaN.

        Args:
            state (State): Immutable host configuration container.
            gene ("BaseGene"): Node or connection gene supplying the ``crossover`` hook.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this gene family.
            attrs1 (evoxc.SymbolicTensor): ``(P, M, A) float32`` parent 1 attributes, NaN
                padded.
            homologous_rows (evoxc.SymbolicTensor): ``(P, M) int32`` output of
                :meth:`find_homologous` — row in parent 2, or ``I_INF``.
            attrs2 (evoxc.SymbolicTensor): ``(P, M, A) float32`` parent 2 attributes, NaN
                padded.

        Returns:
            evoxc.SymbolicTensor: ``(P, M, A) float32`` recombined attributes; entries whose
            homologue is missing equal ``attrs1``, padding rows remain NaN.

        Notes:
            The row of parent 2 is turned into a flat index into the ``(P, M * A)`` view of
            ``attrs2`` (``row * A + arange(A)``), and ``I_INF`` rows are redirected to row
            ``0`` **before** that multiplication: ``I_INF * A`` would overflow ``int32``,
            whereas the gathered value of a redirected row is discarded by the final
            ``select``.  The mask still marks those positions absent, so the gather can
            never read outside the payload.  A family without attribute columns has nothing
            to recombine and comes back unchanged.
        """
        if gene.attr_width == 0:
            return attrs1
        pop_size, count, width = attrs1.shape
        rows = enp.reshape(homologous_rows, (pop_size, count, 1))
        present = evoxc.not_equal(rows, enp.full((), I_INF, dtype=evoxc.int32))
        safe_rows = evoxc.select(present, rows, enp.full((), 0, dtype=evoxc.int32))
        index = evoxc.add(
            evoxc.multiply(safe_rows, enp.full((), width, dtype=evoxc.int32)),
            enp.arange(0, width, dtype=evoxc.int32),
        )
        gathered = masked_flat_gather(
            enp.reshape(attrs2, (pop_size, count * width)),
            index,
            mask=enp.broadcast_to(present, (pop_size, count, width)),
            fill=float("nan"),
        )
        crossed = gene.crossover(state, key, attrs1, gathered)
        return evoxc.select(present, crossed, attrs1)


__all__ = ["DefaultCrossover"]
