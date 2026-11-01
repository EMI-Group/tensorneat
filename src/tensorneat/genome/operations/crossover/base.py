"""Crossover contract: key-aligned recombination of two batches of parent genomes.

Crossover is a trace-time kernel: both parents are ``(P, ...)`` payload batches, parent 1
being the fitter one, so a gene of the offspring is taken from parent 1 unless it has a
homologue in parent 2.
Alignment is by exact key-column match, resolved with one sort per array — never an
``O(M^2)`` scan — and no operator loops over individuals or allocates keys.

This module owns only the generic driver: validate that the two parents are shaped alike,
split the single key into one key per gene family, and hand each family to
:meth:`BaseCrossover.find_homologous` / :meth:`BaseCrossover.crossover_gene_attrs` — the
alignment primitive lives in :mod:`..alignment` (together with the rank normaliser
:func:`~tensorneat.genome.operations.alignment.key_rows`) and the gene hook in
:class:`~tensorneat.genome.gene.base.BaseGene`.
"""

from typing import TYPE_CHECKING

import evoxc

from ....common import ConfigError, State, StatefulBaseClass
from ...utils import GenomeTensors
from ..alignment import key_rows

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps genome -> operations acyclic
    from ...base import BaseGenome


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
    ) -> GenomeTensors:
        """Cross a batch of parent pairs into a batch of offspring.

        Parent 1 is the **fitter** parent: its keys define the offspring's structure and its
        attributes survive wherever parent 2 has no homologue.  The single rank-0 key is
        split internally into one key per gene family (nodes, connections).

        Args:
            state (State): Immutable host configuration container.
            genome ("BaseGenome"): Genome owning the gene objects, their ``key_width``s and
                the ``crossover`` hooks.
            key (evoxc.SymbolicTensor): Rank-0 ``evoxc`` key for this call.
            node_keys1 (evoxc.SymbolicTensor): ``(P, N) int32`` node keys of parent 1
                (fitter parent), ``I_INF`` padded.
            node_attrs1 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes of
                parent 1, NaN padded.
            conn_keys1 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys of
                parent 1, ``I_INF`` padded.
            conn_attrs1 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes
                of parent 1, NaN padded.
            node_keys2 (evoxc.SymbolicTensor): ``(P, N) int32`` node keys of parent 2,
                ``I_INF`` padded.
            node_attrs2 (evoxc.SymbolicTensor): ``(P, N, NA) float32`` node attributes of
                parent 2, NaN padded.
            conn_keys2 (evoxc.SymbolicTensor): ``(P, E, CK) int32`` connection keys of
                parent 2, ``I_INF`` padded.
            conn_attrs2 (evoxc.SymbolicTensor): ``(P, E, CA) float32`` connection attributes
                of parent 2, NaN padded.

        Returns:
            GenomeTensors: ``(node_keys, node_attrs, conn_keys, conn_attrs)`` — parent 1's keys
            with the recombined attributes.

        Raises:
            ConfigError: If the two parents' payload shapes differ (crossover does not
                resize genomes; ``max_nodes``/``max_conns`` are build-time constants).  It
                is a :class:`~tensorneat.common.errors.TensorNEATError` subclass.

        Notes:
            Every shape here is static at trace time, so the equality test runs on the host
            side of the trace and no tensor op is spent on it.  The offspring's key arrays
            are parent 1's tensors, returned untouched: a crossover never invents an
            innovation number.  The two family-specific steps are supplied by the concrete
            subclass — :meth:`find_homologous` (alignment) and :meth:`crossover_gene_attrs`
            (attribute recombination) — so this driver stays family-agnostic.  A node key
            table is two-dimensional (one key column); a ``(P, N, 1)`` view that spells the
            key axis out is accepted as well and returned verbatim, which is what the shared
            :func:`~tensorneat.genome.operations.alignment.key_rows` normaliser allows.
        """
        payload1 = (node_keys1, node_attrs1, conn_keys1, conn_attrs1)
        payload2 = (node_keys2, node_attrs2, conn_keys2, conn_attrs2)
        for name, first, second in zip(GenomeTensors._fields, payload1, payload2):
            if tuple(first.shape) != tuple(second.shape):
                raise ConfigError(
                    f"crossover needs identically shaped parents: {name} is "
                    f"{tuple(first.shape)} in parent 1 and {tuple(second.shape)} "
                    "in parent 2"
                )

        node_key, conn_key = evoxc.random.split_n(key, 2)
        node_attrs = self.crossover_gene_attrs(
            state,
            genome.node_gene,
            node_key,
            node_attrs1,
            self.find_homologous(key_rows(node_keys1), key_rows(node_keys2)),
            node_attrs2,
        )
        conn_attrs = self.crossover_gene_attrs(
            state,
            genome.conn_gene,
            conn_key,
            conn_attrs1,
            self.find_homologous(key_rows(conn_keys1), key_rows(conn_keys2)),
            conn_attrs2,
        )
        return GenomeTensors(node_keys1, node_attrs, conn_keys1, conn_attrs)


__all__ = ["BaseCrossover"]
