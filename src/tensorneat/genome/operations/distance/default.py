from jax import vmap, numpy as jnp

from .base import BaseDistance
from ...gene import BaseGene
from ...utils import extract_gene_attrs


class DefaultDistance(BaseDistance):
    def __init__(
        self,
        compatibility_disjoint: float = 1.0,
        compatibility_weight: float = 0.4,
    ):
        self.compatibility_disjoint = compatibility_disjoint
        self.compatibility_weight = compatibility_weight

    def __call__(self, state, genome, nodes1, conns1, nodes2, conns2):
        """
        The distance between two genomes
        """
        node_distance = self.gene_distance(state, genome.node_gene, nodes1, nodes2)
        conn_distance = self.gene_distance(state, genome.conn_gene, conns1, conns2)
        return node_distance + conn_distance


    def gene_distance(self, state, gene: BaseGene, genes1, genes2):
        """
        The distance between to genes
        genes1: 2-D jax array with shape
        genes2: 2-D jax array with shape
        gene1.shape == gene2.shape
        """
        cnt1 = jnp.sum(~jnp.isnan(genes1[:, 0]))
        cnt2 = jnp.sum(~jnp.isnan(genes2[:, 0]))
        max_cnt = jnp.maximum(cnt1, cnt2)

        # align homologous nodes
        # this process is similar to np.intersect1d in higher dimension
        total_genes = jnp.concatenate((genes1, genes2), axis=0)
        identifiers = total_genes[:, : len(gene.fixed_attrs)]
        sorted_identifiers = jnp.lexsort(identifiers.T[::-1])
        total_genes = total_genes[sorted_identifiers]
        total_genes = jnp.concatenate(
            [total_genes, jnp.full((1, total_genes.shape[1]), jnp.nan)], axis=0
        )  # add a nan row to the end
        fr, sr = total_genes[:-1], total_genes[1:]  # first row, second row

        # intersect part of two genes
        intersect_mask = jnp.all(
            fr[:, : len(gene.fixed_attrs)] == sr[:, : len(gene.fixed_attrs)], axis=1
        ) & ~jnp.isnan(fr[:, 0])

        non_homologous_cnt = cnt1 + cnt2 - 2 * jnp.sum(intersect_mask)

        fr_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(gene, fr)
        sr_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(gene, sr)

        # homologous gene distance
        hgd = vmap(gene.distance, in_axes=(None, 0, 0))(state, fr_attrs, sr_attrs)
        hgd = jnp.where(jnp.isnan(hgd), 0, hgd)
        homologous_distance = jnp.sum(hgd * intersect_mask)

        val = (
            non_homologous_cnt * self.compatibility_disjoint
            + homologous_distance * self.compatibility_weight
        )

        val = jnp.where(max_cnt == 0, 0, val / max_cnt)  # normalize

        return val
