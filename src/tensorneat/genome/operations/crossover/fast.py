import jax
from jax import vmap, numpy as jnp

from .base import BaseCrossover
from ...utils import (
    extract_node_attrs,
    extract_conn_attrs,
    set_node_attrs,
    set_conn_attrs,
)
from tensorneat.common import fetch_first, I_INF
from tensorneat.genome.gene import BaseGene


class FastCrossover(BaseCrossover):
    """
    Crossover function with faster execution speed and less memory usage.
    The behavior of this crossover function is the same as the default crossover function.
    """

    def __call__(self, state, genome, randkey, nodes1, conns1, nodes2, conns2):
        """
        use genome1 and genome2 to generate a new genome
        notice that genome1 should have higher fitness than genome2 (genome1 is winner!)
        """
        randkey1, randkey2 = jax.random.split(randkey, 2)
        node_randkeys = jax.random.split(randkey1, genome.max_nodes)
        conn_randkeys = jax.random.split(randkey2, genome.max_conns)
        batch_create_new_gene = jax.vmap(
            create_new_gene, in_axes=(None, 0, None, 0, 0, None, None)
        )

        # crossover nodes
        node_keys1, node_keys2 = nodes1[:, 0:1], nodes2[:, 0:1]
        node_attrs1 = vmap(extract_node_attrs)(nodes1)
        node_attrs2 = vmap(extract_node_attrs)(nodes2)

        new_node_attrs = batch_create_new_gene(
            state,
            node_randkeys,
            genome.node_gene,
            node_keys1,
            node_attrs1,
            node_keys2,
            node_attrs2,
        )
        new_nodes = vmap(set_node_attrs)(nodes1, new_node_attrs)

        # crossover connections
        conn_keys1, conn_keys2 = conns1[:, 0:2], conns2[:, 0:2]
        conn_attrs1 = vmap(extract_conn_attrs)(conns1)
        conn_attrs2 = vmap(extract_conn_attrs)(conns2)

        new_conn_attrs = batch_create_new_gene(
            state,
            conn_randkeys,
            genome.conn_gene,
            conn_keys1,
            conn_attrs1,
            conn_keys2,
            conn_attrs2,
        )
        new_conns = vmap(set_conn_attrs)(conns1, new_conn_attrs)

        return new_nodes, new_conns


def create_new_gene(
    state,
    randkey,
    gene: BaseGene,
    gene_key,
    gene_attrs,
    genes_keys,
    genes_attrs,
):
    # find homologous genes
    homologous_idx = fetch_first(jnp.all(gene_key == genes_keys, axis=1))

    def none():
        return gene_attrs

    def crossover():  # when homologous gene is found
        return gene.crossover(state, randkey, gene_attrs, genes_attrs[homologous_idx])

    new_attrs = jax.lax.cond(
        homologous_idx == I_INF,  # homologous gene is not found or current gene is nan
        none,
        crossover,
    )

    return new_attrs
