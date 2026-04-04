import jax
from jax import vmap, numpy as jnp

from .base import BaseCrossover
from ...utils import extract_gene_attrs, set_gene_attrs

from tensorneat.common import I_INF, hash_array
from tensorneat.genome.gene import BaseGene

_COLLISION_WINDOW = 4


def _find_homologous_indices(keys1, keys2):
    """
    For each gene in keys1, find the index of its homologous gene in keys2.
    Uses hash + searchsorted: O(M log M) time, O(M) memory.
    Replaces the old nested-vmap linear scan which was O(M^2) and caused OOM
    when pop_size and max_conns were both large.

    keys1: (M, K) fixed attributes of genome1's genes
    keys2: (M, K) fixed attributes of genome2's genes
    Returns: (M,) int32 array, I_INF where no homologous gene is found

    See: https://github.com/EMI-Group/tensorneat/issues/40
    """
    M = keys1.shape[0]

    valid1 = ~jnp.isnan(keys1[:, 0])
    valid2 = ~jnp.isnan(keys2[:, 0])

    safe1 = jnp.where(jnp.isnan(keys1), 0, keys1)
    safe2 = jnp.where(jnp.isnan(keys2), 0, keys2)

    hash1 = vmap(hash_array)(safe1)
    hash2 = vmap(hash_array)(safe2)

    hash1 = jnp.where(valid1, hash1, jnp.uint32(0xFFFFFFFE))
    hash2 = jnp.where(valid2, hash2, jnp.uint32(0xFFFFFFFF))

    sort_idx = jnp.argsort(hash2)
    sorted_h2 = hash2[sort_idx]

    pos = jnp.searchsorted(sorted_h2, hash1, side="left")

    result = jnp.full(M, I_INF, dtype=jnp.int32)
    for offset in range(_COLLISION_WINDOW):
        p = jnp.clip(pos + offset, 0, M - 1)
        orig = sort_idx[p]
        same_hash = sorted_h2[p] == hash1
        exact = jnp.all(keys1 == keys2[orig], axis=1)
        is_new = result == I_INF
        result = jnp.where(same_hash & exact & valid1 & is_new, orig, result)

    return result


def _crossover_with_index(
    state, randkey, gene: BaseGene, attrs1, homologous_idx, all_attrs2
):
    """Crossover a single gene using a pre-computed homologous index."""

    def use_winner():
        return attrs1

    def do_crossover():
        return gene.crossover(state, randkey, attrs1, all_attrs2[homologous_idx])

    return jax.lax.cond(homologous_idx == I_INF, use_winner, do_crossover)


class DefaultCrossover(BaseCrossover):
    def __call__(self, state, genome, randkey, nodes1, conns1, nodes2, conns2):
        """
        use genome1 and genome2 to generate a new genome
        notice that genome1 should have higher fitness than genome2 (genome1 is winner!)
        """
        randkey1, randkey2 = jax.random.split(randkey, 2)
        node_randkeys = jax.random.split(randkey1, genome.max_nodes)
        conn_randkeys = jax.random.split(randkey2, genome.max_conns)

        # crossover nodes
        node_keys1 = nodes1[:, 0 : len(genome.node_gene.fixed_attrs)]
        node_keys2 = nodes2[:, 0 : len(genome.node_gene.fixed_attrs)]
        node_attrs1 = vmap(extract_gene_attrs, in_axes=(None, 0))(
            genome.node_gene, nodes1
        )
        node_attrs2 = vmap(extract_gene_attrs, in_axes=(None, 0))(
            genome.node_gene, nodes2
        )

        node_hom = _find_homologous_indices(node_keys1, node_keys2)
        new_node_attrs = vmap(
            _crossover_with_index, in_axes=(None, 0, None, 0, 0, None)
        )(state, node_randkeys, genome.node_gene, node_attrs1, node_hom, node_attrs2)

        new_nodes = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.node_gene, nodes1, new_node_attrs
        )

        # crossover connections
        # all fixed_attrs together will use to identify a connection
        # if using historical marker, use it
        # related to issue: https://github.com/EMI-Group/tensorneat/issues/11
        conn_keys1 = conns1[:, 0 : len(genome.conn_gene.fixed_attrs)]
        conn_keys2 = conns2[:, 0 : len(genome.conn_gene.fixed_attrs)]
        conn_attrs1 = vmap(extract_gene_attrs, in_axes=(None, 0))(
            genome.conn_gene, conns1
        )
        conn_attrs2 = vmap(extract_gene_attrs, in_axes=(None, 0))(
            genome.conn_gene, conns2
        )

        conn_hom = _find_homologous_indices(conn_keys1, conn_keys2)
        new_conn_attrs = vmap(
            _crossover_with_index, in_axes=(None, 0, None, 0, 0, None)
        )(state, conn_randkeys, genome.conn_gene, conn_attrs1, conn_hom, conn_attrs2)

        new_conns = vmap(set_gene_attrs, in_axes=(None, 0, 0))(
            genome.conn_gene, conns1, new_conn_attrs
        )

        return new_nodes, new_conns
