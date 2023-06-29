"""
Crossover two genomes to generate a new genome.
The calculation method is the same as the crossover operation in NEAT-python.
See https://neat-python.readthedocs.io/en/latest/_modules/genome.html#DefaultGenome.configure_crossover
"""
from typing import Tuple

import jax
from jax import jit, Array, numpy as jnp


@jit
def crossover(randkey: Array, nodes1: Array, cons1: Array, nodes2: Array, cons2: Array) -> Tuple[Array, Array]:
    """
    use genome1 and genome2 to generate a new genome
    notice that genome1 should have higher fitness than genome2 (genome1 is winner!)
    :param randkey:
    :param nodes1:
    :param cons1:
    :param nodes2:
    :param cons2:
    :return:
    """
    randkey_1, randkey_2 = jax.random.split(randkey)

    # crossover nodes
    keys1, keys2 = nodes1[:, 0], nodes2[:, 0]
    # make homologous genes align in nodes2 align with nodes1
    nodes2 = align_array(keys1, keys2, nodes2, 'node')

    # For not homologous genes, use the value of nodes1(winner)
    # For homologous genes, use the crossover result between nodes1 and nodes2
    new_nodes = jnp.where(jnp.isnan(nodes1) | jnp.isnan(nodes2), nodes1, crossover_gene(randkey_1, nodes1, nodes2))

    # crossover connections
    con_keys1, con_keys2 = cons1[:, :2], cons2[:, :2]
    cons2 = align_array(con_keys1, con_keys2, cons2, 'connection')
    new_cons = jnp.where(jnp.isnan(cons1) | jnp.isnan(cons2), cons1, crossover_gene(randkey_2, cons1, cons2))

    return new_nodes, new_cons


def align_array(seq1: Array, seq2: Array, ar2: Array, gene_type: str) -> Array:
    """
    After I review this code, I found that it is the most difficult part of the code. Please never change it!
    make ar2 align with ar1.
    :param seq1:
    :param seq2:
    :param ar2:
    :param gene_type:
    :return:
    align means to intersect part of ar2 will be at the same position as ar1,
    non-intersect part of ar2 will be set to Nan
    """
    seq1, seq2 = seq1[:, jnp.newaxis], seq2[jnp.newaxis, :]
    mask = (seq1 == seq2) & (~jnp.isnan(seq1))

    if gene_type == 'connection':
        mask = jnp.all(mask, axis=2)

    intersect_mask = mask.any(axis=1)
    idx = jnp.arange(0, len(seq1))
    idx_fixed = jnp.dot(mask, idx)

    refactor_ar2 = jnp.where(intersect_mask[:, jnp.newaxis], ar2[idx_fixed], jnp.nan)

    return refactor_ar2


def crossover_gene(rand_key: Array, g1: Array, g2: Array) -> Array:
    """
    crossover two genes
    :param rand_key:
    :param g1:
    :param g2:
    :return:
    only gene with the same key will be crossover, thus don't need to consider change key
    """
    r = jax.random.uniform(rand_key, shape=g1.shape)
    return jnp.where(r > 0.5, g1, g2)
