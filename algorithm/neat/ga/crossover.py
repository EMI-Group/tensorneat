import jax
from jax import Array, numpy as jnp

from core import Genome


def crossover(randkey, genome1: Genome, genome2: Genome):
    """
    use genome1 and genome2 to generate a new genome
    notice that genome1 should have higher fitness than genome2 (genome1 is winner!)
    """
    randkey_1, randkey_2, key = jax.random.split(randkey, 3)

    # crossover nodes
    keys1, keys2 = genome1.nodes[:, 0], genome2.nodes[:, 0]
    # make homologous genes align in nodes2 align with nodes1
    nodes2 = align_array(keys1, keys2, genome2.nodes, False)
    nodes1 = genome1.nodes
    # For not homologous genes, use the value of nodes1(winner)
    # For homologous genes, use the crossover result between nodes1 and nodes2
    new_nodes = jnp.where(jnp.isnan(nodes1) | jnp.isnan(nodes2), nodes1, crossover_gene(randkey_1, nodes1, nodes2))

    # crossover connections
    con_keys1, con_keys2 = genome1.conns[:, :2], genome2.conns[:, :2]
    conns2 = align_array(con_keys1, con_keys2, genome2.conns, True)
    conns1 = genome1.conns

    new_cons = jnp.where(jnp.isnan(conns1) | jnp.isnan(conns2), conns1, crossover_gene(randkey_2, conns1, conns2))

    return genome1.update(new_nodes, new_cons)


def align_array(seq1: Array, seq2: Array, ar2: Array, is_conn: bool) -> Array:
    """
    After I review this code, I found that it is the most difficult part of the code. Please never change it!
    make ar2 align with ar1.
    :param seq1:
    :param seq2:
    :param ar2:
    :param is_conn:
    :return:
    align means to intersect part of ar2 will be at the same position as ar1,
    non-intersect part of ar2 will be set to Nan
    """
    seq1, seq2 = seq1[:, jnp.newaxis], seq2[jnp.newaxis, :]
    mask = (seq1 == seq2) & (~jnp.isnan(seq1))

    if is_conn:
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
