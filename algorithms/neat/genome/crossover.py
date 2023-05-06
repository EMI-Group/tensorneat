from functools import partial
from typing import Tuple

import jax
from jax import jit, vmap, Array
from jax import numpy as jnp

# from .utils import flatten_connections, unflatten_connections
from algorithms.neat.genome.utils import flatten_connections, unflatten_connections


@vmap
def batch_crossover(randkeys: Array, batch_nodes1: Array, batch_connections1: Array, batch_nodes2: Array,
                    batch_connections2: Array) -> Tuple[Array, Array]:
    """
    crossover a batch of genomes
    :param randkeys: batches of random keys
    :param batch_nodes1:
    :param batch_connections1:
    :param batch_nodes2:
    :param batch_connections2:
    :return:
    """
    return crossover(randkeys, batch_nodes1, batch_connections1, batch_nodes2, batch_connections2)


@jit
def crossover(randkey: Array, nodes1: Array, connections1: Array, nodes2: Array, connections2: Array) \
        -> Tuple[Array, Array]:
    """
    use genome1 and genome2 to generate a new genome
    notice that genome1 should have higher fitness than genome2 (genome1 is winner!)
    :param randkey:
    :param nodes1:
    :param connections1:
    :param nodes2:
    :param connections2:
    :return:
    """
    randkey_1, randkey_2 = jax.random.split(randkey)

    # crossover nodes
    keys1, keys2 = nodes1[:, 0], nodes2[:, 0]
    nodes2 = align_array(keys1, keys2, nodes2, 'node')
    new_nodes = jnp.where(jnp.isnan(nodes1) | jnp.isnan(nodes2), nodes1, crossover_gene(randkey_1, nodes1, nodes2))

    # crossover connections
    cons1 = flatten_connections(keys1, connections1)
    cons2 = flatten_connections(keys2, connections2)
    con_keys1, con_keys2 = cons1[:, :2], cons2[:, :2]
    cons2 = align_array(con_keys1, con_keys2, cons2, 'connection')
    new_cons = jnp.where(jnp.isnan(cons1) | jnp.isnan(cons2), cons1, crossover_gene(randkey_2, cons1, cons2))
    new_cons = unflatten_connections(len(keys1), new_cons)

    return new_nodes, new_cons


@partial(jit, static_argnames=['gene_type'])
def align_array(seq1: Array, seq2: Array, ar2: Array, gene_type: str) -> Array:
    """
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


@jit
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


if __name__ == '__main__':
    import numpy as np

    randkey = jax.random.PRNGKey(40)
    nodes1 = np.array([
        [4, 1, 1, 1, 1],
        [6, 2, 2, 2, 2],
        [1, 3, 3, 3, 3],
        [5, 4, 4, 4, 4],
        [np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    nodes2 = np.array([
        [4, 1.5, 1.5, 1.5, 1.5],
        [7, 3.5, 3.5, 3.5, 3.5],
        [5, 4.5, 4.5, 4.5, 4.5],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan],
    ])
    weights1 = np.array([
        [
            [1, 2, 3, 4., np.nan],
            [5, np.nan, 7, 8, np.nan],
            [9, 10, 11, 12, np.nan],
            [13, 14, 15, 16, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan]
        ],
        [
            [0, 1, 0, 1, np.nan],
            [0, np.nan, 0, 1, np.nan],
            [0, 1, 0, 1, np.nan],
            [0, 1, 0, 1, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan]
        ]
    ])
    weights2 = np.array([
        [
            [1.5, 2.5, 3.5, np.nan, np.nan],
            [3.5, 4.5, 5.5, np.nan, np.nan],
            [6.5, 7.5, 8.5, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan]
        ],
        [
            [1, 0, 1, np.nan, np.nan],
            [1, 0, 1, np.nan, np.nan],
            [1, 0, 1, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan]
        ]
    ])

    res = crossover(randkey, nodes1, weights1, nodes2, weights2)
    print(*res, sep='\n')
