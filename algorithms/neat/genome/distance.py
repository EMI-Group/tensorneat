from functools import partial

from jax import jit, vmap, Array
from jax import numpy as jnp

from algorithms.neat.genome.utils import flatten_connections, set_operation_analysis


@jit
def distance(nodes1: Array, connections1: Array, nodes2: Array, connections2: Array) -> Array:
    """
    Calculate the distance between two genomes.
    nodes are a 2-d array with shape (N, 5), its columns are [key, bias, response, act, agg]
    connections are a 3-d array with shape (2, N, N), axis 0 means [weights, enable]
    """

    node_distance = gene_distance(nodes1, nodes2, 'node')

    # refactor connections
    keys1, keys2 = nodes1[:, 0], nodes2[:, 0]
    cons1 = flatten_connections(keys1, connections1)
    cons2 = flatten_connections(keys2, connections2)

    connection_distance = gene_distance(cons1, cons2, 'connection')
    return node_distance + connection_distance


@partial(jit, static_argnames=["gene_type"])
def gene_distance(ar1, ar2, gene_type, compatibility_coe=0.5, disjoint_coe=1.):
    if gene_type == 'node':
        keys1, keys2 = ar1[:, :1], ar2[:, :1]
    else:  # connection
        keys1, keys2 = ar1[:, :2], ar2[:, :2]

    n_sorted_indices, n_intersect_mask, n_union_mask = set_operation_analysis(keys1, keys2)
    nodes = jnp.concatenate((ar1, ar2), axis=0)
    sorted_nodes = nodes[n_sorted_indices]
    fr_sorted_nodes, sr_sorted_nodes = sorted_nodes[:-1], sorted_nodes[1:]

    non_homologous_cnt = jnp.sum(n_union_mask) - jnp.sum(n_intersect_mask)
    if gene_type == 'node':
        node_distance = homologous_node_distance(fr_sorted_nodes, sr_sorted_nodes)
    else:  # connection
        node_distance = homologous_connection_distance(fr_sorted_nodes, sr_sorted_nodes)

    node_distance = jnp.where(jnp.isnan(node_distance), 0, node_distance)
    homologous_distance = jnp.sum(node_distance * n_intersect_mask[:-1])

    gene_cnt1 = jnp.sum(jnp.all(~jnp.isnan(ar1), axis=1))
    gene_cnt2 = jnp.sum(jnp.all(~jnp.isnan(ar2), axis=1))

    val = non_homologous_cnt * disjoint_coe + homologous_distance * compatibility_coe
    return val / jnp.where(gene_cnt1 > gene_cnt2, gene_cnt1, gene_cnt2)


@partial(vmap, in_axes=(0, 0))
def homologous_node_distance(n1, n2):
    d = 0
    d += jnp.abs(n1[1] - n2[1])  # bias
    d += jnp.abs(n1[2] - n2[2])  # response
    d += n1[3] != n2[3]  # activation
    d += n1[4] != n2[4]
    return d


@partial(vmap, in_axes=(0, 0))
def homologous_connection_distance(c1, c2):
    d = 0
    d += jnp.abs(c1[2] - c2[2])  # weight
    d += c1[3] != c2[3]  # enable
    return d
