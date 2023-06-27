"""
Calculate the distance between two genomes.
The calculation method is the same as the distance calculation in NEAT-python.
See https://github.com/CodeReclaimers/neat-python/blob/master/neat/genome.py
"""
from typing import Dict

from jax import jit, vmap, Array
from jax import numpy as jnp

from .utils import EMPTY_NODE, EMPTY_CON


@jit
def distance(nodes1: Array, cons1: Array, nodes2: Array, cons2: Array, jit_config: Dict) -> Array:
    """
    Calculate the distance between two genomes.
    args:
        nodes1: Array(N, 5)
        cons1: Array(C, 4)
        nodes2: Array(N, 5)
        cons2: Array(C, 4)
    returns:
        distance: Array(, )
    """
    nd = node_distance(nodes1, nodes2, jit_config)  # node distance
    cd = connection_distance(cons1, cons2, jit_config)  # connection distance
    return nd + cd


@jit
def node_distance(nodes1: Array, nodes2: Array, jit_config: Dict):
    """
    Calculate the distance between nodes of two genomes.
    """
    # statistics nodes count of two genomes
    node_cnt1 = jnp.sum(~jnp.isnan(nodes1[:, 0]))
    node_cnt2 = jnp.sum(~jnp.isnan(nodes2[:, 0]))
    max_cnt = jnp.maximum(node_cnt1, node_cnt2)

    # align homologous nodes
    # this process is similar to np.intersect1d.
    nodes = jnp.concatenate((nodes1, nodes2), axis=0)
    keys = nodes[:, 0]
    sorted_indices = jnp.argsort(keys, axis=0)
    nodes = nodes[sorted_indices]
    nodes = jnp.concatenate([nodes, EMPTY_NODE], axis=0)  # add a nan row to the end
    fr, sr = nodes[:-1], nodes[1:]  # first row, second row

    # flag location of homologous nodes
    intersect_mask = (fr[:, 0] == sr[:, 0]) & ~jnp.isnan(nodes[:-1, 0])

    # calculate the count of non_homologous of two genomes
    non_homologous_cnt = node_cnt1 + node_cnt2 - 2 * jnp.sum(intersect_mask)

    # calculate the distance of homologous nodes
    hnd = vmap(homologous_node_distance)(fr, sr)
    hnd = jnp.where(jnp.isnan(hnd), 0, hnd)
    homologous_distance = jnp.sum(hnd * intersect_mask)

    val = non_homologous_cnt * jit_config['compatibility_disjoint'] + homologous_distance * jit_config[
        'compatibility_weight']

    return jnp.where(max_cnt == 0, 0, val / max_cnt)  # avoid zero division


@jit
def connection_distance(cons1: Array, cons2: Array, jit_config: Dict):
    """
    Calculate the distance between connections of two genomes.
    Similar process as node_distance.
    """
    con_cnt1 = jnp.sum(~jnp.isnan(cons1[:, 0]))
    con_cnt2 = jnp.sum(~jnp.isnan(cons2[:, 0]))
    max_cnt = jnp.maximum(con_cnt1, con_cnt2)

    cons = jnp.concatenate((cons1, cons2), axis=0)
    keys = cons[:, :2]
    sorted_indices = jnp.lexsort(keys.T[::-1])
    cons = cons[sorted_indices]
    cons = jnp.concatenate([cons, EMPTY_CON], axis=0)  # add a nan row to the end
    fr, sr = cons[:-1], cons[1:]  # first row, second row

    # both genome has such connection
    intersect_mask = jnp.all(fr[:, :2] == sr[:, :2], axis=1) & ~jnp.isnan(fr[:, 0])

    non_homologous_cnt = con_cnt1 + con_cnt2 - 2 * jnp.sum(intersect_mask)
    hcd = vmap(homologous_connection_distance)(fr, sr)
    hcd = jnp.where(jnp.isnan(hcd), 0, hcd)
    homologous_distance = jnp.sum(hcd * intersect_mask)

    val = non_homologous_cnt * jit_config['compatibility_disjoint'] + homologous_distance * jit_config[
        'compatibility_weight']

    return jnp.where(max_cnt == 0, 0, val / max_cnt)


@jit
def homologous_node_distance(n1: Array, n2: Array):
    """
    Calculate the distance between two homologous nodes.
    """
    d = 0
    d += jnp.abs(n1[1] - n2[1])  # bias
    d += jnp.abs(n1[2] - n2[2])  # response
    d += n1[3] != n2[3]  # activation
    d += n1[4] != n2[4]  # aggregation
    return d


@jit
def homologous_connection_distance(c1: Array, c2: Array):
    """
    Calculate the distance between two homologous connections.
    """
    d = 0
    d += jnp.abs(c1[2] - c2[2])  # weight
    d += c1[3] != c2[3]  # enable
    return d
