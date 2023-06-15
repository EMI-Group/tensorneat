from jax import jit, vmap, Array
from jax import numpy as jnp

from .utils import EMPTY_NODE, EMPTY_CON


@jit
def distance(nodes1: Array, cons1: Array, nodes2: Array, cons2: Array, disjoint_coe: float = 1.,
             compatibility_coe: float = 0.5) -> Array:
    """
    Calculate the distance between two genomes.
    nodes are a 2-d array with shape (N, 5), its columns are [key, bias, response, act, agg]
    connections are a 3-d array with shape (2, N, N), axis 0 means [weights, enable]
    """

    nd = node_distance(nodes1, nodes2, disjoint_coe, compatibility_coe)  # node distance

    cd = connection_distance(cons1, cons2, disjoint_coe, compatibility_coe)  # connection distance
    return nd + cd


@jit
def node_distance(nodes1, nodes2, disjoint_coe=1., compatibility_coe=0.5):
    node_cnt1 = jnp.sum(~jnp.isnan(nodes1[:, 0]))
    node_cnt2 = jnp.sum(~jnp.isnan(nodes2[:, 0]))
    max_cnt = jnp.maximum(node_cnt1, node_cnt2)

    nodes = jnp.concatenate((nodes1, nodes2), axis=0)
    keys = nodes[:, 0]
    sorted_indices = jnp.argsort(keys, axis=0)
    nodes = nodes[sorted_indices]
    nodes = jnp.concatenate([nodes, EMPTY_NODE], axis=0)  # add a nan row to the end
    fr, sr = nodes[:-1], nodes[1:]  # first row, second row

    intersect_mask = (fr[:, 0] == sr[:, 0]) & ~jnp.isnan(nodes[:-1, 0])

    non_homologous_cnt = node_cnt1 + node_cnt2 - 2 * jnp.sum(intersect_mask)
    nd = batch_homologous_node_distance(fr, sr)
    nd = jnp.where(jnp.isnan(nd), 0, nd)
    homologous_distance = jnp.sum(nd * intersect_mask)

    val = non_homologous_cnt * disjoint_coe + homologous_distance * compatibility_coe
    return jnp.where(max_cnt == 0, 0, val / max_cnt)


@jit
def connection_distance(cons1, cons2, disjoint_coe=1., compatibility_coe=0.5):
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
    cd = batch_homologous_connection_distance(fr, sr)
    cd = jnp.where(jnp.isnan(cd), 0, cd)
    homologous_distance = jnp.sum(cd * intersect_mask)

    val = non_homologous_cnt * disjoint_coe + homologous_distance * compatibility_coe

    return jnp.where(max_cnt == 0, 0, val / max_cnt)


@vmap
def batch_homologous_node_distance(b_n1, b_n2):
    return homologous_node_distance(b_n1, b_n2)


@vmap
def batch_homologous_connection_distance(b_c1, b_c2):
    return homologous_connection_distance(b_c1, b_c2)


@jit
def homologous_node_distance(n1, n2):
    d = 0
    d += jnp.abs(n1[1] - n2[1])  # bias
    d += jnp.abs(n1[2] - n2[2])  # response
    d += n1[3] != n2[3]  # activation
    d += n1[4] != n2[4]
    return d


@jit
def homologous_connection_distance(c1, c2):
    d = 0
    d += jnp.abs(c1[2] - c2[2])  # weight
    d += c1[3] != c2[3]  # enable
    return d
