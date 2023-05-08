from jax import jit, vmap, Array
from jax import numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .utils import flatten_connections, EMPTY_NODE, EMPTY_CON


def create_distance_function(N, config, type: str, debug: bool = False):
    """
    :param N:
    :param config:
    :param type: {'o2o', 'o2m'}, for one-to-one or one-to-many distance calculation
    :param debug:
    :return:
    """
    disjoint_coe = config.neat.genome.compatibility_disjoint_coefficient
    compatibility_coe = config.neat.genome.compatibility_weight_coefficient

    def distance_with_args(nodes1, connections1, nodes2, connections2):
        return distance(nodes1, connections1, nodes2, connections2, disjoint_coe, compatibility_coe)

    if type == 'o2o':
        nodes1_lower = jnp.zeros((N, 5))
        connections1_lower = jnp.zeros((2, N, N))
        nodes2_lower = jnp.zeros((N, 5))
        connections2_lower = jnp.zeros((2, N, N))

        res_func = jit(distance_with_args).lower(nodes1_lower, connections1_lower,
                                                 nodes2_lower, connections2_lower).compile()
        if debug:
            return lambda *args: res_func(*args)  # for debug
        else:
            return res_func

    elif type == 'o2m':
        vmap_func = vmap(distance_with_args, in_axes=(None, None, 0, 0))
        pop_size = config.neat.population.pop_size
        nodes1_lower = jnp.zeros((N, 5))
        connections1_lower = jnp.zeros((2, N, N))
        nodes2_lower = jnp.zeros((pop_size, N, 5))
        connections2_lower = jnp.zeros((pop_size, 2, N, N))
        res_func = jit(vmap_func).lower(nodes1_lower, connections1_lower, nodes2_lower, connections2_lower).compile()
        if debug:
            return lambda *args: res_func(*args)  # for debug
        else:
            return res_func

    else:
        raise ValueError(f'unknown distance type: {type}, should be one of ["o2o", "o2m"]')


def distance_numpy(nodes1: NDArray, connection1: NDArray, nodes2: NDArray,
                   connection2: NDArray, disjoint_coe: float = 1., compatibility_coe: float = 0.5):
    """
    use in o2o distance.
    o2o can't use vmap, numpy should be faster than jax function
    :param nodes1:
    :param connection1:
    :param nodes2:
    :param connection2:
    :param disjoint_coe:
    :param compatibility_coe:
    :return:
    """

    def analysis(nodes, connections):
        nodes_dict = {}
        idx2key = {}
        for i, node in enumerate(nodes):
            if np.isnan(node[0]):
                continue
            key = int(node[0])
            nodes_dict[key] = (node[1], node[2], node[3], node[4])
            idx2key[i] = key

        connections_dict = {}
        for i in range(connections.shape[1]):
            for j in range(connections.shape[2]):
                if np.isnan(connections[0, i, j]) and np.isnan(connections[1, i, j]):
                    continue
                key = (idx2key[i], idx2key[j])

                weight = connections[0, i, j] if not np.isnan(connections[0, i, j]) else None
                enabled = (connections[1, i, j] == 1) if not np.isnan(connections[1, i, j]) else None
                connections_dict[key] = (weight, enabled)

        return nodes_dict, connections_dict

    nodes1, connections1 = analysis(nodes1, connection1)
    nodes2, connections2 = analysis(nodes2, connection2)

    nd = 0.0
    if nodes1 or nodes2:  # otherwise, both are empty
        disjoint_nodes = 0
        for k2 in nodes2:
            if k2 not in nodes1:
                disjoint_nodes += 1

        for k1, n1 in nodes1.items():
            n2 = nodes2.get(k1)
            if n2 is None:
                disjoint_nodes += 1
            else:
                if np.isnan(n1[0]):  # n1[1] is nan means input nodes
                    continue
                d = abs(n1[0] - n2[0]) + abs(n1[1] - n2[1])
                d += 1 if n1[2] != n2[2] else 0
                d += 1 if n1[3] != n2[3] else 0
                nd += d

        max_nodes = max(len(nodes1), len(nodes2))
        nd = (compatibility_coe * nd + disjoint_coe * disjoint_nodes) / max_nodes

    cd = 0.0
    if connections1 or connections2:
        disjoint_connections = 0
        for k2 in connections2:
            if k2 not in connections1:
                disjoint_connections += 1

        for k1, c1 in connections1.items():
            c2 = connections2.get(k1)
            if c2 is None:
                disjoint_connections += 1
            else:
                # Homologous genes compute their own distance value.
                d = abs(c1[0] - c2[0])
                d += 1 if c1[1] != c2[1] else 0
                cd += d
        max_conn = max(len(connections1), len(connections2))
        cd = (compatibility_coe * cd + disjoint_coe * disjoint_connections) / max_conn

    return nd + cd


@jit
def distance(nodes1: Array, connections1: Array, nodes2: Array, connections2: Array, disjoint_coe: float = 1.,
             compatibility_coe: float = 0.5) -> Array:
    """
    Calculate the distance between two genomes.
    nodes are a 2-d array with shape (N, 5), its columns are [key, bias, response, act, agg]
    connections are a 3-d array with shape (2, N, N), axis 0 means [weights, enable]
    """

    nd = node_distance(nodes1, nodes2, disjoint_coe, compatibility_coe)  # node distance

    # refactor connections
    keys1, keys2 = nodes1[:, 0], nodes2[:, 0]
    cons1 = flatten_connections(keys1, connections1)
    cons2 = flatten_connections(keys2, connections2)
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
    nan_mask = jnp.isnan(nodes[:, 0])

    intersect_mask = (fr[:, 0] == sr[:, 0]) & ~nan_mask[:-1]

    non_homologous_cnt = node_cnt1 + node_cnt2 - 2 * jnp.sum(intersect_mask)
    nd = batch_homologous_node_distance(fr, sr)
    nd = jnp.where(jnp.isnan(nd), 0, nd)
    homologous_distance = jnp.sum(nd * intersect_mask)

    val = non_homologous_cnt * disjoint_coe + homologous_distance * compatibility_coe
    return jnp.where(max_cnt == 0, 0, val / max_cnt)


@jit
def connection_distance(cons1, cons2, disjoint_coe=1., compatibility_coe=0.5):
    con_cnt1 = jnp.sum(~jnp.isnan(cons1[:, 2]))  # weight is not nan, means the connection exists
    con_cnt2 = jnp.sum(~jnp.isnan(cons2[:, 2]))
    max_cnt = jnp.maximum(con_cnt1, con_cnt2)

    cons = jnp.concatenate((cons1, cons2), axis=0)
    keys = cons[:, :2]
    sorted_indices = jnp.lexsort(keys.T[::-1])
    cons = cons[sorted_indices]
    cons = jnp.concatenate([cons, EMPTY_CON], axis=0)  # add a nan row to the end
    fr, sr = cons[:-1], cons[1:]  # first row, second row

    # both genome has such connection
    intersect_mask = jnp.all(fr[:, :2] == sr[:, :2], axis=1) & ~jnp.isnan(fr[:, 2]) & ~jnp.isnan(sr[:, 2])

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
