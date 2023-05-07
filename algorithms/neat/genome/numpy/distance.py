from functools import partial

import numpy as np
from numpy.typing import NDArray

from algorithms.neat.genome.utils import flatten_connections, set_operation_analysis

EMPTY_NODE = np.full((1, 5), np.nan)
EMPTY_CON = np.full((1, 4), np.nan)


def distance(nodes1: NDArray, connections1: NDArray, nodes2: NDArray, connections2: NDArray) -> NDArray:
    """
    Calculate the distance between two genomes.
    nodes are a 2-d array with shape (N, 5), its columns are [key, bias, response, act, agg]
    connections are a 3-d array with shape (2, N, N), axis 0 means [weights, enable]
    """

    nd = node_distance(nodes1, nodes2)  # node distance

    # refactor connections
    keys1, keys2 = nodes1[:, 0], nodes2[:, 0]
    cons1 = flatten_connections(keys1, connections1)
    cons2 = flatten_connections(keys2, connections2)
    cd = connection_distance(cons1, cons2)  # connection distance
    return nd + cd


def node_distance(nodes1, nodes2, disjoint_coe=1., compatibility_coe=0.5):
    node_cnt1 = np.sum(~np.isnan(nodes1[:, 0]))
    node_cnt2 = np.sum(~np.isnan(nodes2[:, 0]))
    max_cnt = np.maximum(node_cnt1, node_cnt2)

    nodes = np.concatenate((nodes1, nodes2), axis=0)
    keys = nodes[:, 0]
    sorted_indices = np.argsort(keys, axis=0)
    nodes = nodes[sorted_indices]
    nodes = np.concatenate([nodes, EMPTY_NODE], axis=0)  # add a nan row to the end
    fr, sr = nodes[:-1], nodes[1:]  # first row, second row
    nan_mask = np.isnan(nodes[:, 0])

    intersect_mask = (fr[:, 0] == sr[:, 0]) & ~nan_mask[:-1]

    non_homologous_cnt = node_cnt1 + node_cnt2 - 2 * np.sum(intersect_mask)
    nd = batch_homologous_node_distance(fr, sr)
    nd = np.where(np.isnan(nd), 0, nd)
    homologous_distance = np.sum(nd * intersect_mask)

    val = non_homologous_cnt * disjoint_coe + homologous_distance * compatibility_coe

    if max_cnt == 0:  # consider the case that both genome has no gene
        return 0
    else:
        return val / max_cnt


def connection_distance(cons1, cons2, disjoint_coe=1., compatibility_coe=0.5):
    con_cnt1 = np.sum(~np.isnan(cons1[:, 2]))  # weight is not nan, means the connection exists
    con_cnt2 = np.sum(~np.isnan(cons2[:, 2]))
    max_cnt = np.maximum(con_cnt1, con_cnt2)

    cons = np.concatenate((cons1, cons2), axis=0)
    keys = cons[:, :2]
    sorted_indices = np.lexsort(keys.T[::-1])
    cons = cons[sorted_indices]
    cons = np.concatenate([cons, EMPTY_CON], axis=0)  # add a nan row to the end
    fr, sr = cons[:-1], cons[1:]  # first row, second row

    # both genome has such connection
    intersect_mask = np.all(fr[:, :2] == sr[:, :2], axis=1) & ~np.isnan(fr[:, 2]) & ~np.isnan(sr[:, 2])

    non_homologous_cnt = con_cnt1 + con_cnt2 - 2 * np.sum(intersect_mask)
    cd = batch_homologous_connection_distance(fr, sr)
    cd = np.where(np.isnan(cd), 0, cd)
    homologous_distance = np.sum(cd * intersect_mask)

    val = non_homologous_cnt * disjoint_coe + homologous_distance * compatibility_coe

    if max_cnt == 0:  # consider the case that both genome has no gene
        return 0
    else:
        return val / max_cnt


def batch_homologous_node_distance(b_n1, b_n2):
    res = []
    for n1, n2 in zip(b_n1, b_n2):
        d = homologous_node_distance(n1, n2)
        res.append(d)
    return np.stack(res, axis=0)


def batch_homologous_connection_distance(b_c1, b_c2):
    res = []
    for c1, c2 in zip(b_c1, b_c2):
        d = homologous_connection_distance(c1, c2)
        res.append(d)
    return np.stack(res, axis=0)


def homologous_node_distance(n1, n2):
    d = 0
    d += np.abs(n1[1] - n2[1])  # bias
    d += np.abs(n1[2] - n2[2])  # response
    d += n1[3] != n2[3]  # activation
    d += n1[4] != n2[4]
    return d


def homologous_connection_distance(c1, c2):
    d = 0
    d += np.abs(c1[2] - c2[2])  # weight
    d += c1[3] != c2[3]  # enable
    return d
