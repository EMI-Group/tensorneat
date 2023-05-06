from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .utils import flatten_connections, unflatten_connections


def batch_crossover(batch_nodes1: NDArray, batch_connections1: NDArray, batch_nodes2: NDArray,
                    batch_connections2: NDArray) -> Tuple[NDArray, NDArray]:
    """
    crossover a batch of genomes
    :param batch_nodes1:
    :param batch_connections1:
    :param batch_nodes2:
    :param batch_connections2:
    :return:
    """
    res_nodes, res_cons = [], []
    for (n1, c1, n2, c2) in zip(batch_nodes1, batch_connections1, batch_nodes2, batch_connections2):
        new_nodes, new_cons = crossover(n1, c1, n2, c2)
        res_nodes.append(new_nodes)
        res_cons.append(new_cons)
    return np.stack(res_nodes, axis=0), np.stack(res_cons, axis=0)


def crossover(nodes1: NDArray, connections1: NDArray, nodes2: NDArray, connections2: NDArray) \
        -> Tuple[NDArray, NDArray]:
    """
    use genome1 and genome2 to generate a new genome
    notice that genome1 should have higher fitness than genome2 (genome1 is winner!)
    :param nodes1:
    :param connections1:
    :param nodes2:
    :param connections2:
    :return:
    """

    # crossover nodes
    keys1, keys2 = nodes1[:, 0], nodes2[:, 0]
    nodes2 = align_array(keys1, keys2, nodes2, 'node')
    new_nodes = np.where(np.isnan(nodes1) | np.isnan(nodes2), nodes1, crossover_gene(nodes1, nodes2))

    # crossover connections
    cons1 = flatten_connections(keys1, connections1)
    cons2 = flatten_connections(keys2, connections2)
    con_keys1, con_keys2 = cons1[:, :2], cons2[:, :2]
    cons2 = align_array(con_keys1, con_keys2, cons2, 'connection')
    new_cons = np.where(np.isnan(cons1) | np.isnan(cons2), cons1, crossover_gene(cons1, cons2))
    new_cons = unflatten_connections(len(keys1), new_cons)

    return new_nodes, new_cons


def align_array(seq1: NDArray, seq2: NDArray, ar2: NDArray, gene_type: str) -> NDArray:
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
    seq1, seq2 = seq1[:, np.newaxis], seq2[np.newaxis, :]
    mask = (seq1 == seq2) & (~np.isnan(seq1))

    if gene_type == 'connection':
        mask = np.all(mask, axis=2)

    intersect_mask = mask.any(axis=1)
    idx = np.arange(0, len(seq1))
    idx_fixed = np.dot(mask, idx)

    refactor_ar2 = np.where(intersect_mask[:, np.newaxis], ar2[idx_fixed], np.nan)

    return refactor_ar2


def crossover_gene(g1: NDArray, g2: NDArray) -> NDArray:
    """
    crossover two genes
    :param g1:
    :param g2:
    :return:
    only gene with the same key will be crossover, thus don't need to consider change key
    """
    r = np.random.rand()
    return np.where(r > 0.5, g1, g2)
