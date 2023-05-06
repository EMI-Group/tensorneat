import numpy as np
from .genome import Genome
from .gene import NodeGene, ConnectionGene
from .feedforward import FeedForwardNetwork

def object2array(genome, N):
    """
    convert objective genome to array
    :param genome:
    :param N: the size of the array
    :return: Tuple(Array, Array), represents the nodes and connections array
    nodes: shape(N, 5), dtype=float
    connections: shape(2, N, N), dtype=float
    con[:, i, j] != nan, means there is a connection from i to j
    """
    nodes = np.full((N, 5), np.nan)
    connections = np.full((2, N, N), np.nan)

    assert len(genome.nodes) + len(genome.input_keys) + 1 <= N  # remain one inf row for mutation adding extra node

    idx = 0
    n2i = {}
    for i in genome.input_keys:
        nodes[idx, 0] = i
        n2i[i] = idx
        idx += 1

    for k, v in genome.nodes.items():
        nodes[idx, 0] = k
        nodes[idx, 1] = v.bias
        nodes[idx, 2] = v.response
        nodes[idx, 3] = 0
        nodes[idx, 4] = 0
        n2i[k] = idx
        idx += 1

    for (f, t), v in genome.connections.items():
        f_i, t_i = n2i[f], n2i[t]
        connections[0, f_i, t_i] = v.weight
        connections[1, f_i, t_i] = v.enabled

    return nodes, connections


def array2object(config, nodes, connections):
    """
    convert array to genome
    :param config:
    :param nodes:
    :param connections:
    :return:
    """
    genome = Genome(0, config, None, init_val=False)
    genome.input_keys = [0, 1]
    genome.output_keys = [2]
    idx2key = {}
    for i in range(nodes.shape[0]):
        key = nodes[i, 0]
        if np.isnan(key):
            continue
        key = int(key)
        idx2key[i] = key
        if key in genome.input_keys:
            continue
        node_gene = NodeGene(key, config, init_val=False)
        node_gene.bias = nodes[i, 1]
        node_gene.response = nodes[i, 2]
        node_gene.act = 'sigmoid'
        node_gene.agg = 'sum'
        genome.nodes[key] = node_gene

    for i in range(connections.shape[1]):
        for j in range(connections.shape[2]):
            if np.isnan(connections[0, i, j]):
                continue
            key = (idx2key[i], idx2key[j])
            connection_gene = ConnectionGene(key, config, init_val=False)
            connection_gene.weight = connections[0, i, j]
            connection_gene.enabled = connections[1, i, j] == 1
            genome.connections[key] = connection_gene

    return genome
