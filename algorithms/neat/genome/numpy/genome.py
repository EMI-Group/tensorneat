"""
Vectorization of genome representation.

Utilizes Tuple[nodes: NDArray, connections: NDArray] to encode the genome, where:

1. N is a pre-set value that determines the maximum number of nodes in the network, and will increase if the genome becomes
too large to be represented by the current value of N.
2. nodes is an array of shape (N, 5), dtype=float, with columns corresponding to: key, bias, response, activation function
(act), and aggregation function (agg).
3. connections is an array of shape (2, N, N), dtype=float, with the first axis representing weight and connection enabled
status.
Empty nodes or connections are represented using np.nan.

"""
from typing import Tuple, Dict
from functools import partial

import numpy as np
from numpy.typing import NDArray

from algorithms.neat.genome.utils import fetch_first

EMPTY_NODE = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])


def create_initialize_function(config):
    pop_size = config.neat.population.pop_size
    N = config.basic.init_maximum_nodes
    num_inputs = config.basic.num_inputs
    num_outputs = config.basic.num_outputs
    default_bias = config.neat.gene.bias.init_mean
    default_response = config.neat.gene.response.init_mean
    # default_act = config.neat.gene.activation.default
    # default_agg = config.neat.gene.aggregation.default
    default_act = 0
    default_agg = 0
    default_weight = config.neat.gene.weight.init_mean
    return partial(initialize_genomes, pop_size, N, num_inputs, num_outputs, default_bias, default_response,
                   default_act, default_agg, default_weight)


def initialize_genomes(pop_size: int,
                       N: int,
                       num_inputs: int, num_outputs: int,
                       default_bias: float = 0.0,
                       default_response: float = 1.0,
                       default_act: int = 0,
                       default_agg: int = 0,
                       default_weight: float = 1.0) \
        -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Initialize genomes with default values.

    Args:
        pop_size (int): Number of genomes to initialize.
        N (int): Maximum number of nodes in the network.
        num_inputs (int): Number of input nodes.
        num_outputs (int): Number of output nodes.
        default_bias (float, optional): Default bias value for output nodes. Defaults to 0.0.
        default_response (float, optional): Default response value for output nodes. Defaults to 1.0.
        default_act (int, optional): Default activation function index for output nodes. Defaults to 1.
        default_agg (int, optional): Default aggregation function index for output nodes. Defaults to 0.
        default_weight (float, optional): Default weight value for connections. Defaults to 0.0.

    Raises:
        AssertionError: If the sum of num_inputs, num_outputs, and 1 is greater than N.

    Returns:
        Tuple[NDArray, NDArray, NDArray, NDArray]: pop_nodes, pop_connections, input_idx, and output_idx arrays.
    """
    # Reserve one row for potential mutation adding an extra node
    assert num_inputs + num_outputs + 1 <= N, f"Too small N: {N} for input_size: " \
                                              f"{num_inputs} and output_size: {num_outputs}!"

    pop_nodes = np.full((pop_size, N, 5), np.nan)
    pop_connections = np.full((pop_size, 2, N, N), np.nan)
    input_idx = np.arange(num_inputs)
    output_idx = np.arange(num_inputs, num_inputs + num_outputs)

    pop_nodes[:, input_idx, 0] = input_idx
    pop_nodes[:, output_idx, 0] = output_idx

    pop_nodes[:, output_idx, 1] = default_bias
    pop_nodes[:, output_idx, 2] = default_response
    pop_nodes[:, output_idx, 3] = default_act
    pop_nodes[:, output_idx, 4] = default_agg

    for i in input_idx:
        for j in output_idx:
            pop_connections[:, 0, i, j] = default_weight
            pop_connections[:, 1, i, j] = 1

    return pop_nodes, pop_connections, input_idx, output_idx


def expand(pop_nodes: NDArray, pop_connections: NDArray, new_N: int) -> Tuple[NDArray, NDArray]:
    """
    Expand the genome to accommodate more nodes.
    :param pop_nodes: (pop_size, N, 5)
    :param pop_connections:  (pop_size, 2, N, N)
    :param new_N:
    :return:
    """
    pop_size, old_N = pop_nodes.shape[0], pop_nodes.shape[1]

    new_pop_nodes = np.full((pop_size, new_N, 5), np.nan)
    new_pop_nodes[:, :old_N, :] = pop_nodes

    new_pop_connections = np.full((pop_size, 2, new_N, new_N), np.nan)
    new_pop_connections[:, :, :old_N, :old_N] = pop_connections
    return new_pop_nodes, new_pop_connections


def expand_single(nodes: NDArray, connections: NDArray, new_N: int) -> Tuple[NDArray, NDArray]:
    """
    Expand a single genome to accommodate more nodes.
    :param nodes: (N, 5)
    :param connections:  (2, N, N)
    :param new_N:
    :return:
    """
    old_N = nodes.shape[0]
    new_nodes = np.full((new_N, 5), np.nan)
    new_nodes[:old_N, :] = nodes

    new_connections = np.full((2, new_N, new_N), np.nan)
    new_connections[:, :old_N, :old_N] = connections

    return new_nodes, new_connections


def analysis(nodes: NDArray, connections: NDArray, input_keys, output_keys) -> \
        Tuple[Dict[int, Tuple[float, float, int, int]], Dict[Tuple[int, int], Tuple[float, bool]]]:
    """
    Convert a genome from array to dict.
    :param nodes: (N, 5)
    :param connections: (2, N, N)
    :param output_keys:
    :param input_keys:
    :return: nodes_dict[key: (bias, response, act, agg)], connections_dict[(f_key, t_key): (weight, enabled)]
    """
    # update nodes_dict
    try:
        nodes_dict = {}
        idx2key = {}
        for i, node in enumerate(nodes):
            if np.isnan(node[0]):
                continue
            key = int(node[0])
            assert key not in nodes_dict, f"Duplicate node key: {key}!"

            bias = node[1] if not np.isnan(node[1]) else None
            response = node[2] if not np.isnan(node[2]) else None
            act = node[3] if not np.isnan(node[3]) else None
            agg = node[4] if not np.isnan(node[4]) else None
            nodes_dict[key] = (bias, response, act, agg)
            idx2key[i] = key

        # check nodes_dict
        for i in input_keys:
            assert i in nodes_dict, f"Input node {i} not found in nodes_dict!"
            bias, response, act, agg = nodes_dict[i]
            assert bias is None and response is None and act is None and agg is None, \
                f"Input node {i} must has None bias, response, act, or agg!"

        for o in output_keys:
            assert o in nodes_dict, f"Output node {o} not found in nodes_dict!"

        for k, v in nodes_dict.items():
            if k not in input_keys:
                bias, response, act, agg = v
                assert bias is not None and response is not None and act is not None and agg is not None, \
                    f"Normal node {k} must has non-None bias, response, act, or agg!"

        # update connections
        connections_dict = {}
        for i in range(connections.shape[1]):
            for j in range(connections.shape[2]):
                if np.isnan(connections[0, i, j]) and np.isnan(connections[1, i, j]):
                    continue
                assert i in idx2key, f"Node index {i} not found in idx2key:{idx2key}!"
                assert j in idx2key, f"Node index {j} not found in idx2key:{idx2key}!"
                key = (idx2key[i], idx2key[j])

                weight = connections[0, i, j] if not np.isnan(connections[0, i, j]) else None
                enabled = (connections[1, i, j] == 1) if not np.isnan(connections[1, i, j]) else None

                assert weight is not None, f"Connection {key} must has non-None weight!"
                assert enabled is not None, f"Connection {key} must has non-None enabled!"
                connections_dict[key] = (weight, enabled)

        return nodes_dict, connections_dict
    except AssertionError:
        print(nodes)
        print(connections)
        raise AssertionError


def pop_analysis(pop_nodes, pop_connections, input_keys, output_keys):
    res = []
    total_nodes, total_connections = 0, 0
    for nodes, connections in zip(pop_nodes, pop_connections):
        nodes, connections = analysis(nodes, connections, input_keys, output_keys)
        res.append((nodes, connections))
        total_nodes += len(nodes)
        total_connections += len(connections)
    print(total_nodes - 200, total_connections)
    return res


def add_node(new_node_key: int, nodes: NDArray, connections: NDArray,
             bias: float = 0.0, response: float = 1.0, act: int = 0, agg: int = 0) -> Tuple[NDArray, NDArray]:
    """
    add a new node to the genome.
    """
    exist_keys = nodes[:, 0]
    idx = fetch_first(np.isnan(exist_keys))
    nodes[idx] = np.array([new_node_key, bias, response, act, agg])
    return nodes, connections


def delete_node(node_key: int, nodes: NDArray, connections: NDArray) -> Tuple[NDArray, NDArray]:
    """
    delete a node from the genome. only delete the node, regardless of connections.
    """
    node_keys = nodes[:, 0]
    idx = fetch_first(node_keys == node_key)
    return delete_node_by_idx(idx, nodes, connections)


def delete_node_by_idx(idx: int, nodes: NDArray, connections: NDArray) -> Tuple[NDArray, NDArray]:
    """
    delete a node from the genome. only delete the node, regardless of connections.
    """
    nodes[idx] = EMPTY_NODE
    return nodes, connections


def add_connection(from_node: int, to_node: int, nodes: NDArray, connections: NDArray,
                   weight: float = 0.0, enabled: bool = True) -> Tuple[NDArray, NDArray]:
    """
    add a new connection to the genome.
    """
    node_keys = nodes[:, 0]
    from_idx = fetch_first(node_keys == from_node)
    to_idx = fetch_first(node_keys == to_node)
    return add_connection_by_idx(from_idx, to_idx, nodes, connections, weight, enabled)


def add_connection_by_idx(from_idx: int, to_idx: int, nodes: NDArray, connections: NDArray,
                          weight: float = 0.0, enabled: bool = True) -> Tuple[NDArray, NDArray]:
    """
    add a new connection to the genome.
    """
    connections[:, from_idx, to_idx] = np.array([weight, enabled])
    return nodes, connections


def delete_connection(from_node: int, to_node: int, nodes: NDArray, connections: NDArray) -> Tuple[NDArray, NDArray]:
    """
    delete a connection from the genome.
    """
    node_keys = nodes[:, 0]
    from_idx = fetch_first(node_keys == from_node)
    to_idx = fetch_first(node_keys == to_node)
    return delete_connection_by_idx(from_idx, to_idx, nodes, connections)


def delete_connection_by_idx(from_idx: int, to_idx: int, nodes: NDArray, connections: NDArray) -> Tuple[
    NDArray, NDArray]:
    """
    delete a connection from the genome.
    """
    connections[:, from_idx, to_idx] = np.nan
    return nodes, connections
