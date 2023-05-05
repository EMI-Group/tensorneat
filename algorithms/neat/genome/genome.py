"""
Vectorization of genome representation.

Utilizes Tuple[nodes: Array, connections: Array] to encode the genome, where:

1. N is a pre-set value that determines the maximum number of nodes in the network, and will increase if the genome becomes
too large to be represented by the current value of N.
2. nodes is an array of shape (N, 5), dtype=float, with columns corresponding to: key, bias, response, activation function
(act), and aggregation function (agg).
3. connections is an array of shape (2, N, N), dtype=float, with the first axis representing weight and connection enabled
status.
Empty nodes or connections are represented using np.nan.

"""
from typing import Tuple
from functools import partial

import numpy as np
from numpy.typing import NDArray
from jax import numpy as jnp
from jax import jit
from jax import Array

from algorithms.neat.genome.utils import fetch_first, fetch_last

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
    :param pop_nodes:
    :param pop_connections:
    :param new_N:
    :return:
    """
    pop_size, old_N = pop_nodes.shape[0], pop_nodes.shape[1]

    new_pop_nodes = np.full((pop_size, new_N, 5), np.nan)
    new_pop_nodes[:, :old_N, :] = pop_nodes

    new_pop_connections = np.full((pop_size, 2, new_N, new_N), np.nan)
    new_pop_connections[:, :, :old_N, :old_N] = pop_connections
    return new_pop_nodes, new_pop_connections


@jit
def add_node(new_node_key: int, nodes: Array, connections: Array,
             bias: float = 0.0, response: float = 1.0, act: int = 0, agg: int = 0) -> Tuple[Array, Array]:
    """
    add a new node to the genome.
    """
    exist_keys = nodes[:, 0]
    idx = fetch_first(jnp.isnan(exist_keys))
    nodes = nodes.at[idx].set(jnp.array([new_node_key, bias, response, act, agg]))
    return nodes, connections


@jit
def delete_node(node_key: int, nodes: Array, connections: Array) -> Tuple[Array, Array]:
    """
    delete a node from the genome. only delete the node, regardless of connections.
    """
    node_keys = nodes[:, 0]
    idx = fetch_first(node_keys == node_key)
    return delete_node_by_idx(idx, nodes, connections)


@jit
def delete_node_by_idx(idx: int, nodes: Array, connections: Array) -> Tuple[Array, Array]:
    """
    delete a node from the genome. only delete the node, regardless of connections.
    """
    node_keys = nodes[:, 0]
    # move the last node to the deleted node's position
    last_real_idx = fetch_last(~jnp.isnan(node_keys))
    nodes = nodes.at[idx].set(nodes[last_real_idx])
    nodes = nodes.at[last_real_idx].set(EMPTY_NODE)
    return nodes, connections


@jit
def add_connection(from_node: int, to_node: int, nodes: Array, connections: Array,
                   weight: float = 0.0, enabled: bool = True) -> Tuple[Array, Array]:
    """
    add a new connection to the genome.
    """
    node_keys = nodes[:, 0]
    from_idx = fetch_first(node_keys == from_node)
    to_idx = fetch_first(node_keys == to_node)
    return add_connection_by_idx(from_idx, to_idx, nodes, connections, weight, enabled)


@jit
def add_connection_by_idx(from_idx: int, to_idx: int, nodes: Array, connections: Array,
                          weight: float = 0.0, enabled: bool = True) -> Tuple[Array, Array]:
    """
    add a new connection to the genome.
    """
    connections = connections.at[:, from_idx, to_idx].set(jnp.array([weight, enabled]))
    return nodes, connections


@jit
def delete_connection(from_node: int, to_node: int, nodes: Array, connections: Array) -> Tuple[Array, Array]:
    """
    delete a connection from the genome.
    """
    node_keys = nodes[:, 0]
    from_idx = fetch_first(node_keys == from_node)
    to_idx = fetch_first(node_keys == to_node)
    return delete_connection_by_idx(from_idx, to_idx, nodes, connections)


@jit
def delete_connection_by_idx(from_idx: int, to_idx: int, nodes: Array, connections: Array) -> Tuple[Array, Array]:
    """
    delete a connection from the genome.
    """
    connections = connections.at[:, from_idx, to_idx].set(np.nan)
    return nodes, connections

# if __name__ == '__main__':
#     pop_nodes, pop_connections, input_keys, output_keys = initialize_genomes(100, 5, 2, 1)
#     print(pop_nodes, pop_connections)
