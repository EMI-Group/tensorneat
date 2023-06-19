"""
Vectorization of genome representation.

Utilizes Tuple[nodes: Array(N, 5), connections: Array(C, 4)] to encode the genome, where:
nodes: [key, bias, response, act, agg]
connections: [in_key, out_key, weight, enable]
N: Maximum number of nodes in the network.
C: Maximum number of connections in the network.
"""

from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray
from jax import jit, numpy as jnp

from .utils import fetch_first


def initialize_genomes(N: int, C: int, config: Dict) -> Tuple[NDArray, NDArray]:
    """
    Initialize genomes with default values.

    Args:
        N (int): Maximum number of nodes in the network.
        C (int): Maximum number of connections in the network.
        config (Dict): Configuration dictionary.

    Returns:
        Tuple[NDArray, NDArray, NDArray, NDArray]: pop_nodes, pop_connections, input_idx, and output_idx arrays.
    """
    # Reserve one row for potential mutation adding an extra node
    assert config['num_inputs'] + config['num_outputs'] + 1 <= N, \
        f"Too small N: {N} for input_size: {config['num_inputs']} and output_size: {config['num_inputs']}!"

    assert config['num_inputs'] * config['num_outputs'] + 1 <= C, \
        f"Too small C: {C} for input_size: {config['num_inputs']} and output_size: {config['num_outputs']}!"

    pop_nodes = np.full((config['pop_size'], N, 5), np.nan)
    pop_cons = np.full((config['pop_size'], C, 4), np.nan)
    input_idx = config['input_idx']
    output_idx = config['output_idx']

    pop_nodes[:, input_idx, 0] = input_idx
    pop_nodes[:, output_idx, 0] = output_idx

    pop_nodes[:, output_idx, 1] = config['bias_init_mean']
    pop_nodes[:, output_idx, 2] = config['response_init_mean']
    pop_nodes[:, output_idx, 3] = config['activation_default']
    pop_nodes[:, output_idx, 4] = config['aggregation_default']

    grid_a, grid_b = np.meshgrid(input_idx, output_idx)
    grid_a, grid_b = grid_a.flatten(), grid_b.flatten()

    p = config['num_inputs'] * config['num_outputs']
    pop_cons[:, :p, 0] = grid_a
    pop_cons[:, :p, 1] = grid_b
    pop_cons[:, :p, 2] = config['weight_init_mean']
    pop_cons[:, :p, 3] = 1

    return pop_nodes, pop_cons


def expand_single(nodes: NDArray, cons: NDArray, new_N: int, new_C: int) -> Tuple[NDArray, NDArray]:
    """
    Expand a single genome to accommodate more nodes or connections.
    :param nodes: (N, 5)
    :param cons:  (C, 4)
    :param new_N:
    :param new_C:
    :return: (new_N, 5), (new_C, 4)
    """
    old_N, old_C = nodes.shape[0], cons.shape[0]
    new_nodes = np.full((new_N, 5), np.nan)
    new_nodes[:old_N, :] = nodes

    new_cons = np.full((new_C, 4), np.nan)
    new_cons[:old_C, :] = cons

    return new_nodes, new_cons


def expand(pop_nodes: NDArray, pop_cons: NDArray, new_N: int, new_C: int) -> Tuple[NDArray, NDArray]:
    """
    Expand the population to accommodate more nodes or connections.
    :param pop_nodes: (pop_size, N, 5)
    :param pop_cons:  (pop_size, C, 4)
    :param new_N:
    :param new_C:
    :return: (pop_size, new_N, 5), (pop_size, new_C, 4)
    """
    pop_size, old_N, old_C = pop_nodes.shape[0], pop_nodes.shape[1], pop_cons.shape[1]

    new_pop_nodes = np.full((pop_size, new_N, 5), np.nan)
    new_pop_nodes[:, :old_N, :] = pop_nodes

    new_pop_cons = np.full((pop_size, new_C, 4), np.nan)
    new_pop_cons[:, :old_C, :] = pop_cons

    return new_pop_nodes, new_pop_cons


@jit
def count(nodes: NDArray, cons: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Count how many nodes and connections are in the genome.
    """
    node_cnt = jnp.sum(~jnp.isnan(nodes[:, 0]))
    cons_cnt = jnp.sum(~jnp.isnan(cons[:, 0]))
    return node_cnt, cons_cnt


@jit
def add_node(nodes: NDArray, cons: NDArray, new_key: int,
             bias: float = 0.0, response: float = 1.0, act: int = 0, agg: int = 0) -> Tuple[NDArray, NDArray]:
    """
    Add a new node to the genome.
    The new node will place at the first NaN row.
    """
    exist_keys = nodes[:, 0]
    idx = fetch_first(jnp.isnan(exist_keys))
    nodes = nodes.at[idx].set(jnp.array([new_key, bias, response, act, agg]))
    return nodes, cons


@jit
def delete_node(nodes: NDArray, cons: NDArray, node_key: int) -> Tuple[NDArray, NDArray]:
    """
    Delete a node from the genome. Only delete the node, regardless of connections.
    Delete the node by its key.
    """
    node_keys = nodes[:, 0]
    idx = fetch_first(node_keys == node_key)
    return delete_node_by_idx(nodes, cons, idx)


@jit
def delete_node_by_idx(nodes: NDArray, cons: NDArray, idx: int) -> Tuple[NDArray, NDArray]:
    """
    Delete a node from the genome. Only delete the node, regardless of connections.
    Delete the node by its idx.
    """
    nodes = nodes.at[idx].set(np.nan)
    return nodes, cons


@jit
def add_connection(nodes: NDArray, cons: NDArray, i_key: int, o_key: int,
                   weight: float = 1.0, enabled: bool = True) -> Tuple[NDArray, NDArray]:
    """
    Add a new connection to the genome.
    The new connection will place at the first NaN row.
    """
    con_keys = cons[:, 0]
    idx = fetch_first(jnp.isnan(con_keys))
    cons = cons.at[idx].set(jnp.array([i_key, o_key, weight, enabled]))
    return nodes, cons


@jit
def delete_connection(nodes: NDArray, cons: NDArray, i_key: int, o_key: int) -> Tuple[NDArray, NDArray]:
    """
    Delete a connection from the genome.
    Delete the connection by its input and output node keys.
    """
    idx = fetch_first((cons[:, 0] == i_key) & (cons[:, 1] == o_key))
    return delete_connection_by_idx(nodes, cons, idx)


@jit
def delete_connection_by_idx(nodes: NDArray, cons: NDArray, idx: int) -> Tuple[NDArray, NDArray]:
    """
    Delete a connection from the genome.
    Delete the connection by its idx.
    """
    cons = cons.at[idx].set(np.nan)
    return nodes, cons
