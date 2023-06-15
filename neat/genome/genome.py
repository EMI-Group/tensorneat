"""
Vectorization of genome representation.

Utilizes Tuple[nodes: Array, connections: Array] to encode the genome, where:

1. N, C are pre-set values that determines the maximum number of nodes and connections in the network, and will increase if the genome becomes
too large to be represented by the current value of N and C.
2. nodes is an array of shape (N, 5), dtype=float, with columns corresponding to: key, bias, response, activation function
(act), and aggregation function (agg).
3. connections is an array of shape (C, 4), dtype=float, with columns corresponding to: i_key, o_key, weight, enabled.
Empty nodes or connections are represented using np.nan.

"""
from typing import Tuple, Dict

import jax
import numpy as np
from numpy.typing import NDArray
from jax import numpy as jnp
from jax import jit
from jax import Array

from .utils import fetch_first


def initialize_genomes(pop_size: int,
                       N: int,
                       C: int,
                       num_inputs: int,
                       num_outputs: int,
                       default_bias: float = 0.0,
                       default_response: float = 1.0,
                       default_act: int = 0,
                       default_agg: int = 0,
                       default_weight: float = 0.0) \
        -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Initialize genomes with default values.

    Args:
        pop_size (int): Number of genomes to initialize.
        N (int): Maximum number of nodes in the network.
        C (int): Maximum number of connections in the network.
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
    assert num_inputs * num_outputs + 1 <= C, f"Too small C: {C} for input_size: " \
                                              f"{num_inputs} and output_size: {num_outputs}!"

    pop_nodes = np.full((pop_size, N, 5), np.nan)
    pop_cons = np.full((pop_size, C, 4), np.nan)
    input_idx = np.arange(num_inputs)
    output_idx = np.arange(num_inputs, num_inputs + num_outputs)

    pop_nodes[:, input_idx, 0] = input_idx
    pop_nodes[:, output_idx, 0] = output_idx

    pop_nodes[:, output_idx, 1] = default_bias
    pop_nodes[:, output_idx, 2] = default_response
    pop_nodes[:, output_idx, 3] = default_act
    pop_nodes[:, output_idx, 4] = default_agg

    grid_a, grid_b = np.meshgrid(input_idx, output_idx)
    grid_a, grid_b = grid_a.flatten(), grid_b.flatten()

    pop_cons[:, :num_inputs * num_outputs, 0] = grid_a
    pop_cons[:, :num_inputs * num_outputs, 1] = grid_b
    pop_cons[:, :num_inputs * num_outputs, 2] = default_weight
    pop_cons[:, :num_inputs * num_outputs, 3] = 1

    return pop_nodes, pop_cons, input_idx, output_idx


def expand(pop_nodes: NDArray, pop_cons: NDArray, new_N: int, new_C: int) -> Tuple[NDArray, NDArray]:
    """
    Expand the genome to accommodate more nodes.
    :param pop_nodes: (pop_size, N, 5)
    :param pop_cons:  (pop_size, C, 4)
    :param new_N:
    :param new_C:
    :return:
    """
    pop_size, old_N, old_C = pop_nodes.shape[0], pop_nodes.shape[1], pop_cons.shape[1]

    new_pop_nodes = np.full((pop_size, new_N, 5), np.nan)
    new_pop_nodes[:, :old_N, :] = pop_nodes

    new_pop_cons = np.full((pop_size, new_C, 4), np.nan)
    new_pop_cons[:, :old_C, :] = pop_cons

    return new_pop_nodes, new_pop_cons


def expand_single(nodes: NDArray, cons: NDArray, new_N: int, new_C: int) -> Tuple[NDArray, NDArray]:
    """
    Expand a single genome to accommodate more nodes.
    :param nodes: (N, 5)
    :param cons:  (2, N, N)
    :param new_N:
    :param new_C:
    :return:
    """
    old_N, old_C = nodes.shape[0], cons.shape[0]
    new_nodes = np.full((new_N, 5), np.nan)
    new_nodes[:old_N, :] = nodes

    new_cons = np.full((new_C, 4), np.nan)
    new_cons[:old_C, :] = cons

    return new_nodes, new_cons


@jit
def count(nodes, cons):
    node_cnt = jnp.sum(~jnp.isnan(nodes[:, 0]))
    cons_cnt = jnp.sum(~jnp.isnan(cons[:, 0]))
    return node_cnt, cons_cnt


@jit
def add_node(nodes: Array, cons: Array, new_key: int,
             bias: float = 0.0, response: float = 1.0, act: int = 0, agg: int = 0) -> Tuple[Array, Array]:
    """
    add a new node to the genome.
    """
    exist_keys = nodes[:, 0]
    idx = fetch_first(jnp.isnan(exist_keys))
    nodes = nodes.at[idx].set(jnp.array([new_key, bias, response, act, agg]))
    return nodes, cons


@jit
def delete_node(nodes: Array, cons: Array, node_key: int) -> Tuple[Array, Array]:
    """
    delete a node from the genome. only delete the node, regardless of connections.
    """
    node_keys = nodes[:, 0]
    idx = fetch_first(node_keys == node_key)
    return delete_node_by_idx(nodes, cons, idx)


@jit
def delete_node_by_idx(nodes: Array, cons: Array, idx: int) -> Tuple[Array, Array]:
    """
    use idx to delete a node from the genome. only delete the node, regardless of connections.
    """
    nodes = nodes.at[idx].set(np.nan)
    return nodes, cons


@jit
def add_connection(nodes: Array, cons: Array, i_key: int, o_key: int,
                   weight: float = 1.0, enabled: bool = True) -> Tuple[Array, Array]:
    """
    add a new connection to the genome.
    """
    con_keys = cons[:, 0]
    idx = fetch_first(jnp.isnan(con_keys))
    return add_connection_by_idx(nodes, cons, idx, i_key, o_key, weight, enabled)


@jit
def add_connection_by_idx(nodes: Array, cons: Array, idx: int, i_key: int, o_key: int,
                          weight: float = 0.0, enabled: bool = True) -> Tuple[Array, Array]:
    """
    use idx to add a new connection to the genome.
    """
    cons = cons.at[idx].set(jnp.array([i_key, o_key, weight, enabled]))
    return nodes, cons


@jit
def delete_connection(nodes: Array, cons: Array, i_key: int, o_key: int) -> Tuple[Array, Array]:
    """
    delete a connection from the genome.
    """
    idx = fetch_first((cons[:, 0] == i_key) & (cons[:, 1] == o_key))
    return delete_connection_by_idx(nodes, cons, idx)


@jit
def delete_connection_by_idx(nodes: Array, cons: Array, idx: int) -> Tuple[Array, Array]:
    """
    use idx to delete a connection from the genome.
    """
    cons = cons.at[idx].set(np.nan)
    return nodes, cons
