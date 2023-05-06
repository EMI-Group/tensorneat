from typing import Tuple
from functools import partial

import numpy as np
from numpy.typing import NDArray
from numpy.random import rand

from .utils import fetch_random, fetch_first, I_INT
from .genome import add_node, add_connection_by_idx, delete_node_by_idx, delete_connection_by_idx
from .graph import check_cycles


def create_mutate_function(config, input_keys, output_keys, batch: bool):
    """
    create mutate function for different situations
    :param output_keys:
    :param input_keys:
    :param config:
    :param batch: mutate for population or not
    :return:
    """
    bias = config.neat.gene.bias
    bias_default = bias.init_mean
    bias_mean = bias.init_mean
    bias_std = bias.init_stdev
    bias_mutate_strength = bias.mutate_power
    bias_mutate_rate = bias.mutate_rate
    bias_replace_rate = bias.replace_rate

    response = config.neat.gene.response
    response_default = response.init_mean
    response_mean = response.init_mean
    response_std = response.init_stdev
    response_mutate_strength = response.mutate_power
    response_mutate_rate = response.mutate_rate
    response_replace_rate = response.replace_rate

    weight = config.neat.gene.weight
    weight_mean = weight.init_mean
    weight_std = weight.init_stdev
    weight_mutate_strength = weight.mutate_power
    weight_mutate_rate = weight.mutate_rate
    weight_replace_rate = weight.replace_rate

    activation = config.neat.gene.activation
    # act_default = activation.default
    act_default = 0
    act_range = len(activation.options)
    act_replace_rate = activation.mutate_rate

    aggregation = config.neat.gene.aggregation
    # agg_default = aggregation.default
    agg_default = 0
    agg_range = len(aggregation.options)
    agg_replace_rate = aggregation.mutate_rate

    enabled = config.neat.gene.enabled
    enabled_reverse_rate = enabled.mutate_rate

    genome = config.neat.genome
    add_node_rate = genome.node_add_prob
    delete_node_rate = genome.node_delete_prob
    add_connection_rate = genome.conn_add_prob
    delete_connection_rate = genome.conn_delete_prob
    single_structure_mutate = genome.single_structural_mutation

    mutate_func = lambda nodes, connections, new_node_key: \
        mutate(nodes, connections, new_node_key, input_keys, output_keys,
               bias_default, bias_mean, bias_std, bias_mutate_strength, bias_mutate_rate,
               bias_replace_rate, response_default, response_mean, response_std,
               response_mutate_strength, response_mutate_rate, response_replace_rate,
               weight_mean, weight_std, weight_mutate_strength, weight_mutate_rate,
               weight_replace_rate, act_default, act_range, act_replace_rate,
               agg_default, agg_range, agg_replace_rate, enabled_reverse_rate,
               add_node_rate, delete_node_rate, add_connection_rate, delete_connection_rate,
               single_structure_mutate)

    if not batch:
        return mutate_func
    else:
        def batch_mutate_func(pop_nodes, pop_connections, new_node_keys):
            res_nodes, res_connections = [], []
            for nodes, connections, new_node_key in zip(pop_nodes, pop_connections, new_node_keys):
                nodes, connections = mutate_func(nodes, connections, new_node_key)
                res_nodes.append(nodes)
                res_connections.append(connections)
            return np.stack(res_nodes, axis=0), np.stack(res_connections, axis=0)

        return batch_mutate_func


def mutate(nodes: NDArray,
           connections: NDArray,
           new_node_key: int,
           input_keys: NDArray,
           output_keys: NDArray,
           bias_default: float = 0,
           bias_mean: float = 0,
           bias_std: float = 1,
           bias_mutate_strength: float = 0.5,
           bias_mutate_rate: float = 0.7,
           bias_replace_rate: float = 0.1,
           response_default: float = 1,
           response_mean: float = 1.,
           response_std: float = 0.,
           response_mutate_strength: float = 0.,
           response_mutate_rate: float = 0.,
           response_replace_rate: float = 0.,
           weight_mean: float = 0.,
           weight_std: float = 1.,
           weight_mutate_strength: float = 0.5,
           weight_mutate_rate: float = 0.7,
           weight_replace_rate: float = 0.1,
           act_default: int = 0,
           act_range: int = 5,
           act_replace_rate: float = 0.1,
           agg_default: int = 0,
           agg_range: int = 5,
           agg_replace_rate: float = 0.1,
           enabled_reverse_rate: float = 0.1,
           add_node_rate: float = 0.2,
           delete_node_rate: float = 0.2,
           add_connection_rate: float = 0.4,
           delete_connection_rate: float = 0.4,
           single_structure_mutate: bool = True):
    """
    :param output_keys:
    :param input_keys:
    :param agg_default:
    :param act_default:
    :param response_default:
    :param bias_default:
    :param nodes: (N, 5)
    :param connections: (2, N, N)
    :param new_node_key:
    :param bias_mean:
    :param bias_std:
    :param bias_mutate_strength:
    :param bias_mutate_rate:
    :param bias_replace_rate:
    :param response_mean:
    :param response_std:
    :param response_mutate_strength:
    :param response_mutate_rate:
    :param response_replace_rate:
    :param weight_mean:
    :param weight_std:
    :param weight_mutate_strength:
    :param weight_mutate_rate:
    :param weight_replace_rate:
    :param act_range:
    :param act_replace_rate:
    :param agg_range:
    :param agg_replace_rate:
    :param enabled_reverse_rate:
    :param add_node_rate:
    :param delete_node_rate:
    :param add_connection_rate:
    :param delete_connection_rate:
    :param single_structure_mutate: a genome is structurally mutate at most once
    :return:
    """

    # mutate_structure
    def nothing(n, c):
        return n, c

    def m_add_node(n, c):
        return mutate_add_node(new_node_key, n, c, bias_default, response_default, act_default, agg_default)

    def m_delete_node(n, c):
        return mutate_delete_node(n, c, input_keys, output_keys)

    def m_add_connection(n, c):
        return mutate_add_connection(n, c, input_keys, output_keys)

    def m_delete_connection(n, c):
        return mutate_delete_connection(n, c)

    if single_structure_mutate:
        d = np.maximum(1, add_node_rate + delete_node_rate + add_connection_rate + delete_connection_rate)

        # shorten variable names for beauty
        anr, dnr = add_node_rate / d, delete_node_rate / d
        acr, dcr = add_connection_rate / d, delete_connection_rate / d

        r = rand()
        if r <= anr:
            nodes, connections = m_add_node(nodes, connections)
        elif r <= anr + dnr:
            nodes, connections = m_delete_node(nodes, connections)
        elif r <= anr + dnr + acr:
            nodes, connections = m_add_connection(nodes, connections)
        elif r <= anr + dnr + acr + dcr:
            nodes, connections = m_delete_connection(nodes, connections)
        else:
            pass  # do nothing

    else:
        # mutate add node
        if rand() < add_node_rate:
            nodes, connections = m_add_node(nodes, connections)

        # mutate delete node
        if rand() < delete_node_rate:
            nodes, connections = m_delete_node(nodes, connections)

        # mutate add connection
        if rand() < add_connection_rate:
            nodes, connections = m_add_connection(nodes, connections)

        # mutate delete connection
        if rand() < delete_connection_rate:
            nodes, connections = m_delete_connection(nodes, connections)

    nodes, connections = mutate_values(nodes, connections, bias_mean, bias_std, bias_mutate_strength,
                                       bias_mutate_rate, bias_replace_rate, response_mean, response_std,
                                       response_mutate_strength, response_mutate_rate, response_replace_rate,
                                       weight_mean, weight_std, weight_mutate_strength,
                                       weight_mutate_rate, weight_replace_rate, act_range, act_replace_rate, agg_range,
                                       agg_replace_rate, enabled_reverse_rate)

    return nodes, connections


def mutate_values(nodes: NDArray,
                  connections: NDArray,
                  bias_mean: float = 0,
                  bias_std: float = 1,
                  bias_mutate_strength: float = 0.5,
                  bias_mutate_rate: float = 0.7,
                  bias_replace_rate: float = 0.1,
                  response_mean: float = 1.,
                  response_std: float = 0.,
                  response_mutate_strength: float = 0.,
                  response_mutate_rate: float = 0.,
                  response_replace_rate: float = 0.,
                  weight_mean: float = 0.,
                  weight_std: float = 1.,
                  weight_mutate_strength: float = 0.5,
                  weight_mutate_rate: float = 0.7,
                  weight_replace_rate: float = 0.1,
                  act_range: int = 5,
                  act_replace_rate: float = 0.1,
                  agg_range: int = 5,
                  agg_replace_rate: float = 0.1,
                  enabled_reverse_rate: float = 0.1) -> Tuple[NDArray, NDArray]:
    """
    Mutate values of nodes and connections.

    Args:
        nodes: A 2D array representing nodes.
        connections: A 3D array representing connections.
        bias_mean: Mean of the bias values.
        bias_std: Standard deviation of the bias values.
        bias_mutate_strength: Strength of the bias mutation.
        bias_mutate_rate: Rate of the bias mutation.
        bias_replace_rate: Rate of the bias replacement.
        response_mean: Mean of the response values.
        response_std: Standard deviation of the response values.
        response_mutate_strength: Strength of the response mutation.
        response_mutate_rate: Rate of the response mutation.
        response_replace_rate: Rate of the response replacement.
        weight_mean: Mean of the weight values.
        weight_std: Standard deviation of the weight values.
        weight_mutate_strength: Strength of the weight mutation.
        weight_mutate_rate: Rate of the weight mutation.
        weight_replace_rate: Rate of the weight replacement.
        act_range: Range of the activation function values.
        act_replace_rate: Rate of the activation function replacement.
        agg_range: Range of the aggregation function values.
        agg_replace_rate: Rate of the aggregation function replacement.
        enabled_reverse_rate: Rate of reversing enabled state of connections.

    Returns:
        A tuple containing mutated nodes and connections.
    """

    bias_new = mutate_float_values(nodes[:, 1], bias_mean, bias_std,
                                   bias_mutate_strength, bias_mutate_rate, bias_replace_rate)
    response_new = mutate_float_values(nodes[:, 2], response_mean, response_std,
                                       response_mutate_strength, response_mutate_rate, response_replace_rate)
    weight_new = mutate_float_values(connections[0, :, :], weight_mean, weight_std,
                                     weight_mutate_strength, weight_mutate_rate, weight_replace_rate)
    act_new = mutate_int_values(nodes[:, 3], act_range, act_replace_rate)
    agg_new = mutate_int_values(nodes[:, 4], agg_range, agg_replace_rate)

    # refactor enabled
    r = np.random.rand(*connections[1, :, :].shape)
    enabled_new = connections[1, :, :] == 1
    enabled_new = np.where(r < enabled_reverse_rate, ~enabled_new, enabled_new)
    enabled_new = np.where(~np.isnan(connections[0, :, :]), enabled_new, np.nan)

    nodes[:, 1] = bias_new
    nodes[:, 2] = response_new
    nodes[:, 3] = act_new
    nodes[:, 4] = agg_new
    connections[0, :, :] = weight_new
    connections[1, :, :] = enabled_new

    return nodes, connections


def mutate_float_values(old_vals: NDArray, mean: float, std: float,
                        mutate_strength: float, mutate_rate: float, replace_rate: float) -> NDArray:
    """
    Mutate float values of a given array.

    Args:
        old_vals: A 1D array of float values to be mutated.
        mean: Mean of the values.
        std: Standard deviation of the values.
        mutate_strength: Strength of the mutation.
        mutate_rate: Rate of the mutation.
        replace_rate: Rate of the replacement.

    Returns:
        A mutated 1D array of float values.
    """
    noise = np.random.normal(size=old_vals.shape) * mutate_strength
    replace = np.random.normal(size=old_vals.shape) * std + mean
    r = rand(*old_vals.shape)
    new_vals = old_vals
    new_vals = np.where(r < mutate_rate, new_vals + noise, new_vals)
    new_vals = np.where(
        np.logical_and(mutate_rate < r, r < mutate_rate + replace_rate),
        replace,
        new_vals
    )
    new_vals = np.where(~np.isnan(old_vals), new_vals, np.nan)
    return new_vals


def mutate_int_values(old_vals: NDArray, range: int, replace_rate: float) -> NDArray:
    """
    Mutate integer values (act, agg) of a given array.

    Args:
        old_vals: A 1D array of integer values to be mutated.
        range: Range of the integer values.
        replace_rate: Rate of the replacement.

    Returns:
        A mutated 1D array of integer values.
    """
    replace_val = np.random.randint(low=0, high=range, size=old_vals.shape)
    r = np.random.rand(*old_vals.shape)
    new_vals = old_vals
    new_vals = np.where(r < replace_rate, replace_val, new_vals)
    new_vals = np.where(~np.isnan(old_vals), new_vals, np.nan)
    return new_vals


def mutate_add_node(new_node_key: int, nodes: NDArray, connections: NDArray,
                    default_bias: float = 0, default_response: float = 1,
                    default_act: int = 0, default_agg: int = 0) -> Tuple[NDArray, NDArray]:
    """
    Randomly add a new node from splitting a connection.
    :param new_node_key:
    :param nodes:
    :param connections:
    :param default_bias:
    :param default_response:
    :param default_act:
    :param default_agg:
    :return:
    """
    # randomly choose a connection
    from_key, to_key, from_idx, to_idx = choice_connection_key(nodes, connections)

    def nothing():
        return nodes, connections

    def successful_add_node():
        # disable the connection
        new_nodes, new_connections = nodes, connections
        new_connections[1, from_idx, to_idx] = False

        # add a new node
        new_nodes, new_connections = \
            add_node(new_node_key, new_nodes, new_connections,
                     bias=default_bias, response=default_response, act=default_act, agg=default_agg)
        new_idx = fetch_first(new_nodes[:, 0] == new_node_key)

        # add two new connections
        weight = new_connections[0, from_idx, to_idx]
        new_nodes, new_connections = add_connection_by_idx(from_idx, new_idx,
                                                           new_nodes, new_connections, weight=0, enabled=True)
        new_nodes, new_connections = add_connection_by_idx(new_idx, to_idx,
                                                           new_nodes, new_connections, weight=weight, enabled=True)
        return new_nodes, new_connections

    # if from_idx == I_INT, that means no connection exist, do nothing
    if from_idx == I_INT:
        nodes, connections = nothing()
    else:
        nodes, connections = successful_add_node()

    return nodes, connections


def mutate_delete_node(nodes: NDArray, connections: NDArray,
                       input_keys: NDArray, output_keys: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Randomly delete a node. Input and output nodes are not allowed to be deleted.
    :param nodes:
    :param connections:
    :param input_keys:
    :param output_keys:
    :return:
    """
    # randomly choose a node
    node_key, node_idx = choice_node_key(nodes, input_keys, output_keys,
                                         allow_input_keys=False, allow_output_keys=False)

    if np.isnan(node_key):
        return nodes, connections

    # delete the node
    aux_nodes, aux_connections = delete_node_by_idx(node_idx, nodes, connections)

    # delete connections
    aux_connections[:, node_idx, :] = np.nan
    aux_connections[:, :, node_idx] = np.nan

    return aux_nodes, aux_connections


def mutate_add_connection(nodes: NDArray, connections: NDArray,
                          input_keys: NDArray, output_keys: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Randomly add a new connection. The output node is not allowed to be an input node. If in feedforward networks,
    cycles are not allowed.
    :param nodes:
    :param connections:
    :param input_keys:
    :param output_keys:
    :return:
    """
    # randomly choose two nodes
    from_key, from_idx = choice_node_key(nodes, input_keys, output_keys,
                                         allow_input_keys=True, allow_output_keys=True)
    to_key, to_idx = choice_node_key(nodes, input_keys, output_keys,
                                     allow_input_keys=False, allow_output_keys=True)

    is_already_exist = ~np.isnan(connections[0, from_idx, to_idx])

    if is_already_exist:
        connections[1, from_idx, to_idx] = True
        return nodes, connections
    elif check_cycles(nodes, connections, from_idx, to_idx):
        return nodes, connections
    else:
        new_nodes, new_connections = add_connection_by_idx(from_idx, to_idx, nodes, connections)
        return new_nodes, new_connections


def mutate_delete_connection(nodes: NDArray, connections: NDArray):
    """
    Randomly delete a connection.
    :param nodes:
    :param connections:
    :return:
    """
    from_key, to_key, from_idx, to_idx = choice_connection_key(nodes, connections)

    def nothing():
        return nodes, connections

    def successfully_delete_connection():
        return delete_connection_by_idx(from_idx, to_idx, nodes, connections)

    if from_idx == I_INT:
        nodes, connections = nothing()
    else:
        nodes, connections = successfully_delete_connection()

    return nodes, connections


def choice_node_key(nodes: NDArray,
                    input_keys: NDArray, output_keys: NDArray,
                    allow_input_keys: bool = False, allow_output_keys: bool = False) -> Tuple[NDArray, NDArray]:
    """
    Randomly choose a node key from the given nodes. It guarantees that the chosen node not be the input or output node.
    :param nodes:
    :param input_keys:
    :param output_keys:
    :param allow_input_keys:
    :param allow_output_keys:
    :return: return its key and position(idx)
    """

    node_keys = nodes[:, 0]
    mask = ~np.isnan(node_keys)

    if not allow_input_keys:
        mask = np.logical_and(mask, ~np.isin(node_keys, input_keys))

    if not allow_output_keys:
        mask = np.logical_and(mask, ~np.isin(node_keys, output_keys))

    idx = fetch_random(mask)

    if idx == I_INT:
        return np.nan, idx
    else:
        return node_keys[idx], idx


def choice_connection_key(nodes: NDArray, connection: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Randomly choose a connection key from the given connections.
    :param nodes:
    :param connection:
    :return: from_key, to_key, from_idx, to_idx
    """
    has_connections_row = np.any(~np.isnan(connection[0, :, :]), axis=1)
    from_idx = fetch_random(has_connections_row)

    if from_idx == I_INT:
        return np.nan, np.nan, from_idx, I_INT

    col = connection[0, from_idx, :]
    to_idx = fetch_random(~np.isnan(col))
    from_key, to_key = nodes[from_idx, 0], nodes[to_idx, 0]

    from_key = np.where(from_idx != I_INT, from_key, np.nan)
    to_key = np.where(to_idx != I_INT, to_key, np.nan)

    return from_key, to_key, from_idx, to_idx
