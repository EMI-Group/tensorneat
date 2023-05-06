from functools import partial

import numpy as np
from numpy.typing import NDArray

from .aggregations import agg
from .activations import act
from .graph import topological_sort, batch_topological_sort
from .utils import I_INT


def create_forward_function(nodes: NDArray, connections: NDArray,
                            N: int, input_idx: NDArray, output_idx: NDArray, batch: bool):
    """
    create forward function for different situations

    :param nodes: shape (N, 5) or (pop_size, N, 5)
    :param connections: shape (2, N, N) or (pop_size, 2, N, N)
    :param N:
    :param input_idx:
    :param output_idx:
    :param batch: using batch or not
    :param debug: debug mode
    :return:
    """

    if nodes.ndim == 2:  # single genome
        cal_seqs = topological_sort(nodes, connections)
        if not batch:
            return lambda inputs: forward_single(inputs, N, input_idx, output_idx,
                                                 cal_seqs, nodes, connections)
        else:
            return lambda batch_inputs: forward_batch(batch_inputs, N, input_idx, output_idx,
                                                      cal_seqs, nodes, connections)
    elif nodes.ndim == 3:  # pop genome
        pop_cal_seqs = batch_topological_sort(nodes, connections)
        if not batch:
            return lambda inputs: pop_forward_single(inputs, N, input_idx, output_idx,
                                                     pop_cal_seqs, nodes, connections)
        else:
            return lambda batch_inputs: pop_forward_batch(batch_inputs, N, input_idx, output_idx,
                                                          pop_cal_seqs, nodes, connections)
    else:
        raise ValueError(f"nodes.ndim should be 2 or 3, but got {nodes.ndim}")


def forward_single(inputs: NDArray, N: int, input_idx: NDArray, output_idx: NDArray,
                   cal_seqs: NDArray, nodes: NDArray, connections: NDArray) -> NDArray:
    """
    jax forward for single input shaped (input_num, )
    nodes, connections are single genome

    :argument inputs: (input_num, )
    :argument N: int
    :argument input_idx: (input_num, )
    :argument output_idx: (output_num, )
    :argument cal_seqs: (N, )
    :argument nodes: (N, 5)
    :argument connections: (2, N, N)

    :return (output_num, )
    """
    ini_vals = np.full((N,), np.nan)
    ini_vals[input_idx] = inputs

    for i in cal_seqs:
        if i in input_idx:
            continue
        if i == I_INT:
            break
        ins = ini_vals * connections[0, :, i]
        z = agg(nodes[i, 4], ins)
        z = z * nodes[i, 2] + nodes[i, 1]
        z = act(nodes[i, 3], z)

        # for some nodes (inputs nodes), the output z will be nan, thus we do not update the vals
        ini_vals[i] = z


    return ini_vals[output_idx]


def forward_batch(batch_inputs: NDArray, N: int, input_idx: NDArray, output_idx: NDArray,
                  cal_seqs: NDArray, nodes: NDArray, connections: NDArray) -> NDArray:
    """
    jax forward for batch_inputs shaped (batch_size, input_num)
    nodes, connections are single genome

    :argument batch_inputs: (batch_size, input_num)
    :argument N: int
    :argument input_idx: (input_num, )
    :argument output_idx: (output_num, )
    :argument cal_seqs: (N, )
    :argument nodes: (N, 5)
    :argument connections: (2, N, N)

    :return (batch_size, output_num)
    """
    res = []
    for inputs in batch_inputs:
        out = forward_single(inputs, N, input_idx, output_idx, cal_seqs, nodes, connections)
        res.append(out)
    return np.stack(res, axis=0)


def pop_forward_single(inputs: NDArray, N: int, input_idx: NDArray, output_idx: NDArray,
                       pop_cal_seqs: NDArray, pop_nodes: NDArray, pop_connections: NDArray) -> NDArray:
    """
    jax forward for single input shaped (input_num, )
    pop_nodes, pop_connections are population of genomes

    :argument inputs: (input_num, )
    :argument N: int
    :argument input_idx: (input_num, )
    :argument output_idx: (output_num, )
    :argument pop_cal_seqs: (pop_size, N)
    :argument pop_nodes: (pop_size, N, 5)
    :argument pop_connections: (pop_size, 2, N, N)

    :return (pop_size, output_num)
    """
    res = []
    for cal_seqs, nodes, connections in zip(pop_cal_seqs, pop_nodes, pop_connections):
        out = forward_single(inputs, N, input_idx, output_idx, cal_seqs, nodes, connections)
        res.append(out)

    return np.stack(res, axis=0)


def pop_forward_batch(batch_inputs: NDArray, N: int, input_idx: NDArray, output_idx: NDArray,
                      pop_cal_seqs: NDArray, pop_nodes: NDArray, pop_connections: NDArray) -> NDArray:
    """
    jax forward for batch input shaped (batch, input_num)
    pop_nodes, pop_connections are population of genomes

    :argument batch_inputs: (batch_size, input_num)
    :argument N: int
    :argument input_idx: (input_num, )
    :argument output_idx: (output_num, )
    :argument pop_cal_seqs: (pop_size, N)
    :argument pop_nodes: (pop_size, N, 5)
    :argument pop_connections: (pop_size, 2, N, N)

    :return (pop_size, batch_size, output_num)
    """
    res = []
    for cal_seqs, nodes, connections in zip(pop_cal_seqs, pop_nodes, pop_connections):
        out = forward_batch(batch_inputs, N, input_idx, output_idx, cal_seqs, nodes, connections)
        res.append(out)

    return np.stack(res, axis=0)
