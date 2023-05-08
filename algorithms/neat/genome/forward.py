from functools import partial

import jax
from jax import Array, numpy as jnp
from jax import jit, vmap
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


@partial(jit, static_argnames=['N'])
def forward_single(inputs: Array, N: int, input_idx: Array, output_idx: Array,
                   cal_seqs: Array, nodes: Array, connections: Array) -> Array:
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
    ini_vals = jnp.full((N,), jnp.nan)
    ini_vals = ini_vals.at[input_idx].set(inputs)

    def scan_body(carry, i):
        def hit():
            ins = carry * connections[0, :, i]
            z = agg(nodes[i, 4], ins)
            z = z * nodes[i, 2] + nodes[i, 1]
            z = act(nodes[i, 3], z)

            new_vals = carry.at[i].set(z)
            return new_vals

        def miss():
            return carry

        return jax.lax.cond((i == I_INT) | (jnp.isin(i, input_idx)), miss, hit), None

    vals, _ = jax.lax.scan(scan_body, ini_vals, cal_seqs)

    return vals[output_idx]


@partial(jit, static_argnames=['N'])
@partial(vmap, in_axes=(0, None, None, None, None, None, None))
def forward_batch(batch_inputs: Array, N: int, input_idx: Array, output_idx: Array,
                  cal_seqs: Array, nodes: Array, connections: Array) -> Array:
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
    return forward_single(batch_inputs, N, input_idx, output_idx, cal_seqs, nodes, connections)


@partial(jit, static_argnames=['N'])
@partial(vmap, in_axes=(None, None, None, None, 0, 0, 0))
def pop_forward_single(inputs: Array, N: int, input_idx: Array, output_idx: Array,
                       pop_cal_seqs: Array, pop_nodes: Array, pop_connections: Array) -> Array:
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
    return forward_single(inputs, N, input_idx, output_idx, pop_cal_seqs, pop_nodes, pop_connections)


@partial(jit, static_argnames=['N'])
@partial(vmap, in_axes=(None, None, None, None, 0, 0, 0))
def pop_forward_batch(batch_inputs: Array, N: int, input_idx: Array, output_idx: Array,
                      pop_cal_seqs: Array, pop_nodes: Array, pop_connections: Array) -> Array:
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
    return forward_batch(batch_inputs, N, input_idx, output_idx, pop_cal_seqs, pop_nodes, pop_connections)
