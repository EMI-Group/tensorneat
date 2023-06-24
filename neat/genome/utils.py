from functools import partial

import jax
from jax import numpy as jnp, Array
from jax import jit, vmap

I_INT = jnp.iinfo(jnp.int32).max  # infinite int
EMPTY_NODE = jnp.full((1, 5), jnp.nan)
EMPTY_CON = jnp.full((1, 4), jnp.nan)


@jit
def unflatten_connections(nodes: Array, cons: Array):
    """
    transform the (C, 4) connections to (2, N, N)
    :param nodes: (N, 5)
    :param cons: (C, 4)
    :return:
    """
    N = nodes.shape[0]
    node_keys = nodes[:, 0]
    i_keys, o_keys = cons[:, 0], cons[:, 1]
    i_idxs = vmap(key_to_indices, in_axes=(0, None))(i_keys, node_keys)
    o_idxs = vmap(key_to_indices, in_axes=(0, None))(o_keys, node_keys)
    res = jnp.full((2, N, N), jnp.nan)

    # Is interesting that jax use clip when attach data in array
    # however, it will do nothing set values in an array
    res = res.at[0, i_idxs, o_idxs].set(cons[:, 2])
    res = res.at[1, i_idxs, o_idxs].set(cons[:, 3])

    return res

def key_to_indices(key, keys):
    return fetch_first(key == keys)


@jit
def fetch_first(mask, default=I_INT) -> Array:
    """
    fetch the first True index
    :param mask: array of bool
    :param default: the default value if no element satisfying the condition
    :return: the index of the first element satisfying the condition. if no element satisfying the condition, return default value
    """
    idx = jnp.argmax(mask)
    return jnp.where(mask[idx], idx, default)


@jit
def fetch_random(rand_key, mask, default=I_INT) -> Array:
    """
    similar to fetch_first, but fetch a random True index
    """
    true_cnt = jnp.sum(mask)
    cumsum = jnp.cumsum(mask)
    target = jax.random.randint(rand_key, shape=(), minval=1, maxval=true_cnt + 1)
    mask = jnp.where(true_cnt == 0, False, cumsum >= target)
    return fetch_first(mask, default)

@jit
def argmin_with_mask(arr: Array, mask: Array) -> Array:
    masked_arr = jnp.where(mask, arr, jnp.inf)
    min_idx = jnp.argmin(masked_arr)
    return min_idx