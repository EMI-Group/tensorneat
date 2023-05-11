from functools import partial
from typing import Tuple

import jax
from jax import numpy as jnp, Array
from jax import jit, vmap

I_INT = jnp.iinfo(jnp.int32).max  # infinite int
EMPTY_NODE = jnp.full((1, 5), jnp.nan)
EMPTY_CON = jnp.full((1, 4), jnp.nan)


@jit
def unflatten_connections(nodes, cons):
    """
    transform the (C, 4) connections to (2, N, N)
    :param cons:
    :param nodes:
    :return:
    """
    N = nodes.shape[0]
    node_keys = nodes[:, 0]
    i_keys, o_keys = cons[:, 0], cons[:, 1]
    i_idxs = key_to_indices(i_keys, node_keys)
    o_idxs = key_to_indices(o_keys, node_keys)
    res = jnp.full((2, N, N), jnp.nan)

    # Is interesting that jax use clip when attach data in array
    # however, it will do nothing set values in an array
    res = res.at[0, i_idxs, o_idxs].set(cons[:, 2])
    res = res.at[1, i_idxs, o_idxs].set(cons[:, 3])
    return res


@partial(vmap, in_axes=(0, None))
def key_to_indices(key, keys):
    return fetch_first(key == keys)


@jit
def fetch_first(mask, default=I_INT) -> Array:
    """
    fetch the first True index
    :param mask: array of bool
    :param default: the default value if no element satisfying the condition
    :return: the index of the first element satisfying the condition. if no element satisfying the condition, return I_INT
    example:
    >>> a = jnp.array([1, 2, 3, 4, 5])
    >>> fetch_first(a > 3)
    3
    >>> fetch_first(a > 30)
    I_INT
    """
    idx = jnp.argmax(mask)
    return jnp.where(mask[idx], idx, default)


@jit
def fetch_last(mask, default=I_INT) -> Array:
    """
    similar to fetch_first, but fetch the last True index
    """
    reversed_idx = fetch_first(mask[::-1], default)
    return jnp.where(reversed_idx == -1, -1, mask.shape[0] - reversed_idx - 1)


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


if __name__ == '__main__':

    a = jnp.array([1, 2, 3, 4, 5])
    print(fetch_first(a > 3))
    print(fetch_first(a > 30))

    print(fetch_last(a > 3))
    print(fetch_last(a > 30))

    rand_key = jax.random.PRNGKey(0)

    for t in [-1, 0, 1, 2, 3, 4, 5]:
        for _ in range(10):
            rand_key, _ = jax.random.split(rand_key)
            print(jax.random.randint(rand_key, shape=(), minval=1, maxval=2))
            print(t, fetch_random(rand_key, a > t))
