from functools import partial
from typing import Tuple

import jax
from jax import numpy as jnp, Array
from jax import jit

I_INT = jnp.iinfo(jnp.int32).max  # infinite int
EMPTY_NODE = jnp.full((1, 5), jnp.nan)
EMPTY_CON = jnp.full((1, 4), jnp.nan)

@jit
def flatten_connections(keys, connections):
    """
    flatten the (2, N, N) connections to (N * N, 4)
    :param keys:
    :param connections:
    :return:
    the first two columns are the index of the node
    the 3rd column is the weight, and the 4th column is the enabled status
    """
    indices_x, indices_y = jnp.meshgrid(keys, keys, indexing='ij')
    indices = jnp.stack((indices_x, indices_y), axis=-1).reshape(-1, 2)

    # make (2, N, N) to (N, N, 2)
    con = jnp.transpose(connections, (1, 2, 0))
    # make (N, N, 2) to (N * N, 2)
    con = jnp.reshape(con, (-1, 2))

    con = jnp.concatenate((indices, con), axis=1)
    return con


@partial(jit, static_argnames=['N'])
def unflatten_connections(N, cons):
    """
    restore the (N * N, 4) connections to (2, N, N)
    :param N:
    :param cons:
    :return:
    """
    cons = cons[:, 2:]  # remove the indices
    unflatten_cons = jnp.moveaxis(cons.reshape(N, N, 2), -1, 0)
    return unflatten_cons


@jit
def set_operation_analysis(ar1: Array, ar2: Array) -> Tuple[Array, Array, Array]:
    """
    Analyze the intersection and union of two arrays by returning their sorted concatenation indices,
    intersection mask, and union mask.

    :param ar1: JAX array of shape (N, M)
        First input array. Should have the same shape as ar2.
    :param ar2: JAX array of shape (N, M)
        Second input array. Should have the same shape as ar1.
    :return: tuple of 3 arrays
        - sorted_indices: Indices that would sort the concatenation of ar1 and ar2.
        - intersect_mask: A boolean array indicating the positions of the common elements between ar1 and ar2
                          in the sorted concatenation.
        - union_mask: A boolean array indicating the positions of the unique elements in the union of ar1 and ar2
                      in the sorted concatenation.

    Examples:
        a = jnp.array([[1, 2], [3, 4], [5, 6]])
        b = jnp.array([[1, 2], [7, 8], [9, 10]])

        sorted_indices, intersect_mask, union_mask = set_operation_analysis(a, b)

        sorted_indices -> array([0, 1, 2, 3, 4, 5])
        intersect_mask -> array([True, False, False, False, False, False])
        union_mask -> array([False, True, True, True, True, True])
    """
    ar = jnp.concatenate((ar1, ar2), axis=0)
    sorted_indices = jnp.lexsort(ar.T[::-1])
    aux = ar[sorted_indices]
    aux = jnp.concatenate((aux, jnp.full((1, ar1.shape[1]), jnp.nan)), axis=0)
    nan_mask = jnp.any(jnp.isnan(aux), axis=1)

    fr, sr = aux[:-1], aux[1:]  # first row, second row
    intersect_mask = jnp.all(fr == sr, axis=1) & ~nan_mask[:-1]
    union_mask = jnp.any(fr != sr, axis=1) & ~nan_mask[:-1]
    return sorted_indices, intersect_mask, union_mask


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
