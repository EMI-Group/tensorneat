from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, Array, jit, vmap

I_INF = np.iinfo(jnp.int32).max  # infinite int


def attach_with_inf(arr, idx):
    target_dim = arr.ndim + idx.ndim - 1
    expand_idx = jnp.expand_dims(idx, axis=tuple(range(idx.ndim, target_dim)))

    return jnp.where(expand_idx == I_INF, jnp.nan, arr[idx])


@jit
def fetch_first(mask, default=I_INF) -> Array:
    """
    fetch the first True index
    :param mask: array of bool
    :param default: the default value if no element satisfying the condition
    :return: the index of the first element satisfying the condition. if no element satisfying the condition, return default value
    """
    idx = jnp.argmax(mask)
    return jnp.where(mask[idx], idx, default)


@jit
def fetch_random(randkey, mask, default=I_INF) -> Array:
    """
    similar to fetch_first, but fetch a random True index
    """
    true_cnt = jnp.sum(mask)
    cumsum = jnp.cumsum(mask)
    target = jax.random.randint(randkey, shape=(), minval=1, maxval=true_cnt + 1)
    mask = jnp.where(true_cnt == 0, False, cumsum >= target)
    return fetch_first(mask, default)


@partial(jit, static_argnames=["reverse"])
def rank_elements(array, reverse=False):
    """
    rank the element in the array.
    if reverse is True, the rank is from small to large. default large to small
    """
    if not reverse:
        array = -array
    return jnp.argsort(jnp.argsort(array))


@jit
def mutate_float(
    randkey, val, init_mean, init_std, mutate_power, mutate_rate, replace_rate
):
    """
    mutate a float value
    uniformly pick r from [0, 1]
    r in [0, mutate_rate) -> add noise
    r in [mutate_rate, mutate_rate + replace_rate) -> create a new value to replace the original value
    otherwise -> keep the original value
    """
    k1, k2, k3 = jax.random.split(randkey, num=3)
    noise = jax.random.normal(k1, ()) * mutate_power
    replace = jax.random.normal(k2, ()) * init_std + init_mean
    r = jax.random.uniform(k3, ())

    val = jnp.where(
        r < mutate_rate,
        val + noise,
        jnp.where((mutate_rate < r) & (r < mutate_rate + replace_rate), replace, val),
    )

    return val


@jit
def mutate_int(randkey, val, options, replace_rate):
    """
    mutate an int value
    uniformly pick r from [0, 1]
    r in [0, replace_rate) -> create a new value to replace the original value
    otherwise -> keep the original value
    """
    k1, k2 = jax.random.split(randkey, num=2)
    r = jax.random.uniform(k1, ())

    val = jnp.where(r < replace_rate, jax.random.choice(k2, options), val)

    return val


def argmin_with_mask(arr, mask):
    """
    find the index of the minimum element in the array, but only consider the element with True mask
    """
    masked_arr = jnp.where(mask, arr, jnp.inf)
    min_idx = jnp.argmin(masked_arr)
    return min_idx


def hash_array(arr: Array):
    arr = jax.lax.bitcast_convert_type(arr, jnp.uint32)

    def update(i, hash_val):
        return hash_val ^ (
            arr[i] + jnp.uint32(0x9E3779B9) + (hash_val << 6) + (hash_val >> 2)
        )

    return jax.lax.fori_loop(0, arr.size, update, jnp.uint32(0))
