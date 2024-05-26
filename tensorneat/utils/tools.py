from functools import partial

import numpy as np
import jax
from jax import numpy as jnp, Array, jit, vmap

I_INF = np.iinfo(jnp.int32).max  # infinite int


def unflatten_conns(nodes, conns):
    """
    transform the (C, CL) connections to (CL-2, N, N), 2 is for the input index and output index), which CL means
    connection length, N means the number of nodes, C means the number of connections
    returns the un_flattened connections with shape (CL-2, N, N)
    """
    N = nodes.shape[0]
    CL = conns.shape[1]
    node_keys = nodes[:, 0]
    i_keys, o_keys = conns[:, 0], conns[:, 1]
    i_idxs = vmap(key_to_indices, in_axes=(0, None))(i_keys, node_keys)
    o_idxs = vmap(key_to_indices, in_axes=(0, None))(o_keys, node_keys)
    res = jnp.full((CL - 2, N, N), jnp.nan)

    # Is interesting that jax use clip when attach data in array
    # however, it will do nothing set values in an array
    # put all attributes include enable in res
    res = res.at[:, i_idxs, o_idxs].set(conns[:, 2:].T)

    return res


def key_to_indices(key, keys):
    return fetch_first(key == keys)


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


def add_node(nodes, new_key: int, attrs):
    """
    Add a new node to the genome.
    The new node will place at the first NaN row.
    """
    exist_keys = nodes[:, 0]
    pos = fetch_first(jnp.isnan(exist_keys))
    new_nodes = nodes.at[pos, 0].set(new_key)
    return new_nodes.at[pos, 1:].set(attrs)


def delete_node_by_pos(nodes, pos):
    """
    Delete a node from the genome.
    Delete the node by its pos in nodes.
    """
    return nodes.at[pos].set(jnp.nan)


def add_conn(conns, i_key, o_key, enable: bool, attrs):
    """
    Add a new connection to the genome.
    The new connection will place at the first NaN row.
    """
    con_keys = conns[:, 0]
    pos = fetch_first(jnp.isnan(con_keys))
    new_conns = conns.at[pos, 0:3].set(jnp.array([i_key, o_key, enable]))
    return new_conns.at[pos, 3:].set(attrs)


def delete_conn_by_pos(conns, pos):
    """
    Delete a connection from the genome.
    Delete the connection by its idx.
    """
    return conns.at[pos].set(jnp.nan)
