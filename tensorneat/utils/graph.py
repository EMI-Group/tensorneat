"""
Some graph algorithm implemented in jax.
Only used in feed-forward networks.
"""

import jax
from jax import jit, Array, numpy as jnp

from .tools import fetch_first, I_INF


@jit
def topological_sort(nodes: Array, conns: Array) -> Array:
    """
    a jit-able version of topological_sort!
    conns: Array[N, N]
    """

    in_degree = jnp.where(jnp.isnan(nodes[:, 0]), jnp.nan, jnp.sum(conns, axis=0))
    res = jnp.full(in_degree.shape, I_INF)

    def cond_fun(carry):
        res_, idx_, in_degree_ = carry
        i = fetch_first(in_degree_ == 0.0)
        return i != I_INF

    def body_func(carry):
        res_, idx_, in_degree_ = carry
        i = fetch_first(in_degree_ == 0.0)

        # add to res and flag it is already in it
        res_ = res_.at[idx_].set(i)
        in_degree_ = in_degree_.at[i].set(-1)

        # decrease in_degree of all its children
        children = conns[i, :]
        in_degree_ = jnp.where(children, in_degree_ - 1, in_degree_)
        return res_, idx_ + 1, in_degree_

    res, _, _ = jax.lax.while_loop(cond_fun, body_func, (res, 0, in_degree))
    return res


@jit
def check_cycles(nodes: Array, conns: Array, from_idx, to_idx) -> Array:
    """
    Check whether a new connection (from_idx -> to_idx) will cause a cycle.
    """

    conns = conns.at[from_idx, to_idx].set(True)

    visited = jnp.full(nodes.shape[0], False)
    new_visited = visited.at[to_idx].set(True)

    def cond_func(carry):
        visited_, new_visited_ = carry
        end_cond1 = jnp.all(visited_ == new_visited_)  # no new nodes been visited
        end_cond2 = new_visited_[from_idx]  # the starting node has been visited
        return jnp.logical_not(end_cond1 | end_cond2)

    def body_func(carry):
        _, visited_ = carry
        new_visited_ = jnp.dot(visited_, conns)
        new_visited_ = jnp.logical_or(visited_, new_visited_)
        return visited_, new_visited_

    _, visited = jax.lax.while_loop(cond_func, body_func, (visited, new_visited))
    return visited[from_idx]
