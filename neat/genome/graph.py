"""
Some graph algorithms implemented in jax.
Only used in feed-forward networks.
"""

import jax
from jax import jit, vmap, Array
from jax import numpy as jnp

# from .configs import fetch_first, I_INT
from neat.genome.utils import fetch_first, I_INT, unflatten_connections


@jit
def topological_sort(nodes: Array, connections: Array) -> Array:
    """
    a jit-able version of topological_sort! that's crazy!
    :param nodes: nodes array
    :param connections: connections array
    :return: topological sorted sequence

        Example:
        nodes = jnp.array([
            [0],
            [1],
            [2],
            [3]
        ])
        connections = jnp.array([
            [
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0]
            ]
        ])

        topological_sort(nodes, connections) -> [0, 1, 2, 3]
    """
    connections_enable = connections[1, :, :] == 1  # forward function. thus use enable
    in_degree = jnp.where(jnp.isnan(nodes[:, 0]), jnp.nan, jnp.sum(connections_enable, axis=0))
    res = jnp.full(in_degree.shape, I_INT)

    def cond_fun(carry):
        res_, idx_, in_degree_ = carry
        i = fetch_first(in_degree_ == 0.)
        return i != I_INT

    def body_func(carry):
        res_, idx_, in_degree_ = carry
        i = fetch_first(in_degree_ == 0.)

        # add to res and flag it is already in it
        res_ = res_.at[idx_].set(i)
        in_degree_ = in_degree_.at[i].set(-1)

        # decrease in_degree of all its children
        children = connections_enable[i, :]
        in_degree_ = jnp.where(children, in_degree_ - 1, in_degree_)
        return res_, idx_ + 1, in_degree_

    res, _, _ = jax.lax.while_loop(cond_fun, body_func, (res, 0, in_degree))
    return res


@jit
def check_cycles(nodes: Array, connections: Array, from_idx: Array, to_idx: Array) -> Array:
    """
    Check whether a new connection (from_idx -> to_idx) will cause a cycle.

    :param nodes: JAX array
        The array of nodes.
    :param connections: JAX array
        The array of connections.
    :param from_idx: int
        The index of the starting node.
    :param to_idx: int
        The index of the ending node.
    :return: JAX array
        An array indicating if there is a cycle caused by the new connection.

    Example:
        nodes = jnp.array([
            [0],
            [1],
            [2],
            [3]
        ])
        connections = jnp.array([
            [
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0]
            ]
        ])

        check_cycles(nodes, connections, 3, 2) -> True
        check_cycles(nodes, connections, 2, 3) -> False
        check_cycles(nodes, connections, 0, 3) -> False
        check_cycles(nodes, connections, 1, 0) -> False
    """

    connections_enable = ~jnp.isnan(connections[0, :, :])
    connections_enable = connections_enable.at[from_idx, to_idx].set(True)

    visited = jnp.full(nodes.shape[0], False)
    new_visited = visited.at[to_idx].set(True)

    def cond_func(carry):
        visited_, new_visited_ = carry
        end_cond1 = jnp.all(visited_ == new_visited_)  # no new nodes been visited
        end_cond2 = new_visited_[from_idx]  # the starting node has been visited
        return jnp.logical_not(end_cond1 | end_cond2)

    def body_func(carry):
        _, visited_ = carry
        new_visited_ = jnp.dot(visited_, connections_enable)
        new_visited_ = jnp.logical_or(visited_, new_visited_)
        return visited_, new_visited_

    _, visited = jax.lax.while_loop(cond_func, body_func, (visited, new_visited))
    return visited[from_idx]


if __name__ == '__main__':
    nodes = jnp.array([
        [0],
        [1],
        [2],
        [3],
        [jnp.nan]
    ])
    connections = jnp.array([
        [
            [jnp.nan, jnp.nan, 1, jnp.nan, jnp.nan],
            [jnp.nan, jnp.nan, 1, 1, jnp.nan],
            [jnp.nan, jnp.nan, jnp.nan, 1, jnp.nan],
            [jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan],
            [jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan]
        ],
        [
            [jnp.nan, jnp.nan, 1, jnp.nan, jnp.nan],
            [jnp.nan, jnp.nan, 1, 1, jnp.nan],
            [jnp.nan, jnp.nan, jnp.nan, 1, jnp.nan],
            [jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan],
            [jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan]
        ]
    ]
    )

    print(topological_sort(nodes, connections))

    print(check_cycles(nodes, connections, 3, 2))
    print(check_cycles(nodes, connections, 2, 3))
    print(check_cycles(nodes, connections, 0, 3))
    print(check_cycles(nodes, connections, 1, 0))
