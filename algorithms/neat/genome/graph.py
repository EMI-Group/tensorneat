"""
Some graph algorithms implemented in jax.
Only used in feed-forward networks.
"""

import jax
from jax import jit, vmap, Array
from jax import numpy as jnp

# from .utils import fetch_first, I_INT
from algorithms.neat.genome.utils import fetch_first, I_INT


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
    connections_enable = connections[1, :, :] == 1
    in_degree = jnp.where(jnp.isnan(nodes[:, 0]), jnp.nan, jnp.sum(connections_enable, axis=0))
    res = jnp.full(in_degree.shape, I_INT)
    idx = 0

    def scan_body(carry, _):
        res_, idx_, in_degree_ = carry
        i = fetch_first(in_degree_ == 0.)

        def hit():
            # add to res and flag it is already in it
            new_res = res_.at[idx_].set(i)
            new_idx = idx_ + 1
            new_in_degree = in_degree_.at[i].set(-1)

            # decrease in_degree of all its children
            children = connections_enable[i, :]
            new_in_degree = jnp.where(children, new_in_degree - 1, new_in_degree)
            return new_res, new_idx, new_in_degree

        def miss():
            return res_, idx_, in_degree_

        return jax.lax.cond(i == I_INT, miss, hit), None

    scan_res, _ = jax.lax.scan(scan_body, (res, idx, in_degree), None, length=in_degree.shape[0])
    res, _, _ = scan_res

    return res


# @jit
def topological_sort_debug(nodes: Array, connections: Array) -> Array:
    connections_enable = connections[1, :, :] == 1
    in_degree = jnp.where(jnp.isnan(nodes[:, 0]), jnp.nan, jnp.sum(connections_enable, axis=0))
    res = jnp.full(in_degree.shape, I_INT)
    idx = 0

    for _ in range(in_degree.shape[0]):
        i = fetch_first(in_degree == 0.)
        if i == I_INT:
            break
        res = res.at[idx].set(i)
        idx += 1
        in_degree = in_degree.at[i].set(-1)
        children = connections_enable[i, :]
        in_degree = jnp.where(children, in_degree - 1, in_degree)

    return res


@vmap
def batch_topological_sort(nodes: Array, connections: Array) -> Array:
    """
    batch version of topological_sort
    :param nodes:
    :param connections:
    :return:
    """
    return topological_sort(nodes, connections)


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
    # connections_enable = connections[0, :, :] == 1
    connections_enable = ~jnp.isnan(connections[0, :, :])

    connections_enable = connections_enable.at[from_idx, to_idx].set(True)
    nodes_visited = jnp.full(nodes.shape[0], False)
    nodes_visited = nodes_visited.at[to_idx].set(True)

    def scan_body(visited, _):
        new_visited = jnp.dot(visited, connections_enable)
        new_visited = jnp.logical_or(visited, new_visited)
        return new_visited, None

    nodes_visited, _ = jax.lax.scan(scan_body, nodes_visited, None, length=nodes_visited.shape[0])

    return nodes_visited[from_idx]


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
            [0, 0, 1, 0, jnp.nan],
            [0, 0, 1, 1, jnp.nan],
            [0, 0, 0, 1, jnp.nan],
            [0, 0, 0, 0, jnp.nan],
            [jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan]
        ],
        [
            [0, 0, 1, 0, jnp.nan],
            [0, 0, 1, 1, jnp.nan],
            [0, 0, 0, 1, jnp.nan],
            [0, 0, 0, 0, jnp.nan],
            [jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan]
        ]
    ]
    )

    print(topological_sort_debug(nodes, connections))
    print(topological_sort(nodes, connections))

    print(check_cycles(nodes, connections, 3, 2))
    print(check_cycles(nodes, connections, 2, 3))
    print(check_cycles(nodes, connections, 0, 3))
    print(check_cycles(nodes, connections, 1, 0))
