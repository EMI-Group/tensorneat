"""
Some graph algorithms implemented in jax.
Only used in feed-forward networks.
"""

import numpy as np
from numpy.typing import NDArray

# from .utils import fetch_first, I_INT
from algorithms.neat.genome.utils import fetch_first, I_INT


def topological_sort(nodes: NDArray, connections: NDArray) -> NDArray:
    """
    a jit-able version of topological_sort! that's crazy!
    :param nodes: nodes array
    :param connections: connections array
    :return: topological sorted sequence

        Example:
        nodes = np.array([
            [0],
            [1],
            [2],
            [3]
        ])
        connections = np.array([
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
    in_degree = np.where(np.isnan(nodes[:, 0]), np.nan, np.sum(connections_enable, axis=0))
    res = np.full(in_degree.shape, I_INT)
    idx = 0

    for _ in range(in_degree.shape[0]):
        i = fetch_first(in_degree == 0.)
        if i == I_INT:
            break
        res[idx] = i
        idx += 1
        in_degree[i] = -1
        children = connections_enable[i, :]
        in_degree = np.where(children, in_degree - 1, in_degree)

    return res


def batch_topological_sort(pop_nodes: NDArray, pop_connections: NDArray) -> NDArray:
    """
    batch version of topological_sort
    :param pop_nodes:
    :param pop_connections:
    :return:
    """
    res = []
    for nodes, connections in zip(pop_nodes, pop_connections):
        seq = topological_sort(nodes, connections)
        res.append(seq)
    return np.stack(res, axis=0)


def check_cycles(nodes: NDArray, connections: NDArray, from_idx: NDArray, to_idx: NDArray) -> NDArray:
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
        nodes = np.array([
            [0],
            [1],
            [2],
            [3]
        ])
        connections = np.array([
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
    connections_enable = ~np.isnan(connections[0, :, :])

    connections_enable[from_idx, to_idx] = True
    nodes_visited = np.full(nodes.shape[0], False)
    nodes_visited[to_idx] = True

    for _ in range(nodes_visited.shape[0]):
        new_visited = np.dot(nodes_visited, connections_enable)
        nodes_visited = np.logical_or(nodes_visited, new_visited)

    return nodes_visited[from_idx]


if __name__ == '__main__':
    nodes = np.array([
        [0],
        [1],
        [2],
        [3],
        [np.nan]
    ])
    connections = np.array([
        [
            [np.nan, np.nan, 1, np.nan, np.nan],
            [np.nan, np.nan, 1, 1, np.nan],
            [np.nan, np.nan, np.nan, 1, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan]
        ],
        [
            [np.nan, np.nan, 1, np.nan, np.nan],
            [np.nan, np.nan, 1, 1, np.nan],
            [np.nan, np.nan, np.nan, 1, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan]
        ]
    ]
    )

    print(topological_sort(nodes, connections))
    print(topological_sort(nodes, connections))

    print(check_cycles(nodes, connections, 3, 2))
    print(check_cycles(nodes, connections, 2, 3))
    print(check_cycles(nodes, connections, 0, 3))
    print(check_cycles(nodes, connections, 1, 0))
