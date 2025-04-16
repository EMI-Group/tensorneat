"""
Some graph algorithm implemented in jax and python.
"""

import jax
from jax import jit, Array, numpy as jnp
from typing import Tuple, Set, List, Union

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


def topological_sort_python(
    nodes: Union[Set[int], List[int]],
    conns: Union[Set[Tuple[int, int]], List[Tuple[int, int]]],
) -> Tuple[List[int], List[List[int]]]:
    # a python version of topological_sort, use python set to store nodes and conns
    # returns the topological order of the nodes and the topological layers
    # written by gpt4 :)

    # Make a copy of the input nodes and connections
    nodes = nodes.copy()
    conns = conns.copy()

    # Initialize the in-degree of each node to 0
    in_degree = {node: 0 for node in nodes}

    # Compute the in-degree for each node
    for conn in conns:
        in_degree[conn[1]] += 1

    topo_order = []
    topo_layer = []

    # Find all nodes with in-degree 0
    zero_in_degree_nodes = [node for node in nodes if in_degree[node] == 0]

    while zero_in_degree_nodes:

        for node in zero_in_degree_nodes:
            nodes.remove(node)

        zero_in_degree_nodes = sorted(
            zero_in_degree_nodes
        )  # make sure the topo_order is from small to large

        topo_layer.append(zero_in_degree_nodes.copy())

        for node in zero_in_degree_nodes:
            topo_order.append(node)

            # Iterate over all connections and reduce the in-degree of connected nodes
            for conn in list(conns):
                if conn[0] == node:
                    in_degree[conn[1]] -= 1
                    conns.remove(conn)

        zero_in_degree_nodes = [node for node in nodes if in_degree[node] == 0]

    # Check if there are still connections left indicating a cycle
    if conns or nodes:
        raise ValueError("Graph has at least one cycle, topological sort not possible")

    return topo_order, topo_layer

def find_useful_nodes(
    nodes: Union[Set[int], List[int]],
    conns: Union[Set[Tuple[int, int]], List[Tuple[int, int]]],
    output_idx: Set[int],
) -> Set[int]:
    """
    Find all useful nodes (really contribute to outputs)
    """
    useful_nodes = set()
    useful_nodes = useful_nodes | output_idx
    while True:
        aux = set()
        for in_, out in conns:
            if out in useful_nodes and in_ not in useful_nodes:
                aux.add(in_)
        if len(aux) == 0:  # no new nodes
            break
        else:
            useful_nodes = useful_nodes | aux
    # print(f"All nodes cnt={len(nodes)}, useful nodes cnt={len(useful_nodes)}")
    return useful_nodes            
        
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
