from typing import Type, Tuple

import numpy as np
import jax
from jax import Array, numpy as jnp

from algorithm import State
from ..gene import BaseGene
from algorithm.utils import fetch_first


def initialize_genomes(state: State, gene_type: Type[BaseGene]):
    o_nodes = np.full((state.N, state.NL), np.nan, dtype=np.float32)  # original nodes
    o_conns = np.full((state.C, state.CL), np.nan, dtype=np.float32)  # original connections

    input_idx = state.input_idx
    output_idx = state.output_idx
    new_node_key = max([*input_idx, *output_idx]) + 1

    o_nodes[input_idx, 0] = input_idx
    o_nodes[output_idx, 0] = output_idx
    o_nodes[new_node_key, 0] = new_node_key
    o_nodes[np.concatenate([input_idx, output_idx]), 1:] = jax.device_get(gene_type.new_node_attrs(state))
    o_nodes[new_node_key, 1:] = jax.device_get(gene_type.new_node_attrs(state))

    input_conns = np.c_[input_idx, np.full_like(input_idx, new_node_key)]
    o_conns[input_idx, 0:2] = input_conns  # in key, out key
    o_conns[input_idx, 2] = True  # enabled
    o_conns[input_idx, 3:] = jax.device_get(gene_type.new_conn_attrs(state))

    output_conns = np.c_[np.full_like(output_idx, new_node_key), output_idx]
    o_conns[output_idx, 0:2] = output_conns  # in key, out key
    o_conns[output_idx, 2] = True  # enabled
    o_conns[output_idx, 3:] = jax.device_get(gene_type.new_conn_attrs(state))

    # repeat origin genome for P times to create population
    pop_nodes = np.tile(o_nodes, (state.P, 1, 1))
    pop_conns = np.tile(o_conns, (state.P, 1, 1))

    return jax.device_put([pop_nodes, pop_conns])


def count(nodes: Array, cons: Array):
    """
    Count how many nodes and connections are in the genome.
    """
    node_cnt = jnp.sum(~jnp.isnan(nodes[:, 0]))
    cons_cnt = jnp.sum(~jnp.isnan(cons[:, 0]))
    return node_cnt, cons_cnt


def add_node(nodes: Array, cons: Array, new_key: int, attrs: Array) -> Tuple[Array, Array]:
    """
    Add a new node to the genome.
    The new node will place at the first NaN row.
    """
    exist_keys = nodes[:, 0]
    idx = fetch_first(jnp.isnan(exist_keys))
    nodes = nodes.at[idx, 0].set(new_key)
    nodes = nodes.at[idx, 1:].set(attrs)
    return nodes, cons


def delete_node(nodes: Array, cons: Array, node_key: Array) -> Tuple[Array, Array]:
    """
    Delete a node from the genome. Only delete the node, regardless of connections.
    Delete the node by its key.
    """
    node_keys = nodes[:, 0]
    idx = fetch_first(node_keys == node_key)
    return delete_node_by_idx(nodes, cons, idx)


def delete_node_by_idx(nodes: Array, cons: Array, idx: Array) -> Tuple[Array, Array]:
    """
    Delete a node from the genome. Only delete the node, regardless of connections.
    Delete the node by its idx.
    """
    nodes = nodes.at[idx].set(np.nan)
    return nodes, cons


def add_connection(nodes: Array, cons: Array, i_key: Array, o_key: Array, enable: bool, attrs: Array) -> Tuple[
    Array, Array]:
    """
    Add a new connection to the genome.
    The new connection will place at the first NaN row.
    """
    con_keys = cons[:, 0]
    idx = fetch_first(jnp.isnan(con_keys))
    cons = cons.at[idx, 0:3].set(jnp.array([i_key, o_key, enable]))
    cons = cons.at[idx, 3:].set(attrs)
    return nodes, cons


def delete_connection(nodes: Array, cons: Array, i_key: Array, o_key: Array) -> Tuple[Array, Array]:
    """
    Delete a connection from the genome.
    Delete the connection by its input and output node keys.
    """
    idx = fetch_first((cons[:, 0] == i_key) & (cons[:, 1] == o_key))
    return delete_connection_by_idx(nodes, cons, idx)


def delete_connection_by_idx(nodes: Array, cons: Array, idx: Array) -> Tuple[Array, Array]:
    """
    Delete a connection from the genome.
    Delete the connection by its idx.
    """
    cons = cons.at[idx].set(np.nan)
    return nodes, cons
