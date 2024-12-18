import jax
from jax import vmap, numpy as jnp
import numpy as np

from .gene import BaseGene
from tensorneat.common import fetch_first, I_INF


def unflatten_conns(nodes, conns):
    """
    transform the (C, CL) connections to (N, N), which contains the idx of the connection in conns
    connection length, N means the number of nodes, C means the number of connections
    returns the unflatten connection indices with shape (N, N)
    """
    N = nodes.shape[0]  # max_nodes
    C = conns.shape[0]  # max_conns
    node_keys = nodes[:, 0]
    i_keys, o_keys = conns[:, 0], conns[:, 1]

    def key_to_indices(key, keys):
        return fetch_first(key == keys)

    i_idxs = vmap(key_to_indices, in_axes=(0, None))(i_keys, node_keys)
    o_idxs = vmap(key_to_indices, in_axes=(0, None))(o_keys, node_keys)

    # Is interesting that jax use clip when attach data in array
    # however, it will do nothing when setting values in an array
    # put the index of connections in the unflatten array
    unflatten = (
        jnp.full((N, N), I_INF, dtype=jnp.int32)
        .at[i_idxs, o_idxs]
        .set(jnp.arange(C, dtype=jnp.int32))
    )

    return unflatten


def valid_cnt(nodes_or_conns):
    return jnp.sum(~jnp.isnan(nodes_or_conns[:, 0]))


def extract_gene_attrs(gene: BaseGene, gene_array):
    """
    extract the custom attributes of the gene
    """
    return gene_array[len(gene.fixed_attrs) :]


def set_gene_attrs(gene: BaseGene, gene_array, attrs):
    """
    set the custom attributes of the gene
    """
    return gene_array.at[len(gene.fixed_attrs) :].set(attrs)


def add_node(nodes, fix_attrs, custom_attrs):
    """
    Add a new node to the genome.
    The new node will place at the first NaN row.
    """
    pos = fetch_first(jnp.isnan(nodes[:, 0]))
    return nodes.at[pos].set(jnp.concatenate((fix_attrs, custom_attrs)))


def delete_node_by_pos(nodes, pos):
    """
    Delete a node from the genome.
    Delete the node by its pos in nodes.
    """
    return nodes.at[pos].set(jnp.nan)


def add_conn(conns, fix_attrs, custom_attrs):
    """
    Add a new connection to the genome.
    The new connection will place at the first NaN row.
    """
    pos = fetch_first(jnp.isnan(conns[:, 0]))
    return conns.at[pos].set(jnp.concatenate((fix_attrs, custom_attrs)))


def delete_conn_by_pos(conns, pos):
    """
    Delete a connection from the genome.
    Delete the connection by its idx.
    """
    return conns.at[pos].set(jnp.nan)


def re_cound_idx(nodes, conns, input_idx, output_idx):
    """
    Make the key of hidden nodes continuous.
    Also update the index of connections.
    """
    nodes, conns = jax.device_get((nodes, conns))
    next_key = max(*input_idx, *output_idx) + 1
    old2new = {}
    for i, key in enumerate(nodes[:, 0]):
        if np.isnan(key):
            continue
        if np.in1d(key, input_idx + output_idx):
            continue
        old2new[int(key)] = next_key
        next_key += 1

    new_nodes = nodes.copy()
    for i, key in enumerate(nodes[:, 0]):
        if (not np.isnan(key)) and int(key) in old2new:
            new_nodes[i, 0] = old2new[int(key)]

    new_conns = conns.copy()
    for i, (i_key, o_key) in enumerate(conns[:, :2]):
        if (not np.isnan(i_key)) and int(i_key) in old2new:
            new_conns[i, 0] = old2new[int(i_key)]
        if (not np.isnan(o_key)) and int(o_key) in old2new:
            new_conns[i, 1] = old2new[int(o_key)]
    return new_nodes, new_conns
