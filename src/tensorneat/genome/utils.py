import jax
from jax import vmap, numpy as jnp

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


def extract_node_attrs(node):
    """
    node: Array(NL, )
    extract the attributes of a node
    """
    return node[1:]  # 0 is for idx


def set_node_attrs(node, attrs):
    """
    node: Array(NL, )
    attrs: Array(NL-1, )
    set the attributes of a node
    """
    return node.at[1:].set(attrs)  # 0 is for idx


def extract_conn_attrs(conn):
    """
    conn: Array(CL, )
    extract the attributes of a connection
    """
    return conn[2:]  # 0, 1 is for in-idx and out-idx


def set_conn_attrs(conn, attrs):
    """
    conn: Array(CL, )
    attrs: Array(CL-2, )
    set the attributes of a connection
    """
    return conn.at[2:].set(attrs)  # 0, 1 is for in-idx and out-idx


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


def add_conn(conns, i_key, o_key, attrs):
    """
    Add a new connection to the genome.
    The new connection will place at the first NaN row.
    """
    con_keys = conns[:, 0]
    pos = fetch_first(jnp.isnan(con_keys))
    new_conns = conns.at[pos, 0:2].set(jnp.array([i_key, o_key]))
    return new_conns.at[pos, 2:].set(attrs)


def delete_conn_by_pos(conns, pos):
    """
    Delete a connection from the genome.
    Delete the connection by its idx.
    """
    return conns.at[pos].set(jnp.nan)
