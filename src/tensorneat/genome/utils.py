import jax
from jax import vmap, numpy as jnp
import numpy as np

from .gene import BaseGene
from tensorneat.common import fetch_first, I_INF

_FLOAT_MAX = jnp.finfo(jnp.float32).max


def _batch_key_to_index(query_keys, ref_keys):
    """
    Map each scalar query key to its index in ref_keys via argsort + searchsorted.
    O(C log N) time, O(C + N) memory — replaces the old vmap linear scan which was
    O(C * N) and caused OOM at large max_conns.
    Returns I_INF for NaN queries or unmatched keys.

    See: https://github.com/EMI-Group/tensorneat/issues/38
    """
    N = ref_keys.shape[0]

    nan_ref = jnp.isnan(ref_keys)
    nan_query = jnp.isnan(query_keys)
    safe_ref = jnp.where(nan_ref, _FLOAT_MAX, ref_keys)
    safe_query = jnp.where(nan_query, _FLOAT_MAX - 1, query_keys)

    sort_idx = jnp.argsort(safe_ref)
    sorted_ref = safe_ref[sort_idx]

    pos = jnp.searchsorted(sorted_ref, safe_query, side="left")
    pos = jnp.clip(pos, 0, N - 1)

    original_idx = sort_idx[pos]
    matched = (ref_keys[original_idx] == query_keys) & ~nan_query

    return jnp.where(matched, original_idx, I_INF)


def map_conn_endpoints(nodes, conns):
    """
    Resolve flat connection endpoint keys to node-row indices.

    Returns ``(src_rows, dst_rows, valid)`` arrays with shape ``(C,)``.
    ``valid`` is false for padded connections and dangling endpoints.
    """
    node_keys = nodes[:, 0]
    src_rows = _batch_key_to_index(conns[:, 0], node_keys)
    dst_rows = _batch_key_to_index(conns[:, 1], node_keys)
    valid = (
        ~jnp.isnan(conns[:, 0])
        & ~jnp.isnan(conns[:, 1])
        & (src_rows != I_INF)
        & (dst_rows != I_INF)
    )
    return src_rows, dst_rows, valid


def build_padded_inbound_adj(
    src_rows, dst_rows, valid, max_nodes, max_in_degree
):
    """
    Pack flat connections into a bounded inbound adjacency.

    The returned ``adj_conns`` has shape ``(max_nodes, max_in_degree)`` and
    stores row indices into the original connection array. Empty slots contain
    ``I_INF``. Slots are sorted by source-node row to preserve the reduction
    order of the legacy dense representation.

    ``overflow`` is true when an admissible destination has more inbound
    connections than the configured cap.
    """
    max_conns = src_rows.shape[0]
    conn_rows = jnp.arange(max_conns, dtype=jnp.int32)
    admissible = (
        valid
        & (src_rows >= 0)
        & (src_rows < max_nodes)
        & (dst_rows >= 0)
        & (dst_rows < max_nodes)
    )

    # Pack in the source-row order used by the legacy dense representation.
    # Invalid connections sort to the tail and are subsequently dropped.
    safe_src = jnp.where(admissible, src_rows, max_nodes)
    safe_dst = jnp.where(admissible, dst_rows, max_nodes)
    order = jnp.lexsort((conn_rows, safe_src, safe_dst))
    sorted_conn_rows = conn_rows[order]
    sorted_dst = dst_rows[order]
    sorted_admissible = admissible[order]

    def assign_slot(counts, item):
        dst, is_admissible = item
        safe_dst = jnp.where(is_admissible, dst, 0)
        slot = counts[safe_dst]
        counts = counts.at[safe_dst].add(is_admissible.astype(jnp.int32))
        return counts, slot

    _, slots = jax.lax.scan(
        assign_slot,
        jnp.zeros((max_nodes,), dtype=jnp.int32),
        (sorted_dst, sorted_admissible),
    )

    within_cap = sorted_admissible & (slots < max_in_degree)
    overflow = jnp.any(sorted_admissible & ~within_cap)
    safe_dst = jnp.where(within_cap, sorted_dst, max_nodes)

    adj_conns = (
        jnp.full((max_nodes, max_in_degree), I_INF, dtype=jnp.int32)
        .at[safe_dst, slots]
        .set(sorted_conn_rows, mode="drop")
    )
    return adj_conns, overflow


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

    i_idxs = _batch_key_to_index(i_keys, node_keys)
    o_idxs = _batch_key_to_index(o_keys, node_keys)

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
        if np.isin(key, input_idx + output_idx):
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
