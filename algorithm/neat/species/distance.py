from typing import Type

from jax import Array, numpy as jnp, vmap

from core import Gene


def create_distance(gene_type: Type[Gene]):
    def node_distance(state, nodes1: Array, nodes2: Array):
        """
        Calculate the distance between nodes of two genomes.
        """
        # statistics nodes count of two genomes
        node_cnt1 = jnp.sum(~jnp.isnan(nodes1[:, 0]))
        node_cnt2 = jnp.sum(~jnp.isnan(nodes2[:, 0]))
        max_cnt = jnp.maximum(node_cnt1, node_cnt2)

        # align homologous nodes
        # this process is similar to np.intersect1d.
        nodes = jnp.concatenate((nodes1, nodes2), axis=0)
        keys = nodes[:, 0]
        sorted_indices = jnp.argsort(keys, axis=0)
        nodes = nodes[sorted_indices]
        nodes = jnp.concatenate([nodes, jnp.full((1, nodes.shape[1]), jnp.nan)], axis=0)  # add a nan row to the end
        fr, sr = nodes[:-1], nodes[1:]  # first row, second row

        # flag location of homologous nodes
        intersect_mask = (fr[:, 0] == sr[:, 0]) & ~jnp.isnan(nodes[:-1, 0])

        # calculate the count of non_homologous of two genomes
        non_homologous_cnt = node_cnt1 + node_cnt2 - 2 * jnp.sum(intersect_mask)

        # calculate the distance of homologous nodes
        hnd = vmap(gene_type.distance_node, in_axes=(None, 0, 0))(state, fr, sr)
        hnd = jnp.where(jnp.isnan(hnd), 0, hnd)
        homologous_distance = jnp.sum(hnd * intersect_mask)

        val = non_homologous_cnt * state.compatibility_disjoint + homologous_distance * state.compatibility_weight

        return jnp.where(max_cnt == 0, 0, val / max_cnt)  # avoid zero division

    def connection_distance(state, cons1: Array, cons2: Array):
        """
        Calculate the distance between connections of two genomes.
        Similar process as node_distance.
        """
        con_cnt1 = jnp.sum(~jnp.isnan(cons1[:, 0]))
        con_cnt2 = jnp.sum(~jnp.isnan(cons2[:, 0]))
        max_cnt = jnp.maximum(con_cnt1, con_cnt2)

        cons = jnp.concatenate((cons1, cons2), axis=0)
        keys = cons[:, :2]
        sorted_indices = jnp.lexsort(keys.T[::-1])
        cons = cons[sorted_indices]
        cons = jnp.concatenate([cons, jnp.full((1, cons.shape[1]), jnp.nan)], axis=0)  # add a nan row to the end
        fr, sr = cons[:-1], cons[1:]  # first row, second row

        # both genome has such connection
        intersect_mask = jnp.all(fr[:, :2] == sr[:, :2], axis=1) & ~jnp.isnan(fr[:, 0])

        non_homologous_cnt = con_cnt1 + con_cnt2 - 2 * jnp.sum(intersect_mask)
        hcd = vmap(gene_type.distance_conn, in_axes=(None, 0, 0))(state, fr, sr)
        hcd = jnp.where(jnp.isnan(hcd), 0, hcd)
        homologous_distance = jnp.sum(hcd * intersect_mask)

        val = non_homologous_cnt * state.compatibility_disjoint + homologous_distance * state.compatibility_weight

        return jnp.where(max_cnt == 0, 0, val / max_cnt)

    def distance(state, genome1, genome2):
        return node_distance(state, genome1.nodes, genome2.nodes) + connection_distance(state, genome1.conns, genome2.conns)

    return distance
