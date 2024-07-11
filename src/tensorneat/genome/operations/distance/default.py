from jax import vmap, numpy as jnp

from .base import BaseDistance
from ...utils import extract_node_attrs, extract_conn_attrs


class DefaultDistance(BaseDistance):
    def __init__(
        self,
        compatibility_disjoint: float = 1.0,
        compatibility_weight: float = 0.4,
    ):
        self.compatibility_disjoint = compatibility_disjoint
        self.compatibility_weight = compatibility_weight

    def __call__(self, state, nodes1, conns1, nodes2, conns2):
        """
        The distance between two genomes
        """
        d = self.node_distance(state, nodes1, nodes2) + self.conn_distance(
            state, conns1, conns2
        )
        return d

    def node_distance(self, state, nodes1, nodes2):
        """
        The distance of the nodes part for two genomes
        """
        node_cnt1 = jnp.sum(~jnp.isnan(nodes1[:, 0]))
        node_cnt2 = jnp.sum(~jnp.isnan(nodes2[:, 0]))
        max_cnt = jnp.maximum(node_cnt1, node_cnt2)

        # align homologous nodes
        # this process is similar to np.intersect1d.
        nodes = jnp.concatenate((nodes1, nodes2), axis=0)
        keys = nodes[:, 0]
        sorted_indices = jnp.argsort(keys, axis=0)
        nodes = nodes[sorted_indices]
        nodes = jnp.concatenate(
            [nodes, jnp.full((1, nodes.shape[1]), jnp.nan)], axis=0
        )  # add a nan row to the end
        fr, sr = nodes[:-1], nodes[1:]  # first row, second row

        # flag location of homologous nodes
        intersect_mask = (fr[:, 0] == sr[:, 0]) & ~jnp.isnan(nodes[:-1, 0])

        # calculate the count of non_homologous of two genomes
        non_homologous_cnt = node_cnt1 + node_cnt2 - 2 * jnp.sum(intersect_mask)

        # calculate the distance of homologous nodes
        fr_attrs = vmap(extract_node_attrs)(fr)
        sr_attrs = vmap(extract_node_attrs)(sr)
        hnd = vmap(self.genome.node_gene.distance, in_axes=(None, 0, 0))(
            state, fr_attrs, sr_attrs
        )  # homologous node distance
        hnd = jnp.where(jnp.isnan(hnd), 0, hnd)
        homologous_distance = jnp.sum(hnd * intersect_mask)

        val = (
            non_homologous_cnt * self.compatibility_disjoint
            + homologous_distance * self.compatibility_weight
        )

        val = jnp.where(max_cnt == 0, 0, val / max_cnt)  # normalize

        return val

    def conn_distance(self, state, conns1, conns2):
        """
        The distance of the conns part for two genomes
        """
        con_cnt1 = jnp.sum(~jnp.isnan(conns1[:, 0]))
        con_cnt2 = jnp.sum(~jnp.isnan(conns2[:, 0]))
        max_cnt = jnp.maximum(con_cnt1, con_cnt2)

        cons = jnp.concatenate((conns1, conns2), axis=0)
        keys = cons[:, :2]
        sorted_indices = jnp.lexsort(keys.T[::-1])
        cons = cons[sorted_indices]
        cons = jnp.concatenate(
            [cons, jnp.full((1, cons.shape[1]), jnp.nan)], axis=0
        )  # add a nan row to the end
        fr, sr = cons[:-1], cons[1:]  # first row, second row

        # both genome has such connection
        intersect_mask = jnp.all(fr[:, :2] == sr[:, :2], axis=1) & ~jnp.isnan(fr[:, 0])

        non_homologous_cnt = con_cnt1 + con_cnt2 - 2 * jnp.sum(intersect_mask)

        fr_attrs = vmap(extract_conn_attrs)(fr)
        sr_attrs = vmap(extract_conn_attrs)(sr)
        hcd = vmap(self.genome.conn_gene.distance, in_axes=(None, 0, 0))(
            state, fr_attrs, sr_attrs
        )  # homologous connection distance
        hcd = jnp.where(jnp.isnan(hcd), 0, hcd)
        homologous_distance = jnp.sum(hcd * intersect_mask)

        val = (
            non_homologous_cnt * self.compatibility_disjoint
            + homologous_distance * self.compatibility_weight
        )

        val = jnp.where(max_cnt == 0, 0, val / max_cnt)  # normalize

        return val
