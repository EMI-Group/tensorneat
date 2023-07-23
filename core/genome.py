from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp

from utils.tools import fetch_first


@register_pytree_node_class
class Genome:

    def __init__(self, nodes, conns):
        self.nodes = nodes
        self.conns = conns

    def update(self, nodes, conns):
        return self.__class__(nodes, conns)

    def update_nodes(self, nodes):
        return self.update(nodes, self.conns)

    def update_conns(self, conns):
        return self.update(self.nodes, conns)

    def count(self):
        """Count how many nodes and connections are in the genome."""
        nodes_cnt = jnp.sum(~jnp.isnan(self.nodes[:, 0]))
        conns_cnt = jnp.sum(~jnp.isnan(self.conns[:, 0]))
        return nodes_cnt, conns_cnt

    def add_node(self, new_key: int, attrs):
        """
        Add a new node to the genome.
        The new node will place at the first NaN row.
        """
        exist_keys = self.nodes[:, 0]
        pos = fetch_first(jnp.isnan(exist_keys))
        new_nodes = self.nodes.at[pos, 0].set(new_key)
        new_nodes = new_nodes.at[pos, 1:].set(attrs)
        return self.update_nodes(new_nodes)

    def delete_node_by_pos(self, pos):
        """
        Delete a node from the genome.
        Delete the node by its pos in nodes.
        """
        nodes = self.nodes.at[pos].set(jnp.nan)
        return self.update_nodes(nodes)

    def add_conn(self, i_key, o_key, enable: bool, attrs):
        """
        Add a new connection to the genome.
        The new connection will place at the first NaN row.
        """
        con_keys = self.conns[:, 0]
        pos = fetch_first(jnp.isnan(con_keys))
        new_conns = self.conns.at[pos, 0:3].set(jnp.array([i_key, o_key, enable]))
        new_conns = new_conns.at[pos, 3:].set(attrs)
        return self.update_conns(new_conns)

    def delete_conn_by_pos(self, pos):
        """
        Delete a connection from the genome.
        Delete the connection by its idx.
        """
        conns = self.conns.at[pos].set(jnp.nan)
        return self.update_conns(conns)

    def tree_flatten(self):
        children = self.nodes, self.conns
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self):
        return f"Genome(nodes={self.nodes}, conns={self.conns})"
