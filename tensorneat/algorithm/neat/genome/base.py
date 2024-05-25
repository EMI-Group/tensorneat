import jax.numpy as jnp
from ..gene import BaseNodeGene, BaseConnGene, DefaultNodeGene, DefaultConnGene
from utils import fetch_first, State


class BaseGenome:
    network_type = None

    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            max_nodes: int,
            max_conns: int,
            node_gene: BaseNodeGene = DefaultNodeGene(),
            conn_gene: BaseConnGene = DefaultConnGene(),
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_idx = jnp.arange(num_inputs)
        self.output_idx = jnp.arange(num_inputs, num_inputs + num_outputs)
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self.node_gene = node_gene
        self.conn_gene = conn_gene

    def setup(self, state=State()):
        return state

    def transform(self, state, nodes, conns):
        raise NotImplementedError

    def forward(self, state, inputs, transformed):
        raise NotImplementedError

    def add_node(self, nodes, new_key: int, attrs):
        """
        Add a new node to the genome.
        The new node will place at the first NaN row.
        """
        exist_keys = nodes[:, 0]
        pos = fetch_first(jnp.isnan(exist_keys))
        new_nodes = nodes.at[pos, 0].set(new_key)
        return new_nodes.at[pos, 1:].set(attrs)

    def delete_node_by_pos(self, nodes, pos):
        """
        Delete a node from the genome.
        Delete the node by its pos in nodes.
        """
        return nodes.at[pos].set(jnp.nan)

    def add_conn(self, conns, i_key, o_key, enable: bool, attrs):
        """
        Add a new connection to the genome.
        The new connection will place at the first NaN row.
        """
        con_keys = conns[:, 0]
        pos = fetch_first(jnp.isnan(con_keys))
        new_conns = conns.at[pos, 0:3].set(jnp.array([i_key, o_key, enable]))
        return new_conns.at[pos, 3:].set(attrs)

    def delete_conn_by_pos(self, conns, pos):
        """
        Delete a connection from the genome.
        Delete the connection by its idx.
        """
        return conns.at[pos].set(jnp.nan)
