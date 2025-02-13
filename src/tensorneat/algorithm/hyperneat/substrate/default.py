from jax import vmap, numpy as jnp
import numpy as np

from .base import BaseSubstrate
from tensorneat.genome.utils import set_gene_attrs


class DefaultSubstrate(BaseSubstrate):

    connection_type = "recurrent"

    def __init__(self, num_inputs, num_outputs, coors, nodes, conns):
        self.inputs = num_inputs
        self.outputs = num_outputs
        self.coors = jnp.array(coors)
        self.nodes = jnp.array(nodes)
        self.conns = jnp.array(conns)

    def make_nodes(self, query_res):
        return self.nodes

    def make_conns(self, query_res):
        # change weight of conns
        # the last column is the weight
        # print(f"{self.conns.shape=}, {query_res.shape=}")
        return self.conns.at[:, -1].set(query_res.flatten())

    @property
    def query_coors(self):
        return self.coors

    @property
    def num_inputs(self):
        return self.inputs

    @property
    def num_outputs(self):
        return self.outputs

    @property
    def nodes_cnt(self):
        return self.nodes.shape[0]

    @property
    def conns_cnt(self):
        return self.conns.shape[0]
