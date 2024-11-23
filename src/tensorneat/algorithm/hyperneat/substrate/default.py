from jax import vmap
import numpy as np

from .base import BaseSubstrate
from tensorneat.genome.utils import set_conn_attrs


class DefaultSubstrate(BaseSubstrate):

    connection_type = "recurrent"

    def __init__(self, num_inputs, num_outputs, coors, nodes, conns):
        self.inputs = num_inputs
        self.outputs = num_outputs
        self.coors = np.array(coors)
        self.nodes = np.array(nodes)
        self.conns = np.array(conns)

    def make_nodes(self, query_res):
        return self.nodes

    def make_conns(self, query_res):
        # change weight of conns
        return vmap(set_conn_attrs)(self.conns, query_res)

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
