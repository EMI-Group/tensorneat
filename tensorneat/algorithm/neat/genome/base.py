from typing import Callable, Sequence

import numpy as np
import jax
from jax import vmap, numpy as jnp
from ..gene import BaseNodeGene, BaseConnGene
from .operations import BaseMutation, BaseCrossover, BaseDistance
from tensorneat.common import (
    State,
    StatefulBaseClass,
    hash_array,
)
from .utils import valid_cnt


class BaseGenome(StatefulBaseClass):
    network_type = None

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes: int,
        max_conns: int,
        node_gene: BaseNodeGene,
        conn_gene: BaseConnGene,
        mutation: BaseMutation,
        crossover: BaseCrossover,
        distance: BaseDistance,
        output_transform: Callable = None,
        input_transform: Callable = None,
        init_hidden_layers: Sequence[int] = (),
    ):
        
        # check transform functions
        if input_transform is not None:
            try:
                _ = input_transform(jnp.zeros(num_inputs))
            except Exception as e:
                raise ValueError(f"Output transform function failed: {e}")

        if output_transform is not None:
            try:
                _ = output_transform(jnp.zeros(num_outputs))
            except Exception as e:
                raise ValueError(f"Output transform function failed: {e}")

        # prepare for initialization
        all_layers = [num_inputs] + list(init_hidden_layers) + [num_outputs]
        layer_indices = []
        next_index = 0
        for layer in all_layers:
            layer_indices.append(list(range(next_index, next_index + layer)))
            next_index += layer

        all_init_nodes = []
        all_init_conns_in_idx = []
        all_init_conns_out_idx = []
        for i in range(len(layer_indices) - 1):
            in_layer = layer_indices[i]
            out_layer = layer_indices[i + 1]
            for in_idx in in_layer:
                for out_idx in out_layer:
                    all_init_conns_in_idx.append(in_idx)
                    all_init_conns_out_idx.append(out_idx)
            all_init_nodes.extend(in_layer)
        all_init_nodes.extend(layer_indices[-1])

        if max_nodes < len(all_init_nodes):
            raise ValueError(
                f"max_nodes={max_nodes} must be greater than or equal to the number of initial nodes={len(all_init_nodes)}"
            )

        if max_conns < len(all_init_conns_in_idx):
            raise ValueError(
                f"max_conns={max_conns} must be greater than or equal to the number of initial connections={len(all_init_conns_in_idx)}"
            )
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self.node_gene = node_gene
        self.conn_gene = conn_gene
        self.mutation = mutation
        self.crossover = crossover
        self.distance = distance
        self.output_transform = output_transform
        self.input_transform = input_transform

        self.input_idx = np.array(layer_indices[0])
        self.output_idx = np.array(layer_indices[-1])
        self.all_init_nodes = np.array(all_init_nodes)
        self.all_init_conns = np.c_[all_init_conns_in_idx, all_init_conns_out_idx]
        print(self.output_idx)

    def setup(self, state=State()):
        state = self.node_gene.setup(state)
        state = self.conn_gene.setup(state)
        state = self.mutation.setup(state, self)
        state = self.crossover.setup(state, self)
        state = self.distance.setup(state, self)
        return state

    def transform(self, state, nodes, conns):
        raise NotImplementedError

    def forward(self, state, transformed, inputs):
        raise NotImplementedError

    def sympy_func(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

    def execute_mutation(self, state, randkey, nodes, conns, new_node_key):
        return self.mutation(state, randkey, nodes, conns, new_node_key)

    def execute_crossover(self, state, randkey, nodes1, conns1, nodes2, conns2):
        return self.crossover(state, randkey, nodes1, conns1, nodes2, conns2)

    def execute_distance(self, state, nodes1, conns1, nodes2, conns2):
        return self.distance(state, nodes1, conns1, nodes2, conns2)

    def initialize(self, state, randkey):
        k1, k2 = jax.random.split(randkey)  # k1 for nodes, k2 for conns

        all_nodes_cnt = len(self.all_init_nodes)
        all_conns_cnt = len(self.all_init_conns)

        # initialize nodes
        nodes = jnp.full((self.max_nodes, self.node_gene.length), jnp.nan)
        # create node indices
        node_indices = self.all_init_nodes
        # create node attrs
        rand_keys_n = jax.random.split(k1, num=all_nodes_cnt)
        node_attr_func = vmap(self.node_gene.new_random_attrs, in_axes=(None, 0))
        node_attrs = node_attr_func(state, rand_keys_n)

        nodes = nodes.at[:all_nodes_cnt, 0].set(node_indices)  # set node indices
        nodes = nodes.at[:all_nodes_cnt, 1:].set(node_attrs)  # set node attrs

        # initialize conns
        conns = jnp.full((self.max_conns, self.conn_gene.length), jnp.nan)
        # create input and output indices
        conn_indices = self.all_init_conns
        # create conn attrs
        rand_keys_c = jax.random.split(k2, num=all_conns_cnt)
        conns_attr_func = jax.vmap(
            self.conn_gene.new_random_attrs,
            in_axes=(
                None,
                0,
            ),
        )
        conns_attrs = conns_attr_func(state, rand_keys_c)

        conns = conns.at[:all_conns_cnt, :2].set(conn_indices)  # set conn indices
        conns = conns.at[:all_conns_cnt, 2:].set(conns_attrs)  # set conn attrs

        return nodes, conns

    def network_dict(self, state, nodes, conns):
        return {
            "nodes": self._get_node_dict(state, nodes),
            "conns": self._get_conn_dict(state, conns),
        }

    def get_input_idx(self):
        return self.input_idx.tolist()

    def get_output_idx(self):
        return self.output_idx.tolist()

    def hash(self, nodes, conns):
        nodes_hashs = vmap(hash_array)(nodes)
        conns_hashs = vmap(hash_array)(conns)
        return hash_array(jnp.concatenate([nodes_hashs, conns_hashs]))

    def repr(self, state, nodes, conns, precision=2):
        nodes, conns = jax.device_get([nodes, conns])
        nodes_cnt, conns_cnt = valid_cnt(nodes), valid_cnt(conns)
        s = f"{self.__class__.__name__}(nodes={nodes_cnt}, conns={conns_cnt}):\n"
        s += f"\tNodes:\n"
        for node in nodes:
            if np.isnan(node[0]):
                break
            s += f"\t\t{self.node_gene.repr(state, node, precision=precision)}"
            node_idx = int(node[0])
            if np.isin(node_idx, self.input_idx):
                s += " (input)"
            elif np.isin(node_idx, self.output_idx):
                s += " (output)"
            s += "\n"

        s += f"\tConns:\n"
        for conn in conns:
            if np.isnan(conn[0]):
                break
            s += f"\t\t{self.conn_gene.repr(state, conn, precision=precision)}\n"
        return s

    def _get_conn_dict(self, state, conns):
        conns = jax.device_get(conns)
        conn_dict = {}
        for conn in conns:
            if np.isnan(conn[0]):
                continue
            cd = self.conn_gene.to_dict(state, conn)
            in_idx, out_idx = cd["in"], cd["out"]
            conn_dict[(in_idx, out_idx)] = cd
        return conn_dict

    def _get_node_dict(self, state, nodes):
        nodes = jax.device_get(nodes)
        node_dict = {}
        for node in nodes:
            if np.isnan(node[0]):
                continue
            nd = self.node_gene.to_dict(state, node)
            idx = nd["idx"]
            node_dict[idx] = nd
        return node_dict
