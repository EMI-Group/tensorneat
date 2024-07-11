from typing import Callable

import jax
from jax import vmap, numpy as jnp

from .substrate import *
from tensorneat.common import State, ACT, AGG
from tensorneat.algorithm import BaseAlgorithm, NEAT
from tensorneat.genome import BaseNode, BaseConn, RecurrentGenome


class HyperNEAT(BaseAlgorithm):
    def __init__(
        self,
        substrate: BaseSubstrate,
        neat: NEAT,
        weight_threshold: float = 0.3,
        max_weight: float = 5.0,
        aggregation: Callable = AGG.sum,
        activation: Callable = ACT.sigmoid,
        activate_time: int = 10,
        output_transform: Callable = ACT.standard_sigmoid,
    ):
        assert (
            substrate.query_coors.shape[1] == neat.num_inputs
        ), "Query coors of Substrate should be equal to NEAT input size"
        
        self.substrate = substrate
        self.neat = neat
        self.weight_threshold = weight_threshold
        self.max_weight = max_weight
        self.hyper_genome = RecurrentGenome(
            num_inputs=substrate.num_inputs,
            num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt,
            max_conns=substrate.conns_cnt,
            node_gene=HyperNEATNode(aggregation, activation),
            conn_gene=HyperNEATConn(),
            activate_time=activate_time,
            output_transform=output_transform,
        )
        self.pop_size = neat.pop_size

    def setup(self, state=State()):
        state = self.neat.setup(state)
        state = self.substrate.setup(state)
        return self.hyper_genome.setup(state)

    def ask(self, state):
        return self.neat.ask(state)

    def tell(self, state, fitness):
        state = self.neat.tell(state, fitness)
        return state

    def transform(self, state, individual):
        transformed = self.neat.transform(state, individual)
        query_res = vmap(self.neat.forward, in_axes=(None, None, 0))(
            state, transformed, self.substrate.query_coors
        )
        # mute the connection with weight weight threshold
        query_res = jnp.where(
            (-self.weight_threshold < query_res) & (query_res < self.weight_threshold),
            0.0,
            query_res,
        )

        # make query res in range [-max_weight, max_weight]
        query_res = jnp.where(
            query_res > 0, query_res - self.weight_threshold, query_res
        )
        query_res = jnp.where(
            query_res < 0, query_res + self.weight_threshold, query_res
        )
        query_res = query_res / (1 - self.weight_threshold) * self.max_weight

        h_nodes, h_conns = self.substrate.make_nodes(
            query_res
        ), self.substrate.make_conns(query_res)

        return self.hyper_genome.transform(state, h_nodes, h_conns)

    def forward(self, state, transformed, inputs):
        # add bias
        inputs_with_bias = jnp.concatenate([inputs, jnp.array([1])])

        res = self.hyper_genome.forward(state, transformed, inputs_with_bias)
        return res

    @property
    def num_inputs(self):
        return self.substrate.num_inputs - 1  # remove bias

    @property
    def num_outputs(self):
        return self.substrate.num_outputs

    def show_details(self, state, fitness):
        return self.neat.show_details(state, fitness)


class HyperNEATNode(BaseNode):
    def __init__(
        self,
        aggregation=AGG.sum,
        activation=ACT.sigmoid,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.activation = activation

    def forward(self, state, attrs, inputs, is_output_node=False):
        return jax.lax.cond(
            is_output_node,
            lambda: self.aggregation(inputs),  # output node does not need activation
            lambda: self.activation(self.aggregation(inputs)),
        )


class HyperNEATConn(BaseConn):
    custom_attrs = ["weight"]

    def forward(self, state, attrs, inputs):
        weight = attrs[0]
        return inputs * weight
