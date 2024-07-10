from typing import Callable

import jax, jax.numpy as jnp

from tensorneat.common import State, Act, Agg
from .. import BaseAlgorithm, NEAT
from ..neat.gene import BaseNodeGene, BaseConnGene
from ..neat.genome import RecurrentGenome
from .substrate import *


class HyperNEAT(BaseAlgorithm):
    def __init__(
        self,
        substrate: BaseSubstrate,
        neat: NEAT,
        below_threshold: float = 0.3,
        max_weight: float = 5.0,
        aggregation=Agg.sum,
        activation=Act.sigmoid,
        activate_time: int = 10,
        output_transform: Callable = Act.sigmoid,
    ):
        assert (
            substrate.query_coors.shape[1] == neat.num_inputs
        ), "Substrate input size should be equal to NEAT input size"

        self.substrate = substrate
        self.neat = neat
        self.below_threshold = below_threshold
        self.max_weight = max_weight
        self.hyper_genome = RecurrentGenome(
            num_inputs=substrate.num_inputs,
            num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt,
            max_conns=substrate.conns_cnt,
            node_gene=HyperNodeGene(aggregation, activation),
            conn_gene=HyperNEATConnGene(),
            activate_time=activate_time,
            output_transform=output_transform,
        )

    def setup(self, state=State()):
        state = self.neat.setup(state)
        state = self.substrate.setup(state)
        return self.hyper_genome.setup(state)

    def ask(self, state: State):
        return self.neat.ask(state)

    def tell(self, state: State, fitness):
        state = self.neat.tell(state, fitness)
        return state

    def transform(self, state, individual):
        transformed = self.neat.transform(state, individual)
        query_res = jax.vmap(self.neat.forward, in_axes=(None, None, 0))(
            state, transformed, self.substrate.query_coors
        )
        # mute the connection with weight below threshold
        query_res = jnp.where(
            (-self.below_threshold < query_res) & (query_res < self.below_threshold),
            0.0,
            query_res,
        )

        # make query res in range [-max_weight, max_weight]
        query_res = jnp.where(
            query_res > 0, query_res - self.below_threshold, query_res
        )
        query_res = jnp.where(
            query_res < 0, query_res + self.below_threshold, query_res
        )
        query_res = query_res / (1 - self.below_threshold) * self.max_weight

        h_nodes, h_conns = self.substrate.make_nodes(
            query_res
        ), self.substrate.make_conn(query_res)

        return self.hyper_genome.transform(state, h_nodes, h_conns)

    def forward(self, state, inputs, transformed):
        # add bias
        inputs_with_bias = jnp.concatenate([inputs, jnp.array([1])])

        res = self.hyper_genome.forward(state, inputs_with_bias, transformed)
        return res

    @property
    def num_inputs(self):
        return self.substrate.num_inputs - 1  # remove bias

    @property
    def num_outputs(self):
        return self.substrate.num_outputs

    @property
    def pop_size(self):
        return self.neat.pop_size

    def member_count(self, state: State):
        return self.neat.member_count(state)

    def generation(self, state: State):
        return self.neat.generation(state)


class HyperNodeGene(BaseNodeGene):
    def __init__(
        self,
        aggregation=Agg.sum,
        activation=Act.sigmoid,
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


class HyperNEATConnGene(BaseConnGene):
    custom_attrs = ["weight"]

    def forward(self, state, attrs, inputs):
        weight = attrs[0]
        return inputs * weight
