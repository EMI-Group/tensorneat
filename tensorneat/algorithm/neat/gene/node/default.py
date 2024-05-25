from typing import Tuple

import jax, jax.numpy as jnp

from utils import Act, Agg, act, agg, mutate_int, mutate_float
from . import BaseNodeGene


class DefaultNodeGene(BaseNodeGene):
    "Default node gene, with the same behavior as in NEAT-python."

    custom_attrs = ['bias', 'response', 'aggregation', 'activation']

    def __init__(
            self,
            bias_init_mean: float = 0.0,
            bias_init_std: float = 1.0,
            bias_mutate_power: float = 0.5,
            bias_mutate_rate: float = 0.7,
            bias_replace_rate: float = 0.1,

            response_init_mean: float = 1.0,
            response_init_std: float = 0.0,
            response_mutate_power: float = 0.5,
            response_mutate_rate: float = 0.7,
            response_replace_rate: float = 0.1,

            activation_default: callable = Act.sigmoid,
            activation_options: Tuple = (Act.sigmoid,),
            activation_replace_rate: float = 0.1,

            aggregation_default: callable = Agg.sum,
            aggregation_options: Tuple = (Agg.sum,),
            aggregation_replace_rate: float = 0.1,
    ):
        super().__init__()
        self.bias_init_mean = bias_init_mean
        self.bias_init_std = bias_init_std
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate

        self.response_init_mean = response_init_mean
        self.response_init_std = response_init_std
        self.response_mutate_power = response_mutate_power
        self.response_mutate_rate = response_mutate_rate
        self.response_replace_rate = response_replace_rate

        self.activation_default = activation_options.index(activation_default)
        self.activation_options = activation_options
        self.activation_indices = jnp.arange(len(activation_options))
        self.activation_replace_rate = activation_replace_rate

        self.aggregation_default = aggregation_options.index(aggregation_default)
        self.aggregation_options = aggregation_options
        self.aggregation_indices = jnp.arange(len(aggregation_options))
        self.aggregation_replace_rate = aggregation_replace_rate

    def new_attrs(self, state, key):
        return jnp.array(
            [self.bias_init_mean, self.response_init_mean, self.activation_default, self.aggregation_default]
        )

    def mutate(self, state, key, node):
        k1, k2, k3, k4 = jax.random.split(key, num=4)
        index = node[0]

        bias = mutate_float(k1, node[1], self.bias_init_mean, self.bias_init_std,
                            self.bias_mutate_power, self.bias_mutate_rate, self.bias_replace_rate)

        res = mutate_float(k2, node[2], self.response_init_mean, self.response_init_std,
                           self.response_mutate_power, self.response_mutate_rate, self.response_replace_rate)

        act = mutate_int(k3, node[3], self.activation_indices, self.activation_replace_rate)

        agg = mutate_int(k4, node[4], self.aggregation_indices, self.aggregation_replace_rate)

        return jnp.array([index, bias, res, act, agg])

    def distance(self, state, node1, node2):
        return (
                jnp.abs(node1[1] - node2[1]) +
                jnp.abs(node1[2] - node2[2]) +
                (node1[3] != node2[3]) +
                (node1[4] != node2[4])
        )

    def forward(self, state, attrs, inputs, is_output_node=False):
        bias, res, act_idx, agg_idx = attrs

        z = agg(agg_idx, inputs, self.aggregation_options)
        z = bias + res * z

        # the last output node should not be activated
        z = jax.lax.cond(
            is_output_node,
            lambda: z,
            lambda: act(act_idx, z, self.activation_options)
        )

        return z
