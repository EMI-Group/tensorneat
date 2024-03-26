import jax.numpy as jnp

from utils import mutate_float
from . import BaseConnGene


class DefaultConnGene(BaseConnGene):
    "Default connection gene, with the same behavior as in NEAT-python."

    custom_attrs = ['weight']

    def __init__(
            self,
            weight_init_mean: float = 0.0,
            weight_init_std: float = 1.0,
            weight_mutate_power: float = 0.5,
            weight_mutate_rate: float = 0.8,
            weight_replace_rate: float = 0.1,
    ):
        super().__init__()
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.weight_mutate_power = weight_mutate_power
        self.weight_mutate_rate = weight_mutate_rate
        self.weight_replace_rate = weight_replace_rate

    def new_custom_attrs(self):
        return jnp.array([self.weight_init_mean])

    def mutate(self, key, conn):
        input_index = conn[0]
        output_index = conn[1]
        enabled = conn[2]
        weight = mutate_float(key,
                              conn[3],
                              self.weight_init_mean,
                              self.weight_init_std,
                              self.weight_mutate_power,
                              self.weight_mutate_rate,
                              self.weight_replace_rate
                              )

        return jnp.array([input_index, output_index, enabled, weight])

    def distance(self, attrs1, attrs2):
        return (attrs1[2] != attrs2[2]) + jnp.abs(attrs1[3] - attrs2[3])  # enable + weight

    def forward(self, attrs, inputs):
        weight = attrs[0]
        return inputs * weight
