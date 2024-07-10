import jax.numpy as jnp
from . import BaseNodeGene
from tensorneat.common import Agg


class KANNode(BaseNodeGene):
    "Node gene for KAN, with only a sum aggregation."

    custom_attrs = []

    def __init__(self):
        super().__init__()

    def new_identity_attrs(self, state):
        return jnp.array([])

    def new_random_attrs(self, state, randkey):
        return jnp.array([])

    def mutate(self, state, randkey, attrs):
        return jnp.array([])

    def distance(self, state, attrs1, attrs2):
        return 0

    def forward(self, state, attrs, inputs, is_output_node=False):
        return Agg.sum(inputs)
