import jax, jax.numpy as jnp
from .. import BaseGene


class BaseNodeGene(BaseGene):
    "Base class for node genes."
    fixed_attrs = ["index"]

    def __init__(self):
        super().__init__()

    def crossover(self, state, randkey, gene1, gene2):
        return jnp.where(
            jax.random.normal(randkey, gene1.shape) > 0,
            gene1,
            gene2,
        )

    def forward(self, state, attrs, inputs, is_output_node=False):
        raise NotImplementedError
