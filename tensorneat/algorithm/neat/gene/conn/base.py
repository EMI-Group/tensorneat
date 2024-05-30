import jax, jax.numpy as jnp
from .. import BaseGene


class BaseConnGene(BaseGene):
    "Base class for connection genes."
    fixed_attrs = ["input_index", "output_index", "enabled"]

    def __init__(self):
        super().__init__()

    def crossover(self, state, randkey, gene1, gene2):
        def crossover_attr():
            return jnp.where(
                jax.random.normal(randkey, gene1.shape) > 0,
                gene1,
                gene2,
            )

        return jax.lax.cond(
            gene1[2] == gene2[2],  # if both genes are enabled or disabled
            crossover_attr,  # then randomly pick attributes from gene1 or gene2
            lambda: jnp.where(  # one gene is enabled and the other is disabled
                gene1[2],  # if gene1 is enabled
                gene1,  # then return gene1
                gene2,  # else return gene2
            ),
        )

    def forward(self, state, attrs, inputs):
        raise NotImplementedError
