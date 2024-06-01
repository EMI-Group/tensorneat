import jax, jax.numpy as jnp
from .. import BaseGene


class BaseConnGene(BaseGene):
    "Base class for connection genes."
    fixed_attrs = ["input_index", "output_index"]

    def __init__(self):
        super().__init__()

    def new_zero_attrs(self, state):
        # the attrs which make the least influence on the network, used in mutate add conn
        raise NotImplementedError

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    def update_by_batch(self, state, attrs, batch_inputs):
        # default: do not update attrs, but to calculate batch_res
        return (
            jax.vmap(self.forward, in_axes=(None, None, 0))(state, attrs, batch_inputs),
            attrs,
        )
