import jax, jax.numpy as jnp
from .. import BaseGene


class BaseNodeGene(BaseGene):
    "Base class for node genes."
    fixed_attrs = ["index"]

    def __init__(self):
        super().__init__()

    def forward(self, state, attrs, inputs, is_output_node=False):
        raise NotImplementedError

    def update_by_batch(self, state, attrs, batch_inputs, is_output_node=False):
        # default: do not update attrs, but to calculate batch_res
        return (
            jax.vmap(self.forward, in_axes=(None, None, 0, None))(
                state, attrs, batch_inputs, is_output_node
            ),
            attrs,
        )
