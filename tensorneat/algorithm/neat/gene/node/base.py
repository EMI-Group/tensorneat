import jax, jax.numpy as jnp
from .. import BaseGene


class BaseNodeGene(BaseGene):
    "Base class for node genes."
    fixed_attrs = ["index"]

    def __init__(self):
        super().__init__()

    def forward(self, state, attrs, inputs, is_output_node=False):
        raise NotImplementedError

    def input_transform(self, state, attrs, inputs):
        """
        make transformation in the input node.
        default: do nothing
        """
        return inputs

    def update_by_batch(self, state, attrs, batch_inputs, is_output_node=False):
        # default: do not update attrs, but to calculate batch_res
        return (
            jax.vmap(self.forward, in_axes=(None, None, 0, None))(
                state, attrs, batch_inputs, is_output_node
            ),
            attrs,
        )

    def update_input_transform(self, state, attrs, batch_inputs):
        """
        update the attrs for transformation in the input node.
        default: do nothing
        """
        return (
            jax.vmap(self.input_transform, in_axes=(None, None, 0))(
                state, attrs, batch_inputs
            ),
            attrs,
        )

    def repr(self, state, node, precision=2, idx_width=3, func_width=8):
        idx = node[0]

        idx = int(idx)
        return "{}(idx={:<{idx_width}})".format(
            self.__class__.__name__, idx, idx_width=idx_width
        )

    def to_dict(self, state, node):
        idx = node[0]
        return {
            "idx": int(idx),
        }

    def sympy_func(self, state, node_dict, inputs, is_output_node=False, precision=None):
        raise NotImplementedError
