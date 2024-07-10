import jax, jax.numpy as jnp
from .. import BaseGene


class BaseNodeGene(BaseGene):
    "Base class for node genes."
    fixed_attrs = ["index"]

    def __init__(self):
        super().__init__()

    def forward(self, state, attrs, inputs, is_output_node=False):
        raise NotImplementedError

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

    def sympy_func(self, state, node_dict, inputs, is_output_node=False):
        raise NotImplementedError
