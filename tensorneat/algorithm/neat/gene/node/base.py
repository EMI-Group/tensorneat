from .. import BaseGene


class BaseNodeGene(BaseGene):
    "Base class for node genes."
    fixed_attrs = ["index"]

    def __init__(self):
        super().__init__()

    def forward(self, state, attrs, inputs, is_output_node=False):
        raise NotImplementedError
