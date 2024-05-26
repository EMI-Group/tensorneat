from .. import BaseGene


class BaseConnGene(BaseGene):
    "Base class for connection genes."
    fixed_attrs = ["input_index", "output_index", "enabled"]

    def __init__(self):
        super().__init__()

    def forward(self, state, attrs, inputs):
        raise NotImplementedError
