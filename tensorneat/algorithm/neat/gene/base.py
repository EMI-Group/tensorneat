from utils import State


class BaseGene:
    "Base class for node genes or connection genes."
    fixed_attrs = []
    custom_attrs = []

    def __init__(self):
        pass

    def setup(self, key, state=State()):
        return state

    def new_attrs(self, state, key):
        raise NotImplementedError

    def mutate(self, state, key, gene):
        raise NotImplementedError

    def distance(self, state, gene1, gene2):
        raise NotImplementedError

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    @property
    def length(self):
        return len(self.fixed_attrs) + len(self.custom_attrs)
