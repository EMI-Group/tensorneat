from utils import State


class BaseGene:
    "Base class for node genes or connection genes."
    fixed_attrs = []
    custom_attrs = []

    def __init__(self):
        pass

    def setup(self, state=State()):
        return state

    def new_custom_attrs(self, state):
        # the attrs which make the least influence on the network, used in add node or add conn in mutation
        raise NotImplementedError

    def new_random_attrs(self, state, randkey):
        # random attributes of the gene. used in initialization.
        raise NotImplementedError

    def mutate(self, state, randkey, gene):
        raise NotImplementedError

    def distance(self, state, gene1, gene2):
        raise NotImplementedError

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    @property
    def length(self):
        return len(self.fixed_attrs) + len(self.custom_attrs)
