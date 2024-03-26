class BaseGene:
    "Base class for node genes or connection genes."
    fixed_attrs = []
    custom_attrs = []

    def __init__(self):
        pass

    def new_custom_attrs(self):
        raise NotImplementedError

    def mutate(self, randkey, gene):
        raise NotImplementedError

    def distance(self, gene1, gene2):
        raise NotImplementedError

    def forward(self, attrs, inputs):
        raise NotImplementedError

    @property
    def length(self):
        return len(self.fixed_attrs) + len(self.custom_attrs)