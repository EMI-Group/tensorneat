import jax, jax.numpy as jnp
from utils import State, StatefulBaseClass


class BaseGene(StatefulBaseClass):
    "Base class for node genes or connection genes."
    fixed_attrs = []
    custom_attrs = []

    def __init__(self):
        pass

    def new_identity_attrs(self, state):
        # the attrs which do identity transformation, used in mutate add node
        raise NotImplementedError

    def new_random_attrs(self, state, randkey):
        # random attributes of the gene. used in initialization.
        raise NotImplementedError

    def mutate(self, state, randkey, attrs):
        raise NotImplementedError

    def crossover(self, state, randkey, attrs1, attrs2):
        return jnp.where(
            jax.random.normal(randkey, attrs1.shape) > 0,
            attrs1,
            attrs2,
        )

    def distance(self, state, attrs1, attrs2):
        raise NotImplementedError

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    def update_by_batch(self, state, attrs, batch_inputs):
        raise NotImplementedError

    @property
    def length(self):
        return len(self.fixed_attrs) + len(self.custom_attrs)

    def repr(self, state, gene, precision=2):
        raise NotImplementedError
