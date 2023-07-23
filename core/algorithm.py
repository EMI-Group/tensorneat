from jax import Array
from .state import State
from .genome import Genome

EMPTY = lambda *args: args


class Algorithm:

    def setup(self, randkey, state: State = State()):
        """initialize the state of the algorithm"""
        pass

    def ask(self, state: State):
        """require the population to be evaluated"""
        pass

    def tell(self, state: State, fitness):
        """update the state of the algorithm"""
        pass

    def forward(self, inputs: Array, transformed: Array):
        """the forward function of a single forward transformation"""
        pass

    def forward_transform(self, state: State, genome: Genome):
        """create the forward transformation of a genome"""
        pass
