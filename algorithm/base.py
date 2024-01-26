from utils import State


class BaseAlgorithm:

    def setup(self, randkey):
        """initialize the state of the algorithm"""

        raise NotImplementedError

    def ask(self, state: State):
        """require the population to be evaluated"""
        raise NotImplementedError

    def tell(self, state: State, fitness):
        """update the state of the algorithm"""
        raise NotImplementedError

    def transform(self, state: State):
        """transform the genome into a neural network"""
        raise NotImplementedError

    def forward(self, inputs, transformed):
        raise NotImplementedError