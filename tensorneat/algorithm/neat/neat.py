from tensorneat.common import State
from .. import BaseAlgorithm
from .species import *


class NEAT(BaseAlgorithm):
    def __init__(
        self,
        species: BaseSpecies,
    ):
        self.species = species
        self.genome = species.genome

    def setup(self, state=State()):
        state = self.species.setup(state)
        return state

    def ask(self, state: State):
        return self.species.ask(state)

    def tell(self, state: State, fitness):
        return self.species.tell(state, fitness)

    def transform(self, state, individual):
        """transform the genome into a neural network"""
        nodes, conns = individual
        return self.genome.transform(state, nodes, conns)

    def restore(self, state, transformed):
        return self.genome.restore(state, transformed)

    def forward(self, state, transformed, inputs):
        return self.genome.forward(state, transformed, inputs)

    def update_by_batch(self, state, batch_input, transformed):
        return self.genome.update_by_batch(state, batch_input, transformed)

    @property
    def num_inputs(self):
        return self.genome.num_inputs

    @property
    def num_outputs(self):
        return self.genome.num_outputs

    @property
    def pop_size(self):
        return self.species.pop_size

    def member_count(self, state: State):
        return state.member_count

    def generation(self, state: State):
        # to analysis the algorithm
        return state.generation
