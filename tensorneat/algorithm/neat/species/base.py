from utils import State
from ..genome import BaseGenome


class BaseSpecies:
    genome: BaseGenome
    pop_size: int
    species_size: int

    def setup(self, state=State()):
        return state

    def ask(self, state: State):
        raise NotImplementedError

    def update_species(self, state, fitness):
        raise NotImplementedError

    def speciate(self, state):
        raise NotImplementedError
