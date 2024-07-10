from tensorneat.common import State, StatefulBaseClass
from tensorneat.genome import BaseGenome


class BaseSpecies(StatefulBaseClass):
    genome: BaseGenome
    pop_size: int
    species_size: int

    def ask(self, state: State):
        raise NotImplementedError

    def tell(self, state: State, fitness):
        raise NotImplementedError

    def update_species(self, state, fitness):
        raise NotImplementedError

    def speciate(self, state):
        raise NotImplementedError
