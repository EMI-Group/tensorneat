from utils import State


class BaseSpecies:
    def setup(self, key, state=State()):
        return state

    def ask(self, state: State):
        raise NotImplementedError

    def update_species(self, state, fitness, generation):
        raise NotImplementedError

    def speciate(self, state, generation):
        raise NotImplementedError
