from utils import State


class BaseCrossover:

    def setup(self, key, state=State()):
        return state

    def __call__(self, state, key, genome, nodes1, nodes2, conns1, conns2):
        raise NotImplementedError
