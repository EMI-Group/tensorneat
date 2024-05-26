from utils import State


class BaseCrossover:
    def setup(self, state=State()):
        return state

    def __call__(self, state, randkey, genome, nodes1, nodes2, conns1, conns2):
        raise NotImplementedError
