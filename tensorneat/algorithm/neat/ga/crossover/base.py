from utils import StatefulBaseClass


class BaseCrossover(StatefulBaseClass):
    def __call__(self, state, randkey, genome, nodes1, nodes2, conns1, conns2):
        raise NotImplementedError
