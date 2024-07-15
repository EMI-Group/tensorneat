from tensorneat.common import StatefulBaseClass, State


class BaseCrossover(StatefulBaseClass):

    def __call__(self, state, genome, randkey, nodes1, nodes2, conns1, conns2):
        raise NotImplementedError
