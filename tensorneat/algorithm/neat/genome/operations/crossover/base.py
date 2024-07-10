from tensorneat.common import StatefulBaseClass, State


class BaseCrossover(StatefulBaseClass):

    def setup(self, state=State(), genome = None):
        assert genome is not None, "genome should not be None"
        self.genome = genome
        return state

    def __call__(self, state, randkey, nodes1, nodes2, conns1, conns2):
        raise NotImplementedError
