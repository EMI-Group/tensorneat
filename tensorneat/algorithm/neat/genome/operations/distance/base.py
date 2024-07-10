from tensorneat.common import StatefulBaseClass, State


class BaseDistance(StatefulBaseClass):

    def setup(self, state=State(), genome = None):
        assert genome is not None, "genome should not be None"
        self.genome = genome
        return state

    def __call__(self, state, nodes1, nodes2, conns1, conns2):
        """
        The distance between two genomes
        """
        raise NotImplementedError
