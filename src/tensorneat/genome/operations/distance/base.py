from tensorneat.common import StatefulBaseClass, State


class BaseDistance(StatefulBaseClass):

    def __call__(self, state, genome, nodes1, nodes2, conns1, conns2):
        """
        The distance between two genomes
        """
        raise NotImplementedError
