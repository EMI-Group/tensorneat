from tensorneat.common import StatefulBaseClass, State


class BaseMutation(StatefulBaseClass):

    def setup(self, state=State(), genome = None):
        assert genome is not None, "genome should not be None"
        self.genome = genome
        return state

    def __call__(self, state, randkey, nodes, conns, new_node_key):
        raise NotImplementedError
