from utils import StatefulBaseClass


class BaseMutation(StatefulBaseClass):
    def __call__(self, state, randkey, genome, nodes, conns, new_node_key):
        raise NotImplementedError
