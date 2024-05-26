from utils import State


class BaseMutation:
    def setup(self, state=State()):
        return state

    def __call__(self, state, randkey, genome, nodes, conns, new_node_key):
        raise NotImplementedError
