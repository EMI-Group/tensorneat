from utils import State


class BaseMutation:

    def setup(self, key, state=State()):
        return state

    def __call__(self, state, key, genome, nodes, conns, new_node_key):
        raise NotImplementedError
