from config import GeneConfig
from .state import State


class Gene:
    node_attrs = []
    conn_attrs = []

    def __init__(self, config: GeneConfig):
        raise NotImplementedError

    def setup(self, state=State()):
        raise NotImplementedError

    def update(self, state):
        raise NotImplementedError

    def new_node_attrs(self, state: State):
        raise NotImplementedError

    def new_conn_attrs(self, state: State):
        raise NotImplementedError

    def mutate_node(self, state: State, randkey, node_attrs):
        raise NotImplementedError

    def mutate_conn(self, state: State, randkey, conn_attrs):
        raise NotImplementedError

    def distance_node(self, state: State, node_attrs1, node_attrs2):
        raise NotImplementedError

    def distance_conn(self, state: State, conn_attrs1, conn_attrs2):
        raise NotImplementedError

    def forward_transform(self, state: State, genome):
        raise NotImplementedError

    def forward(self, state: State, inputs, transform):
        raise NotImplementedError
