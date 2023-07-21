from jax import Array, numpy as jnp, vmap


class BaseGene:
    node_attrs = []
    conn_attrs = []

    @staticmethod
    def setup(state, config):
        return state

    @staticmethod
    def new_node_attrs(state):
        return jnp.zeros(0)

    @staticmethod
    def new_conn_attrs(state):
        return jnp.zeros(0)

    @staticmethod
    def mutate_node(state, attrs: Array, key):
        return attrs

    @staticmethod
    def mutate_conn(state, attrs: Array, key):
        return attrs

    @staticmethod
    def distance_node(state, node1: Array, node2: Array):
        return node1

    @staticmethod
    def distance_conn(state, conn1: Array, conn2: Array):
        return conn1

    @staticmethod
    def forward_transform(state, nodes, conns):
        return nodes, conns

    @staticmethod
    def create_forward(config):
        return None
