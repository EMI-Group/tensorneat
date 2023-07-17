from jax import Array, numpy as jnp


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
    def distance_node(state, array1: Array, array2: Array):
        return array1

    @staticmethod
    def distance_conn(state, array1: Array, array2: Array):
        return array1

    @staticmethod
    def forward(state, array: Array):
        return array
