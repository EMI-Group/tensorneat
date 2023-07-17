from jax import Array, numpy as jnp

from . import BaseGene


class NormalGene(BaseGene):
    node_attrs = ['bias', 'response', 'aggregation', 'activation']
    conn_attrs = ['weight']

    @staticmethod
    def setup(state, config):
        return state

    @staticmethod
    def new_node_attrs(state):
        return jnp.array([0, 0, 0, 0])

    @staticmethod
    def new_conn_attrs(state):
        return jnp.array([0])

    @staticmethod
    def mutate_node(state, attrs: Array, key):
        return attrs

    @staticmethod
    def mutate_conn(state, attrs: Array, key):
        return attrs

    @staticmethod
    def distance_node(state, array: Array):
        return array

    @staticmethod
    def distance_conn(state, array: Array):
        return array

    @staticmethod
    def forward(state, array: Array):
        return array
