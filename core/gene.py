from jax import Array, numpy as jnp

from config import GeneConfig
from .state import State
from .genome import Genome


class Gene:
    node_attrs = []
    conn_attrs = []

    @staticmethod
    def setup(config: GeneConfig, state: State):
        return state

    @staticmethod
    def new_node_attrs(state: State):
        return jnp.zeros(0)

    @staticmethod
    def new_conn_attrs(state: State):
        return jnp.zeros(0)

    @staticmethod
    def mutate_node(state: State, attrs: Array, randkey: Array):
        return attrs

    @staticmethod
    def mutate_conn(state: State, attrs: Array, randkey: Array):
        return attrs

    @staticmethod
    def distance_node(state: State, node1: Array, node2: Array):
        return node1

    @staticmethod
    def distance_conn(state: State, conn1: Array, conn2: Array):
        return conn1

    @staticmethod
    def forward_transform(state: State, genome: Genome):
        return jnp.zeros(0)  # transformed
    @staticmethod
    def create_forward(state: State, config: GeneConfig):
        return lambda *args: args  # forward function

