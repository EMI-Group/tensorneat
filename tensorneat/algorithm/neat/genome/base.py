import jax, jax.numpy as jnp
from ..gene import BaseNodeGene, BaseConnGene
from ..ga import BaseMutation, BaseCrossover
from utils import State, StatefulBaseClass


class BaseGenome(StatefulBaseClass):
    network_type = None

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes: int,
        max_conns: int,
        node_gene: BaseNodeGene,
        conn_gene: BaseConnGene,
        mutation: BaseMutation,
        crossover: BaseCrossover,
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_idx = jnp.arange(num_inputs)
        self.output_idx = jnp.arange(num_inputs, num_inputs + num_outputs)
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self.node_gene = node_gene
        self.conn_gene = conn_gene
        self.mutation = mutation
        self.crossover = crossover

    def setup(self, state=State()):
        state = self.node_gene.setup(state)
        state = self.conn_gene.setup(state)
        state = self.mutation.setup(state)
        state = self.crossover.setup(state)
        return state

    def transform(self, state, nodes, conns):
        raise NotImplementedError

    def restore(self, state, transformed):
        raise NotImplementedError

    def forward(self, state, transformed, inputs):
        raise NotImplementedError

    def execute_mutation(self, state, randkey, nodes, conns, new_node_key):
        return self.mutation(state, randkey, self, nodes, conns, new_node_key)

    def execute_crossover(self, state, randkey, nodes1, conns1, nodes2, conns2):
        return self.crossover(state, randkey, self, nodes1, conns1, nodes2, conns2)

    def initialize(self, state, randkey):
        """
        Default initialization method for the genome.
        Add an extra hidden node.
        Make all input nodes and output nodes connected to the hidden node.
        All attributes will be initialized randomly using gene.new_random_attrs method.

        For example, a network with 2 inputs and 1 output, the structure will be:
        nodes:
            [
                [0, attrs0],  # input node 0
                [1, attrs1],  # input node 1
                [2, attrs2],  # output node 0
                [3, attrs3],  # hidden node
                [NaN, NaN],  # empty node
            ]
        conns:
            [
                [0, 3, attrs0],  # input node 0 -> hidden node
                [1, 3, attrs1],  # input node 1 -> hidden node
                [3, 2, attrs2], # hidden node -> output node 0
                [NaN, NaN],
                [NaN, NaN],
            ]
        """

        k1, k2 = jax.random.split(randkey)  # k1 for nodes, k2 for conns
        # initialize nodes
        new_node_key = (
            max([*self.input_idx, *self.output_idx]) + 1
        )  # the key for the hidden node
        node_keys = jnp.concatenate(
            [self.input_idx, self.output_idx, jnp.array([new_node_key])]
        )  # the list of all node keys

        # initialize nodes and connections with NaN
        nodes = jnp.full((self.max_nodes, self.node_gene.length), jnp.nan)
        conns = jnp.full((self.max_conns, self.conn_gene.length), jnp.nan)

        # set keys for input nodes, output nodes and hidden node
        nodes = nodes.at[node_keys, 0].set(node_keys)

        # generate random attributes for nodes
        node_keys = jax.random.split(k1, len(node_keys))
        random_node_attrs = jax.vmap(
            self.node_gene.new_random_attrs, in_axes=(None, 0)
        )(state, node_keys)
        nodes = nodes.at[: len(node_keys), 1:].set(random_node_attrs)

        # initialize conns
        # input-hidden connections
        input_conns = jnp.c_[
            self.input_idx, jnp.full_like(self.input_idx, new_node_key)
        ]
        conns = conns.at[self.input_idx, :2].set(input_conns)  # in-keys, out-keys

        # output-hidden connections
        output_conns = jnp.c_[
            jnp.full_like(self.output_idx, new_node_key), self.output_idx
        ]
        conns = conns.at[self.output_idx, :2].set(output_conns)  # in-keys, out-keys

        conn_keys = jax.random.split(k2, num=len(self.input_idx) + len(self.output_idx))
        # generate random attributes for conns
        random_conn_attrs = jax.vmap(
            self.conn_gene.new_random_attrs, in_axes=(None, 0)
        )(state, conn_keys)
        conns = conns.at[: len(conn_keys), 2:].set(random_conn_attrs)

        return nodes, conns

    def update_by_batch(self, state, batch_input, transformed):
        """
        Update the genome by a batch of data.
        """
        raise NotImplementedError
