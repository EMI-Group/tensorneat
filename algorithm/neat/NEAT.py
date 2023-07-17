from algorithm.state import State
from .gene import *
from .genome import initialize_genomes


class NEAT:
    def __init__(self, config):
        self.config = config
        if self.config['gene_type'] == 'normal':
            self.gene_type = NormalGene
        else:
            raise NotImplementedError

    def setup(self, randkey):

        state = State(
            randkey=randkey,
            P=self.config['pop_size'],
            N=self.config['maximum_nodes'],
            C=self.config['maximum_connections'],
            S=self.config['maximum_species'],
            NL=1 + len(self.gene_type.node_attrs),  # node length = (key) + attributes
            CL=3 + len(self.gene_type.conn_attrs),  # conn length = (in, out, key) + attributes
            input_idx=self.config['input_idx'],
            output_idx=self.config['output_idx']
        )

        pop_nodes, pop_conns = initialize_genomes(state, self.gene_type)
        next_node_key = max(*state.input_idx, *state.output_idx) + 2
        state = state.update(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            next_node_key=next_node_key
        )

        return state

    def tell(self, state, fitness):
        return State()

    def ask(self, state):
        return State()
