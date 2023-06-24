from functools import partial

import numpy as np
import jax

from configs.configer import Configer
from .genome.genome import initialize_genomes
from .function_factory import FunctionFactory


class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config, function_factory=None, seed=42):
        self.randkey = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        self.config = config  # global config
        self.jit_config = Configer.create_jit_config(config)  # config used in jit-able functions
        self.function_factory = function_factory or FunctionFactory(self.config)

        self.symbols = {
            'P': self.config['pop_size'],
            'N': self.config['init_maximum_nodes'],
            'C': self.config['init_maximum_connections'],
            'S': self.config['init_maximum_species'],
        }

        self.generation = 0
        self.best_genome = None

        self.pop_nodes, self.pop_cons = initialize_genomes(self.symbols['N'], self.symbols['C'], self.config)


    def ask(self):
        """
        Creates a function that receives a genome and returns a forward function.
        There are 3 types of config['forward_way']: {'single', 'pop', 'common'}

        single:
            Create pop_size number of forward functions.
            Each function receive (batch_size, input_size) and returns (batch_size, output_size)
            e.g. RL task

        pop:
            Create a single forward function, which use only once calculation for the population.
            The function receives (pop_size, batch_size, input_size) and returns (pop_size, batch_size, output_size)

        common:
            Special case of pop. The population has the same inputs.
            The function receives (batch_size, input_size) and returns (pop_size, batch_size, output_size)
            e.g. numerical regression; Hyper-NEAT

        """
        u_pop_cons = self.get_func('pop_unflatten_connections')(self.pop_nodes, self.pop_cons)
        pop_seqs = self.get_func('pop_topological_sort')(self.pop_nodes, u_pop_cons)

        if self.config['forward_way'] == 'single':
            forward_funcs = []
            for seq, nodes, cons in zip(pop_seqs, self.pop_nodes, u_pop_cons):
                func = lambda x: self.get_func('forward')(x, seq, nodes, cons)
                forward_funcs.append(func)
            return forward_funcs

        elif self.config['forward_way'] == 'pop':
            func = lambda x: self.get_func('pop_batch_forward')(x, pop_seqs, self.pop_nodes, u_pop_cons)
            return func

        elif self.config['forward_way'] == 'common':
            func = lambda x: self.get_func('common_forward')(x, pop_seqs, self.pop_nodes, u_pop_cons)
            return func

        else:
            raise NotImplementedError
    def get_func(self, name):
        return self.function_factory.get(name, self.symbols)
