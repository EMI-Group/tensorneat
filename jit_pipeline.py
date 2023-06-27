import time
from typing import Union, Callable

import numpy as np
import jax

from configs import Configer
from function_factory import FunctionFactory
from algorithms.neat import initialize_genomes, expand, expand_single

from algorithms.neat.jit_species import update_species
from algorithms.neat.operations import create_next_generation_then_speciate


class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config, function_factory=None, seed=42):
        self.randkey = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        self.config = config  # global config
        self.jit_config = Configer.create_jit_config(config)  # config used in jit-able functions
        self.function_factory = function_factory or FunctionFactory(self.config, self.jit_config)

        self.symbols = {
            'P': self.config['pop_size'],
            'N': self.config['init_maximum_nodes'],
            'C': self.config['init_maximum_connections'],
            'S': self.config['init_maximum_species'],
        }

        self.generation = 0
        self.best_genome = None

        self.pop_nodes, self.pop_cons = initialize_genomes(self.symbols['N'], self.symbols['C'], self.config)
        self.species_info = np.full((self.symbols['S'], 3), np.nan)
        self.species_info[0, :] = 0, -np.inf, 0
        self.idx2species = np.zeros(self.symbols['P'], dtype=np.int32)
        self.center_nodes = np.full((self.symbols['S'], self.symbols['N'], 5), np.nan)
        self.center_cons = np.full((self.symbols['S'], self.symbols['C'], 4), np.nan)
        self.center_nodes[0, :, :] = self.pop_nodes[0, :, :]
        self.center_cons[0, :, :] = self.pop_cons[0, :, :]

        self.best_fitness = float('-inf')
        self.best_genome = None
        self.generation_timestamp = time.time()

        self.evaluate_time = 0
        print(self.config)

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

    def tell(self, fitnesses):
        self.generation += 1

        species_info, center_nodes, center_cons, winner, loser, elite_mask = \
            update_species(self.randkey, fitnesses, self.species_info, self.idx2species, self.center_nodes,
                           self.center_cons, self.generation, self.jit_config)

        # node keys to be used in the mutation process
        new_node_keys = np.arange(self.generation * self.config['pop_size'],
                                  self.generation * self.config['pop_size'] + self.config['pop_size'])

        # create the next generation and then speciate the population
        self.pop_nodes, self.pop_cons, idx2specie, center_nodes, center_cons, species_keys = \
            create_next_generation_then_speciate(self.randkey, self.pop_nodes, self.pop_cons, winner, loser, elite_mask, new_node_keys, center_nodes,
                 center_cons, species_keys, species_key_start, self.jit_config)

        # carry data to cpu
        self.pop_nodes, self.pop_cons, idx2specie, center_nodes, center_cons, species_keys = \
            jax.device_get([self.pop_nodes, self.pop_cons, idx2specie, center_nodes, center_cons, species_keys])

        # update randkey
        self.randkey = jax.random.split(self.randkey)[0]

    def get_func(self, name):
        return self.function_factory.get(name, self.symbols)

    def auto_run(self, fitness_func, analysis: Union[Callable, str] = "default"):
        for _ in range(self.config['generation_limit']):
            forward_func = self.ask()

            tic = time.time()
            fitnesses = fitness_func(forward_func)
            self.evaluate_time += time.time() - tic

            assert np.all(~np.isnan(fitnesses)), "fitnesses should not be nan!"

            if analysis is not None:
                if analysis == "default":
                    self.default_analysis(fitnesses)
                else:
                    assert callable(analysis), f"What the fuck you passed in? A {analysis}?"
                    analysis(fitnesses)

            if max(fitnesses) >= self.config['fitness_threshold']:
                print("Fitness limit reached!")
                return self.best_genome

            self.tell(fitnesses)
        print("Generation limit reached!")
        return self.best_genome

    def default_analysis(self, fitnesses):
        max_f, min_f, mean_f, std_f = max(fitnesses), min(fitnesses), np.mean(fitnesses), np.std(fitnesses)

        new_timestamp = time.time()
        cost_time = new_timestamp - self.generation_timestamp
        self.generation_timestamp = new_timestamp

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_genome = (self.pop_nodes[max_idx], self.pop_cons[max_idx])

        print(f"Generation: {self.generation}",
              f"fitness: {max_f}, {min_f}, {mean_f}, {std_f}, Cost time: {cost_time}")
