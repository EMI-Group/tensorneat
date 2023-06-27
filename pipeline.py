import time
from typing import Union, Callable

import numpy as np
import jax

from configs import Configer
from function_factory import FunctionFactory
from algorithms.neat import initialize_genomes, expand, expand_single, SpeciesController


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

        self.species_controller = SpeciesController(self.config)
        self.pop_nodes, self.pop_cons = initialize_genomes(self.symbols['N'], self.symbols['C'], self.config)
        self.species_controller.init_speciate(self.pop_nodes, self.pop_cons)

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

        winner, loser, elite_mask, center_nodes, center_cons, species_keys, species_key_start = \
            self.species_controller.ask(fitnesses, self.generation, self.symbols)

        # node keys to be used in the mutation process
        new_node_keys = np.arange(self.generation * self.config['pop_size'],
                                  self.generation * self.config['pop_size'] + self.config['pop_size'])

        # create the next generation and then speciate the population
        self.pop_nodes, self.pop_cons, idx2specie, center_nodes, center_cons, species_keys = \
            self.get_func('create_next_generation_then_speciate') \
                (self.randkey, self.pop_nodes, self.pop_cons, winner, loser, elite_mask, new_node_keys, center_nodes,
                 center_cons, species_keys, species_key_start, self.jit_config)

        # carry data to cpu
        self.pop_nodes, self.pop_cons, idx2specie, center_nodes, center_cons, species_keys = \
            jax.device_get([self.pop_nodes, self.pop_cons, idx2specie, center_nodes, center_cons, species_keys])

        self.species_controller.tell(idx2specie, center_nodes, center_cons, species_keys, self.generation)

        # expand the population if needed
        self.expand()

        # update randkey
        self.randkey = jax.random.split(self.randkey)[0]

    def expand(self):
        """
        Expand the population if needed.
        when the maximum node number >= N or the maximum connection number of >= C
        the population will expand
        """

        # analysis nodes
        pop_node_keys = self.pop_nodes[:, :, 0]
        pop_node_sizes = np.sum(~np.isnan(pop_node_keys), axis=1)
        max_node_size = np.max(pop_node_sizes)

        # analysis connections
        pop_con_keys = self.pop_cons[:, :, 0]
        pop_node_sizes = np.sum(~np.isnan(pop_con_keys), axis=1)
        max_con_size = np.max(pop_node_sizes)

        # expand if needed
        if max_node_size >= self.symbols['N'] or max_con_size >= self.symbols['C']:
            if max_node_size > self.symbols['N'] * self.config['pre_expand_threshold']:
                self.symbols['N'] = int(self.symbols['N'] * self.config['expand_coe'])
                print(f"pre node expand to {self.symbols['N']}!")

            if max_con_size > self.symbols['C'] * self.config['pre_expand_threshold']:
                self.symbols['C'] = int(self.symbols['C'] * self.config['expand_coe'])
                print(f"pre connection expand to {self.symbols['C']}!")

            self.pop_nodes, self.pop_cons = expand(self.pop_nodes, self.pop_cons, self.symbols['N'], self.symbols['C'])
            # don't forget to expand representation genome in species
            for s in self.species_controller.species.values():
                s.representative = expand_single(*s.representative, self.symbols['N'], self.symbols['C'])

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
                    assert callable(analysis), f"Callable is needed here😅😅😅 A {analysis}?"
                    analysis(fitnesses)

            if max(fitnesses) >= self.config['fitness_threshold']:
                print("Fitness limit reached!")
                return self.best_genome

            self.tell(fitnesses)
        print("Generation limit reached!")
        return self.best_genome

    def default_analysis(self, fitnesses):
        max_f, min_f, mean_f, std_f = max(fitnesses), min(fitnesses), np.mean(fitnesses), np.std(fitnesses)
        species_sizes = [len(s.members) for s in self.species_controller.species.values()]

        new_timestamp = time.time()
        cost_time = new_timestamp - self.generation_timestamp
        self.generation_timestamp = new_timestamp

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_genome = (self.pop_nodes[max_idx], self.pop_cons[max_idx])

        print(f"Generation: {self.generation}",
              f"fitness: {max_f}, {min_f}, {mean_f}, {std_f}, Species sizes: {species_sizes}, Cost time: {cost_time}")
