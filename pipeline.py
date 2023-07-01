import time
from typing import Union, Callable

import numpy as np
import jax
from jax import jit, vmap

from configs import Configer
from algorithms import neat


class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config, seed=42):
        self.randkey = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        self.config = config  # global config
        self.jit_config = Configer.create_jit_config(config)  # config used in jit-able functions

        self.P = config['pop_size']
        self.N = config['init_maximum_nodes']
        self.C = config['init_maximum_connections']
        self.S = config['init_maximum_species']

        self.generation = 0
        self.best_genome = None

        self.pop_nodes, self.pop_cons = neat.initialize_genomes(self.N, self.C, self.config)
        self.species_info = np.full((self.S, 4), np.nan)
        self.species_info[0, :] = 0, -np.inf, 0, self.P
        self.idx2species = np.zeros(self.P, dtype=np.float32)
        self.center_nodes = np.full((self.S, self.N, 5), np.nan)
        self.center_cons = np.full((self.S, self.C, 4), np.nan)
        self.center_nodes[0, :, :] = self.pop_nodes[0, :, :]
        self.center_cons[0, :, :] = self.pop_cons[0, :, :]

        self.best_fitness = float('-inf')
        self.best_genome = None
        self.generation_timestamp = time.time()

        self.evaluate_time = 0

        self.pop_unflatten_connections = jit(vmap(neat.unflatten_connections))
        self.pop_topological_sort = jit(vmap(neat.topological_sort))
        self.forward = neat.create_forward_function(config)

        # fitness_lower = np.zeros(self.P, dtype=np.float32)
        # randkey_lower = np.zeros(2, dtype=np.uint32)
        # pop_nodes_lower = np.zeros((self.P, self.N, 5), dtype=np.float32)
        # pop_cons_lower = np.zeros((self.P, self.C, 4), dtype=np.float32)
        # species_info_lower = np.zeros((self.S, 4), dtype=np.float32)
        # idx2species_lower = np.zeros(self.P, dtype=np.float32)
        # center_nodes_lower = np.zeros((self.S, self.N, 5), dtype=np.float32)
        # center_cons_lower = np.zeros((self.S, self.C, 4), dtype=np.float32)
        #
        # self.tell_func = jit(neat.tell).lower(fitness_lower,
        #                                       randkey_lower,
        #                                       pop_nodes_lower,
        #                                       pop_cons_lower,
        #                                       species_info_lower,
        #                                       idx2species_lower,
        #                                       center_nodes_lower,
        #                                       center_cons_lower,
        #                                       0,
        #                                       self.jit_config).compile()

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
        u_pop_cons = self.pop_unflatten_connections(self.pop_nodes, self.pop_cons)
        pop_seqs = self.pop_topological_sort(self.pop_nodes, u_pop_cons)

        # only common mode is supported currently
        assert self.config['forward_way'] == 'common'
        return lambda x: self.forward(x, pop_seqs, self.pop_nodes, u_pop_cons)

    def tell(self, fitness):

        self.randkey, self.pop_nodes, self.pop_cons, self.species_info, self.idx2species, self.center_nodes, \
            self.center_cons, self.generation = neat.tell(fitness, self.randkey, self.pop_nodes, self.pop_cons,
                                                          self.species_info, self.idx2species, self.center_nodes,
                                                          self.center_cons, self.generation, self.jit_config)

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

        species_sizes = [int(i) for i in self.species_info[:, 3] if i > 0]

        print(f"Generation: {self.generation}",
              f"species: {len(species_sizes)}, {species_sizes}",
              f"fitness: {max_f}, {min_f}, {mean_f}, {std_f}, Cost time: {cost_time}")
