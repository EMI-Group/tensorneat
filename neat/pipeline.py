from typing import List, Union, Tuple, Callable
import time

import jax
import numpy as np

from .species import SpeciesController
from .genome import expand, expand_single
from .function_factory import FunctionFactory

from .population import *


class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config, function_factory, seed=42):
        self.time_dict = {}
        self.function_factory = function_factory

        self.randkey = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        self.config = config
        self.N = config.basic.init_maximum_nodes
        self.C = config.basic.init_maximum_connections
        self.S = config.basic.init_maximum_species
        self.expand_coe = config.basic.expands_coe
        self.pop_size = config.neat.population.pop_size

        self.species_controller = SpeciesController(config)
        self.initialize_func = self.function_factory.create_initialize(self.N, self.C)
        self.pop_nodes, self.pop_cons, self.input_idx, self.output_idx = self.initialize_func()

        self.create_and_speciate = self.function_factory.create_update_speciate(self.N, self.C, self.S)

        self.generation = 0
        self.generation_time_list = []
        self.species_controller.init_speciate(self.pop_nodes, self.pop_cons)

        self.best_fitness = float('-inf')
        self.best_genome = None
        self.generation_timestamp = time.time()

        self.evaluate_time = 0

    def ask(self):
        """
        Create a forward function for the population.
        :return:
        Algorithm gives the population a forward function, then environment gives back the fitnesses.
        """
        return self.function_factory.ask_pop_batch_forward(self.pop_nodes, self.pop_cons)

    def tell(self, fitnesses):

        self.generation += 1

        winner_part, loser_part, elite_mask, pre_spe_center_nodes, pre_spe_center_cons, pre_species_keys, new_species_key_start = self.species_controller.ask(
            fitnesses,
            self.generation,
            self.S, self.N, self.C)

        new_node_keys = np.arange(self.generation * self.pop_size, self.generation * self.pop_size + self.pop_size)
        self.pop_nodes, self.pop_cons, idx2specie, new_center_nodes, new_center_cons, new_species_keys = self.create_and_speciate(
            self.randkey, self.pop_nodes, self.pop_cons, winner_part, loser_part, elite_mask,
            new_node_keys,
            pre_spe_center_nodes, pre_spe_center_cons, pre_species_keys, new_species_key_start)


        self.pop_nodes, self.pop_cons, idx2specie, new_center_nodes, new_center_cons, new_species_keys = \
            jax.device_get([self.pop_nodes, self.pop_cons, idx2specie, new_center_nodes, new_center_cons, new_species_keys])

        self.species_controller.tell(idx2specie, new_center_nodes, new_center_cons, new_species_keys, self.generation)

        self.expand()

    def auto_run(self, fitness_func, analysis: Union[Callable, str] = "default"):
        for _ in range(self.config.neat.population.generation_limit):
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

            if max(fitnesses) >= self.config.neat.population.fitness_threshold:
                print("Fitness limit reached!")
                return self.best_genome

            self.tell(fitnesses)
        print("Generation limit reached!")
        return self.best_genome

    def expand(self):
        """
        Expand the population if needed.
        :return:
        when the maximum node number of the population >= N
        the population will expand
        """
        pop_node_keys = self.pop_nodes[:, :, 0]
        pop_node_sizes = np.sum(~np.isnan(pop_node_keys), axis=1)
        max_node_size = np.max(pop_node_sizes)
        if max_node_size >= self.N:
            self.N = int(self.N * self.expand_coe)
            # self.C = int(self.C * self.expand_coe)
            print(f"node expand to {self.N}!")
            self.pop_nodes, self.pop_cons = expand(self.pop_nodes, self.pop_cons, self.N, self.C)

            # don't forget to expand representation genome in species
            for s in self.species_controller.species.values():
                s.representative = expand_single(*s.representative, self.N, self.C)


        pop_con_keys = self.pop_cons[:, :, 0]
        pop_node_sizes = np.sum(~np.isnan(pop_con_keys), axis=1)
        max_con_size = np.max(pop_node_sizes)
        if max_con_size >= self.C:
            # self.N = int(self.N * self.expand_coe)
            self.C = int(self.C * self.expand_coe)
            print(f"connections expand to {self.C}!")
            self.pop_nodes, self.pop_cons = expand(self.pop_nodes, self.pop_cons, self.N, self.C)

            # don't forget to expand representation genome in species
            for s in self.species_controller.species.values():
                s.representative = expand_single(*s.representative, self.N, self.C)

        self.create_and_speciate = self.function_factory.create_update_speciate(self.N, self.C, self.S)

        

    def default_analysis(self, fitnesses):
        max_f, min_f, mean_f, std_f = max(fitnesses), min(fitnesses), np.mean(fitnesses), np.std(fitnesses)
        species_sizes = [len(s.members) for s in self.species_controller.species.values()]

        new_timestamp = time.time()
        cost_time = new_timestamp - self.generation_timestamp
        self.generation_time_list.append(cost_time)
        self.generation_timestamp = new_timestamp

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_genome = (self.pop_nodes[max_idx], self.pop_cons[max_idx])

        print(f"Generation: {self.generation}",
              f"fitness: {max_f}, {min_f}, {mean_f}, {std_f}, Species sizes: {species_sizes}, Cost time: {cost_time}")
