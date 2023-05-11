from typing import List, Union, Tuple, Callable
import time

import jax
import numpy as np

from .species import SpeciesController
from .genome import expand, expand_single
from .function_factory import FunctionFactory
from .genome.genome import count
from .genome.debug.tools import check_array_valid

class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config, seed=42):
        self.time_dict = {}
        self.function_factory = FunctionFactory(config, debug=True)

        self.randkey = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        self.config = config
        self.N = config.basic.init_maximum_nodes
        self.C = config.basic.init_maximum_connections
        self.expand_coe = config.basic.expands_coe
        self.pop_size = config.neat.population.pop_size

        self.species_controller = SpeciesController(config)
        self.initialize_func = self.function_factory.create_initialize()
        self.pop_nodes, self.pop_connections, self.input_idx, self.output_idx = self.initialize_func()

        self.compile_functions(debug=True)

        self.generation = 0
        self.species_controller.init_speciate(self.pop_nodes, self.pop_connections)

        self.best_fitness = float('-inf')
        self.best_genome = None
        self.generation_timestamp = time.time()

    def ask(self):
        """
        Create a forward function for the population.
        :return:
        Algorithm gives the population a forward function, then environment gives back the fitnesses.
        """
        return self.function_factory.ask_pop_batch_forward(self.pop_nodes, self.pop_connections)

    def tell(self, fitnesses):

        self.generation += 1

        self.species_controller.update_species_fitnesses(fitnesses)

        winner_part, loser_part, elite_mask = self.species_controller.reproduce(fitnesses, self.generation)

        self.update_next_generation(winner_part, loser_part, elite_mask)

        # pop_analysis(self.pop_nodes, self.pop_connections, self.input_idx, self.output_idx)

        self.species_controller.speciate(self.pop_nodes, self.pop_connections, self.generation,
                                         self.o2o_distance, self.o2m_distance)

        self.expand()

    def auto_run(self, fitness_func, analysis: Union[Callable, str] = "default"):
        for _ in range(self.config.neat.population.generation_limit):
            forward_func = self.ask()
            fitnesses = fitness_func(forward_func)

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

    def update_next_generation(self, winner_part, loser_part, elite_mask) -> None:
        """
        create next generation
        :param winner_part:
        :param loser_part:
        :param elite_mask:
        :return:
        """

        assert self.pop_nodes.shape[0] == self.pop_size
        k1, k2, self.randkey = jax.random.split(self.randkey, 3)

        crossover_rand_keys = jax.random.split(k1, self.pop_size)
        mutate_rand_keys = jax.random.split(k2, self.pop_size)

        # batch crossover
        wpn = self.pop_nodes[winner_part]  # winner pop nodes
        wpc = self.pop_connections[winner_part]  # winner pop connections
        lpn = self.pop_nodes[loser_part]  # loser pop nodes
        lpc = self.pop_connections[loser_part]  # loser pop connections

        npn, npc = self.crossover_func(crossover_rand_keys, wpn, wpc, lpn,
                                       lpc)  # new pop nodes, new pop connections

        # for i in range(self.pop_size):
        #     n, c = np.array(npn[i]), np.array(npc[i])
        #     check_array_valid(n, c, self.input_idx, self.output_idx)

        # mutate
        new_node_keys = np.arange(self.generation * self.pop_size, self.generation * self.pop_size + self.pop_size)

        m_npn, m_npc = self.mutate_func(mutate_rand_keys, npn, npc, new_node_keys)  # mutate_new_pop_nodes

        # for i in range(self.pop_size):
        #     n, c = np.array(m_npn[i]), np.array(m_npc[i])
        #     check_array_valid(n, c, self.input_idx, self.output_idx)

        # elitism don't mutate
        npn, npc, m_npn, m_npc = jax.device_get([npn, npc, m_npn, m_npc])

        self.pop_nodes = np.where(elite_mask[:, None, None], npn, m_npn)
        self.pop_connections = np.where(elite_mask[:, None, None], npc, m_npc)

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
            print(f"node expand to {self.N}!")
            self.pop_nodes, self.pop_connections = expand(self.pop_nodes, self.pop_connections, self.N, self.C)

            # don't forget to expand representation genome in species
            for s in self.species_controller.species.values():
                s.representative = expand_single(*s.representative, self.N, self.C)

            # update functions
            self.compile_functions(debug=True)


        pop_con_keys = self.pop_connections[:, :, 0]
        pop_node_sizes = np.sum(~np.isnan(pop_con_keys), axis=1)
        max_con_size = np.max(pop_node_sizes)
        if max_con_size >= self.C:
            self.C = int(self.C * self.expand_coe)
            print(f"connections expand to {self.C}!")
            self.pop_nodes, self.pop_connections = expand(self.pop_nodes, self.pop_connections, self.N, self.C)

            # don't forget to expand representation genome in species
            for s in self.species_controller.species.values():
                s.representative = expand_single(*s.representative, self.N, self.C)

            # update functions
            self.compile_functions(debug=True)



    def compile_functions(self, debug=False):
        self.mutate_func = self.function_factory.create_mutate(self.N, self.C)
        self.crossover_func = self.function_factory.create_crossover(self.N, self.C)
        self.o2o_distance, self.o2m_distance = self.function_factory.create_distance(self.N, self.C)

    def default_analysis(self, fitnesses):
        max_f, min_f, mean_f, std_f = max(fitnesses), min(fitnesses), np.mean(fitnesses), np.std(fitnesses)
        species_sizes = [len(s.members) for s in self.species_controller.species.values()]

        new_timestamp = time.time()
        cost_time = new_timestamp - self.generation_timestamp
        self.generation_timestamp = new_timestamp

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_genome = (self.pop_nodes[max_idx], self.pop_connections[max_idx])

        print(f"Generation: {self.generation}",
              f"fitness: {max_f}, {min_f}, {mean_f}, {std_f}, Species sizes: {species_sizes}, Cost time: {cost_time}")
