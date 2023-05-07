from typing import List, Union, Tuple, Callable
import time

import jax
import numpy as np

from .species import SpeciesController
from .genome import create_initialize_function, create_mutate_function, create_forward_function
from .genome import create_crossover_function
from .genome import expand, expand_single


class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config, seed=42):
        self.randkey = jax.random.PRNGKey(seed)

        self.config = config
        self.N = config.basic.init_maximum_nodes
        self.expand_coe = config.basic.expands_coe
        self.pop_size = config.neat.population.pop_size

        self.species_controller = SpeciesController(config)
        self.initialize_func = create_initialize_function(config)
        self.pop_nodes, self.pop_connections, self.input_idx, self.output_idx = self.initialize_func()
        self.mutate_func = create_mutate_function(config, self.input_idx, self.output_idx, batch=True)
        self.crossover_func = create_crossover_function(batch=True)

        self.generation = 0

        self.species_controller.speciate(self.pop_nodes, self.pop_connections, self.generation)

        self.new_node_keys_pool: List[int] = [max(self.output_idx) + 1]

        self.generation_timestamp = time.time()
        self.best_fitness = float('-inf')

    def ask(self, batch: bool):
        """
        Create a forward function for the population.
        :param batch:
        :return:
        Algorithm gives the population a forward function, then environment gives back the fitnesses.
        """
        func = create_forward_function(self.pop_nodes, self.pop_connections, self.N, self.input_idx, self.output_idx,
                                       batch=batch)
        return func

    def tell(self, fitnesses):

        self.generation += 1

        self.species_controller.update_species_fitnesses(fitnesses)

        crossover_pair = self.species_controller.reproduce(self.generation)

        self.update_next_generation(crossover_pair)

        self.species_controller.speciate(self.pop_nodes, self.pop_connections, self.generation)

        self.expand()

    def auto_run(self, fitness_func, analysis: Union[Callable, str] = "default"):
        for _ in range(self.config.neat.population.generation_limit):
            forward_func = self.ask(batch=True)
            fitnesses = fitness_func(forward_func)

            if analysis is not None:
                if analysis == "default":
                    self.default_analysis(fitnesses)
                else:
                    assert callable(analysis), f"What the fuck you passed in? A {analysis}?"
                    analysis(fitnesses)

            self.tell(fitnesses)
        print("Generation limit reached!")

    def update_next_generation(self, crossover_pair: List[Union[int, Tuple[int, int]]]) -> None:
        """
        create the next generation
        :param crossover_pair: created from self.reproduce()
        """

        assert self.pop_nodes.shape[0] == self.pop_size
        k1, k2, self.randkey = jax.random.split(self.randkey, 3)

        # crossover
        # prepare elitism mask and crossover pair
        elitism_mask = np.full(self.pop_size, False)
        for i, pair in enumerate(crossover_pair):
            if not isinstance(pair, tuple):  # elitism
                elitism_mask[i] = True
                crossover_pair[i] = (pair, pair)
        crossover_pair = np.array(crossover_pair)

        crossover_rand_keys = jax.random.split(k1, self.pop_size)

        # batch crossover
        wpn = self.pop_nodes[crossover_pair[:, 0]]  # winner pop nodes
        wpc = self.pop_connections[crossover_pair[:, 0]]  # winner pop connections
        lpn = self.pop_nodes[crossover_pair[:, 1]]  # loser pop nodes
        lpc = self.pop_connections[crossover_pair[:, 1]]  # loser pop connections
        npn, npc = self.crossover_func(crossover_rand_keys, wpn, wpc, lpn, lpc)  # new pop nodes, new pop connections

        # mutate
        mutate_rand_keys = jax.random.split(k2, self.pop_size)
        new_node_keys = np.array(self.fetch_new_node_keys())

        m_npn, m_npc = self.mutate_func(mutate_rand_keys, npn, npc, new_node_keys)  # mutate_new_pop_nodes
        m_npn, m_npc = jax.device_get(m_npn), jax.device_get(m_npc)
        # elitism don't mutate
        # (pop_size, ) to (pop_size, 1, 1)
        self.pop_nodes = np.where(elitism_mask[:, None, None], npn, m_npn)
        # (pop_size, ) to (pop_size, 1, 1, 1)
        self.pop_connections = np.where(elitism_mask[:, None, None, None], npc, m_npc)
        # print(pop_analysis(self.pop_nodes, self.pop_connections, self.input_idx, self.output_idx))

        # recycle unused node keys
        unused = []
        for i, nodes in enumerate(self.pop_nodes):
            node_keys, key = nodes[:, 0], new_node_keys[i]
            if not np.isin(key, node_keys):  # the new node key is not used
                unused.append(key)
        self.new_node_keys_pool = unused + self.new_node_keys_pool

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
            print(f"expand to {self.N}!")
            self.pop_nodes, self.pop_connections = expand(self.pop_nodes, self.pop_connections, self.N)

            # don't forget to expand representation genome in species
            for s in self.species_controller.species.values():
                s.representative = expand_single(*s.representative, self.N)

    def fetch_new_node_keys(self):
        # if remain unused keys are not enough, create new keys
        if len(self.new_node_keys_pool) < self.pop_size:
            max_unused_key = max(self.new_node_keys_pool) if self.new_node_keys_pool else -1
            new_keys = list(range(max_unused_key + 1, max_unused_key + 1 + 10 * self.pop_size))
            self.new_node_keys_pool.extend(new_keys)

        # fetch keys from pool
        res = self.new_node_keys_pool[:self.pop_size]
        self.new_node_keys_pool = self.new_node_keys_pool[self.pop_size:]
        return res

    def default_analysis(self, fitnesses):
        max_f, min_f, mean_f, std_f = max(fitnesses), min(fitnesses), np.mean(fitnesses), np.std(fitnesses)
        species_sizes = [len(s.members) for s in self.species_controller.species.values()]

        new_timestamp = time.time()
        cost_time = new_timestamp - self.generation_timestamp
        self.generation_timestamp = new_timestamp

        print(f"Generation: {self.generation}",
              f"fitness: {max_f}, {min_f}, {mean_f}, {std_f}, Species sizes: {species_sizes}, Cost time: {cost_time}")
