import time
from typing import Union, Callable

import jax
from jax import vmap, jit
import numpy as np

from config import Config
from core import Algorithm, Genome


class Pipeline:
    """
    Simple pipeline.
    """

    def __init__(self, config: Config, algorithm: Algorithm):
        self.config = config
        self.algorithm = algorithm

        randkey = jax.random.PRNGKey(config.basic.seed)
        self.state = algorithm.setup(randkey)

        self.best_genome = None
        self.best_fitness = float('-inf')
        self.generation_timestamp = time.time()

        self.evaluate_time = 0

        self.act_func = jit(self.algorithm.act)
        self.batch_act_func = jit(vmap(self.act_func, in_axes=(None, 0, None)))
        self.pop_batch_act_func = jit(vmap(self.batch_act_func, in_axes=(None, None, 0)))
        self.forward_transform_func = jit(vmap(self.algorithm.forward_transform, in_axes=(None, 0)))
        self.tell_func = jit(self.algorithm.tell)

    def ask(self):
        pop_transforms = self.forward_transform_func(self.state, self.state.pop_genomes)
        return lambda inputs: self.pop_batch_act_func(self.state, inputs, pop_transforms)

    def tell(self, fitness):
        # self.state = self.tell_func(self.state, fitness)
        new_state = self.tell_func(self.state, fitness)
        self.state = new_state

    def auto_run(self, fitness_func, analysis: Union[Callable, str] = "default"):
        for _ in range(self.config.basic.generation_limit):
            forward_func = self.ask()

            fitnesses = fitness_func(forward_func)

            if analysis is not None:
                if analysis == "default":
                    self.default_analysis(fitnesses)
                else:
                    assert callable(analysis), f"What the fuck you passed in? A {analysis}?"
                    analysis(fitnesses)

            if max(fitnesses) >= self.config.basic.fitness_target:
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
            self.best_genome = Genome(self.state.pop_genomes.nodes[max_idx], self.state.pop_genomes.conns[max_idx])

        member_count = jax.device_get(self.state.species_info.member_count)
        species_sizes = [int(i) for i in member_count if i > 0]

        print(f"Generation: {self.state.generation}",
              f"species: {len(species_sizes)}, {species_sizes}",
              f"fitness: {max_f:.6f}, {min_f:.6f}, {mean_f:.6f}, {std_f:.6f}, Cost time: {cost_time * 1000:.6f}ms")