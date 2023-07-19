import time
from typing import Union, Callable

import jax
from jax import vmap, jit
import numpy as np


class Pipeline:
    """
    Neat algorithm pipeline.
    """

    def __init__(self, config, algorithm):
        self.config = config
        self.algorithm = algorithm
        randkey = jax.random.PRNGKey(config['random_seed'])
        self.state = algorithm.setup(randkey)

        self.best_genome = None
        self.best_fitness = float('-inf')
        self.generation_timestamp = time.time()

        self.evaluate_time = 0

        self.forward_func = algorithm.gene_type.create_forward(config)
        self.batch_forward_func = jit(vmap(self.forward_func, in_axes=(0, None)))
        self.pop_batch_forward_func = jit(vmap(self.batch_forward_func, in_axes=(None, 0)))

        self.pop_transform_func = jit(vmap(algorithm.gene_type.forward_transform))

    def ask(self):
        pop_transforms = self.pop_transform_func(self.state.pop_nodes, self.state.pop_conns)
        return lambda inputs: self.pop_batch_forward_func(inputs, pop_transforms)

    def tell(self, fitness):
        self.state = self.algorithm.step(self.state, fitness)

    def auto_run(self, fitness_func, analysis: Union[Callable, str] = "default"):
        for _ in range(self.config['generation_limit']):
            forward_func = self.ask()

            fitnesses = fitness_func(forward_func)

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
            self.best_genome = (self.state.pop_nodes[max_idx], self.state.pop_conns[max_idx])

        member_count = jax.device_get(self.state.species_info[:, 3])
        species_sizes = [int(i) for i in member_count if i > 0]

        print(f"Generation: {self.state.generation}",
              f"species: {len(species_sizes)}, {species_sizes}",
              f"fitness: {max_f}, {min_f}, {mean_f}, {std_f}, Cost time: {cost_time}")
