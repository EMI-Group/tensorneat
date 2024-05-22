from functools import partial

import jax, jax.numpy as jnp
import time
import numpy as np

from algorithm import BaseAlgorithm
from problem import BaseProblem
from utils import State


class Pipeline:

    def __init__(
            self,
            algorithm: BaseAlgorithm,
            problem: BaseProblem,
            seed: int = 42,
            fitness_target: float = 1,
            generation_limit: int = 1000,
    ):
        assert problem.jitable, "Currently, problem must be jitable"

        self.algorithm = algorithm
        self.problem = problem
        self.seed = seed
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.pop_size = self.algorithm.pop_size

        print(self.problem.input_shape, self.problem.output_shape)

        # TODO: make each algorithm's input_num and output_num
        assert algorithm.num_inputs == self.problem.input_shape[-1], \
            f"algorithm input shape is {algorithm.num_inputs} but problem input shape is {self.problem.input_shape}"

        # self.act_func = self.algorithm.act

        # for _ in range(len(self.problem.input_shape) - 1):
        #     self.act_func = jax.vmap(self.act_func, in_axes=(None, 0, None))

        self.best_genome = None
        self.best_fitness = float('-inf')
        self.generation_timestamp = None

    def setup(self):
        key = jax.random.PRNGKey(self.seed)
        key, algorithm_key, evaluate_key = jax.random.split(key, 3)

        # TODO: Problem should has setup function to maintain state
        return State(
            randkey=key,
            alg=self.algorithm.setup(algorithm_key),
            pro=self.problem.setup(evaluate_key),
        )

    def step(self, state):
        key, sub_key = jax.random.split(state.randkey)
        keys = jax.random.split(key, self.pop_size)

        pop = self.algorithm.ask(state.alg)

        pop_transformed = jax.vmap(self.algorithm.transform)(pop)

        fitnesses = jax.vmap(self.problem.evaluate, in_axes=(0, None, None, 0))(
                        keys,
                        state.pro,
                        self.algorithm.forward,
                        pop_transformed
                    )

        # fitnesses = jnp.where(jnp.isnan(fitnesses), -1e6, fitnesses)

        alg_state = self.algorithm.tell(state.alg, fitnesses)

        return state.update(
            randkey=sub_key,
            alg=alg_state,
        ), fitnesses

    def auto_run(self, ini_state):
        state = ini_state
        print("start compile")
        tic = time.time()
        compiled_step = jax.jit(self.step).lower(ini_state).compile()
        print(f"compile finished, cost time: {time.time() - tic:.6f}s", )
        for _ in range(self.generation_limit):

            self.generation_timestamp = time.time()

            previous_pop = self.algorithm.ask(state.alg)

            state, fitnesses = compiled_step(state)

            fitnesses = jax.device_get(fitnesses)

            self.analysis(state, previous_pop, fitnesses)

            if max(fitnesses) >= self.fitness_target:
                print("Fitness limit reached!")
                return state, self.best_genome

        print("Generation limit reached!")
        return state, self.best_genome

    def analysis(self, state, pop, fitnesses):

        max_f, min_f, mean_f, std_f = max(fitnesses), min(fitnesses), np.mean(fitnesses), np.std(fitnesses)

        new_timestamp = time.time()

        cost_time = new_timestamp - self.generation_timestamp

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_genome = pop[0][max_idx], pop[1][max_idx]

        member_count = jax.device_get(self.algorithm.member_count(state.alg))
        species_sizes = [int(i) for i in member_count if i > 0]

        print(f"Generation: {self.algorithm.generation(state.alg)}",
              f"species: {len(species_sizes)}, {species_sizes}",
              f"fitness: {max_f:.6f}, {min_f:.6f}, {mean_f:.6f}, {std_f:.6f}, Cost time: {cost_time * 1000:.6f}ms")

    def show(self, state, best, *args, **kwargs):
        transformed = self.algorithm.transform(best)
        self.problem.show(state.randkey, state.pro, self.algorithm.forward, transformed, *args, **kwargs)
