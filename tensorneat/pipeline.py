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

        # print(self.problem.input_shape, self.problem.output_shape)

        # TODO: make each algorithm's input_num and output_num
        assert algorithm.num_inputs == self.problem.input_shape[-1], \
            f"algorithm input shape is {algorithm.num_inputs} but problem input shape is {self.problem.input_shape}"

        self.best_genome = None
        self.best_fitness = float('-inf')
        self.generation_timestamp = None

    def setup(self, state=State()):
        state = state.register(randkey=jax.random.PRNGKey(self.seed))
        state = self.algorithm.setup(state)
        state = self.problem.setup(state)
        return state

    def step(self, state):
        randkey_, randkey = jax.random.split(state.randkey)
        keys = jax.random.split(randkey_, self.pop_size)

        state, pop = self.algorithm.ask(state)

        state, pop_transformed = jax.vmap(self.algorithm.transform, in_axes=(None, 0), out_axes=(None, 0))(state, pop)

        state, fitnesses = jax.vmap(self.problem.evaluate, in_axes=(0, None, None, 0), out_axes=(None, 0))(
            keys,
            state,
            self.algorithm.forward,
            pop_transformed
        )

        state = self.algorithm.tell(state, fitnesses)

        return state.update(randkey=randkey), fitnesses

    def auto_run(self, state):
        print("start compile")
        tic = time.time()
        compiled_step = jax.jit(self.step).lower(state).compile()
        print(f"compile finished, cost time: {time.time() - tic:.6f}s", )

        for _ in range(self.generation_limit):

            self.generation_timestamp = time.time()

            state, previous_pop = self.algorithm.ask(state)

            state, fitnesses = compiled_step(state)

            fitnesses = jax.device_get(fitnesses)

            self.analysis(state, previous_pop, fitnesses)

            if max(fitnesses) >= self.fitness_target:
                print("Fitness limit reached!")
                return state, self.best_genome

            # node = previous_pop[0][0][:, 0]
            # node_count = jnp.sum(~jnp.isnan(node))
            # conn = previous_pop[1][0][:, 0]
            # conn_count = jnp.sum(~jnp.isnan(conn))
            # if (w % 5 == 0):
            #     print("node_count", node_count)
            #     print("conn_count", conn_count)

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

        member_count = jax.device_get(self.algorithm.member_count(state))
        species_sizes = [int(i) for i in member_count if i > 0]

        print(f"Generation: {self.algorithm.generation(state)}",
              f"species: {len(species_sizes)}, {species_sizes}",
              f"fitness: {max_f:.6f}, {min_f:.6f}, {mean_f:.6f}, {std_f:.6f}, Cost time: {cost_time * 1000:.6f}ms")

    def show(self, state, best, *args, **kwargs):
        state, transformed = self.algorithm.transform(state, best)
        self.problem.show(state.randkey, state, self.algorithm.forward, transformed, *args, **kwargs)
