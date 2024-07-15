import json
import os

import jax, jax.numpy as jnp
import datetime, time
import numpy as np

from tensorneat.algorithm import BaseAlgorithm
from tensorneat.problem import BaseProblem
from tensorneat.common import State, StatefulBaseClass


class Pipeline(StatefulBaseClass):
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        problem: BaseProblem,
        seed: int = 42,
        fitness_target: float = 1,
        generation_limit: int = 1000,
        is_save: bool = False,
        save_dir=None,
    ):
        assert problem.jitable, "Currently, problem must be jitable"

        self.algorithm = algorithm
        self.problem = problem
        self.seed = seed
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.pop_size = self.algorithm.pop_size

        np.random.seed(self.seed)

        assert (
            algorithm.num_inputs == self.problem.input_shape[-1]
        ), f"algorithm input shape is {algorithm.num_inputs} but problem input shape is {self.problem.input_shape}"

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.generation_timestamp = None
        self.is_save = is_save

        if is_save:
            if save_dir is None:
                now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                self.save_dir = f"./{self.__class__.__name__} {now}"
            else:
                self.save_dir = save_dir
            print(f"save to {self.save_dir}")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.genome_dir = os.path.join(self.save_dir, "genomes")
            if not os.path.exists(self.genome_dir):
                os.makedirs(self.genome_dir)

    def setup(self, state=State()):
        print("initializing")
        state = state.register(randkey=jax.random.PRNGKey(self.seed))

        state = self.algorithm.setup(state)
        state = self.problem.setup(state)

        if self.is_save:
            # self.save(state=state, path=os.path.join(self.save_dir, "pipeline.pkl"))
            with open(os.path.join(self.save_dir, "config.txt"), "w") as f:
                f.write(json.dumps(self.show_config(), indent=4))
            # create log file
            with open(os.path.join(self.save_dir, "log.txt"), "w") as f:
                f.write("Generation,Max,Min,Mean,Std,Cost Time\n")

        print("initializing finished")
        return state

    def step(self, state):

        randkey_, randkey = jax.random.split(state.randkey)
        keys = jax.random.split(randkey_, self.pop_size)

        pop = self.algorithm.ask(state)

        pop_transformed = jax.vmap(self.algorithm.transform, in_axes=(None, 0))(
            state, pop
        )

        fitnesses = jax.vmap(self.problem.evaluate, in_axes=(None, 0, None, 0))(
            state, keys, self.algorithm.forward, pop_transformed
        )

        # replace nan with -inf
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        previous_pop = self.algorithm.ask(state)
        state = self.algorithm.tell(state, fitnesses)

        return state.update(randkey=randkey), previous_pop, fitnesses

    def auto_run(self, state):
        print("start compile")
        tic = time.time()
        compiled_step = jax.jit(self.step).lower(state).compile()
        # compiled_step = self.step
        print(
            f"compile finished, cost time: {time.time() - tic:.6f}s",
        )

        for _ in range(self.generation_limit):

            self.generation_timestamp = time.time()

            state, previous_pop, fitnesses = compiled_step(state)

            fitnesses = jax.device_get(fitnesses)

            self.analysis(state, previous_pop, fitnesses)

            if max(fitnesses) >= self.fitness_target:
                print("Fitness limit reached!")
                break

        if int(state.generation) >= self.generation_limit:
            print("Generation limit reached!")

        if self.is_save:
            best_genome = jax.device_get(self.best_genome)
            with open(os.path.join(self.genome_dir, f"best_genome.npz"), "wb") as f:
                np.savez(
                    f,
                    nodes=best_genome[0],
                    conns=best_genome[1],
                    fitness=self.best_fitness,
                )

        return state, self.best_genome

    def analysis(self, state, pop, fitnesses):
        
        generation = int(state.generation)

        valid_fitnesses = fitnesses[~np.isinf(fitnesses)]

        max_f, min_f, mean_f, std_f = (
            max(valid_fitnesses),
            min(valid_fitnesses),
            np.mean(valid_fitnesses),
            np.std(valid_fitnesses),
        )

        new_timestamp = time.time()

        cost_time = new_timestamp - self.generation_timestamp

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_genome = pop[0][max_idx], pop[1][max_idx]

        if self.is_save:
            # save best
            best_genome = jax.device_get((pop[0][max_idx], pop[1][max_idx]))
            file_name = os.path.join(
                self.genome_dir, f"{generation}.npz"
            )
            with open(file_name, "wb") as f:
                np.savez(
                    f,
                    nodes=best_genome[0],
                    conns=best_genome[1],
                    fitness=self.best_fitness,
                )

            # append log
            with open(os.path.join(self.save_dir, "log.txt"), "a") as f:
                f.write(
                    f"{generation},{max_f},{min_f},{mean_f},{std_f},{cost_time}\n"
                )

        print(
            f"Generation: {generation}, Cost time: {cost_time * 1000:.2f}ms\n",
            f"\tfitness: valid cnt: {len(valid_fitnesses)}, max: {max_f:.4f}, min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n",
        )

        self.algorithm.show_details(state, fitnesses)

    def show(self, state, best, *args, **kwargs):
        transformed = self.algorithm.transform(state, best)
        return self.problem.show(
            state, state.randkey, self.algorithm.forward, transformed, *args, **kwargs
        )
