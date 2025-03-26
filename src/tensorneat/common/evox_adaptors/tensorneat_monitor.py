import warnings
import os
import time
import numpy as np

import jax
from jax.experimental import io_callback
from evox import Monitor
from evox import State as EvoXState

from tensorneat.algorithm import BaseAlgorithm as TensorNEATAlgorithm
from tensorneat.common import State as TensorNEATState


class TensorNEATMonitor(Monitor):

    def __init__(
        self,
        tensorneat_algorithm: TensorNEATAlgorithm,
        save_dir: str = None,
        is_save: bool = False,
    ):
        super().__init__()
        self.tensorneat_algorithm = tensorneat_algorithm

        self.generation_timestamp = time.time()
        self.alg_state: TensorNEATState = None
        self.fitness = None
        self.best_fitness = -np.inf
        self.best_genome = None

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

    def clear_history(self):
        self.alg_state: TensorNEATState = None
        self.fitness = None
        self.best_fitness = -np.inf
        self.best_genome = None

    def hooks(self):
        return ["pre_tell"]

    def pre_tell(self, monitor_state, workflow_state, transformed_fitness):
        io_callback(
            self.store_info,
            None,
            workflow_state,
            transformed_fitness,
        )
        return monitor_state 

    def store_info(self, state: EvoXState, fitness):
        self.alg_state: TensorNEATState = state.query_state("algorithm").alg_state
        self.fitness = jax.device_get(fitness)

    def show(self):
        io_callback(
            self._show,
            None
        )

    def _show(self):
        pop = self.tensorneat_algorithm.ask(self.alg_state)
        generation = int(self.alg_state.generation)

        valid_fitnesses = self.fitness[~np.isinf(self.fitness)]

        max_f, min_f, mean_f, std_f = (
            max(valid_fitnesses),
            min(valid_fitnesses),
            np.mean(valid_fitnesses),
            np.std(valid_fitnesses),
        )

        new_timestamp = time.time()

        cost_time = new_timestamp - self.generation_timestamp
        self.generation_timestamp = time.time()
        
        max_idx = np.argmax(self.fitness)
        if self.fitness[max_idx] > self.best_fitness:
            self.best_fitness = self.fitness[max_idx]
            self.best_genome = pop[0][max_idx], pop[1][max_idx]

        if self.is_save:
            # save best
            best_genome = jax.device_get(self.best_genome)
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

        self.tensorneat_algorithm.show_details(self.alg_state, self.fitness)
