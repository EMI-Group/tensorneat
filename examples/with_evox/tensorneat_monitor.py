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
        neat_algorithm: TensorNEATAlgorithm,
        save_dir: str = None,
        is_save: bool = False,
    ):
        super().__init__()
        self.neat_algorithm = neat_algorithm

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

    def hooks(self):
        return ["pre_tell"]

    def pre_tell(self, state: EvoXState, cand_sol, transformed_cand_sol, fitness, transformed_fitness):
        io_callback(
            self.store_info,
            None,
            state,
            transformed_fitness,
        )

    def store_info(self, state: EvoXState, fitness):
        self.alg_state: TensorNEATState = state.query_state("algorithm").alg_state
        self.fitness = jax.device_get(fitness)

    def show(self):
        pop = self.neat_algorithm.ask(self.alg_state)
        valid_fitnesses = self.fitness[~np.isinf(self.fitness)]

        max_f, min_f, mean_f, std_f = (
            max(valid_fitnesses),
            min(valid_fitnesses),
            np.mean(valid_fitnesses),
            np.std(valid_fitnesses),
        )

        new_timestamp = time.time()

        cost_time = new_timestamp - self.generation_timestamp
        self.generation_timestamp = new_timestamp

        max_idx = np.argmax(self.fitness)
        if self.fitness[max_idx] > self.best_fitness:
            self.best_fitness = self.fitness[max_idx]
            self.best_genome = pop[0][max_idx], pop[1][max_idx]

        if self.is_save:
            best_genome = jax.device_get((pop[0][max_idx], pop[1][max_idx]))
            with open(
                os.path.join(
                    self.genome_dir,
                    f"{int(self.neat_algorithm.generation(self.alg_state))}.npz",
                ),
                "wb",
            ) as f:
                np.savez(
                    f,
                    nodes=best_genome[0],
                    conns=best_genome[1],
                    fitness=self.best_fitness,
                )

        # save best if save path is not None
        member_count = jax.device_get(self.neat_algorithm.member_count(self.alg_state))
        species_sizes = [int(i) for i in member_count if i > 0]

        pop = jax.device_get(pop)
        pop_nodes, pop_conns = pop  # (P, N, NL), (P, C, CL)
        nodes_cnt = (~np.isnan(pop_nodes[:, :, 0])).sum(axis=1)  # (P,)
        conns_cnt = (~np.isnan(pop_conns[:, :, 0])).sum(axis=1)  # (P,)

        max_node_cnt, min_node_cnt, mean_node_cnt = (
            max(nodes_cnt),
            min(nodes_cnt),
            np.mean(nodes_cnt),
        )

        max_conn_cnt, min_conn_cnt, mean_conn_cnt = (
            max(conns_cnt),
            min(conns_cnt),
            np.mean(conns_cnt),
        )

        print(
            f"Generation: {self.neat_algorithm.generation(self.alg_state)}, Cost time: {cost_time * 1000:.2f}ms\n",
            f"\tnode counts: max: {max_node_cnt}, min: {min_node_cnt}, mean: {mean_node_cnt:.2f}\n",
            f"\tconn counts: max: {max_conn_cnt}, min: {min_conn_cnt}, mean: {mean_conn_cnt:.2f}\n",
            f"\tspecies: {len(species_sizes)}, {species_sizes}\n",
            f"\tfitness: valid cnt: {len(valid_fitnesses)}, max: {max_f:.4f}, min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n",
        )

        # append log
        if self.is_save:
            with open(os.path.join(self.save_dir, "log.txt"), "a") as f:
                f.write(
                    f"{self.neat_algorithm.generation(self.alg_state)},{max_f},{min_f},{mean_f},{std_f},{cost_time}\n"
                )
