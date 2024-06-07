import jax, jax.numpy as jnp
import time
import numpy as np

from algorithm import BaseAlgorithm
from problem import BaseProblem
from problem.rl_env import RLEnv
from problem.func_fit import FuncFit
from utils import State


class Pipeline:
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        problem: BaseProblem,
        seed: int = 42,
        fitness_target: float = 1,
        generation_limit: int = 1000,
        pre_update: bool = False,
        update_batch_size: int = 10000,
        save_path=None,
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
        assert (
            algorithm.num_inputs == self.problem.input_shape[-1]
        ), f"algorithm input shape is {algorithm.num_inputs} but problem input shape is {self.problem.input_shape}"

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.generation_timestamp = None
        self.pre_update = pre_update
        self.update_batch_size = update_batch_size
        if pre_update:
            if isinstance(problem, RLEnv):
                assert problem.record_episode, "record_episode must be True"
                self.fetch_data = lambda episode: episode["obs"]
            elif isinstance(problem, FuncFit):
                assert problem.return_data, "return_data must be True"
                self.fetch_data = lambda data: data
            else:
                raise NotImplementedError
        else:
            if isinstance(problem, RLEnv):
                assert not problem.record_episode, "record_episode must be False"
            elif isinstance(problem, FuncFit):
                assert not problem.return_data, "return_data must be False"
        self.save_path = save_path

    def setup(self, state=State()):
        print("initializing")
        state = state.register(randkey=jax.random.PRNGKey(self.seed))

        if self.pre_update:
            # initial with mean = 0 and std = 1
            state = state.register(
                data=jax.random.normal(
                    state.randkey, (self.update_batch_size, self.algorithm.num_inputs)
                )
            )

        state = self.algorithm.setup(state)
        state = self.problem.setup(state)
        print("initializing finished")
        return state

    def step(self, state):

        randkey_, randkey = jax.random.split(state.randkey)
        keys = jax.random.split(randkey_, self.pop_size)

        pop = self.algorithm.ask(state)

        pop_transformed = jax.vmap(self.algorithm.transform, in_axes=(None, 0))(
            state, pop
        )

        if self.pre_update:
            # update the population
            _, pop_transformed = jax.vmap(
                self.algorithm.update_by_batch, in_axes=(None, None, 0)
            )(state, state.data, pop_transformed)

            # raw_data: (Pop, Batch, num_inputs)
            fitnesses, raw_data = jax.vmap(
                self.problem.evaluate, in_axes=(None, 0, None, 0)
            )(state, keys, self.algorithm.forward, pop_transformed)

            # update population
            pop_nodes, pop_conns = jax.vmap(self.algorithm.restore, in_axes=(None, 0))(
                state, pop_transformed
            )
            state = state.update(pop_nodes=pop_nodes, pop_conns=pop_conns)

            # update data for next generation
            data = self.fetch_data(raw_data)
            assert (
                data.ndim == 3
                and data.shape[0] == self.pop_size
                and data.shape[2] == self.algorithm.num_inputs
            )
            # reshape to (Pop * Batch, num_inputs)
            data = data.reshape(
                data.shape[0] * data.shape[1], self.algorithm.num_inputs
            )
            # shuffle
            data = jax.random.permutation(randkey_, data, axis=0)
            # cutoff or expand
            if data.shape[0] >= self.update_batch_size:
                data = data[: self.update_batch_size]  # cutoff
            else:
                data = (
                    jnp.full(state.data.shape, jnp.nan).at[: data.shape[0]].set(data)
                )  # expand
            state = state.update(data=data)

        else:
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
                return state, self.best_genome

        print("Generation limit reached!")
        return state, self.best_genome

    def analysis(self, state, pop, fitnesses):

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

        # save best if save path is not None
        if self.save_path is not None:
            best_genome = jax.device_get(self.best_genome)
            with open(self.save_path, "wb") as f:
                np.savez(
                    f,
                    nodes=best_genome[0],
                    conns=best_genome[1],
                    fitness=self.best_fitness,
                )

        member_count = jax.device_get(self.algorithm.member_count(state))
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
            f"Generation: {self.algorithm.generation(state)}, Cost time: {cost_time * 1000:.2f}ms\n",
            f"\tnode counts: max: {max_node_cnt}, min: {min_node_cnt}, mean: {mean_node_cnt:.2f}\n",
            f"\tconn counts: max: {max_conn_cnt}, min: {min_conn_cnt}, mean: {mean_conn_cnt:.2f}\n",
            f"\tspecies: {len(species_sizes)}, {species_sizes}\n",
            f"\tfitness: valid cnt: {len(valid_fitnesses)}, max: {max_f:.4f}, min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n",
        )

    def show(self, state, best, *args, **kwargs):
        transformed = self.algorithm.transform(state, best)
        self.problem.show(
            state, state.randkey, self.algorithm.forward, transformed, *args, **kwargs
        )
