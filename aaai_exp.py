from typing import Callable
from time import time

import jax
from jax import numpy as jnp, vmap, jit
import gymnax
import numpy as np

from config import *
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.rl_env import GymNaxConfig, GymNaxEnv


def conf_cartpole():
    return Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=500,
            generation_limit=150,
            pop_size=10000
        ),
        neat=NeatConfig(
            inputs=3,
            outputs=1,
        ),
        gene=NormalGeneConfig(
            activation_default=Act.tanh,
            activation_options=(Act.tanh,),
        ),
        problem=GymNaxConfig(
            env_name='Pendulum-v1',
            output_transform=lambda out: out * 2  # the action of pendulum is [-2, 2]
        )
    )


def batch_evaluate(
        key,
        alg_state,
        genomes,
        env_params,
        batch_transform: Callable,
        batch_act: Callable,
        batch_reset: Callable,
        batch_step: Callable,
):
    alg_time, env_time, forward_time = 0, 0, 0
    pop_size = genomes.nodes.shape[0]

    alg_tic = time()
    genomes_transform = batch_transform(alg_state, genomes)
    alg_time += time() - alg_tic

    reset_keys = jax.random.split(key, pop_size)
    observations, states = batch_reset(reset_keys, env_params)

    done = np.zeros(pop_size, dtype=bool)
    fitnesses = np.zeros(pop_size)

    while not np.all(done):
        key, _ = jax.random.split(key)
        vmap_keys = jax.random.split(key, pop_size)

        forward_tic = time()
        actions = batch_act(alg_state, observations, genomes_transform).block_until_ready()
        forward_time += time() - forward_tic

        env_tic = time()
        observations, states, reward, current_done, _ = batch_step(vmap_keys, states, actions, env_params)
        reward, current_done = jax.device_get([reward, current_done])
        env_time += time() - env_tic

        fitnesses += reward * np.logical_not(done)
        done = np.logical_or(done, current_done)

    return fitnesses, alg_time, env_time, forward_time


def main():
    conf = conf_cartpole()
    algorithm = NEAT(conf, NormalGene)

    def act(state, inputs, genome):
        res = algorithm.act(state, inputs, genome)
        return conf.problem.output_transform(res)

    batch_transform = jit(vmap(algorithm.transform, in_axes=(None, 0)))
    # (state, obs, genome_transform) -> action
    batch_act = jit(vmap(act, in_axes=(None, 0, 0)))

    env, env_params = gymnax.make(conf.problem.env_name)
    # (seed, params) -> (ini_obs, ini_state)
    batch_reset = jit(vmap(env.reset, in_axes=(0, None)))
    # (seed, state, action, params) -> (obs, state, reward, done, info)
    batch_step = jit(vmap(env.step, in_axes=(0, 0, 0, None)))

    key = jax.random.PRNGKey(conf.basic.seed)
    alg_key, pro_key = jax.random.split(key)
    alg_state = algorithm.setup(alg_key)

    for i in range(conf.basic.generation_limit):

        total_tic = time()

        pro_key, _ = jax.random.split(pro_key)

        fitnesses, a1, env_time, forward_time = batch_evaluate(
            pro_key,
            alg_state,
            algorithm.ask(alg_state),
            env_params,
            batch_transform,
            batch_act,
            batch_reset,
            batch_step
        )
        alg_tic = time()
        alg_state = algorithm.tell(alg_state, fitnesses)
        alg_state = jax.tree_map(lambda x: x.block_until_ready(), alg_state)
        a2 = time() - alg_tic

        alg_time = a1 + a2
        total_time = time() - total_tic

        print(f"generation: {i}, alg_time: {alg_time:.2f}, env_time: {env_time:.2f}, forward_time: {forward_time:.2f}, total_time: {total_time: .2f}, "
              f"max_fitness: {np.max(fitnesses):.2f}", f"avg_fitness: {np.mean(fitnesses):.2f}")


if __name__ == '__main__':
    main()
