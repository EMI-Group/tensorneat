import time

import jax.numpy as jnp
from config import *
from pipeline_jitable_env import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.rl_env import GymNaxConfig, GymNaxEnv


def conf_with_seed(seed):
    return Config(
        basic=BasicConfig(
            seed=seed,
            fitness_target=501,
            pop_size=10000,
            generation_limit=100
        ),
        neat=NeatConfig(
            inputs=4,
            outputs=1,
            max_species=10000
        ),
        gene=NormalGeneConfig(
            activation_default=Act.sigmoid,
            activation_options=(Act.sigmoid,),
        ),
        problem=GymNaxConfig(
            env_name='CartPole-v1',
            output_transform=lambda out: jnp.where(out[0] > 0.5, 1, 0)  # the action of cartpole is {0, 1}
        )
    )


if __name__ == '__main__':

    times = []

    for seed in range(10):
        conf = conf_with_seed(seed)
        algorithm = NEAT(conf, NormalGene)
        pipeline = Pipeline(conf, algorithm, GymNaxEnv)
        state = pipeline.setup()
        pipeline.pre_compile(state)
        tic = time.time()
        state, best = pipeline.auto_run(state)
        time_cost = time.time() - tic
        times.append(time_cost)
        print(times)

    print(f"total_times: {times}")


