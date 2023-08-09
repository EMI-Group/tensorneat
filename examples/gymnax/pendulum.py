import jax.numpy as jnp

from config import *
from pipeline_jitable_env import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.rl_env import GymNaxConfig, GymNaxEnv


def example_conf():
    return Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=0,
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


if __name__ == '__main__':
    conf = example_conf()

    algorithm = NEAT(conf, NormalGene)
    pipeline = Pipeline(conf, algorithm, GymNaxEnv)
    state = pipeline.setup()
    pipeline.pre_compile(state)
    state, best = pipeline.auto_run(state)
