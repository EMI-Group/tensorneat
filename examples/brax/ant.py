import jax.numpy as jnp

from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.rl_env import BraxEnv, BraxConfig


def example_conf():
    return Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=10000,
            pop_size=1000
        ),
        neat=NeatConfig(
            inputs=27,
            outputs=8,
        ),
        gene=NormalGeneConfig(
            activation_default=Act.tanh,
            activation_options=(Act.tanh,),
        ),
        problem=BraxConfig(
            env_name="ant"
        )
    )


if __name__ == '__main__':
    conf = example_conf()

    algorithm = NEAT(conf, NormalGene)
    pipeline = Pipeline(conf, algorithm, BraxEnv)
    state = pipeline.setup()
    pipeline.pre_compile(state)
    state, best = pipeline.auto_run(state)
