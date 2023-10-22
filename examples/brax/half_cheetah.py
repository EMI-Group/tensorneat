import jax.numpy as jnp

from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.rl_env import BraxEnv, BraxConfig


# ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']


def example_conf():
    return Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=10000,
            generation_limit=10,
            pop_size=100
        ),
        neat=NeatConfig(
            inputs=17,
            outputs=6,
        ),
        gene=NormalGeneConfig(
            activation_default=Act.tanh,
            activation_options=(Act.tanh,),
        ),
        problem=BraxConfig(
            env_name="halfcheetah"
        )
    )


if __name__ == '__main__':
    conf = example_conf()
    algorithm = NEAT(conf, NormalGene)
    pipeline = Pipeline(conf, algorithm, BraxEnv)
    state = pipeline.setup()
    pipeline.pre_compile(state)
    state, best = pipeline.auto_run(state)
    pipeline.show(state, best, save_path="half_cheetah.gif", )
