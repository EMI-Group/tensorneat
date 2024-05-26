import jax.numpy as jnp

from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from algorithm.hyperneat import HyperNEAT, NormalSubstrateConfig, NormalSubstrate
from problem.rl_env import GymNaxConfig, GymNaxEnv


def example_conf():
    return Config(
        basic=BasicConfig(seed=42, fitness_target=500, pop_size=10000),
        neat=NeatConfig(
            inputs=4,
            outputs=1,
        ),
        gene=NormalGeneConfig(
            activation_default=Act.tanh,
            activation_options=(Act.tanh,),
        ),
        hyperneat=HyperNeatConfig(activation=Act.sigmoid, inputs=4, outputs=2),
        substrate=NormalSubstrateConfig(
            input_coors=((-1, -1), (-0.5, -1), (0, -1), (0.5, -1), (1, -1)),
            hidden_coors=(
                # (-1, -0.5), (-0.5, -0.5), (0, -0.5), (0.5, -0.5),
                (1, 0),
                (-1, 0),
                (-0.5, 0),
                (0, 0),
                (0.5, 0),
                (1, 0),
                # (1, 0.5), (-1, 0.5), (-0.5, 0.5), (0, 0.5), (0.5, 0.5), (1, 0.5),
            ),
            output_coors=((-1, 1), (1, 1)),
        ),
        problem=GymNaxConfig(
            env_name="CartPole-v1",
            output_transform=lambda out: jnp.argmax(
                out
            ),  # the action of cartpole is {0, 1}
        ),
    )


if __name__ == "__main__":
    conf = example_conf()

    algorithm = HyperNEAT(conf, NormalGene, NormalSubstrate)
    pipeline = Pipeline(conf, algorithm, GymNaxEnv)
    state = pipeline.setup()
    pipeline.pre_compile(state)
    state, best = pipeline.auto_run(state)
