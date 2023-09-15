from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.rl_env import GymNaxConfig, GymNaxEnv


def example_conf():
    return Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=500,
            pop_size=10000
        ),
        neat=NeatConfig(
            inputs=8,
            outputs=2,
        ),
        gene=NormalGeneConfig(
            activation_default=Act.sigmoid,
            activation_options=(Act.sigmoid,),
        ),
        problem=GymNaxConfig(
            env_name='Reacher-misc',
        )
    )


if __name__ == '__main__':
    conf = example_conf()

    algorithm = NEAT(conf, NormalGene)
    pipeline = Pipeline(conf, algorithm, GymNaxEnv)
    state = pipeline.setup()
    pipeline.pre_compile(state)
    state, best = pipeline.auto_run(state)
