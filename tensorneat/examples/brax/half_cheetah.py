import jax

from pipeline import Pipeline
from algorithm.neat import *

from problem.rl_env import BraxEnv
from utils import Act


def sample_policy(randkey, obs):
    return jax.random.uniform(randkey, (6,), minval=-1, maxval=1)


if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=17,
                    num_outputs=6,
                    max_nodes=50,
                    max_conns=100,
                    node_gene=DefaultNodeGene(
                        activation_options=(Act.tanh,),
                        activation_default=Act.tanh,
                    ),
                    output_transform=Act.tanh,
                ),
                pop_size=1000,
                species_size=10,
            ),
        ),
        problem=BraxEnv(
            env_name="halfcheetah",
            max_step=1000,
            obs_normalization=True,
            sample_episodes=1000,
            sample_policy=sample_policy,
        ),
        generation_limit=10000,
        fitness_target=5000,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
