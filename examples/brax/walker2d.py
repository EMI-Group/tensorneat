from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode

from tensorneat.problem.rl import BraxEnv
from tensorneat.common import ACT, AGG

import jax, jax.numpy as jnp


def random_sample_policy(randkey, obs):
    return jax.random.uniform(randkey, (6,), minval=-1.0, maxval=1.0)


if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=1000,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            genome=DefaultGenome(
                max_nodes=100,
                max_conns=200,
                num_inputs=17,
                num_outputs=6,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=ACT.tanh,
                    aggregation_options=AGG.sum,
                ),
                output_transform=ACT.standard_tanh,
            ),
        ),
        problem=BraxEnv(
            env_name="walker2d",
            max_step=1000,
            obs_normalization=True,
            sample_episodes=1000,
            sample_policy=random_sample_policy,
        ),
        seed=42,
        generation_limit=100,
        fitness_target=5000,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
