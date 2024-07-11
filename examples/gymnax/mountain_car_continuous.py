import jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode

from tensorneat.problem.rl import GymNaxEnv
from tensorneat.common import ACT, AGG



if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=1000,
            species_size=20,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            genome=DefaultGenome(
                num_inputs=2,
                num_outputs=1,
                init_hidden_layers=(),
                node_gene=BiasNode(
                    activation_options=ACT.tanh,
                    aggregation_options=AGG.sum,
                ),
                output_transform=ACT.tanh,
            ),
        ),
        problem=GymNaxEnv(
            env_name="MountainCarContinuous-v0",
            repeat_times=5,
        ),
        seed=42,
        generation_limit=100,
        fitness_target=99,
    )

    # initialize state
    state = pipeline.setup()

    # run until terminate
    state, best = pipeline.auto_run(state)
