import jax.numpy as jnp

from pipeline import Pipeline
from algorithm.neat import *

from problem.rl_env import GymNaxEnv

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=8,
                    num_outputs=2,
                    max_nodes=50,
                    max_conns=100,
                ),
                pop_size=10000,
                species_size=10,
            ),
        ),
        problem=GymNaxEnv(
            env_name="Reacher-misc",
        ),
        generation_limit=10000,
        fitness_target=90,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
