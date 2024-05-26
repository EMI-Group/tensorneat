import jax.numpy as jnp

from pipeline import Pipeline
from algorithm.neat import *

from problem.rl_env import GymNaxEnv

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=2,
                    num_outputs=3,
                    max_nodes=50,
                    max_conns=100,
                    output_transform=lambda out: jnp.argmax(
                        out
                    ),  # the action of mountain car is {0, 1, 2}
                ),
                pop_size=10000,
                species_size=10,
            ),
        ),
        problem=GymNaxEnv(
            env_name="MountainCar-v0",
        ),
        generation_limit=10000,
        fitness_target=0,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
