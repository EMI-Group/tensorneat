from pipeline import Pipeline
from algorithm.neat import *

from problem.rl_env import BraxEnv
from utils import Act

import jax, jax.numpy as jnp


def split_right_left(randkey, forward_func, obs):
    right_obs_keys = jnp.array([2, 3, 4, 11, 12, 13])
    left_obs_keys = jnp.array([5, 6, 7, 14, 15, 16])
    right_action_keys = jnp.array([0, 1, 2])
    left_action_keys = jnp.array([3, 4, 5])

    right_foot_obs = obs
    left_foot_obs = obs
    left_foot_obs = left_foot_obs.at[right_obs_keys].set(obs[left_obs_keys])
    left_foot_obs = left_foot_obs.at[left_obs_keys].set(obs[right_obs_keys])

    right_action, left_action = jax.vmap(forward_func)(jnp.stack([right_foot_obs, left_foot_obs]))
    # print(right_action.shape)
    # print(left_action.shape)

    return jnp.concatenate([right_action, left_action])


if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=17,
                    num_outputs=3,
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
            env_name="walker2d",
            max_step=1000,
            action_policy=split_right_left
        ),
        generation_limit=10000,
        fitness_target=5000,
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
