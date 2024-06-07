import jax, jax.numpy as jnp
import jax.random
from problem.rl_env.jumanji.jumanji_2048 import Jumanji_2048


def random_policy(state, params, obs):
    # key = jax.random.key(obs.sum())
    # actions = jax.random.normal(key, (4,))
    # actions = actions.at[2:].set(-9999)
    return jnp.array([4, 4, 0, 1])
    # return jnp.array([1, 2, 3, 4])
    return actions


if __name__ == "__main__":
    problem = Jumanji_2048(
        max_step=10000, repeat_times=1000, guarantee_invalid_action=True
    )
    state = problem.setup()
    jit_evaluate = jax.jit(
        lambda state, randkey: problem.evaluate(state, randkey, random_policy, None)
    )
    randkey = jax.random.PRNGKey(0)
    reward = jit_evaluate(state, randkey)
    print(reward)
