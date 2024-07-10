import jax
from problem.rl_env import BraxEnv


def random_policy(randkey, forward_func, obs):
    return jax.random.uniform(randkey, (6,), minval=-1, maxval=1)


if __name__ == "__main__":
    problem = BraxEnv(env_name="walker2d", max_step=1000, action_policy=random_policy)
    state = problem.setup()
    randkey = jax.random.key(0)
    problem.show(
        state,
        randkey,
        act_func=lambda state, params, obs: obs,
        params=None,
        save_path="walker2d_random_policy",
    )
