import jax

import brax
from brax import envs


def inference_func(key, *args):
    return jax.random.normal(key, shape=(env.action_size,))


env_name = "ant"
backend = "generalized"

env = envs.create(env_name=env_name, backend=backend)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_func)


rollout = []
rng = jax.random.PRNGKey(seed=1)
ori_state = jit_env_reset(rng=rng)
state = ori_state

for _ in range(100):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act = jit_inference_fn(act_rng, state.obs)
    state = jit_env_step(state, act)
    reward = state.reward
    # print(reward)

a = 1


