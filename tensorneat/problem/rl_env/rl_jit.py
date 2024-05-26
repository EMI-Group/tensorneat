from functools import partial

import jax

from .. import BaseProblem


class RLEnv(BaseProblem):
    jitable = True

    def __init__(self, max_step=1000):
        super().__init__()
        self.max_step = max_step

    def evaluate(self, state, randkey, act_func, params):
        rng_reset, rng_episode = jax.random.split(randkey)
        init_obs, init_env_state = self.reset(rng_reset)

        def cond_func(carry):
            _, _, _, done, _, count = carry
            return ~done & (count < self.max_step)

        def body_func(carry):
            obs, env_state, rng, done, tr, count = carry  # tr -> total reward
            action = act_func(state, obs, params)
            next_obs, next_env_state, reward, done, _ = self.step(
                rng, env_state, action
            )
            next_rng, _ = jax.random.split(rng)
            return next_obs, next_env_state, next_rng, done, tr + reward, count + 1

        _, _, _, _, total_reward, _ = jax.lax.while_loop(
            cond_func, body_func, (init_obs, init_env_state, rng_episode, False, 0.0, 0)
        )

        return total_reward

    # @partial(jax.jit, static_argnums=(0,))
    def step(self, randkey, env_state, action):
        return self.env_step(randkey, env_state, action)

    # @partial(jax.jit, static_argnums=(0,))
    def reset(self, randkey):
        return self.env_reset(randkey)

    def env_step(self, randkey, env_state, action):
        raise NotImplementedError

    def env_reset(self, randkey):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        raise NotImplementedError
