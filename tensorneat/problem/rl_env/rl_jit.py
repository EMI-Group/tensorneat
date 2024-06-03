from functools import partial

import jax
import jax.numpy as jnp

from .. import BaseProblem


class RLEnv(BaseProblem):
    jitable = True

    def __init__(self, max_step=1000, record_episode=False):
        super().__init__()
        self.max_step = max_step
        self.record_episode = record_episode

    def evaluate(self, state, randkey, act_func, params):
        rng_reset, rng_episode = jax.random.split(randkey)
        init_obs, init_env_state = self.reset(rng_reset)

        if self.record_episode:
            obs_array = jnp.full((self.max_step, *self.input_shape), jnp.nan)
            action_array = jnp.full((self.max_step, *self.output_shape), jnp.nan)
            reward_array = jnp.full((self.max_step,), jnp.nan)
            episode = {
                "obs": obs_array,
                "action": action_array,
                "reward": reward_array,
            }
        else:
            episode = None

        def cond_func(carry):
            _, _, _, done, _, count, _ = carry
            return ~done & (count < self.max_step)

        def body_func(carry):
            obs, env_state, rng, done, tr, count, epis = carry  # tr -> total reward
            action = act_func(state, params, obs)
            next_obs, next_env_state, reward, done, _ = self.step(
                rng, env_state, action
            )
            next_rng, _ = jax.random.split(rng)

            if self.record_episode:
                epis["obs"] = epis["obs"].at[count].set(obs)
                epis["action"] = epis["action"].at[count].set(action)
                epis["reward"] = epis["reward"].at[count].set(reward)

            return (
                next_obs,
                next_env_state,
                next_rng,
                done,
                tr + reward,
                count + 1,
                epis,
            )

        _, _, _, _, total_reward, _, episode = jax.lax.while_loop(
            cond_func,
            body_func,
            (init_obs, init_env_state, rng_episode, False, 0.0, 0, episode),
        )

        if self.record_episode:
            return total_reward, episode
        else:
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
