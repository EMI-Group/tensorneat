from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from utils import State
from .. import BaseProblem


class RLEnv(BaseProblem):
    jitable = True

    def __init__(
        self,
        max_step=1000,
        repeat_times=1,
        record_episode=False,
        action_policy: Callable = None,
        obs_normalization: bool = False,
        sample_policy: Callable = None,
        sample_episodes: int = 0,
    ):
        """
        action_policy take three args:
            randkey, forward_func, obs
            randkey is a random key for jax.random
            forward_func is a function which receive obs and return action forward_func(obs) - > action
            obs is the observation of the environment

        sample_policy take two args:
            randkey, obs -> action
        """

        super().__init__()
        self.max_step = max_step
        self.record_episode = record_episode
        self.repeat_times = repeat_times
        self.action_policy = action_policy

        if obs_normalization:
            assert sample_policy is not None, "sample_policy must be provided"
            assert sample_episodes > 0, "sample_size must be greater than 0"
            self.sample_policy = sample_policy
            self.sample_episodes = sample_episodes
        self.obs_normalization = obs_normalization

    def setup(self, state=State()):
        if self.obs_normalization:
            print("Sampling episodes for normalization")
            keys = jax.random.split(state.randkey, self.sample_episodes)
            dummy_act_func = (
                lambda s, p, o: o
            )  # receive state, params, obs and return the original obs
            dummy_sample_func = lambda rk, act_func, obs: self.sample_policy(
                rk, obs
            )  # ignore act_func

            def sample(rk):
                return self.evaluate_once(
                    state, rk, dummy_act_func, None, dummy_sample_func, True
                )

            rewards, episodes = jax.jit(jax.vmap(sample))(keys)

            obs = jax.device_get(episodes["obs"])  # shape: (sample_episodes, max_step, *input_shape)
            obs = obs.reshape(
                -1, *self.input_shape
            )  # shape: (sample_episodes * max_step, *input_shape)

            obs_axis = tuple(range(obs.ndim))
            valid_data_flag = np.all(~jnp.isnan(obs), axis=obs_axis[1:])
            obs = obs[valid_data_flag]

            obs_mean = np.mean(obs, axis=0)
            obs_std = np.std(obs, axis=0)

            state = state.register(
                problem_obs_mean=obs_mean,
                problem_obs_std=obs_std,
            )

            print("Sampling episodes for normalization finished.")
            print("valid data count: ", obs.shape[0])
            print("obs_mean: ", obs_mean)
            print("obs_std: ", obs_std)
        return state

    def evaluate(self, state: State, randkey, act_func: Callable, params):
        keys = jax.random.split(randkey, self.repeat_times)
        if self.record_episode:
            rewards, episodes = jax.vmap(
                self.evaluate_once, in_axes=(None, 0, None, None, None, None, None)
            )(
                state,
                keys,
                act_func,
                params,
                self.action_policy,
                True,
                self.obs_normalization,
            )

            episodes["obs"] = episodes["obs"].reshape(
                self.max_step * self.repeat_times, *self.input_shape
            )
            episodes["action"] = episodes["action"].reshape(
                self.max_step * self.repeat_times, *self.output_shape
            )
            episodes["reward"] = episodes["reward"].reshape(
                self.max_step * self.repeat_times,
            )

            return rewards.mean(), episodes

        else:
            rewards = jax.vmap(
                self.evaluate_once, in_axes=(None, 0, None, None, None, None, None)
            )(
                state,
                keys,
                act_func,
                params,
                self.action_policy,
                False,
                self.obs_normalization,
            )

            return rewards.mean()

    def evaluate_once(
        self,
        state,
        randkey,
        act_func,
        params,
        action_policy,
        record_episode,
        normalize_obs=False,
    ):
        rng_reset, rng_episode = jax.random.split(randkey)
        init_obs, init_env_state = self.reset(rng_reset)

        if record_episode:
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
            _, _, _, done, _, count, _, rk = carry
            return ~done & (count < self.max_step)

        def body_func(carry):
            (
                obs,
                env_state,
                rng,
                done,
                tr,
                count,
                epis,
                rk,
            ) = carry  # tr -> total reward; rk -> randkey

            if normalize_obs:
                obs = norm_obs(state, obs)

            if action_policy is not None:
                forward_func = lambda obs: act_func(state, params, obs)
                action = action_policy(rk, forward_func, obs)
            else:
                action = act_func(state, params, obs)
            next_obs, next_env_state, reward, done, _ = self.step(
                rng, env_state, action
            )
            next_rng, _ = jax.random.split(rng)

            if record_episode:
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
                jax.random.split(rk)[0],
            )

        _, _, _, _, total_reward, _, episode, _ = jax.lax.while_loop(
            cond_func,
            body_func,
            (init_obs, init_env_state, rng_episode, False, 0.0, 0, episode, randkey),
        )

        if record_episode:
            return total_reward, episode
        else:
            return total_reward

    def step(self, randkey, env_state, action):
        return self.env_step(randkey, env_state, action)

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


def norm_obs(state, obs):
    return (obs - state.problem_obs_mean) / (state.problem_obs_std + 1e-6)
