from dataclasses import dataclass
from typing import Callable
from functools import partial

import jax

from config import ProblemConfig

from core import Problem, State


@dataclass(frozen=True)
class RLEnvConfig(ProblemConfig):
    output_transform: Callable = lambda x: x


class RLEnv(Problem):

    jitable = True

    def __init__(self, config: RLEnvConfig = RLEnvConfig()):
        super().__init__(config)
        self.config = config

    def evaluate(self, randkey, state: State, act_func: Callable, params):
        rng_reset, rng_episode = jax.random.split(randkey)
        init_obs, init_env_state = self.reset(rng_reset)

        def cond_func(carry):
            _, _, _, done, _ = carry
            return ~done

        def body_func(carry):
            obs, env_state, rng, _, tr = carry  # total reward
            net_out = act_func(state, obs, params)
            action = self.config.output_transform(net_out)
            next_obs, next_env_state, reward, done, _ = self.step(rng, env_state, action)
            next_rng, _ = jax.random.split(rng)
            return next_obs, next_env_state, next_rng, done, tr + reward

        _, _, _, _, total_reward = jax.lax.while_loop(
            cond_func,
            body_func,
            (init_obs, init_env_state, rng_episode, False, 0.0)
        )

        return total_reward

    @partial(jax.jit, static_argnums=(0,))
    def step(self, randkey, env_state, action):
        return self.env_step(randkey, env_state, action)

    @partial(jax.jit, static_argnums=(0,))
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

    def show(self, randkey, state: State, act_func: Callable, params):
        raise NotImplementedError
