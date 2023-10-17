from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from brax import envs
from core import State
from .rl_jit import RLEnv, RLEnvConfig


@dataclass(frozen=True)
class BraxConfig(RLEnvConfig):
    env_name: str = "ant"
    backend: str = "generalized"

    def __post_init__(self):
        # TODO: Check if env_name is registered
        # assert self.env_name in gymnax.registered_envs, f"Env {self.env_name} not registered"
        pass


class BraxEnv(RLEnv):
    def __init__(self, config: BraxConfig = BraxConfig()):
        super().__init__(config)
        self.config = config
        self.env = envs.create(env_name=config.env_name, backend=config.backend)

    def env_step(self, randkey, env_state, action):
        state = self.env.step(env_state, action)
        return state.obs, state, state.reward, state.done.astype(jnp.bool_), state.info

    def env_reset(self, randkey):
        init_state = self.env.reset(randkey)
        return init_state.obs, init_state

    @property
    def input_shape(self):
        return (self.env.observation_size, )

    @property
    def output_shape(self):
        return (self.env.action_size, )

    def show(self, randkey, state: State, act_func: Callable, params):
        # TODO
        raise NotImplementedError("im busy! to de done!")
