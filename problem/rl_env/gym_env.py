from dataclasses import dataclass
from typing import Callable

import gym

from core import State
from .rl_unjit import RLEnv, RLEnvConfig


@dataclass(frozen=True)
class GymConfig(RLEnvConfig):
    env_name: str = "CartPole-v1"

    def __post_init__(self):
        assert self.env_name in gym.registered_envs, f"Env {self.env_name} not registered"


class GymNaxEnv(RLEnv):

    def __init__(self, config: GymConfig = GymConfig()):
        super().__init__(config)
        self.config = config
        self.env, self.env_params = gym.make(config.env_name)

    def env_step(self, randkey, env_state, action):
        return self.env.step(randkey, env_state, action, self.env_params)

    def env_reset(self, randkey):
        return self.env.reset(randkey, self.env_params)

    @property
    def input_shape(self):
        return self.env.observation_space(self.env_params).shape

    @property
    def output_shape(self):
        return self.env.action_space(self.env_params).shape

    def show(self, randkey, state: State, act_func: Callable, params):
        raise NotImplementedError("GymNax render must rely on gym 0.19.0(old version).")
