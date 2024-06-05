import jax, jax.numpy as jnp
import jumanji

from utils import State
from ..rl_jit import RLEnv


class Jumanji_2048(RLEnv):
    def __init__(
        self, max_step=1000, repeat_times=1, record_episode=False, guarantee_invalid_action=True
    ):
        super().__init__(max_step, repeat_times, record_episode)
        self.guarantee_invalid_action = guarantee_invalid_action
        self.env = jumanji.make("Game2048-v1")

    def env_step(self, randkey, env_state, action):
        action_mask = env_state["action_mask"]
        if self.guarantee_invalid_action:
            score_with_mask = jnp.where(action_mask, action, -jnp.inf)
            action = jnp.argmax(score_with_mask)
        else:
            action = jnp.argmax(action)

        done = ~action_mask[action]

        env_state, timestep = self.env.step(env_state, action)
        reward = timestep["reward"]

        board, action_mask = timestep["observation"]
        extras = timestep["extras"]

        done = done | (jnp.sum(action_mask) == 0)  # all actions of invalid

        return board.reshape(-1), env_state, reward, done, extras

    def env_reset(self, randkey):
        env_state, timestep = self.env.reset(randkey)
        step_type = timestep["step_type"]
        reward = timestep["reward"]
        discount = timestep["discount"]
        observation = timestep["observation"]
        extras = timestep["extras"]
        board, action_mask = observation

        return board.reshape(-1), env_state

    @property
    def input_shape(self):
        return (16,)

    @property
    def output_shape(self):
        return (4,)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        raise NotImplementedError("GymNax render must rely on gym 0.19.0(old version).")
