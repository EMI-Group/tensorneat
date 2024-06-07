import gymnax

from .rl_jit import RLEnv


class GymNaxEnv(RLEnv):
    def __init__(self, env_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert env_name in gymnax.registered_envs, f"Env {env_name} not registered"
        self.env, self.env_params = gymnax.make(env_name)

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

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        raise NotImplementedError("GymNax render must rely on gym 0.19.0(old version).")
