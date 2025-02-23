import jax.numpy as jnp
from brax import envs

from .rl_jit import RLEnv, norm_obs


class BraxEnv(RLEnv):
    def __init__(
        self, env_name: str = "ant", backend: str = "generalized", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_name = env_name
        self.env = envs.create(env_name=env_name, backend=backend)

    def env_step(self, randkey, env_state, action):
        state = self.env.step(env_state, action)
        return state.obs, state, state.reward, state.done.astype(jnp.bool_), state.info

    def env_reset(self, randkey):
        init_state = self.env.reset(randkey)
        return init_state.obs, init_state

    @property
    def input_shape(self):
        return (self.env.observation_size,)

    @property
    def output_shape(self):
        return (self.env.action_size,)

    def show(
            self,
            state,
            randkey,
            act_func,
            params,
            save_path=None,
            height=480,
            width=480,
            output_type="rgb_array",
            *args,
            **kwargs,
        ):

        assert output_type in ["rgb_array", "gif", "mp4"]

        import jax
        import imageio
        from brax.io import image
        import numpy as np

        obs, env_state = self.reset(randkey)
        reward, done = 0.0, False
        state_histories = [env_state.pipeline_state]

        def step(key, env_state, obs):
            key, _ = jax.random.split(key)

            if self.obs_normalization:
                obs = norm_obs(state, obs)

            if self.action_policy is not None:
                forward_func = lambda obs: act_func(state, params, obs)
                action = self.action_policy(key, forward_func, obs)
            else:
                action = act_func(state, params, obs)

            obs, env_state, r, done, info = self.step(randkey, env_state, action)
            return key, env_state, obs, r, done

        jit_step = jax.jit(step)

        for _ in range(self.max_step):
            key, env_state, obs, r, done = jit_step(randkey, env_state, obs)
            state_histories.append(env_state.pipeline_state)
            reward += r
            if done:
                break

        print("Total reward: ", reward)

        try:
            imgs = image.render_array(
                sys=self.env.sys, trajectory=state_histories, height=height, width=width, camera="track"
            )
        except ValueError:
            imgs = image.render_array(
                sys=self.env.sys, trajectory=state_histories, height=height, width=width
            )

        if output_type == "rgb_array":
            imgs = np.array(imgs)
            return imgs

        if save_path is None:
            save_path = f"{self.env_name}.{output_type}"

        imageio.mimsave(save_path, imgs, *args, **kwargs)

        if output_type == "gif":
            imageio.mimsave(save_path, imgs, *args, **kwargs)
        elif output_type == "mp4":
            fps = kwargs.get("fps", 30)
            imageio.mimsave(save_path, imgs, fps=fps, codec="libx264", format="mp4")

        print(f"{output_type} saved to: ", save_path)
