import jax.numpy as jnp
from brax import envs

from .rl_jit import RLEnv


class BraxEnv(RLEnv):
    def __init__(self, max_step=1000, record_episode=False, env_name: str = "ant", backend: str = "generalized"):
        super().__init__(max_step, record_episode)
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
        height=512,
        width=512,
        duration=0.1,
        *args,
        **kwargs
    ):

        import jax
        import imageio
        import numpy as np
        from brax.io import image
        from tqdm import tqdm

        obs, env_state = self.reset(randkey)
        reward, done = 0.0, False
        state_histories = []

        def step(key, env_state, obs):
            key, _ = jax.random.split(key)
            action = act_func(obs, params)
            obs, env_state, r, done, _ = self.step(randkey, env_state, action)
            return key, env_state, obs, r, done

        while not done:
            state_histories.append(env_state.pipeline_state)
            key, env_state, obs, r, done = jax.jit(step)(randkey, env_state, obs)
            reward += r

        imgs = [
            image.render_array(sys=self.env.sys, state=s, width=width, height=height)
            for s in tqdm(state_histories, desc="Rendering")
        ]

        def create_gif(image_list, gif_name, duration):
            with imageio.get_writer(gif_name, mode="I", duration=duration) as writer:
                for image in image_list:
                    formatted_image = np.array(image, dtype=np.uint8)
                    writer.append_data(formatted_image)

        create_gif(imgs, save_path, duration=0.1)
        print("Gif saved to: ", save_path)
        print("Total reward: ", reward)
