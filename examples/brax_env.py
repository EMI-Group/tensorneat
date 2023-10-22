import imageio
import jax

import brax
from brax import envs
from brax.io import image
import matplotlib.pyplot as plt

import time
from tqdm import tqdm
import numpy as np


def inference_func(key, *args):
    return jax.random.normal(key, shape=(env.action_size,))


env_name = "ant"
backend = "generalized"

env = envs.create(env_name=env_name, backend=backend)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_func)

rng = jax.random.PRNGKey(seed=1)
ori_state = jit_env_reset(rng=rng)
state = ori_state

render_history = []

for i in range(100):
    act_rng, rng = jax.random.split(rng)

    tic = time.time()
    act = jit_inference_fn(act_rng, state.obs)
    state = jit_env_step(state, act)
    print("step time: ", time.time() - tic)

    render_history.append(state.pipeline_state)

    # img = image.render_array(sys=env.sys, state=pipeline_state, width=512, height=512)
    # print("render time: ", time.time() - tic)

    # plt.imsave("../images/ant_{}.png".format(i), img)

    reward = state.reward
    done = state.done
    print(i, reward)

render_history = jax.device_get(render_history)
# print(render_history)

imgs = [image.render_array(sys=env.sys, state=s, width=512, height=512) for s in tqdm(render_history)]


# for i, s in enumerate(tqdm(render_history)):
#     img = image.render_array(sys=env.sys, state=s, width=512, height=512)
#     print(img.shape)
#     # print(type(img))
#     plt.imsave("../images/ant_{}.png".format(i), img)


def create_gif(image_list, gif_name, duration):
    with imageio.get_writer(gif_name, mode='I', duration=duration) as writer:
        for image in image_list:
            # 确保图像的数据类型正确
            formatted_image = np.array(image, dtype=np.uint8)
            writer.append_data(formatted_image)


create_gif(imgs, "../images/ant.gif", 0.1)
