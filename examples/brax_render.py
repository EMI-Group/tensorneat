import brax
from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.io import image
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import traceback

# print(f"Using Brax {brax.__version__}, Jax {jax.__version__}")
# print("From GymWrapper, env.reset()")
# try:
#     env = envs.create("inverted_pendulum",
#                       batch_size=1,
#                       episode_length=150,
#                       backend='generalized')
#     env = gym_wrapper.GymWrapper(env)
#     env.reset()
#     img = env.render(mode='rgb_array')
#     plt.imshow(img)
# except Exception:
#     traceback.print_exc()
#
# print("From GymWrapper, env.reset() and action")
# try:
#     env = envs.create("inverted_pendulum",
#                       batch_size=1,
#                       episode_length=150,
#                       backend='generalized')
#     env = gym_wrapper.GymWrapper(env)
#     env.reset()
#     action = jnp.zeros(env.action_space.shape)
#     env.step(action)
#     img = env.render(mode='rgb_array')
#     plt.imshow(img)
# except Exception:
#     traceback.print_exc()

print("From brax env")
try:
    env = envs.create("inverted_pendulum",
                      batch_size=1,
                      episode_length=150,
                      backend='generalized')
    key = jax.random.PRNGKey(0)
    initial_env_state = env.reset(key)
    base_state = initial_env_state.pipeline_state
    pipeline_state = env.pipeline_init(base_state.q.ravel(), base_state.qd.ravel())
    img = image.render_array(sys=env.sys, state=pipeline_state, width=256, height=256)
    print(f"pixel values: [{img.min()}, {img.max()}]")
    plt.imshow(img)
    plt.show()
except Exception:
    traceback.print_exc()