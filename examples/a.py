import jax.random
import numpy as np
import jax.numpy as jnp
import time


def random_array(key):
    return jax.random.normal(key, (1000,))

def random_array_np():
    return np.random.normal(size=(1000,))


def t_jax():
    key = jax.random.PRNGKey(42)
    max_li = []
    tic = time.time()
    for _ in range(100):
        key, sub_key = jax.random.split(key)
        array = random_array(sub_key)
        array = jax.device_get(array)
        max_li.append(max(array))
    print(max_li, time.time() - tic)

def t_np():
    max_li = []
    tic = time.time()
    for _ in range(100):
        max_li.append(max(random_array_np()))
    print(max_li, time.time() - tic)

if __name__ == '__main__':
    t_np()