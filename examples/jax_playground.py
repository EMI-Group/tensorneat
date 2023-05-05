import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import vmap, jit


def plus1(x):
    return x + 1


def minus1(x):
    return x - 1


def func(rand_key, x):
    r = jax.random.uniform(rand_key, shape=())
    return jax.lax.cond(r > 0.5, plus1, minus1, x)


def func2(rand_key):
    r = jax.random.uniform(rand_key, ())
    if r < 0.3:
        return 1
    elif r < 0.5:
        return 2
    else:
        return 3



key = random.PRNGKey(0)
print(func(key, 0))

batch_func = vmap(jit(func))
keys = random.split(key, 100)
print(batch_func(keys, jnp.zeros(100)))