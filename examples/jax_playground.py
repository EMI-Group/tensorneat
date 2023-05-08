import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import vmap, jit
from functools import partial

from examples.time_utils import using_cprofile


@jit
def func(x, y):
    return x + y


a, b, c = jnp.array([1]), jnp.array([2]), jnp.array([3])
li = [a, b, c]

cpu_li = jax.device_get(li)

print(cpu_li)