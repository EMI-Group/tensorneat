import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import vmap, jit


seed = jax.random.PRNGKey(42)
seed, *subkeys = random.split(seed, 3)


c = random.split(seed, 1)
print(seed, subkeys)
print(c)