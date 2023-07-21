import numpy as np
import jax.numpy as jnp

a = jnp.zeros((5, 5))
k1 = jnp.array([1, 2, 3])
k2 = jnp.array([2, 3, 4])
v = jnp.array([1, 1, 1])

a = a.at[k1, k2].set(v)

print(a)
