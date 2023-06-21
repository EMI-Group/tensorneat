import numpy as np

import jax.numpy as jnp
import jax

a = jnp.array([1, 0, 1, 0, np.nan])
b = jnp.array([1, 1, 1, 1, 1])
c = jnp.array([1, 1, 1, 1, 1])

full = jnp.array([
    [1, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 1, 1],
])

print(jnp.column_stack([a[:, None], b[:, None], c[:, None]]))

aux0 = full[:, 0, None]
aux1 = full[:, 1, None]

print(aux0, aux0.shape)

print(jnp.concatenate([aux0, aux1], axis=1))

f_a = jnp.array([False, False, True, True])
f_b = jnp.array([True, False, False, False])

print(jnp.logical_and(f_a, f_b))
print(f_a & f_b)

print(f_a + jnp.nan * 0.0)
print(f_a + 1 * 0.0)


@jax.jit
def main():
    return func('happy') + func('sad')


def func(x):
    if x == 'happy':
        return 1
    else:
        return 2


print(main())