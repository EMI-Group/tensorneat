import jax, jax.numpy as jnp

arr = jnp.ones((10, 10))
a = jnp.array([
    [1, 2, 3],
    [4, 5, 6]
])

def attach_with_inf(arr, idx):
    target_dim = arr.ndim + idx.ndim - 1
    expand_idx = jnp.expand_dims(idx, axis=tuple(range(idx.ndim, target_dim)))

    return jnp.where(expand_idx == 1, jnp.nan, arr[idx])

b = attach_with_inf(arr, a)
print(b)