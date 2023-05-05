"""
aggregations, two special case need to consider:
1. extra 0s
2. full of 0s
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


@jit
def sum_agg(z):
    z = jnp.where(jnp.isnan(z), 0, z)
    return jnp.sum(z, axis=0)


@jit
def product_agg(z):
    z = jnp.where(jnp.isnan(z), 1, z)
    return jnp.prod(z, axis=0)


@jit
def max_agg(z):
    z = jnp.where(jnp.isnan(z), -jnp.inf, z)
    return jnp.max(z, axis=0)


@jit
def min_agg(z):
    z = jnp.where(jnp.isnan(z), jnp.inf, z)
    return jnp.min(z, axis=0)


@jit
def maxabs_agg(z):
    z = jnp.where(jnp.isnan(z), 0, z)
    abs_z = jnp.abs(z)
    max_abs_index = jnp.argmax(abs_z)
    return z[max_abs_index]


@jit
def median_agg(z):

    non_zero_mask = ~jnp.isnan(z)
    n = jnp.sum(non_zero_mask, axis=0)

    z = jnp.where(jnp.isnan(z), jnp.inf, z)
    sorted_valid_values = jnp.sort(z)

    def _even_case():
        return (sorted_valid_values[n // 2 - 1] + sorted_valid_values[n // 2]) / 2

    def _odd_case():
        return sorted_valid_values[n // 2]

    median = jax.lax.cond(n % 2 == 0, _even_case, _odd_case)

    return median


@jit
def mean_agg(z):
    non_zero_mask = ~jnp.isnan(z)
    valid_values_sum = sum_agg(z)
    valid_values_count = jnp.sum(non_zero_mask, axis=0)
    mean_without_zeros = valid_values_sum / valid_values_count
    return mean_without_zeros


AGG_TOTAL_LIST = [sum_agg, product_agg, max_agg, min_agg, maxabs_agg, median_agg, mean_agg]

agg_name2key = {
    'sum': 0,
    'product': 1,
    'max': 2,
    'min': 3,
    'maxabs': 4,
    'median': 5,
    'mean': 6,
}


@jit
def agg(idx, z):
    idx = jnp.asarray(idx, dtype=jnp.int32)

    def full_zero():
        return 0.

    def not_full_zero():
        return jax.lax.switch(idx, AGG_TOTAL_LIST, z)

    return jax.lax.cond(jnp.all(z == 0.), full_zero, not_full_zero)


vectorized_agg = jax.vmap(agg, in_axes=(0, 0))

if __name__ == '__main__':
    array = jnp.asarray([1, 2, np.nan, np.nan, 3, 4, 5, np.nan, np.nan, np.nan, np.nan], dtype=jnp.float32)
    for names in agg_name2key.keys():
        print(names, agg(agg_name2key[names], array))

    array2 = jnp.asarray([0, 0, 0, 0], dtype=jnp.float32)
    for names in agg_name2key.keys():
        print(names, agg(agg_name2key[names], array2))
