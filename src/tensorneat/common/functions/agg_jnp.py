import jax.numpy as jnp


def sum_(z):
    return jnp.sum(z, axis=0, where=~jnp.isnan(z), initial=0)


def product_(z):
    return jnp.prod(z, axis=0, where=~jnp.isnan(z), initial=1)


def max_(z):
    return jnp.max(z, axis=0, where=~jnp.isnan(z), initial=-jnp.inf)


def min_(z):
    return jnp.min(z, axis=0, where=~jnp.isnan(z), initial=jnp.inf)


def maxabs_(z):
    z = jnp.where(jnp.isnan(z), 0, z)
    abs_z = jnp.abs(z)
    max_abs_index = jnp.argmax(abs_z)
    return z[max_abs_index]


def median_(z):
    n = jnp.sum(~jnp.isnan(z), axis=0)

    z = jnp.sort(z)  # sort

    idx1, idx2 = (n - 1) // 2, n // 2
    median = (z[idx1] + z[idx2]) / 2

    return median


def mean_(z):
    sumation = sum_(z)
    valid_count = jnp.sum(~jnp.isnan(z), axis=0)
    return sumation / valid_count
