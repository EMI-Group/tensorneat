import jax.numpy as jnp


def sum_(z, mask):
    return jnp.sum(z, axis=0, where=mask, initial=0)


def product_(z, mask):
    return jnp.prod(z, axis=0, where=mask, initial=1)


def max_(z, mask):
    return jnp.max(z, axis=0, where=mask, initial=-jnp.inf)


def min_(z, mask):
    return jnp.min(z, axis=0, where=mask, initial=jnp.inf)


def maxabs_(z, mask):
    z = jnp.where(mask, z, 0)
    abs_z = jnp.abs(z)
    max_abs_index = jnp.argmax(abs_z)
    return z[max_abs_index]

def mean_(z, mask):
    s = jnp.sum(z, axis=0, where=mask, initial=0)
    valid_count = jnp.sum(mask, axis=0)
    return s / valid_count
