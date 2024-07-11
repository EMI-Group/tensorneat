import jax.numpy as jnp

SCALE = 5


def scaled_sigmoid_(z):
    z = 1 / (1 + jnp.exp(-z))
    return z * SCALE


def sigmoid_(z):
    z = 1 / (1 + jnp.exp(-z))
    return z


def scaled_tanh_(z):
    return jnp.tanh(z) * SCALE


def tanh_(z):
    return jnp.tanh(z)


def sin_(z):
    return jnp.sin(z)


def relu_(z):
    return jnp.maximum(z, 0)


def lelu_(z):
    leaky = 0.005
    return jnp.where(z > 0, z, leaky * z)


def identity_(z):
    return z


def inv_(z):
    # avoid division by zero
    z = jnp.where(z > 0, jnp.maximum(z, 1e-7), jnp.minimum(z, -1e-7))
    return 1 / z


def log_(z):
    z = jnp.maximum(z, 1e-7)
    return jnp.log(z)


def exp_(z):
    return jnp.exp(z)


def abs_(z):
    return jnp.abs(z)
