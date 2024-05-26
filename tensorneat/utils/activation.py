import jax
import jax.numpy as jnp


class Act:
    @staticmethod
    def sigmoid(z):
        z = jnp.clip(5 * z, -10, 10)
        return 1 / (1 + jnp.exp(-z))

    @staticmethod
    def tanh(z):
        return jnp.tanh(0.6 * z)

    @staticmethod
    def sin(z):
        return jnp.sin(z)

    @staticmethod
    def relu(z):
        return jnp.maximum(z, 0)

    @staticmethod
    def lelu(z):
        leaky = 0.005
        return jnp.where(z > 0, z, leaky * z)

    @staticmethod
    def identity(z):
        return z

    @staticmethod
    def clamped(z):
        return jnp.clip(z, -1, 1)

    @staticmethod
    def inv(z):
        z = jnp.where(z > 0, jnp.maximum(z, 1e-7), jnp.minimum(z, -1e-7))
        return 1 / z

    @staticmethod
    def log(z):
        z = jnp.maximum(z, 1e-7)
        return jnp.log(z)

    @staticmethod
    def exp(z):
        z = jnp.clip(z, -10, 10)
        return jnp.exp(z)

    @staticmethod
    def abs(z):
        return jnp.abs(z)


ACT_ALL = (
    Act.sigmoid,
    Act.tanh,
    Act.sin,
    Act.relu,
    Act.lelu,
    Act.identity,
    Act.clamped,
    Act.inv,
    Act.log,
    Act.exp,
    Act.abs,
)


def act(idx, z, act_funcs):
    """
    calculate activation function for each node
    """
    idx = jnp.asarray(idx, dtype=jnp.int32)
    # change idx from float to int
    res = jax.lax.switch(idx, act_funcs, z)
    return res
