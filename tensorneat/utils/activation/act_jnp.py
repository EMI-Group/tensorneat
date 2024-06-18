import jax
import jax.numpy as jnp


sigma_3 = 2.576


class Act:
    @staticmethod
    def name2func(name):
        return getattr(Act, name)

    @staticmethod
    def sigmoid(z):
        z = jnp.clip(5 * z / sigma_3, -5, 5)
        z = 1 / (1 + jnp.exp(-z))

        return z * sigma_3  # (0, sigma_3)

    @staticmethod
    def tanh(z):
        z = jnp.clip(5 * z / sigma_3, -5, 5)
        return jnp.tanh(z) * sigma_3  # (-sigma_3, sigma_3)

    @staticmethod
    def standard_tanh(z):
        z = jnp.clip(5 * z / sigma_3, -5, 5)
        return jnp.tanh(z)  # (-1, 1)

    @staticmethod
    def sin(z):
        z = jnp.clip(jnp.pi / 2 * z / sigma_3, -jnp.pi / 2, jnp.pi / 2)
        return jnp.sin(z) * sigma_3  # (-sigma_3, sigma_3)

    @staticmethod
    def relu(z):
        z = jnp.clip(z, -sigma_3, sigma_3)
        return jnp.maximum(z, 0)  # (0, sigma_3)

    @staticmethod
    def lelu(z):
        leaky = 0.005
        z = jnp.clip(z, -sigma_3, sigma_3)
        return jnp.where(z > 0, z, leaky * z)

    @staticmethod
    def identity(z):
        z = jnp.clip(z, -sigma_3, sigma_3)
        return z

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
        z = jnp.clip(z, -1, 1)
        return jnp.abs(z)


ACT_ALL = (
    Act.sigmoid,
    Act.tanh,
    Act.sin,
    Act.relu,
    Act.lelu,
    Act.identity,
    Act.inv,
    Act.log,
    Act.exp,
    Act.abs,
)


def act_func(idx, z, act_funcs):
    """
    calculate activation function for each node
    """
    idx = jnp.asarray(idx, dtype=jnp.int32)
    # change idx from float to int

    # -1 means identity activation
    res = jax.lax.cond(
        idx == -1,
        lambda: z,
        lambda: jax.lax.switch(idx, act_funcs, z),
    )

    return res
