import jax
import jax.numpy as jnp


class Activation:
    name2func = {}

    @staticmethod
    def sigmoid_act(z):
        z = jnp.clip(z * 5, -60, 60)
        return 1 / (1 + jnp.exp(-z))

    @staticmethod
    def tanh_act(z):
        z = jnp.clip(z * 2.5, -60, 60)
        return jnp.tanh(z)

    @staticmethod
    def sin_act(z):
        z = jnp.clip(z * 5, -60, 60)
        return jnp.sin(z)

    @staticmethod
    def gauss_act(z):
        z = jnp.clip(z * 5, -3.4, 3.4)
        return jnp.exp(-z ** 2)

    @staticmethod
    def relu_act(z):
        return jnp.maximum(z, 0)

    @staticmethod
    def elu_act(z):
        return jnp.where(z > 0, z, jnp.exp(z) - 1)

    @staticmethod
    def lelu_act(z):
        leaky = 0.005
        return jnp.where(z > 0, z, leaky * z)

    @staticmethod
    def selu_act(z):
        lam = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717
        return jnp.where(z > 0, lam * z, lam * alpha * (jnp.exp(z) - 1))

    @staticmethod
    def softplus_act(z):
        z = jnp.clip(z * 5, -60, 60)
        return 0.2 * jnp.log(1 + jnp.exp(z))

    @staticmethod
    def identity_act(z):
        return z

    @staticmethod
    def clamped_act(z):
        return jnp.clip(z, -1, 1)

    @staticmethod
    def inv_act(z):
        z = jnp.maximum(z, 1e-7)
        return 1 / z

    @staticmethod
    def log_act(z):
        z = jnp.maximum(z, 1e-7)
        return jnp.log(z)

    @staticmethod
    def exp_act(z):
        z = jnp.clip(z, -60, 60)
        return jnp.exp(z)

    @staticmethod
    def abs_act(z):
        return jnp.abs(z)

    @staticmethod
    def hat_act(z):
        return jnp.maximum(0, 1 - jnp.abs(z))

    @staticmethod
    def square_act(z):
        return z ** 2

    @staticmethod
    def cube_act(z):
        return z ** 3


def act(idx, z, act_funcs):
    """
    calculate activation function for each node
    """
    idx = jnp.asarray(idx, dtype=jnp.int32)
    # change idx from float to int
    res = jax.lax.switch(idx, act_funcs, z)
    return res
