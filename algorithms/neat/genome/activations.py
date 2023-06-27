import jax
import jax.numpy as jnp
from jax import jit


@jit
def sigmoid_act(z):
    z = jnp.clip(z * 5, -60, 60)
    return 1 / (1 + jnp.exp(-z))


@jit
def tanh_act(z):
    z = jnp.clip(z * 2.5, -60, 60)
    return jnp.tanh(z)


@jit
def sin_act(z):
    z = jnp.clip(z * 5, -60, 60)
    return jnp.sin(z)


@jit
def gauss_act(z):
    z = jnp.clip(z * 5, -3.4, 3.4)
    return jnp.exp(-z ** 2)


@jit
def relu_act(z):
    return jnp.maximum(z, 0)


@jit
def elu_act(z):
    return jnp.where(z > 0, z, jnp.exp(z) - 1)


@jit
def lelu_act(z):
    leaky = 0.005
    return jnp.where(z > 0, z, leaky * z)


@jit
def selu_act(z):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return jnp.where(z > 0, lam * z, lam * alpha * (jnp.exp(z) - 1))


@jit
def softplus_act(z):
    z = jnp.clip(z * 5, -60, 60)
    return 0.2 * jnp.log(1 + jnp.exp(z))


@jit
def identity_act(z):
    return z


@jit
def clamped_act(z):
    return jnp.clip(z, -1, 1)


@jit
def inv_act(z):
    z = jnp.maximum(z, 1e-7)
    return 1 / z


@jit
def log_act(z):
    z = jnp.maximum(z, 1e-7)
    return jnp.log(z)


@jit
def exp_act(z):
    z = jnp.clip(z, -60, 60)
    return jnp.exp(z)


@jit
def abs_act(z):
    return jnp.abs(z)


@jit
def hat_act(z):
    return jnp.maximum(0, 1 - jnp.abs(z))


@jit
def square_act(z):
    return z ** 2


@jit
def cube_act(z):
    return z ** 3


act_name2func = {
    'sigmoid': sigmoid_act,
    'tanh': tanh_act,
    'sin': sin_act,
    'gauss': gauss_act,
    'relu': relu_act,
    'elu': elu_act,
    'lelu': lelu_act,
    'selu': selu_act,
    'softplus': softplus_act,
    'identity': identity_act,
    'clamped': clamped_act,
    'inv': inv_act,
    'log': log_act,
    'exp': exp_act,
    'abs': abs_act,
    'hat': hat_act,
    'square': square_act,
    'cube': cube_act,
}
