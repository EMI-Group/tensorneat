import numpy as np


def sigmoid_act(z):
    z = np.clip(z * 5, -60, 60)
    return 1 / (1 + np.exp(-z))


def tanh_act(z):
    z = np.clip(z * 2.5, -60, 60)
    return np.tanh(z)


def sin_act(z):
    z = np.clip(z * 5, -60, 60)
    return np.sin(z)


def gauss_act(z):
    z = np.clip(z, -3.4, 3.4)
    return np.exp(-5 * z ** 2)


def relu_act(z):
    return np.maximum(z, 0)


def elu_act(z):
    return np.where(z > 0, z, np.exp(z) - 1)


def lelu_act(z):
    leaky = 0.005
    return np.where(z > 0, z, leaky * z)


def selu_act(z):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return np.where(z > 0, lam * z, lam * alpha * (np.exp(z) - 1))


def softplus_act(z):
    z = np.clip(z * 5, -60, 60)
    return 0.2 * np.log(1 + np.exp(z))


def identity_act(z):
    return z


def clamped_act(z):
    return np.clip(z, -1, 1)


def inv_act(z):
    return 1 / z


def log_act(z):
    z = np.maximum(z, 1e-7)
    return np.log(z)


def exp_act(z):
    z = np.clip(z, -60, 60)
    return np.exp(z)


def abs_act(z):
    return np.abs(z)


def hat_act(z):
    return np.maximum(0, 1 - np.abs(z))


def square_act(z):
    return z ** 2


def cube_act(z):
    return z ** 3


ACT_TOTAL_LIST = [sigmoid_act, tanh_act, sin_act, gauss_act, relu_act, elu_act, lelu_act, selu_act, softplus_act,
                  identity_act, clamped_act, inv_act, log_act, exp_act, abs_act, hat_act, square_act, cube_act]

act_name2key = {
    'sigmoid': 0,
    'tanh': 1,
    'sin': 2,
    'gauss': 3,
    'relu': 4,
    'elu': 5,
    'lelu': 6,
    'selu': 7,
    'softplus': 8,
    'identity': 9,
    'clamped': 10,
    'inv': 11,
    'log': 12,
    'exp': 13,
    'abs': 14,
    'hat': 15,
    'square': 16,
    'cube': 17,
}


def act(idx, z):
    idx = np.asarray(idx, dtype=np.int32)
    return ACT_TOTAL_LIST[idx](z)
