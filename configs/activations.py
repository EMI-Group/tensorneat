from neat.genome.activations import *

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


def refactor_act(config):
    config['activation_default'] = act_name2key[config['activation_default']]
    config['activation_options'] = [
        act_name2key[act_name] for act_name in config['activation_options']
    ]
