"""
Has the built-in activation functions,
code for using them,
and code for adding new user-defined ones
"""
import math

def sigmoid_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


activation_dict = {
    "sigmoid": sigmoid_activation,
}

full_activation_list = list(activation_dict.keys())