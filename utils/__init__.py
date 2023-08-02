from .activation import Activation, act
from .aggregation import Aggregation, agg
from .tools import *
from .graph import *

Activation.name2func = {
    'sigmoid': Activation.sigmoid_act,
    'tanh': Activation.tanh_act,
    'sin': Activation.sin_act,
    'gauss': Activation.gauss_act,
    'relu': Activation.relu_act,
    'elu': Activation.elu_act,
    'lelu': Activation.lelu_act,
    'selu': Activation.selu_act,
    'softplus': Activation.softplus_act,
    'identity': Activation.identity_act,
    'clamped': Activation.clamped_act,
    'inv': Activation.inv_act,
    'log': Activation.log_act,
    'exp': Activation.exp_act,
    'abs': Activation.abs_act,
    'hat': Activation.hat_act,
    'square': Activation.square_act,
    'cube': Activation.cube_act,
}

Aggregation.name2func = {
    'sum': Aggregation.sum_agg,
    'product': Aggregation.product_agg,
    'max': Aggregation.max_agg,
    'min': Aggregation.min_agg,
    'maxabs': Aggregation.maxabs_agg,
    'median': Aggregation.median_agg,
    'mean': Aggregation.mean_agg,
}
