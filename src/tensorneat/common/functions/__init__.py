from .act_jnp import *
from .act_sympy import *
from .agg_jnp import *
from .agg_sympy import *
from .manager import FunctionManager

act_name2jnp = {
    "scaled_sigmoid": scaled_sigmoid_,
    "sigmoid": sigmoid_,
    "scaled_tanh": scaled_tanh_,
    "tanh": tanh_,
    "sin": sin_,
    "relu": relu_,
    "lelu": lelu_,
    "identity": identity_,
    "inv": inv_,
    "log": log_,
    "exp": exp_,
    "abs": abs_,
}

act_name2sympy = {
    "scaled_sigmoid": SympyScaledSigmoid,
    "sigmoid": SympySigmoid,
    "scaled_tanh": SympyScaledTanh,
    "tanh": SympyTanh,
    "sin": SympySin,
    "relu": SympyRelu,
    "lelu": SympyLelu,
    "identity": SympyIdentity,
    "inv": SympyIdentity,
    "log": SympyLog,
    "exp": SympyExp,
    "abs": SympyAbs,
}

agg_name2jnp = {
    "sum": sum_,
    "product": product_,
    "max": max_,
    "min": min_,
    "maxabs": maxabs_,
    "median": median_,
    "mean": mean_,
}

agg_name2sympy = {
    "sum": SympySum,
    "product": SympyProduct,
    "max": SympyMax,
    "min": SympyMin,
    "maxabs": SympyMaxabs,
    "median": SympyMedian,
    "mean": SympyMean,
}

ACT = FunctionManager(act_name2jnp, act_name2sympy)
AGG = FunctionManager(agg_name2jnp, agg_name2sympy)
