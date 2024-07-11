import jax, jax.numpy as jnp

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
    "clip": SympyClip,
}

agg_name2jnp = {
    "sum": sum_,
    "product": product_,
    "max": max_,
    "min": min_,
    "maxabs": maxabs_,
    "mean": mean_,
}

agg_name2sympy = {
    "sum": SympySum,
    "product": SympyProduct,
    "max": SympyMax,
    "min": SympyMin,
    "maxabs": SympyMaxabs,
    "mean": SympyMean,
}

ACT = FunctionManager(act_name2jnp, act_name2sympy)
AGG = FunctionManager(agg_name2jnp, agg_name2sympy)

def apply_activation(idx, z, act_funcs):
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

def apply_aggregation(idx, z, agg_funcs):
    """
    calculate activation function for inputs of node
    """
    idx = jnp.asarray(idx, dtype=jnp.int32)

    return jax.lax.cond(
        jnp.all(jnp.isnan(z)),
        lambda: jnp.nan,  # all inputs are nan
        lambda: jax.lax.switch(idx, agg_funcs, z),  # otherwise
    )

def get_func_name(func):
    name = func.__name__
    if name.endswith("_"):
        name = name[:-1]
    return name