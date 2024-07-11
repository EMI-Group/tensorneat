from tensorneat.common.aggregation.agg_jnp import Agg, agg_func, AGG_ALL
from .tools import *
from .graph import *
from .state import State
from .stateful_class import StatefulBaseClass

from .aggregation.agg_jnp import Agg, AGG_ALL, agg_func
from .activation.act_jnp import Act, ACT_ALL, act_func
from .aggregation.agg_sympy import *
from .activation.act_sympy import *

from typing import Callable, Union

name2sympy = {
    "sigmoid": SympySigmoid,
    "standard_sigmoid": SympyStandardSigmoid,
    "tanh": SympyTanh,
    "standard_tanh": SympyStandardTanh,
    "sin": SympySin,
    "relu": SympyRelu,
    "lelu": SympyLelu,
    "identity": SympyIdentity,
    "inv": SympyInv,
    "log": SympyLog,
    "exp": SympyExp,
    "abs": SympyAbs,
    "sum": SympySum,
    "product": SympyProduct,
    "max": SympyMax,
    "min": SympyMin,
    "maxabs": SympyMaxabs,
    "mean": SympyMean,
    "clip": SympyClip,
}


def convert_to_sympy(func: Union[str, Callable]):
    if isinstance(func, str):
        name = func
    else:
        name = func.__name__
    if name in name2sympy:
        return name2sympy[name]
    else:
        raise ValueError(
            f"Can not convert to sympy! Function {name} not found in name2sympy"
        )


SYMPY_FUNCS_MODULE_NP = {}
SYMPY_FUNCS_MODULE_JNP = {}
for cls in name2sympy.values():
    if hasattr(cls, "numerical_eval"):
        SYMPY_FUNCS_MODULE_NP[cls.__name__] = cls.numerical_eval
        SYMPY_FUNCS_MODULE_JNP[cls.__name__] = partial(cls.numerical_eval, backend=jnp)
