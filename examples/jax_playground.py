import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import vmap, jit
from functools import partial

from examples.time_utils import using_cprofile


def func(x, y):
    """
    :param x: (100, )
    :param y: (100,
    :return:
    """
    return x * y


def func2(x, y, s):
    """
    :param x: (100, )
    :param y: (100,
    :return:
    """
    if s == '123':
        return 0
    else:
        return x * y


@jit
def func3(x, y):
    return func2(x, y, '123')


# @using_cprofile
def main():
    key = jax.random.PRNGKey(42)

    x1, y1 = jax.random.normal(key, shape=(1000,)), jax.random.normal(key, shape=(1000,))

    jit_lower_func = jit(func).lower(1, 2).compile()
    print(type(jit_lower_func))
    print(jit_lower_func.memory_analysis())

    jit_compiled_func2 = jit(func2, static_argnames=['s']).lower(x1, y1, '123').compile()
    print(jit_compiled_func2(x1, y1))

    # print(jit_compiled_func2(x1, y1))

    f = func3.lower(x1, y1).compile()

    print(f(x1, y1))

    # print(jit_lower_func(x1, y1))

    # x2, y2 = jax.random.normal(key, shape=(200,)), jax.random.normal(key, shape=(200,))
    # print(jit_lower_func(x2, y2))


if __name__ == '__main__':
    main()
