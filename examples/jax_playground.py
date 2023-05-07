import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import vmap, jit

from examples.time_utils import using_cprofile


def func(x, y):
    """
    :param x: (100, )
    :param y: (100,
    :return:
    """
    return x * y


# @using_cprofile
def main():
    key = jax.random.PRNGKey(42)

    x1, y1 = jax.random.normal(key, shape=(100,)), jax.random.normal(key, shape=(100,))

    jit_func = jit(func)

    z = jit_func(x1, y1)
    print(z)

    jit_lower_func = jit(func).lower(x1, y1).compile()
    print(type(jit_lower_func))
    import pickle

    with open('jit_function.pkl', 'wb') as f:
        pickle.dump(jit_lower_func, f)

    new_jit_lower_func = pickle.load(open('jit_function.pkl', 'rb'))

    print(jit_lower_func(x1, y1))
    print(new_jit_lower_func(x1, y1))

    # x2, y2 = jax.random.normal(key, shape=(200,)), jax.random.normal(key, shape=(200,))
    # print(jit_lower_func(x2, y2))


if __name__ == '__main__':
    main()
