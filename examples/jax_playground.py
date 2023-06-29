from functools import partial

from jax import numpy as jnp, jit


@partial(jit, static_argnames=['reverse'])
def rank_element(array, reverse=False):
    """
    rank the element in the array.
    if reverse is True, the rank is from large to small.
    """
    if reverse:
        array = -array
    return jnp.argsort(jnp.argsort(array))


a = jnp.array([1, 5, 3, 5, 2, 1, 0])
print(rank_element(a, reverse=True))
