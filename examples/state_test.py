import jax
from jax import numpy as jnp
from algorithm.state import State


@jax.jit
def func(state: State, a):
    return state.update(a=a)


state = State(c=1, b=2)
print(state)

vmap_func = jax.vmap(func, in_axes=(None, 0))
print(vmap_func(state, jnp.array([1, 2, 3])))