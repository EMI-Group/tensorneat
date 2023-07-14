import jax
from algorithm.state import State

@jax.jit
def func(state: State, a):
    return state.update(a=a)


state = State(c=1, b=2)
print(state)

state = func(state, 1111111)

print(state)
