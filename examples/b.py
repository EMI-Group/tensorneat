from enum import Enum
from jax import jit

class NetworkType(Enum):
    ANN = 0
    SNN = 1
    LSTM = 2




@jit
def func(d):
    return d[0] + 1


d = {0: 1, 1: NetworkType.ANN.value}
n = None

print(n or d)
print(d)

print(func(d))
