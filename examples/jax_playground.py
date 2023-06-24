import numpy as np
from jax import jit

from configs import Configer
from neat.pipeline import Pipeline

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
xor_outputs = np.array([[0], [1], [1], [0]], dtype=np.float32)

def main():
    config = Configer.load_config("xor.ini")
    print(config)
    pipeline = Pipeline(config)
    forward_func = pipeline.ask()
    # inputs = np.tile(xor_inputs, (150, 1, 1))
    outputs = forward_func(xor_inputs)
    print(outputs)



@jit
def f(x, jit_config):
    return x + jit_config["bias_mutate_rate"]


if __name__ == '__main__':
    main()
