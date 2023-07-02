import jax
import numpy as np

from configs import Configer
from pipeline import Pipeline

xor_inputs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
xor_outputs = np.array([[0], [1], [1], [0], [1], [0], [0], [1]], dtype=np.float32)


def evaluate(forward_func):
    """
    :param forward_func: (4: batch, 2: input size) -> (pop_size, 4: batch, 1: output size)
    :return:
    """
    outs = forward_func(xor_inputs)
    outs = jax.device_get(outs)
    fitnesses = 8 - np.sum((outs - xor_outputs) ** 2, axis=(1, 2))
    return fitnesses


def main():
    config = Configer.load_config("xor3d.ini")
    pipeline = Pipeline(config)
    nodes, cons = pipeline.auto_run(evaluate)
    # g = Genome(nodes, cons, config)
    # print(g)


if __name__ == '__main__':
    main()
