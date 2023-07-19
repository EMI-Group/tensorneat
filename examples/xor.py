import jax
import numpy as np

from algorithm import Configer, NEAT
from algorithm.neat import NormalGene, RecurrentGene, Pipeline

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
xor_outputs = np.array([[0], [1], [1], [0]], dtype=np.float32)


def evaluate(forward_func):
    """
    :param forward_func: (4: batch, 2: input size) -> (pop_size, 4: batch, 1: output size)
    :return:
    """
    outs = forward_func(xor_inputs)
    outs = jax.device_get(outs)
    fitnesses = 4 - np.sum((outs - xor_outputs) ** 2, axis=(1, 2))
    return fitnesses


def main():
    config = Configer.load_config("xor.ini")
    # algorithm = NEAT(config, NormalGene)
    algorithm = NEAT(config, RecurrentGene)
    pipeline = Pipeline(config, algorithm)
    best = pipeline.auto_run(evaluate)
    print(best)


if __name__ == '__main__':
    main()
