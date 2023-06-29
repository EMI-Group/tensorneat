import numpy as np

from configs import Configer
from algorithms.neat import Genome
from pipeline import Pipeline

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
xor_outputs = np.array([[0], [1], [1], [0]], dtype=np.float32)


def evaluate(forward_func):
    """
    :param forward_func: (4: batch, 2: input size) -> (pop_size, 4: batch, 1: output size)
    :return:
    """
    outs = forward_func(xor_inputs)
    fitnesses = 4 - np.sum((outs - xor_outputs) ** 2, axis=(1, 2))
    return np.array(fitnesses)  # returns a list


def main():
    config = Configer.load_config("xor.ini")
    pipeline = Pipeline(config, seed=6)
    nodes, cons = pipeline.auto_run(evaluate)
    g = Genome(nodes, cons, config)
    print(g)


if __name__ == '__main__':
    main()
