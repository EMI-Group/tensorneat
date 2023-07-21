import jax
import numpy as np

from pipeline import Pipeline
from config import Configer
from algorithm import NEAT, HyperNEAT
from algorithm.neat import RecurrentGene
from algorithm.hyperneat import BaseSubstrate

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
    algorithm = HyperNEAT(config, RecurrentGene, BaseSubstrate)
    pipeline = Pipeline(config, algorithm)
    pipeline.auto_run(evaluate)


if __name__ == '__main__':
    main()
