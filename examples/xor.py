from typing import Callable, List

import jax
import numpy as np

from utils import Configer
from algorithms.neat import Pipeline

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([[0], [1], [1], [0]])


def evaluate(forward_func: Callable) -> List[float]:
    """
    :param forward_func: (4: batch, 2: input size) -> (pop_size, 4: batch, 1: output size)
    :return:
    """
    outs = forward_func(xor_inputs)
    outs = jax.device_get(outs)
    fitnesses = np.mean((outs - xor_outputs) ** 2, axis=(1, 2))
    return fitnesses.tolist()  # returns a list


def main():
    config = Configer.load_config()
    pipeline = Pipeline(config)
    forward_func = pipeline.ask(batch=True)
    fitnesses = evaluate(forward_func)
    pipeline.tell(fitnesses)


    # for i in range(100):
    #     forward_func = pipeline.ask(batch=True)
    #     fitnesses = evaluate(forward_func)
    #     pipeline.tell(fitnesses)


if __name__ == '__main__':
    main()
