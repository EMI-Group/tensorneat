from typing import Callable, List
from functools import partial

import jax
import numpy as np

from utils import Configer
from algorithms.neat import Pipeline
from time_utils import using_cprofile

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
xor_outputs = np.array([[0], [1], [1], [0]])


def evaluate(forward_func: Callable) -> List[float]:
    """
    :param forward_func: (4: batch, 2: input size) -> (pop_size, 4: batch, 1: output size)
    :return:
    """
    outs = forward_func(xor_inputs)
    outs = jax.device_get(outs)
    fitnesses = 4 - np.sum((outs - xor_outputs) ** 2, axis=(1, 2))
    return fitnesses.tolist()  # returns a list


@using_cprofile
# @partial(using_cprofile, root_abs_path='/mnt/e/neat-jax/', replace_pattern="/mnt/e/neat-jax/")
def main():
    config = Configer.load_config()
    pipeline = Pipeline(config, seed=114514)
    pipeline.auto_run(evaluate)


if __name__ == '__main__':
    main()
