from typing import Callable, List
import time

import numpy as np

from configs import Configer
from neat import Pipeline

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([[0], [1], [1], [0]])


def evaluate(forward_func: Callable) -> List[float]:
    """
    :param forward_func: (4: batch, 2: input size) -> (pop_size, 4: batch, 1: output size)
    :return:
    """
    outs = forward_func(xor_inputs)
    fitnesses = 4 - np.sum((outs - xor_outputs) ** 2, axis=(1, 2))
    # print(fitnesses)
    return fitnesses.tolist()  # returns a list


# @using_cprofile
# @partial(using_cprofile, root_abs_path='/mnt/e/neatax/', replace_pattern="/mnt/e/neat-jax/")
def main():
    tic = time.time()
    config = Configer.load_config("xor.ini")
    print(config)
    function_factory = FunctionFactory(config)
    pipeline = Pipeline(config, function_factory, seed=6)
    nodes, cons = pipeline.auto_run(evaluate)
    print(nodes, cons)
    total_time = time.time() - tic
    compile_time = pipeline.function_factory.compile_time
    total_it = pipeline.generation
    mean_time_per_it = (total_time - compile_time) / total_it
    evaluate_time = pipeline.evaluate_time
    print(
        f"total time: {total_time:.2f}s, compile time: {compile_time:.2f}s, real_time: {total_time - compile_time:.2f}s, evaluate time: {evaluate_time:.2f}s")
    print(f"total it: {total_it}, mean time per it: {mean_time_per_it:.2f}s")


if __name__ == '__main__':
    main()
