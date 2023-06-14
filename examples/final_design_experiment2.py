
import numpy as np
import jax
from utils import Configer
from algorithms.neat import Pipeline
from time_utils import using_cprofile
from algorithms.neat.function_factory import FunctionFactory
from problems import EnhanceLogic
import time


def evaluate(problem, func):
    outs = func(problem.inputs)
    outs = jax.device_get(outs)
    fitnesses = -np.mean((problem.outputs - outs) ** 2, axis=(1, 2))
    return fitnesses


def main():
    config = Configer.load_config()
    problem = EnhanceLogic("xor", n=3)
    problem.refactor_config(config)

    evaluate_func = lambda func: evaluate(problem, func)

    for p in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        config.neat.population.pop_size = p
        tic = time.time()
        function_factory = FunctionFactory(config)
        print(f"running: {p}")

        pipeline = Pipeline(config, function_factory, seed=2)
        pipeline.auto_run(evaluate_func)

        total_time = time.time() - tic
        evaluate_time = pipeline.evaluate_time
        total_it = pipeline.generation
        print(f"total time: {total_time:.2f}s, evaluate time: {evaluate_time:.2f}s, total_it: {total_it}")

        with open("2060_log2", "ab") as f:
            f.write \
                (f"{p}, total time: {total_time:.2f}s, compile time: {function_factory.compile_time:.2f}s, total_it: {total_it}\n".encode
                    ("utf-8"))
            f.write(f"{str(pipeline.generation_time_list)}\n".encode("utf-8"))

    compile_time = function_factory.compile_time

    print("total_compile_time:", compile_time)


if __name__ == '__main__':
    main()
