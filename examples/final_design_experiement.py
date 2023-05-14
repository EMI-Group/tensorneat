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
    function_factory = FunctionFactory(config)
    evaluate_func = lambda func: evaluate(problem, func)

    # precompile
    pipeline = Pipeline(config, function_factory, seed=114514)
    pipeline.auto_run(evaluate_func)

    for r in range(10):
        print(f"running: {r}/{10}")
        tic = time.time()

        pipeline = Pipeline(config, function_factory, seed=r)
        pipeline.auto_run(evaluate_func)

        total_time = time.time() - tic
        evaluate_time = pipeline.evaluate_time
        total_it = pipeline.generation
        print(f"total time: {total_time:.2f}s, evaluate time: {evaluate_time:.2f}s, total_it: {total_it}")

        if total_it >= 500:
            res = "fail"
        else:
            res = "success"

        with open("log", "wb") as f:
            f.write(f"{res}, total time: {total_time:.2f}s, evaluate time: {evaluate_time:.2f}s, total_it: {total_it}\n".encode("utf-8"))
            f.write(str(pipeline.generation_time_list).encode("utf-8"))

    compile_time = function_factory.compile_time

    print("total_compile_time:", compile_time)


if __name__ == '__main__':
    main()
