import numpy as np
import jax
from utils import Configer
from neat import Pipeline
from neat import FunctionFactory
from problems import EnhanceLogic
import time


def evaluate(problem, func):
    inputs = problem.ask_for_inputs()
    pop_predict = jax.device_get(func(inputs))
    # print(pop_predict)
    fitnesses = []
    for predict in pop_predict:
        f = problem.evaluate_predict(predict)
        fitnesses.append(f)
    return np.array(fitnesses)


# @using_cprofile
# @partial(using_cprofile, root_abs_path='/mnt/e/neatax/', replace_pattern="/mnt/e/neat-jax/")
def main():
    tic = time.time()
    config = Configer.load_config()
    problem = EnhanceLogic("xor", n=3)
    problem.refactor_config(config)
    function_factory = FunctionFactory(config)
    evaluate_func = lambda func: evaluate(problem, func)
    pipeline = Pipeline(config, function_factory, seed=33413)
    print("start run")
    pipeline.auto_run(evaluate_func)

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
