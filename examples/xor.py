from neat import FunctionFactory
from utils import Configer
from neat import Pipeline
from problems import Xor
import time


# @using_cprofile
# @partial(using_cprofile, root_abs_path='/mnt/e/neatax/', replace_pattern="/mnt/e/neat-jax/")
def main():
    tic = time.time()
    config = Configer.load_config()
    print(config)
    assert False
    problem = Xor()
    problem.refactor_config(config)
    function_factory = FunctionFactory(config)
    pipeline = Pipeline(config, function_factory, seed=6)
    nodes, cons = pipeline.auto_run(problem.evaluate)
    print(nodes, cons)
    total_time = time.time() - tic
    compile_time = pipeline.function_factory.compile_time
    total_it = pipeline.generation
    mean_time_per_it = (total_time - compile_time) / total_it
    evaluate_time = pipeline.evaluate_time
    print(f"total time: {total_time:.2f}s, compile time: {compile_time:.2f}s, real_time: {total_time - compile_time:.2f}s, evaluate time: {evaluate_time:.2f}s")
    print(f"total it: {total_it}, mean time per it: {mean_time_per_it:.2f}s")


if __name__ == '__main__':
    main()
