from functools import partial

from utils import Configer
from algorithms.neat import Pipeline
from time_utils import using_cprofile
from problems import Sin, Xor, DIY


@using_cprofile
# @partial(using_cprofile, root_abs_path='/mnt/e/neatax/', replace_pattern="/mnt/e/neat-jax/")
def main():
    config = Configer.load_config()
    problem = Xor()
    problem.refactor_config(config)
    pipeline = Pipeline(config, seed=1)
    pipeline.auto_run(problem.evaluate)


if __name__ == '__main__':
    main()
