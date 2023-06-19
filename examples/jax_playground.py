from functools import partial

import jax
from jax import jit

from configs import Configer
from neat.pipeline_ import Pipeline


def main():
    config = Configer.load_config("xor.ini")
    print(config)
    pipeline = Pipeline(config)


@jit
def f(x, jit_config):
    return x + jit_config["bias_mutate_rate"]


if __name__ == '__main__':
    main()
