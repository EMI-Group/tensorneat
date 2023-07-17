import jax

from algorithm.config import Configer
from algorithm.neat import NEAT
from algorithm.neat.genome import create_mutate

if __name__ == '__main__':
    config = Configer.load_config()
    neat = NEAT(config)
    randkey = jax.random.PRNGKey(42)
    state = neat.setup(randkey)
    mutate_func = jax.jit(create_mutate(config, neat.gene_type))
    state = mutate_func(state)
    print(state)


