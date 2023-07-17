import jax

from algorithm.config import Configer
from algorithm.neat import NEAT

if __name__ == '__main__':
    config = Configer.load_config()
    neat = NEAT(config)
    randkey = jax.random.PRNGKey(42)
    state = neat.setup(randkey)
    state = neat.mutate(state)
    print(state)
    pop_nodes, pop_conns = state.pop_nodes, state.pop_conns
    print(neat.distance(state, pop_nodes[0], pop_conns[0], pop_nodes[1], pop_conns[1]))
    print(neat.crossover(state, pop_nodes[0], pop_conns[0], pop_nodes[1], pop_conns[1]))


