import jax
import numpy as np

from algorithm.config import Configer
from algorithm.neat import NEAT, NormalGene, Pipeline
from algorithm.neat.genome import create_mutate

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

def single_genome(func, nodes, conns):
    t = NormalGene.forward_transform(nodes, conns)
    out1 = func(xor_inputs[0], t)
    out2 = func(xor_inputs[1], t)
    out3 = func(xor_inputs[2], t)
    out4 = func(xor_inputs[3], t)
    print(out1, out2, out3, out4)

if __name__ == '__main__':
    config = Configer.load_config()
    neat = NEAT(config, NormalGene)
    randkey = jax.random.PRNGKey(42)
    state = neat.setup(randkey)
    forward_func = NormalGene.create_forward(config)
    mutate_func =  create_mutate(config, NormalGene)


    nodes, conns = state.pop_nodes[0], state.pop_conns[0]
    single_genome(forward_func, nodes, conns)
    nodes, conns = mutate_func(state, randkey, nodes, conns, 10000)
    single_genome(forward_func, nodes, conns)


