import time

import jax.random

from utils import Configer
from algorithms.neat.genome.genome import *

from algorithms.neat.species import SpeciesController
from algorithms.neat.genome.forward import create_forward_function
from algorithms.neat.genome.mutate import create_mutate_function

if __name__ == '__main__':
    N = 10
    pop_nodes, pop_connections, input_idx, output_idx = initialize_genomes(10000, N, 2, 1,
                                                                           default_act=9, default_agg=0)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # forward = create_forward_function(pop_nodes, pop_connections, 5, input_idx, output_idx, batch=True)
    nodes, connections = pop_nodes[0], pop_connections[0]

    forward = create_forward_function(pop_nodes, pop_connections, N, input_idx, output_idx, batch=True)
    out = forward(inputs)
    print(out.shape)
    print(out)

    config = Configer.load_config()
    s_c = SpeciesController(config.neat)
    s_c.speciate(pop_nodes, pop_connections, 0)
    s_c.speciate(pop_nodes, pop_connections, 0)
    print(s_c.genome_to_species)

    start = time.time()
    for i in range(100):
        print(i)
        s_c.speciate(pop_nodes, pop_connections, i)
    print(time.time() - start)

    seed = jax.random.PRNGKey(42)
    mutate_func = create_mutate_function(config, input_idx, output_idx, batch=False)
    print(nodes, connections, sep='\n')
    print(*mutate_func(seed, nodes, connections, 100), sep='\n')

    randseeds = jax.random.split(seed, 10000)
    new_node_keys = jax.random.randint(randseeds[0], minval=0, maxval=10000, shape=(10000,))
    batch_mutate_func = create_mutate_function(config, input_idx, output_idx, batch=True)
    pop_nodes, pop_connections = batch_mutate_func(randseeds, pop_nodes, pop_connections, new_node_keys)
    print(pop_nodes, pop_connections, sep='\n')

    start = time.time()
    for i in range(100):
        print(i)
        pop_nodes, pop_connections = batch_mutate_func(randseeds, pop_nodes, pop_connections, new_node_keys)
    print(time.time() - start)

    print(nodes, connections, sep='\n')
    nodes, connections = add_node(6, nodes, connections)
    nodes, connections = add_node(7, nodes, connections)
    print(nodes, connections, sep='\n')

    nodes, connections = add_connection(6, 7, nodes, connections)
    nodes, connections = add_connection(0, 7, nodes, connections)
    nodes, connections = add_connection(1, 7, nodes, connections)
    print(nodes, connections, sep='\n')

    nodes, connections = delete_connection(6, 7, nodes, connections)
    print(nodes, connections, sep='\n')

    nodes, connections = delete_node(6, nodes, connections)
    print(nodes, connections, sep='\n')

    nodes, connections = delete_node(7, nodes, connections)
    print(nodes, connections, sep='\n')
