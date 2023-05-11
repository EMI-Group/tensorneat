import jax
import numpy as np
from algorithms.neat.function_factory import FunctionFactory
from algorithms.neat.genome.debug.tools import check_array_valid
from utils import Configer

from algorithms.neat.genome.crossover import crossover


if __name__ == '__main__':
    config = Configer.load_config()
    function_factory = FunctionFactory(config, debug=True)
    initialize_func = function_factory.create_initialize()
    pop_nodes, pop_connections, input_idx, output_idx = initialize_func()
    mutate_func = function_factory.create_mutate(pop_nodes.shape[1], pop_connections.shape[1])
    crossover_func = function_factory.create_crossover(pop_nodes.shape[1], pop_connections.shape[1])
    key = jax.random.PRNGKey(0)
    new_node_idx = 100
    while True:
        key, subkey = jax.random.split(key)
        mutate_keys = jax.random.split(subkey, len(pop_nodes))
        new_nodes = np.arange(new_node_idx, new_node_idx + len(pop_nodes))
        new_node_idx += len(pop_nodes)
        pop_nodes, pop_connections = mutate_func(mutate_keys, pop_nodes, pop_connections, new_nodes)
        pop_nodes, pop_connections = jax.device_get([pop_nodes, pop_connections])
        # for i in range(len(pop_nodes)):
        #     check_array_valid(pop_nodes[i], pop_connections[i], input_idx, output_idx)
        idx1 = np.random.permutation(len(pop_nodes))
        idx2 = np.random.permutation(len(pop_nodes))

        n1, c1 = pop_nodes[idx1], pop_connections[idx1]
        n2, c2 = pop_nodes[idx2], pop_connections[idx2]
        crossover_keys = jax.random.split(subkey, len(pop_nodes))

        # for idx, (zn1, zc1, zn2, zc2) in enumerate(zip(n1, c1, n2, c2)):
        #     n, c = crossover(crossover_keys[idx], zn1, zc1, zn2, zc2)
        #     try:
        #         check_array_valid(n, c, input_idx, output_idx)
        #     except AssertionError as e:
        #         crossover(crossover_keys[idx], zn1, zc1, zn2, zc2)

        pop_nodes, pop_connections = crossover_func(crossover_keys, n1, c1, n2, c2)

        for i in range(len(pop_nodes)):
            check_array_valid(pop_nodes[i], pop_connections[i], input_idx, output_idx)

        print(new_node_idx)


