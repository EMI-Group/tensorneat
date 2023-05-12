import jax
import jax.numpy as jnp
import numpy as np
from algorithms.neat.function_factory import FunctionFactory
from algorithms.neat.genome.debug.tools import check_array_valid
from utils import Configer
from algorithms.neat.jitable_speciate import jitable_speciate
from algorithms.neat.genome.crossover import crossover
from algorithms.neat.genome.utils import I_INT
from time import time

if __name__ == '__main__':
    config = Configer.load_config()
    function_factory = FunctionFactory(config, debug=True)
    initialize_func = function_factory.create_initialize()

    pop_nodes, pop_connections, input_idx, output_idx = initialize_func()
    mutate_func = function_factory.create_mutate(pop_nodes.shape[1], pop_connections.shape[1])
    crossover_func = function_factory.create_crossover(pop_nodes.shape[1], pop_connections.shape[1])

    N, C, species_size = function_factory.init_N, function_factory.init_C, 20
    spe_center_nodes = np.full((species_size, N, 5), np.nan)
    spe_center_connections = np.full((species_size, C, 4), np.nan)
    spe_center_nodes[0] = pop_nodes[0]
    spe_center_connections[0] = pop_connections[0]

    key = jax.random.PRNGKey(0)
    new_node_idx = 100

    while True:
        start_time = time()
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

        # for i in range(len(pop_nodes)):
        #     check_array_valid(pop_nodes[i], pop_connections[i], input_idx, output_idx)

        #speciate next generation

        idx2specie, spe_center_nodes, spe_center_cons = jitable_speciate(pop_nodes, pop_connections, spe_center_nodes, spe_center_connections,
                               compatibility_threshold=2.5)

        idx2specie = np.array(idx2specie)
        spe_dict = {}
        for i in range(len(idx2specie)):
            spe_idx = idx2specie[i]
            if spe_idx not in spe_dict:
                spe_dict[spe_idx] = 1
            else:
                spe_dict[spe_idx] += 1

        print(spe_dict)
        assert np.all(idx2specie != I_INT)
        print(time() - start_time)
        # print(idx2specie)
