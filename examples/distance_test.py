from typing import Callable, List
from functools import partial

import numpy as np

from utils import Configer
from algorithms.neat.genome.numpy import analysis, distance
from algorithms.neat.genome.numpy import create_initialize_function, create_mutate_function


def real_distance(nodes1, connections1, nodes2, connections2, input_idx, output_idx):
    nodes1, connections1 = analysis(nodes1, connections1, input_idx, output_idx)
    nodes2, connections2 = analysis(nodes2, connections2, input_idx, output_idx)
    compatibility_coe = 0.5
    disjoint_coe = 1.
    node_distance = 0.0
    if nodes1 or nodes2:  # otherwise, both are empty
        disjoint_nodes = 0
        for k2 in nodes2:
            if k2 not in nodes1:
                disjoint_nodes += 1

        for k1, n1 in nodes1.items():
            n2 = nodes2.get(k1)
            if n2 is None:
                disjoint_nodes += 1
            else:
                if n1[0] is None:
                    continue
                d = abs(n1[0] - n2[0]) + abs(n1[1] - n2[1])
                d += 1 if n1[2] != n2[2] else 0
                d += 1 if n1[3] != n2[3] else 0
                node_distance += d

        max_nodes = max(len(nodes1), len(nodes2))
        node_distance = (compatibility_coe * node_distance + disjoint_coe * disjoint_nodes) / max_nodes

    connection_distance = 0.0
    if connections1 or connections2:
        disjoint_connections = 0
        for k2 in connections2:
            if k2 not in connections1:
                disjoint_connections += 1

        for k1, c1 in connections1.items():
            c2 = connections2.get(k1)
            if c2 is None:
                disjoint_connections += 1
            else:
                # Homologous genes compute their own distance value.
                d = abs(c1[0] - c2[0])
                d += 1 if c1[1] != c2[1] else 0
                connection_distance += d
        max_conn = max(len(connections1), len(connections2))
        connection_distance = (compatibility_coe * connection_distance + disjoint_coe * disjoint_connections) / max_conn

    return node_distance + connection_distance


def main():
    config = Configer.load_config()
    keys_idx = config.basic.num_inputs + config.basic.num_outputs
    pop_size = config.neat.population.pop_size
    init_func = create_initialize_function(config)
    pop_nodes, pop_connections, input_idx, output_idx = init_func()

    mutate_func = create_mutate_function(config, input_idx, output_idx, batch=True)

    while True:
        pop_nodes, pop_connections = mutate_func(pop_nodes, pop_connections, list(range(keys_idx, keys_idx + pop_size)))
        keys_idx += pop_size
        for i in range(pop_size):
            for j in range(pop_size):
                nodes1, connections1 = pop_nodes[i], pop_connections[i]
                nodes2, connections2 = pop_nodes[j], pop_connections[j]
                numpy_d = distance(nodes1, connections1, nodes2, connections2)
                real_d = real_distance(nodes1, connections1, nodes2, connections2, input_idx, output_idx)
                assert np.isclose(numpy_d, real_d), f'{numpy_d} != {real_d}'
                print(numpy_d, real_d)


if __name__ == '__main__':
    np.random.seed(0)
    main()
