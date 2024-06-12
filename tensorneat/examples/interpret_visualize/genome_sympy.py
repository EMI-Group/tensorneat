import jax, jax.numpy as jnp

from algorithm.neat import *
from algorithm.neat.genome.advance import AdvanceInitialize
from utils.graph import topological_sort_python

if __name__ == '__main__':
    genome = AdvanceInitialize(
        num_inputs=17,
        num_outputs=6,
        hidden_cnt=8,
        max_nodes=50,
        max_conns=500,
    )

    state = genome.setup()

    randkey = jax.random.PRNGKey(42)
    nodes, conns = genome.initialize(state, randkey)

    network = genome.network_dict(state, nodes, conns)
    print(set(network["nodes"]), set(network["conns"]))
    order, _ = topological_sort_python(set(network["nodes"]), set(network["conns"]))
    print(order)

    input_idx, output_idx = genome.get_input_idx(), genome.get_output_idx()
    print(input_idx, output_idx)

    print(genome.repr(state, nodes, conns))
    print(network)

    res = genome.sympy_func(state, network, precision=3)
    print(res)

