import jax, jax.numpy as jnp

from algorithm.neat import *
from algorithm.neat.genome.dense import DenseInitialize
from utils.graph import topological_sort_python
from utils import *

if __name__ == "__main__":
    genome = DenseInitialize(
        num_inputs=3,
        num_outputs=1,
        max_nodes=50,
        max_conns=500,
    )

    state = genome.setup()

    randkey = jax.random.PRNGKey(42)
    nodes, conns = genome.initialize(state, randkey)

    network = genome.network_dict(state, nodes, conns)

    input_idx, output_idx = genome.get_input_idx(), genome.get_output_idx()

    res = genome.sympy_func(state, network, sympy_input_transform=lambda x: 999999999*x, sympy_output_transform=SympyStandardSigmoid)
    (symbols,
    args_symbols,
    input_symbols,
    nodes_exprs,
    output_exprs,
    forward_func,) = res

    print(symbols)
    print(output_exprs[0].subs(args_symbols))

    inputs = jnp.zeros(3)
    print(forward_func(inputs))
