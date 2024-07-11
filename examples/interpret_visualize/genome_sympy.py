import jax, jax.numpy as jnp

from tensorneat.genome import DefaultGenome
from tensorneat.common import *
from tensorneat.common.functions import SympySigmoid

if __name__ == "__main__":
    genome = DefaultGenome(
        num_inputs=3,
        num_outputs=1,
        max_nodes=50,
        max_conns=500,
        output_transform=ACT.sigmoid,
    )

    state = genome.setup()

    randkey = jax.random.PRNGKey(42)
    nodes, conns = genome.initialize(state, randkey)

    network = genome.network_dict(state, nodes, conns)

    input_idx, output_idx = genome.get_input_idx(), genome.get_output_idx()

    res = genome.sympy_func(state, network, sympy_input_transform=lambda x: 999*x, sympy_output_transform=SympySigmoid)
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

    print(genome.forward(state, genome.transform(state, nodes, conns), inputs))

    print(AGG.sympy_module("jax"))
    print(AGG.sympy_module("numpy"))

    print(ACT.sympy_module("jax"))
    print(ACT.sympy_module("numpy"))