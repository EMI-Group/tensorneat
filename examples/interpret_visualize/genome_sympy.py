"""Print the symbolic formula, Python code and LaTeX code of one genome (host only)."""
import evoxc

from tensorneat import genome
from tensorneat.common import ACT, AGG
from tensorneat.common.sympy_tools import to_latex_code, to_python_code

if __name__ == "__main__":
    g = genome.DefaultGenome(
        num_inputs=3,
        num_outputs=1,
        max_nodes=10,
        max_conns=20,
        init_hidden_layers=(3,),
        output_transform=ACT.sigmoid,
        node_gene=genome.BiasNode(
            activation_options=ACT.tanh, aggregation_options=AGG.sum
        ),
    )
    state = g.setup()

    # initialize() is a trace-time kernel, so one host graph call draws one individual;
    # pop_size=1 yields the P == 1 payload the host helpers accept directly.
    exe = evoxc.build(
        lambda key: g.initialize(state, key, 1),
        evoxc.TensorSpec((), evoxc.int64),
        backend="numpy",
    )
    payload = evoxc.run(exe, evoxc.random.key(42))

    network = g.network_dict(state, payload)
    print(g.repr(state, payload))

    # The 8-tuple is (symbols, args_symbols, input_symbols, nodes_exprs, output_exprs,
    # forward_func, topo_order, useful_nodes); subs() rewrites the output in terms of inputs.
    res = g.sympy_func(
        state, network, sympy_output_transform=ACT.obtain_sympy(ACT.sigmoid)
    )
    print(res[4][0].subs(res[1]))
    print(to_python_code(*res))
    print(to_latex_code(*res))

    # sympy_func is host-only: it walks the decoded dicts in Python and substitutes gene
    # values into sympy expressions, so it never enters an evoxc graph. A RecurrentGenome
    # refuses it by design -- a cyclic network has no closed-form expression.
