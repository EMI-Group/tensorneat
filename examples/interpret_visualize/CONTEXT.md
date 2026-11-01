# Interpret / visualize examples

## Intent
Two host-only demos of genome introspection: turn one genome into sympy formulas / Python code / LaTeX, and draw its topology as an SVG.
Neither script needs `Pipeline`; both run in a couple of seconds and are guarded by `if __name__ == "__main__":`.

## API Surface
- `genome_sympy.py` — `DefaultGenome(num_inputs=3, num_outputs=1, max_nodes=10, max_conns=20, init_hidden_layers=(3,), output_transform=ACT.sigmoid, node_gene=BiasNode(activation_options=ACT.tanh, aggregation_options=AGG.sum))`; prints `repr(state, payload)`, `output_expr.subs(args_symbols)`, `to_python_code(*res)`, `to_latex_code(*res)`.
- `visualize_genome.py` — same genome with `num_outputs=2`; writes `network.svg` next to the script via `Path(__file__).with_name(...)` and prints the path.
- Scripts are run as `python examples/interpret_visualize/<name>.py`.

## Constraints
- One individual is drawn by a single host graph call: `evoxc.build(lambda key: g.initialize(state, key, 1), evoxc.TensorSpec((), evoxc.int64), backend="numpy")` then `evoxc.run(exe, evoxc.random.key(42))`; the result is a `P == 1` payload the host helpers accept.
- Only `DefaultGenome` can be inspected this way: `RecurrentGenome.sympy_func` / `.visualize` raise `TensorNEATError` by design (a cyclic network has no closed form).
- `sympy_func(state, network, ..., backend="numpy")` returns the 8-tuple `(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func, topo_order, useful_nodes)`; `to_python_code` / `to_latex_code` take it positionally.
- `network_dict` / `repr` / `sympy_func` / `visualize` are host-only: never feed their results back into a graph.
- No `jax` / `torch` import or reference anywhere; `network.svg` is a runtime artifact of `visualize_genome.py`, not a tracked file.
