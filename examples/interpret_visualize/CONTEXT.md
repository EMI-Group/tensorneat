# examples/interpret_visualize

## Intent
Two host-only demos of genome introspection: turn one genome into sympy formulas / Python code / LaTeX, and draw its topology as an SVG.
Neither script needs `Pipeline`; both are guarded by `if __name__ == "__main__":`.
`genome_sympy.py` takes 0.85 s and `visualize_genome.py` 0.95 s on this CPU, wall clock including `uv run` startup.

## API Surface
- `genome_sympy.py` — `DefaultGenome(num_inputs=3, num_outputs=1, max_nodes=10, max_conns=20, init_hidden_layers=(3,), output_transform=ACT.sigmoid, node_gene=BiasNode(activation_options=ACT.tanh, aggregation_options=AGG.sum))`; prints `g.repr(state, payload)`, then `res[4][0].subs(res[1])`, `to_python_code(*res)` and `to_latex_code(*res)` for `res = g.sympy_func(state, network, sympy_output_transform=ACT.obtain_sympy(ACT.sigmoid))`.
- `visualize_genome.py` — same genome with `num_outputs=2`; writes `network.svg` next to the script via `Path(__file__).with_name(...)` and prints the path.
- Scripts are run from the repository root as `uv run python examples/interpret_visualize/<name>.py`.

## Constraints
- One individual is drawn by a single host graph call: `evoxc.build(lambda key: g.initialize(state, key, 1), evoxc.TensorSpec((), evoxc.int64), backend="numpy")` then `evoxc.run(exe, evoxc.random.key(42))`; the result is a `P == 1` payload the host helpers accept.
- Only `DefaultGenome` can be inspected this way: `RecurrentGenome.sympy_func` / `.visualize` raise `TensorNEATError` by design (a cyclic network has no closed form).
- `sympy_func(state, network, ..., backend="numpy")` returns the 8-tuple `(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func, topo_order, useful_nodes)`; `to_python_code` / `to_latex_code` take it positionally.
- `network_dict` / `repr` / `sympy_func` / `visualize` are host-only: never feed their results back into a graph.
- No `jax` / `torch` import or reference anywhere.

## Test Strategy
`genome_sympy.py` prints the genome `repr`, the substituted sympy formula, the Python code and the LaTeX code, then exits 0.
`visualize_genome.py` prints the genome `repr` and `saved network figure to .../network.svg`, then exits 0.
`network.svg` is a runtime artefact the script writes next to itself: it is git-ignored by the repo `.gitignore` and never tracked, so a regenerated copy stays out of a commit.
Both demos need only the base install (`sympy` and `matplotlib` are project dependencies, not extras).
No pytest target lives here: `test/` owns the automated suites.
