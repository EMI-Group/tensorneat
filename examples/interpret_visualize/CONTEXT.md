# examples/interpret_visualize

## Intent
Three host-side demo scripts around one genome payload.
Two are introspection: turn one genome into sympy formulas / Python code / LaTeX, and draw its topology as an SVG.
The third is the library's exact gradient path through an evolved topology (`genome.grad_from_payload`), verified against central finite differences.
No script needs `Pipeline`; all three are guarded by `if __name__ == "__main__":` (`grad_descent.py` through a `main()`).
Machine-specific wall clock from the repository root with a warm `.venv`, including `uv run` startup: `genome_sympy.py` takes 0.82–0.89 s, `visualize_genome.py` 0.95–0.97 s and `grad_descent.py` 0.62–0.63 s (a first run that must build the venv adds ~2.5 s on top).

## API Surface
- `genome_sympy.py` — `DefaultGenome(num_inputs=3, num_outputs=1, max_nodes=10, max_conns=20, init_hidden_layers=(3,), output_transform=ACT.sigmoid, node_gene=BiasNode(activation_options=ACT.tanh, aggregation_options=AGG.sum))`; prints `g.repr(state, payload)`, then `res[4][0].subs(res[1])`, `to_python_code(*res)` and `to_latex_code(*res)` for `res = g.sympy_func(state, network, sympy_output_transform=ACT.obtain_sympy(ACT.sigmoid))`.
- `visualize_genome.py` — same genome with `num_outputs=2`; writes `network.svg` next to the script via `Path(__file__).with_name(...)` and prints the path.
- `grad_descent.py` — `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=8, max_conns=12, init_hidden_layers=(2,))`, one individual from `evoxc.random.key(0)`, query `[[0.5, -1.0]]`, traced loss `lambda out: (out - 0.5) * (out - 0.5)` (a squared error to the constant-0.5 predictor); prints the genome `repr`, the forward output, both gradient shapes/norms/nonzero counts, then a central-difference check (`eps=1e-3`, the four largest-magnitude entries of each gradient array) and `PASS`.
- Scripts are run from the repository root as `uv run python examples/interpret_visualize/<name>.py`.

## Constraints
- One individual is drawn by a single host graph call: `evoxc.build(lambda key: g.initialize(state, key, 1), evoxc.TensorSpec((), evoxc.int64), backend="numpy")` then `evoxc.run(exe, evoxc.random.key(42))`; the result is a `P == 1` payload the host helpers accept (`grad_descent.py` uses `evoxc.random.key(0)` and converts the leaves with `Tensor.numpy()`).
- Only `DefaultGenome` can be inspected this way: `RecurrentGenome.sympy_func` / `.visualize` raise `TensorNEATError` by design (a cyclic network has no closed form).
- `sympy_func(state, network, ..., backend="numpy")` returns the 8-tuple `(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func, topo_order, useful_nodes)`; `to_python_code` / `to_latex_code` take it positionally.
- `network_dict` / `repr` / `sympy_func` / `visualize` / `grad_from_payload` are host-only: never feed their results back into a graph.
- `grad_from_payload(state, payload, inputs, loss_fn, *, backend="numpy")` returns `(node_attr_grads, conn_attr_grads)`; its `loss_fn` is traced (evoxc tensors and Python scalars only), squaring must be a multiply (a `power` pullback emits `log`), the payload must stay batch-first, and the loss value is not returned — so `grad_descent.py` reads it from one reusable `transform`+`forward` graph it builds itself, which also serves every finite-difference evaluation (a fresh gradient graph is traced per `grad_from_payload` call).
- No `jax` / `torch` import or reference anywhere.

## Test Strategy
`genome_sympy.py` prints the genome `repr`, the substituted sympy formula, the Python code and the LaTeX code, then exits 0.
`visualize_genome.py` prints the genome `repr` and `saved network figure to .../network.svg`, then exits 0.
`grad_descent.py` prints the genome `repr`, forward output `0.33965498`, loss `2.571052e-02`, gradient shapes `(1, 8, 4)` / `(1, 12, 1)` with 6 nonzero entries each, a worst analytic-vs-finite-difference discrepancy of `8.33e-06` against the `1e-3` tolerance, and `PASS`, then exits 0.
`network.svg` is a runtime artefact the script writes next to itself: it is git-ignored by the repo `.gitignore` and never tracked, so a regenerated copy stays out of a commit.
All three scripts need only the base install (`numpy`, `sympy` and `matplotlib` are project dependencies, not extras).
No pytest target lives here: `test/` owns the automated suites.
