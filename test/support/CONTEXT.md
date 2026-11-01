# ./test/support — shared, host-side test infrastructure

## Intent
Hold everything more than one suite needs, so no test module re-invents it: the one correct way to run an evoxc graph from pytest, hand-built gene payloads (so a test controls topology instead of drawing it), and the numpy reference implementations the assertions are written against.
This is test code: it is imported (`from support.runner import run_defn ...`) and never packaged, and it must stay free of `jax`/`torch` and of any `backend` other than the numpy reference interpreter.
`conftest.py` puts this directory's parent on `sys.path`, so `import support.<module>` works from every suite.

## API Surface
### `runner.py` (implemented)
`build_defn(fn, *args, backend="numpy", **options)` → reusable `Executable`; a `numpy.ndarray` argument becomes a `TensorSpec`, every other leaf is a static baked into the trace.
`run_defn(fn, *args, backend="numpy", **options)` → build + run, returns host numpy leaves (namedtuples keep their type, so a `Transformed` comes back readable).
`run_graph(graph, *arrays, backend="numpy")` → `lower`→`compile`→`load`→`run` for an already traced `evoxc.Graph` (what `genome.grad` returns).
`compile_graph(graph, backend="numpy")`, `to_numpy(tree)`, `spec_like(array)`, `tensor(array)`, `DEFAULT_BACKEND = "numpy"`.
Rules the runner exists to encode: statics must be passed **again** at `run` (they are baked per call here), an `Executable` is not callable, and `evoxc.Tensor` has no `__array__`.
A test that needs a *method* inside a graph closes over the host objects and passes only tensors/statics: `run_defn(lambda nk, ck: genome.transform(state, nk, ck), node_keys, conn_keys)` — never nest two `@evoxc.defn` functions.
### `payloads.py`
`I_INF` — the `int32` sentinel re-exported from `tensorneat.common.tools`, so a payload and the kernels agree on one value.
`new_payload(pop_size, max_nodes, max_conns, *, node_attr_width=3, conn_key_width=3, conn_attr_width=1) -> Payload` — `I_INF`-padded `node_keys`/`conn_keys` (`int32`) and NaN-padded `node_attrs`/`conn_attrs` (`float32`).
`Payload` — the four arrays in `GenomeTensors` order with `.args()` (a tuple for `run_defn`), `.of(p)` (one individual's view) and `__getitem__`.
`set_node(payload, p, row, key, attrs=None)` / `set_conn(payload, p, row, in_key, out_key, attrs=None)` — write one gene row in place, host-side, with attribute count validated against the payload widths.
`default_genome(**kwargs)` / `recurrent_genome(**kwargs)` / `genome_state(genome)` — a `DefaultGenome`/`RecurrentGenome` with default genes and operators over `num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16`, and the `State` from its `setup()`.
`chain_payload(...)`, `skip_payload(...)`, `isolated_payload(...)`, `ring_payload(...)` — the four canonical topologies the forward/grad tests start from (dense chain, a node bypassed by its consumers, a node with no edges, a recurrent cycle).
### `oracle.py`
`schedule(node_keys, conn_in, conn_out, input_keys, output_keys, *, max_in_degree, key_span)` → a `Transformed`-shaped namedtuple of numpy arrays derived independently of the evoxc kernel (dense key→slot table, in-group rank, counting trick, longest-path levels), for `test/unit/common/test_tensorization.py` and for cross-checking the forward oracle.
`forward(payload, transformed, inputs, *, node_attrs_of, conn_attr_of, activation, aggregation, is_output, network_type="feedforward", activate_time=10)` → `(P, num_outputs)` NaN-aware numpy forward that follows the documented level-synchronous algorithm (NaN→0 sanitization, neutral-padded aggregation, `has_valid` guard, output nodes not activated by the node gene).
`func_fit_loss(pred, targets, error_method)` → per-individual loss, matching `problem/func_fit`'s reduction over sample *and* output axes.
`cartpole_rollout(actions, obs0, *, max_step, params)` → per-episode return of the documented Euler dynamics with the frozen-state-after-death rule.
`mutate_float_oracle(...)`, `distance_oracle(...)` — only if a unit test needs them; parity covers the real ones.

## Constraints
No `jax`, no `torch`, no network, no GPU; `runner.py` is the only module allowed to call `evoxc.build`/`evoxc.run`/`evoxc.lower`/`evoxc.compile`/`evoxc.load`.
Oracles are written from the *documented algorithm* (docstrings in `../../src/tensorneat/`), never by calling the implementation they check — otherwise the test is a tautology.
Files ≤ ~1000 lines, one concern per file; helpers take plain numpy in and return plain numpy out so a test can print them on failure.
Nothing here is collected by pytest (no `test_*` names) and nothing here asserts.

## Routing Table
No subdirectories — this node is a leaf.
Consumers: `../unit/`, `../parity/`, `../integration/` and `../release_gate_no_jax.py`.
Read-only contracts it encodes: `../../src/tensorneat/common/tensorization.py` (`Transformed`), `../../src/tensorneat/genome/utils.py` (`GenomeTensors`), `../../src/tensorneat/problem/` (loss + dynamics).
