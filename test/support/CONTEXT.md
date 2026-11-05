# ./test/support — shared, host-side test infrastructure

## Intent
Hold everything more than one suite needs, so no test module re-invents it: the one correct way to run an evoxc graph from pytest, hand-built gene payloads (so a test controls topology instead of drawing it), and the numpy reference implementations the assertions are written against.
This is test code: it is imported (`from support.runner import run_defn ...`) and never packaged, and it must stay free of `jax`/`torch` and of any `backend` other than the numpy reference interpreter.
`test_lazy.py` is the single exception to "imported, never collected": it is the pytest module that pins `lazy.py`'s skip contract.
`conftest.py` puts this directory's parent on `sys.path`, so `import support.<module>` works from every suite.

## API Surface
### `runner.py` (implemented)
`build_defn(fn, *args, backend="numpy", **options)` → reusable `Executable`; a `numpy.ndarray` argument becomes a `TensorSpec`, every other leaf is a static baked into the trace.
`run_defn(fn, *args, backend="numpy", **options)` → build + run, returns host numpy leaves (namedtuples keep their type, so a `Transformed` comes back readable).
`run_graph(graph, *arrays, backend="numpy")` → `lower`→`compile`→`load`→`run` for an already traced `evoxc.Graph` (what `genome.grad` returns).
`compile_graph(graph, backend="numpy")`, `to_numpy(tree)`, `spec_like(array)`, `tensor(array)`, `DEFAULT_BACKEND = "numpy"`.
Arguments may be nested pytrees of arrays (a `Transformed` namedtuple, a dict of payloads): `_split_arg` recurses, so the spec tree always mirrors the runtime tree; the spec shape is read off the *wrapped* tensor because `evoxc.from_numpy` reports a 0-d array as shape `(1,)`.
Rules the runner exists to encode: statics must be passed **again** at `run` (they are baked per call here), an `Executable` is not callable, and `evoxc.Tensor` has no `__array__`.
A test that needs a *method* inside a graph closes over the host objects and passes only tensors/statics: `run_defn(lambda nk, ck: genome.transform(state, nk, ck), node_keys, conn_keys)` — never nest two `@evoxc.defn` functions.
### `lazy.py` (implemented)
`require(module: str, *, allow_pending: bool = False) -> ModuleType` imports a module at call time inside a fixture, so a failing import is attributed to the test that needed it.
Every error propagates by default, `NotImplementedError` included: a pending implementation in the library under test is a defect, not a skip.
`allow_pending=True` converts only a `NotImplementedError` whose innermost traceback frame is a file inside `LIBRARY_DIR` into `pytest.skip`, with a message naming the dotted module, the absolute source file, the raising line and the raising function name.
`LIBRARY_DIR` is derived from this file's location (`Path(__file__).resolve().parents[2] / "src" / "tensorneat"`), never hardcoded, and is a plain module attribute, so a test monkeypatches it to drive the in-library case without touching `src/`.
A `NotImplementedError` raised anywhere else — a `support` module, `evoxc`, site-packages — propagates even with `allow_pending=True`, and `ImportError` and every other error always propagate.
The exported surface is `__all__ == ["require"]`; `LIBRARY_DIR` is a documented module attribute, not an export.
Every suite's `conftest.py` resolves its package through `require(<package>)` without the flag inside a module-scoped fixture.
### `test_lazy.py` (implemented)
The one pytest module of this node: it pins the skip channel above so it cannot hide a real pending implementation (9 tests, measured `9 passed in 0.4 s`).
Every test that needs a subject module writes it on disk under `tmp_path`, inserts that directory at the front of `sys.path` through the `tmp_sys_path` fixture, and purges the temporary packages from `sys.modules` afterwards, so no test depends on the current state of `src/tensorneat`.
It pins `LIBRARY_DIR`'s derivation from this file's location and `__all__`; default-mode propagation of `NotImplementedError`; the in-library opt-in skip, including the module, file, line 1 and `in <module>()` in the message; propagation of a `NotImplementedError` raised outside `LIBRARY_DIR` with `allow_pending=True` (the anti-hole case); `ImportError` propagation in both modes and from an in-library module (only `NotImplementedError` converts); and identity of a healthy import with `importlib.import_module`.
### `payloads.py` (implemented)
`I_INF` — the `int32` sentinel re-exported from `tensorneat.common.tools`, so a payload and the kernels agree on one value.
`new_payload(pop_size=1, max_nodes=8, max_conns=16, *, node_attr_width=4, conn_key_width=2, conn_attr_width=1) -> Payload` — `I_INF`-padded `node_keys`/`conn_keys` (`int32`) and NaN-padded `node_attrs`/`conn_attrs` (`float32`).
The default widths are the real gene layouts (`NA=4` = `bias, response, aggregation, activation`; `CK=2` = `(input_index, output_index)`; `CA=1` = `weight`), and a `NA==3` payload reproduces the pre-rewrite node layout that has no activation column.
`Payload` — the four arrays in `GenomeTensors` order with `.args()` (a tuple for `run_defn`), `.of(p)` (a **one-individual batch**, `P == 1`, so it feeds straight back into a kernel) and `__getitem__` (`int` → `.of(p)`, `str` → the field array).
`Payload` also exposes `.pop_size`/`.max_nodes`/`.max_conns`/`.node_attr_width`/`.conn_key_width`/`.conn_attr_width`/`.field_names`, and `__repr__` prints the three shapes.
`set_node(payload, p, row, key, attrs=None)` / `set_conn(payload, p, row, in_key, out_key, attrs=None)` — write one gene row in place, host-side, with attribute count validated against the payload widths.
`attrs=None` writes the neutral row (`(bias=0, response=1, aggregation=0, activation=0)` / `(weight=1)`), and `key == I_INF` (any endpoint `I_INF` for `set_conn`) *clears* the row back to the sentinels instead of writing garbage.
`set_conn` writes the connection's row index into the key columns past `(input_index, output_index)` — the documented `historical_marker` of `OriginConn`.
`default_genome(**kwargs)` / `recurrent_genome(**kwargs)` / `genome_state(genome)` — a `DefaultGenome`/`RecurrentGenome` with default genes and operators over `num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16`, and the `State` from its `setup(State())`.
The two factories only construct the genome and call `setup`; any error a `src` constructor raises propagates untouched.
`chain_payload(...)`, `skip_payload(...)`, `isolated_payload(...)`, `ring_payload(...)` — the four canonical topologies the forward/grad tests start from (dense chain, a node bypassed by its consumers, a node with no edges, a recurrent cycle).
Each topology builder takes `pop_size`, `max_nodes`, `max_conns`, `num_inputs`, `num_outputs`, the three widths and `weight`, uses node key == row index, and documents the exact rows/edges/levels it writes in its docstring.
### `functions_ref.py` (implemented)
The numpy kernels of the two option registries, split out because a registry is not a network algorithm and `oracle.py` must stay inside the line budget.
`ACT_FUNCS` — the 12 activations in the documented order (`scaled_sigmoid, sigmoid, scaled_tanh, tanh, sin, relu, lelu, identity, inv, log, exp, abs`) with the `ACT_SCALE`/`LELU_SLACK`/`INV_EPS`/`LOG_EPS` constants, and `AGG_FUNCS` — the 6 aggregations (`sum, product, max, min, maxabs, mean`); option **ids** are positions in the configured list, `-1` is identity for an activation and never an aggregation id.
`apply_activation(z, act_ids, act_fns=(act_sigmoid,))` / `apply_aggregation(z, mask, agg_ids, agg_fns=(agg_sum,))` — dispatch per row, neutral-pad the masked-out slots per option (a fully dead row aggregates to exactly `0.0`), and fall back to `0.0` for an out-of-range id.
`oracle.py` re-exports all of it, so a test may import kernels and algorithms from `support.oracle` alone.
### `oracle.py` (implemented)
`schedule(node_keys, conn_in, conn_out, input_keys, output_keys, *, max_in_degree, key_span)` → a `Transformed`-shaped namedtuple of numpy arrays derived independently of the evoxc kernel (dense key→slot table, in-group rank, counting trick, longest-path levels), for `test/unit/common/test_tensorization.py` and for cross-checking the forward oracle.
`schedule` accepts either sentinel for an absent row (`I_INF` from the payloads, or the `NaN` `derive_schedule` documents), and reports a node on or downstream of a cycle as `level = I_INF`.
`default_node_attrs_of(node_attrs)` → `(bias, response, agg_ids, act_ids)` and `default_conn_attr_of(conn_attrs)` → `(P, E)` weights are the column readers `forward` uses by default; a test overrides them for another gene layout.
`forward(payload, transformed, inputs, *, node_attrs_of=None, conn_attr_of=None, activation=None, aggregation=None, is_output=None, network_type="feedforward", activate_time=10)` → `(P, num_outputs)` NaN-aware numpy forward that follows the documented level-synchronous algorithm (NaN→0 sanitization, neutral-padded aggregation, `has_valid` guard, output nodes not activated by the node gene).
`activation`/`aggregation` are either one callable or a sequence of callables (the configured option list the id columns index), and `is_output` defaults to the last `num_outputs` node slots.
`func_fit_loss(pred, targets, error_method="mse")` → per-individual `(P,)` loss, matching `problem/func_fit`'s reduction over sample *and* output axes; `FUNC_FIT_ERROR_METHODS` names the four accepted strings.
`cartpole_params(**knobs)` → the documented Gym `CartPole-v1` constants (`theta_threshold` in **degrees**, `length` = half-pole), and `cartpole_rollout(actions, obs0, *, max_step=500, params=None)` → per-episode return of the documented Euler dynamics with the frozen-state-after-death rule.
No distance or mutation oracle here: parity compares the real operators against the frozen legacy library, and `./unit/algorithm` builds the controlled distance table it needs inside its own module.

## Constraints
No `jax`, no `torch`, no network, no GPU; `runner.py` is the only module allowed to call `evoxc.build`/`evoxc.run`/`evoxc.lower`/`evoxc.compile`/`evoxc.load`.
Oracles are written from the *documented algorithm* (docstrings in `../../src/tensorneat/`), never by calling the implementation they check — otherwise the test is a tautology.
Files ≤ ~1000 lines, one concern per file; helpers take plain numpy in and return plain numpy out so a test can print them on failure.
`./test_lazy.py` is the only file here pytest collects and the only one that asserts; every other module is imported, never collected, and never asserts.

## Routing Table
No subdirectories — this node is a leaf.
Consumers: `../unit/`, `../parity/`, `../integration/` and `../release_gate_no_jax.py`; inside this node, `./test_lazy.py` pins `lazy.py`.
Read-only contracts it encodes: `../../src/tensorneat/common/tensorization.py` (`Transformed`), `../../src/tensorneat/genome/utils.py` (`GenomeTensors`), `../../src/tensorneat/problem/` (loss + dynamics).
