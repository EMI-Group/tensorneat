# ./test/unit/common — numpy-backend suites for `tensorneat.common`

## Intent
Prove that the bottom layer of the package computes what its docstrings promise: the execution schedule (`tensorization`), the flat-index kernel helpers (`tools`), the ACT/AGG registries and dispatchers (`functions`), the immutable host container (`state`) and the host graph algorithms (`host_graph`).
Every test body now holds real assertions; there is no `pytest.skip` and no `xfail` in this directory.
Every assertion compares against something *independent* of the implementation it checks: a plain-numpy oracle written in the test module, `support.oracle.schedule`, `host_graph.node_depth` as the counterpart of the in-graph `node_level`, a closed form, or a hand-enumerated literal for a genome small enough to write out.
The suite runs in the default gate (`pytest -m "not parity and not slow"`) on `backend="numpy"`.

## How to run
`/tmp/tn-venv/bin/python -m pytest test/unit/common` from the repo root (`python` is not on PATH in this checkout) — 155 tests, ~2.5 s.
155 = `test_tools.py` 49 + `test_functions.py` 36 + `test_tensorization.py` 22 + `test_host_graph.py` 21 + `test_state.py` 17 + `test_tensorization_validate.py` 10.

## What each module proves
`./test_tensorization.py` (22) — `derive_schedule` on hand-built payloads: the twelve `Transformed` fields with their documented shapes and integer dtypes (`level`/`overflow` are read-only aliases of `node_level`/`edge_overflow`, not extra fields), input nodes pinned at level `0`, the longest-path recursion over the payload's own decoded edges, `I_INF` for absent/isolated/cyclic nodes, incoming-edge windows dense in connection-row order with `I_INF` padding and `in_src`/`in_conn`/`in_valid` agreeing cell for cell, `edge_overflow` counting exactly the dropped edges, field-by-field equality with `support.oracle.schedule`, `topo_key` as a valid execution order, `key_slot_table`/`slot_lookup`, the missing-output-key miss slot, and the `float32`+`NaN` spelling producing the identical schedule as `int32`+`I_INF`.
`./test_tensorization_validate.py` (10) — `validate_schedule` accepts the kernel output and rejects each tampered invariant (field shape, dtype, `I_INF` level with a finite `topo_key` and vice versa, wrong `topo_key` formula, negative `edge_overflow`, out-of-range input slot, live level with dead `in_valid`, `in_conn`/`in_src` disagreement), matched on the specific `ConfigError` message.
`./test_tools.py` (49) — `I_INF`/`KEY_MAX` sentinels, `batch_offsets`, `flat_gather`/`masked_flat_gather`, `flat_scatter`/`masked_flat_scatter`, `fetch_first`, `fetch_random`, `rank_elements`, `argmin_with_mask`, `mutate_float`, `mutate_int` and `hash_rows` against a numpy expression, a Python loop or a stated law, including the out-of-range behaviour (evoxc raises a `ShapeError` where JAX clamped, so the masked helpers sanitize *and* mask), the dump-slot rule for dropped writes, the neutral-element fills, and key-only determinism of both mutators.
`./test_functions.py` (36) — literal `ACT`/`AGG` option tables (12 / 6) and sympy counterparts, `FunctionManager` name↔id↔kernel↔sympy pairing plus its `ValueError`/`IndexError`/`AttributeError` paths and the `get_func_name`-is-absent check, `random_index` staying in `[0, len)` over seeded keys, `apply_activation`/`apply_aggregation` against both a per-option runner loop and `support.functions_ref` for every id (incl. NaN, masked-out slots, all-`False`/all-`True` columns, `maxabs` sign/tie, `mean` real edge count, `-1`/out-of-range ⇒ identity, integral `float32` ids), each aggregation's neutral padding and `has_valid` zero guard, and the sympy counterparts against the kernels through `sp.lambdify` plus `to_python_code` source `exec`-ed against numpy.
`./test_state.py` (17) — `State` access/`KeyError`/`__contains__`/insertion order/legacy `repr`, `register`/`update`/`remove` returning new containers and leaving the original untouched (wrong direction ⇒ `ConfigError`, assignment ⇒ `AttributeError`), `save`/`load` round-trip in `tmp_path` with the raw pickle payload read back by a recording `Unpickler`, error propagation for a missing/garbage/truncated file, `tree_flatten`/`tree_unflatten` pairing and length validation, and evoxc pytree registration through `evoxc.tree_flatten`/`tree_map`/`tree_unflatten`.
`./test_host_graph.py` (21) — `decode_graph` (`NaN` padding skipped, keys cast to `int`, `GraphError` on a duplicate node key / absent endpoint / bad fixed prefix), `topological_sort` (ascending layers, deterministic, argument not mutated, `GraphError` on a cycle or unknown endpoint), `find_useful_nodes` (backward closure, terminates on a cycle), `check_cycles` (self-loop, back-edge, existing-edge and forward-edit cases), `node_depth` (hand-enumerated longest path, `GraphError` on a cycle / unreachable / feedless node).

## Fixtures
`./conftest.py` owns the three lazy inputs: `common` (a plain module-scoped accessor for `tensorneat.common`), and `payloads` / `oracle` (`support.payloads`, `support.oracle`, resolved with `pytest.importorskip` so a bare environment skips instead of erroring).
`seed`, `rng` and `evoxc_key` come from `../conftest.py`; a key-taking helper is wrapped by `test_tools._seeded` / `_run_seeded` because the runner cannot carry an `evoxc.Tensor` key as a graph input.
Module-local fixtures name the genome each test starts from so a test signature states its topology (`tiny`, `chain_payload`, `skip_payload`, `isolated_payload`, `ring_payload`, `sparse_payload`, `state`, `feedforward`, `cyclic`, `gene_rows`).
`test_host_graph.py` adds an `errors` fixture that imports the public `tensorneat.common.errors` rather than the names `host_graph` happens to re-export.

## Gotchas of this module family
`derive_schedule`'s five statics are positional-or-keyword and `evoxc.trace.defn.Defn` is directly callable (it also exposes `.fn`), so `run_defn(derive_schedule, …, int(max_in_degree), int(key_span))` is the whole call — no `functools.partial` dance.
`derive_schedule` accepts **both** key spellings (`int32`+`I_INF` canonical, `float32`+`NaN` the legacy/parity spelling); the tests assert the two spellings yield identical schedules instead of treating the mismatch as a doc conflict.
`support.oracle.schedule` returns the full post-merge `Transformed` field set, so the comparison is field-by-field with no substitution.
`edge_overflow` is *data*, never an exception, and `level == I_INF` is the schedule's way of reporting a cycle where `host_graph.node_depth` raises `GraphError` — a recurrent genome has no depth agreement by design and none is asserted.
A missing output key is *reported* as `output_slot == I_INF` by the kernel; only the host-side helpers raise.
The dispatchers take `act_fns`/`agg_fns` as trace-time static arguments, so a custom option tuple must be bound (e.g. `functools.partial(apply_activation, act_fns=...)`), never passed as a traced argument.
`FunctionManager.random_index` draws `evoxc.random.randint(key, (1,), 0, len)` and reshapes to rank 0, because a 0-d numpy-backend draw is rejected while the output tensor is built.
`sympy_module()` emits only the counterpart classes that define `numerical_eval` (`SympyClip` for ACT, an empty mapping for AGG) — not every registered option.
`to_python_code` emits bare `exp(...)`, so the generated snippet needs numpy's names in its namespace to run.
`test_tensorization_validate.py` is the `validate_schedule` half of a two-module pair covering one source module; it relative-imports the schedule fixtures from `test_tensorization.py`, and pytest resolves a fixture's own dependencies in the *using* module's namespace, so the payload fixtures travel with that import.
A parametrized case may legitimately carry several variants (mutation rates, neutral elements, tampered invariants) — the counts are the design, not an accident.

## Constraints
Backend is always `"numpy"`; no `requires_iree` case lives here (every kernel under test is numpy-backend only, so an iree variant would prove nothing new).
No test module imports `jax`/`torch` at module scope or in a body; `sympy` is imported inside the tests that need it.
`import tensorneat.common` executes `src/tensorneat/__init__.py`, which imports no optional framework, so the guard's baseline stays empty and the per-test diff watches only what a test itself pulls in.
`import tensorneat` is usable, so nothing here is gated behind a "not implemented" skip.
Files stay ≤ ~1000 lines with one source module per file (the tensorization pair is the one documented exception); nothing here is exported or packaged.
A test may not edit `../../../src`, `../../../benchmarks` or `pyproject.toml` — a discovered defect is reported to the parent instead of patched here.

## Routing Table
No subdirectories — this node is a leaf.
Read-only contracts it encodes: `../../../src/tensorneat/common/{tensorization,tools,state,host_graph,errors}.py` and `common/functions/`.
Consumers of its findings: `../../../src/tensorneat/genome` (forward/grad build on the schedule) and `../problem` (fitness on the forward output).
