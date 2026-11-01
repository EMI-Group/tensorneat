# ./test/unit/common — numpy-backend suites for `tensorneat.common`

## Intent
Prove that the bottom layer of the package computes what its docstrings promise: the execution schedule (`tensorization`), the flat-index kernel helpers (`tools`), the ACT/AGG registries and dispatchers (`functions`), the immutable host container (`state`) and the host graph algorithms (`host_graph`).
Every assertion compares against something *independent* of the implementation it checks: a plain-numpy oracle written in the test module, `support.oracle.schedule`, `host_graph.node_depth` as the counterpart of the in-graph `level`, or a hand-enumerated literal for a genome small enough to write out.
The suite runs in the default gate (`pytest -m "not parity and not slow"`) on `backend="numpy"`, with no optional dependency and no `jax`/`torch` in the process.

## What each module proves
`./test_tensorization.py` — `derive_schedule` yields the correct level order (input pinned at `0`, longest-path depths, `I_INF` for absent/unreachable/cyclic nodes) and the correct incoming-edge windows (columns `0..d-1` dense in connection-row order, `I_INF` padding, `in_conn`/`in_src` agreeing cell for cell), `edge_overflow` counts exactly the edges a too-small `max_in_degree` drops, the whole schedule equals `support.oracle.schedule` field by field, `level` agrees with `host_graph.node_depth` on a feedforward genome, and `validate_schedule` accepts the kernel output while rejecting each tampered invariant.
`./test_tools.py` — `batch_offsets`, `flat_gather`, `masked_flat_gather`, `flat_scatter`, `masked_flat_scatter`, `fetch_first`, `fetch_random`, `rank_elements`, `argmin_with_mask`, `mutate_float`, `mutate_int` and `hash_rows` match a numpy expression, a Python loop or a stated law, including the out-of-range behaviour (evoxc raises where JAX clamped, so the masked helpers sanitize *and* mask) and the dump-slot rule for dropped writes.
`./test_functions.py` — the `ACT`/`AGG` option order (gene data, asserted literally), the `FunctionManager` name↔id↔kernel↔sympy pairing and its `ValueError`s, `apply_activation`/`apply_aggregation` against a per-option loop (evoxc and sympy), `-1`⇒identity, each aggregation's neutral padding plus the `has_valid` zero guard, and `sympy_module`/`to_python_code` agreement with the kernels.
`./test_state.py` — `State` immutability (`register`/`update`/`remove` return new containers and reject the wrong direction with `ConfigError`), `KeyError` on a missing entry (and it escapes `hasattr`), the legacy `repr`, `save`/`load` round-trip in `tmp_path`, and the evoxc pytree registration.
`./test_host_graph.py` — `decode_graph` (NaN padding skipped, `int` keys, `GraphError` on a duplicate key or an absent endpoint), `topological_sort` (ascending layers, deterministic, `GraphError` on a cycle), `find_useful_nodes` (backward fixpoint, terminates on a cycle), `check_cycles` (self-loop/back-edge/existing-edge/forward-edit) and `node_depth` (hand-enumerated depths, `GraphError` on a cycle or an unreachable node).

## Fixtures
`./conftest.py` owns the three lazy inputs: `common` (the `tensorneat.common` package under test — `import tensorneat` still raises `NotImplementedError` from the `src/` design skeletons, so the import is deferred and reported as a skip), `payloads` (`support.payloads`) and `oracle` (`support.oracle`), the last two resolved with `pytest.importorskip`.
Module-local fixtures name the genome each test starts from (`tiny`, `chain_payload`, `skip_payload`, `isolated_payload`, `ring_payload`, `state`, `feedforward`, `cyclic`, `gene_rows`) so a test signature states its topology.
`seed`, `rng` and `evoxc_key` come from `../conftest.py`; a key-taking helper is wrapped by `test_tools._seeded` because the runner cannot carry an `evoxc.Tensor` key as a graph input.

## Gotchas of this module family
`derive_schedule`'s statics are keyword-only and `evoxc.build` mirrors a positional call, so the kernel is bound with `functools.partial(...fn, **statics)`; a `Defn` object is not callable, hence the `.fn` unwrap (`test_tensorization._schedule`).
Two key conventions meet here: `tensorization` documents `float32` keys with `NaN` padding while the gene payload (`genome/utils.py`, `support/payloads.py`) stores `int32` keys with `I_INF` padding — `test_tensorization._float_keys` reconciles them in exactly one place (reported upward as a doc conflict; `host_graph.decode_graph` has the same float/NaN signature).
`edge_overflow` is *data*, never an exception, and `level == I_INF` is the schedule's way of reporting a cycle where `host_graph` raises `GraphError` — so a recurrent genome has no depth agreement by design.
Keys are integer-valued and exact; `I_INF` lives in integer arrays only and `NaN` in float arrays only, and no test mixes them.
A parametrized case may legitimately carry five or six variants (mutation rates, neutral elements) — the counts are the design, not an accident.

## Constraints
Backend is always `"numpy"`; no `requires_iree` case lives here (every kernel under test is numpy-backend only, so an iree variant would prove nothing new).
No `jax`/`torch` import at module scope or in a body (`../conftest.py` enforces it per test); `sympy` is imported inside the tests that need it.
Every test has a real name, real fixtures and a docstring naming the invariant and its oracle; until the implementation wave its body is `pytest.skip("not implemented: <what will be asserted>")` naming the owner module/function.
Files stay under ~1000 lines with one concern each; nothing here is exported or packaged.
A test may not edit `../../../src`, `../../../benchmarks` or `pyproject.toml` — a discovered defect is reported to the parent.

## Routing Table
No subdirectories — this node is a leaf.
Read-only contracts it encodes: `../../../src/tensorneat/common/{tensorization,tools,state,host_graph}.py` and `common/functions/`.
Consumers of its findings: `../../../src/tensorneat/genome` (forward/grad build on the schedule) and `../problem` (fitness on the forward output).
