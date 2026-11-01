# ./test — the pytest suite of the evoxc rewrite

## Intent
Prove three separate things about the rewrite: the evoxc kernels are *correct* (unit, against numpy oracles), they are *equivalent* to the frozen legacy library where the legacy semantics are well defined (parity, against `benchmarks/legacy_jax`), and the installed package is *JAX-free and runnable end to end* (release gate + integration).
Every suite reads `../src/tensorneat` and `../benchmarks/legacy_jax` as read-only inputs; nothing in `src/` imports anything from here.
`./support` holds the host-side helpers the suites share (one graph runner, hand-built gene payloads, numpy reference oracles); it is test code and is not packaged.

## How to run
Commands an owner runs from the repo root (`[tool.pytest.ini_options]` in `pyproject.toml` sets `testpaths = ["test"]`, `addopts = "-q"`):
- `pytest -m "not parity and not slow"` — the default gate: `./unit` plus the release gate, numpy backend only, no optional dependency, target under a minute.
- `pytest -m parity` — kernel equality against the frozen legacy JAX implementation; needs `pip install -e ".[bench]"`, and every parity module calls `pytest.importorskip("jax")` so a bare environment skips rather than errors.
- `pytest -m slow` — `./integration` (XOR evolution, determinism, rebuild-on-shape-change, checkpoint round-trip) and the oversized-population unit cases.
- `pytest test/release_gate_no_jax.py` — the import-hygiene gate alone; it runs subprocesses and belongs to the default gate.
Markers `parity` and `slow` are declared in `pyproject.toml`; `requires_iree` is declared in `conftest.py` and is in no gate (it needs `evoxc[iree]`).

## Layout
```
test/
  conftest.py            sys.path guards (src/, benchmarks/legacy_jax), marker registration, seed fixture
  release_gate_no_jax.py the import-hygiene gate (collected by a conftest hook, see Constraints)
  support/               shared host-side test infrastructure (runner, payloads, numpy oracles)
  unit/                  numpy-backend kernel and invariant tests, mirrors src/tensorneat/
    common/  genome/  algorithm/  problem/
  parity/                same kernels vs the frozen legacy JAX library, marker `parity`
  integration/           whole-Pipeline runs, marker `slow`
```

## What each suite proves
- `./unit/common` — the schedule derived from a hand-built genome has the right level order, incoming-edge windows and sentinels, `validate_schedule` accepts what `derive_schedule` produces and rejects tampered schedules, flat gather/scatter/mutation helpers agree with a plain-numpy oracle, the ACT/AGG dispatchers agree with a per-option Python loop, and host graph walks (`topological_sort`, `node_depth`) agree with the in-graph `level`.
- `./unit/genome` — forward values on hand-built genomes (dense chain, skipped node, unconnected node ⇒ NaN propagation, activation `-1` ⇒ identity), the recurrent tick, `grad` against the numpy oracle, and the genetic operators only through their *invariants* (capacity, key uniqueness, in-degree bound, structural stability of crossover).
- `./unit/algorithm` — speciation, stagnation, spawn allocation and parent selection invariants: `sum(spawn) == pop_size`, no NaN fitness survives `tell`, a one-member species never yields NaN probabilities, every genome ends in exactly one species.
- `./unit/problem` — func-fit losses against the closed form and each `error_method`, dataset tiling order, `NaN` prediction ⇒ `-inf` fitness, and CartPole returns against a host-numpy re-implementation of the same Euler dynamics.
- `./parity` — on *identical* genomes: `forward` values and NaN pattern, `execute_distance`, `XOR`/`XOR3d` fitness and `grad` within tolerance.
- `./integration` — `Pipeline.auto_run` reaches the XOR fitness target inside a small generation budget, the same seed reproduces the same run, a shape-signature change rebuilds (and does not corrupt) the cached executable, and `is_save` writes a loadable checkpoint.
- `release_gate_no_jax.py` — in a clean subprocess, `import tensorneat` leaves `jax`, `torch`, `brax`, `gymnax`, `mujoco` out of `sys.modules`, and a full XOR generation still runs.

## Shared infrastructure (`./support`)
`support/runner.py` is the ONLY place that touches `evoxc.build`/`evoxc.run`/`evoxc.lower`/`evoxc.compile`/`evoxc.load`: `build_defn(fn, *args, backend="numpy")` maps a `@evoxc.defn` (or any plain callable closure) plus its arguments to an executable, `run_defn(fn, *args)` builds and runs in one call, `run_graph(graph, *arrays)` compiles and runs a `Graph` (what `genome.grad` returns), and `to_numpy(tree)` unwraps `evoxc.Tensor` leaves of nested tuples/lists/dicts/namedtuples.
`support/payloads.py` builds `int32`/`float32` gene payloads by hand (`node_keys`, `node_attrs`, `conn_keys`, `conn_attrs` as plain numpy arrays, with `I_INF`/`NaN` padding) plus the canonical small genomes (dense chain, skipped node, isolated node, recurrent ring).
`support/oracle.py` holds the numpy reference implementations the assertions compare against: the schedule derivation, a NaN-aware forward pass, the four func-fit losses, and the CartPole Euler rollout.

## Known divergences parity must tolerate
RNG draws differ by construction (`evoxc.random` splitmix64 vs `jax.random`), so parity never compares populations, mutations or crossovers — only kernels on identical payloads.
`genome.hash()` hashes differently (float32 row hash vs the legacy encoding) and is never compared.
Activation index `-1` means identity now, while the legacy `lax.switch(-1, …)` clamped to the last option: parity genomes avoid `-1` or exclude those nodes from the comparison.
All configured ACT/AGG options are evaluated and selected, so a non-total option can poison a gradient it is not selected into; parity gradient tests use single-option configurations.
The legacy forward aggregated over a full `max_nodes`-wide column with no in-degree truncation, so a parity genome must be constructed with `max_in_degree >= max_nodes` (or at least ≥ every node's real in-degree) to be comparable; otherwise dropped edges show up as `edge_overflow`.
`CustomFuncFit` samples from a seeded `Generator` instead of the global `np.random` state, so parity supplies fixed datasets (`method="grid"` or explicit arrays).
The legacy RL problems are gone, so CartPole has no parity oracle — `./unit/problem`'s numpy reference is the only oracle.

## Conventions
Every test that depends on unimplemented source code has a real signature and docstring and a body of `pytest.skip("not implemented")`, so the file is importable and collectable before the implementation wave and the skip text names the owner.
`jax` is imported only inside `./parity` modules, and only after `pytest.importorskip("jax")`; no other test may import it at module scope.
No network, no GPU, `backend="numpy"` everywhere except tests marked `requires_iree`.
Shapes and keys in the suite are small (pop ≤ 16, `max_nodes` ≤ 16) except where a test exists specifically to exceed a static bound.
Files stay under ~1000 lines; one test module per source module, named `test_<module>.py`.

## Constraints
`release_gate_no_jax.py` does not match pytest's default `python_files` patterns, so `conftest.py::pytest_collect_file` explicitly collects that one filename; do not rename it without editing the hook.
Notebooks (`*.ipynb`) at this level are legacy JAX-era artifacts kept for reference; pytest never collects them, so `testpaths` needs no exclusion.
A test may not edit `../src`, `../benchmarks` or `pyproject.toml`: cross-node needs go into the report to the parent.
Suite output must be deterministic — a flaky test is a bug in the test, not a `pytest.retries` candidate (the plugin is not installed).

## Routing Table
| Path | Purpose |
|---|---|
| `./unit` | numpy-backend kernel/invariant suites, mirroring `../src/tensorneat/`. |
| `./parity` | kernel equality vs the frozen legacy JAX library (marker `parity`). |
| `./integration` | whole-`Pipeline` runs, determinism, rebuild, checkpointing (marker `slow`). |
| `./support` | shared runner, hand-built payloads and numpy oracles for the suites above. |
| `./release_gate_no_jax.py` | import-hygiene gate on `tensorneat` (module file, collected by the conftest hook). |
| `../src/tensorneat` | code under test — read-only, escalate changes to the parent. |
| `../benchmarks/legacy_jax` | frozen legacy implementation used as parity oracle — read-only. |
| `../pyproject.toml` | pytest configuration (`testpaths`, `addopts`, `parity`/`slow` markers) — read-only. |
