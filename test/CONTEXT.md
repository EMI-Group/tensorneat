# ./test — the pytest suite of the evoxc rewrite

## Intent
Prove three separate things about the rewrite: the evoxc kernels are *correct* (unit, against numpy oracles), they are *equivalent* to the frozen legacy library where the legacy semantics are well defined (parity, against `benchmarks/legacy_jax`), and the installed package is *JAX-free and runnable end to end* (release gate + integration).
This is a verifier, not a specification: every test body executes a real kernel and asserts on its output.
Every suite reads `../src/tensorneat` and `../benchmarks/legacy_jax` as read-only inputs; nothing in `src/` imports anything from here.
`./support` holds the host-side helpers the suites share (one graph runner, hand-built gene payloads, numpy reference oracles); it is test code and is not packaged.

## How to run
Commands an owner runs from the repo root (`[tool.pytest.ini_options]` in `pyproject.toml` sets `testpaths = ["test"]`, `addopts = "-q"`; a doubled `-q` suppresses the summary line, so read the counts with `-o addopts=""`):
- `pytest -m "not parity and not slow"` — the default gate: `./unit` plus the release gate, numpy backend only, no optional dependency. Measured on this machine: **466 passed, 0 failed, 0 xfailed in 80 s**, with 39 tests deselected and the four JAX parity modules reported as skips at import.
- `pytest -m parity` — kernel equality against the frozen legacy JAX implementation; needs `pip install -e ".[bench]"` and finishes in **19 s** (53 passed). Without the extra, the four JAX oracle modules skip at import and `test_convert.py` alone runs **18 passed**. Every parity module calls `pytest.importorskip("jax")` so a bare environment skips rather than errors.
- `pytest -m slow` — `./integration` (XOR evolution, determinism, rebuild-on-shape-change, checkpoint round-trip) plus the release gate's end-to-end XOR run: **21 passed, 4 skipped in 86 s**.
- `pytest test/release_gate_no_jax.py` — the import-hygiene gate alone; **4 passed**.
- `pytest` (no marker) — the whole tree: **505 passed, 4 skipped in 163 s** in a bare environment, the skips being the four JAX parity modules.
Markers `parity` and `slow` are declared in `pyproject.toml`; `requires_iree` is declared in `conftest.py` and is in no gate (it needs `evoxc[iree]`).

## Layout
```
test/
  conftest.py            sys.path guards (src/, benchmarks/legacy_jax, test/), marker registration, seed/rng/evoxc_key fixtures
  release_gate_no_jax.py the import-hygiene gate (collected by a conftest hook, see Constraints)
  support/               shared host-side test infrastructure (runner, payloads, numpy oracles, lazy imports)
  unit/                  numpy-backend kernel and invariant tests, mirrors src/tensorneat/
    conftest.py          session-wide optional-framework leak guard
    common/  genome/  algorithm/  problem/  benchmarks/
  parity/                same kernels vs the frozen legacy JAX library, marker `parity`
  integration/           whole-Pipeline runs, marker `slow`
  backends/              compiled-backend runs (marker `requires_iree`), skips without `evoxc[iree]`
```
Every directory that holds test modules carries an `__init__.py`: pytest's default prepend import mode names a module by its basename, so un-namespaced directories collide (`test/unit/genome/test_forward.py` vs `test/parity/test_forward.py`) and abort the session with an import-mismatch error.
Current counts: `unit/common` 155, `unit/genome` 100, `unit/algorithm` 144, `unit/problem` 62, `unit/benchmarks` 16, `unit/test_version.py` 2, `parity` 53 (18 without jax), `integration` 20, release gate 4.

## What each suite proves
- `./unit/common` — the schedule derived from a hand-built genome has the right level order, incoming-edge windows and sentinels, `validate_schedule` accepts what `derive_schedule` produces and rejects every tampered invariant, flat gather/scatter/mutation helpers agree with a plain-numpy oracle, the ACT/AGG dispatchers agree with a per-option Python loop for all 12 ACT / 6 AGG ids (including `-1` and out-of-range ⇒ identity), `State` is an immutable pytree container, and the host graph walks (`decode_graph`, `topological_sort`, `find_useful_nodes`, `node_depth`, `check_cycles`) agree with the in-graph `node_level`.
- `./unit/genome` — forward values on hand-built genomes (dense chain, skipped node, unconnected node ⇒ NaN propagation, activation `-1` ⇒ identity), the recurrent tick, `grad` against the numpy / finite-difference reference, `execute_distance`, and the genetic operators through their *invariants* (capacity, key uniqueness, in-degree bound, cycle and duplicate rejection, structural stability of crossover), plus the host-only helpers (`network_dict`, `sympy_func`, `visualize`, `repr`) and the in-graph `hash`.
- `./unit/algorithm` — speciation, stagnation, spawn allocation, parent selection and `NEAT.tell` bookkeeping: `sum(spawn) == pop_size`, the last-qualifying-block layout rule, damping law and `min_species_size` floor, elite pinning, father-wins-tie, no NaN fitness surviving `tell`, a one-member species never yielding NaN probabilities, every genome ending in exactly one species.
- `./unit/problem` — func-fit losses against the closed form for each `error_method`, individual-major dataset tiling order, `NaN` prediction ⇒ `-inf` fitness, `CustomFuncFit` sampling from a seeded `Generator` (`method="grid"` pinned), `CartPole` construction validation and returns against a host-numpy re-implementation of the same Euler dynamics, and NaN logits ⇒ `-inf` fitness.
- `./unit/benchmarks` — the benchmark harness fingerprint: `git_report()` on a real temporary git repository created under `tmp_path` (clean tree ⇒ clean, unstaged modification ⇒ dirty, untracked file ⇒ dirty, staged rename ⇒ dirty naming the new path), the porcelain `XY PATH` parser on the documented records, and the `-z` NUL-record reader that consumes a rename's source field; this is the regression suite for the `git_dirty` false-negative (a stripped status line read the path one character short).
- `./parity` — on *identical* genomes: `forward` values and NaN pattern (rtol 1e-5) on chain / skip / isolated / random-DAG genomes, `execute_distance` closed forms, `XOR`/`XOR3d` fitness for every `error_method`, and gradients w.r.t. `conn_attrs`/`node_attrs` (rtol 1e-4), plus the lossless-schedule guard in `convert.py`.
- `./integration` — `Pipeline.auto_run` reaches the XOR fitness target inside a small generation budget, the same seed reproduces the same run, a shape-signature change rebuilds (and does not corrupt) the cached executable, `is_save` writes a loadable checkpoint, and the documented reporting tail `show(state, best)` prints the XOR report for the batched best and for its bare rows.
- `./backends` — the `iree` compiled backend runs a real XOR generation end to end (`setup` + 2 `step`s, finite `(P,)` fitness, one cached build), agrees with numpy on the same seed (int/key payloads exactly, floats within a documented reduction-reassociation tolerance), and rejects an unknown compile option instead of falling back.
- `release_gate_no_jax.py` — in a clean subprocess, `import tensorneat` leaves `jax`, `torch`, `brax`, `gymnax`, `mujoco` out of `sys.modules`, and a full XOR generation still runs.

## Shared infrastructure (`./support`)
`support/runner.py` is the ONLY place that touches `evoxc.build`/`evoxc.run`/`evoxc.lower`/`evoxc.compile`/`evoxc.load`: `build_defn(fn, *args, backend="numpy")` maps a `@evoxc.defn` (or any plain callable closure) plus its arguments to an executable, `run_defn(fn, *args)` builds and runs in one call, `run_graph(graph, *arrays)` compiles and runs a `Graph` (what `genome.grad` returns), and `to_numpy(tree)` unwraps `evoxc.Tensor` leaves of nested tuples/lists/dicts/namedtuples (a plain `tuple` is rebuilt rather than called).
`support/payloads.py` builds `int32`/`float32` gene payloads by hand (`node_keys`, `node_attrs`, `conn_keys`, `conn_attrs` as plain numpy arrays, with `I_INF`/`NaN` padding) plus the canonical small genomes (dense chain, skipped node, isolated node, recurrent ring).
`support/oracle.py` holds the numpy reference implementations the assertions compare against: the schedule derivation (field-for-field against `Transformed`), a NaN-aware forward pass, the four func-fit losses, and the CartPole Euler rollout; `support/functions_ref.py` keeps the 12 ACT / 6 AGG reference kernels.
`support/lazy.py::require(module)` is the deferred-import helper the fixtures use: it imports a module at call time and turns a `NotImplementedError` raised during that import into a named skip, while every other error surfaces.

## Known items
`evoxc`'s power pullback computes `d(x²)/dx` as `y*b/a`, yielding `NaN` at `x == 0` where jax yields `0`; `./parity` differentiates at a differentiable point and records the corner rather than depending on it.

## Known divergences parity must tolerate
RNG draws differ by construction (`evoxc.random` splitmix64 vs `jax.random`), so parity never compares populations, mutations or crossovers — only kernels on identical payloads.
`genome.hash()` hashes differently (float32 row hash vs the legacy encoding) and is never compared.
Activation index `-1` means identity now, while the legacy `lax.switch(-1, …)` clamped to the last option: parity genomes avoid `-1` or exclude those nodes from the comparison.
All configured ACT/AGG options are evaluated and selected, so a non-total option can poison a gradient it is not selected into; parity gradient tests use single-option configurations.
The legacy forward aggregated over a full `max_nodes`-wide column with no in-degree truncation, so a parity genome must be constructed with `max_in_degree >= max_nodes` (or at least ≥ every node's real in-degree) to be comparable; otherwise dropped edges show up as `edge_overflow`.
`CustomFuncFit` samples from a seeded `Generator` instead of the global `np.random` state, so parity supplies fixed datasets (`method="grid"` or explicit arrays).
The legacy RL problems are gone, so CartPole has no parity oracle — `./unit/problem`'s numpy reference is the only oracle.
`problem.finalize_fitness` maps NaN → `-inf` while the legacy pipeline left NaN in the fitness array, so parity compares finite fitness only and the mapping is pinned as the one deliberate divergence.
`mape` with a zero target is `1` for a zero prediction but `NaN` (`0/0`) when prediction and target are both exactly zero on the legacy side; parity genomes are chosen to avoid that corner.
A NaN *source slot* is dropped when the destination also has a live source, so an isolated node poisons only a read-out that has no other live input; both halves of that rule are pinned in `./parity` and `./unit/genome`.

## Conventions
No test body is a placeholder: every test either asserts on a real kernel or is skipped because an optional dependency is absent (`./parity` without `jax`).
`jax` is imported only inside `./parity` modules, and only after `pytest.importorskip("jax")`; no other test may import it at module scope, and `test/unit/conftest.py` enforces that with a session-baseline `sys.modules` diff plus an import recorder that attributes the offending file.
A behaviour the implementation documents but does not deliver is pinned with `xfail(strict=True)` and a reason starting `src defect:` — never removed, never softened.
Shapes and keys in the suite are small (pop ≤ 16, `max_nodes` ≤ 16) except where a test exists specifically to exceed a static bound.
No network, no GPU, `backend="numpy"` everywhere except tests marked `requires_iree`.
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
| `./backends` | compiled-backend (`iree`) end-to-end XOR run plus numpy/iree agreement; marker `requires_iree`, in no gate (`./backends/CONTEXT.md`). |
| `./support` | shared runner, hand-built payloads and numpy oracles for the suites above. |
| `./release_gate_no_jax.py` | import-hygiene gate on `tensorneat` (module file, collected by the conftest hook). |
| `../src/tensorneat` | code under test — read-only, escalate changes to the parent. |
| `../benchmarks/legacy_jax` | frozen legacy implementation used as parity oracle — read-only. |
| `../pyproject.toml` | pytest configuration (`testpaths`, `addopts`, `parity`/`slow` markers) — read-only. |
