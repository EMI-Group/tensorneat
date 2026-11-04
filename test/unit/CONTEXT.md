# ./unit — numpy-backend verifier suite for the evoxc kernels

## Intent
Prove that every evoxc kernel in `../../src/tensorneat` computes what its docstring promises, on the reference numpy interpreter, with no optional framework imported by the tests themselves.
Each assertion compares against something independent of the implementation under test: a closed-form value, a plain-numpy oracle, a hand-built payload, or a structural invariant (`sum(spawn) == pop_size`, "no NaN survives `tell`").
This directory is the whole content of the default gate; `./parity` (legacy JAX oracle) and `./integration` (whole `Pipeline`) are the two suites that go beyond it.

## Counts
`./common` 155 passed, `./genome` 100, `./algorithm` 144, `./problem` 62, `./benchmarks` 16 — 477 passed and 0 xfailed in the subdirectories, plus `./test_version.py` (2) at this node's root.
`./genome` dominates the wall clock at ~52 s (one traced executable per graph), `./algorithm` takes ~28 s, `./common` ~12 s and `./problem` ~4 s.
Everything in this node is green: there is no xfail and no skip (the `./benchmarks` fingerprint tests skip only on a box without a `git` binary).

## How to run
`python -m pytest test/unit -m "not parity and not slow"` (or plain `pytest` from the repo root) — numpy backend, no optional dependency, ~95 s.
`pytest test/unit/<sub> -k <expr>` narrows the same gate while keeping the leak guard active.
`test/unit/algorithm` and `test/unit/genome` are also comfortable alone (`~28 s` / `~52 s`); no module here needs a network, a GPU or parallelism.

## Layout
`./common` — schedule (`tensorization`) derivation and validation, flat gather/scatter/mutation helpers, ACT/AGG dispatch, host graph walks, `State`.
`./genome` — gene payloads, forward/recurrent/grad kernels, mutation/crossover/distance invariants, host-only introspection.
`./algorithm` — NEAT `tell` algebra: speciation, stagnation, spawn allocation, parent selection.
`./problem` — func-fit losses (`mse`/`rmse`/`mae`/`mape`) and the CartPole rollout.
`./benchmarks` — the benchmark harness fingerprint (`git_report`, porcelain parsing) on a real temporary git repository.
`./test_version.py` — the package's `__version__` metadata and its `PackageNotFoundError` fallback.
Every directory here carries `__init__.py` (pytest's prepend import mode names a module by basename, so un-namespaced directories collide with `./parity`/`./integration`).

## Oracles
`../support/oracle.py` holds the numpy reference implementations the suites compare against: the schedule derivation, a NaN-aware forward pass, the four func-fit losses, the CartPole Euler rollout.
`../support/functions_ref.py` holds the 12 ACT and 6 AGG reference kernels the dispatch tests loop over.
`../support/payloads.py` builds every `int32`/`float32` gene payload by hand (`node_keys`/`node_attrs`/`conn_keys`/`conn_attrs`, `I_INF`/`NaN` padding) plus the canonical small genomes, so no test draws a topology it could have written.
A test that needs neither of those compares against a closed form written in the module itself (`np.where(fitness, -inf)`, the sigmoid chain, the species-count algebra).
Kernels are built only through `../support/runner.py` (`run_defn`/`run_graph`/`to_numpy`), and `./genome/conftest.py` caches one `evoxc.Executable` per graph and replays it with `evoxc.run` because tracing costs about ten times a run.

## The optional-framework guard (`conftest.py`)
`pytest_sessionstart` freezes the session baseline — the forbidden modules already in `sys.modules` — and a session-scoped autouse fixture does it when the conftest is loaded only at collection time; the freeze is idempotent, so it happens exactly once.
A session-wide recorder wraps `builtins.__import__` and notes every import *statement* whose root module is in `("jax", "jaxlib", "torch", "brax", "gymnax", "mujoco")` together with the file that executed it.
The autouse per-test fixture fails a test whose body or module let a forbidden module in, reporting the offending module and the importing file.
The recorder is what gives the guard teeth: it attributes a forbidden import to the file that executed it, including an import performed by a module under test during a test body.
It also supplies the allowances the guard needs: an import executed from `src/`, from `pytest.importorskip`, or from another suite (parity imports `jax` while it is collected, so `pytest test/unit test/parity` must not fail here) is not this suite's leak; one executed from a file below this directory is.
Each leak is reported once and forgotten, so one bad import does not turn every later test red.
Not covered: a dynamic `importlib.import_module` of an already-loaded framework, and anything a subprocess does — that is `../release_gate_no_jax.py`'s job, which checks the package itself in a clean interpreter.

## Known items
`./genome/test_mutation.py` is at the ~1000-line concern threshold; split it by operation family (add-node / add-conn / mutate-weight / toggle / delete) if it grows further.
No test in this node is xfailed or skipped.

## Conventions
One module per source module, named `test_<module>.py`; the directory tree mirrors `../../src/tensorneat/`.
Shapes stay small (pop ≤ 16, `max_nodes` ≤ 16) except where a test exists specifically to trip a static bound.
Backend is always `"numpy"`; a test that needs `evoxc[iree]` is marked `requires_iree` and belongs to no gate.
A failure names the independent oracle it disagrees with, so a red test says which side of the comparison is wrong.

## Constraints
No network, no GPU, no wall-clock or ordering dependence; `tmp_path` appears only in `./common/test_state.py`, for the `State` save/load round-trip.
A test may not edit `../../src`, `../../benchmarks` or `pyproject.toml`; a discovered defect is pinned by an xfail here and reported upward.
`../conftest.py` puts `src/`, `benchmarks/legacy_jax` and `test/` on `sys.path`; `conftest.py` here has one process-wide side effect, the import recorder, and it uses nothing outside the standard library and pytest.

## Routing Table
| Path | Purpose |
|---|---|
| `./common` | tensorization schedule invariants, flat gather/scatter/mutation helpers, ACT/AGG dispatch, host graph algorithms, `State`. |
| `./genome` | payload layout, forward/recurrent/grad kernels, mutation/crossover/distance invariants, host introspection. |
| `./algorithm` | NEAT `tell` algebra: speciation, stagnation, spawn allocation, parent selection. |
| `./problem` | func-fit losses (`mse`/`rmse`/`mae`/`mape`) and the CartPole rollout. |
| `./benchmarks` | harness fingerprint: `git_report`/porcelain parsing against a real temporary git repository. |
| `./test_version.py` | `tensorneat.__version__` against the distribution metadata, plus the `PackageNotFoundError` fallback. |
| `./conftest.py` | session baseline + import recorder: the per-test optional-framework guard. |
| `../support` | shared runner, hand-built payloads and numpy oracles used by every suite here. |
| `../release_gate_no_jax.py` | import-hygiene gate on the package in a clean subprocess (test root, not this node). |
