# ./test/backends — the optional compiled backend (`iree`) smoke + parity suite

## Intent
Prove that `Pipeline(..., backend="iree")` really executes the whole NEAT generation graph on the compiled backend, and that the result is the *same run* the numpy reference interpreter produces from the same seed.
This suite is the only place in `./test` where a backend other than `"numpy"` runs, so it is the guard against the failure mode "the compiled path silently falls back / silently diverges".
It asserts on real kernels — nothing here skips unless the optional extra is genuinely absent.

## What it proves
`test_iree_pipeline.py` drives one tiny XOR configuration (`pop_size=8`, `max_nodes=8`, `max_conns=16`, `seed=42`, `fitness_target=-1e-3`) through `setup()` plus two `step()` calls and checks three separate things.
1. The iree run is populated: `(P,)` finite fitness with more than one distinct value, the four-array payload at its static shape (int32 keys, float32 attrs), a generation counter advancing by one per step, an advanced `randkey`, and exactly one `evoxc.build` (the second step reuses the cached executable).
2. The same seed on `backend="numpy"` is the same run: generation counters and the int32 key payloads are bit-identical, while the float32 attribute payloads and the fitness vectors agree to `rtol=1e-5`/`atol=1e-6` with `equal_nan=True`.
   The float tolerance is derived, not guessed: iree's llvm-cpu target reassociates floating-point reductions, so a float32 result moves by ~1 ulp per reordering; measured `max |iree - numpy| <= 6e-8` at this size, and a real divergence is `>> 1e-2`.
   `==` is never used on floats.
3. No silent fallback: `compile_options={"not_a_real_compile_option": 1}` raises `evoxc.core.errors.BackendError` on `backend="iree"` (named option, named backend) while the numpy reference interpreter — which documents no option validation — runs the very same dictionary.
   The error surfaces in `setup()`, because `NEAT.setup` builds the genetic-operator graph with the state's build options.
Both backends are driven once per module through module-scoped fixtures, so the ~8 s iree compile happens once and the three cases cost ~9 s together.

## How to run
`uv pip install "evoxc[iree]"` installs `iree-base-compiler` + `iree-base-runtime`; without the extra the module skips at import via `pytest.importorskip("iree")` with an install hint.
- `pytest -q test/backends` — this suite alone; with the extra installed it is **3 passed in ~9 s**.
- `pytest -m requires_iree` — the same tests selected by marker (registered in `../conftest.py`, in no gate).
- `pytest -m "not requires_iree"` — deselect them from a run that installed the extra but does not want the compile cost.
- `pytest -q test/backends` in an environment without the extra must report one skip, not an error; that is the release-shape check for this directory.
Markers: `requires_iree` is declared in `../conftest.py` (`pyproject.toml` is read-only for this node).

## Compile time is separate from per-generation time
`Pipeline.setup()` is host-only plus NEAT's small operator graph; the generation graph is compiled on the **first** `step()` (~8–9 s at pop 8–100 on this box) and then reused for every later generation (milliseconds at pop 8, ~200 ms at pop 100).
So a test that drives several generations does not pay the compile per generation, and any timing claim must state which of the two it measures.
The standalone CLI for that is `../benchmarks/backend_measure.py` (`--backend numpy|iree`, `--case xor|xor3d`, `--pop N`, 3 warmup + 20 timed generations, `--opt-level` forwarded as `compile_options={"opt_level": ...}`, `--json out.json`), and its peak-RSS column is a process-lifetime high-water mark, so each `(case, backend)` must be measured in a fresh process.

## Coverage gaps (what this suite does not prove)
`"xla"` and `"tvm"` are claimed backends with **no test anywhere** in `./test` (no test file mentions either name), so their support rests on `../benchmarks/BACKENDS.md` hand-measurements only.
The no-fallback case is pinned only for a bad *compile option* on `iree`; nothing tests an unknown backend *name* (`backend="nope"`), a kernel `iree` cannot lower, or the claim that a failing backend never retries on numpy.
The numpy-vs-iree comparison takes `numpy` (another `evoxc` backend) as its oracle, so a defect shared by evoxc's frontend/lowering is invisible here; the independent JAX oracle in `../parity` covers the numpy leg only.
Only XOR (`func_fit`) is exercised: no `scan`/`while_loop` rollout (`CartPole`/`Pendulum`/`MountainCarContinuous`), no `RecurrentGenome` tick loop, no `HyperNEAT` — the control-flow shapes most likely to differ per backend.
Generation 1's payload is the host-seeded `setup()` population, so it is identical across backends by construction; only generation 2's payload exercises iree-built mutation/crossover.
`../.github/workflows/ci.yaml` runs this leg with `continue-on-error: true`, so a red iree job does not fail CI.

## Constraints
`importorskip("iree")` is the first executable statement after `import pytest`; every other import of the file is below it, so a missing extra can never surface as a collection error.
No test here imports `jax`; the `iree` import is the only optional-dependency coupling, and it does not pull JAX in.
The configuration stays tiny on purpose: the compile dominates the runtime, and a bigger population buys nothing that the parity comparison does not already prove.
`test/backends/__init__.py` namespaces the directory so pytest's prepend import mode cannot collide with another suite's `test_*.py` basename.
This directory writes nothing outside `test/backends`; `../src`, `../benchmarks` and `pyproject.toml` are read-only inputs.

## Routing Table
| Path | Purpose |
|---|---|
| `./test_iree_pipeline.py` | iree execution, iree-vs-numpy same-seed equality, and the no-silent-fallback option check (marker `requires_iree`). |
| `../conftest.py` | registers the `requires_iree` marker and puts `src/` on `sys.path`. |
| `../../benchmarks/backend_measure.py` | build time vs per-generation time vs peak RSS for one `(case, backend)`. |
