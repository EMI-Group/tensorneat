# ./test/backends — the optional compiled backend (`iree`) suite and its request guard

## Intent
Prove that `Pipeline(..., backend="iree")` really executes the whole NEAT generation graph on the compiled backend, and that the result is the *same run* the numpy reference interpreter produces from the same seed.
This suite is the only place in `./test` where a backend other than `"numpy"` runs, so it is the guard against the failure mode "the compiled path silently falls back / silently diverges".
It asserts on real kernels; the only skip is the structural optional-dependency skip of `pytest.importorskip("iree")`, and `./conftest.py` turns the same absence into a hard error when the compiled-backend leg was explicitly requested.

## What it proves
`test_iree_pipeline.py` drives one tiny XOR configuration (`pop_size=8`, `max_nodes=8`, `max_conns=16`, `seed=42`, `fitness_target=-1e-3`) through `setup()` plus two `step()` calls and checks three separate things.
1. The iree run is populated: `(P,)` finite fitness with more than one distinct value, the four-array payload at its static shape (int32 keys, float32 attrs), a generation counter advancing by one per step, an advanced `randkey`, and exactly one `evoxc.build` (the second step reuses the cached executable).
2. The same seed on `backend="numpy"` is the same run: generation counters and the int32 key payloads are bit-identical, while the float32 attribute payloads and the fitness vectors agree to `rtol=1e-5`/`atol=1e-6` with `equal_nan=True`.
   The float tolerance is derived, not guessed: iree's llvm-cpu target reassociates floating-point reductions, so a float32 result moves by ~1 ulp per reordering; measured `max |iree - numpy| <= 6e-8` at this size, and a real divergence is `>> 1e-2`.
   `==` is never used on floats.
   Both fixtures are first pinned to the same `backend`/`seed`/`pop_size`/`num_inputs`/`num_outputs`/`max_nodes`/`max_conns`, both fitness vectors must hold more than one distinct value, and every float payload field must carry a finite non-zero entry on the iree side, so the agreement can never be established by comparing padding against padding.
3. No silent fallback: `compile_options={"not_a_real_compile_option": 1}` raises `evoxc.core.errors.BackendError` on `backend="iree"` (named option, named backend) while the numpy reference interpreter — which documents no option validation — runs the very same dictionary.
   The error surfaces in `setup()`, because `NEAT.setup` builds the genetic-operator graph with the state's build options.
Both backends are driven once per module through module-scoped fixtures, so the ~8 s iree compile happens once and the three cases cost ~9 s together.

## The request guard (`./conftest.py`)
The module-level `importorskip` is correct for a whole-tree run and wrong for a run that explicitly asks for the compiled-backend leg, because a bare "1 skipped" reads as success either way.
`iree_available()` decides availability with `importlib.util.find_spec("iree")`, so the guard never imports the optional extension and never pays its ~80 MB import.
`explicit_marker_expressions(config)` reads only `config.invocation_params.args` (an `-m` arriving through `addopts` is configuration, not a request) and recognises `-m EXPR`, `-m=EXPR` and `-mEXPR`.
`implies_requires_iree(expression)` compiles the selector with pytest's own `_pytest.mark.expression.Expression` and evaluates it against a matcher that accepts only the bare `requires_iree` marker: `requires_iree` and `requires_iree or parity` imply the leg, while `not requires_iree`, `requires_iree and slow` and `parity` do not, and a malformed expression is left to pytest's own `-m` validation.
`names_the_backends_directory(config)` resolves each non-option command-line token and accepts only this directory or a path inside it, so an option value such as `-o addopts=` or `-k iree` is never mistaken for a path and a broad `pytest test` is not a request for this suite.
`pytest_configure` is a historic hook, so it also runs for this sub-directory conftest when pytest reaches `test/backends` during collection, before the module-level skip can be produced.
An explicit marker request with iree absent stops the session through `pytest.exit(..., returncode=1)` and names both install spellings (`uv sync --extra iree`, i.e. `uv pip install "evoxc[iree]"`); no skip is collected.
Naming the directory without a marker request and with iree absent keeps the single structural skip and exit code 0, and announces the install command through `config.issue_config_time_warning`, because a scope request is not a capability request.
With iree importable the guard returns immediately and the module runs its assertions for real.

## The two skip meanings
A default or whole-tree run (or one that merely names this directory) skips because the optional extra is absent and was not requested, which is expected and green.
A run whose `-m` selector implies `requires_iree` treats the same absence as "the requested leg is unavailable", which is never a skip: it exits non-zero with the install command.

## How to run
`uv sync --extra iree` installs `iree-base-compiler` + `iree-base-runtime` (equivalent: `uv pip install "evoxc[iree]"`).
`uv run --extra bench --extra iree pytest test/backends -q -o addopts=""` — this suite alone with the extra installed: **3 passed in ~9–10 s**.
`uv run --extra bench --extra iree pytest test/backends -q -o addopts="" -m requires_iree` — the same three cases selected by marker (registered in `../conftest.py`, in no gate): **3 passed**.
`-m "not requires_iree"` deselects all three (`3 deselected`) in a run that installed the extra but does not want the compile cost.
Without the extra (bare `uv sync`), `uv run pytest test/backends -q -o addopts=""` reports **1 skipped, 1 warning in ~0.05 s, exit 0**, the warning naming the install command.
Without the extra, `uv run pytest -m requires_iree -q -o addopts="" test/backends` reports **exit 1** and the message `test/backends: this run explicitly requests the compiled-backend leg (-m 'requires_iree') but the optional iree backend is not installed.` followed by the install command.
Without the extra, `uv run pytest` (whole tree) stays green and this directory contributes one structural skip; that bare environment's total is 5 skips, the other four being the `jax` parity modules.
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
The guard has no committed test of its own: its absent-extra behaviour is evidenced by the two-environment commands above, and its helpers are exercised by a direct throwaway call with a fake config and a monkeypatched `find_spec` (not committed).

## Constraints
`./conftest.py` imports nothing optional and decides availability with `find_spec`; it reads `_pytest.mark.expression` (pytest's own `-m` parser — private, but stable and the only way to reuse that grammar exactly) and `pytest` itself.
`importorskip("iree")` is the first executable statement after `import pytest`; every other import of the module is below it, so a missing extra can never surface as a collection error.
No test here imports `jax`; the `iree` import is the only optional-dependency coupling, and it does not pull JAX in.
The configuration stays tiny on purpose: the compile dominates the runtime, and a bigger population buys nothing that the parity comparison does not already prove.
`test/backends/__init__.py` namespaces the directory so pytest's prepend import mode cannot collide with another suite's `test_*.py` basename.
This directory writes nothing outside `test/backends`; `../src`, `../benchmarks` and `pyproject.toml` are read-only inputs.

## Routing Table
| Path | Purpose |
|---|---|
| `./conftest.py` | the request guard: hard error for an explicit `-m requires_iree` request with iree absent, announced structural skip when the directory is merely named. |
| `./test_iree_pipeline.py` | iree execution, iree-vs-numpy same-seed equality, and the no-silent-fallback option check (marker `requires_iree`). |
| `../conftest.py` | registers the `requires_iree` marker and puts `src/` on `sys.path`. |
| `../../benchmarks/backend_measure.py` | build time vs per-generation time vs peak RSS for one `(case, backend)`. |
