# ./test/backends — the compiled-backend suites (IREE llvm-cpu and ROCm/HIP) and their request guard

## Intent
Prove that both compiled legs really execute a whole NEAT generation graph and produce *the same run* the numpy reference interpreter produces from the same seed: llvm-cpu via `test_iree_pipeline.py`, and an AMD GPU via `test_rocm_pipeline.py`.
This is the only place in `./test` where a backend other than `"numpy"` runs, so it is the guard against the two failure modes "the compiled path silently falls back" and "the compiled path silently diverges".
Both legs drive one configuration and one comparison, defined once in `_tiny_xor.py`, so they cannot drift into two different experiments.
Every test asserts on a real kernel; the only skips are structural (optional dependency absent, device absent), and `./conftest.py` turns the same absence into a hard error when a compiled leg was explicitly requested.

## Shared harness (`_tiny_xor.py`, not collected)
`POP_SIZE=8`, `GENERATIONS=2`, `MAX_NODES=8`, `MAX_CONNS=16`, `SEED=42`, `UNKNOWN_OPTION`, `PAYLOAD_FIELDS` — the geometry both legs share, and the reason the two legs' numbers are comparable at all.
`tiny_xor_pipeline(backend, *, device=None, compile_options=None)` builds the one `Pipeline` (sigmoid read-out so `-1e-3` is reachable, `species_size=4` inside `pop_size=8`, `is_detailed=False`).
`drive(pipeline)` runs `setup()` plus `GENERATIONS` `step()` calls and keeps one `(payload, fitnesses, generation)` tuple per generation.
`run_configuration(run)` returns the seven values two runs must agree on before any number is compared: `(backend, seed, pop_size, num_inputs, num_outputs, max_nodes, max_conns)`.
`assert_populated(run, *, label)` is the generation-shaped assertion set: advancing counter, four-entry payload at static shape (int32 keys, float32 attrs), `(P,)` finite fitness with more than one distinct value, advanced `randkey`, exactly one cached `evoxc.build`, positive `build_seconds`.
`assert_same_run(reference, other, *, reference_label, other_label, rtol, atol)` compares **every** traced generation (generation 1 is the host-seeded population, every later one also exercises the compiled mutation/crossover path): equal counters, int32 key payloads bit-identical, float fields and fitness within the caller's tolerance with `equal_nan=True`, plus the non-vacuous guard that each compared side carries a finite non-zero entry, and more-than-one-distinct fitness on both sides.
Tolerances stay in the test modules, because each of them documents its own measurement; `assert_same_run` takes them as arguments.
The leading underscore keeps the module out of collection, and it imports no optional dependency, so both test modules import it only *after* their `importorskip` guard.

## The llvm-cpu leg (`test_iree_pipeline.py`)
`Pipeline(..., backend="iree")` executes `setup()` plus two generations and returns a populated payload (the shared `assert_populated` set).
The same seed on `backend="numpy"` is the same run: counters and int32 key payloads bit-identical, float payloads and fitness within `FLOAT_RTOL=1e-5` / `FLOAT_ATOL=1e-6`.
No silent fallback: `compile_options={"not_a_real_compile_option": 1}` raises `evoxc.core.BackendError` on `backend="iree"` naming the option and the backend (from `setup()`, because `NEAT.setup` builds the genetic-operator graph with the state's build options), while the numpy reference interpreter — which documents no option validation — runs the very same dictionary.
Measured on this box (IREE 3.13.0): `setup()` ~0.89 s, `builds == 1`, `build_seconds` ~5.9 s, first `step()` 5.92 s (it pays the compile), second `step()` 0.002 s; worst |iree − numpy| = **2.384e-07** (2 ulp of float32, eps 1.1920929e-07) over seeds 42/43/44/7/123/2024 × both generations × every float field and the fitness vector, so `atol=1e-6` is 4.2× that worst difference and `rtol=1e-5` ~42×.
The tolerance is derived, not guessed: each reduction reassociation moves a float32 result by ~1 ulp, a real divergence (wrong operand, dropped edge, changed activation) moves values by ≫ 1e-2, and `==` is never used on floats.

## The ROCm/HIP leg (`test_rocm_pipeline.py`)
`Pipeline(..., backend="iree", device="rocm", compile_options={"target_backends": ["rocm"]})` runs `setup()` plus two generations on the AMD GPU, with the shared populated set plus a **placement** assertion: every `evoxc.Tensor` leaf of the runtime state reports `device.kind == "rocm"` (`randkey`, `pop`, `generation`, `species`; the host configuration entries are not tensors) and `state.device == "rocm"`, so a silent CPU fallback — which would leave the leaves on `cpu` while the fitness vector still looked fine — cannot pass.
The same seed on `backend="numpy"` is the same run under the shared `assert_same_run`.
Measured on this box (AMD Instinct MI300X VF, IREE 3.13.0): `setup()` 1.15–1.28 s, `builds == 1`, `build_seconds` 6.3–7.0 s, first `step()` 6.3–6.4 s, second `step()` 0.004 s; worst |rocm − numpy| = **2.3841858e-07** (2 ulp) over the same six-seed sweep, int32 keys bit-identical and NaN patterns identical, so it uses the same `rtol=1e-5` / `atol=1e-6` as the CPU leg and documents its own 4.2× headroom.
The device is not a guess: `iree.runtime.get_driver("hip").query_available_devices()` reports one device in ~0.04 s, and `import iree.runtime` costs ~0.14 s / a few MiB on IREE 3.13 (the runtime is lazy — the claimed "~80 MB import" of the 3.11 era does not apply).

## The request guard (`./conftest.py`)
`iree_available()` decides availability with `importlib.util.find_spec`, so the guard never imports the optional extension; `explicit_marker_expressions(config)` reads only `config.invocation_params.args` (an `-m` arriving through `addopts` is configuration, not a request) and recognises `-m EXPR`, `-m=EXPR` and `-mEXPR`.
`implies_marker(expression, marker)` compiles the selector with pytest's own `_pytest.mark.expression.Expression` and evaluates it **twice** against the same synthetic test that carries only that marker: once with a matcher that accepts it, once with a matcher that accepts nothing.
The expression implies the leg only when the first selects the test and the second does not, so a marker-free selector cannot masquerade as a request: `requires_iree`, `requires_rocm` and their disjunctions imply, while `not requires_iree`, `requires_iree and slow`, `parity` and the documented default gate `not parity and not slow` do not; a malformed expression is left to pytest's own `-m` validation.
`implies_requires_iree` is the marker-defaulting wrapper over `implies_marker`; `REQUESTED_MARKERS = (requires_iree, requires_rocm)` are the two capability requests, because the GPU leg needs the same IREE wheels *plus* a HIP device.
`names_the_backends_directory(config)` resolves each non-option command-line token and accepts only this directory or a path inside it, so an option value such as `-o addopts=` or `-k iree` is never mistaken for a path and a broad `pytest test` is not a request for this suite.
`pytest_configure` is a historic hook, so it runs for this sub-directory conftest before collection — before a module-level skip can be produced — and `pytest.exit(..., returncode=1)` stops an explicit marker request with iree absent, announcing the install step and never a skip.
`INSTALL_COMMAND` is the spelling `pyproject.toml`/`uv.lock` resolve: the `iree` extra names the IREE 3.13 wheels directly (`iree-base-compiler>=3.13,<3.14` + `iree-base-runtime>=3.13,<3.14`, same release for compiler and runtime) and no longer pulls `evoxc[iree]`, whose pin is 3.11 and whose HIP runtime aborts on loop-carried `while` graphs; the pip half of the message is the equivalent pair of pinned wheels.

## Skip conditions (exhaustive)
`iree` not installed: each module contributes one structural skip from `pytest.importorskip("iree")`, whose reason names the install step and says an explicit marker run fails instead of skipping.
`iree` installed and `/dev/kfd` absent: the ROCm module skips with a reason naming `/dev/kfd`, the AMD compute kernel driver's character device.
`iree` installed and `get_driver("hip")` **or** `query_available_devices()` raises: the ROCm module skips with the exception type and text, so the missing piece is diagnosable from the skip line alone; the `try` covers the probe only, so an error from a test body always propagates.
`iree` installed and the HIP driver reports zero devices: the ROCm module skips saying there is no AMD GPU to run on.
Residual: with `iree` installed but no AMD device, an explicit `-m requires_rocm` run skips (exit 0) rather than failing — the guard cannot probe `/dev/kfd` or the driver without importing the extension in `conftest.py`, which this directory forbids; only the iree-absent case is loud for `requires_rocm`.

## How to run
Commands from the repo root with every extra installed (`uv sync --all-extras`; IREE 3.13.0, `evoxc` git v0.1.1).
- `uv run pytest test/backends -q -o addopts=""` — this directory alone: **46 passed in ~17.5 s** (41 guard cases + 3 llvm-cpu + 2 ROCm; the ROCm compile happens once and ~10 s of the total is that leg).
- `uv run pytest test/backends -q -o addopts="" -m requires_iree` — **5 passed, 41 deselected in ~18.8 s**, because the ROCm module also carries `requires_iree` (a GPU leg needs iree too).
- `uv run pytest test/backends -q -o addopts="" -m requires_rocm` — **2 passed, 44 deselected in ~9.5 s**.
- `uv run pytest test/backends -q -o addopts="" -m "not requires_iree"` — **41 passed, 5 deselected in ~0.45 s**: the guard's own cases only, no compile.
- `uv run pytest test/backends/test_conftest.py -q -o addopts=""` — **41 passed in 0.03 s**, no optional dependency.
- Without the extra (bare `uv sync`): `uv run pytest test/backends -q -o addopts=""` reports **41 passed, 2 skipped, 1 warning, exit 0** — the guard's own cases run, each leg contributes its one structural skip, and the warning names the install step. With iree absent an explicit `-m requires_iree` **or** `-m requires_rocm` **exits 1** with the message `this run explicitly requests the compiled-backend leg (-m '…', which selects the '…' marker) but the optional iree backend is not installed.` followed by the install step (naming the directory keeps exit 1; a whole-tree marker run exits 2 because the same `pytest.exit` fires during collection — non-zero either way).
- The documented default gate `-m "not parity and not slow"` selects both compiled legs; on this box it is **1312 passed, 104 deselected in ~59 s**, and the whole tree with every extra is **1416 passed, 0 skipped in ~99 s**.
- Absence can be simulated without uninstalling anything: a throwaway pytest plugin (`-p block_iree`) that sets `sys.modules["iree"] = None` makes `find_spec("iree")` return `None` and `import iree` fail, which reproduces the bare-environment counts above.
Markers: `requires_iree` and `requires_rocm` are declared in `../conftest.py`; both are in no gate of their own, but the default gate selects them.

## Compile time is separate from per-generation time
`Pipeline.setup()` is host-only plus NEAT's small operator graph; the generation graph is compiled on the **first** `step()` (~5.9 s on llvm-cpu, ~6.3–7.0 s on ROCm at this size) and then reused for every later generation (2–4 ms at pop 8, ~200 ms at pop 100).
So a test that drives several generations does not pay the compile per generation, and any timing claim must state which of the two it measures.
The standalone CLI for that is `../benchmarks/backend_measure.py` (`--backend numpy|iree`, `--case xor|xor3d`, `--pop N`, 3 warmup + 20 timed generations, `--opt-level` forwarded as `compile_options={"opt_level": ...}`, `--json out.json`), and its peak-RSS column is a process-lifetime high-water mark, so each `(case, backend)` must be measured in a fresh process.

## Coverage gaps (what this directory does not prove)
`"xla"` and `"tvm"` are claimed backends with **no test anywhere** in `./test`, so their support rests on `../benchmarks/BACKENDS.md` hand-measurements only.
The no-silent-fallback case is pinned for a bad *compile option* and for device placement on ROCm; nothing tests an unknown backend *name* (`backend="nope"`), a kernel `iree` cannot lower, or a failing backend never retrying on numpy.
Both legs compare against numpy — another `evoxc` backend — as the oracle, so a defect shared by evoxc's frontend/lowering is invisible here; the independent JAX oracle in `../parity` covers the numpy leg only.
Only XOR (`func_fit`) is exercised on either compiled leg: no `scan`/`while_loop` rollout (`CartPole`/`Pendulum`/`MountainCarContinuous`), no `RecurrentGenome` tick loop, no `HyperNEAT` — though these loops are exactly why IREE 3.13 is the floor, so the loop-carrying shapes are the largest untested risk on both legs.
Generation 1's payload is the host-seeded `setup()` population, so it is identical across backends by construction; only generation 2's payload exercises compiled mutation/crossover.
`../.github/workflows/ci.yaml` runs the iree leg with `continue-on-error: true`, so a red iree job does not fail CI, and a GPU-less runner gets the ROCm module's structural skip.
The guard's pure helpers are pinned by `test_conftest.py` (41 cases: the 25-row marker table, the install-command pins, the explicit `implies_requires_iree` coverage), but the absent-extra *hook* itself — `pytest_configure` calling `pytest.exit` — is evidenced by the two-environment commands above and the `block_iree` simulation rather than by a committed test, because it needs a process whose `iree` is genuinely absent.

## Constraints
`./conftest.py` imports nothing optional and decides availability with `find_spec`; it must never probe `/dev/kfd` or the HIP driver — those live in `test_rocm_pipeline.py`, which runs its probe once at import and short-circuits on `/dev/kfd` before touching the driver.
It reads `_pytest.mark.expression` (pytest's own `-m` parser — private, but stable and the only way to reuse that grammar exactly) and `pytest` itself.
In both test modules `pytest.importorskip("iree")` is the first executable statement after `import pytest`; every other import sits below it, so a missing extra can never surface as a collection error.
No test here imports `jax`; `iree` is the only optional-dependency coupling, and `iree.runtime` does not pull JAX in.
The configuration stays tiny on purpose: the compile dominates the runtime, and a bigger population buys nothing the numpy comparison does not already prove.
`test/backends/__init__.py` namespaces the directory so pytest's prepend import mode cannot collide with another suite's `test_*.py` basename, and both test modules import the shared harness as `backends._tiny_xor`.
This directory writes nothing outside `test/backends`; `../src`, `../benchmarks`, `../pyproject.toml` and `../conftest.py` are read-only inputs.

## Routing Table
| Path | Purpose |
|---|---|
| `./_tiny_xor.py` | the shared tiny-XOR configuration, driver, run-configuration tuple and the `assert_populated`/`assert_same_run` assertions both compiled legs use (not collected). |
| `./conftest.py` | the request guard: hard error for an explicit `-m requires_iree`/`-m requires_rocm` request with iree absent, announced structural skip when the directory is merely named; `INSTALL_COMMAND` spells the IREE 3.13 wheels. |
| `./test_iree_pipeline.py` | the llvm-cpu leg: execution, same-seed equality against numpy, and the no-silent-fallback option check (marker `requires_iree`). |
| `./test_rocm_pipeline.py` | the AMD GPU leg: `device="rocm"` + `target_backends=["rocm"]`, GPU-resident state, same-seed equality against numpy; skips on a missing `/dev/kfd`, a failing HIP driver query or zero devices (markers `requires_iree` + `requires_rocm`). |
| `./test_conftest.py` | the guard's truth table (`implies_marker`/`implies_requires_iree` for both markers, 25 rows), the install-command pins and the explicit wrapper coverage — no optional dependency, collected in every gate. |
| `../conftest.py` | registers the `requires_iree` and `requires_rocm` markers and puts `src/` on `sys.path`. |
| `../../benchmarks/backend_measure.py` | build time vs per-generation time vs peak RSS for one `(case, backend)`. |
