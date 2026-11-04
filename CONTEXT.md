# TensorNEAT

## Intent
TensorNEAT is a NEAT / HyperNEAT neuroevolution library whose computation layer is the **EvoX Compiler (`evoxc`)** — no JAX and no PyTorch anywhere in `src/`.
The NEAT idea is unchanged: irregular network topologies are *tensorized* into fixed-shape arrays so a whole population of differently-shaped networks is evaluated at once.
What changed is the engine: evoxc graphs are built once (`@evoxc.defn` → `evoxc.build`) and executed on a switchable backend (reference numpy interpreter by default; `iree`, `tvm`, `xla` opt-in), instead of being `jax.jit`/`vmap`-traced.
The library keeps its research identity (GECCO 2024 Best Paper, arXiv 2404.01817): batched gradient descent through evolved topologies, network visualization, and sympy formula/Python-code generation are still supported.

## Layout
- `src/tensorneat/` — the only installable package (setuptools src-layout); everything importable lives here.
- `benchmarks/` — old-vs-new performance harness **and** the frozen pre-rewrite JAX implementation used as the timing baseline and numerical oracle.
- `test/` — pytest suite: `unit/`, `parity/` (vs the frozen legacy implementation), `integration/`.
- `examples/` — runnable scripts per problem family (`func_fit/`, `rl/`, `interpret_visualize/`).
- `docs/` — Sphinx autodoc site; `tutorials/` — teaching notebooks; `imgs/` — static assets.
- `pyproject.toml` — build config, runtime deps, optional extras, pytest settings.

## API Surface
- `from tensorneat import Pipeline, algorithm, genome, problem, common` is the documented entry point; `tensorneat.__version__` is the installed distribution version (`"0.0.0"` outside an installed checkout).
- `Pipeline(algorithm, problem, seed=42, fitness_target=..., generation_limit=1000, backend="numpy", device=None, compile_options=None, is_save=False, is_detailed=True)` → `setup()`, `auto_run(state)`, `step(state)`, `show(state, best)`.
- `algorithm.NEAT(...)`, `algorithm.HyperNEAT(substrate, neat)`, `genome.DefaultGenome/RecurrentGenome`, `problem.XOR/XOR3d/CustomFuncFit/CartPole`, `common.ACT/AGG/State`.
- Configuration objects are constructor-level and host-side; every numerical operation is a batch-first evoxc kernel (population on axis 0).

## Backends
`backend=` is forwarded verbatim to `evoxc.build`; `"numpy"` (reference interpreter, always available) is the measured CPU backend.
`"iree"` (llvm-cpu) compiles and runs every shipped case end to end since the graph formulations were made v1-exportable; on this box it is **slower** than numpy per generation (5× at pop 100, 24× at pop 500, 23× at pop 1000) and costs ~9–11 s of `iree-compile` per static shape — the numbers, protocol and per-gate fix record live in `benchmarks/BACKENDS.md`.
`"xla"` is unavailable without a user-supplied PJRT plugin `.so` (`EVOXC_PJRT_PLUGIN`); `"tvm"` is untested here.
Unsupported shapes/kernels must raise the backend's error — never silently fall back to another backend.

## Graph portability (what every kernel must respect)
The StableHLO v1 exporter is the portability gate: `evoxc`'s `DEFERRED_OPS` (`cumsum`, `cumprod`, `tril`, `triu`, `flip`, `erf`, `gelu`, `where`, `scan`, `convolution`, `runtime_call`, `external_call`, `sparse_*`) have no emission, `scatter` accepts only rank-1 axis-0 and rank-2 row/column writes, and iree declares no unsigned dtype.
Kernels therefore use static constants, static-index `gather`s, log2-doubling folds, same-width signed carriers (plus an explicit sign flip where unsigned *order* matters), and at most one `scatter` per prepared index block.
The rules with their evidence live in `src/tensorneat/common/CONTEXT.md`, `src/tensorneat/algorithm/neat/CONTEXT.md` and `src/tensorneat/genome/operations/CONTEXT.md`.

## Performance Contract
The rewrite must be at least as fast as the frozen JAX baseline on CPU (pop 100/500/1000, XOR and XOR3d), and the measurement must live in `benchmarks/RESULTS.md` (numbers + how to reproduce).
That contract is measured on the `numpy` reference backend; the `iree` compiled leg was measured separately (see `benchmarks/BACKENDS.md`) and does not meet it, so it is not part of the verdict.
Legacy measured on this machine (JAX 0.12 CPU, 1 thread-bound box): 51 ms/gen at pop 100, 109 ms/gen at pop 500, 157 ms/gen at pop 1000.
The winning structural idea is *edge-proportional, depth-parallel* evaluation: work scales with real connections and network depth, not with `max_nodes × max_nodes` padding waste.

## Measured Cost of one generation
`backend="numpy"`, XOR, seed 42, `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100)`, `NEAT(pop_size=P, species_size=20, survival_threshold=0.01)`, build excluded, 3 warmup + 20 timed `Pipeline.step`, one process per cell, before/after one op-count pass taken in a single session (`benchmarks/PERF_NUMPY.md` holds the full record, the profile split and the reproduce commands):
| cell | median ms/gen before | median ms/gen now | peak RSS before → now |
| --- | --- | --- | --- |
| xor p100 | 43.47 | 38.58 | 118.2 → 114.4 MiB |
| xor p500 | 268.39 | 159.90 | 260.8 → 240.5 MiB |
| xor p1000 | 342.21 | 270.08 | 431.6 → 391.5 MiB |
| xor3d p500 | 290.04 | 176.98 | 291.3 → 264.7 MiB |
A generation is now **8 150 executed IR ops** (was 10 767), and ~63 % of the remaining wall time is the numpy interpreter's fixed per-op cost (~13 µs: operand resolution + dtype/shape validation + `Tensor` construction in `evoxc/backends/numpy/interpreter.py::_run_block`), not array work — so **executed op count is the lever for any further win**, and host-side bookkeeping in `Pipeline.step` is 0.1 %.
The interpreter keeps every SSA result of a block alive until that block returns, so a kernel's live set is charged once per enclosing block: **never hoist a kernel out of a `while`/`scan` without measuring peak RSS** (hoisting the 20 speciation rows was 28 % faster and 1.7× the memory; measuring them inside one `evoxc.scan` is 40 % faster and below the starting memory).
`benchmarks/RESULTS.md` predates this pass (its new-side rows are stale; the verdict must be re-rendered by `run_benchmark.py`, never hand-edited).

## Constraints
- `import tensorneat` must never import `jax`, `torch`, `brax`, `gymnax`, or `mujoco` (a release-gate test enforces this).
- Requires Python >= 3.11 (evoxc's floor); runtime deps are `evoxc`, `numpy`, `sympy`, `networkx`, `matplotlib`.
- All computation lives inside `@evoxc.defn` graphs or helpers called during an active trace; there is no eager mode — host code only orchestrates build/run/log.
- Shapes are static: population, `max_nodes`, `max_conns` are fixed at build time; a shape change forces a rebuild (the Pipeline caches per-shape executables).
- Control flow (`scan`/`cond`/`while_loop`) appears only at the outermost layer of a kernel; `vmap` is never applied over control flow (evoxc defers that) — batch dimensions are written explicitly instead.
- `State` is an immutable host container for configuration + evoxc RNG keys + tensors; all randomness flows through `evoxc.random` (stateless keys), so a `(seed, config, shape)` triple reproduces a run.
- NaN padding semantics from the pre-rewrite library are preserved (unconnected node value = NaN, NaN fitness = `-inf`), because `test/parity/` compares against the frozen implementation.
- Files stay under ~1000 lines; a file that grows past that has more than one responsibility.
- No silent error swallowing: configuration errors raise `tensorneat.common.errors.TensorNEATError`, backend/shape errors propagate from evoxc unchanged.

## Test Strategy
`pytest` from the repo root (`[tool.pytest.ini_options]` in `pyproject.toml`, `testpaths = ["test"]`); `test/` mirrors the source tree.
`test/unit/` runs on the numpy backend with no optional dependency; `test/parity/` requires the `bench` extra (JAX) and compares kernels against `benchmarks/legacy_jax`; `test/integration/` evolves XOR to the fitness target and asserts determinism for a fixed seed; `test/backends/` runs a real XOR generation on `backend="iree"` and compares it with numpy, and skips cleanly without the `evoxc[iree]` extra (marker `requires_iree`, in no gate).
`test/release_gate_no_jax.py` asserts that `import tensorneat` pulls in no JAX/torch module.

## Routing Table
| Path | Purpose |
|---|---|
| `./src/tensorneat` | The package: pipeline (compile site), kernel contract, genome/algorithm/problem/common subtrees. |
| `./benchmarks` | Old-vs-new benchmark harness, frozen legacy JAX implementation, spike results, `RESULTS.md`. |
| `./test` | Pytest suites: unit, parity vs legacy, integration, backends (`iree`), release gates. |
| `./examples` | Runnable scripts for each supported problem family. |
| `./docs` | Sphinx configuration and autodoc stubs. |
| `./tutorials` | Teaching notebooks (JAX-era; read for concepts, not for API accuracy). |
| `./imgs` | Static assets referenced by README/tutorials. |
| `./.github` | Release-to-PyPI workflow only; it runs no tests. |
