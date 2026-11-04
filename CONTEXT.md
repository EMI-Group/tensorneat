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
`backend=` is forwarded verbatim to `evoxc.build`; `"numpy"` (reference interpreter, always available), `"iree"`, `"tvm"`, `"xla"` are opt-in extras.
Unsupported shapes/kernels must raise the backend's error — never silently fall back to another backend.

## Performance Contract
The rewrite must be at least as fast as the frozen JAX baseline on CPU (pop 100/500/1000, XOR and XOR3d), and the measurement must live in `benchmarks/RESULTS.md` (numbers + how to reproduce).
Legacy measured on this machine (JAX 0.12 CPU, 1 thread-bound box): 51 ms/gen at pop 100, 109 ms/gen at pop 500, 157 ms/gen at pop 1000.
The winning structural idea is *edge-proportional, depth-parallel* evaluation: work scales with real connections and network depth, not with `max_nodes × max_nodes` padding waste.

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
`test/unit/` runs on the numpy backend with no optional dependency; `test/parity/` requires the `bench` extra (JAX) and compares kernels against `benchmarks/legacy_jax`; `test/integration/` evolves XOR to the fitness target and asserts determinism for a fixed seed.
`test/release_gate_no_jax.py` asserts that `import tensorneat` pulls in no JAX/torch module.

## Routing Table
| Path | Purpose |
|---|---|
| `./src/tensorneat` | The package: pipeline (compile site), kernel contract, genome/algorithm/problem/common subtrees. |
| `./benchmarks` | Old-vs-new benchmark harness, frozen legacy JAX implementation, spike results, `RESULTS.md`. |
| `./test` | Pytest suites: unit, parity vs legacy, integration, release gates. |
| `./examples` | Runnable scripts for each supported problem family. |
| `./docs` | Sphinx configuration and autodoc stubs. |
| `./tutorials` | Teaching notebooks (JAX-era; read for concepts, not for API accuracy). |
| `./imgs` | Static assets referenced by README/tutorials. |
| `./.github` | Release-to-PyPI workflow only; it runs no tests. |
