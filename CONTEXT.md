# TensorNEAT 0.2.0
## Intent
TensorNEAT is a NEAT / HyperNEAT neuroevolution library whose computation layer is the **EvoX Compiler (`evoxc`)** — no JAX and no PyTorch anywhere in `src/`.
The NEAT idea is unchanged: irregular network topologies are *tensorized* into fixed-shape arrays so a whole population of differently-shaped networks is evaluated at once, with the population on axis 0 of every tensor.
What changed is the engine: every generation is one `@evoxc.defn` graph, built once per shape (`evoxc.build`) and then run once per generation on a switchable backend, instead of being `jax.jit`/`vmap`-traced per call.
The library keeps its research identity (GECCO 2024 Best Paper, arXiv 2404.01817): batched gradient descent through evolved topologies, network visualization, and sympy formula / Python-code generation.
## Layout
- `src/tensorneat/` — the only installable package (setuptools src-layout); everything importable lives here.
- `test/` — pytest suite: `unit/`, `parity/` (vs a frozen legacy implementation), `integration/`, `backends/` (compiled backends).
- `benchmarks/` — old-vs-new harness, the frozen pre-rewrite JAX implementation used as timing baseline and numerical oracle, and the committed measured reports.
- `examples/` — runnable scripts per problem family (`func_fit/`, `rl/`, `interpret_visualize/`).
- `docs/` — Sphinx autodoc site (strict build is clean); `tutorials/` — `quickstart.ipynb` (current API) plus `legacy_jax/` (pre-rewrite notebooks, concepts only).
- `pyproject.toml` + `uv.lock` + `.python-version` — packaging, the locked environment, and the interpreter pin; `README.md` is the front page and must match the code.
## API Surface
- `from tensorneat import Pipeline, algorithm, genome, problem, common`; `tensorneat.__version__` is the installed distribution version.
- `Pipeline(algorithm, problem, seed=42, fitness_target=1, generation_limit=1000, backend="numpy", device=None, compile_options=None, is_save=False, save_dir=None, is_detailed=True, show_problem_details=False, pop_batch_size=None)` → `setup()` (host-only, builds nothing), `step(state)`, `auto_run(state)`, `show(state, best)`.
- `algorithm.NEAT(...)`, `algorithm.HyperNEAT(substrate, neat)`, `genome.DefaultGenome/RecurrentGenome`, `problem.XOR/XOR3d/CustomFuncFit/CartPole/Pendulum/MountainCarContinuous`, `common.ACT/AGG/State`.
- A genome payload is a four-array record (`node_keys`, `node_attrs`, `conn_keys`, `conn_attrs`) and is **batch-first**: `auto_run` returns `best` as a one-individual batch (`(1, max_nodes)`, …), which `show` and `genome.network_dict(state, best)` consume; bare rows are accepted by the host helpers and written one-genome-per-file by checkpointing.
- Configuration objects are constructor-level and host-side; every numerical operation is a batch-first kernel inside an `evoxc` graph.
## Environment (uv)
`uv` is the supported environment manager: `uv lock` then `uv sync` (add `--extra bench` for the parity oracle, `--extra iree`/`--extra compilers` for compiled backends, `--extra docs` for Sphinx), `uv run pytest`, `uv run python examples/func_fit/xor.py`.
`requires-python = ">=3.11"`; `.python-version` pins 3.12; `uv.lock` is committed and covers all 59 packages of the resolved set.
`evoxc` is resolved by `[tool.uv.sources]` from git tag `v0.1.1` because PyPI only ships 0.1.0, which predates the frontend work that makes a build 0.32 s instead of 1.55 s; the metadata floor is `evoxc >= 0.1.1`, which PyPI cannot satisfy, so a plain `pip install tensorneat` fails and the README's git-pinned `evoxc` plus `pip install --no-deps` recipe is the supported pip path.
## Backends
`backend=` is forwarded verbatim to `evoxc.build`; `compile_options` is expanded into its `**options`.
`"numpy"` is the default, the fastest leg on CPU today, and the only one with no optional dependency.
`"iree"` lowers, compiles and runs every shipped problem family on llvm-cpu, agrees with numpy numerically, and is 5–24× slower per generation here at 8–11 s of compile per static shape (`benchmarks/BACKENDS.md`).
`"xla"` requires a user-supplied PJRT C API plugin (unavailable on this box) and `"tvm"` is untested.
A backend that cannot run a kernel raises; nothing ever silently falls back to another backend.
## Performance
`benchmarks/RESULTS.md` is the generated full-matrix old-vs-new report and its **timing** verdict is **FAIL**: the numpy interpreter beats the frozen JAX baseline on XOR (1.15×/1.68×/2.79× faster at pop 100/500/1000) but loses on XOR3d (0.61×/0.59×/0.88× of the old speed) and Pagie (0.17×), so node-for-node the rewrite is a portability/dependency trade with a mixed CPU-speed record, not a uniform speed-up.
Its quality rows predate the search-quality fixes below; `benchmarks/results/xor3d_quality_2026-11-04.md` is the post-fix search-quality record.
The rewrite's wins are real and recorded: no JAX/torch runtime dependency, one explicit persistent graph per shape, ~5× lower build latency, lower memory on the compiled leg, and numpy/iree/xla/tvm behind one `backend=` argument.
`benchmarks/PERF_NUMPY.md` records the numpy op-count pass that cut a generation by 39 % (pop 500) and 45 % (pop 1000) with bit-identical results; its committed before/after pair is the reference for further work.
Remaining lever, measured and out of scope so far: the padded gene geometry (`max_nodes`/`max_conns` multiply every per-pair op) and `evoxc`'s interpreter per-op overhead (~63 % of a generation).
## Constraints
- `import tensorneat` must never import `jax`, `torch`, `brax`, `gymnax` or `mujoco` (a release-gate test enforces this); compiled backends are pulled in only by `evoxc.build(backend=...)`.
- All computation lives inside `@evoxc.defn` graphs or helpers called during an active trace; there is no eager mode — host code only orchestrates build/run/log.
- Shapes are static: population, `max_nodes`, `max_conns` are fixed at build time; a shape change forces a rebuild (the Pipeline caches one executable per shape signature).
- Portability rules for every graph, from the StableHLO v1 exporter: rank ≤ 2 `scatter` (one prepared index block per scatter), no unsigned dtypes, static dims, non-negative axes, no `cumsum`/`tril`/`triu`/`flip`/`erf`/`gelu`; never seed a `while`-carried value with a `scatter` (it crashes `iree-compile` while exporting cleanly).
- Control flow (`scan`/`cond`/`while_loop`) appears only at the outermost layer of a kernel, and `vmap` is never applied over control flow — batch dimensions are written explicitly instead.
- `State` is an immutable host container for configuration, `evoxc` RNG keys and tensors; all randomness flows through `evoxc.random` stateless keys, so a `(seed, config, shape)` triple reproduces a run.
- NaN padding semantics of the pre-rewrite library are preserved (unconnected node value = `NaN`, `NaN` fitness = `-inf`), because `test/parity/` compares against the frozen implementation.
- Files stay under ~1000 lines; `common/tensorization.py` is the one documented exception (one cohesive schedule derivation).
- No silent error swallowing: configuration errors raise `tensorneat.common.errors.TensorNEATError`; trace/shape/backend errors propagate from `evoxc` unchanged.
- Dropped on purpose: the Brax/Gymnax/mujoco-playground RL environments required JAX; `problem.rl` ships three simulator-free evoxc-native replacements instead — `CartPole` (discrete), `Pendulum` (continuous torque), `MountainCarContinuous` (continuous force, one-time goal bonus).
## Test Strategy
`uv run pytest` from the repo root (`testpaths = ["test"]` in `pyproject.toml`); `test/` mirrors the source tree and the current full run is green with `uv sync --all-extras` (see `test/CONTEXT.md` for the live per-directory counts).
`test/unit/` runs on the numpy backend with no optional dependency (the Pendulum and MountainCarContinuous suites carry their own in-file host-numpy oracles, in the same spirit as `test_cartpole.py`); `test/parity/` needs `--extra bench` and compares kernels against `benchmarks/legacy_jax` within documented tolerances (rtol 1e-5 forward, 1e-4 gradients; 53 tests); `test/integration/` evolves XOR, asserts determinism per seed, rebuild-on-shape-change and checkpoint round-trips; `test/backends/` runs a real `iree` generation and skips when the extra is absent; `test/unit/benchmarks/` covers the harness's own git fingerprint; `test/release_gate_no_jax.py` enforces the import hygiene.
## Routing Table
| Path | Responsibility |
|---|---|
| `./src/tensorneat` | The package: pipeline (the single compile site), host payload helpers, genome/algorithm/problem/common subtrees. |
| `./test` | Pytest suites: unit, parity vs the frozen legacy library, integration, compiled backends, release gates. |
| `./benchmarks` | Old-vs-new harness + frozen JAX oracle; `RESULTS.md`/`THREADS.md`/`iree.md` are generated, `BACKENDS.md` is hand-written, `PERF_NUMPY.md` is the hand-measured op-count pass. |
| `./examples` | Runnable scripts per problem family; six of twelve cross their fitness target inside the shipped demo budget (all twelve exit 0). |
| `./docs` | Sphinx configuration, autodoc stubs (one per public module) and the built-site rules. |
| `./tutorials` | `quickstart.ipynb` for the current API (`legacy_jax/` keeps the pre-rewrite notebooks, concepts only). |
| `./imgs` | Static assets; only the two `evox_brand_*.svg` logos are referenced (README header) — the network/RL figures and GIFs are unreferenced pre-rewrite leftovers. |
| `./.github/workflows` | `ci.yaml` (uv sync + test + parity + import-hygiene + optional iree) and the PyPI release workflow. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Packaging metadata, the locked environment, the interpreter pin — a dependency change needs `uv lock`. |
| `.readthedocs.yaml` | Read the Docs build (Python 3.12, uv-based install, sphinx-build). |
## License
`LICENSE` is GPL-3.0 and the packaging metadata says so; nothing in this repo is BSD-licensed.
