# TensorNEAT

## Intent
TensorNEAT is a JAX-based library for NeuroEvolution of Augmenting Topologies (NEAT) and HyperNEAT, targeting GPU acceleration.
Its central idea is "tensorization" of irregular network topologies: each genome is padded into fixed-shape arrays so a whole population of differently-shaped networks is evaluated by `jax.vmap` inside one `jax.jit`-ed generation step.
It additionally supports batched gradient computation through evolved topologies (`genome.grad`), network visualization, and sympy-based formula/Python-code generation.
It won the GECCO 2024 Best Paper Award (arXiv 2404.01817, ACM TELO 10.1145/3730406) and is compatible with the EvoX framework.

## Layout
- `src/tensorneat/` — the installable package (setuptools src-layout); all library code lives here, nothing importable exists outside it.
- `pyproject.toml` — build config only (`setuptools>=61` / `setuptools.build_meta`, packages found under `src`), runtime deps, `requires-python >= 3.9`.
- `test/` — pytest-style tests, throughput benchmarks, GPU memory profilers, and exploration notebooks.
- `examples/` — runnable scripts grouped by problem family: `func_fit/`, `brax/`, `gymnax/`, `mujoco_playground/`, `interpret_visualize/`, `with_evox/`.
- `docs/` — Sphinx site with autodoc stubs in `docs/source/`; built by Read the Docs via `.readthedocs.yaml`.
- `tutorials/` — three teaching notebooks (functional programming/`State`, genome, usage) plus sample `.svg`/`.pkl` fixtures.
- `imgs/` — README logos, evolved-network SVG diagrams, and RL animation GIFs; static assets, no code.
- `.github/workflows/pypi_publish.yaml` — the only workflow: on release, `python -m build` then `twine upload dist/*`.

## API Surface
- `from tensorneat import Pipeline, algorithm, genome, problem, common` is the documented entry point (`src/tensorneat/__init__.py`).
- `Pipeline(algorithm, problem, seed=42, fitness_target=1, generation_limit=1000, is_save=False, save_dir=None, show_problem_details=False, using_multidevice=False, pop_batch_size=None)`; `setup()`, `auto_run(state) -> (state, best)`, `show(state, best)`, `step(state)`.
- `algorithm.NEAT(genome, pop_size, species_size=10, ...)`, `algorithm.HyperNEAT(substrate, neat, ...)`, `algorithm.HyperNEATFeedForward`.
- `genome.DefaultGenome(...)` / `genome.RecurrentGenome(...)`, composed from `node_gene`, `conn_gene`, `mutation`, `crossover`, `distance`.
- `problem.XOR()`, `problem.XOR3d()`, `problem.CustomFuncFit(...)`, `problem.BraxEnv(...)`, `problem.GymNaxEnv(...)`, `problem.MujocoEnv(...)`.
- `common.ACT` / `common.AGG` registries (`add_func` + `update_sympy` to extend), `common.State`, `common.sympy_tools`.

## Generation Loop
One generation is `Pipeline.step`, compiled once via `jax.jit(self.step).lower(state).compile()` in `auto_run`.
Per generation: `algorithm.ask` → `vmap(algorithm.transform)` → `vmap(problem.evaluate)` (which calls `algorithm.forward`) → `algorithm.tell`.
`pop_batch_size` swaps the population `vmap` for `lax.scan` over batches to cap peak memory.

## Constraints
- `State` is an immutable registered pytree: `register`/`update`/`remove` return a new `State` and `__setattr__` raises.
- Everything reachable from `Pipeline.step` must be traceable: no host reads (`device_get`, `.item()`), no `print`, no Python branching on traced values.
- All genome arrays are fixed-shape and NaN-padded; `I_INF` (`np.iinfo(jnp.int32).max`) is the "absent index" sentinel for integer index paths.
- `problem.jitable` must be truthy (asserted in `Pipeline.__init__`) and `problem.input_shape[-1]` must equal `algorithm.num_inputs`.
- No in-place mutation anywhere — always `.at[].set` / `.at[].add`.
- Heavy RL dependencies (`brax`, `gymnax`, `mujoco_playground`) are imported lazily inside constructors, never at module import.
- `flax`, `jaxopt` and `optax` are declared in `pyproject.toml` but imported nowhere in `src/`; `evox` is a package-level optional extra, not a declared dependency.

## Conventions
- Style is PEP8 with 4-space indentation; there is no enforced formatter or linter (no ruff/black/isort/flake8/pre-commit config).
- Documentation uses Sphinx `napoleon` style; `docs/conf.py` mocks `evox` so autodoc works without it.
- There is no pytest configuration (`conftest.py`, `pytest.ini`, test extra all absent) and `pytest` is not a declared dependency.
- Only `test/test_grad.py` is pytest-collectable; most other `test/` files are scripts that execute at import time, and several still reference stale `algorithm.neat...` module paths.
- CI runs no tests at all — only the PyPI publish job.
- Public API changes must keep the README examples and `docs/source/*.rst` stubs in sync.
- Per-module known issues are recorded in the child `CONTEXT.md` files rather than at the root.

## Routing Table
| Path | Purpose |
|---|---|
| `./src` | Library source root; `./src/tensorneat` is the package — start any code change here. |
| `./src/tensorneat/algorithm` | Evolutionary layer: `NEAT` ask/tell, speciation, HyperNEAT + substrates. |
| `./src/tensorneat/genome` | Genome representation, genes, genetic operators, and the tensorized forward pass. |
| `./src/tensorneat/problem` | Task layer: function fitting (XOR/XOR3d/custom) and RL envs (brax/gymnax/mujoco). |
| `./src/tensorneat/common` | `State`, graph/tensorization helpers, `ACT`/`AGG` registries, EvoX adaptors. |
| `./test` | Tests, benchmarks (`benchmark_optimization.py`), memory profilers, notebooks. |
| `./examples` | Runnable usage scripts per problem family and integration target. |
| `./docs` | Sphinx autodoc site configuration and generated module stubs. |
| `./tutorials` | Step-by-step teaching notebooks. |
| `./imgs` | Static assets referenced by the README and tutorials. |
| `./.github` | Release-to-PyPI workflow only. |
