# tutorials — teaching material for the current TensorNEAT

## Intent
One executed notebook that teaches the current 0.2.0 flow end to end (genome → algorithm → problem → `Pipeline` → `evoxc` graph introspection), plus an archive of the pre-0.2.0 JAX notebooks kept for concepts only.
Nothing here is imported by `src/`, `test/` or `examples/`; tutorials are read by humans and executed by hand.

## API Surface
- `quickstart.ipynb` — the current tutorial. Runs top to bottom on the default `backend="numpy"` with no optional dependency, solves 2-input XOR, and carries its executed outputs.
  Its configuration: `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=8, output_transform=ACT.sigmoid)`, `NEAT(pop_size=100, species_size=10, survival_threshold=0.01)`, `XOR()`, `seed=0`, `fitness_target=-1e-3`, `generation_limit=100`, `is_detailed=False`; at HEAD it reaches the target at generation 81 with best fitness `-0.000934` and shape signature `(100, 8, 100, 4, 16, 108)`, ~0.3 s of that wall clock being the `evoxc.build`.
- Sections 2–3 teach that `Pipeline.setup()` builds no graph — it only creates the state, so `pipeline.builds` is `0` and `pipeline.shape_signature` is `None` right after it.
  The generation graph is compiled lazily by `Pipeline._executable_for` on the first `step()`, which is where `builds` goes to `1` and stays there; a different shape is a second cache entry keyed by `(pop_size, max_nodes, max_conns, max_in_degree, max_depth, key_span)`.
- Sections 4–6 cover `auto_run(state) -> (state, best)`, `show(state, best)` and the batch-first payload: `best` is a one-individual `GenomeTensors` (field shapes `(1, 8)`, `(1, 8, 4)`, `(1, 100, 2)`, `(1, 100, 1)`), `genome.network_dict(state, best)` takes it as one argument, and the pre-0.2.0 loose-row form `network_dict(state, *best)` is demonstrated raising `TypeError`.
- Section 7 is backend *documentation plus an install probe*, not a backend switch: only `"numpy"` is built and run anywhere in the notebook, and its one code cell prints `importlib.util.find_spec("iree") is not None` and `pipeline.backend`, compiling and running nothing on a second backend.
  Its install route for `"iree"` is `uv sync --extra iree`, or without uv the pinned compiler `pip install "evoxc @ git+https://github.com/EMI-Group/evoxc@fe85a534192f749a97b2f6e3b13b0800f69b8d90"` plus the IREE wheels `iree-base-compiler` / `iree-base-runtime` 3.13 (named directly, because `evoxc[iree]` pins IREE `>=3.11,<3.12`), and it states that plain `pip install "evoxc[iree]"` is unsupported because PyPI only ships `evoxc` 0.1.0 while `pyproject.toml` resolves the upstream `main` commit `fe85a53` through `[tool.uv.sources]`.
  The same section quotes the measured cost of that leg — one 8.0–9.6 s `iree-compile` per static shape and 1.54× / 5.47× / 6.63× slower per generation than numpy at pop 100 / 500 / 1000 from `benchmarks/BACKENDS.md` — so no speed-up is implied.
  It also states that `"xla"` needs a PJRT C API plugin `.so` and no CPU plugin is pip-installable (the `xla-rocm` extra installs the `jax-rocm7-pjrt` wheel, which lands at `<site-packages>/jax_plugins/xla_rocm7/xla_rocm_plugin.so`; `evoxc` discovers an installed `jax_plugins/*/*_plugin.so` by itself, and `compile_options["plugin_path"]` or `EVOXC_PJRT_PLUGIN` only pins a different one), that a long `while` on ROCm needs `XLA_FLAGS=--xla_gpu_enable_command_buffer=WHILE`, and that `"tvm"` is an `evoxc[tvm]` extra whose backend this project never measured.
- Section 8 and the graph-inspection block that follows it show the graph mechanics: a standalone `@evoxc.defn` function built with `evoxc.build` and run with `evoxc.run`, then `pipeline._executable_for(state)` reporting one `evoxc.pipeline.Executable` on the numpy backend with 16 graph inputs, 18 outputs and one cached executable; the private accessor is labelled illustrative and `pipeline.builds` / `pipeline.build_seconds` / `pipeline.shape_signature` are named as the supported counters.
- The "where to go next" section points at `examples/` (both HyperNEAT variants, `CustomFuncFit`, `RecurrentGenome`) and names the three simulator-free RL environments and their scripts: `CartPole` (`examples/rl/cartpole.py`, discrete), `Pendulum` (`examples/rl/pendulum.py`, continuous torque) and `MountainCarContinuous` (`examples/rl/mountaincar.py`, continuous force).
- Re-execute it after any API change, from the repo root: `uv sync --no-dev`, then `MPLBACKEND=Agg uv run --with nbconvert,nbclient,ipykernel jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 tutorials/quickstart.ipynb` — `jupyter` is not a project extra, so `--with` provisions nbconvert / nbclient / ipykernel for that run only.

## Constraints
- Outputs are committed as part of the tutorial and must come from an execution of the current tree (exit 0, zero error outputs, ~6 s wall clock at HEAD).
- The notebook needs no optional extra: only the numpy default is exercised, so its `iree installed here:` line reports the environment (`False` after the default `uv sync --no-dev`, `True` only when the `iree` extra is synced) and must never be read as a measurement.
- Keep the whole notebook under ~3 minutes of compute; cap evolution with a modest `fitness_target` / `generation_limit`.
- Evolution figures are a function of the current tree: a `src/` change that alters the RNG stream, mutation order or tensorization schedule changes the generation count, fitnesses and predictions, so re-execute the notebook in the same change.
- Numbers borrowed from `benchmarks/BACKENDS.md` / `benchmarks/RESULTS.md` / `benchmarks/GPU.md` must match those files as committed at HEAD, and are quoted with their source; never invent a measurement.
- Teach, do not duplicate: runnable copy-paste demos live in `examples/`, so a tutorial cell explains a mechanism the scripts do not.
- The archive under `./legacy_jax` is byte-identical historical material — never execute it or lift code from it.

## Test Strategy
- There is no pytest target here; the check is `nbconvert --execute`, which must exit 0 with zero error outputs, plus the printed best fitness reaching the target (`generations: 81`, `best fitness: -0.000934`, `target reached: True`).
- For a fixed tree the run is deterministic: two consecutive executions produce identical evolution figures (generation counts, fitnesses, predictions, `network_dict` output), so a re-execution differs only in the compile-time figure, wall clock, and the environment-reporting `iree installed here:` flag.
- Every code cell also runs headless as a plain script (concatenate the code cells, `MPLBACKEND=Agg .venv/bin/python script.py`), which is the cheap check when `nbconvert` is unavailable.

## Routing Table
| Path | Purpose |
|---|---|
| `./quickstart.ipynb` | Current tutorial: build a genome/algorithm/problem, `Pipeline` setup/step/auto_run, the batch-first `best` payload, backend selection documented with an `iree` install probe, evoxc graph inspection. |
| `./legacy_jax` | Archived pre-0.2.0 JAX notebooks and their assets — concepts only, none of the code runs on this library. |
