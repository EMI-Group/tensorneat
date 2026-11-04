# tutorials — teaching material for the current TensorNEAT

## Intent
One executed notebook that teaches the current 0.2.0 flow end to end (genome → algorithm → problem → `Pipeline` → backend switch → evoxc introspection), plus an archive of the pre-0.2.0 JAX notebooks kept for concepts only.
Nothing here is imported by `src/`, `test/` or `examples/`; tutorials are read by humans and executed by hand.

## API Surface
- `quickstart.ipynb` — the current tutorial. Runs top to bottom on the default `backend="numpy"` with no optional dependency, solves 2-input XOR, and carries its executed outputs.
  Its configuration: `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=8, output_transform=ACT.sigmoid)`, `NEAT(pop_size=100, species_size=10, survival_threshold=0.01)`, `XOR()`, `seed=0`, `fitness_target=-1e-3`, `generation_limit=100`, `is_detailed=False`; it reaches the target at generation 85 with best fitness `-0.000789`.
- Section 2 teaches that `Pipeline.setup()` builds no graph — it only creates the state, so `pipeline.builds` is `0` and `pipeline.shape_signature` is `None` right after it.
  The generation graph is compiled lazily by `Pipeline._executable_for` on the first `step()`, which is where `builds` goes to `1` and stays there; a different shape is a second cache entry.
- Section 7 states the backend facts: `"numpy"` is the only backend the notebook exercises; `"iree"` needs `pip install "evoxc[iree]"`, pays one 7.0–10.4 s `iree-compile` per static shape, and is 9.3× / 93.9× / 102.7× slower per generation than numpy at pop 100 / 500 / 1000 (`benchmarks/BACKENDS.md`), so no speed-up is implied.
- The same section says `"xla"` has no pip extra and no installable dependency at all: the evoxc xla adapter drives a user-supplied PJRT C API plugin `.so` found through `plugin_path` in `compile_options` or the `EVOXC_PJRT_PLUGIN` environment variable, and nothing provides one on this box.
  `"tvm"` is an `evoxc[tvm]` extra that this project's measurements never attempted.
- The "where to go next" section names the three simulator-free RL environments the package ships and their scripts: `CartPole` (`examples/rl/cartpole.py`, discrete), `Pendulum` (`examples/rl/pendulum.py`, continuous torque) and `MountainCarContinuous` (`examples/rl/mountaincar.py`, continuous force).
- Re-execute it after any API change, from the repo root: `export PATH=/root/.local/bin:$PATH`, `uv sync --all-extras`, then
  `uv run --with jupyter,nbconvert,nbclient,ipykernel jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 tutorials/quickstart.ipynb`
  — the committed outputs were regenerated from the current tree by exactly that command (exit 0, ~7 s wall clock, ~0.26 s of it the `evoxc.build` for the generation graph).

## Constraints
- Outputs are committed as part of the tutorial: a stale output is a bug, so re-run the command above and commit the new notebook.
- The notebook needs no optional extra: only the numpy default is exercised, and the `iree` cell merely prints whether `evoxc[iree]` happens to be installed (`True` in an `--all-extras` environment) without compiling or running anything on it; anything not actually executed is stated as unverified instead of being implied.
- Keep the whole notebook under ~3 minutes of compute; cap evolution with a modest `fitness_target` / `generation_limit`.
- Numbers borrowed from `benchmarks/BACKENDS.md` / `benchmarks/RESULTS.md` must match those files as committed at HEAD, and are quoted with their source; never invent a measurement.
- Teach, do not duplicate: runnable copy-paste demos live in `examples/`, so a tutorial cell explains a mechanism the scripts do not.
- The archive under `./legacy_jax` is byte-identical historical material — never execute it or lift code from it.

## Test Strategy
- There is no pytest target here; the check is `nbconvert --execute`, which must exit 0 with zero error outputs, plus the printed best fitness reaching the target (`generations: 85`, `best fitness: -0.000789`, `target reached: True`).
- The notebook is deterministic (`seed`, config and shape are fixed), so a re-execution differs only in wall-clock figures (compile time, timestamps) and in the `iree installed here:` flag, which reports the environment rather than a measurement.

## Routing Table
| Path | Purpose |
|---|---|
| `./quickstart.ipynb` | Current tutorial: build a genome/algorithm/problem, `Pipeline` setup/step/auto_run, batch-first `best` payload, backend switch, evoxc graph inspection. |
| `./legacy_jax` | Archived pre-0.2.0 JAX notebooks and their assets — concepts only, none of the code runs on this library. |
