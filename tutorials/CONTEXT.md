# tutorials — teaching material for the current TensorNEAT

## Intent
One executed notebook that teaches the current 0.2.0 flow end to end (genome → algorithm → problem → `Pipeline` → backend switch → evoxc introspection), plus an archive of the pre-0.2.0 JAX notebooks kept for concepts only.
Nothing here is imported by `src/`, `test/` or `examples/`; tutorials are read by humans and executed by hand.

## API Surface
- `quickstart.ipynb` — the current tutorial. Runs top to bottom on the default `backend="numpy"` with no optional dependency, solves 2-input XOR, and carries its executed outputs.
  Its configuration: `DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=8, output_transform=ACT.sigmoid)`, `NEAT(pop_size=100, species_size=10, survival_threshold=0.01)`, `XOR()`, `seed=0`, `fitness_target=-1e-3`, `generation_limit=100`; it reaches the target at generation 85.
- Re-execute it after any API change:
  `uv run --with jupyter,nbconvert,nbclient,ipykernel jupyter nbconvert --to notebook --execute --inplace tutorials/quickstart.ipynb`
  (on this box that takes ~7 s wall clock, ~0.3 s of which is `evoxc.build` for the generation graph).

## Constraints
- Outputs are committed as part of the tutorial: a stale output is a bug, so re-run the command above and commit the new notebook.
- The notebook must never need an optional extra (`evoxc[iree]`, `evoxc[xla]`, `evoxc[tvm]`); anything that was not actually executed is stated as unverified in the markdown instead of being implied.
- Keep the whole notebook under ~3 minutes of compute; cap evolution with a modest `fitness_target` / `generation_limit`.
- Numbers borrowed from `benchmarks/BACKENDS.md` / `benchmarks/RESULTS.md` must match those files, and are quoted with their source; never invent a measurement.
- Teach, do not duplicate: runnable copy-paste demos live in `examples/`, so a tutorial cell explains a mechanism the scripts do not.
- The archive under `./legacy_jax` is byte-identical historical material — never execute it or lift code from it.

## Test Strategy
- There is no pytest target here; the check is `nbconvert --execute`, which must exit 0 with zero error outputs, plus the printed best fitness reaching the target.
- The notebook is deterministic (`seed`, config and shape are fixed), so a re-execution differs only in the compile time and other wall-clock figures.

## Routing Table
| Path | Purpose |
|---|---|
| `./quickstart.ipynb` | Current tutorial: build a genome/algorithm/problem, `Pipeline` setup/step/auto_run, batch-first `best` payload, backend switch, evoxc graph inspection. |
| `./legacy_jax` | Archived pre-0.2.0 JAX notebooks and their assets — concepts only, none of the code runs on this library. |
