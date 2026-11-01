# benchmarks — performance and fidelity evidence
## Intent
Prove that the evoxc rewrite is at least as fast as the frozen pre-rewrite JAX library and searches just as well, on one box, with numbers anyone can regenerate.
Nothing here ships: the harness never imports both implementations in one process (JAX thread pools and evoxc/iree allocators would corrupt timings), it spawns one subprocess per run and renders `RESULTS.md` from the measured JSON.
`RESULTS.md` is generated evidence, never hand-edited.
## Layout
- `run_benchmark.py` — CLI: plans the case matrix, spawns the worker subprocesses, aggregates, regenerates `RESULTS.md`.
- `cases.py` — the case matrix: the logical workload (problem, genome, NEAT config, pop/gen budget, target) declared as data and shared by both implementations.
- `harness/` — protocol (schema, aggregation, verdict rules), per-implementation adapters, worker entry point, report writer.
- `spikes/` — measured formulation evidence behind the depth-parallel design (`FORMULATION.md` + runnable micro-benchmark).
- `legacy_jax/` — frozen pre-rewrite JAX implementation; the timing baseline and numerical oracle (read-only, see its `CONTEXT.md`).
- `results/` — machine-readable run records; `latest.json` is what the committed `RESULTS.md` was rendered from (git-ignored).
- `RESULTS.md` — the generated report (tables, verdict, how-to-reproduce).
## Metric Protocol
Every number comes from one worker run of one `(case, impl, backend, repeat)` tuple.
- `compile_time_s` — one-off build/compile of the generation step: on the old side `jax.jit(pipeline.step).lower(state).compile()`; on the new side the first `Pipeline.step` call, which performs the evoxc build (it also advances the state by one generation, reported as `generations_consumed_before_timing`).
- warmup — `--warmup` generations run untimed and discarded before any measurement.
- `generation_ms` — wall clock of one `step` call *including* the host sync (`jax.device_get` on the old side, evoxc host materialisation on the new side), sampled over `--generations` generations after the warmup.
- `ms_per_generation` — median of `generation_ms`; `gens_per_second` = `1000 / median`.
- `peak_rss_mb` — `resource.getrusage(RUSAGE_SELF).ru_maxrss` sampled after import, after compile and at process end; because it is a high-water mark, the three samples attribute memory to phases.
- quality — from the same timed run: per-generation `best_fitness` / `mean_fitness` / `valid_count` curves, `final_best_fitness`, and `generations_to_target` (first generation whose best fitness reaches the case's `fitness_target`; `null` means "not within the budget", rendered as `>N`).
- repeats — repeat `r` uses `seed + r`, so repeats sample timing noise *and* search variance; per-case values are medians over repeats with min/max kept in JSON.
## Fairness Rules (binding)
1. One implementation per process: the old side loads JAX, the new side asserts `jax`/`torch` never entered `sys.modules` after `import tensorneat` (a built-in release-gate check).
2. Runs are strictly sequential — never two benchmark processes at once, one fresh interpreter per run.
3. Thread count is set identically for both sides: `--threads N` exports `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=N` and, on the old side, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=N`; the default `--threads 0` leaves runtime defaults ("as users run it") and `--threads 1` reproduces a thread-bound box.
4. Identical work: same seed, same case config (pop size, genome sizes, species size, mutation defaults), same generation budget and fitness target for both sides; `cases.py` declares the config as plain data that the builders consume, so the JSON record and the constructed objects cannot drift.
5. No logging inside the timed region: stdout/stderr are redirected to `os.devnull` for the warmup+timed loop on both sides, the new pipeline runs with `is_detailed=False`, and the harness drives the compiled `step` directly so the legacy `analysis()`/print path is never entered.
6. Compile is measured, not charged to generations: the one-off build is reported separately and the generation it consumes on the new side is excluded from the quality curve.
7. Same dataset: XOR/XOR3d literals; `CustomFuncFit` uses the same sampler settings (`num_samples=500`, box `[-1,1]^2`) — legacy draws from the global `np.random` (seeded explicitly before construction), the new side from `default_rng(seed)`: same distribution, different draws.
8. The measured baseline wins: the declared numbers in `../CONTEXT.md` are printed next to the measured old row for sanity, but the verdict compares against the old row measured in the same run on the same box.
## What Is Not Compared
- Trajectories or bitwise equality: the sides use different PRNGs (JAX `PRNGKey` vs evoxc splitmix64), so one seed identifies the same *workload*, not the same search. Numerical fidelity is `test/parity/`'s job.
- Accelerator numbers: the contract is CPU; `--backend iree|xla|tvm` rows are informational.
- Import time and visualization/codegen cost (only `rss_after_import_mb` is recorded).
- CartPole against the legacy side: the pre-rewrite library has no evoxc CartPole (its RL problems are gym/brax/mujoco based and need extra dependencies), so that case is new-only and marked `n/a` for old.
## Verdict Rules
- timing PASS when `new_ms_per_generation <= old_ms_per_generation` for every case both sides ran (speed-up >= 1.00x).
- quality PASS when `new_final_best >= old_final_best - 0.10 * |old_final_best|` for every such case.
- overall PASS when both pass; `INCOMPLETE` while any run is `error`/`skipped` (listing them); otherwise FAIL with the offending rows named.
## How To Run
```
python benchmarks/run_benchmark.py --list
python benchmarks/run_benchmark.py                                    # both impls, numpy backend, all cases, 150 gens, 2 repeats
python benchmarks/run_benchmark.py --impl old --cases xor_p100 --generations 30 --repeats 1
python benchmarks/run_benchmark.py --impl new --backend numpy --cases xor_p100,xor_p1000
python benchmarks/run_benchmark.py --impl new --backend iree --cases xor_p500
```
The worker prepends `benchmarks/legacy_jax` (old side) or `src` (new side) to `sys.path`, so an installed `tensorneat` never shadows the tree under test; the legacy side needs the `bench` extra (`jax`), the new side only `evoxc` + `numpy`.
`--json` (default `benchmarks/results/latest.json`) holds per-run and aggregated records; `--out` (default `benchmarks/RESULTS.md`) is the regenerated report; `--dry-run` prints the planned subprocess commands without running them.
## Notes for Agents
- `--list` and `--dry-run` must work on an interpreter without numpy: `cases.py` is stdlib-only at module scope and `harness/protocol.py` guards its numpy import (`np = None`), so keep new module-scope imports on the numpy-free paths clean.
- A failing run is always an `error`/`skipped` row, never a missing row: `harness/worker.py` writes its record even on a traceback, and the CLI synthesises a row when a worker dies without writing one.
- `benchmarks/.gitignore` keeps `results/*.json` untracked; `RESULTS.md` is committed evidence and is only ever rewritten by `run_benchmark.py`.
- The committed `RESULTS.md` is currently INCOMPLETE because the rewrite is unimplemented: `import tensorneat` fails in the `common.State` skeleton (`NotImplementedError`) before any agent code runs.
- Bench environment: `jax` only on the old side, `evoxc` on the new side; the verdict compares the old row measured on the same box, never the declared 51/109/157 ms contract numbers.
## Constraints
- Python >= 3.11; harness dependencies are stdlib + `numpy` (plus `jax` on the old side only, via the `bench` extra).
- Never edit `legacy_jax/` — a moving baseline makes every number meaningless.
- Nothing outside `benchmarks/` is read except the two implementation trees under test, and nothing outside `benchmarks/` is written.
- Files stay under ~1000 lines with one responsibility each: protocol / adapters / worker / reporter / CLI / cases.
- A run that cannot work must fail loudly (`status="error"` with the traceback in JSON and a non-zero exit from the worker) — never a silently missing row.
## Routing Table
| Path | Purpose |
|---|---|
| `./run_benchmark.py` | CLI: matrix planning, subprocess orchestration, JSON and `RESULTS.md` regeneration. |
| `./cases.py` | Case matrix (problem + genome + NEAT config as data) and the per-implementation builders. |
| `./harness` | Schema/aggregation/verdict, adapter contract, worker entry point, report writer (`./harness/CONTEXT.md`). |
| `./spikes` | Measured formulation evidence for the depth-parallel design. |
| `./legacy_jax` | Frozen pre-rewrite JAX implementation: timing baseline and numerical oracle (read-only). |
| `./results` | Generated JSON run records (`latest.json`); git-ignored, safe to delete. |
| `./RESULTS.md` | Generated report: environment, protocol, tables, verdict, how to reproduce. |
| `../test/parity` | Numerical-fidelity suite using `./legacy_jax` as the oracle (sibling — read-only, escalate writes to parent). |
