# ./test/integration — whole-`Pipeline` runs on the numpy backend

## Intent
Prove the *assembled* library rather than its kernels: a `Pipeline` built from the documented public API evolves XOR, reproduces itself from `(seed, config, shapes)`, survives a shape-signature change, and leaves a loadable checkpoint.
Twenty-one tests across five modules drive the real `Pipeline` (`setup` → `step` / `auto_run`) on the numpy reference interpreter and assert on what the pipeline returns, prints or writes, never on an intermediate kernel.
No `jax`/`torch`, no network, no GPU, no wall-clock assertions: a run is judged by generations, fitness, payload shapes and compile counts.
Every module carries `pytestmark = [pytest.mark.slow]`; `pytest -m slow` selects the directory, the default gate leaves it out.

## Current state: every test runs
`src/tensorneat/pipeline.py` is the evoxc compile site, so `conftest.py::missing_evoxc_pipeline_features(tn)` returns an empty list and `require_evoxc_pipeline(tn)` does not skip; all 23 tests run on the numpy backend.
The probe is structural, never hardcoded: it inspects `inspect.signature(tn.Pipeline.__init__)` for the four documented compile kwargs (`PIPELINE_COMPILE_KWARGS = ("backend", "device", "compile_options", "is_detailed")`), checks whether the loaded `tensorneat.pipeline` module binds `jax`/`jnp`, and scans that module's source file for a top-level `import jax` (it never imports jax to find out).
The autouse fixture `_evoxc_pipeline_required(tn)` applies the probe to every test in the directory, so a regression that reintroduces an evoxc-free pipeline reports a skip naming the missing features instead of a directory of opaque errors.
No module imports `tensorneat` or a `support` module at module scope: `tn`, `payloads` and `runner` are `support.lazy.require` fixtures, and only `numpy`/`json`/`pytest` plus `from integration.conftest import …` appear at module scope.

## How a test builds and drives a `Pipeline`
`./conftest.py` owns the single configuration, `TINY_CONFIG`, grouped by the object that owns each knob: genome `num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, max_in_degree=4, max_depth=8, output_transform="sigmoid"`, `NEAT(pop_size=16, species_size=4)`, `XOR(error_method="mse")`, `Pipeline(seed=0, fitness_target=-0.01, generation_limit=60, backend="numpy", device=None, compile_options=None, is_save=False, save_dir=None, is_detailed=True, show_problem_details=False)`.
`tiny_pipeline(**overrides)` returns that `Pipeline` **configured but not run**; an unknown knob raises `TypeError`, and `output_transform` is either an `ACT` registry name (`"sigmoid"`), a callable, or `None`.
`fitness_target=-0.01` on XOR's negated `mse` means "mean squared error ≤ 0.01"; tests that need a fixed-length run pass `fitness_target=1.0`, which XOR can never reach, so exactly `generation_limit` generations are consumed.
`run_pipeline(pipeline)` is `setup()` + `auto_run()` and returns `(state, best)` — `auto_run`'s documented pair, nothing derived.
`step_once(pipeline, state)` runs one `pipeline.step` and returns `(state, (P,) fitness)`, validating the documented step contract: a tuple whose `[0]` is the new state and whose `[-1]` is a length-`pipeline.pop_size` floating array.
`step_loop(pipeline, state, generations=None, target=None)` drives `step_once` for a fixed count (or until `max(fitness) >= target`) and returns `(state, [max_fitness_per_generation])` — the trajectory oracle, read from what the pipeline reports rather than from an undocumented history attribute.
`state.pop` is a `genome.utils.GenomeTensors` (`node_keys (P,N) int32`, `node_attrs (P,N,4) float32`, `conn_keys (P,C,2) int32`, `conn_attrs (P,C,1) float32`); `state.generation` is a 0-d float32 evoxc tensor and must be read as `int(to_numpy(state.generation))`.

## Shared helpers (`./conftest.py`)
| Name | Contract |
|---|---|
| `missing_evoxc_pipeline_features(tn)` | List of missing evoxc-pipeline features; empty once the rewrite lands. |
| `require_evoxc_pipeline(tn)` | `pytest.skip` naming those features. |
| `tn`, `payloads`, `runner` | Module-scope lazy fixtures for the package, `support.payloads`, `support.runner`. |
| `tiny_pipeline` | Factory for the configured, unflagged `Pipeline` of `TINY_CONFIG`. |
| `run_pipeline` | `setup()` + `auto_run()` → `(state, best)`. |
| `step_once`, `step_loop` | One / N generations through `pipeline.step`, fitness materialised on the host. |
| `record_compiles()` / `compiles` | Context manager and fixture that count calls to `COMPILE_ENTRY_POINTS = ("build", "build_cached", "lower", "compile")` on `evoxc`, restoring them in `finally`. |
| `evaluate_population` | `(pipeline, state, pop) -> (P,) numpy`: `algorithm.transform` → replicated `algorithm.forward` → `problem.evaluate` in one graph; accepts a `GenomeTensors` or `support.payloads.Payload` at P=1 or P=pop_size. |
| `transform_of` | Host `Transformed` for a payload, so `edge_overflow` is visible. |
| `fitness_history`, `_read_log` | Per-generation `(generation, max)` pairs from `pipeline.fitness_history` or `<save_dir>/log.txt`; raises when neither exists. |
| `TINY_CONFIG`, `LOG_COLUMNS` | The one configuration and the documented log header `Generation,Max,Min,Mean,Std,Cost Time`. |

`evaluate_population` must replicate the schedule and the payload before calling `algorithm.forward`: func-fit tiles its dataset to `(P*S, I)` and calls `act_fn` once, so `act_fn` derives `reps = batch.shape[0] // P` and rebuilds every `Transformed` field and payload array as `reshape(x, (P,1,…)) → broadcast(…, (P,reps,…)) → reshape((P*reps,…))` (individual-major, row `p*S+s`); `evoxc` offers `broadcast`, not `broadcast_to`.
The recipe is validated today against a hand-written numpy forward pass (max abs diff 2.7e-8 on the initial XOR population), independently of `pipeline.py`.

## What each module proves
| Module | Tests | Proves |
|---|---|---|
| `./test_xor_run.py` | 5 | `auto_run` at `pop_size=100/species_size=10/generation_limit=250/fitness_target=-0.05` returns a best genome that **re-scores** to ≥ `-0.05`; a one-generation `auto_run` returns `(state, best)` with a P=1 batched payload padded to `max_nodes`/`max_conns`; `auto_run`'s reported best equals the maximum of a manually stepped 5-generation run at the same seed; `step` advances `generation` 0→1 and its in-graph fitness equals `evaluate_population` of the population it consumed; the population maximum never drops across 10 generations (elitism). |
| `./test_determinism.py` | 5 | Same seed ⇒ bit-identical trajectory and `state.pop` payloads; a different seed differs; disturbing the global `numpy.random` state between two runs changes nothing; a run neither reseeds nor consumes the global stream; `state.randkey` advances and is a function of the seed. Eight generations per comparison, exact equality (NaN-padded attrs compared with `equal_nan=True`, no tolerance). |
| `./test_shape_rebuild.py` | 4 | `pop_size` 8/16 each evolve at their own width, and a fresh 8-wide pipeline still works after the 16-wide one; widening `max_nodes` 8→16 with `max_conns` 16→32 and `max_in_degree` 4→8 produces the wider padded payload with zero `edge_overflow`; a second step of the same signature adds no compile call while a pipeline of a different width does; an extra generation on an already-run pipeline keeps the population scorable at its width. |
| `./test_checkpoint.py` | 3 | `is_save=True` with `save_dir=tmp_path` writes a JSON `config.txt`, a `log.txt` with the documented header and one line per generation, `genomes/<generation>.npz` and `genomes/best_genome.npz`; the log's `Generation`/`Max` columns agree with `fitness_history`; the npz's four split arrays plus its `fitness` scalar round-trip through `numpy.load` into a P=1 payload that re-scores to the recorded and logged value. |
| `./test_show_best.py` | 4 | The documented reporting tail `state, best = auto_run(state)` → `show(state, best)`: a one-generation run's best prints the XOR report (`input:/target:/predict:` for all four samples plus one finite `loss:` line); the best is the batch-first `(1, max_nodes)`/`(1, max_conns, key_width)` payload and `show` accepts the bare per-individual rows a checkpoint stores as the same individual (character-identical reports); `show(state, None)` raises `common.TensorNEATError` naming both accepted spellings and `auto_run`/`best_genome` (never a bare `TypeError`) and a bare-row payload still reports all four samples; `show_problem_details=True` reports `Generation: 1,` and prints no `<label> failed (...)` line, so the `_constant_act_fn` diagnostics path runs to completion at the population's own `P`. |

## Requirements this suite imposes on `pipeline.py` (cross-node)
`Pipeline.step` must return a tuple with the new `State` first and the `(pop_size,)` fitness last; `step_once` rejects anything else with a message quoting that contract.
`Pipeline.auto_run` must return `(state, best_genome)` with the best a P=1 batched `GenomeTensors` padded to the static signature, and must record the run's best fitness so the checkpoint and the re-score agree.
`Pipeline` must expose `pop_size`, and must keep `State` keys `pop`, `generation`, `randkey` (`NEAT.setup`/`tell` own them).
`Pipeline.show` must accept `auto_run`'s batch-first `best_genome` unchanged and the same genome as bare per-individual rows, print the problem's report and return `None`; a `None` (missing) payload or a payload holding `P > 1` individuals must raise `common.TensorNEATError` naming the accepted spellings, never a bare `TypeError` or a population-axis message; `show_problem_details=True` must not abort a run when a diagnostics hook raises (it reports the failure on stdout instead), so `_constant_act_fn` takes its payload as given (the population, for diagnostics) rather than through the one-individual funnel.
The compile site must reach the compiler through `evoxc.build`, `build_cached`, `lower` or `compile`; otherwise `test_a_second_generation_of_the_same_shape_does_not_compile_again` skips, naming exactly that requirement.
`is_save` must keep the legacy layout (`config.txt` as `json.dumps(show_config(), indent=4)`, the `LOG_COLUMNS` header, one appended line per generation, per-generation and best-genome npz files), and `best_genome.npz` must store `node_keys`/`node_attrs`/`conn_keys`/`conn_attrs` plus a scalar `fitness`, because the payload has no bitcast and cannot be packed into a mixed-dtype `nodes`/`conns` pair.
`pipeline.fitness_history` is optional: `fitness_history()` falls back to `<save_dir>/log.txt`, which `is_save=True` always writes.

## Budget
Measured on this machine: `pytest -m slow` is **24 passed, 0 skipped in ~20 s** — the 23 tests here plus the release gate's XOR run (no parity test carries the `slow` marker, so the `--extra bench` oracle is never pulled into this selection).
`test_xor_run::test_auto_run_reaches_the_xor_fitness_target` dominates (`pop_size=100`, up to 250 generations, one compile per shape); every other module runs 1–10 generations at `pop_size≤16`; the three evolving cases of `./test_show_best.py` spend one generation each, and its missing-payload case runs no generation at all (only `setup()`).
The convergence budget is calibrated from the legacy curve in `benchmarks/RESULTS.md` (`pop_size=100` reached `mse ≤ 0.1` at generation 77 and `mse ≈ 0.0099` by generation 150); the evoxc rewrite reaches its target inside the configured budget on this machine.

## Known issues / uncertainty
The convergence budget is the legacy-calibrated one; the evoxc run reaches the target inside it, so it stays headroom rather than a measured floor.
`test_step_advances_one_generation_and_reports_one_fitness_per_individual` assumes the fitness returned by `step` belongs to the population the step consumed (`state_before.pop`); a pipeline that returned post-`tell` fitness instead would fail it loudly.
`test_the_best_fitness_is_non_decreasing_across_generations` rests on NEAT's per-species elitism preserving the population maximum; if species spawn allocation can drop the elite, this is the assertion to revisit (it is not weakened).
`state.generation` is asserted to reach exactly `generation_limit` on fixed-length runs and `1` after a single step; a different counter base would surface there.
Nothing in this directory can observe the pipeline's per-shape cache object itself — only the compile counts — and the off-shape "must raise" contract is not testable from outside a cached executable, so it is not asserted here.

## Conventions
Every test has a real body and none skips; the structural probe stays armed but no longer fires, and the compile-count test finds a real entry point.
Assertions are never weakened to get green: a genuine `src/` defect is reported (or pinned with `xfail(strict=True)` and a one-line reason), never hidden behind a tolerance or a skip.
Shapes stay small (`pop_size` 8/16, `max_nodes ≤ 16`), checkpoint tests always pass `save_dir=tmp_path`, and `log.txt`'s `Cost Time` column is parsed but never asserted on.
Files stay under ~1000 lines; one test module per documented behaviour area, module basenames unique across the tree.

## Constraints
`../src/tensorneat`, `../../benchmarks` and `../../pyproject.toml` are read-only inputs: a defect or contract gap found from here goes into the report to the parent, never into this directory's diff.
`pytest`'s prepend import mode names a test module by bare basename, so `./__init__.py` namespaces this directory as `integration.*`; do not delete it.
No new dependency, no `import jax`, no network, no GPU, `backend="numpy"` everywhere.

## Routing Table
| Path | Purpose |
|---|---|
| `./conftest.py` | `TINY_CONFIG`, the structural probe, the step/compile/re-score helpers, the lazy imports. |
| `./test_xor_run.py` | End-to-end XOR: re-scored convergence, the `auto_run` return pair, in-graph vs host fitness, elitism. |
| `./test_determinism.py` | Reproducibility from `(seed, config, shapes)`; global numpy RNG neither read nor written; `randkey` advance. |
| `./test_shape_rebuild.py` | Static-shape contract: per-width runs, wider genome signature, compile-once-then-reuse, rerun consistency. |
| `./test_checkpoint.py` | `is_save` artefacts: `config.txt`, `log.txt`, `genomes/*.npz`, and the npz round trip. |
| `./test_show_best.py` | The documented reporting tail: `auto_run`'s best is directly showable, batch-first and bare rows describe the same individual, a missing payload is a `TensorNEATError`, `show_problem_details` never aborts a run. |
| `../support` | `runner` (only `evoxc` entry point for test-built graphs), `payloads`, `lazy.require`. |
| `../unit` | Kernel-level correctness suite (the oracle this directory builds on, never re-tests). |
| `../../src/tensorneat` | Code under test — read-only; `pipeline.py` owns the compile site, the cache and the checkpoint layout. |
