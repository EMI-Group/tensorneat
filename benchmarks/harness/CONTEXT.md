# harness — protocol, adapters, worker, report
## Intent
The machinery behind `../run_benchmark.py`: one schema for every measurement, one adapter per implementation, one worker process per run, one renderer for `../RESULTS.md`.
This package is import-safe for both sides: nothing here imports `jax`, `tensorneat` or `evoxc` at module scope — the implementation-specific imports live inside `adapters/old.py` and `adapters/new.py`, which the worker loads *after* it has put the right tree on `sys.path`.
## Module Map
- `protocol.py` — schema constants, `RunResult`, timing/quality summarisers, repeat aggregation, verdict rules, environment reporting, RSS sampling. Stdlib + `numpy` only.
- `worker.py` — script entry point (not a module entry): parses run arguments, sets `sys.path` + thread env, imports the adapter, times one run, writes one `RunResult` JSON.
- `adapters/__init__.py` — `get_adapter(impl)`; the adapter contract (below).
- `adapters/old.py` — `tensorneat_legacy` + JAX; one-off cost is `jax.jit(pipeline.step).lower(state).compile()`.
- `adapters/new.py` — `src/tensorneat` + evoxc; one-off cost is the first `Pipeline.step` call (the build), and it asserts no `jax`/`torch` module was pulled in.
- `reporter.py` — aggregated records + run metadata -> `RESULTS.md` markdown (tables, verdict, fairness, reproduce block).
## Report Document
The CLI writes one JSON document to `--json` and hands the same dict to the reporter: `{schema, generated_at, meta, runs, aggregated, verdict}`.
`meta` carries `env`, `cases`, `impls`, `backends`, `protocol` (generations/warmup/repeats/seed/threads/timeout), `command`, `case_matrix` (each case's `as_dict()`) and the paths.
`reporter.render_report(doc) -> str` and `reporter.write_report(doc, path)` are the only consumers; the reporter is presentation only and reuses `protocol.comparisons` / `doc["verdict"]` instead of recomputing verdicts.
`protocol.short_error(text)` is the single error summariser: it keeps the exception line of a traceback (not `Traceback (most recent call last):`) and is used by `aggregate`, `verdict`, the CLI progress line and the reporter.
The worker redirects stdout/stderr for `prepare`, the warmup and the timed loop; the quality curve covers warmup + timed generations, so the new side's build generation (excluded by `generations_consumed_before_timing`) is the only index the two sides do not share.
## Adapter Contract
```python
class Adapter:
    impl: str                    # "old" | "new"
    display: str                 # human row label, e.g. "tensorneat-legacy (JAX)"
    backend_label: str           # backend actually used ("jax" for old, the requested evoxc backend for new)
    def build(self, case, backend: str, seed: int)                 # -> pipeline object
    def setup(self, pipeline)                                      # -> state
    def prepare(self, pipeline, state)                             # -> (state, compile_seconds, generations_consumed, step_fn)
    # step_fn(state) -> (state, fitness float64 ndarray of shape (P,))
```
`prepare` is where the two compile models meet: the old side compiles in `prepare` and does not advance the state (0 generations consumed); the new side performs its build by running one real generation and reports 1 consumed generation, which the worker subtracts from the warmup so both sides time the same generation index.
`step_fn` must return fitness already materialised on the host, so the sync cost sits inside the timed region.
## Fitness Extraction Contract (pipeline <-> harness)
The worker reads fitness from the **last** element of the step result: `(state, pop, fitness)` and `(state, fitness)` are both accepted, anything else fails loudly with the observed type/shape.
`step` must not print; the harness redirects stdout/stderr to `os.devnull` for the whole warmup+timed loop anyway.
## RunResult JSON (schema `tensorneat-bench/1`)
```
{schema, case, impl, backend, repeat, seed, status: ok|skipped|error, error|null,
 case_config: {family, algorithm, genome, problem, pop_size, generation_limit, fitness_target, notes},
 env: {python, platform, machine, cpu_count, numpy, jax, evoxc, tensorneat},
 protocol: {warmup_generations, timed_generations, threads},
 compile_time_s, generations_consumed_before_timing,
 generation_ms: [...], ms_per_generation, ms_per_generation_mean, ms_per_generation_min,
 ms_per_generation_max, gens_per_second,
 rss_after_import_mb, rss_after_compile_mb, rss_final_mb,
 quality: {target, best_fitness: [...], mean_fitness: [...], valid_count: [...],
           final_best_fitness, generations_to_target|null}}
```
Aggregated records (`aggregate()` output) carry the same identity keys plus `n_repeats`, `seeds`, medians over repeats and `{metric}_min`/`{metric}_max` spreads, and `gens_to_target_all`.
## Worker CLI
`worker.py --case NAME --impl {old,new} --backend NAME --repeat R --seed S --generations N --warmup N --threads N --out PATH`.
It writes JSON to `--out` even on failure (`status="error"`, traceback in `error`) and exits non-zero; the parent CLI treats a missing file as an error too.
## Constraints
- Never import the implementation under test at module scope here; the worker controls `sys.path` before the adapter import.
- `protocol.py` holds the only copy of the schema, the verdict rules and the tolerance constants — the CLI and reporter must not re-implement them.
- Files <= ~1000 lines; `reporter.py` is presentation only (no measurement logic).
## Routing Table
| Path | Purpose |
|---|---|
| `./adapters` | Per-implementation contracts (`old.py`, `new.py`) and the factory. |
| `../cases.py` | Case definitions the adapter builds a pipeline from (parent level, read-only from here). |
| `../CONTEXT.md` | Protocol, fairness rules and verdict semantics (parent level). |
