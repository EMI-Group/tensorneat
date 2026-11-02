# harness — protocol, adapters, worker, report
## Intent
The machinery behind `../run_benchmark.py`: one schema for every measurement, one adapter per implementation, one worker process per run, one renderer for `../RESULTS.md`.
This package is import-safe for both sides: nothing here imports `jax`, `tensorneat` or `evoxc` at module scope — the implementation-specific imports live inside `adapters/old.py` and `adapters/new.py`, which the worker loads *after* it has put the right tree on `sys.path`.
## Module Map
- `protocol.py` — schema constants, `RunResult`, timing/quality summarisers, repeat aggregation, verdict rules (including the quality tolerance `max(10 % × |old|, 1e-3)` and its absolute floor), environment reporting (host/CPU/RAM probes, git SHA + code-scope dirty flag, thread env, load average), RSS sampling. Stdlib + `numpy` only.
- `worker.py` — script entry point (not a module entry): parses run arguments, sets `sys.path` + thread env, imports the adapter, samples the load average at start and end, times one run, writes one `RunResult` JSON.
- `adapters/__init__.py` — `get_adapter(impl)`; the adapter contract (below).
- `adapters/old.py` — `tensorneat_legacy` + JAX; one-off cost is `jax.jit(pipeline.step).lower(state).compile()`.
- `adapters/new.py` — `src/tensorneat` + evoxc; one-off cost is the first `Pipeline.step` call (the build), and it asserts no `jax`/`torch` module was pulled in.
- `reporter.py` — assembles `RESULTS.md` from aggregated records + run metadata (title and optional note, environment incl. load and the code-scope dirty state, protocol, case matrix, per-case tables, verdict, per-repeat quality curves, reference baseline, fairness, reproduce block) and decides which companion files (`BACKENDS.md`/`THREADS.md`) are next to the report.
- `report_format.py` — shared null-safe cell/table formatters and document accessors (`_cell`, `_table`, `_num`, `_gens_to_target`, repo-relative paths, case order, companion-file probe); presentation only.
- `report_sections.py` — the data-derived `## Headline: old vs new` table (median speed-up, per-repeat range, quality overlap) and the `## Why the new side wins and loses` narrative incl. the generated memory paragraph, all written from the measured numbers rather than assuming a winner.
## Report Document
The CLI writes one JSON document to `--json` and hands the same dict to the reporter: `{schema, generated_at, meta, runs, aggregated, verdict}`.
`meta` carries `env`, `cases`, `impls`, `backends`, `protocol` (generations/warmup/repeats/seed/threads/timeout), `command`, `case_matrix` (each case's `as_dict()`) and the paths; a `--render-from-json` re-render keeps the stored measurement fields, refreshes `json_path`/`out_path` and adds `rendered_at` (the report banner then says it was re-rendered without a measurement).
`meta.env` / each run's `env` is the environment fingerprint: Python/platform/machine, `cpu_model` (`/proc/cpuinfo`), `cpu_count`, `memory_mb`, metadata-only `numpy`/`jax`/`evoxc`/`tensorneat` versions, `git_commit` + `git_dirty` + `git_dirty_paths` of this checkout, the `--threads` value with the observed `thread_env` values (`None` = unset) for `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` + `XLA_FLAGS`, and the document-build `loadavg` triple.
`meta.report_title`/`meta.report_note` (optional, `null` when the CLI flags were not given) make a companion document self-describing: `reporter` renders the title as the H1 (default `reporter.TITLE`) and the note as one italic line directly under the banner; both absent = today's `RESULTS.md` byte for byte.
`git_dirty` is scoped to the **code under test** (`protocol.GIT_DIRTY_PATHS`, minus `protocol.GIT_EVIDENCE_PATHS`): the harness writes `RESULTS.md`, `THREADS.md` and `results/*.json` while it samples this fingerprint, so a full-tree status would always report a clean measurement as dirty. `git_dirty_paths` names the offending repo-relative paths (the report prints the first `protocol.GIT_DIRTY_PATH_LIMIT` and counts the rest); a clean tree renders as `(clean — generated report artifacts are not counted)`.
Each run also carries optional `loadavg_start`/`loadavg_end` (1/5/15-minute averages sampled by that worker): measured conditions, never verdict inputs, and their absence in older JSON is rendered as `n/a (not recorded)`.
The fingerprint never imports an implementation: versions are package metadata and the host probes degrade to `None` (rendered `n/a`) when `/proc` or git is unavailable.
The report opens with the headline old-vs-new table (one row per case/backend: ms/gen both sides, median speed-up, per-repeat speed-up range, quality overlap, compile seconds, peak RSS, final best, gens-to-target, timing/quality verdict), then environment (with the load-average row and the shared-box caveat), protocol, case matrix, per-case tables, old-vs-new, verdict, the data-derived win/lose narrative with the memory paragraph, per-repeat quality curves (one row per repeat plus the element-wise-median aggregate), declared baseline, non-comparisons (accelerator pointer when `BACKENDS.md` is next to the report) and the reproduce block (thread-bound pointer when `THREADS.md` is).
`reporter.render_report(doc, report_path=None) -> str` and `reporter.write_report(doc, path)` are the only consumers; the reporter is presentation only and reuses `protocol.comparisons` / `doc["verdict"]` instead of recomputing verdicts. `report_path` decides the companion-file probe (explicit path when given, else the recorded absolute `meta.out_path`, else `benchmarks/` of this checkout).
`protocol.short_error(text)` is the single error summariser: it keeps the exception line of a traceback (not `Traceback (most recent call last):`) and is used by `aggregate`, `verdict`, the CLI progress line and the reporter.
Paths in the report are repo-relative (`_display_path`): `meta.json_path`/`meta.out_path` and the interpreter recorded in `meta.command` drop the worktree prefix when they sit inside the checkout and fall back to the basename otherwise, so a committed `RESULTS.md` never names the machine that produced it.
The `## How to reproduce` block therefore starts with the portable `python benchmarks/run_benchmark.py <flags...>` and states the interpreter that actually ran on its own line.
The case matrix is one table (family, problem, pop, timed generations, target, and `yes`/`not run`/`n/a` per implementation); the verdict section drops protocol's per-case "no verdict" reasons and keeps the error rows, the `Sub-verdicts` line and the closing `N run(s) not ok` line.
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
 env: {python, platform, machine, cpu_model, cpu_count, memory_mb,
        numpy, jax, evoxc, tensorneat, git_commit, git_dirty,
        threads, thread_env: {OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, XLA_FLAGS}},
 protocol: {warmup_generations, timed_generations, threads, thread_env,
            untimed_warmup_generations},
 compile_time_s, generations_consumed_before_timing,
 generation_ms: [...], ms_per_generation, ms_per_generation_mean, ms_per_generation_min,
 ms_per_generation_max, gens_per_second,
 rss_after_import_mb, rss_after_compile_mb, rss_final_mb,
 loadavg_start: [1, 5, 15]|null, loadavg_end: [1, 5, 15]|null,
 quality: {target, best_fitness: [...], mean_fitness: [...], valid_count: [...],
           final_best_fitness, generations_to_target|null}}
```
`loadavg_start`/`loadavg_end` are optional keys (absent in JSON written before they existed; absence renders as `n/a`, the schema stays `tensorneat-bench/1`).
Aggregated records (`aggregate()` output) carry the same identity keys plus `n_repeats`, `seeds`, medians over repeats and `{metric}_min`/`{metric}_max` spreads, and `gens_to_target_all`.
## Worker CLI
`worker.py --case NAME --impl {old,new} --backend NAME --repeat R --seed S --generations N --warmup N --threads N --out PATH`.
It writes JSON to `--out` even on failure (`status="error"`, traceback in `error`) and exits non-zero; the parent CLI treats a missing file as an error too.
## Constraints
- Never import the implementation under test at module scope here; the worker controls `sys.path` before the adapter import.
- `protocol.py` holds the only copy of the schema, the verdict rules and the tolerance constants (`QUALITY_TOLERANCE_REL` and the `QUALITY_ABS_TOL` floor) — the CLI and reporter must not re-implement them.
- The quality rule is `new >= old - max(QUALITY_TOLERANCE_REL × |old|, QUALITY_ABS_TOL)`: the absolute floor is what lets two runs that are both at the optimum pass, and every report note / verdict reason renders the active tolerance from `protocol.quality_rule_text()` / the pair's `quality_tolerance` instead of hard-coding it.
- Files <= ~1000 lines; `reporter.py` assembles the document, `report_format.py` owns the null-safe table primitives, the companion-file probe and the data-derived headline/win-lose sections: split per report section when one grows instead of moving measurement code into presentation.
## Routing Table
| Path | Purpose |
|---|---|
| `./adapters` | Per-implementation contracts (`old.py`, `new.py`) and the factory. |
| `../cases.py` | Case definitions the adapter builds a pipeline from (parent level, read-only from here). |
| `../CONTEXT.md` | Protocol, fairness rules and verdict semantics (parent level). |
