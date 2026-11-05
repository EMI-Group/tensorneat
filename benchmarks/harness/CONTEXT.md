# harness — protocol, adapters, worker, report
## Intent
The machinery behind `../run_benchmark.py`: one schema for every measurement, one adapter per implementation, one worker process per run, one renderer for the committed reports (`../RESULTS.md`, `../THREADS.md`, `../iree.md`).
This package is import-safe for both sides: nothing here imports `jax`, `tensorneat` or `evoxc` at module scope — the implementation-specific imports live inside `adapters/old.py` and `adapters/new.py`, which the worker loads *after* it has put the right tree on `sys.path`.
## Module Map
- `protocol.py` — schema constants, `RunResult`, timing/quality summarisers, repeat aggregation, verdict rules (including the quality tolerance `max(10 % × |old|, 1e-3)` and its absolute floor), environment reporting (host/CPU/RAM probes, git SHA + code-scope dirty flag, thread env, load average), RSS sampling. Stdlib + `numpy` only.
- `worker.py` — script entry point (not a module entry): parses run arguments, sets `sys.path` + thread env, imports the adapter, samples the load average at start and end, times one run, writes one `RunResult` JSON.
- `adapters/__init__.py` — `get_adapter(impl)`; the adapter contract (below).
- `adapters/old.py` — `tensorneat_legacy` + JAX; one-off cost is `jax.jit(pipeline.step).lower(state).compile()`.
- `adapters/new.py` — `src/tensorneat` + evoxc; one-off cost is the first `Pipeline.step` call (the build), and it asserts no `jax`/`torch` module was pulled in.
- `reporter.py` — assembles `RESULTS.md` from aggregated records + run metadata (title and optional note, environment incl. load and the code-scope dirty state, protocol, case matrix, per-case tables, verdict, per-repeat quality curves, reference baseline, fairness) and decides which companion files (`BACKENDS.md`/`THREADS.md`) are next to the report, dropping the one that names the report itself.
- `report_format.py` — shared null-safe cell/table formatters and document accessors (`cell`, `table`, `num`, `gens_to_target` with round-half-up for a median generation count, `reach` for the `(reached/total)` annotation of an aggregated generation count, repo-relative paths and the `scrub_paths` normaliser for free text, case order, companion-file probe); presentation only.
- `report_sections.py` — the data-derived `## Headline: old vs new` table (median speed-up, per-repeat range, quality overlap), the `## Why the new side wins and loses` narrative incl. the generated memory paragraph, and the `## How to reproduce` block, all written from the measured numbers rather than assuming a winner; its `has_old_side(doc)` decides the "the old side always runs on JAX" phrasing, and its contract note quotes the old side's XOR medians read from `../results/authoritative.json` and `../results/threads1.json`, so "the declared 51/109/157 ms/gen baseline is not this box's reading" is a measurement rather than an assertion.
## Report Document
The CLI writes one JSON document to `--json` and hands the same dict to the reporter: `{schema, generated_at, meta, runs, aggregated, verdict}`.
`meta` carries `env`, `cases`, `impls`, `backends`, `protocol` (generations/warmup/repeats/seed/threads/timeout), `command`, `case_matrix` (each case's `as_dict()`) and the paths; a `--render-from-json` re-render keeps the stored measurement fields, refreshes `json_path`/`out_path` and adds `rendered_at` (the report banner then says it was re-rendered without a measurement).
`meta.env` / each run's `env` is the environment fingerprint: Python/platform/machine, `cpu_model` (`/proc/cpuinfo`), `cpu_count`, `memory_mb`, metadata-only `numpy`/`jax`/`evoxc`/`tensorneat` versions, `git_commit` + `git_dirty` + `git_dirty_paths` of this checkout, the `--threads` value with the observed `thread_env` values (`None` = unset) for `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` + `XLA_FLAGS`, and the document-build `loadavg` triple.
`meta.report_title`/`meta.report_note` (optional, `null` when the CLI flags were not given) make a companion document self-describing: `reporter` renders the title as the H1 (default `reporter.TITLE`) and the note as one italic line directly under the banner; both absent = the default `RESULTS.md` banner.
`--render-from-json … --report-note TEXT --persist-note` writes that note into the record's `meta.report_note` before rendering, so the note survives into every later bare re-render; it is metadata-only (the runs are re-read and compared, never rewritten) and refuses to run without both flags.
Every committed record now stores its own note (`results/authoritative.json`, `results/threads1.json`, `results/iree.json`, `results/xor3d_quality_2026-11-04.json`), so a bare `--render-from-json` reproduces each committed report's banner without a flag.
`reporter._banner` writes the provenance blockquote as whole sentences — what produced the file and from which record and timestamp, then the optional `--render-from-json` notice, then the do-not-edit rule — so a re-rendered file never splits "It must not be hand-edited: rerun the CLI instead…" across the inserted line.
`git_dirty` is scoped to the **code under test** (`protocol.GIT_DIRTY_PATHS` = `src`, `benchmarks/harness`, `benchmarks/legacy_jax`, `benchmarks/cases.py`, `benchmarks/run_benchmark.py`, `benchmarks/backend_measure.py`, `test`, `pyproject.toml`, `uv.lock`, `.python-version`, minus `protocol.GIT_EVIDENCE_PATHS`): the harness writes `RESULTS.md`, `THREADS.md`, `iree.md` and `results/*.json` while it samples this fingerprint, so a full-tree status would always report a clean measurement as dirty.
The probe reads `git status --porcelain -z` (`_git(..., strip=False)`, so the leading space of `" M path"` survives) and parses each NUL-separated record as git's two-character `XY` status field, one space, then the unquoted path (`_porcelain_paths`), taking a rename's destination record and consuming its source field. `git_dirty_paths` names the offending repo-relative paths (the report prints the first `protocol.GIT_DIRTY_PATH_LIMIT` and counts the rest); a clean tree renders as `(clean — generated report artifacts are not counted)`.
Each run also carries optional `loadavg_start`/`loadavg_end` (1/5/15-minute averages sampled by that worker): measured conditions, never verdict inputs, and their absence in older JSON is rendered as `n/a (not recorded)`.
The fingerprint never imports an implementation: versions are package metadata and the host probes degrade to `None` (rendered `n/a`) when `/proc` or git is unavailable.
The report opens with the headline old-vs-new table (one row per case/backend: ms/gen both sides, median speed-up, per-repeat speed-up range, quality overlap, compile seconds, peak RSS, final best, gens-to-target, timing/quality verdict), then environment (with the load-average row and the shared-box caveat), protocol, case matrix, per-case tables, old-vs-new, verdict, the data-derived win/lose narrative with the memory paragraph, per-repeat quality curves (one row per repeat plus the element-wise-median aggregate), declared baseline, non-comparisons (accelerator pointer when `BACKENDS.md` is next to the report) and the reproduce block (thread-bound pointer when `THREADS.md` is, never when the report *is* `THREADS.md`).
## Generations-to-target rendering contract
A case's `quality` block carries `target`, `best_fitness`, `mean_fitness`, `valid_count`, `final_best_fitness` and `generations_to_target` (`null` when the target was not reached, and also when the case was measured with no target at all — `protocol.quality_curve(target=None)`).
A case may declare `gens_to_target=False` (e.g. `../cases.py` `cartpole_p100`, whose horizon-capped return the initial population already maxes out); `worker.py` then measures with `target=None`, so `quality.target` is `null` in the record and the report renders `n/a (no informative target)` instead of a reach claim nobody should read.
`report_format.reach(value, all_values, length, target=…)` is the single renderer for an aggregated generation count and distinguishes "key absent" from "recorded as None" with the module sentinel `report_format.TARGET_UNKNOWN`:
- every repeat reached → the rounded median, e.g. `78`;
- part of the repeats reached (`k` of `n`) → `78 (2/3)`, the median of the repeats that reached;
- none reached → `>152 (0/3)`;
- `target` recorded as `null` → `n/a (no informative target)`;
- a legacy record with no `gens_to_target_all`/`target` key → the pre-existing plain form (`78` / `>152`).
The headline `Gens to target (old)`/`(new)` columns, the per-case table, the quality-curve `— aggregate` row and the per-repeat curve rows all go through it, so no table overstates reach.
## Speed-up range
`protocol.speedup_bounds` keeps the field names `best`/`worst`, but `best` is `old_min/new_max` — the best case *for the old side*, i.e. the **pessimistic** bound for the new side, while `worst` is `old_max/new_min`, the optimistic one.
The report prints the pessimistic value first and titles the column `Speed-up range (worst–best)`; the computed values are unchanged (`xor_p100` prints `0.59x–1.37x`).
`reporter.render_report(doc, report_path=None) -> str` and `reporter.write_report(doc, path)` are the only consumers; the reporter is presentation only and reuses `protocol.comparisons` / `doc["verdict"]` instead of recomputing verdicts. `report_path` decides the companion-file probe (explicit path when given, else the recorded absolute `meta.out_path`, else `benchmarks/` of this checkout) and `reporter._companions` drops the entry naming the report's own file, so a companion never points at itself.
`protocol.short_error(text)` is the single error summariser: it keeps the exception line of a traceback (not `Traceback (most recent call last):`) and is used by `aggregate`, `verdict`, the CLI progress line and the reporter.
Paths in the report are repo-relative (`report_format.display_path`): `meta.json_path`/`meta.out_path` and the interpreter recorded in `meta.command` drop the worktree prefix when they sit inside the checkout and fall back to the basename otherwise, so a committed `RESULTS.md` never names the machine that produced it.
Free text — a recorded traceback or backend error, which is stored verbatim in the record — is normalised at render time by `report_format.scrub_paths` (applied to the whole document at the end of `render_report`): an absolute path that reaches a checkout directory (`src/`, `benchmarks/`, `test/`) is rewritten to its repo-relative form, so `<checkout>/src/tensorneat/…/species.py:295` renders as `src/tensorneat/algorithm/neat/species.py:295`, while a path with no such component (a virtualenv's `site-packages`, `/proc/...`) is left alone.
The records themselves are evidence: their `runs`/`aggregated`/`verdict` are never rewritten for presentation, and `--persist-note` writes nothing but `meta.report_note`.
The `## How to reproduce` block therefore starts with the portable `python benchmarks/run_benchmark.py <flags...>` and states the interpreter that actually ran on its own line; `meta.command` is written as `shlex.join([sys.executable, *sys.argv])` and rendered verbatim (so `shlex.split` of the line returns that argv), while a legacy record stored as unquoted `" ".join(argv)` has its `--report-title`/`--report-note` span re-quoted from `meta`, and a render override that replaced that stored value falls back to the structural span (flag → next `--` token) so the line stays the run's own argv rather than the override; its closing sentence names the record the report was rendered from (`reporter._record_sentence`): a record that `results/.gitignore` re-includes is committed evidence and is called traceable, a scratch record is labelled as not committed.
The case matrix is one table (family, problem, pop, timed generations, target, and `yes`/`not run`/`n/a` per implementation); the verdict section drops protocol's per-case "no verdict" reasons and keeps the error rows, the `Sub-verdicts` line and the closing `N run(s) not ok` line.
The worker redirects stdout/stderr for `prepare`, the warmup and the timed loop.
The worker sets `untimed = max(0, warmup - generations_consumed_before_timing)` and records it as `protocol.untimed_warmup_generations`, so the fitness curve holds `untimed + timed` entries: one fewer on the new side, whose build generation is run but excluded — a target reached at absolute generation `g` is reported as index `g - 1` there.
Both sides still evolve the same total `warmup + timed` generations when `warmup >= 1`; with `--warmup 0` the new side's build generation cannot be subtracted and it evolves one generation more than the old side for the same `--generations`.
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
`prepare` is where the two compile models meet: the old side compiles in `prepare` and does not advance the state (0 generations consumed); the new side performs its build by running one real generation and reports 1 consumed generation, which the worker subtracts from the warmup so both sides' timing samples cover the same generation indices (the new side's quality curve starts one generation later and is one entry shorter).
`step_fn` must return fitness already materialised on the host, so the sync cost sits inside the timed region.
## Fitness Extraction Contract (pipeline <-> harness)
The worker reads fitness from the **last** element of the step result: `(state, pop, fitness)` and `(state, fitness)` are both accepted, anything else fails loudly with the observed type/shape.
`step` must not print; the harness redirects stdout/stderr to `os.devnull` for the whole warmup+timed loop anyway.
## RunResult JSON (schema `tensorneat-bench/1`)
```
{schema, case, impl, backend, repeat, seed, status: ok|skipped|error, error|null,
 case_config: {family, algorithm, genome, problem, pop_size, generation_limit, fitness_target, gens_to_target, notes},
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
Aggregated records (`aggregate()` output) carry the same identity keys plus `n_repeats`, `seeds`, medians over repeats and `{metric}_min`/`{metric}_max` spreads, `gens_to_target_all` (per-repeat, seed-sorted) and its summary `target`, `gens_to_target_reached`, `gens_to_target_repeats`.
## Worker CLI
`worker.py --case NAME --impl {old,new} --backend NAME --repeat R --seed S --generations N --warmup N --threads N --out PATH`.
It writes JSON to `--out` even on failure (`status="error"`, traceback in `error`) and exits non-zero; the parent CLI treats a missing file as an error too.
## Known Issues
- The reach contract and `--persist-note` are exercised only by hand (renders of the committed records, a `$TMPDIR` copy round-trip); `test/unit/benchmarks/test_protocol.py` covers just the git fingerprint, so `aggregate()`'s new keys, the six `reach` forms and the note round-trip have no regression test.
- `Interpreter used:` is host-dependent: `display_path` yields `.venv/bin/python` only when the recording machine's checkout still exists on the rendering host and otherwise the basename `python`, so a re-render is not byte-identical across boxes.
- A report rendered outside this checkout (`$TMPDIR`) loses the `BACKENDS.md`/`THREADS.md` companion pointers (an `os.path.exists` probe next to the report) and prints the bare output filename in its footer, so a temp-path render is never byte-identical to the committed file even when every cell is.
- Report text is not a pure function of the record: `report_sections._contract_baseline_text` reads `../results/authoritative.json` + `../results/threads1.json`, `_record_sentence` reads `../results/.gitignore`, `report_format.companion_files` probes the report directory, and `reporter.PARITY_TEST_COUNT` is a hand-maintained `53` (the `meta.parity_tests` override it documents is written by no code path) — a record rendered outside the checkout is labelled a scratch record and loses the measured-baseline sentence.
- A partly failed group is rendered as `error` in the headline and per-case tables while `_curves_section` still prints the ok repeats' curves; `aggregate`'s `n_errors` and `meta.impls` are written as record data and read by nothing.
- `_json_safe` maps +/-inf to `null`, so an all-invalid generation reads `n/a` on every per-repeat curve row while `median_curve` can still print `-inf` on the aggregate row.
- Harness modules import each other as bare `harness.*`: `import benchmarks.harness.reporter` fails with `ModuleNotFoundError: No module named 'harness'` (only `benchmarks.harness.protocol`, the spelling the tests use, is importable that way), and importing both spellings yields two distinct `protocol` modules.
- `OldAdapter.display`/`adapters/__init__.py`'s `Adapter` protocol still declare `display` although rows are labelled from `report_format.IMPL_LABELS`; `worker._split_step` accepts only tuples of length >= 2 and trusts the last element as fitness.
- A synthesised `error`/`skipped` row (a timeout, an `OSError` from `subprocess.run`, a worker that wrote no record) omits the `schema`/`env`/`protocol`/`loadavg_*` fields a worker record carries; an `OSError` is caught, so the matrix continues instead of aborting before the JSON and report are written.
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
