"""Presentation only: render ``benchmarks/RESULTS.md`` from one benchmark document.

:func:`render_report` turns the JSON document written by ``../run_benchmark.py`` into
markdown; :func:`write_report` writes it to a path. There is no measurement maths and no
verdict rule here — ``protocol.py`` owns both, this module only formats what it produced.
Null-safe cells and document accessors live in :mod:`harness.report_format`; the
data-derived headline and win/lose tables live in :mod:`harness.report_sections`.
Imports are stdlib plus those modules; it never touches numpy, jax, evoxc or tensorneat.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from harness.protocol import (
    GIT_DIRTY_PATH_LIMIT,
    QUALITY_ABS_TOL,
    QUALITY_TOLERANCE_REL,
    STATUS_OK,
    THREAD_ENV_VARS,
    comparisons,
    loadavg_summary,
    median_curve,
)
from harness.report_format import (
    CURVE_HEADERS,
    IMPL_LABELS,
    by_case,
    case_order,
    cell,
    companion_files,
    declared_length,
    first_error,
    fitness,
    fitness_at,
    gens_to_target,
    impl_label,
    json_path,
    meta,
    mib,
    ms,
    num,
    number,
    out_path,
    pct_index,
    plain,
    protocol_meta,
    recorded_lengths,
    repo_relative,
    run_flag,
    seconds,
    sort_key,
    spaced_headings,
    table,
    text,
    timestamp,
    verdict_cell,
)
from harness.report_sections import headline_section, winlose_section

__all__ = ["render_report", "write_report", "IMPL_LABELS"]

TITLE = "# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline"
"""Default H1; ``--report-title`` (stored as ``meta.report_title``) overrides it."""

DEFAULT_REPORT_NOTE = ""
"""Default note under the banner: none, so an unflagged render is byte-for-byte the old one."""

PARITY_TEST_COUNT = 53
"""Collected parity tests on this box, measured 2026-11-02 with
``pytest test/parity --collect-only -q`` (18 convert + 5 distance + 13 fitness + 12 forward
+ 5 grad). The report reads ``meta.parity_tests`` first so a future CLI can override it."""


# --- Environment --------------------------------------------------------------------------


def _memory_text(env: dict) -> str:
    """Total physical memory in MiB, or ``n/a`` when the probe could not read it."""
    value = number(env.get("memory_mb"))
    return f"{value:.0f} MiB" if value is not None else "n/a"


def _git_state(env: dict) -> str:
    """The dirty state of the *code under test*, naming the offending paths when there are some.

    A clean tree says so explicitly: the harness rewrites its own report and JSON while it
    samples this fingerprint, so "clean" must not be mistaken for "unchecked" — generated
    evidence is outside ``protocol.GIT_DIRTY_PATHS`` and never counts.
    """
    dirty = env.get("git_dirty")
    if not isinstance(dirty, bool):
        return "dirty state n/a"
    if not dirty:
        return "clean — generated report artifacts are not counted"
    paths = [text(path) for path in (env.get("git_dirty_paths") or []) if text(path)]
    if not paths:
        return "dirty working tree"  # JSON written before the paths were recorded
    shown = ", ".join(paths[:GIT_DIRTY_PATH_LIMIT])
    more = len(paths) - GIT_DIRTY_PATH_LIMIT
    return f"dirty: {shown}" + (f" +{more} more" if more > 0 else "")


def _git_text(env: dict) -> str:
    """Commit SHA and dirty state, with the reminder that both sides come from this tree."""
    commit = text(env.get("git_commit"), "")
    if not commit:
        return "n/a (git unavailable)"
    return (
        f"`{commit[:12]}` ({_git_state(env)}) — `benchmarks/legacy_jax` and `src/` both come "
        "from this checkout"
    )


def _thread_env_text(env: dict, threads) -> str:
    """Which thread variables are set (and to what) in the process that wrote the record."""
    recorded = env.get("thread_env")
    if not isinstance(recorded, dict) or not recorded:
        return "n/a (not recorded)"
    parts = []
    for name in THREAD_ENV_VARS:
        value = text(recorded.get(name), "")
        parts.append(f"{name} " + (f"set (`{value}`)" if value else "unset"))
    joined = " · ".join(parts)
    if isinstance(threads, int) and not isinstance(threads, bool) and threads > 0:
        return (
            joined
            + " (CLI process, inherited by every worker before the export) — every worker "
            f"exports the four `*_NUM_THREADS` variables as {threads} and sets `XLA_FLAGS` on "
            "the old side before numpy/jax import"
        )
    return (
        joined
        + " (CLI process, inherited unchanged by every worker — `--threads 0` exports nothing)"
    )


def _loadavg_text(doc: dict) -> str:
    """Load average of the box, as measured by the document build and by the workers.

    Two sources, kept apart on purpose: ``meta.env.loadavg`` is the moment the document was
    assembled, while each worker's optional ``loadavg_start``/``loadavg_end`` bracket its own
    timed run, so the reported range says how busy the box was *while measuring*.
    """
    recorded = (meta(doc).get("env") or {}).get("loadavg")
    parts = ["n/a (not recorded)"]
    if isinstance(recorded, (list, tuple)) and len(recorded) >= 3:
        triple = " / ".join(num(value, 2) for value in recorded[:3])
        parts = [f"{triple} (1/5/15 min, document build)"]
    summary = loadavg_summary(doc.get("runs") or [])
    if summary.get("n_runs"):
        parts.append(
            f"workers {num(summary.get('start_min'), 2)}–{num(summary.get('start_max'), 2)} "
            f"at start, {num(summary.get('end_min'), 2)}–{num(summary.get('end_max'), 2)} at "
            f"end (1-min, {plain(summary['n_runs'])} run(s))"
        )
    return "; ".join(parts)


SHARED_BOX_CAVEAT = (
    "**Shared box.** These numbers were measured on a machine that was not exclusively ours: "
    "other jobs ran on it while the workers timed generations, so the per-repeat spread "
    "(the headline's speed-up range and the per-repeat quality-curve rows) is the honest "
    "uncertainty measure — a median alone would hide it."
)
"""One caveat sentence: load and repeat spread are the measured conditions, not noise to drop."""


def _environment_section(doc: dict) -> str:
    """Environment fingerprint table: host, versions, git provenance, thread settings, load."""
    env = meta(doc).get("env") or {}
    proto = protocol_meta(doc)
    versions = " / ".join(
        f"{name} {text(env.get(name))}" for name in ("numpy", "jax", "evoxc", "tensorneat")
    )
    rows = [
        ["Date (UTC)", timestamp(doc)],
        ["Python", text(env.get("python"))],
        ["Platform", text(env.get("platform"))],
        ["CPU model", text(env.get("cpu_model"))],
        ["CPU cores (`os.cpu_count`)", text(env.get("cpu_count"))],
        ["Memory", _memory_text(env)],
        ["Load average (1/5/15 min)", _loadavg_text(doc)],
        ["numpy / jax / evoxc / tensorneat (installed metadata)", versions],
        ["Git commit", _git_text(env)],
        ["Thread pinning (`--threads`)", _threads_text(proto.get("threads"))],
        ["Thread env (set/unset)", _thread_env_text(env, proto.get("threads"))],
    ]
    return (
        "## Environment\n\n"
        + table(["Item", "Value"], rows)
        + "\n\n"
        + SHARED_BOX_CAVEAT
    )


# --- Protocol -----------------------------------------------------------------------------


def _repeats_text(repeats, seed) -> str:
    """Repeat count plus the concrete seed list (repeat ``r`` runs with ``seed + r``)."""
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        return "n/a"
    if isinstance(seed, int) and not isinstance(seed, bool):
        seeds = ", ".join(str(seed + offset) for offset in range(repeats))
        return f"{repeats} (`seed + r`, seeds {seeds})"
    return str(repeats)


def _threads_text(threads) -> str:
    """Thread pinning as requested on the command line."""
    if not isinstance(threads, int) or isinstance(threads, bool):
        return "n/a"
    if threads == 0:
        return "`--threads 0` — runtime defaults (as users run it), nothing exported"
    return (
        f"`--threads {threads}` — exported to OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS and, on the "
        "old side, to `XLA_FLAGS` intra-op parallelism"
    )


def _protocol_section(doc: dict, companions: dict | None = None) -> str:
    """Protocol table plus the metric definitions the template states."""
    doc_meta = meta(doc)
    proto = protocol_meta(doc)
    backends = ", ".join(f"`{text(name)}`" for name in (doc_meta.get("backends") or []))
    rows = [
        ["Backend(s) of the new implementation", backends or "n/a"],
        ["Warmup generations (untimed)", plain(proto.get("warmup"))],
        ["Timed generations", plain(proto.get("generations"))],
        ["Repeats (seed + r)", _repeats_text(proto.get("repeats"), proto.get("seed"))],
        ["Thread pinning", _threads_text(proto.get("threads"))],
        [
            "New-side build",
            "inside the first `Pipeline.step` call (the evoxc build); the generation it "
            "consumes is recorded as `generations_consumed_before_timing` and excluded from "
            "the timed samples and from the quality curve — the untimed warmup is shortened "
            "by it so both sides time and record the same generation index",
        ],
        [
            "Compile time",
            "reported separately as `compile_time_s` (old: "
            "`jax.jit(pipeline.step).lower(state).compile()`; new: the first "
            "`Pipeline.step` build) and never charged to a generation",
        ],
        [
            "Scheduling",
            "one worker process per run, strictly sequential — the two implementations never "
            "share a process",
        ],
    ]
    lines = [
        "Metrics: `compile_time_s` (one-off build), `ms_per_generation` (median wall clock "
        "per generation",
        "after warmup, including host sync), `gens_per_second`, `peak_rss_mb`, and quality",
        "(`final_best_fitness`, `generations_to_target`, best-fitness curve).",
        "The new side's build generation is excluded from both the timing and the quality "
        "curve, and compile/build seconds are never charged to a generation.",
        "Definitions, fairness rules and verdict rules live in `benchmarks/CONTEXT.md`.",
    ]
    if (companions or {}).get("threads"):
        lines.append(
            "A thread-bound sensitivity run (`--threads 1`, XOR family, same CLI) is recorded "
            "in `benchmarks/THREADS.md` next to this report."
        )
    return (
        "## Protocol\n\n"
        + table(["Parameter", "Value"], rows)
        + "\n\n"
        + "\n".join(lines)
    )


# --- Case matrix --------------------------------------------------------------------------


def _timed_generations(doc: dict, entry: dict) -> str:
    """Timed generation budget of this run, falling back to the case's declared limit."""
    generations = protocol_meta(doc).get("generations")
    if isinstance(generations, int) and not isinstance(generations, bool):
        return plain(generations)
    return plain(entry.get("generation_limit"))


def _case_matrix_section(doc: dict) -> str:
    """The case matrix as one table: workload, budget and which implementations ran."""
    matrix = meta(doc).get("case_matrix") or []
    if not matrix:
        return "## Case matrix\n\nno cases recorded"
    ran = {
        (text(run.get("case"), ""), text(run.get("impl"), ""))
        for run in doc.get("runs") or []
    }
    rows = []
    for entry in matrix:
        name = text(entry.get("name"))
        impls = [str(impl) for impl in (entry.get("impls") or [])]
        rows.append(
            [
                name,
                text(entry.get("family")),
                text(entry.get("problem")),
                plain(entry.get("pop_size")),
                _timed_generations(doc, entry),
                plain(entry.get("fitness_target")),
                run_flag(impls, ran, name, "old"),
                run_flag(impls, ran, name, "new"),
                cell(entry.get("notes"), "—"),
            ]
        )
    body = table(
        [
            "Case",
            "Family",
            "Problem",
            "Pop",
            "Timed generations",
            "Fitness target",
            "Old",
            "New",
            "Notes",
        ],
        rows,
    )
    return "## Case matrix\n\n" + body


# --- Per-case results ---------------------------------------------------------------------


def _result_row(record: dict, recorded_generations) -> list:
    """One aggregated record as a table row.

    ``recorded_generations`` is the budget ``generations_to_target`` is indexed against:
    the length of the recorded best-fitness curve (warmup + timed), never just the timed
    part, so this table and the quality-curves table print the same ``>N`` bound.
    """
    targets = record.get("generations_to_target")
    return [
        impl_label(record.get("impl")),
        f"`{text(record.get('backend'))}`",
        seconds(record.get("compile_time_s")),
        ms(record.get("ms_per_generation")),
        num(record.get("gens_per_second"), 2),
        mib(record.get("rss_final_mb")),
        fitness(record.get("final_best_fitness")),
        gens_to_target(targets, recorded_generations),
        plain(record.get("n_repeats")),
    ]


def _empty_row(record: dict) -> list:
    """A not-ok aggregated record: identity, status and ``n/a`` metrics (never dropped)."""
    status = text(record.get("status"), "error")
    return [
        f"{impl_label(record.get('impl'))} — {status}",
        f"`{text(record.get('backend'))}`",
        "n/a",
        "n/a",
        "n/a",
        "n/a",
        "n/a",
        "n/a",
        plain(record.get("n_repeats")),
    ]


def _results_section(doc: dict) -> str:
    """Per-case table of every aggregated ``(impl, backend)`` record."""
    aggregated = doc.get("aggregated") or []
    grouped = by_case(aggregated)
    lengths = recorded_lengths(doc)
    fallback_length = declared_length(doc)
    parts = ["## Results per case", ""]
    for case in case_order(doc, list(grouped)):
        records = grouped.get(case)
        parts.append(f"### {case}")
        parts.append("")
        if not records:
            parts.extend(["no runs recorded", ""])
            continue
        rows, notes = [], []
        for record in sorted(
            records, key=lambda item: sort_key(item.get("impl"), item.get("backend"))
        ):
            status = text(record.get("status"), "ok")
            if status == STATUS_OK:
                key = (
                    text(record.get("case"), ""),
                    text(record.get("impl"), ""),
                    text(record.get("backend"), ""),
                )
                rows.append(_result_row(record, lengths.get(key) or fallback_length))
            else:
                rows.append(_empty_row(record))
                notes.append(
                    f"- `{text(record.get('impl'))}` / `{text(record.get('backend'))}`: "
                    f"**{status}** — {cell(first_error(record))}"
                )
        parts.append(
            table(
                [
                    "Implementation",
                    "Backend",
                    "Compile (s)",
                    "ms/generation",
                    "generations/s",
                    "peak RSS (MiB)",
                    "Final best",
                    "Gens to target",
                    "Repeats",
                ],
                rows,
            )
        )
        parts.append("")
        parts.extend(notes)
        if notes:
            parts.append("")
    return "\n".join(parts).rstrip()


# --- Old vs new ---------------------------------------------------------------------------


def _comparison_section(doc: dict) -> str:
    """Old-vs-new table with the two sub-verdicts per comparison."""
    verdict = doc.get("verdict") or {}
    pairs = verdict.get("comparisons")
    if pairs is None:
        pairs = comparisons(doc.get("aggregated") or [])
    if not pairs:
        return "## Old vs new\n\nno old-vs-new comparison available."
    rows = []
    for pair in pairs:
        rows.append(
            [
                text(pair.get("case")),
                f"`{text(pair.get('backend'))}`",
                ms(pair.get("old_ms")),
                ms(pair.get("new_ms")),
                num(pair.get("speedup"), 2, "x"),
                num(pair.get("delta_ms"), 2, signed=True),
                fitness(pair.get("old_final_best")),
                fitness(pair.get("new_final_best")),
                num(pair.get("quality_delta"), 4, signed=True),
                verdict_cell(pair.get("timing_pass")),
                verdict_cell(pair.get("quality_pass")),
            ]
        )
    note = (
        "Timing PASS means new ms/generation <= old; quality PASS means the new final best "
        "stays within the active tolerance "
        f"`max({QUALITY_TOLERANCE_REL:.0%} × |old|, {QUALITY_ABS_TOL:g})` of the old final best "
        "(the absolute floor keeps two runs that are both at the optimum from failing on "
        "float noise)."
    )
    body = table(
        [
            "Case",
            "Backend",
            "Old ms/gen",
            "New ms/gen",
            "Speed-up",
            "Delta ms/gen",
            "Old final best",
            "New final best",
            "Quality delta",
            "Timing",
            "Quality",
        ],
        rows,
    )
    return "## Old vs new\n\n" + body + "\n\n" + note


# --- Verdict ------------------------------------------------------------------------------


NO_VERDICT_PREFIXES = ("timing: no verdict for ", "quality: no verdict for ")
"""Per-case "no verdict" reasons that repeat the failing rows already listed above them."""


def _verdict_reasons(verdict: dict) -> list[str]:
    """Verdict reasons without the per-case "no verdict" restatement.

    The error rows, the ``Sub-verdicts`` summary and the closing ``N run(s) not ok`` line
    already say that a comparison could not be judged, so one line per case adds noise.
    """
    return [
        str(reason)
        for reason in (verdict.get("reasons") or [])
        if not str(reason).startswith(NO_VERDICT_PREFIXES)
    ]


def _verdict_section(doc: dict) -> str:
    """Overall verdict, its reasons and the two sub-verdicts."""
    verdict = doc.get("verdict") or {}
    status = text(verdict.get("status"), "INCOMPLETE")
    reasons = _verdict_reasons(verdict)
    lines = ["## Verdict", "", f"**{status}**", ""]
    if reasons:
        lines.extend(f"- {cell(reason)}" for reason in reasons)
    else:
        lines.append("- no reason recorded")
    timing = text(verdict.get("timing"), "n/a")
    quality = text(verdict.get("quality"), "n/a")
    lines.extend(
        [
            "",
            f"Sub-verdicts: timing `{timing}`, quality `{quality}` — timing compares measured "
            "ms/generation, quality passes when `new >= old - "
            f"max({QUALITY_TOLERANCE_REL:.0%} × |old|, {QUALITY_ABS_TOL:g})`.",
        ]
    )
    return "\n".join(lines)


# --- Quality curves -----------------------------------------------------------------------


def _curve_row(label: str, backend: str, seed: str, curve, gens: str) -> list:
    """One quality-curve row: the checkpoints every curve row shares."""
    length = len(curve)
    return [
        label,
        f"`{text(backend)}`",
        seed,
        fitness_at(curve, 0),
        fitness_at(curve, 9),
        fitness_at(curve, pct_index(length, 0.25)),
        fitness_at(curve, pct_index(length, 0.50)),
        fitness_at(curve, pct_index(length, 1.00)),
        fitness_at(curve, length - 1),
        gens,
    ]


def _curves_section(doc: dict) -> str:
    """Best-fitness checkpoints: one row per repeat (seed) plus one aggregated row per group.

    One row per repeat is what makes the sampling visible: two seeds of the same case can land
    on opposite sides of the fitness target, and a single "first ok run" row would hide that.
    The aggregated row is the element-wise median of those repeats (``protocol.median_curve``),
    with the median generations-to-target from the aggregated record.
    """
    grouped_runs: dict[tuple[str, str, str], list[dict]] = {}
    for run in doc.get("runs") or []:
        if text(run.get("status"), "ok") != STATUS_OK:
            continue
        quality = run.get("quality") or {}
        curve = quality.get("best_fitness")
        if not isinstance(curve, list) or not curve:
            continue
        key = (
            text(run.get("case"), ""),
            text(run.get("impl"), ""),
            text(run.get("backend"), ""),
        )
        grouped_runs.setdefault(key, []).append(run)

    parts = ["## Quality curves", ""]
    if not grouped_runs:
        parts.append("no fitness curves recorded")
        return "\n".join(parts)

    aggregated_by_key: dict[tuple[str, str, str], dict] = {}
    for record in doc.get("aggregated") or []:
        if text(record.get("status"), "ok") != STATUS_OK:
            continue
        aggregated_by_key.setdefault(
            (
                text(record.get("case"), ""),
                text(record.get("impl"), ""),
                text(record.get("backend"), ""),
            ),
            record,
        )

    for case in case_order(doc, [key[0] for key in grouped_runs]):
        keys = sorted(
            (key for key in grouped_runs if key[0] == case),
            key=lambda key: sort_key(key[1], key[2]),
        )
        if not keys:
            continue
        parts.append(f"### {case}")
        parts.append("")
        rows: list[list] = []
        for key in keys:
            runs = sorted(
                grouped_runs[key],
                key=lambda run: (
                    number(run.get("seed")) is None,
                    number(run.get("seed")) or 0,
                ),
            )
            curves = []
            for run in runs:
                curve = (run.get("quality") or {}).get("best_fitness")
                curves.append(curve)
                rows.append(
                    _curve_row(
                        impl_label(key[1]),
                        key[2],
                        plain(run.get("seed")),
                        curve,
                        gens_to_target(
                            (run.get("quality") or {}).get("generations_to_target"),
                            len(curve),
                        ),
                    )
                )
            aggregate = median_curve(curves)
            if not aggregate:
                continue
            seeds = ", ".join(plain(run.get("seed")) for run in runs)
            record = aggregated_by_key.get(key) or {}
            rows.append(
                _curve_row(
                    impl_label(key[1]) + " — aggregate",
                    key[2],
                    seeds,
                    aggregate,
                    gens_to_target(record.get("generations_to_target"), len(aggregate)),
                )
            )
        parts.append(table(CURVE_HEADERS, rows))
        parts.append("")
    parts.append(
        "Percent columns are generation `round(fraction × N)` of the N recorded generations "
        "(warmup + timed — the budget `generations_to_target` is indexed against), and `>N` means "
        "the case's fitness target was not reached within those N recorded generations. "
        "Each repeat (seed) has its own row, and the `— aggregate` row is the element-wise "
        "median of those repeats: a bimodal case (one seed at the target, one not) shows up "
        "as two different `Final` values rather than one average."
    )
    return "\n".join(parts)


# --- Declared baseline --------------------------------------------------------------------


def _reference_section(doc: dict) -> str:
    """Declared legacy baseline next to the measured old row."""
    measured: dict[str, object] = {}
    for record in doc.get("aggregated") or []:
        if record.get("impl") == "old":
            measured.setdefault(text(record.get("case"), ""), record.get("ms_per_generation"))

    rows = []
    for entry in meta(doc).get("case_matrix") or []:
        declared_map = entry.get("reference_ms_per_generation") or {}
        if not declared_map:
            continue
        name = text(entry.get("name"))
        pop = entry.get("pop_size")
        declared = declared_map.get(str(pop))
        if declared is None and len(declared_map) == 1:
            declared = next(iter(declared_map.values()))
        elif declared is None:
            continue
        old_ms = measured.get(name)
        old_number = number(old_ms)
        declared_number = number(declared)
        ratio = (
            old_number / declared_number
            if old_number is not None and declared_number not in (None, 0)
            else None
        )
        rows.append(
            [
                name,
                plain(pop),
                ms(declared_number),
                ms(old_number),
                num(ratio, 2, "x"),
            ]
        )

    explanation = "\n".join(
        [
            "The performance contract in `../CONTEXT.md` declares legacy CPU numbers measured on a",
            "1-thread-bound box (51 ms/gen at pop 100, 109 ms/gen at pop 500, 157 ms/gen at pop 1000).",
            "The generator prints those next to the old row measured on this box; the verdict uses the",
            "measured row.",
        ]
    )
    if not rows:
        return (
            "## Reference numbers (declared baseline)\n\n"
            "no case declares a reference baseline\n\n"
            + explanation
        )
    body = table(
        ["Case", "Pop", "Declared legacy ms/gen", "Measured old ms/gen", "Ratio (measured/declared)"],
        rows,
    )
    return "## Reference numbers (declared baseline)\n\n" + body + "\n\n" + explanation


# --- What is not compared -----------------------------------------------------------------


def _parity_count(doc: dict) -> int:
    """Parity tests collected on this box (``meta.parity_tests`` overrides the constant)."""
    recorded = number(meta(doc).get("parity_tests"))
    return int(recorded) if recorded else PARITY_TEST_COUNT


def _not_compared_section(doc: dict, companions: dict | None = None) -> str:
    """The explicit non-comparisons, including the measured parity-test count."""
    count = _parity_count(doc)
    accelerators = [
        "- **No accelerator row exists.** The contract is CPU and the only measured backend is "
        "the numpy reference interpreter; an `iree`/`xla`/`tvm` row would be informational and "
        "is never part of the old-vs-new verdict."
    ]
    if (companions or {}).get("backends"):
        accelerators.append(
            " `benchmarks/BACKENDS.md` records the measured attempt: `iree` cannot lower a "
            "single TensorNEAT case (it dies in `Pipeline.setup` before generation 0), so there "
            "is no accelerator number to report."
        )
    return "\n".join(
        [
            "## What is not compared",
            "",
            "- **Same workload, not the same search.** The sides use different PRNGs (JAX "
            "`PRNGKey` vs evoxc splitmix64), so one seed identifies the same *workload*, not "
            "the same trajectory or a bitwise-equal population. Numerical fidelity is "
            "`test/parity/`'s job: it compares kernels against `benchmarks/legacy_jax`, and "
            f"`pytest test/parity --collect-only -q` collects {count} tests on this box.",
            "".join(accelerators),
            "- **CartPole has no old-side row.** The frozen library's RL problems need "
            "gym/brax/mujoco, so `cartpole_p100` is new-only and marked `n/a` for old.",
            "- **Import time, visualization and code generation are not timed.** Only "
            "`rss_after_import_mb` is recorded for the import phase; `Pipeline.show`, network "
            "drawing and sympy codegen sit outside every timed region.",
            "- **No trajectory equality is expected.** The two sides evolve different "
            "populations from the same workload seed; parity of kernels (not of search) is "
            "what `test/parity/` verifies.",
        ]
    )


# --- How to reproduce ---------------------------------------------------------------------


def _command_tokens(command) -> list[str]:
    """``meta["command"]`` split back into argv tokens (empty for a missing command)."""
    text_command = str(command or "").strip()
    if not text_command:
        return []
    try:
        return shlex.split(text_command)
    except ValueError:
        return text_command.split()


def _reproduce_command(doc: dict) -> tuple[str, str]:
    """``(portable command, interpreter used)`` for the reproduce block.

    The recorded command starts with the absolute interpreter path of the machine that
    produced the run, which must not become the documented way to reproduce it: the
    first command is the portable ``python benchmarks/run_benchmark.py <flags...>`` and
    the interpreter that actually ran is stated on its own line.
    """
    tokens = _command_tokens(meta(doc).get("command"))
    if not tokens:
        return "python benchmarks/run_benchmark.py", "n/a"
    interpreter = repo_relative(tokens[0]) or tokens[0]
    arguments = [repo_relative(token) or token for token in tokens[1:]]
    if not arguments:
        arguments = ["benchmarks/run_benchmark.py"]
    return shlex.join(["python", *arguments]), interpreter


def _protocol_flags(proto: dict) -> str:
    """``--generations/--warmup/--repeats/--threads`` as recorded, for the variant commands."""
    flags = []
    for name in ("generations", "warmup", "repeats", "threads"):
        value = proto.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            flags.extend([f"--{name}", str(value)])
    return " ".join(flags)


def _variant(arguments: str) -> str:
    """One ``python benchmarks/run_benchmark.py ...`` command line."""
    return "python benchmarks/run_benchmark.py " + arguments


def _reproduce_section(doc: dict, companions: dict | None = None) -> str:
    """Portable command line, the interpreter that produced the run, and its settings."""
    doc_meta = meta(doc)
    proto = protocol_meta(doc)
    command, interpreter = _reproduce_command(doc)
    backends = [text(name) for name in (doc_meta.get("backends") or []) if name]
    primary = backends[0] if backends else "numpy"
    flags = _protocol_flags(proto)
    cases = ", ".join(text(name) for name in (doc_meta.get("cases") or []))
    threads = proto.get("threads")
    thread_text = (
        "runtime defaults (`--threads 0`)" if threads == 0 else f"`--threads {plain(threads)}`"
    )
    repeats, seed = proto.get("repeats"), proto.get("seed")
    seed_text = ""
    if isinstance(repeats, int) and isinstance(seed, int) and repeats >= 1:
        seed_text = " (seeds " + ", ".join(str(seed + offset) for offset in range(repeats)) + ")"

    lines = [
        "```bash",
        "# regenerate the committed report: the literal command of this run",
        command,
        "",
        "# full matrix, both implementations on the primary backend",
        _variant(f"--impl both --backend {primary} --cases all {flags}".rstrip()),
        "",
        "# baseline only (frozen JAX)",
        _variant(f"--impl old --cases all {flags}".rstrip()),
        "",
        "# candidate only, one backend at a time",
    ]
    for backend in backends or [primary]:
        lines.append(_variant(f"--impl new --backend {backend} --cases all {flags}".rstrip()))
    if "iree" not in backends:
        lines.append(
            _variant(f"--impl new --backend iree --cases xor_p500 {flags}".rstrip())
            + "   # informational; errors cleanly when the IREE runtime is absent"
        )
    lines.extend(
        [
            "",
            "# inspect the matrix and the planned worker commands (no run, no report)",
            "python benchmarks/run_benchmark.py --list",
            "python benchmarks/run_benchmark.py --dry-run",
            "```",
        ]
    )
    note = (
        f"Cases: {cases or 'all declared cases'}. Timed generations: "
        f"{plain(proto.get('generations'))}, warmup generations (untimed): "
        f"{plain(proto.get('warmup'))}, repeats: {plain(proto.get('repeats'))}{seed_text}, "
        f"threads: {thread_text}. Backend(s) of the new side: {backends and ', '.join(backends) or 'n/a'}; "
        "the old side always runs on JAX, once per case. `--out` defaults to "
        "`benchmarks/RESULTS.md` and `--json` to `benchmarks/results/latest.json` (git-ignored); "
        "the report is regenerated in place from that JSON."
    )
    if (companions or {}).get("threads"):
        note += (
            " A thread-bound sensitivity run of the XOR family (`--threads 1`, same CLI) is "
            "recorded in `benchmarks/THREADS.md` next to this report."
        )
    return (
        "## How to reproduce\n\n"
        + "\n".join(lines)
        + "\n\n"
        + f"Interpreter used: {interpreter}\n\n"
        + note
    )


# --- Entry points -------------------------------------------------------------------------


def _banner(doc: dict) -> str:
    """The blockquote that marks the file as generated (and re-rendered, when it was)."""
    lines = [
        "> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from",
        f"> `{json_path(doc)}` at {timestamp(doc)}. It must not be hand-edited:",
        "> rerun the CLI instead, or the numbers below stop matching the recorded evidence.",
    ]
    rendered = text(meta(doc).get("rendered_at"), "")
    if rendered and rendered != timestamp(doc):
        lines.insert(
            2,
            f"> Re-rendered from that same JSON at {rendered} with `--render-from-json` "
            "(no worker was re-run; the git fingerprint is re-probed).",
        )
    return "\n".join(lines)


def _title(doc: dict) -> str:
    """The H1: ``meta.report_title`` when the run declared one, else :data:`TITLE`.

    A bare title ('Thread sensitivity') gets the ``#`` marker; one that already carries its
    own hashes is kept verbatim, so both spellings render a single H1.
    """
    custom = text(meta(doc).get("report_title"), "").strip()
    if not custom:
        return TITLE
    return custom if custom.startswith("#") else f"# {custom}"


def _report_note(doc: dict) -> str:
    """``meta.report_note`` as one italic line under the banner (``""`` when none was given).

    Companion documents (``THREADS.md``) must not be confused with the authoritative report,
    so the note is rendered from the document rather than hard-coded here; a multi-line
    value is collapsed to the one line the report promises.
    """
    note = " ".join(text(meta(doc).get("report_note"), "").split())
    return f"*{note}*" if note else DEFAULT_REPORT_NOTE


def _footer(doc: dict) -> str:
    """Closing line: provenance and the do-not-edit reminder."""
    return (
        f"Generated from `{json_path(doc)}` into `{out_path(doc)}` at "
        f"{timestamp(doc)} — do not hand-edit."
    )


def render_report(doc: dict, report_path=None) -> str:
    """Render the whole ``RESULTS.md`` body for one benchmark document.

    Args:
        doc: The JSON document written by ``run_benchmark.py`` (``meta``, ``runs``,
            ``aggregated``, ``verdict``); a partial document renders with ``n/a`` cells.
        report_path: Where the report will be written. It decides which companion evidence
            files are ``next to`` the report, so a pointer is only rendered when the file
            really is there (a ``$TMPDIR`` render does not claim a checkout's files).

    Returns:
        Markdown text with a trailing newline. Without ``meta.report_title`` /
        ``meta.report_note`` the output is exactly the default ``RESULTS.md``.
    """
    document = doc or {}
    companions = companion_files(document, report_path)
    blocks = [
        _title(document),
        _banner(document),
        _report_note(document),
        headline_section(document),
        _environment_section(document),
        _protocol_section(document, companions),
        _case_matrix_section(document),
        _results_section(document),
        _comparison_section(document),
        _verdict_section(document),
        winlose_section(document),
        _curves_section(document),
        _reference_section(document),
        _not_compared_section(document, companions),
        _reproduce_section(document, companions),
        _footer(document),
    ]
    return (
        spaced_headings(
            "\n\n".join(block.rstrip() for block in blocks if block.rstrip())
        )
        + "\n"
    )


def write_report(doc: dict, path) -> None:
    """Write :func:`render_report` to ``path`` (parent dirs created, UTF-8, trailing newline)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(doc, target), encoding="utf-8")
