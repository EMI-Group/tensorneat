"""Presentation only: render ``benchmarks/RESULTS.md`` from one benchmark document.

:func:`render_report` turns the JSON document written by ``../run_benchmark.py`` into
markdown; :func:`write_report` writes it to a path. There is no measurement maths and no
verdict rule here — ``protocol.py`` owns both, this module only formats what it produced.
Imports are stdlib plus ``harness.protocol`` (constants and the old-vs-new pairing); it
never touches numpy, jax, evoxc or tensorneat.
"""

from __future__ import annotations

import math
from pathlib import Path

from harness.protocol import QUALITY_TOLERANCE_REL, STATUS_OK, comparisons

__all__ = ["render_report", "write_report", "IMPL_LABELS"]

TITLE = "# TensorNEAT benchmark — evoxc rewrite vs frozen JAX baseline"

IMPL_LABELS = {"old": "tensorneat-legacy (JAX)", "new": "tensorneat (evoxc)"}
"""Row labels: the implementation that was measured, not just its short id."""

IMPL_ORDER = {"old": 0, "new": 1}
"""Report row order inside one case (baseline first, candidate second)."""

CURVE_HEADERS = [
    "Implementation",
    "Backend",
    "Seed",
    "Best @1",
    "Best @10",
    "Best @25%",
    "Best @50%",
    "Best @100%",
    "Final",
    "Gens to target",
]


# --- Null-safe cell formatters ------------------------------------------------------------


def _cell(value, fallback: str = "n/a") -> str:
    """One markdown table cell: never ``None``, never a raw newline or pipe."""
    if value is None:
        return fallback
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()
    return text or fallback


def _table(headers, rows) -> str:
    """A pipe table; every cell goes through :func:`_cell`."""
    lines = [
        "| " + " | ".join(_cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _number(value) -> float | None:
    """Coerce to a float, or ``None`` for missing / non-numeric input (NaN included)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _num(value, digits: int = 2, unit: str = "", signed: bool = False) -> str:
    """Fixed-point number, ``+``/``-`` sign when asked, ``-inf``/``inf`` preserved."""
    number = _number(value)
    if number is None:
        return "n/a"
    if math.isinf(number):
        return "-inf" if number < 0 else "inf"
    return f"{number:+.{digits}f}{unit}" if signed else f"{number:.{digits}f}{unit}"


def _ms(value) -> str:
    """Wall clock per generation, milliseconds with 2 decimals."""
    return _num(value, 2)


def _seconds(value) -> str:
    """One-off compile/build time, seconds with 3 decimals."""
    return _num(value, 3)


def _mib(value) -> str:
    """Peak RSS, MiB with 1 decimal."""
    return _num(value, 1)


def _fitness(value) -> str:
    """Fitness with 4 decimals; ``-inf`` means every individual was invalid."""
    return _num(value, 4)


def _plain(value) -> str:
    """Compact human number (``100``, ``-0.05``, ``180``) for configuration cells."""
    if value is None or isinstance(value, bool):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    number = _number(value)
    if number is None:
        return "n/a"
    if math.isinf(number):
        return "-inf" if number < 0 else "inf"
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _flag(value) -> str:
    """Boolean -> ``yes``/``no``; anything missing -> ``n/a``."""
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _verdict_cell(value) -> str:
    """``True`` -> ``PASS``, ``False`` -> ``FAIL``, ``None`` -> ``n/a``."""
    if value is None:
        return "n/a"
    return "PASS" if value else "FAIL"


def _text(value, fallback: str = "n/a") -> str:
    """Non-empty string or the fallback."""
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _first_error(record: dict) -> str:
    """First line of an aggregated record's error list (never empty)."""
    errors = record.get("errors")
    candidates = []
    if isinstance(errors, (list, tuple)):
        candidates.extend(str(item) for item in errors if item)
    if record.get("error"):
        candidates.append(str(record["error"]))
    for candidate in candidates:
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        if lines:
            return lines[0][:200]
    return "(no error recorded)"


def _impl_label(impl) -> str:
    """``old``/``new`` plus its human label, e.g. ```old` tensorneat-legacy (JAX)``."""
    key = _text(impl, "unknown")
    label = IMPL_LABELS.get(key)
    return f"`{key}` {label}" if label else f"`{key}`"


def _sort_key(impl, backend) -> tuple[int, str]:
    """Old before new, then backend name — a stable order for every table."""
    return (IMPL_ORDER.get(_text(impl, ""), len(IMPL_ORDER)), _text(backend, ""))


# --- Document accessors -------------------------------------------------------------------


def _meta(doc: dict) -> dict:
    """``doc["meta"]`` or an empty dict."""
    return doc.get("meta") or {}


def _protocol_meta(doc: dict) -> dict:
    """``doc["meta"]["protocol"]`` or an empty dict."""
    return _meta(doc).get("protocol") or {}


def _json_path(doc: dict) -> str:
    """Path of the JSON the report was rendered from."""
    return _text(_meta(doc).get("json_path"), "benchmarks/results/latest.json")


def _timestamp(doc: dict) -> str:
    """Generation timestamp recorded in the document."""
    return _text(doc.get("generated_at"), "unknown")


def _case_order(doc: dict, extra_cases=()) -> list[str]:
    """Cases in the declared matrix order, then any case only seen in the runs."""
    order: list[str] = []
    for entry in _meta(doc).get("case_matrix") or []:
        name = entry.get("name")
        if name and name not in order:
            order.append(name)
    for name in extra_cases:
        if name and name not in order:
            order.append(name)
    return order


def _by_case(records) -> dict[str, list[dict]]:
    """Group records by case, preserving first-seen case order."""
    grouped: dict[str, list[dict]] = {}
    for record in records or []:
        grouped.setdefault(_text(record.get("case"), ""), []).append(record)
    return grouped


# --- Sections -----------------------------------------------------------------------------


def _banner(doc: dict) -> str:
    """The blockquote that marks the file as generated."""
    return "\n".join(
        [
            "> **GENERATED FILE.** Produced by `python benchmarks/run_benchmark.py`, rendered from",
            f"> `{_json_path(doc)}` at {_timestamp(doc)}. It must not be hand-edited:",
            "> rerun the CLI instead, or the numbers below stop matching the recorded evidence.",
        ]
    )


def _platform_cpu(env: dict) -> str:
    """Platform, machine and core count as one cell."""
    bits = [bit for bit in (_text(env.get("platform"), ""), _text(env.get("machine"), "")) if bit]
    cores = env.get("cpu_count")
    if isinstance(cores, int) and cores > 0:
        bits.append(f"{cores} cores")
    return " · ".join(bits) if bits else "n/a"


def _environment_section(doc: dict) -> str:
    """Environment fingerprint table."""
    env = _meta(doc).get("env") or {}
    versions = " / ".join(
        f"{name} {_text(env.get(name))}" for name in ("numpy", "jax", "evoxc", "tensorneat")
    )
    rows = [
        ["Date (UTC)", _timestamp(doc)],
        ["Python", _text(env.get("python"))],
        ["Platform / CPU", _platform_cpu(env)],
        ["numpy / jax / evoxc / tensorneat", versions],
    ]
    return "## Environment\n\n" + _table(["Item", "Value"], rows)


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
        return "`--threads 0` — runtime defaults (as users run it)"
    return (
        f"`--threads {threads}` — exported to OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS and, on the "
        "old side, to `XLA_FLAGS` intra-op parallelism"
    )


def _protocol_section(doc: dict) -> str:
    """Protocol table plus the metric definitions the template states."""
    meta = _meta(doc)
    proto = _protocol_meta(doc)
    backends = ", ".join(f"`{_text(name)}`" for name in (meta.get("backends") or []))
    rows = [
        ["Backend(s) of the new implementation", backends or "n/a"],
        ["Warmup generations (untimed)", _plain(proto.get("warmup"))],
        ["Timed generations", _plain(proto.get("generations"))],
        ["Repeats (seed + r)", _repeats_text(proto.get("repeats"), proto.get("seed"))],
        ["Thread pinning", _threads_text(proto.get("threads"))],
        [
            "Scheduling",
            "one worker process per run, strictly sequential — the two implementations never "
            "share a process",
        ],
    ]
    paragraph = "\n".join(
        [
            "Metrics: `compile_time_s` (one-off build), `ms_per_generation` (median wall clock "
            "per generation",
            "after warmup, including host sync), `gens_per_second`, `peak_rss_mb`, and quality",
            "(`final_best_fitness`, `generations_to_target`, best-fitness curve).",
            "Definitions, fairness rules and verdict rules live in `benchmarks/CONTEXT.md`.",
        ]
    )
    return "## Protocol\n\n" + _table(["Parameter", "Value"], rows) + "\n\n" + paragraph


def _case_matrix_section(doc: dict) -> str:
    """The case matrix, plus family/notes for each case."""
    matrix = _meta(doc).get("case_matrix") or []
    if not matrix:
        return "## Case matrix\n\nno cases recorded"
    rows, notes = [], []
    for entry in matrix:
        impls = [str(impl) for impl in (entry.get("impls") or [])]
        rows.append(
            [
                _text(entry.get("name")),
                _text(entry.get("problem")),
                _plain(entry.get("pop_size")),
                _plain(entry.get("generation_limit")),
                _plain(entry.get("fitness_target")),
                _flag("old" in impls),
                _flag("new" in impls),
            ]
        )
        notes.append(
            [
                _text(entry.get("name")),
                _text(entry.get("family")),
                _cell(entry.get("notes"), "—"),
            ]
        )
    body = _table(
        ["Case", "Problem", "Pop", "Generations", "Fitness target", "Old", "New"], rows
    )
    body += "\n\n" + _table(["Case", "Family", "Notes"], notes)
    return "## Case matrix\n\n" + body


def _result_row(record: dict, timed_generations) -> list:
    """One aggregated record as a table row."""
    targets = record.get("generations_to_target")
    return [
        _impl_label(record.get("impl")),
        f"`{_text(record.get('backend'))}`",
        _seconds(record.get("compile_time_s")),
        _ms(record.get("ms_per_generation")),
        _num(record.get("gens_per_second"), 2),
        _mib(record.get("rss_final_mb")),
        _fitness(record.get("final_best_fitness")),
        _gens_to_target(targets, timed_generations),
        _plain(record.get("n_repeats")),
    ]


def _empty_row(record: dict) -> list:
    """A not-ok aggregated record: identity, status and ``n/a`` metrics (never dropped)."""
    status = _text(record.get("status"), "error")
    return [
        f"{_impl_label(record.get('impl'))} — {status}",
        f"`{_text(record.get('backend'))}`",
        "n/a",
        "n/a",
        "n/a",
        "n/a",
        "n/a",
        "n/a",
        _plain(record.get("n_repeats")),
    ]


def _results_section(doc: dict) -> str:
    """Per-case table of every aggregated ``(impl, backend)`` record."""
    aggregated = doc.get("aggregated") or []
    grouped = _by_case(aggregated)
    timed = _protocol_meta(doc).get("generations")
    parts = ["## Results per case", ""]
    for case in _case_order(doc, list(grouped)):
        records = grouped.get(case)
        parts.append(f"### {case}")
        parts.append("")
        if not records:
            parts.extend(["no runs recorded", ""])
            continue
        rows, notes = [], []
        for record in sorted(
            records, key=lambda item: _sort_key(item.get("impl"), item.get("backend"))
        ):
            status = _text(record.get("status"), "ok")
            if status == STATUS_OK:
                rows.append(_result_row(record, timed))
            else:
                rows.append(_empty_row(record))
                notes.append(
                    f"- `{_text(record.get('impl'))}` / `{_text(record.get('backend'))}`: "
                    f"**{status}** — {_cell(_first_error(record))}"
                )
        parts.append(
            _table(
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
                _text(pair.get("case")),
                f"`{_text(pair.get('backend'))}`",
                _ms(pair.get("old_ms")),
                _ms(pair.get("new_ms")),
                _num(pair.get("speedup"), 2, "x"),
                _num(pair.get("delta_ms"), 2, signed=True),
                _fitness(pair.get("old_final_best")),
                _fitness(pair.get("new_final_best")),
                _num(pair.get("quality_delta"), 4, signed=True),
                _verdict_cell(pair.get("timing_pass")),
                _verdict_cell(pair.get("quality_pass")),
            ]
        )
    note = (
        "Timing PASS means new ms/generation <= old; quality PASS means the new final best "
        f"stays within {QUALITY_TOLERANCE_REL:.0%} of the old final best."
    )
    table = _table(
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
    return "## Old vs new\n\n" + table + "\n\n" + note


def _verdict_section(doc: dict) -> str:
    """Overall verdict, its reasons and the two sub-verdicts."""
    verdict = doc.get("verdict") or {}
    status = _text(verdict.get("status"), "INCOMPLETE")
    reasons = [str(reason) for reason in (verdict.get("reasons") or [])]
    lines = ["## Verdict", "", f"**{status}**", ""]
    if reasons:
        lines.extend(f"- {_cell(reason)}" for reason in reasons)
    else:
        lines.append("- no reason recorded")
    timing = _text(verdict.get("timing"), "n/a")
    quality = _text(verdict.get("quality"), "n/a")
    lines.extend(
        [
            "",
            f"Sub-verdicts: timing `{timing}`, quality `{quality}` — timing compares measured "
            f"ms/generation, quality compares final best fitness at a {QUALITY_TOLERANCE_REL:.0%} "
            "tolerance.",
        ]
    )
    return "\n".join(lines)


def _pct_index(length: int, fraction: float) -> int:
    """Index of generation ``round(fraction * length)`` (1-based generation, clamped)."""
    return max(0, min(length - 1, int(round(fraction * length)) - 1))


def _fitness_at(curve, index: int) -> str:
    """Fitness at ``index`` of ``curve`` (negative counts from the end, clamped)."""
    if not isinstance(curve, list) or not curve:
        return "n/a"
    position = index + len(curve) if index < 0 else index
    return _fitness(curve[max(0, min(len(curve) - 1, position))])


def _gens_to_target(value, length) -> str:
    """Reached generation, or ``>N`` when the target was not reached within N generations."""
    number = _number(value)
    if number is not None and not math.isinf(number):
        return str(int(number))
    return f">{length}" if isinstance(length, int) and length > 0 else "n/a"


def _curves_section(doc: dict) -> str:
    """Best-fitness curve checkpoints from the first ok run of every case/impl/backend."""
    picked: dict[tuple[str, str, str], dict] = {}
    for run in doc.get("runs") or []:
        if _text(run.get("status"), "ok") != STATUS_OK:
            continue
        quality = run.get("quality") or {}
        curve = quality.get("best_fitness")
        if not isinstance(curve, list) or not curve:
            continue
        key = (
            _text(run.get("case"), ""),
            _text(run.get("impl"), ""),
            _text(run.get("backend"), ""),
        )
        picked.setdefault(key, run)

    parts = ["## Quality curves", ""]
    if not picked:
        parts.append("no fitness curves recorded")
        return "\n".join(parts)

    for case in _case_order(doc, [key[0] for key in picked]):
        keys = sorted((key for key in picked if key[0] == case), key=lambda k: _sort_key(k[1], k[2]))
        if not keys:
            continue
        parts.append(f"### {case}")
        parts.append("")
        rows = []
        for key in keys:
            run = picked[key]
            quality = run.get("quality") or {}
            curve = quality.get("best_fitness")
            length = len(curve)
            rows.append(
                [
                    _impl_label(key[1]),
                    f"`{_text(key[2])}`",
                    _plain(run.get("seed")),
                    _fitness_at(curve, 0),
                    _fitness_at(curve, 9),
                    _fitness_at(curve, _pct_index(length, 0.25)),
                    _fitness_at(curve, _pct_index(length, 0.50)),
                    _fitness_at(curve, _pct_index(length, 1.00)),
                    _fitness_at(curve, length - 1),
                    _gens_to_target(quality.get("generations_to_target"), length),
                ]
            )
        parts.append(_table(CURVE_HEADERS, rows))
        parts.append("")
    parts.append(
        "Percent columns are generation `round(fraction × N)` of the N-generation timed run; "
        "`>N` means the case's fitness target was not reached within those N generations. "
        "Each row is the first ok run recorded for that case/implementation/backend."
    )
    return "\n".join(parts)


def _reference_section(doc: dict) -> str:
    """Declared legacy baseline next to the measured old row."""
    measured: dict[str, object] = {}
    for record in doc.get("aggregated") or []:
        if record.get("impl") == "old":
            measured.setdefault(_text(record.get("case"), ""), record.get("ms_per_generation"))

    rows = []
    for entry in _meta(doc).get("case_matrix") or []:
        declared_map = entry.get("reference_ms_per_generation") or {}
        if not declared_map:
            continue
        name = _text(entry.get("name"))
        pop = entry.get("pop_size")
        declared = declared_map.get(str(pop))
        if declared is None and len(declared_map) == 1:
            declared = next(iter(declared_map.values()))
        elif declared is None:
            continue
        old_ms = measured.get(name)
        old_number = _number(old_ms)
        declared_number = _number(declared)
        ratio = (
            old_number / declared_number
            if old_number is not None and declared_number not in (None, 0)
            else None
        )
        rows.append(
            [
                name,
                _plain(pop),
                _ms(declared_number),
                _ms(old_number),
                _num(ratio, 2, "x"),
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
    table = _table(
        ["Case", "Pop", "Declared legacy ms/gen", "Measured old ms/gen", "Ratio (measured/declared)"],
        rows,
    )
    return "## Reference numbers (declared baseline)\n\n" + table + "\n\n" + explanation


def _not_compared_section() -> str:
    """The three explicit non-comparisons."""
    return "\n".join(
        [
            "## What is not compared",
            "",
            "- Different PRNGs (JAX `PRNGKey` vs evoxc splitmix64) mean a shared seed identifies "
            "the same *workload*, not the same search trajectory; numerical fidelity is covered "
            "by `test/parity/`.",
            "- Accelerator numbers are informational.",
            "- CartPole is new-only (the legacy library has no equivalent problem).",
        ]
    )


def _reproduce_section(doc: dict) -> str:
    """Exact command line plus the settings it used."""
    meta = _meta(doc)
    proto = _protocol_meta(doc)
    command = _text(meta.get("command"), "python benchmarks/run_benchmark.py")
    cases = ", ".join(_text(name) for name in (meta.get("cases") or []))
    backends = ", ".join(_text(name) for name in (meta.get("backends") or []))
    threads = proto.get("threads")
    thread_text = "runtime defaults (`--threads 0`)" if threads == 0 else f"`--threads {_plain(threads)}`"
    repeats, seed = proto.get("repeats"), proto.get("seed")
    seed_text = ""
    if isinstance(repeats, int) and isinstance(seed, int) and repeats >= 1:
        seed_text = " (seeds " + ", ".join(str(seed + offset) for offset in range(repeats)) + ")"
    block = "\n".join(
        [
            "```bash",
            command,
            "python benchmarks/run_benchmark.py --list   # print the case matrix",
            "python benchmarks/run_benchmark.py --dry-run  # print the planned worker commands",
            "```",
        ]
    )
    note = (
        f"Cases: {cases or 'all declared cases'}. Timed generations: "
        f"{_plain(proto.get('generations'))}, warmup generations (untimed): "
        f"{_plain(proto.get('warmup'))}, repeats: {_plain(proto.get('repeats'))}{seed_text}, "
        f"threads: {thread_text}. Backend(s) of the new side: {backends or 'n/a'}; the old side "
        "always ran on JAX, once per case."
    )
    return "## How to reproduce\n\n" + block + "\n\n" + note


def _footer(doc: dict) -> str:
    """Closing line: provenance and the do-not-edit reminder."""
    return f"Generated from `{_json_path(doc)}` at {_timestamp(doc)} — do not hand-edit."


# --- Entry points -------------------------------------------------------------------------


def render_report(doc: dict) -> str:
    """Render the whole ``RESULTS.md`` body for one benchmark document.

    Args:
        doc: The JSON document written by ``run_benchmark.py`` (``meta``, ``runs``,
            ``aggregated``, ``verdict``); a partial document renders with ``n/a`` cells.

    Returns:
        Markdown text with a trailing newline.
    """
    document = doc or {}
    blocks = [
        TITLE,
        _banner(document),
        _environment_section(document),
        _protocol_section(document),
        _case_matrix_section(document),
        _results_section(document),
        _comparison_section(document),
        _verdict_section(document),
        _curves_section(document),
        _reference_section(document),
        _not_compared_section(),
        _reproduce_section(document),
        _footer(document),
    ]
    return "\n\n".join(block.rstrip() for block in blocks) + "\n"


def write_report(doc: dict, path) -> None:
    """Write :func:`render_report` to ``path`` (parent dirs created, UTF-8, trailing newline)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(doc), encoding="utf-8")
