"""Shared, null-safe markdown formatting and document accessors for ``RESULTS.md``.

Presentation only: no measurement maths and no verdict rule lives here — ``protocol.py``
owns both. Every renderer imports these primitives so a cell can never print ``None``, a
raw newline or a stray pipe, and so paths and case order are computed in one place.

Imports are stdlib plus ``harness.protocol`` (constants and ``short_error``); it never
touches numpy, jax, evoxc or tensorneat, and therefore stays importable on the bare
interpreter that runs ``--list``/``--dry-run``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from harness.protocol import NO_ERROR_TEXT, REPO_ROOT, short_error

__all__ = [
    "COMPANION_FILES",
    "CURVE_HEADERS",
    "HEADINGS",
    "IMPL_LABELS",
    "IMPL_ORDER",
    "by_case",
    "case_order",
    "cell",
    "companion_files",
    "declared_length",
    "display_path",
    "fitness",
    "fitness_at",
    "first_error",
    "gens_to_target",
    "impl_label",
    "json_path",
    "mib",
    "ms",
    "num",
    "number",
    "out_path",
    "pct_index",
    "plain",
    "protocol_meta",
    "recorded_lengths",
    "repo_relative",
    "run_flag",
    "seconds",
    "sort_key",
    "spaced_headings",
    "table",
    "text",
    "timestamp",
    "verdict_cell",
]

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

HEADINGS = ("## ", "### ")
"""The heading levels this report emits; :func:`spaced_headings` keeps their spacing uniform."""

COMPANION_FILES = {"backends": "BACKENDS.md", "threads": "THREADS.md"}
"""Evidence files the report cross-references only when they sit next to it.

A fresh checkout without them must not claim they exist, so the sections render the pointer
only when :func:`companion_files` finds the file in the report's own directory.
"""


# --- Null-safe cell formatters ------------------------------------------------------------


def cell(value, fallback: str = "n/a") -> str:
    """One markdown table cell: never ``None``, never a raw newline or pipe."""
    if value is None:
        return fallback
    text_value = (
        str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()
    )
    return text_value or fallback


def table(headers, rows) -> str:
    """A pipe table; every cell goes through :func:`cell`."""
    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def spaced_headings(markdown: str) -> str:
    """Put exactly one blank line before and after every ``##``/``###`` heading.

    Markdown renders a heading followed directly by a table, but the section blocks are
    assembled by hand, so this single normaliser guarantees the blank lines instead of
    every section having to remember them.
    """
    lines = markdown.split("\n")
    spaced: list[str] = []
    for index, line in enumerate(lines):
        if not line.startswith(HEADINGS):
            spaced.append(line)
            continue
        if spaced and spaced[-1].strip():
            spaced.append("")
        spaced.append(line)
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if following.strip():
            spaced.append("")
    return "\n".join(spaced)


def number(value) -> float | None:
    """Coerce to a float, or ``None`` for missing / non-numeric input (NaN included)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric


def num(value, digits: int = 2, unit: str = "", signed: bool = False) -> str:
    """Fixed-point number, ``+``/``-`` sign when asked, ``-inf``/``inf`` preserved."""
    numeric = number(value)
    if numeric is None:
        return "n/a"
    if math.isinf(numeric):
        return "-inf" if numeric < 0 else "inf"
    return f"{numeric:+.{digits}f}{unit}" if signed else f"{numeric:.{digits}f}{unit}"


def ms(value) -> str:
    """Wall clock per generation, milliseconds with 2 decimals."""
    return num(value, 2)


def seconds(value) -> str:
    """One-off compile/build time, seconds with 3 decimals."""
    return num(value, 3)


def mib(value) -> str:
    """Peak RSS, MiB with 1 decimal."""
    return num(value, 1)


def fitness(value) -> str:
    """Fitness with 4 decimals; ``-inf`` means every individual was invalid."""
    return num(value, 4)


def plain(value) -> str:
    """Compact human number (``100``, ``-0.05``, ``180``) for configuration cells."""
    if value is None or isinstance(value, bool):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    numeric = number(value)
    if numeric is None:
        return "n/a"
    if math.isinf(numeric):
        return "-inf" if numeric < 0 else "inf"
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"


def run_flag(declared, ran, case: str, impl: str) -> str:
    """``yes`` when the implementation ran, ``not run`` when it could but was not asked,
    ``n/a`` when the case cannot run that implementation at all."""
    if impl not in declared:
        return "n/a"
    return "yes" if (case, impl) in ran else "not run"


def verdict_cell(value) -> str:
    """``True`` -> ``PASS``, ``False`` -> ``FAIL``, ``None`` -> ``n/a``."""
    if value is None:
        return "n/a"
    return "PASS" if value else "FAIL"


def text(value, fallback: str = "n/a") -> str:
    """Non-empty string or the fallback."""
    if value is None:
        return fallback
    text_value = str(value).strip()
    return text_value if text_value else fallback


def first_error(record: dict) -> str:
    """The record's error, summarised by :func:`harness.protocol.short_error` (never empty)."""
    errors = record.get("errors")
    candidates = []
    if isinstance(errors, (list, tuple)):
        candidates.extend(errors)
    if record.get("error") is not None:
        candidates.append(record["error"])
    for candidate in candidates:
        summary = short_error(candidate)
        if summary:
            return summary
    return NO_ERROR_TEXT


def impl_label(impl) -> str:
    """``old``/``new`` plus its human label, e.g. ```old` tensorneat-legacy (JAX)``."""
    key = text(impl, "unknown")
    label = IMPL_LABELS.get(key)
    return f"`{key}` {label}" if label else f"`{key}`"


def sort_key(impl, backend) -> tuple[int, str]:
    """Old before new, then backend name — a stable order for every table."""
    return (IMPL_ORDER.get(text(impl, ""), len(IMPL_ORDER)), text(backend, ""))


# --- Document accessors -------------------------------------------------------------------


def meta(doc: dict) -> dict:
    """``doc["meta"]`` or an empty dict."""
    return doc.get("meta") or {}


def protocol_meta(doc: dict) -> dict:
    """``doc["meta"]["protocol"]`` or an empty dict."""
    return meta(doc).get("protocol") or {}


def _repo_root(path: Path) -> Path | None:
    """The checkout root above ``path``: the directory that holds ``benchmarks/``.

    The marker files are the strongest signal, but a path recorded on the machine that ran
    the benchmark already names the root structurally (``<root>/benchmarks/results/x.json``),
    so a ``benchmarks`` path component identifies it even when the tree is not readable.
    Returns ``None`` for a path outside any checkout — which is what makes
    :func:`display_path` fall back to the basename.
    """
    for parent in path.parents:
        if (parent / "benchmarks").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
        if parent.name == "benchmarks":
            return parent.parent
    return None


def repo_relative(value) -> str | None:
    """``value`` relative to the checkout root when it is an absolute path inside it.

    Relative paths are returned normalised, a path outside the checkout as ``None``.
    """
    path = Path(str(value))
    if not path.is_absolute():
        return path.as_posix()
    root = _repo_root(path)
    if root is None:
        return None
    return path.relative_to(root).as_posix()


def display_path(value, fallback: str) -> str:
    """A path as it belongs in the committed report: repo-relative, else just the basename.

    A worktree path such as ``/root/.../benchmarks/results/latest.json`` must never leak
    into ``RESULTS.md``: whoever reads it needs a path relative to the repo they cloned.
    """
    text_value = text(value, fallback)
    relative = repo_relative(text_value)
    return relative if relative is not None else Path(text_value).name


def json_path(doc: dict) -> str:
    """Path of the JSON the report was rendered from."""
    return display_path(meta(doc).get("json_path"), "benchmarks/results/latest.json")


def out_path(doc: dict) -> str:
    """Path of the report itself, rendered with the same relativisation."""
    return display_path(meta(doc).get("out_path"), "benchmarks/RESULTS.md")


def timestamp(doc: dict) -> str:
    """Generation timestamp recorded in the document."""
    return text(doc.get("generated_at"), "unknown")


def case_order(doc: dict, extra_cases=()) -> list[str]:
    """Cases in the declared matrix order, then any case only seen in the runs."""
    order: list[str] = []
    for entry in meta(doc).get("case_matrix") or []:
        name = entry.get("name")
        if name and name not in order:
            order.append(name)
    for name in extra_cases:
        if name and name not in order:
            order.append(name)
    return order


def by_case(records) -> dict[str, list[dict]]:
    """Group records by case, preserving first-seen case order."""
    grouped: dict[str, list[dict]] = {}
    for record in records or []:
        grouped.setdefault(text(record.get("case"), ""), []).append(record)
    return grouped


def _curve_length(curve) -> int | None:
    """Number of recorded generations in one best-fitness curve (``None`` when absent)."""
    return len(curve) if isinstance(curve, list) and curve else None


def recorded_lengths(doc: dict) -> dict[tuple[str, str, str], int]:
    """Length of the first recorded best-fitness curve per ``(case, impl, backend)``."""
    lengths: dict[tuple[str, str, str], int] = {}
    for run in doc.get("runs") or []:
        quality = run.get("quality") or {}
        length = _curve_length(quality.get("best_fitness"))
        if length is None:
            continue
        key = (
            text(run.get("case"), ""),
            text(run.get("impl"), ""),
            text(run.get("backend"), ""),
        )
        lengths.setdefault(key, length)
    return lengths


def declared_length(doc: dict) -> int | None:
    """``warmup + timed`` generations from the protocol, for records without a curve."""
    proto = protocol_meta(doc)
    timed, warmup = proto.get("generations"), proto.get("warmup")
    if not isinstance(timed, int) or isinstance(timed, bool):
        return None
    if isinstance(warmup, int) and not isinstance(warmup, bool):
        return timed + warmup
    return timed


def pct_index(length: int, fraction: float) -> int:
    """Index of generation ``round(fraction * length)`` (1-based generation, clamped)."""
    return max(0, min(length - 1, int(round(fraction * length)) - 1))


def report_dir(doc: dict, report_path=None) -> Path:
    """Directory the report is (or will be) written to, for companion-file probes.

    The explicit ``report_path`` wins — that is where ``write_report`` is about to put the
    file, and it is the only correct place for a ``--render-from-json --out $TMPDIR/x.md``
    smoke. Without it, the recorded absolute ``meta.out_path`` names the directory, and a
    document that records nothing falls back to ``benchmarks/`` of this checkout.
    """
    if report_path:
        return Path(report_path).expanduser().resolve().parent
    recorded = meta(doc).get("out_path")
    if recorded:
        path = Path(str(recorded))
        return path.parent if path.is_absolute() else (REPO_ROOT / path).parent
    return REPO_ROOT / "benchmarks"


def companion_files(doc: dict, report_path=None) -> dict[str, bool]:
    """Which companion evidence files exist next to the report (``backends``/``threads``)."""
    directory = report_dir(doc, report_path)
    return {
        key: os.path.exists(os.path.join(directory, name))
        for key, name in COMPANION_FILES.items()
    }


def fitness_at(curve, index: int) -> str:
    """Fitness at ``index`` of ``curve`` (negative counts from the end, clamped)."""
    if not isinstance(curve, list) or not curve:
        return "n/a"
    position = index + len(curve) if index < 0 else index
    return fitness(curve[max(0, min(len(curve) - 1, position))])


def gens_to_target(value, length) -> str:
    """Reached generation, or ``>N`` when the target was not reached within the N recorded ones.

    The aggregated value is a median over the repeats, so it can be fractional (``77.5``);
    the rendered integer rounds half up (``77.5 -> 78``) instead of truncating. The raw value
    stays untouched in the JSON.
    """
    numeric = number(value)
    if numeric is not None and not math.isinf(numeric):
        return str(int(math.floor(numeric + 0.5)))
    return f">{length}" if isinstance(length, int) and length > 0 else "n/a"
