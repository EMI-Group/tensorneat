"""Schema, measurement summarisers, repeat aggregation and verdicts for the harness.

This module is the single copy of the ``tensorneat-bench/1`` schema, of the aggregation
maths, of the error summaries and of the PASS/FAIL/INCOMPLETE rules: the CLI, the worker
and the reporter import them from here instead of re-implementing them.
It never imports the implementation under test (jax / tensorneat / evoxc are only looked up
in installed-package metadata), and numpy is optional so ``--list``/``--dry-run`` work on a
bare interpreter.
"""

from __future__ import annotations

import math
import os
import platform
import resource
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from importlib import metadata
from pathlib import Path

try:
    import numpy as np
except ImportError:  # --list/--dry-run must work without numpy
    np = None

# --- Schema and protocol constants (the only copy in the repository) ---------------------

SCHEMA = "tensorneat-bench/1"

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error"

QUALITY_TOLERANCE_REL = 0.10
"""Relative part of the quality tolerance: a new final best may trail ``old`` by 10 % of ``|old|``."""

QUALITY_ABS_TOL = 1e-3
"""Absolute floor of the quality tolerance, in fitness units.

A relative-only tolerance degenerates at the optimum: for ``old = -7.9e-15`` it demands the
new side beat ``-8.7e-16`` and two runs that are both at the optimum fail on float noise.
The floor makes "both at the optimum within 1e-3" a PASS without turning a real gap into one:
the tolerance is still ``max(10 % of |old|, 1e-3)``, so every case whose old fitness is
meaningfully below the optimum keeps the relative rule.
"""

DEFAULT_GENERATIONS = 150
DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 2
DEFAULT_SEED = 42
DEFAULT_THREADS = 0
DEFAULT_BACKEND = "numpy"
DEFAULT_TIMEOUT = 3600
OLD_BACKEND_LABEL = "jax"

NO_ERROR_TEXT = "(no error recorded)"
"""Placeholder for a run that recorded no error text at all (``short_error`` returns ``""``)."""

REPO_ROOT = Path(__file__).resolve().parents[2]
"""The checkout under test: ``benchmarks/harness/protocol.py`` -> repository root."""

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "XLA_FLAGS",
)
"""Every thread variable recorded in the environment fingerprint.

``worker.py`` keeps its own copy because it must not import this module before numpy/jax
read the thread variables at import time; keep the two lists in step.
"""

GIT_TIMEOUT = 5
"""Seconds allowed for one ``git`` probe before the fingerprint degrades to ``n/a``."""

GIT_DIRTY_PATHS: tuple[str, ...] = (
    "src",
    "benchmarks/harness",
    "benchmarks/legacy_jax",
    "benchmarks/cases.py",
    "benchmarks/run_benchmark.py",
    "benchmarks/backend_measure.py",
    "test",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
)
"""The *code under test*: only a change here makes the fingerprint ``git_dirty``.

``benchmarks/harness`` covers ``benchmarks/harness/worker.py`` and the rest of the harness;
``benchmarks/backend_measure.py`` is the standalone protocol behind ``benchmarks/BACKENDS.md``;
``pyproject.toml``, ``uv.lock`` and ``.python-version`` are the dependency and interpreter pins,
so a lock-only change still measures a different environment. Everything else (report text,
generated JSON, spikes) is measurement output, not an input of the measurement, so a harness
that renders ``benchmarks/RESULTS.md`` while it samples its own fingerprint must not report
itself as dirty — that was the false alarm this scope removes.
"""

GIT_EVIDENCE_PATHS: tuple[str, ...] = (
    "benchmarks/RESULTS.md",
    "benchmarks/THREADS.md",
    "benchmarks/iree.md",
    "benchmarks/results",
    "benchmarks/BACKENDS.md",
    "benchmarks/spikes",
)
"""Generated evidence, explicitly outside :data:`GIT_DIRTY_PATHS` even if a scope ever widens."""

GIT_DIRTY_PATH_LIMIT = 5
"""How many offending paths the report names before it says ``+N more``."""

__all__ = [
    "SCHEMA",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "STATUS_ERROR",
    "QUALITY_TOLERANCE_REL",
    "QUALITY_ABS_TOL",
    "DEFAULT_GENERATIONS",
    "DEFAULT_WARMUP",
    "DEFAULT_REPEATS",
    "DEFAULT_SEED",
    "DEFAULT_THREADS",
    "DEFAULT_BACKEND",
    "DEFAULT_TIMEOUT",
    "OLD_BACKEND_LABEL",
    "RunResult",
    "NO_ERROR_TEXT",
    "REPO_ROOT",
    "THREAD_ENV_VARS",
    "GIT_DIRTY_PATHS",
    "GIT_EVIDENCE_PATHS",
    "GIT_DIRTY_PATH_LIMIT",
    "git_report",
    "short_error",
    "rss_mb",
    "env_report",
    "loadavg_summary",
    "summarize_timing",
    "quality_curve",
    "median_curve",
    "aggregate",
    "speedup",
    "speedup_bounds",
    "quality_tolerance",
    "quality_threshold",
    "quality_rule_text",
    "quality_overlap",
    "comparisons",
    "verdict",
]


# --- Small numeric helpers --------------------------------------------------------------


def _median(values) -> float | None:
    """Arithmetic median of the given floats (``None`` when there are none)."""
    vals = sorted(values)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _finite(value) -> float | None:
    """Coerce to a finite float, or ``None`` for missing / non-numeric / non-finite input."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value, unit: str = "") -> str:
    """Render an optional float for a human-readable verdict reason."""
    return "n/a" if value is None else f"{value:.3f}{unit}"


def _require_numpy(func: str) -> None:
    """Fail loudly when a numpy-only helper is called on a bare interpreter."""
    if np is None:
        raise RuntimeError(
            f"{func} requires numpy, which is not installed — "
            "install numpy to summarise fitness curves"
        )


def _json_safe(value):
    """Recursively turn numpy scalars/arrays and non-finite floats into JSON-safe values."""
    if np is not None and isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if np is not None and isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


# --- Error summaries --------------------------------------------------------------------


_TRACEBACK_MARKER = "Traceback (most recent call last)"


def _truncate(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters, ending in ``…`` when anything had to go."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def short_error(text: str | None, limit: int = 200) -> str:
    """One-line, actionable summary of an error string (``""`` when there is none).

    A plain message is returned as its first non-empty line, cut to ``limit`` characters.
    A traceback — anything containing ``"Traceback (most recent call last)"`` or spanning
    several lines — is summarised by its *last* non-empty line, because that is the
    exception line (``ModuleNotFoundError: No module named 'evoxc'``) and not the header.
    The result is always a single line of at most ``limit`` characters (``…`` marks a cut),
    so it can go straight into a verdict reason or a markdown table cell.
    """
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""
    if _TRACEBACK_MARKER in raw or len(lines) > 1:
        return _truncate(" ".join(lines[-1].split()), limit)
    return _truncate(raw, limit)


# --- Environment ------------------------------------------------------------------------


def rss_mb() -> float:
    """Peak resident set size of this process in MiB (``ru_maxrss`` is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load_average() -> list[float] | None:
    """1/5/15-minute load average of this host, or ``None`` where the OS does not report it.

    Load is recorded as measured conditions, never as a verdict input: it is what makes a
    spread between repeats readable (a shared box under other jobs moves the medians).
    """
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _version(package: str) -> str | None:
    """Installed version from package metadata, without importing the package itself."""
    try:
        return metadata.version(package)
    except Exception:
        # Importing jax/evoxc/tensorneat just to read a version would trip the release gate.
        return None


def _cpu_model() -> str | None:
    """CPU model string from ``/proc/cpuinfo`` (``platform.processor()`` as fallback).

    The first ``model name`` (x86), ``Model`` (ARM) or ``Hardware`` line wins; an
    unreadable file or an empty value degrades to ``None`` instead of raising.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key, separator, value = line.partition(":")
                if separator and key.strip() in ("model name", "Model", "Hardware"):
                    text = value.strip()
                    if text:
                        return text
    except OSError:
        pass
    return platform.processor() or None


def _memory_mb() -> float | None:
    """Total physical memory in MiB from ``/proc/meminfo``, else ``sysconf``, else ``None``."""
    try:
        with open("/proc/meminfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(float(line.split()[1]) / 1024.0)
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024.0 * 1024.0))
    except (AttributeError, OSError, ValueError):
        return None


def _git(*arguments: str, root: Path = REPO_ROOT, strip: bool = True) -> str | None:
    """Run one ``git`` command in the checkout; ``None`` when git or the repo is unavailable.

    ``strip`` must be off for ``git status --porcelain -z``: the status field is two
    characters wide, so stripping eats the leading space of ``" M path"`` and shifts the path.
    """
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() if strip else completed.stdout


def _porcelain_path(entry: str) -> str:
    """The path named by one ``git status --porcelain`` record (rename -> its new name).

    Git's format is ``XY PATH``: two status characters, one space, then the path, so the path
    starts at offset 3 (``" M path"``, ``"M  path"``, ``"?? path"``, ``"R  old -> new"``); a
    record whose leading space was stripped has no two-character field to read.
    """
    if len(entry) <= 3:
        return ""
    payload = entry[3:]
    if entry[0] in "RC" and " -> " in payload:  # line form: "R  old -> new"
        payload = payload.split(" -> ", 1)[1]
    return payload[1:-1] if payload[:1] == payload[-1:] == '"' else payload  # line quotes


def _porcelain_paths(output: str) -> list[str]:
    """The paths named by ``git status --porcelain -z`` output (NUL fields, unquoted paths).

    A rename/copy carries its source in its own following field, never parsed as a record.
    """
    fields = iter(output.split("\0"))
    paths: list[str] = []
    for record in fields:
        if record.strip() and (path := _porcelain_path(record)):
            paths.append(path)
        if record[:1] in ("R", "C") or record[1:2] in ("R", "C"):
            next(fields, None)  # the rename/copy source follows in its own field
    return paths


def _in_code_scope(path: str) -> bool:
    """True when ``path`` is code under test (inside a dirty scope, never an evidence artifact)."""

    def inside(candidate: str, prefix: str) -> bool:
        return candidate == prefix or candidate.startswith(prefix + "/")

    return any(inside(path, prefix) for prefix in GIT_DIRTY_PATHS) and not any(
        inside(path, prefix) for prefix in GIT_EVIDENCE_PATHS
    )


def git_report(root: Path = REPO_ROOT) -> dict:
    """Commit SHA and dirty state of the checkout the two implementations come from.

    The dirty flag is computed over the **code under test only** (:data:`GIT_DIRTY_PATHS`,
    with :data:`GIT_EVIDENCE_PATHS` excluded), because the harness writes its own report and
    JSON while it samples this fingerprint: a full-tree ``git status`` would always call a
    perfectly clean measurement "dirty" and make the SHA untrustworthy.

    A missing git binary, a non-checkout directory or a timeout yields ``None`` values,
    which the report renders as ``n/a``; measuring must never fail because of provenance.

    Returns:
        ``{"git_commit", "git_dirty", "git_dirty_paths"}``: the full SHA, ``True`` only when
        one code path is modified/untracked, and the offending repo-relative paths (empty
        when clean). The list is not truncated here — the report names
        :data:`GIT_DIRTY_PATH_LIMIT` of them and counts the rest.
    """
    commit = _git("rev-parse", "HEAD", root=root)
    if commit is None:
        return {"git_commit": None, "git_dirty": None, "git_dirty_paths": []}
    status = _git("status", "--porcelain", "-z", "--", *GIT_DIRTY_PATHS, root=root, strip=False)
    if status is None:
        return {"git_commit": commit, "git_dirty": None, "git_dirty_paths": []}
    paths = [path for path in _porcelain_paths(status) if path and _in_code_scope(path)]
    return {
        "git_commit": commit,
        "git_dirty": bool(paths),
        "git_dirty_paths": list(dict.fromkeys(paths)),
    }


def _thread_env() -> dict:
    """The thread variables as they are *actually set in this process* (``None`` = unset)."""
    return {name: os.environ.get(name) for name in THREAD_ENV_VARS}


def env_report(threads: int | None = None, root: Path = REPO_ROOT) -> dict:
    """Environment fingerprint recorded once per run; versions come from metadata only.

    Args:
        threads: The ``--threads`` value in force. ``0`` means runtime defaults and exports
            nothing, so ``thread_env`` is exactly what every worker inherited; a positive
            value is what the workers export over the inherited environment (the caller's
            own environment is still reported verbatim, which is why the value is recorded).
        root: Checkout whose git commit/dirty flag fingerprint the tree under test.

    Returns:
        A JSON-safe mapping with no implementation imports: only stdlib probes plus
        package metadata.
    """
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_mb": _memory_mb(),
        "numpy": _version("numpy"),
        "jax": _version("jax"),
        "evoxc": _version("evoxc"),
        "tensorneat": _version("tensorneat"),
        "threads": threads,
        "thread_env": _thread_env(),
        "loadavg": _load_average(),
    }
    report.update(git_report(root))
    return report


def loadavg_summary(runs: list[dict]) -> dict:
    """Min/max 1-minute load recorded by the workers at start and end, over the ok runs.

    Returns ``{"n_runs", "start_min", "start_max", "end_min", "end_max"}``. Every bound is
    ``None`` when no run recorded a load average (JSON from a harness that predates the
    optional ``loadavg_start``/``loadavg_end`` keys, or a platform without ``getloadavg``),
    so the report can say "not recorded" instead of inventing a condition.
    """
    start: list[float] = []
    end: list[float] = []
    for run in runs or []:
        if run.get("status", STATUS_OK) != STATUS_OK:
            continue
        for key, bucket in (("loadavg_start", start), ("loadavg_end", end)):
            values = run.get(key)
            if isinstance(values, (list, tuple)) and values:
                first = _finite(values[0])
                if first is not None:
                    bucket.append(first)
    return {
        "n_runs": len(start),
        "start_min": min(start) if start else None,
        "start_max": max(start) if start else None,
        "end_min": min(end) if end else None,
        "end_max": max(end) if end else None,
    }


# --- Per-run record ---------------------------------------------------------------------


@dataclass
class RunResult:
    """One worker run of one ``(case, impl, backend, repeat)`` tuple (schema ``1``)."""

    case: str
    impl: str
    backend: str
    repeat: int
    seed: int
    schema: str = SCHEMA
    status: str = STATUS_OK
    error: str | None = None
    case_config: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    protocol: dict = field(default_factory=dict)
    compile_time_s: float | None = None
    generations_consumed_before_timing: int = 0
    generation_ms: list[float] = field(default_factory=list)
    ms_per_generation: float | None = None
    ms_per_generation_mean: float | None = None
    ms_per_generation_min: float | None = None
    ms_per_generation_max: float | None = None
    gens_per_second: float | None = None
    rss_after_import_mb: float | None = None
    rss_after_compile_mb: float | None = None
    rss_final_mb: float | None = None
    loadavg_start: list[float] | None = None
    loadavg_end: list[float] | None = None
    quality: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe dict: non-finite floats become ``None``, numpy values become Python ones."""
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        """Rebuild a record tolerantly: unknown keys are dropped, missing identity keys default."""
        names = {f.name for f in fields(cls)}
        known = {key: value for key, value in (d or {}).items() if key in names}
        for required, fallback in (
            ("case", ""),
            ("impl", ""),
            ("backend", ""),
            ("repeat", 0),
            ("seed", 0),
        ):
            known.setdefault(required, fallback)
        return cls(**known)


# --- Timing and quality summarisers -----------------------------------------------------


def summarize_timing(generation_ms: list[float]) -> dict:
    """Median/mean/min/max and throughput of per-generation wall clock samples."""
    samples = [value for value in (_finite(x) for x in generation_ms or []) if value is not None]
    median = _median(samples)
    if median is None:
        return {
            "ms_per_generation": None,
            "ms_per_generation_mean": None,
            "ms_per_generation_min": None,
            "ms_per_generation_max": None,
            "gens_per_second": None,
        }
    return {
        "ms_per_generation": median,
        "ms_per_generation_mean": float(sum(samples) / len(samples)),
        "ms_per_generation_min": float(min(samples)),
        "ms_per_generation_max": float(max(samples)),
        "gens_per_second": (1000.0 / median) if median > 0 else None,
    }


def quality_curve(fitnesses, target: float | None) -> dict:
    """Per-generation fitness curve; ``generations_to_target=None`` renders as ``>N``.

    Each element of ``fitnesses`` is one generation's per-individual fitness. NaN/inf
    fitness means "invalid" and is treated as ``-inf``; ``mean_fitness`` averages the finite
    values only (legacy ``analysis()`` semantics) and is ``-inf`` when none are finite.
    """
    _require_numpy("quality_curve")
    resolved_target = None if target is None else float(target)
    best_fitness: list[float] = []
    mean_fitness: list[float] = []
    valid_count: list[int] = []

    for generation in fitnesses:
        values = np.asarray(generation, dtype=np.float64).ravel()
        values = np.where(np.isfinite(values), values, -np.inf)
        finite = values[np.isfinite(values)]
        best_fitness.append(float(np.max(values)) if values.size else float("-inf"))
        mean_fitness.append(float(np.mean(finite)) if finite.size else float("-inf"))
        valid_count.append(int(finite.size))

    generations_to_target = None
    if resolved_target is not None:
        for index, best in enumerate(best_fitness, start=1):
            if best >= resolved_target:
                generations_to_target = index
                break

    return {
        "target": resolved_target,
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "valid_count": valid_count,
        "final_best_fitness": best_fitness[-1] if best_fitness else None,
        "generations_to_target": generations_to_target,
    }


def median_curve(curves: list[list[float]]) -> list[float]:
    """Element-wise median of per-repeat fitness curves: the aggregated quality-curve row.

    Only finite values enter each point's median (a ``-inf`` from an all-invalid generation
    stays ``-inf`` only while every repeat agrees). Curves of different lengths are truncated
    to the shortest — the reporter labels that row as covering the common range — and an
    empty input returns an empty curve.
    """
    usable = [curve for curve in (curves or []) if isinstance(curve, list) and curve]
    if not usable:
        return []
    length = min(len(curve) for curve in usable)
    aggregated: list[float] = []
    for index in range(length):
        values = [_finite(curve[index]) for curve in usable]
        finite = [value for value in values if value is not None]
        aggregated.append(_median(finite) if finite else float("-inf"))
    return aggregated


# --- Repeat aggregation -----------------------------------------------------------------


def _grouped(runs: list[dict]) -> list[tuple[tuple[str, str, str], list[dict]]]:
    """Group run records by ``(case, impl, backend)``, preserving first-seen order."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for run in runs or []:
        key = (run.get("case", ""), run.get("impl", ""), run.get("backend", ""))
        groups.setdefault(key, []).append(run)
    return list(groups.items())


def _group_status(statuses: list[str]) -> str:
    """``ok`` when all runs are ok, ``error`` when any errored, else the majority status.

    The error case returns before the vote, so the vote only ever chooses between ``skipped``
    and ``ok``; a tie goes to ``skipped``, the cautious side of those two.
    """
    if not statuses or all(status == STATUS_OK for status in statuses):
        return STATUS_OK
    if STATUS_ERROR in statuses:
        return STATUS_ERROR
    counts = Counter(statuses)
    priority = {STATUS_SKIPPED: 1, STATUS_OK: 0}
    return max(counts, key=lambda status: (counts[status], priority.get(status, 0)))


def _values(group: list[dict], key: str) -> list[float]:
    """Finite values of a top-level metric over the ok runs only."""
    out = []
    for run in group:
        if run.get("status", STATUS_OK) != STATUS_OK:
            continue
        value = _finite(run.get(key))
        if value is not None:
            out.append(value)
    return out


def _quality_values(group: list[dict], key: str) -> list[float]:
    """Finite values of a quality metric over the ok runs only."""
    out = []
    for run in group:
        if run.get("status", STATUS_OK) != STATUS_OK:
            continue
        quality = run.get("quality") or {}
        value = _finite(quality.get(key))
        if value is not None:
            out.append(value)
    return out


def _max_value(group: list[dict], key: str) -> float | None:
    """High-water mark of a metric over the ok runs (RSS is a high-water mark by nature)."""
    values = _values(group, key)
    return max(values) if values else None


def _seed_int(run: dict) -> int | None:
    try:
        return int(run.get("seed"))
    except (TypeError, ValueError):
        return None


def aggregate(runs: list[dict]) -> list[dict]:
    """Collapse repeat records into one record per ``(case, impl, backend)``.

    Numeric medians use the ok runs only; error/skipped runs are excluded from the medians
    but still counted in ``n_errors`` and summarised in ``errors``, so a group never vanishes.
    ``gens_to_target_all`` plus ``target``/``gens_to_target_reached``/``gens_to_target_repeats``
    let the report say how many repeats reached the target, not only the median of those that did.
    """
    aggregated = []
    for (case, impl, backend), group in _grouped(runs):
        statuses = [run.get("status", STATUS_OK) for run in group]
        ms_values = _values(group, "ms_per_generation")
        median_ms = _median(ms_values)
        compile_times = _values(group, "compile_time_s")
        final_best = _quality_values(group, "final_best_fitness")

        ordered = sorted(
            group, key=lambda run: (_seed_int(run) is None, _seed_int(run) or 0)
        )
        gens_to_target_all = []
        target = None
        for run in ordered:
            quality = run.get("quality") or {}
            if target is None and run.get("status", STATUS_OK) == STATUS_OK:
                target = quality.get("target")
            value = _finite(quality.get("generations_to_target"))
            gens_to_target_all.append(int(value) if value is not None else None)
        gens_to_target = _median(_quality_values(group, "generations_to_target"))
        gens_reached = sum(1 for item in gens_to_target_all if item is not None)

        errors = [
            short_error(run.get("error"))
            for run in group
            if run.get("status", STATUS_OK) != STATUS_OK and run.get("error")
        ]

        aggregated.append(
            {
                "case": case,
                "impl": impl,
                "backend": backend,
                "n_repeats": len(group),
                "seeds": sorted({s for s in (_seed_int(run) for run in group) if s is not None}),
                "status": _group_status(statuses),
                "ms_per_generation": median_ms,
                "ms_per_generation_min": min(ms_values) if ms_values else None,
                "ms_per_generation_max": max(ms_values) if ms_values else None,
                "ms_per_generation_mean": _median(_values(group, "ms_per_generation_mean")),
                "gens_per_second": (
                    1000.0 / median_ms if median_ms and median_ms > 0 else None
                ),
                "compile_time_s": _median(compile_times),
                "compile_time_s_min": min(compile_times) if compile_times else None,
                "compile_time_s_max": max(compile_times) if compile_times else None,
                "rss_after_import_mb": _max_value(group, "rss_after_import_mb"),
                "rss_after_compile_mb": _max_value(group, "rss_after_compile_mb"),
                "rss_final_mb": _max_value(group, "rss_final_mb"),
                "final_best_fitness": _median(final_best),
                "final_best_fitness_min": min(final_best) if final_best else None,
                "final_best_fitness_max": max(final_best) if final_best else None,
                "generations_to_target": gens_to_target,
                "gens_to_target_all": gens_to_target_all,
                "target": target,
                "gens_to_target_reached": gens_reached,
                "gens_to_target_repeats": len(gens_to_target_all),
                "n_errors": sum(1 for status in statuses if status != STATUS_OK),
                "errors": errors,
            }
        )
    return aggregated


# --- Old vs new -------------------------------------------------------------------------


def speedup(old_ms: float | None, new_ms: float | None) -> float | None:
    """``old/new`` when both values are finite and ``new > 0``, else ``None``."""
    old = _finite(old_ms)
    new = _finite(new_ms)
    if old is None or new is None or new <= 0:
        return None
    return old / new


def speedup_bounds(old: dict, new: dict) -> dict:
    """Best and worst old/new speed-up over the repeats of one case.

    ``best`` is the fastest old repeat over the slowest new repeat (``old_min/new_max``): the
    best case *for the old side*, which is the pessimistic bound for the new side. ``worst`` is
    the slowest old over the fastest new (``old_max/new_min``), the optimistic bound for the
    new side. Both come from the per-repeat medians the aggregation already recorded and are
    ``None`` when a side has no timing spread, so the report prints the range only when the
    repeats support it — and it prints the new side's worst ratio first.
    """
    old_low = _finite((old or {}).get("ms_per_generation_min"))
    old_high = _finite((old or {}).get("ms_per_generation_max"))
    new_low = _finite((new or {}).get("ms_per_generation_min"))
    new_high = _finite((new or {}).get("ms_per_generation_max"))
    return {
        "best": speedup(old_low, new_high),
        "worst": speedup(old_high, new_low),
    }


def quality_tolerance(old_best) -> float | None:
    """Active quality tolerance for one old final best: ``max(10 % of |old|, 1e-3)``."""
    value = _finite(old_best)
    if value is None:
        return None
    return max(QUALITY_TOLERANCE_REL * abs(value), QUALITY_ABS_TOL)


def quality_threshold(old_best) -> float | None:
    """Lowest new final best that still passes quality against this old final best."""
    value = _finite(old_best)
    tolerance = quality_tolerance(value)
    return None if value is None or tolerance is None else value - tolerance


def quality_rule_text() -> str:
    """The active quality tolerance in one phrase, for the verdict and the report notes."""
    return (
        f"max({QUALITY_TOLERANCE_REL:.0%} × |old|, {QUALITY_ABS_TOL:g} absolute floor)"
    )


def quality_overlap(old: dict, new: dict) -> str | None:
    """Where the new side's per-repeat final-best range sits against the old side's range.

    A factual observation for the report, never a replacement for the PASS/FAIL verdict:
    ``"new worse"`` when every new repeat is below the old range, ``"new better"`` when every
    new repeat is above it, ``"inside"`` when the two per-repeat ranges intersect (every new
    value inside the old range, or the ranges partially overlap). ``None`` when either side
    recorded no final best.
    """
    old_min = _finite((old or {}).get("final_best_fitness_min"))
    old_max = _finite((old or {}).get("final_best_fitness_max"))
    new_min = _finite((new or {}).get("final_best_fitness_min"))
    new_max = _finite((new or {}).get("final_best_fitness_max"))
    if old_min is None or old_max is None or new_min is None or new_max is None:
        return None
    if new_max < old_min:
        return "new worse"
    if new_min > old_max:
        return "new better"
    return "inside"


def _pair(old: dict, new: dict) -> dict:
    """Compare one aggregated old record with one aggregated new record."""
    both_ok = old.get("status") == STATUS_OK and new.get("status") == STATUS_OK
    old_ms = _finite(old.get("ms_per_generation"))
    new_ms = _finite(new.get("ms_per_generation"))
    old_best = _finite(old.get("final_best_fitness"))
    new_best = _finite(new.get("final_best_fitness"))
    tolerance = quality_tolerance(old_best)
    threshold = quality_threshold(old_best)

    timing_pass = None
    if both_ok and old_ms is not None and new_ms is not None:
        timing_pass = new_ms <= old_ms

    quality_absent = old_best is None or new_best is None
    quality_pass = None
    if both_ok and not quality_absent:
        quality_pass = new_best >= threshold

    return {
        "case": new.get("case", ""),
        "backend": new.get("backend", ""),
        "old_ms": old_ms,
        "new_ms": new_ms,
        "speedup": speedup(old_ms, new_ms),
        "speedup_range": speedup_bounds(old, new),
        "delta_ms": (new_ms - old_ms) if old_ms is not None and new_ms is not None else None,
        "old_final_best": old_best,
        "new_final_best": new_best,
        "quality_delta": (new_best - old_best) if not quality_absent else None,
        "quality_tolerance": tolerance,
        "quality_threshold": threshold,
        "quality_overlap": quality_overlap(old, new),
        "timing_pass": timing_pass,
        "quality_absent": quality_absent,
        "quality_pass": quality_pass,
    }


def comparisons(aggregated: list[dict]) -> list[dict]:
    """Pair every case's old record with each of its new backend records."""
    old_by_case: dict[str, dict] = {}
    new_by_case: dict[str, list[dict]] = {}
    order: list[str] = []
    for record in aggregated or []:
        case = record.get("case", "")
        if case not in order:
            order.append(case)
        if record.get("impl") == "old":
            old_by_case.setdefault(case, record)
        elif record.get("impl") == "new":
            new_by_case.setdefault(case, []).append(record)

    pairs = []
    for case in order:
        old = old_by_case.get(case)
        if old is None:
            continue
        for new in new_by_case.get(case, []):
            pairs.append(_pair(old, new))
    return pairs


def verdict(aggregated: list[dict], runs: list[dict] | None = None) -> dict:
    """Overall PASS / FAIL / INCOMPLETE per the rules in ``benchmarks/CONTEXT.md``.

    Any error/skipped run makes the overall status INCOMPLETE (each one is listed).
    Timing passes when every comparison is at least 1.00x, quality when the new final best
    stays within the active tolerance ``max(QUALITY_TOLERANCE_REL × |old|, QUALITY_ABS_TOL)``
    of the old one. The absolute floor is what keeps two runs that are both at the optimum
    from failing on float noise; it never rescues a case whose old fitness is meaningfully
    below the optimum, because there the relative term is the larger one. With no old-vs-new
    pair at all the timing/quality entries are ``n/a`` and the overall status is INCOMPLETE
    with the reason ``no old-vs-new comparison available`` (there is nothing to pass).
    """
    pairs = comparisons(aggregated)
    records = list(aggregated or []) if runs is None else list(runs)

    reasons: list[str] = []
    incomplete = [
        record for record in records if record.get("status", STATUS_OK) != STATUS_OK
    ]
    for record in incomplete:
        error = record.get("error")
        if error is None:
            error = record.get("errors")
        if isinstance(error, (list, tuple)):
            error = next((item for item in error if item), None)
        reasons.append(
            f"{record.get('case', '?')}/{record.get('impl', '?')}/"
            f"{record.get('backend', '?')}: {record.get('status')} — "
            f"{short_error(error) or NO_ERROR_TEXT}"
        )

    if not pairs:
        reasons.append("no old-vs-new comparison available")
        return {
            "status": "INCOMPLETE",
            "timing": "n/a",
            "quality": "n/a",
            "reasons": reasons,
            "comparisons": [],
        }

    timing_fail = [pair for pair in pairs if pair["timing_pass"] is False]
    timing_missing = [pair for pair in pairs if pair["timing_pass"] is None]
    judged = [pair for pair in pairs if pair["quality_pass"] is not None]
    quality_fail = [pair for pair in judged if pair["quality_pass"] is False]
    quality_missing = [pair for pair in pairs if pair["quality_pass"] is None]

    timing = "FAIL" if timing_fail else ("INCOMPLETE" if timing_missing else "PASS")
    if quality_fail:
        quality = "FAIL"
    elif len(judged) == len(pairs):
        quality = "PASS"
    elif not judged:
        quality = "n/a"
    else:
        quality = "INCOMPLETE"

    for pair in timing_fail:
        reasons.append(
            f"timing: {pair['case']}/{pair['backend']} new {_fmt(pair['new_ms'])} ms/gen > "
            f"old {_fmt(pair['old_ms'])} ms/gen (speed-up {_fmt(pair['speedup'])}x)"
        )
    for pair in quality_fail:
        old_best = pair["old_final_best"]
        tolerance = pair.get("quality_tolerance")
        threshold = pair.get("quality_threshold")
        if tolerance is None:
            tolerance = quality_tolerance(old_best)
        if threshold is None:
            threshold = quality_threshold(old_best)
        reasons.append(
            f"quality: {pair['case']}/{pair['backend']} new final best "
            f"{pair['new_final_best']:.4f} < threshold {threshold:.4f} "
            f"(old {old_best:.4f} - tolerance {tolerance:.4f}, i.e. {quality_rule_text()})"
        )
    if timing != "PASS":
        for pair in timing_missing:
            reasons.append(
                f"timing: no verdict for {pair['case']}/{pair['backend']} "
                "(both sides need status ok and a measured ms/generation)"
            )
    if quality != "PASS":
        for pair in quality_missing:
            reasons.append(
                f"quality: no verdict for {pair['case']}/{pair['backend']} "
                "(final best fitness missing on at least one side)"
            )

    if incomplete:
        status = "INCOMPLETE"
        reasons.append(
            f"{len(incomplete)} run(s) not ok — cannot finalise the verdict "
            f"(timing={timing}, quality={quality})"
        )
    elif timing == "FAIL" or quality == "FAIL":
        status = "FAIL"
    elif timing == "PASS" and quality == "PASS":
        status = "PASS"
        reasons.append(f"timing: all {len(pairs)} comparison(s) at or above 1.00x")
        reasons.append(
            f"quality: all {len(judged)} comparison(s) within "
            f"{quality_rule_text()} of the old final best"
        )
    else:
        status = "INCOMPLETE"
        reasons.append(
            "verdict needs both timing and quality to pass "
            f"(timing={timing}, quality={quality})"
        )

    return {
        "status": status,
        "timing": timing,
        "quality": quality,
        "reasons": reasons,
        "comparisons": pairs,
    }
