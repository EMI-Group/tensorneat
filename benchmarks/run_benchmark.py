#!/usr/bin/env python3
"""Benchmark CLI: plan the case matrix, run one worker per run, aggregate, render ``RESULTS.md``.

One worker subprocess per ``(case, implementation, backend, repeat)`` tuple, strictly
sequential — the JAX baseline and the evoxc rewrite never share an interpreter, so their
thread pools and allocators cannot disturb each other's timings. Every schema, aggregation
and verdict decision comes from ``harness.protocol``; this file only orchestrates.

``--list`` and ``--dry-run`` work on a bare interpreter (no numpy): the numpy-dependent
modules are imported lazily, from inside the functions that need them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `import cases` / `import harness...` must resolve from anywhere
    sys.path.insert(0, str(HERE))

import cases  # noqa: E402  (needs the sys.path fix above; stdlib-only, so --list stays bare)

WORKER = HERE / "harness" / "worker.py"
"""The per-run worker script, spawned with the current interpreter."""

IMPL_ORDER = ("old", "new")
"""Planning order: the baseline first, then the candidate."""

SELECTIONS = ("old", "new", "both")
"""Valid values of ``--impl``."""

STDERR_TAIL_LINES = 15
"""How many trailing stderr lines are summarised into a failed run's ``error`` field."""


# --- Lazy imports of the numpy-touching half of the harness -------------------------------


def _protocol():
    """Import ``harness.protocol`` (numpy-optional) and return the module."""
    from harness import protocol

    return protocol


def _reporter():
    """Import ``harness.reporter`` (formatting only) and return the module."""
    from harness import reporter

    return reporter


# --- Argument parsing ---------------------------------------------------------------------


def _comma_list(text: str | None) -> list[str]:
    """Split a comma-separated CLI value, dropping blanks and duplicates."""
    if not text:
        return []
    out: list[str] = []
    for chunk in str(text).split(","):
        name = chunk.strip()
        if name and name not in out:
            out.append(name)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser; the defaults are the protocol's, never a second copy of them."""
    protocol = _protocol()
    parser = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--impl",
        choices=SELECTIONS,
        default="both",
        help="which implementation(s) to run; 'both' compares them",
    )
    parser.add_argument(
        "--backend",
        default=protocol.DEFAULT_BACKEND,
        help="comma-separated evoxc backends for the new implementation "
        "(the old side always runs on JAX)",
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="comma-separated case names, or 'all' for the whole matrix",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=protocol.DEFAULT_GENERATIONS,
        help="timed generations per run",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=protocol.DEFAULT_WARMUP,
        help="untimed, discarded generations before the timed loop",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=protocol.DEFAULT_REPEATS,
        help="repeats per (case, impl, backend); repeat r uses seed + r",
    )
    parser.add_argument(
        "--seed", type=int, default=protocol.DEFAULT_SEED, help="base seed of repeat 0"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=protocol.DEFAULT_THREADS,
        help="thread count exported to every worker (0 = runtime defaults)",
    )
    parser.add_argument(
        "--out",
        default=str(HERE / "RESULTS.md"),
        help="markdown report path (regenerated in place)",
    )
    parser.add_argument(
        "--json",
        default=str(HERE / "results" / "latest.json"),
        help="machine-readable run record the report is rendered from",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_cases",
        help="print the case matrix and exit (no run, no report)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned worker commands and exit without running anything",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=protocol.DEFAULT_TIMEOUT,
        help="per-run wall clock limit in seconds",
    )
    return parser


def _validation_error(args: argparse.Namespace) -> str | None:
    """Return a message for an unusable protocol argument, else ``None``."""
    if args.generations < 1:
        return "--generations must be >= 1"
    if args.warmup < 0:
        return "--warmup must be >= 0"
    if args.repeats < 1:
        return "--repeats must be >= 1"
    if args.timeout < 1:
        return "--timeout must be >= 1 second"
    if not _comma_list(args.backend):
        return "--backend needs at least one backend name"
    return None


def _fail(message: str) -> int:
    """Report a plan-time error on stderr and return the plan-error exit code."""
    print(f"run_benchmark.py: error: {message}", file=sys.stderr)
    return 2


# --- Planning -----------------------------------------------------------------------------


def _resolve_cases(case_names: list[str] | None) -> list:
    """Resolve names to :class:`cases.Case` objects in declaration order.

    Raises:
        ValueError: If a name is unknown (the message lists the valid names).
    """
    names = list(case_names) if case_names else ["all"]
    if "all" in names:
        names = ["all"]
    order = {name: index for index, name in enumerate(cases.all_names())}
    return sorted(
        cases.get_cases(names), key=lambda case: order.get(case.name, len(order))
    )


def plan_runs(impl_selection, backends, case_names, repeats, seed):
    """Plan one run per ``(case, impl, backend, repeat)`` tuple.

    The old side runs on JAX only, once per case; the new side runs once per requested
    evoxc backend. Selecting ``all`` plans strictly from each case's ``impls``, so a
    new-only case simply has no old row; naming that case (and ``old``) explicitly instead
    yields a ``skipped`` record, because there the user asked for something that cannot run.

    Args:
        impl_selection: ``"old"``, ``"new"`` or ``"both"``.
        backends: evoxc backends for the new implementation.
        case_names: case names, or ``None``/``["all"]`` for the whole matrix.
        repeats: repeats per group; repeat ``r`` uses ``seed + r``.
        seed: base seed.

    Returns:
        ``(runs, skipped)``: planned runs as dicts with ``case/impl/backend/repeat/seed``,
        and skipped records in the ``RunResult`` shape (``status="skipped"``).

    Raises:
        ValueError: If ``case_names`` names an unknown case.
    """
    protocol = _protocol()
    impls = IMPL_ORDER if impl_selection == "both" else (impl_selection,)
    names = list(case_names) if case_names else ["all"]
    explicit = "all" not in names
    new_backends = list(backends) or [protocol.DEFAULT_BACKEND]

    runs: list[dict] = []
    skipped: list[dict] = []
    for case in _resolve_cases(names):
        for impl in IMPL_ORDER:
            if impl not in impls:
                continue
            if impl not in case.impls:
                if not explicit:
                    continue
                skipped.append(
                    {
                        "case": case.name,
                        "impl": impl,
                        "backend": (
                            protocol.OLD_BACKEND_LABEL
                            if impl == "old"
                            else new_backends[0]
                        ),
                        "repeat": 0,
                        "seed": seed,
                        "status": protocol.STATUS_SKIPPED,
                        "error": (
                            f"case {case.name} declares impls={tuple(case.impls)}; "
                            f"{impl} is not applicable"
                        ),
                    }
                )
                continue
            backend_list = (
                [protocol.OLD_BACKEND_LABEL] if impl == "old" else new_backends
            )
            for backend in backend_list:
                for repeat in range(repeats):
                    runs.append(
                        {
                            "case": case.name,
                            "impl": impl,
                            "backend": backend,
                            "repeat": repeat,
                            "seed": seed + repeat,
                        }
                    )
    return runs, skipped


# --- Execution ----------------------------------------------------------------------------


def _worker_command(run: dict, args: argparse.Namespace, out_path) -> list[str]:
    """The exact argv of one worker invocation."""
    return [
        sys.executable,
        str(WORKER),
        "--case",
        str(run["case"]),
        "--impl",
        str(run["impl"]),
        "--backend",
        str(run["backend"]),
        "--repeat",
        str(run["repeat"]),
        "--seed",
        str(run["seed"]),
        "--generations",
        str(args.generations),
        "--warmup",
        str(args.warmup),
        "--threads",
        str(args.threads),
        "--out",
        str(out_path),
    ]


def _tail(text, lines: int = STDERR_TAIL_LINES) -> str:
    """Last ``lines`` lines of captured output (``""`` for nothing captured)."""
    if not text:
        return ""
    chunks = [line for line in str(text).strip().splitlines()]
    return "\n".join(chunks[-lines:])


def _load_record(path) -> dict | None:
    """Read a worker JSON record, or ``None`` when it is missing or not a JSON object."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _error_record(run: dict, args: argparse.Namespace, case_by_name: dict, message: str) -> dict:
    """Build a ``status="error"`` run record so a failed run never vanishes from the report."""
    protocol = _protocol()
    case = case_by_name.get(run["case"])
    payload = {
        "case": run["case"],
        "impl": run["impl"],
        "backend": run["backend"],
        "repeat": run["repeat"],
        "seed": run["seed"],
        "status": protocol.STATUS_ERROR,
        "error": protocol.short_error(message) or protocol.NO_ERROR_TEXT,
        "case_config": case.as_dict() if case is not None else {},
        "env": protocol.env_report(threads=args.threads),
        "protocol": {
            "warmup_generations": args.warmup,
            "timed_generations": args.generations,
            "threads": args.threads,
        },
    }
    return protocol.RunResult.from_dict(payload).to_dict()


def execute_run(
    run: dict,
    args: argparse.Namespace,
    tmpdir,
    case_by_name: dict,
    index: int,
    total: int,
) -> dict:
    """Run one worker to completion and return its record (never ``None``).

    A missing/invalid output file, a timeout and a non-zero exit all produce a
    ``status="error"`` record carrying the stderr tail, so the failure is visible in both
    the JSON and the report.
    """
    protocol = _protocol()
    out_path = Path(tmpdir) / f"run-{index:03d}.json"
    command = _worker_command(run, args, out_path)
    print(
        f"[{index}/{total}] {run['case']} impl={run['impl']} backend={run['backend']} "
        f"repeat={run['repeat']} seed={run['seed']} ",
        end="",
        flush=True,
    )
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        message = f"timeout after {args.timeout}s"
        tail = protocol.short_error(_tail(stderr))
        if tail:
            message = f"{message}: {tail}"
        print(f"ERROR: timeout after {args.timeout}s", flush=True)
        return _error_record(run, args, case_by_name, message)

    record = _load_record(out_path)
    if proc.returncode != 0:
        tail = protocol.short_error(_tail(proc.stderr) or _tail(proc.stdout))
        if record is not None and record.get("status") == protocol.STATUS_ERROR:
            record = protocol.RunResult.from_dict(record).to_dict()
            if not record.get("error"):
                record["error"] = tail or f"worker exited with code {proc.returncode}"
        else:
            record = _error_record(
                run,
                args,
                case_by_name,
                tail or f"worker exited with code {proc.returncode} and wrote no run record",
            )
        summary = protocol.short_error(record.get("error")) or protocol.NO_ERROR_TEXT
        print(f"ERROR: {summary}", flush=True)
        return record

    if record is None:
        message = f"worker exited 0 but wrote no valid JSON to {out_path}"
        print(f"ERROR: {message}", flush=True)
        return _error_record(run, args, case_by_name, message)

    record = protocol.RunResult.from_dict(record).to_dict()
    if record.get("status") != protocol.STATUS_OK:
        summary = protocol.short_error(record.get("error")) or protocol.NO_ERROR_TEXT
        print(f"ERROR: {summary}", flush=True)
        return record
    milliseconds = record.get("ms_per_generation")
    if isinstance(milliseconds, (int, float)):
        print(f"{milliseconds:.2f} ms/gen  {record.get('status')}", flush=True)
    else:
        print(f"{record.get('status')}", flush=True)
    return record


def _merge_in_order(run_records, skipped_records, case_order) -> list[dict]:
    """Interleave executed and skipped records in case order, then old-before-new."""
    order = {name: position for position, name in enumerate(case_order)}
    buckets: dict[tuple[str, str], list[dict]] = {}
    for record in list(run_records) + list(skipped_records):
        key = (record.get("case", ""), record.get("impl", ""))
        buckets.setdefault(key, []).append(record)
    keys = sorted(
        buckets,
        key=lambda key: (
            order.get(key[0], len(order)),
            IMPL_ORDER.index(key[1]) if key[1] in IMPL_ORDER else len(IMPL_ORDER),
        ),
    )
    return [record for key in keys for record in buckets[key]]


# --- Reporting ----------------------------------------------------------------------------


def _build_document(
    args: argparse.Namespace,
    selected,
    backends,
    run_records,
    aggregated,
    verdict,
    json_path,
) -> dict:
    """Assemble the JSON document the reporter renders (schema ``protocol.SCHEMA``)."""
    protocol = _protocol()
    impls = list(IMPL_ORDER) if args.impl == "both" else [args.impl]
    return {
        "schema": protocol.SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": {
            "env": protocol.env_report(threads=args.threads),
            "cases": [case.name for case in selected],
            "impls": impls,
            "backends": list(backends),
            "protocol": {
                "generations": args.generations,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "seed": args.seed,
                "threads": args.threads,
                "timeout": args.timeout,
            },
            "command": " ".join([sys.executable, *sys.argv]),
            "case_matrix": [case.as_dict() for case in selected],
            "json_path": str(json_path),
            "out_path": str(Path(args.out)),
        },
        "runs": run_records,
        "aggregated": aggregated,
        "verdict": verdict,
    }


def _write_json(document: dict, path) -> None:
    """Write the run record (parent directories created, trailing newline)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def _print_verdict(verdict: dict) -> None:
    """Print the final verdict line plus every reason behind it."""
    status = verdict.get("status", "INCOMPLETE")
    reasons = [str(reason) for reason in (verdict.get("reasons") or [])]
    headline = reasons[-1] if reasons else "no reason recorded"
    print(f"VERDICT: {status} — {headline}")
    for reason in reasons[:-1]:
        print(f"  - {reason}")


def _print_case_matrix() -> None:
    """Print the case matrix from ``cases`` alone (no numpy, no report files)."""
    header = ("name", "family", "pop", "gens", "target", "impls", "problem", "notes")
    selected = cases.get_cases()
    rows = [
        (
            case.name,
            case.family,
            str(case.pop_size),
            str(case.generation_limit),
            f"{case.fitness_target:g}",
            "/".join(case.impls),
            str(case.problem_config["class"]),
            case.notes or "",
        )
        for case in selected
    ]
    widths = [
        max(len(row[column]) for row in (header, *rows))
        for column in range(len(header))
    ]
    for row in (header, *rows):
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip())
    print(f"\n{len(rows)} case(s); select them with --cases <name>[,<name>] or --cases all")


def _print_plan(runs, skipped, args: argparse.Namespace) -> None:
    """Print the skipped records and the exact planned worker commands, one per line."""
    for record in skipped:
        print(f"# skipped {record['case']} impl={record['impl']}: {record['error']}")
    if not runs:
        print("# no runs planned")
        return
    for index, run in enumerate(runs, start=1):
        out_path = Path("<tmpdir>") / f"run-{index:03d}.json"
        print(shlex.join(_worker_command(run, args, out_path)))


# --- Entry point --------------------------------------------------------------------------


def main(argv=None) -> int:
    """Parse the CLI, run the plan, write the JSON and the report; return the exit code."""
    args = build_parser().parse_args(argv)

    if args.list_cases:
        _print_case_matrix()
        return 0

    error = _validation_error(args)
    if error:
        return _fail(error)

    backends = _comma_list(args.backend)
    case_names = _comma_list(args.cases) or ["all"]
    try:
        selected = _resolve_cases(case_names)
        runs, skipped = plan_runs(args.impl, backends, case_names, args.repeats, args.seed)
    except ValueError as exc:
        return _fail(str(exc))

    if args.dry_run:
        _print_plan(runs, skipped, args)
        return 0

    if not WORKER.is_file():
        return _fail(f"worker script missing: {WORKER} — the harness is incomplete")

    protocol = _protocol()
    reporter = _reporter()
    case_by_name = {case.name: case for case in selected}
    json_path = Path(args.json)
    out_path = Path(args.out)
    tmp_parent = os.environ.get("TMPDIR") or None

    run_records: list[dict] = []
    if runs:
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmpdir:
            total = len(runs)
            for index, run in enumerate(runs, start=1):
                run_records.append(
                    execute_run(run, args, tmpdir, case_by_name, index, total)
                )

    records = _merge_in_order(run_records, skipped, [case.name for case in selected])
    aggregated = protocol.aggregate(records)
    verdict = protocol.verdict(aggregated, records)
    document = _build_document(
        args, selected, backends, records, aggregated, verdict, json_path
    )

    if not runs:
        _write_json(document, json_path)
        print(
            f"no runs planned for impl={args.impl} cases={','.join(case_names)}: "
            f"{len(skipped)} skipped record(s); wrote {json_path} only "
            "(no report, nothing was measured)",
            file=sys.stderr,
        )
        _print_verdict(verdict)
        return 1

    _write_json(document, json_path)
    reporter.write_report(document, out_path)
    print(f"wrote {json_path} and {out_path}")
    _print_verdict(verdict)

    ok = all(record.get("status") == protocol.STATUS_OK for record in records)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
