"""One-process-per-run worker: measure exactly one benchmark run and record it.

The CLI spawns this script once per ``(case, impl, backend, repeat)`` tuple and reads the
single ``RunResult`` JSON it writes to ``--out``. One implementation per process is
mandatory (fairness rule 1 in ``../CONTEXT.md``): the old side loads JAX, the new side
would corrupt its timings if JAX were already resident, so nothing here imports the
implementation under test at module scope.

Module scope is stdlib only, on purpose: ``numpy`` (and, on the old side, ``jax``) must
see ``OMP_NUM_THREADS``/``MKL_NUM_THREADS``/``OPENBLAS_NUM_THREADS``/``NUMEXPR_NUM_THREADS``
in the environment *at import time*, so the CLI-facing imports of ``harness.protocol``,
``cases`` and ``harness.adapters`` all happen inside :func:`main`, after the thread env
vars and ``sys.path`` have been fixed.

Every failure is recorded, never raised: the CLI reads ``--out`` even on a non-zero exit,
so a hard crash would silently lose a row.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

__all__ = ["main", "build_parser", "THREAD_ENV_VARS", "BENCH_DIR", "REPO_ROOT"]

# ``benchmarks/`` and the repository root, derived from this file so the worker behaves
# identically from any cwd (the CLI spawns it as a subprocess).
BENCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
"""Thread-count variables exported to *both* sides so the comparison is thread-for-thread."""

_XLA_THREADS_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={threads}"
"""``XLA_FLAGS`` for the old side: pin XLA's eigen pools to the same ``--threads``."""


# --- CLI --------------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    """``argparse`` type for a strictly positive count (a 0-generation run has no timing)."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    """``argparse`` type for a count where 0 is meaningful (warmup, threads, repeat)."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the documented worker CLI (see ``../CONTEXT.md``)."""
    parser = argparse.ArgumentParser(
        prog="benchmarks/harness/worker.py",
        description=(
            "Time one benchmark run (one case, implementation, backend and repeat) and "
            "write exactly one RunResult JSON to --out."
        ),
    )
    parser.add_argument(
        "--case", required=True, metavar="NAME", help="case name from benchmarks/cases.py"
    )
    parser.add_argument(
        "--impl", required=True, choices=("old", "new"), help="implementation under test"
    )
    # Mirrors protocol.DEFAULT_BACKEND; protocol is only importable after the thread env
    # vars are set, so the default is spelled out here rather than imported.
    parser.add_argument(
        "--backend",
        default="numpy",
        metavar="NAME",
        help='evoxc backend for --impl new (default: "numpy"); --impl old always runs JAX',
    )
    parser.add_argument(
        "--repeat",
        type=_non_negative_int,
        default=0,
        metavar="R",
        help="repeat index recorded in the result; the planner owns the seed + r offset",
    )
    parser.add_argument(
        "--seed", type=int, required=True, metavar="S", help="seed used verbatim for this run"
    )
    parser.add_argument(
        "--generations", type=_positive_int, required=True, metavar="N", help="timed generations"
    )
    parser.add_argument(
        "--warmup",
        type=_non_negative_int,
        default=3,
        metavar="N",
        help=(
            "untimed generations (default: 3), counted from the first generation the step "
            "function has not already consumed while preparing"
        ),
    )
    parser.add_argument(
        "--threads",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="thread count exported to both sides; 0 leaves the runtime defaults",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="file to write the RunResult JSON to (parents created)",
    )
    return parser


# --- Process setup (must happen before numpy / jax / evoxc are imported) -----------------


def _configure_sys_path(impl: str) -> None:
    """Put the tree under test first on ``sys.path``.

    An installed ``tensorneat`` (or ``tensorneat_legacy``) must never shadow the checkout
    being measured, so the implementation tree is inserted *last* and therefore wins
    ``sys.path[0]``. ``benchmarks/`` itself carries ``cases.py`` and the ``harness``
    package; the repository root keeps the checkout's own layout importable.

    Args:
        impl: ``"old"`` (``benchmarks/legacy_jax``) or ``"new"`` (``src/``).

    Raises:
        FileNotFoundError: If the implementation tree does not exist in this checkout.
    """
    impl_tree = (REPO_ROOT / "src") if impl == "new" else (BENCH_DIR / "legacy_jax")
    if not impl_tree.is_dir():
        raise FileNotFoundError(f"implementation tree for --impl {impl} is missing: {impl_tree}")
    for entry in (str(BENCH_DIR), str(REPO_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    if str(impl_tree) in sys.path:
        sys.path.remove(str(impl_tree))
    sys.path.insert(0, str(impl_tree))


def _configure_threads(threads: int, impl: str) -> dict[str, str]:
    """Export the thread count before numpy/jax import; return the variables actually set.

    ``--threads 0`` leaves the runtime defaults untouched ("as users run it") and therefore
    exports nothing — guessing the effective values would be a lie in the record.

    Args:
        threads: Requested thread count.
        impl: ``"old"`` also gets ``XLA_FLAGS``, which XLA only reads at import time.

    Returns:
        The ``{name: value}`` pairs written to ``os.environ`` (empty for ``--threads 0``).
    """
    if threads <= 0:
        return {}
    value = str(threads)
    exported = {name: value for name in THREAD_ENV_VARS}
    if impl == "old":
        exported["XLA_FLAGS"] = _XLA_THREADS_FLAGS.format(threads=threads)
    os.environ.update(exported)
    return exported


# --- Small helpers ----------------------------------------------------------------------


def _protocol_block(
    *, warmup: int, timed: int, threads: int, thread_env: dict[str, str], untimed: int
) -> dict:
    """The ``protocol`` block of a result: the budgets and the thread configuration used."""
    return {
        "warmup_generations": warmup,
        "timed_generations": timed,
        "threads": threads,
        "thread_env": dict(thread_env),
        "untimed_warmup_generations": untimed,
    }


def _split_step(result: object) -> tuple[object, object]:
    """Split a ``step_fn`` result into ``(state, fitness)``.

    Both documented shapes are accepted: ``(state, fitness)`` (the adapter contract) and
    ``(state, population, fitness)`` (a raw pipeline step). Anything else fails loudly with
    the observed type and length/shape instead of guessing which element is fitness.
    """
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[-1]
    observed = len(result) if isinstance(result, tuple) else getattr(result, "shape", None)
    raise TypeError(
        "step_fn must return (state, fitness) or (state, population, fitness); "
        f"got {type(result).__name__} (length/shape={observed!r})"
    )


def _summary_line(error: str) -> str:
    """Last non-empty line of a traceback — the one-line stderr summary of a failed run."""
    lines = [line.strip() for line in error.strip().splitlines() if line.strip()]
    return lines[-1][:300] if lines else "(no traceback)"


def _write_json(record: dict, out_path: Path) -> None:
    """Write exactly one JSON record to ``out_path``, creating parent directories.

    The record goes to a sibling temp file and is renamed into place, so the CLI never
    reads a half-written JSON when it inspects ``--out`` after the worker exits.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)


def _skipped_record(protocol, case, args, thread_env: dict[str, str]) -> dict:
    """Build the ``status="skipped"`` record for a case that declares other implementations."""
    return protocol.RunResult(
        schema=protocol.SCHEMA,
        case=case.name,
        impl=args.impl,
        backend=args.backend,
        repeat=args.repeat,
        seed=args.seed,
        status=protocol.STATUS_SKIPPED,
        error=(
            f"case {case.name} declares impls={list(case.impls)}; "
            f"{args.impl} is not applicable"
        ),
        case_config=case.as_dict(),
        env=protocol.env_report(threads=args.threads),
        protocol=_protocol_block(
            warmup=args.warmup,
            timed=args.generations,
            threads=args.threads,
            thread_env=thread_env,
            untimed=0,
        ),
    ).to_dict()


def _error_record(
    args,
    case,
    protocol,
    thread_env: dict[str, str],
    untimed: int,
    error: str,
) -> dict:
    """Best-effort ``status="error"`` record: identity keys, the traceback and what we know.

    The case snapshot and the environment probe are the only best-effort parts: if either
    breaks (a half-constructed case, an unusable ``importlib.metadata``), the record must
    still be written, because it is the only channel back to the CLI.
    """
    case_config: dict = {}
    if case is not None:
        try:
            case_config = case.as_dict()
        except Exception:  # noqa: BLE001 - a broken snapshot must not lose the traceback
            case_config = {}
    env: dict = {}
    if protocol is not None:
        try:
            env = protocol.env_report(threads=args.threads)
        except Exception:  # noqa: BLE001 - a broken probe must not lose the traceback
            env = {}
    return {
        "schema": getattr(protocol, "SCHEMA", None),
        "case": getattr(case, "name", args.case),
        "impl": args.impl,
        "backend": args.backend,
        "repeat": args.repeat,
        "seed": args.seed,
        "status": getattr(protocol, "STATUS_ERROR", "error"),
        "error": error,
        "case_config": case_config,
        "env": env,
        "protocol": _protocol_block(
            warmup=args.warmup,
            timed=args.generations,
            threads=args.threads,
            thread_env=thread_env,
            untimed=untimed,
        ),
    }


# --- The measurement --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Measure one run and write its ``RunResult`` JSON to ``--out``.

    Args:
        argv: Arguments as for ``sys.argv[1:]`` (``None`` reads the real command line).

    Returns:
        ``0`` when the record was written (``ok`` or ``skipped``), ``1`` when the run failed
        and an ``error`` record was written instead.
    """
    args = build_parser().parse_args(argv)
    out_path = Path(args.out)

    protocol = None
    case = None
    thread_env: dict[str, str] = {}
    untimed = 0

    try:
        # sys.path and the thread env come first: numpy reads the thread variables and the
        # implementation must be importable from the tree under test, not from site-packages.
        _configure_sys_path(args.impl)
        thread_env = _configure_threads(args.threads, args.impl)

        import harness.protocol as protocol  # noqa: PLC0415 - see module docstring
        import cases  # noqa: PLC0415 - needs BENCH_DIR on sys.path

        case = cases.get_cases([args.case])[0]
        if args.impl not in case.impls:
            _write_json(_skipped_record(protocol, case, args, thread_env), out_path)
            return 0

        # Imported only here: the adapter module imports the implementation tree (jax /
        # tensorneat), and a case that cannot run this impl is recorded as skipped without
        # touching either side. RSS is sampled right after, with both sides loaded.
        from harness.adapters import get_adapter  # noqa: PLC0415 - imports the impl tree

        adapter = get_adapter(args.impl)
        rss_after_import = protocol.rss_mb()
        pipeline = adapter.build(case, args.backend, args.seed)
        state = adapter.setup(pipeline)

        # No logging inside the measured region (fairness rule 5): both streams go to devnull
        # for prepare + warmup + the timed loop.
        with (
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            state, compile_s, consumed, step_fn = adapter.prepare(pipeline, state)
            consumed = int(consumed)
            rss_after_compile = protocol.rss_mb()

            # The old side consumes no generation while compiling, the new side consumes one
            # (its build runs a real generation). Skipping that one from the warmup makes both
            # sides time the same generation index; the warmup fitness values still enter the
            # quality curve, so both sides' curves cover the same generations too.
            untimed = max(0, args.warmup - consumed)
            fitness_curve = []
            for _ in range(untimed):
                state, fitness = _split_step(step_fn(state))
                fitness_curve.append(fitness)

            generation_ms: list[float] = []
            for _ in range(args.generations):
                # The whole call is timed, so the host sync inside step_fn is charged.
                t0 = time.perf_counter()
                result = step_fn(state)
                elapsed = time.perf_counter() - t0
                state, fitness = _split_step(result)
                generation_ms.append(elapsed * 1000.0)
                fitness_curve.append(fitness)

        rss_final = protocol.rss_mb()
        timing = protocol.summarize_timing(generation_ms)
        record = protocol.RunResult(
            schema=protocol.SCHEMA,
            case=case.name,
            impl=args.impl,
            backend=adapter.backend_label,
            repeat=args.repeat,
            seed=args.seed,
            status=protocol.STATUS_OK,
            case_config=case.as_dict(),
            env=protocol.env_report(threads=args.threads),
            protocol=_protocol_block(
                warmup=args.warmup,
                timed=args.generations,
                threads=args.threads,
                thread_env=thread_env,
                untimed=untimed,
            ),
            compile_time_s=compile_s,
            generations_consumed_before_timing=consumed,
            generation_ms=generation_ms,
            ms_per_generation=timing["ms_per_generation"],
            ms_per_generation_mean=timing["ms_per_generation_mean"],
            ms_per_generation_min=timing["ms_per_generation_min"],
            ms_per_generation_max=timing["ms_per_generation_max"],
            gens_per_second=timing["gens_per_second"],
            rss_after_import_mb=rss_after_import,
            rss_after_compile_mb=rss_after_compile,
            rss_final_mb=rss_final,
            quality=protocol.quality_curve(fitness_curve, case.fitness_target),
        ).to_dict()
        _write_json(record, out_path)
        return 0
    except Exception:
        error = traceback.format_exc()
        record = _error_record(args, case, protocol, thread_env, untimed, error)
        try:
            _write_json(record, out_path)
        except Exception:  # noqa: BLE001 - the operator must still see why the run died
            print(error, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        print(
            f"[worker] {args.case}/{args.impl}/{args.backend}: {_summary_line(error)}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
