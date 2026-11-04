#!/usr/bin/env python3
"""Measure one ``(case, backend)`` pair: build time, per-generation time, peak RSS.

This is the standalone, dependency-light companion of the full harness
(``benchmarks/run_benchmark.py``): one process, one configuration, one table on
stdout (plus an optional JSON record). It exists so a backend can be measured
without the old-vs-new protocol, in particular to time the ``iree`` compiled
path against the ``numpy`` reference interpreter with the same code.

Protocol
--------
For one ``(case, backend, pop, seed)``:

1. ``Pipeline.setup()`` is called and its wall time reported as ``setup_s``.
   That call is host-only *plus* NEAT's own genetic-operator graph build (a
   smaller ``evoxc.build``), so it is not the expensive compile.
2. The first ``Pipeline.step()`` pays the generation graph -- trace, lower,
   compile, load -- and its cost is reported twice: as ``first_step_s`` (wall
   time of that step, build + one run) and as ``build_s`` (``Pipeline``'s own
   ``build_seconds``, i.e. the ``evoxc.build`` alone). **The build is never
   charged to a generation and the build step is excluded from every timing
   sample below.**
3. ``--warmup`` further generations run untimed (default 3), then
   ``--generations`` generations are timed (default 20) and reported as
   ``ms/generation`` mean, median, min and max.
4. Peak RSS is ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` converted to MiB.

Peak RSS is a **process-lifetime high-water mark**, not the footprint of this
configuration: once anything in the process has touched more memory, the value
never goes back down. **Measure each ``(case, backend)`` in a FRESH process**, or
the second measurement silently inherits the first one's high-water mark. This
CLI therefore measures exactly one configuration per invocation by design.

Backend options
---------------
``backend`` is forwarded verbatim to ``evoxc.build``, and
``Pipeline(compile_options={...})`` is expanded into that call's ``**options``.
The one evoxc-defined, value-validated option is spelled ``opt_level`` (accepted
forms: ``"O0"``..``"O3"``, ``"0"``..``"3"``, or ints 0..3; evoxc normalizes and
translates it per backend). ``--opt-level`` passes it through unchanged as
``compile_options={"opt_level": <value>}``.

Verified on this box with iree-base-compiler/-runtime 3.11.0 (llvm-cpu target):
``O0`` and ``O1`` compile and run, while ``O2``/``O3`` currently fail *inside*
iree-compile (a ``vector.gather`` lowering error) and surface as
``evoxc.core.errors.BackendError`` -- the value is forwarded, the backend
validates it, and nothing falls back to another backend.

Cases
-----
``xor`` and ``xor3d`` mirror the canonical matrix of ``benchmarks/cases.py``:
``NEAT(pop_size=..., species_size=20, survival_threshold=0.01)``,
``DefaultGenome(num_inputs=2|3, num_outputs=1, max_nodes=50, max_conns=100,
output_transform=sigmoid)``, ``problem.XOR()`` / ``problem.XOR3d()``. The
fitness target only affects ``auto_run`` (never ``step``), so it is recorded but
does not bound this measurement.

Usage
-----
    python benchmarks/backend_measure.py --backend numpy --pop 100
    python benchmarks/backend_measure.py --backend iree --case xor3d --pop 100 \\
        --json benchmarks/results/iree_backend_measure.json

The default protocol is 3 warmup + 20 timed generations on every backend. For a
quick iree sanity check (its compile is ~8 s and dominates anything under a few
tens of generations) reduce the sample: ``--backend iree --pop 32 --warmup 1
--generations 3`` (``--pop`` must be at least ``species_size``, which the
canonical matrix fixes at 20).

Requires ``tensorneat`` (installed, or importable from ``../src``), ``numpy``
and -- for ``--backend iree`` -- the optional ``evoxc[iree]`` extra. Nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Repository root (``<repo>/benchmarks/backend_measure.py``).
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
#: The src-layout package, so the CLI also works from a bare checkout.
SRC_DIR = REPO_ROOT / "src"

#: The two benchmark families this CLI knows, mirroring ``benchmarks/cases.py``.
CASES: dict[str, dict[str, Any]] = {
    "xor": {"problem": "XOR", "num_inputs": 2, "fitness_target": -0.1},
    "xor3d": {"problem": "XOR3d", "num_inputs": 3, "fitness_target": -0.05},
}
#: Static genome geometry of the canonical matrix (``benchmarks/cases.py``).
MAX_NODES = 50
MAX_CONNS = 100
#: NEAT configuration of the canonical matrix.
SPECIES_SIZE = 20
SURVIVAL_THRESHOLD = 0.01


def _tensorneat() -> Any:
    """Import ``tensorneat``, falling back to the in-repo ``src`` directory.

    Kept lazy so ``--help`` and the argument validation work on a bare
    interpreter that has neither the package installed nor numpy available.
    """
    try:
        import tensorneat
    except ImportError:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        import tensorneat
    return tensorneat


def _peak_rss_mib() -> float:
    """This process's peak resident set size in MiB.

    ``ru_maxrss`` is KiB on Linux and bytes on macOS; it is a *lifetime*
    high-water mark, which is why one configuration is measured per process.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def _build_pipeline(args: argparse.Namespace) -> tuple[Any, dict | None]:
    """Construct the ``Pipeline`` for ``args`` and return it with its compile options."""
    tn = _tensorneat()
    case = CASES[args.case]
    compile_options = None if args.opt_level is None else {"opt_level": args.opt_level}
    pipeline = tn.Pipeline(
        algorithm=tn.algorithm.NEAT(
            pop_size=args.pop,
            species_size=SPECIES_SIZE,
            survival_threshold=SURVIVAL_THRESHOLD,
            genome=tn.genome.DefaultGenome(
                num_inputs=case["num_inputs"],
                num_outputs=1,
                max_nodes=MAX_NODES,
                max_conns=MAX_CONNS,
                output_transform=tn.common.ACT.sigmoid,
            ),
        ),
        problem=getattr(tn.problem, case["problem"])(),
        seed=args.seed,
        generation_limit=args.warmup + args.generations,
        fitness_target=case["fitness_target"],
        backend=args.backend,
        compile_options=compile_options,
        is_detailed=False,
    )
    return pipeline, compile_options


def measure(args: argparse.Namespace) -> dict[str, Any]:
    """Run the protocol for ``args`` and return the record (also printed as a table).

    Args:
        args: Parsed CLI arguments (``backend``, ``case``, ``pop``, ``generations``,
            ``warmup``, ``seed``, ``opt_level``).

    Returns:
        dict: The JSON-safe record; every timing is in seconds except the
        ``ms_per_generation`` block and ``peak_rss_mib``.
    """
    import numpy as np

    from tensorneat.common import to_numpy

    pipeline, compile_options = _build_pipeline(args)

    started = time.perf_counter()
    state = pipeline.setup()
    setup_s = time.perf_counter() - started

    started = time.perf_counter()
    state, _, fitnesses = pipeline.step(state)  # pays the evoxc build
    first_step_s = time.perf_counter() - started
    build_s = float(pipeline.build_seconds)

    for _ in range(args.warmup):
        state, _, fitnesses = pipeline.step(state)

    samples: list[float] = []
    for _ in range(args.generations):
        started = time.perf_counter()
        state, _, fitnesses = pipeline.step(state)
        samples.append(time.perf_counter() - started)

    milliseconds = [sample * 1000.0 for sample in samples]
    mean_ms = statistics.fmean(milliseconds)
    return {
        "case": args.case,
        "backend": args.backend,
        "pop": args.pop,
        "seed": args.seed,
        "warmup_generations": args.warmup,
        "timed_generations": args.generations,
        "compile_options": compile_options,
        "setup_s": round(setup_s, 6),
        "build_s": round(build_s, 6),
        "first_step_s": round(first_step_s, 6),
        "ms_per_generation": {
            "mean": round(mean_ms, 4),
            "median": round(statistics.median(milliseconds), 4),
            "min": round(min(milliseconds), 4),
            "max": round(max(milliseconds), 4),
            "samples": [round(value, 4) for value in milliseconds],
        },
        "generations_per_second": round(1000.0 / mean_ms, 4) if mean_ms > 0 else None,
        "peak_rss_mib": round(_peak_rss_mib(), 2),
        "final_generation": int(to_numpy(state.generation)),
        "final_max_fitness": float(np.max(fitnesses)),
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }


def render(record: dict[str, Any]) -> str:
    """Render ``record`` as an aligned two-column table plus the measurement caveats."""
    per_generation = record["ms_per_generation"]
    lines = [
        f"backend_measure: {record['case']} / {record['backend']} / "
        f"pop {record['pop']} / seed {record['seed']}",
        "",
        f"{'metric':<38}{'value':>24}",
        f"{'-' * 38} {'-' * 24:>24}",
        f"{'setup (Pipeline.setup)':<38}{record['setup_s']:>18.3f} s",
        f"{'evoxc build (pipeline.build_seconds)':<38}{record['build_s']:>18.3f} s",
        f"{'first step (build + 1 run)':<38}{record['first_step_s']:>18.3f} s",
        f"{'warmup generations (untimed)':<38}{record['warmup_generations']:>24d}",
        f"{'timed generations':<38}{record['timed_generations']:>24d}",
        f"{'ms/generation mean':<38}{per_generation['mean']:>21.3f} ms",
        f"{'ms/generation median':<38}{per_generation['median']:>21.3f} ms",
        f"{'ms/generation min / max':<38}"
        f"{per_generation['min']:>14.3f} / {per_generation['max']:.3f} ms",
        f"{'generations/second (from mean)':<38}"
        f"{record['generations_per_second']:>18.3f} /s",
        f"{'peak RSS (process high-water mark)':<38}{record['peak_rss_mib']:>18.2f} MiB",
        f"{'final generation':<38}{record['final_generation']:>24d}",
        f"{'final generation max fitness':<38}{record['final_max_fitness']:>24.6f}",
        "",
        f"compile_options: {record['compile_options']!r}",
        "build time is one-off and per shape; it is excluded from every "
        "ms/generation sample.",
        "peak RSS is a process-lifetime high-water mark -- measure each "
        "(case, backend) in a fresh process.",
    ]
    return "\n".join(lines)


def _positive_int(text: str) -> int:
    """argparse type for a strictly positive integer."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    """argparse type for an integer >= 0 (a zero warmup is legal)."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI; ``--opt-level`` keeps evoxc's accepted spellings verbatim."""
    parser = argparse.ArgumentParser(
        prog="backend_measure.py",
        description=(
            "Measure build time, ms/generation and peak RSS for one "
            "(case, backend) pair in this process."
        ),
    )
    parser.add_argument("--backend", choices=("numpy", "iree"), default="numpy")
    parser.add_argument("--case", choices=tuple(CASES), default="xor")
    parser.add_argument("--pop", type=_positive_int, default=100)
    parser.add_argument(
        "--generations", type=_positive_int, default=20, help="timed generations"
    )
    parser.add_argument(
        "--warmup", type=_non_negative_int, default=3, help="untimed generations"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--opt-level",
        default=None,
        choices=("O0", "O1", "O2", "O3", "0", "1", "2", "3"),
        help=(
            "forwarded verbatim as compile_options={'opt_level': <value>}; evoxc "
            "normalizes/validates it and translates it per backend"
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="out.json",
        help="also write the record as JSON to this path",
    )
    args = parser.parse_args(argv)
    if args.pop < SPECIES_SIZE:
        parser.error(
            f"--pop must be at least species_size={SPECIES_SIZE} (the canonical "
            f"matrix of benchmarks/cases.py), got {args.pop}: NEAT refuses "
            "pop_size < species_size * min_species_size"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    """Run one measurement, print its table and optionally write the JSON record."""
    args = parse_args(argv)
    record = measure(args)
    print(render(record))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
