#!/usr/bin/env python3
"""Measure one ``(case, target)`` accelerator row, and run the accelerator matrix.

The standalone, one-configuration-per-process companion of the full harness
(``benchmarks/run_benchmark.py``) and the GPU sibling of
``benchmarks/backend_measure.py``: ``run`` measures exactly one row in this process,
``matrix`` spawns one such process per row strictly sequentially and writes one
record, and ``render`` re-renders the measured block of ``benchmarks/GPU.md`` from
that record with no measurement. Nothing here is wired into the old-vs-new verdict
(``RESULTS.md`` stays numpy-only; see ``## What this does not change`` in
``GPU.md``); the record contract -- schema, markers, tolerances, the environment
fingerprint and every rendered line -- lives in ``benchmarks/gpu_report.py``.

Targets
-------
A *target* is a named way to build the ``Pipeline`` (data in
:data:`benchmarks.gpu_targets.TARGETS`): ``numpy`` -> backend numpy, no device, no
options; ``iree-llvm-cpu`` -> iree with ``target_backends=["llvm-cpu"]``;
``iree-rocm`` -> iree with ``device="rocm"`` and ``target_backends=["rocm"]``;
``xla-rocm`` -> evoxc's ``xla`` adapter with ``device="rocm"``, driving the PJRT
C API plugin named by ``--pjrt-plugin PATH`` or ``$EVOXC_PJRT_PLUGIN`` (the
``xla-rocm`` extra installs the AMD ``jax-rocm7-pjrt`` wheel, so
``uv sync --extra xla-rocm`` supplies it). Raw passthrough flags override a named
target's defaults, for hardware this file does not name: ``--device KIND`` with
``--device-index N`` (together they build ``evoxc.Device(KIND, N)``; an index alone
re-indexes the target's own device kind), ``--target-backends a,b``,
``--rocm-target ARCH`` (appends ``--iree-rocm-target=ARCH`` to the
``iree_compile_args`` compile option) and ``--iree-compile-arg FLAG`` (repeatable,
appended to that same list). They apply to **every** row of a ``matrix`` run, so a
combination that no longer pairs a loadable device with its target backends becomes
an ``error`` row instead of being repaired here: ``device`` and ``target_backends``
must stay aligned.

The xla target's plugin (``--pjrt-plugin``, forwarded by ``matrix`` to every row)
-------------------------------------------------------------------------------
The plugin path goes into ``compile_options["plugin_path"]`` **and** into
``EVOXC_PJRT_PLUGIN``, because the adapter's availability probe reads only the
variable (resolution, fingerprint and that rule live in
``benchmarks/gpu_targets.py``). Resolution order is ``--pjrt-plugin`` then
``$EVOXC_PJRT_PLUGIN``; without one an xla row is a named ``error`` row and
``matrix`` exits 2 before measuring -- never a fallback, never a made-up number.

The xla target's ``XLA_FLAGS`` workaround (``--xla-flags``)
----------------------------------------------------------
On this box an xla generation graph that holds a long ``while`` (the RL rollout)
SIGSEGVs inside ``PJRT_LoadedExecutable_Execute``: XLA captures the graph as a HIP
graph / FUSION command buffer and replays it through ``hipGraphLaunch``. XLA reads
``XLA_FLAGS`` when its backend is initialised -- before the plugin builds the
executable -- so the cure is a **process environment** setting, not a
``compile_options`` entry: :data:`gpu_targets.XLA_WORKAROUND_FLAGS`
(``--xla_gpu_enable_command_buffer=WHILE``) is exported as ``XLA_FLAGS`` for a
target of :data:`gpu_targets.PJRT_TARGETS` and for no other target, whose
environment is inherited untouched. ``matrix`` forwards ``--xla-flags`` to every
row child and applies the same rule there, and ``--xla-flags ''`` measures
**without** the workaround: the variable is then removed from the child
environment, so the A/B leg is a real absence rather than a stale inherited value.
The effective value is recorded per row (``record["xla_flags"]``) and once in the
document (``meta["xla_flags"]``, ``null`` when no planned target is an xla
target), so a number is never mis-attributed to a flag it did not run with.

Cases
-----
The canonical matrix of ``benchmarks/cases.py`` (``get_cases``), built for the new
implementation only (``case.build(tn, "new")``) and wrapped in a ``Pipeline`` with
``generation_limit=warmup+generations``, ``is_detailed=False`` and the target's
``backend``/``device``/``compile_options``. The case table is never duplicated here.

Protocol (per row, one fresh process)
-------------------------------------
``Pipeline.setup()`` is timed into ``setup_s``; the first ``Pipeline.step()`` pays
the generation graph and is reported as ``first_step_s`` and, on its own, as
``build_s`` (``Pipeline.build_seconds``) -- **the build is never charged to a
generation and that step is excluded from every timing sample**. ``--warmup``
untimed generations then ``--generations`` timed ones run with stdout/stderr
redirected to ``os.devnull`` (no logging inside the measured region); each is timed
with ``time.perf_counter`` around the call, including host materialisation, and
summarised as mean / median / min / max / samples plus ``generations_per_second``.
``peak_rss_mib`` is ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` -- a
process-lifetime high-water mark, which is why one row is one process -- and
``final_generation``, ``final_max_fitness``, the environment fingerprint,
``device_info`` and the exact ``protocol`` block are recorded.

Numerical cross-check (default on, ``--no-cross-check`` disables it)
-------------------------------------------------------------------
After the timed region, a *fresh* ``backend="numpy"`` pipeline with the same case,
seed and step count is driven in this same process -- same implementation, so "one
implementation per process" is untouched and no JAX is imported anywhere -- and two
snapshots are compared: **generation 1** (both legs' first evaluated population,
from the same seed and graph, so it must agree) and the **final state** (the
post-``tell`` population after the same number of steps plus the last step's fitness
vector). Integer keys are compared exactly, floats need an identical NaN (and, for
fitness, ``-inf``) mask plus :data:`~gpu_report.FLOAT_RTOL`/:data:`~gpu_report.FLOAT_ATOL`,
and the record carries a per-field verdict, the max absolute difference and the
mask-mismatch counts. Both legs share one PRNG, so beyond generation 1 a
float-reassociation difference can reorder a near-tied selection and legitimately
diverge the *keys*: a final-state key mismatch is search-trajectory drift, not
necessarily a kernel defect (:data:`gpu_report.TRAJECTORY_NOTE`). The ``numpy``
target records ``{"status": "skipped"}`` (it is the reference), and a cross-check
that cannot run records its traceback while the measured row keeps its timings, so
missing numerical evidence is visible instead of silently green.

Lockstep trajectory trace (``--trace N``, off by default)
--------------------------------------------------------
``--trace N`` (also forwarded by ``matrix`` to every child) re-runs the row after the
timed region and after ``peak_rss_mib`` is sampled as a lockstep pair: a fresh
accelerated pipeline and a fresh ``backend="numpy"`` pipeline of the same case and
seed, each stepped once (each pays its own build), compared for generations ``1..N``
with the cross-check rules. ``record["trajectory"]`` holds one row per generation plus
``first_key_divergence``, ``first_float_exceedance`` and the trace's wall time. It
costs ``N`` x (accelerated generation + numpy generation) on top of the timed region,
so it is off by default, and it is the reproducible form of the near-tie drift
analysis (:data:`gpu_report.LOCKSTEP_NOTE`); a failing leg records ``status="error"``.

Subcommands
-----------
``run`` measures exactly one row, prints its table and optionally writes ``--json``
(exit 0 for an ``ok`` row, 1 for an ``error`` row). ``matrix`` plans
``cases x targets``, spawns one ``run`` subprocess per row *strictly sequentially*
(one fresh process each, the same interpreter), collects each child's JSON,
synthesises an ``error`` row when a child wrote none (crash, kill, timeout,
non-JSON, carrying its returncode and stderr tail), writes one document
(``schema``, ``meta``, ``records``) and, with ``--out REPORT``, refreshes that
report's measured block; it exits 0 only when every planned row is ``ok``, 1 when
any row is ``error`` (a blocked device path is loud by construction) and 2 when the
report could not be rendered. ``render`` re-renders that block from an existing
record and replaces only the text between the report's markers, exiting 2 with a
named error on a missing marker, an unreadable document, a wrong schema or a
missing ``records`` list. ``--list`` prints the targets, the passthrough flags and
the canonical case matrix.

Usage
-----
    python3 benchmarks/gpu_measure.py --list                 # numpy-free
    uv run --extra iree python benchmarks/gpu_measure.py run \\
        --case xor_p100 --target iree-rocm --json $TMPDIR/row.json
    uv run --extra iree python benchmarks/gpu_measure.py matrix \\
        --cases xor_p100,xor_p500,xor3d_p500 --targets numpy,iree-llvm-cpu,iree-rocm \\
        --warmup 3 --generations 10 --seed 42 \\
        --json benchmarks/results/gpu_2026-11-05.json --out benchmarks/GPU.md
    uv run --extra iree python benchmarks/gpu_measure.py matrix \\
        --cases xor_p100 --targets numpy,xla-rocm --pjrt-plugin /path/to/pjrt_plugin.so \\
        --json $TMPDIR/xla.json --out $TMPDIR/xla.md
    uv run --extra iree python benchmarks/gpu_measure.py render \\
        --json benchmarks/results/gpu_2026-11-05.json --out benchmarks/GPU.md

Requires ``tensorneat`` (installed, or importable from ``../src``) and ``numpy``; the
``iree`` targets additionally need the optional ``iree`` extra
(``uv sync --extra iree``) plus, for ``iree-rocm``, a working ROCm runtime; the
``xla-rocm`` target additionally needs the ``xla-rocm`` extra
(``uv sync --extra xla-rocm``), which installs the pinned ``jax-rocm7-pjrt`` plugin
wheel for ``--pjrt-plugin``/``$EVOXC_PJRT_PLUGIN`` to name.

Notes
-----
The IREE runtime can print ``nanobind: leaked ...`` lines to stderr when the
interpreter exits; that is a binding teardown message emitted *after* the record is
written and never turns a row into an error. Nothing falls back to another backend:
a target that cannot run raises, and the raise becomes an ``error`` row.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import resource
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Repository root (``<repo>/benchmarks/gpu_measure.py``).
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
#: The src-layout package, so the CLI also works from a bare checkout.
SRC_DIR = REPO_ROOT / "src"

if str(HERE) not in sys.path:  # the record/report half is a sibling module, not a package
    sys.path.insert(0, str(HERE))
from gpu_report import (  # noqa: E402  (after the sys.path guard)
    FLOAT_ATOL,
    FLOAT_RTOL,
    LOCKSTEP_NOTE,
    SCHEMA,
    TRAJECTORY_NOTE,
    RenderError,
    compare_snapshots,
    trajectory_row,
    display_path,
    environment,
    last_error_line,
    read_document,
    render_block,
    render_row,
    snapshot,
    splice_block,
    summary_line,
)
from gpu_targets import (  # noqa: E402  (after the sys.path guard)
    NUMPY_PLAN,
    PJRT_ENV_VAR,
    PJRT_FLAG,
    PJRT_TARGETS,
    TARGETS,
    XLA_WORKAROUND_FLAGS,
    PluginError,
    fingerprint_plugin,
    plugin_for,
    require_plugin,
)

#: The protocol defaults of the accelerator matrix.
DEFAULT_CASES = "xor_p100,xor_p500,xor3d_p500"
DEFAULT_TARGETS = "numpy,iree-llvm-cpu,iree-rocm"
DEFAULT_SEED = 42
DEFAULT_WARMUP = 3
DEFAULT_GENERATIONS = 10
DEFAULT_TRACE = 0  #: generations the lockstep trajectory trace compares (0 = off)
#: Seconds one row subprocess may take before it is killed and recorded as an error.
DEFAULT_TIMEOUT = 3600
DEFAULT_JSON = HERE / "results" / "gpu.json"
DEFAULT_REPORT = HERE / "GPU.md"
DEFAULT_TITLE = "TensorNEAT accelerator measurement — numpy / iree llvm-cpu / iree rocm"


# --- Lazy imports (module scope stays numpy/evoxc/tensorneat-free) ----------------------


def _numpy() -> Any:
    """Import numpy lazily so ``--help``/``--list`` work on a bare interpreter."""
    import numpy

    return numpy


def _tensorneat() -> Any:
    """Import ``tensorneat``, falling back to the in-repo ``src`` directory."""
    try:
        import tensorneat
    except ImportError:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        import tensorneat
    return tensorneat


def _cases_module() -> Any:
    """Import the canonical case matrix (``benchmarks/cases.py``, stdlib-only)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import cases

    return cases


def _case(name: str) -> Any:
    """The registered :class:`cases.Case` for ``name`` (ValueError when unknown)."""
    return _cases_module().get_cases([name])[0]


def _case_names() -> list[str]:
    """Every canonical case name, in declaration order."""
    return [case.name for case in _cases_module().get_cases(None)]


# --- Target plans and the pipeline ------------------------------------------------------


def _split_csv(text: str | None) -> list[str]:
    """Split a comma-separated CLI value, dropping empty fields."""
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _target_plan(
    target_name: str, args: argparse.Namespace, plugin: str | None = None
) -> dict[str, Any]:
    """Resolve one named target plus the raw passthrough flags into a build plan.

    Args:
        target_name: A key of :data:`gpu_targets.TARGETS`.
        args: Parsed CLI arguments; ``device``/``device_index``/``target_backends``/
            ``rocm_target``/``iree_compile_arg`` each override the named defaults.
        plugin: The resolved PJRT plugin path (:func:`gpu_targets.plugin_for`); it
            becomes ``compile_options["plugin_path"]`` for a target of
            :data:`gpu_targets.PJRT_TARGETS` and is ignored otherwise.

    Returns:
        A JSON-safe ``{"name", "backend", "device", "device_index",
        "compile_options"}``; ``device_index`` is set only when a device is being
        materialised, so an index without a device kind is dropped.
    """
    base = TARGETS[target_name]
    compile_options = (
        None if base["compile_options"] is None else copy.deepcopy(base["compile_options"])
    )
    device = base["device"]
    if args.target_backends:
        compile_options = dict(compile_options or {})
        compile_options["target_backends"] = _split_csv(args.target_backends)
    if args.device is not None:
        device = args.device
    if args.rocm_target or args.iree_compile_arg:
        compile_options = dict(compile_options or {})
        flags = list(compile_options.get("iree_compile_args") or [])
        if args.rocm_target:
            flags.append(f"--iree-rocm-target={args.rocm_target}")
        flags.extend(args.iree_compile_arg or [])
        compile_options["iree_compile_args"] = flags
    if target_name in PJRT_TARGETS and plugin is not None:
        compile_options = dict(compile_options or {})
        compile_options["plugin_path"] = plugin
    explicit = args.device is not None or args.device_index is not None
    device_index = (0 if args.device_index is None else args.device_index) if (
        device is not None and explicit) else None
    return {
        "name": target_name,
        "backend": base["backend"],
        "device": device,
        "device_index": device_index,
        "compile_options": compile_options,
    }


def _device_value(plan: dict[str, Any]) -> Any:
    """The ``device`` value forwarded to ``Pipeline`` for ``plan``.

    ``None`` stays ``None``; a plan produced by the raw ``--device``/
    ``--device-index`` flags materialises ``evoxc.Device(kind, index)``, while a
    named target's plain kind string is forwarded verbatim (evoxc coerces it to
    ``Device(kind, 0)``).
    """
    if plan["device"] is None:
        return None
    if plan["device_index"] is None:
        return plan["device"]
    import evoxc

    return evoxc.Device(plan["device"], int(plan["device_index"]))


def _build_pipeline(
    tn: Any, case: Any, plan: dict[str, Any], args: argparse.Namespace
) -> Any:
    """Construct the ``Pipeline`` of one ``(case, target)`` row."""
    algorithm, problem = case.build(tn, "new")
    return tn.Pipeline(
        algorithm=algorithm,
        problem=problem,
        seed=args.seed,
        fitness_target=case.fitness_target,
        generation_limit=args.warmup + args.generations,
        backend=plan["backend"],
        device=_device_value(plan),
        compile_options=plan["compile_options"],
        is_detailed=False,
    )


# --- The xla ``XLA_FLAGS`` workaround (row environment, never a compile option) --------


def _effective_xla_flags(target_name: str, requested: str) -> str | None:
    """The ``XLA_FLAGS`` value this target is measured with (``None`` = leave alone).

    The ROCm command-buffer workaround (:data:`gpu_targets.XLA_WORKAROUND_FLAGS`)
    only concerns a target of :data:`gpu_targets.PJRT_TARGETS`: every other target
    keeps the inherited environment untouched, so an ``XLA_FLAGS`` value is never
    injected into a numpy/iree row. An empty ``requested`` is meaningful for an xla
    target -- it asks for the *absence* of the workaround (``--xla-flags ''``) --
    so it is returned as the empty string, not folded into ``None``.
    """
    return requested if target_name in PJRT_TARGETS else None


def _export_xla_flags(
    environment: MutableMapping[str, str], flags: str | None
) -> MutableMapping[str, str]:
    """Apply the workaround rule to ``environment`` in place and return it.

    ``None`` (a target that does not use XLA) leaves the environment byte-identical;
    a non-empty value is exported as ``XLA_FLAGS``; an empty value **removes** the
    variable, so "measured without the workaround" is a real absence rather than a
    value inherited from the parent process (XLA reads it at backend init).
    """
    if flags is None:
        return environment
    if flags:
        environment["XLA_FLAGS"] = flags
    else:
        environment.pop("XLA_FLAGS", None)
    return environment


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    """The exact protocol block recorded with every row (deterministic given ``args``)."""
    return {
        "one_configuration_per_process": True,
        "rows_strictly_sequential": True,
        "warmup_generations": args.warmup,
        "timed_generations": args.generations,
        "seed": args.seed,
        "compile_measured_separately": True,
        "build_charged_to_a_generation": False,
        "timing_clock": "time.perf_counter() wall clock around each step() call, "
        "including host materialisation of the result",
        "peak_rss": "resource.getrusage(RUSAGE_SELF).ru_maxrss in MiB -- "
        "a process-lifetime high-water mark",
        "logging_in_timed_region": False,
        "cross_check_enabled": not args.no_cross_check,
        "cross_check_reference": "numpy",
        "trajectory_generations": args.trace,
        "float_rtol": FLOAT_RTOL,
        "float_atol": FLOAT_ATOL,
        "errors": "recorded as status='error' rows carrying the traceback, never dropped",
    }


# --- The measurement --------------------------------------------------------------------


def _peak_rss_mib() -> float:
    """This process's peak resident set size in MiB (``ru_maxrss`` is KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def _numpy_reference(
    tn: Any, case: Any, args: argparse.Namespace, steps: int
) -> dict[str, Any]:
    """Drive a fresh ``backend="numpy"`` pipeline over ``steps`` generations.

    Same implementation, same case, same seed, same graph -- a numerical oracle for
    the accelerated row, not a second implementation.

    Returns:
        ``{"generation_1", "final_state", "build_s"}``: the first step's evaluated
        payload plus fitness, the state after the last step (its post-``tell``
        population) plus the last step's fitness, and the numpy build time (the
        check's one-off cost, so the check is not mistaken for free).
    """
    from tensorneat.common import to_numpy

    pipeline = _build_pipeline(tn, case, NUMPY_PLAN, args)
    state = pipeline.setup()
    state, evaluated, fitnesses = pipeline.step(state)
    generation_1 = snapshot(evaluated, fitnesses)
    for _ in range(steps - 1):
        state, evaluated, fitnesses = pipeline.step(state)
    final_state = snapshot([to_numpy(field) for field in state.pop], fitnesses)
    return {
        "generation_1": generation_1,
        "final_state": final_state,
        "build_s": float(pipeline.build_seconds),
    }


def _cross_check(
    tn: Any,
    case: Any,
    args: argparse.Namespace,
    generation_1: dict[str, Any],
    final_state: dict[str, Any],
) -> dict[str, Any]:
    """Run the numpy reference leg and compare both snapshots against it.

    A reference leg that cannot run is recorded as ``status="error"`` with its
    traceback; the measured row keeps its timings so the flops are not thrown away,
    and the missing numerical evidence stays visible in the report.
    """
    steps = 1 + args.warmup + args.generations
    base = {"reference": "numpy", "steps": steps, "rtol": FLOAT_RTOL,
            "atol": FLOAT_ATOL, "trajectory_note": TRAJECTORY_NOTE}
    try:
        reference = _numpy_reference(tn, case, args, steps)
    except Exception:
        return {**base, "status": "error", "error": traceback.format_exc()}
    first = compare_snapshots(generation_1, reference["generation_1"])
    final = compare_snapshots(final_state, reference["final_state"])
    status = "pass" if first["status"] == "pass" and final["status"] == "pass" else "fail"
    return {
        **base,
        "status": status,
        "generation_1": first,
        "final_state": final,
        "reference_build_s": round(reference["build_s"], 6),
    }


def _species_count(state: Any) -> int | None:
    """Live species count from ``state.species.member_count`` (``None`` if unreadable).

    ``NaN`` marks a dead slot and ``NaN > 0`` is False (as in ``show_details``).
    """
    try:
        from tensorneat.common import to_numpy

        return int(sum(1 for count in to_numpy(state.species.member_count) if count > 0))
    except Exception:
        return None


def _trajectory(
    tn: Any, case: Any, plan: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Per-generation lockstep trace of this row, inside the devnull redirect.

    Never raises: a failing leg is recorded as ``status="error"`` with its traceback
    so the measured row keeps its timings (see the module docstring).
    """
    steps = args.trace
    base = {"steps": steps, "rtol": FLOAT_RTOL, "atol": FLOAT_ATOL, "note": LOCKSTEP_NOTE}
    if steps <= 0:
        return {**base, "status": "skipped", "reason": "--trace 0 (off)"}
    if plan["name"] == "numpy":
        return {**base, "status": "skipped", "reason": "target is the numpy reference"}
    started = time.perf_counter()
    try:
        with (open(os.devnull, "w", encoding="utf-8") as devnull,
              contextlib.redirect_stdout(devnull),
              contextlib.redirect_stderr(devnull)):
            accelerated = _build_pipeline(tn, case, plan, args)
            reference = _build_pipeline(tn, case, NUMPY_PLAN, args)
            acc = list(accelerated.step(accelerated.setup()))  # each pays its own build
            ref = list(reference.step(reference.setup()))
            generations, first_key, first_float = [], None, None
            for generation in range(1, steps + 1):
                row, exceeds = trajectory_row(
                    generation, snapshot(acc[1], acc[2]), snapshot(ref[1], ref[2]),
                    _species_count(acc[0]), _species_count(ref[0]))
                generations.append(row)
                if first_key is None and row["node_keys_mismatches"] != 0:
                    first_key = generation
                if first_float is None and exceeds:
                    first_float = generation
                if generation == steps:
                    break
                acc, ref = list(accelerated.step(acc[0])), list(reference.step(ref[0]))
    except Exception:
        return {**base, "status": "error", "error": traceback.format_exc(),
                "trace_s": round(time.perf_counter() - started, 6)}
    return {**base, "status": "ok", "generations": generations,
            "first_key_divergence": first_key, "first_float_exceedance": first_float,
            "trace_s": round(time.perf_counter() - started, 6)}


def _measure(
    tn: Any, case: Any, plan: dict[str, Any], args: argparse.Namespace, record: dict[str, Any]
) -> None:
    """Fill ``record`` with the measured row (raises on any failure).

    Writing into the caller's dict is deliberate: an exception halfway through
    leaves every already-measured field on the ``error`` record.
    """
    from tensorneat.common import to_numpy

    pipeline = _build_pipeline(tn, case, plan, args)
    started = time.perf_counter()
    try:
        state = pipeline.setup()
    except Exception:
        record["setup_s"] = round(time.perf_counter() - started, 6)
        record["setup_failed"] = True
        raise
    record["setup_s"] = round(time.perf_counter() - started, 6)

    started = time.perf_counter()
    state, evaluated, fitnesses = pipeline.step(state)  # pays the evoxc build
    record["first_step_s"] = round(time.perf_counter() - started, 6)
    record["build_s"] = round(float(pipeline.build_seconds), 6)
    generation_1 = snapshot(evaluated, fitnesses)

    with (
        open(os.devnull, "w", encoding="utf-8") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        for _ in range(args.warmup):
            state, evaluated, fitnesses = pipeline.step(state)
        samples: list[float] = []
        for _ in range(args.generations):
            started = time.perf_counter()
            state, evaluated, fitnesses = pipeline.step(state)
            samples.append(time.perf_counter() - started)

    record["peak_rss_mib"] = round(_peak_rss_mib(), 2)
    milliseconds = [sample * 1000.0 for sample in samples]
    mean_ms = statistics.fmean(milliseconds)
    record["ms_per_generation"] = {
        "mean": round(mean_ms, 4), "median": round(statistics.median(milliseconds), 4),
        "min": round(min(milliseconds), 4), "max": round(max(milliseconds), 4),
        "samples": [round(value, 4) for value in milliseconds],
    }
    record["generations_per_second"] = round(1000.0 / mean_ms, 4) if mean_ms > 0 else None
    record["final_generation"] = int(to_numpy(state.generation))
    record["final_max_fitness"] = float(_numpy().max(fitnesses))
    final_state = snapshot([to_numpy(field) for field in state.pop], fitnesses)

    if args.no_cross_check:
        record["cross_check"] = {"status": "skipped", "reason": "--no-cross-check"}
    elif plan["name"] == "numpy":
        record["cross_check"] = {"status": "skipped", "reason": "target is the numpy reference"}
    else:
        record["cross_check"] = _cross_check(tn, case, args, generation_1, final_state)
    record["trajectory"] = _trajectory(tn, case, plan, args)  # last: after the RSS sample


def measure_row(args: argparse.Namespace) -> dict[str, Any]:
    """Measure ``args.case``/``args.target`` in this process and return the row record.

    Args:
        args: Parsed ``run`` arguments.

    Returns:
        The JSON-safe row; ``status`` is ``"ok"`` or ``"error"``, and an ``error``
        row carries ``error`` (the full traceback) plus whatever was measured before
        the failure. Never raises for a measurement failure. A target of
        :data:`gpu_targets.PJRT_TARGETS` resolves its PJRT plugin first -- the path
        goes into the plan's ``compile_options["plugin_path"]`` *and* into
        :data:`gpu_targets.PJRT_ENV_VAR`, which is what the adapter's
        ``check_available`` probe reads -- and an unresolvable plugin is a named
        ``error`` row, never a fallback to another backend.

        The row also records the ``XLA_FLAGS`` value it ran with
        (``record["xla_flags"]``; ``None`` for a target that does not use XLA) and
        exports it into :data:`os.environ` before the pipeline is built, because XLA
        reads the variable at backend init (see the module docstring's workaround
        section): the value recorded is the value that applied.
    """
    named = TARGETS[args.target]
    record: dict[str, Any] = {
        "case": args.case,
        "target": args.target,
        "backend": named["backend"],
        "device": named["device"],
        "device_index": None,
        "compile_options": copy.deepcopy(named["compile_options"]),
        "seed": args.seed,
        "warmup_generations": args.warmup,
        "timed_generations": args.generations,
        "xla_flags": _effective_xla_flags(args.target, args.xla_flags),
        "protocol": _protocol(args),
        "status": "ok",
        "error": None,
    }
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Before any backend init: XLA reads the variable when its client is created.
    _export_xla_flags(os.environ, record["xla_flags"])
    try:
        plugin = plugin_for(args.target, args.pjrt_plugin)
    except PluginError as exc:
        # Nothing was measured and nothing fell back: the message is the record.
        return {**record, "status": "error", "error": str(exc), "plugin_error": True}
    if plugin is not None:
        os.environ[PJRT_ENV_VAR] = plugin  # read by the adapter's check_available probe
        record["pjrt_plugin"] = fingerprint_plugin(plugin)
    plan = _target_plan(args.target, args, plugin)
    record.update(
        backend=plan["backend"], device=plan["device"],
        device_index=plan["device_index"], compile_options=plan["compile_options"],
    )
    record["env"] = environment()
    record["device_info"] = {
        **plan,
        "hip_devices": record["env"]["iree"]["hip_devices"],
        "hip_error": record["env"]["iree"]["hip_error"],
    }
    try:
        case = _case(args.case)
        record["case_config"] = case.as_dict()
        _measure(_tensorneat(), case, plan, args, record)
    except Exception:
        record["status"] = "error"
        record["error"] = traceback.format_exc()
    return record


def run_command(args: argparse.Namespace) -> int:
    """``run``: measure one row, print it, optionally write its JSON; exit 0/1."""
    record = measure_row(args)
    print(render_row(record))
    if record["status"] != "ok":
        print(last_error_line(record.get("error")), file=sys.stderr)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if record["status"] == "ok" else 1


# --- Matrix driver ----------------------------------------------------------------------


def _selected_cases(text: str) -> list[Any]:
    """Resolve ``--cases`` (CSV, or ``all``) to registered cases.

    Raises:
        ValueError: If the value is empty or names an unknown case.
    """
    names = _split_csv(text)
    if not names:
        raise ValueError("--cases is empty; pass at least one case name (or 'all')")
    return _cases_module().get_cases(names)


def _selected_targets(text: str) -> list[str]:
    """Resolve ``--targets`` (CSV, or ``all``) to named targets.

    Raises:
        ValueError: If the value is empty or names an unknown target.
    """
    names = _split_csv(text)
    if not names:
        raise ValueError(
            f"--targets is empty; pass at least one of {sorted(TARGETS)} (or 'all')"
        )
    if names == ["all"]:
        return list(TARGETS)
    unknown = [name for name in names if name not in TARGETS]
    if unknown:
        raise ValueError(
            f"unknown target(s) {unknown!r}; valid names: {sorted(TARGETS)} (or 'all')"
        )
    return list(dict.fromkeys(names))


def _child_command(
    args: argparse.Namespace,
    script: Path,
    case_name: str,
    target: str,
    row_path: Path,
    plugin: str | None = None,
) -> list[str]:
    """The ``run`` argv that measures one matrix row in a fresh process."""
    command = [
        sys.executable, str(script), "run",
        "--case", case_name, "--target", target,
        "--warmup", str(args.warmup), "--generations", str(args.generations),
        "--seed", str(args.seed), "--json", str(row_path),
    ]
    if plugin is not None:
        command += [PJRT_FLAG, plugin]
    if args.device is not None:
        command += ["--device", args.device]
    if args.device_index is not None:
        command += ["--device-index", str(args.device_index)]
    if args.target_backends:
        command += ["--target-backends", args.target_backends]
    if args.rocm_target:
        command += ["--rocm-target", args.rocm_target]
    for flag in args.iree_compile_arg or []:
        command += ["--iree-compile-arg", flag]
    if args.no_cross_check:
        command.append("--no-cross-check")
    if args.trace:
        command += ["--trace", str(args.trace)]
    # Forwarded explicitly so `--xla-flags ''` reaches the child as an empty value.
    # The `=` form is required: the default value itself begins with `--`, which
    # argparse would otherwise read as the next option instead of this one's value.
    command += [f"--xla-flags={args.xla_flags}"]
    return command


def _child_environment(plugin: str | None, xla_flags: str | None) -> dict[str, str]:
    """The environment of a row subprocess: this one, plus the resolved plugin.

    The child re-resolves the plugin from its own ``--pjrt-plugin`` argv, but the
    variable is exported too because ``XlaBackend.check_available`` reads it during
    backend registration and never sees the build's compile options.

    ``xla_flags`` applies the workaround rule (:func:`_effective_xla_flags`): the
    value is exported as ``XLA_FLAGS`` for an xla target, an empty value removes the
    variable, and ``None`` (a target that does not use XLA) leaves the inherited
    environment untouched.
    """
    env = dict(os.environ)
    if plugin is not None:
        env[PJRT_ENV_VAR] = plugin
    return _export_xla_flags(env, xla_flags)


def _tail(text: str | None, lines: int = 20, limit: int = 4000) -> str:
    """The last ``lines`` lines of ``text``, at most ``limit`` characters."""
    if not text:
        return ""
    return "\n".join(str(text).splitlines()[-lines:])[-limit:]


def _crash_row(
    case_name: str,
    target: str,
    command: list[str],
    reason: str,
    returncode: int | None,
    stderr_text: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Synthesise an ``error`` row for a child that wrote no record."""
    tail = _tail(stderr_text)
    parts = [reason] + (["", "--- child stderr tail ---", tail] if tail else [])
    return {
        "case": case_name, "target": target, "status": "error",
        "error": "\n".join(parts), "returncode": returncode,
        "child_command": shlex.join(command), "stderr_tail": tail,
        "xla_flags": _effective_xla_flags(target, args.xla_flags),
        "protocol": _protocol(args),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _read_child_row(
    row_path: Path,
    case_name: str,
    target: str,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Read the child's JSON record, or synthesise an error row explaining why not."""
    if not row_path.exists():
        return _crash_row(
            case_name, target, command,
            f"the child process exited {completed.returncode} without writing a record",
            completed.returncode, completed.stderr, args)
    try:
        record = json.loads(row_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _crash_row(
            case_name, target, command,
            f"the child's record at {row_path.name} could not be read: "
            f"{type(exc).__name__}: {exc}",
            completed.returncode, completed.stderr, args)
    if not isinstance(record, dict) or "status" not in record:
        return _crash_row(
            case_name, target, command,
            f"the child's record at {row_path.name} is not a row record (no 'status')",
            completed.returncode, completed.stderr, args)
    return record


def _document_meta(
    args: argparse.Namespace,
    started: datetime,
    finished: datetime,
    pjrt_plugin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``meta`` block of a matrix document (command, title/note, env, protocol).

    ``pjrt_plugin`` is the plugin fingerprint when an ``xla`` target was
    measured; the key is absent otherwise, so a record of the numpy/iree targets
    keeps the shape it always had (and renders without a plugin line).
    ``xla_flags`` is the ``XLA_FLAGS`` value the ``xla`` rows ran with, and ``None``
    when no planned target is an xla target (a numpy/iree matrix records no flag).
    """
    targets = _selected_targets(args.targets)
    meta = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "report_title": args.report_title, "report_note": args.report_note,
        "created_utc": started.isoformat(timespec="seconds"),
        "session_utc": {"start": started.isoformat(timespec="seconds"),
                        "end": finished.isoformat(timespec="seconds")},
        "protocol": _protocol(args),
        "cases": [case.name for case in _selected_cases(args.cases)],
        "targets": targets,
        "xla_flags": args.xla_flags if any(t in PJRT_TARGETS for t in targets) else None,
        "env": environment(),
    }
    if pjrt_plugin is not None:
        meta["pjrt_plugin"] = pjrt_plugin
    return meta


def matrix_command(args: argparse.Namespace) -> int:
    """``matrix``: measure every planned row, one fresh process each, sequentially."""
    try:
        cases = _selected_cases(args.cases)
        targets = _selected_targets(args.targets)
    except ValueError as exc:
        print(f"gpu_measure matrix: {exc}", file=sys.stderr)
        return 2
    # Resolve once, before any row: a missing plugin must not cost a session.
    try:
        plugin, plugin_fingerprint = require_plugin(targets, args.pjrt_plugin)
    except PluginError as exc:
        print(f"gpu_measure matrix: {exc}", file=sys.stderr)
        return 2
    plan = [(case.name, target) for case in cases for target in targets]
    script = Path(__file__).resolve()
    started = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="gpu_measure_") as tmp:
        for index, (case_name, target) in enumerate(plan, 1):
            row_path = Path(tmp) / f"row_{index:03d}.json"
            command = _child_command(args, script, case_name, target, row_path, plugin)
            # Per row, because the workaround applies to xla targets only.
            child_env = _child_environment(
                plugin, _effective_xla_flags(target, args.xla_flags))
            print(f"[{index}/{len(plan)}] {case_name} / {target}: running", flush=True)
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=args.timeout,
                    env=child_env,
                )
            except subprocess.TimeoutExpired as exc:
                record = _crash_row(
                    case_name, target, command,
                    f"the child process timed out after {args.timeout} s",
                    None, exc.stderr if isinstance(exc.stderr, str) else None, args)
            except OSError as exc:
                record = _crash_row(
                    case_name, target, command,
                    f"could not spawn the child process: {type(exc).__name__}: {exc}",
                    None, None, args)
            else:
                record = _read_child_row(
                    row_path, case_name, target, command, completed, args
                )
            records.append(record)
            print(summary_line(index, len(plan), record), flush=True)
    finished = datetime.now(timezone.utc)

    document = {
        "schema": SCHEMA,
        "meta": _document_meta(args, started, finished, plugin_fingerprint),
        "records": records,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.json}")

    blocked = [record for record in records if record.get("status") != "ok"]
    for record in blocked:
        print(
            f"BLOCKED: {record.get('case', '?')} / {record.get('target', '?')}: "
            f"{last_error_line(record.get('error'))}"
        )
    print(f"{len(records) - len(blocked)}/{len(records)} row(s) ok, {len(blocked)} error")

    if args.out is not None:
        try:
            splice_block(args.out, render_block(document, record_path=args.json))
        except RenderError as exc:
            print(f"gpu_measure matrix: {exc}", file=sys.stderr)
            return 2
        print(f"rendered the measured block of {args.out} from {args.json}")
    return 0 if not blocked else 1


# --- ``render`` and ``--list`` ----------------------------------------------------------


def render_command(args: argparse.Namespace) -> int:
    """``render``: re-render the measured block from an existing record (no measurement)."""
    try:
        document = read_document(args.json)
        splice_block(args.out, render_block(document, record_path=args.json))
    except RenderError as exc:
        print(f"gpu_measure render: {exc}", file=sys.stderr)
        return 2
    print(
        f"rendered the measured block of {args.out} from {args.json} "
        f"({len(document['records'])} record(s))"
    )
    return 0


def list_command(args: argparse.Namespace) -> int:
    """``--list``: the targets, the passthrough flags and the canonical case matrix."""
    del args
    print("targets (named label -> how the Pipeline is built):")
    for name, base in TARGETS.items():
        plugin = (f"  plugin from {PJRT_FLAG}/${PJRT_ENV_VAR}"
                  if name in PJRT_TARGETS else "")
        print(f"  {name:<16} backend={base['backend']:<6} device={base['device']!r:<8} "
              f"compile_options={base['compile_options']!r}{plugin}")
    print()
    print("passthrough overrides (apply to every matrix row, by target name):")
    print("  --pjrt-plugin PATH                 the PJRT C API plugin the xla targets drive")
    print(f"  (or export {PJRT_ENV_VAR}=PATH; the flag wins, `uv sync --extra xla-rocm`")
    print("   installs the AMD one)")
    print("  --device KIND / --device-index N   build evoxc.Device(KIND, N)")
    print("  --target-backends a,b              replace target_backends")
    print("  --rocm-target ARCH                 append --iree-rocm-target=ARCH")
    print("  --iree-compile-arg FLAG            append one raw iree-compile flag (repeatable)")
    print(f"  --xla-flags TEXT                   XLA_FLAGS for the xla targets only "
          f"(default: {XLA_WORKAROUND_FLAGS!r};")
    print("                                     '' measures without the ROCm "
          "command-buffer workaround)")
    print()
    print(f"default matrix: cases {DEFAULT_CASES}; targets {DEFAULT_TARGETS}; "
          f"seed {DEFAULT_SEED}; {DEFAULT_WARMUP} warmup + {DEFAULT_GENERATIONS} timed")
    print(f"default record: {display_path(DEFAULT_JSON)}; default report: "
          f"{display_path(DEFAULT_REPORT)}")
    print()
    print("cases (from benchmarks/cases.py):")
    for case in _cases_module().get_cases(None):
        print(
            f"  {case.name:<20} pop={case.pop_size:<5} family={case.family:<9} "
            f"problem={case.problem_config['class']:<22} "
            f"fitness_target={case.fitness_target:g} impls={' '.join(case.impls)}"
        )
    return 0


# --- CLI --------------------------------------------------------------------------------


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


def _add_measurement_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments ``run`` and ``matrix`` share: budget plus passthrough overrides."""
    parser.add_argument(
        "--warmup", type=_non_negative_int, default=DEFAULT_WARMUP,
        help="untimed generations before the timed region (default: %(default)s)")
    parser.add_argument(
        "--generations", type=_positive_int, default=DEFAULT_GENERATIONS,
        help="timed generations (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-cross-check", action="store_true",
        help="skip the numpy numerical cross-check (timing only)")
    parser.add_argument(
        "--trace", type=_non_negative_int, default=DEFAULT_TRACE, metavar="N",
        help="compare N generations in lockstep against a fresh numpy leg after the "
             "timed region (default: %(default)s = off; costs N x (accelerated + numpy "
             "generation); the reproducible form of the near-tie drift analysis)")
    parser.add_argument(
        "--device", default=None, metavar="KIND",
        help=("raw device kind (rocm, hip, cuda, ...); with --device-index it builds "
              "evoxc.Device(KIND, N) and overrides the named target's device"))
    parser.add_argument(
        "--pjrt-plugin", default=None, metavar="PATH",
        help=(f"PJRT C API plugin .so (exporting GetPjRtApi) that the xla targets "
              f"drive: it becomes compile_options['plugin_path'] and is exported as "
              f"{PJRT_ENV_VAR} for the adapter's availability probe. Resolution order: "
              f"this flag, then ${PJRT_ENV_VAR}; an xla target with neither is a named "
              "error, never a fallback to another backend"))
    parser.add_argument(
        "--device-index", type=_non_negative_int, default=None, metavar="N",
        help="device index; materialises evoxc.Device(<kind>, N)")
    parser.add_argument(
        "--target-backends", default=None, metavar="a,b",
        help="comma-separated target_backends override (default: the target's own)")
    parser.add_argument(
        "--rocm-target", default=None, metavar="ARCH",
        help="append --iree-rocm-target=ARCH to the iree_compile_args compile option")
    parser.add_argument(
        "--iree-compile-arg", action="append", default=None, metavar="FLAG",
        help="append one raw iree-compile flag to iree_compile_args (repeatable)")
    parser.add_argument(
        "--xla-flags", default=XLA_WORKAROUND_FLAGS, metavar="TEXT",
        help=(f"XLA_FLAGS exported for the xla targets only (default: "
              f"{XLA_WORKAROUND_FLAGS!r}, the ROCm command-buffer workaround: a long "
              "`while` in the RL rollout SIGSEGVs inside PJRT_LoadedExecutable_Execute "
              "under FUSION capture). It is a process-environment setting because XLA "
              "reads it at backend init, so it cannot be a compile option. Pass "
              "--xla-flags '' to measure WITHOUT the workaround: the variable is then "
              "removed from the child environment, never inherited. Targets that do "
              "not use XLA keep their environment untouched"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI (``--help``/``--list`` must work without numpy)."""
    parser = argparse.ArgumentParser(
        prog="gpu_measure.py",
        description=(
            "Measure accelerator rows (numpy / iree llvm-cpu / iree rocm / xla rocm) "
            "one configuration per process, and render the measured block of "
            "benchmarks/GPU.md."),
        epilog=(
            "Try: gpu_measure.py --list  |  gpu_measure.py run --case xor_p100 "
            "--target iree-rocm  |  gpu_measure.py run --case xor_p100 --target xla-rocm "
            "--pjrt-plugin /path/to/pjrt_plugin.so  |  gpu_measure.py matrix "
            "--out benchmarks/GPU.md  |  gpu_measure.py render --json "
            "benchmarks/results/gpu_2026-11-05.json. "
            "'matrix' exits 1 when any planned row is an error row (a blocked device "
            "path must be loud), 2 when the report could not be rendered or when a "
            "planned xla target has no PJRT plugin."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="print the targets, the overrides and the canonical case matrix")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser(
        "run", help="measure exactly one (case, target) row in this process")
    run.add_argument(
        "--case", choices=tuple(_case_names()), default=DEFAULT_CASES.split(",")[0],
        help="case name from benchmarks/cases.py")
    run.add_argument(
        "--target", choices=tuple(TARGETS), default="numpy",
        help="named accelerator target (see --list)")
    _add_measurement_arguments(run)
    run.add_argument(
        "--json", type=Path, default=None, metavar="row.json",
        help="also write this row's record as JSON")

    matrix = subparsers.add_parser(
        "matrix", help="measure every planned row sequentially, one process each")
    matrix.add_argument(
        "--cases", default=DEFAULT_CASES, metavar="a,b",
        help="comma-separated case names, or 'all' (default: %(default)s)")
    matrix.add_argument(
        "--targets", default=DEFAULT_TARGETS, metavar="a,b",
        help="comma-separated target labels, or 'all' (default: %(default)s)")
    _add_measurement_arguments(matrix)
    matrix.add_argument(
        "--timeout", type=_positive_int, default=DEFAULT_TIMEOUT, metavar="SECONDS",
        help="per-row wall-clock budget before the child is killed (default: %(default)s)")
    matrix.add_argument(
        "--json", type=Path, default=DEFAULT_JSON, metavar="out.json",
        help="matrix document to write (default: %(default)s)")
    matrix.add_argument(
        "--out", type=Path, default=None, metavar="REPORT",
        help="refresh this report's measured block after the run (e.g. benchmarks/GPU.md)")
    matrix.add_argument(
        "--report-title", default=DEFAULT_TITLE, metavar="TEXT",
        help="title stored in the record's meta (default: the accelerator matrix title)")
    matrix.add_argument(
        "--report-note", default=None, metavar="TEXT",
        help="one-line note stored in the record's meta and rendered in its provenance")

    render = subparsers.add_parser(
        "render", help="re-render the measured block from an existing record")
    render.add_argument(
        "--json", type=Path, default=DEFAULT_JSON, metavar="record.json",
        help="matrix document to render from (default: %(default)s)")
    render.add_argument(
        "--out", type=Path, default=DEFAULT_REPORT, metavar="REPORT",
        help="report whose marked block is replaced (default: %(default)s)")

    subparsers.add_parser("list", help="print the targets, the overrides and the case matrix")

    args = parser.parse_args(argv)
    if args.list:
        args.command = "list"
    if args.command is None:
        parser.error("a subcommand is required (run, matrix, render, list) — or --list")
    return args


def main(argv: list[str] | None = None) -> int:
    """Dispatch one subcommand and return its exit code."""
    args = parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "matrix":
        return matrix_command(args)
    if args.command == "render":
        return render_command(args)
    if args.command == "list":
        return list_command(args)
    raise AssertionError(f"unknown subcommand {args.command!r}")  # argparse guarantees this


if __name__ == "__main__":
    raise SystemExit(main())
