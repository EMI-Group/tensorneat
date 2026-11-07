#!/usr/bin/env python3
"""Measure the **frozen pre-rewrite JAX implementation** per generation on the device.

The accelerator report (``benchmarks/GPU.md``, block source
``results/gpu_rocm_xla_2026-11-07.json``) compares the new evoxc engine against the
``numpy`` reference interpreter *on* the GPU. The headline it cannot answer is the one
this file measures: **new evoxc (``xla`` or ``iree``) against the old JAX version on the
same physical device**, per generation. ``benchmarks/legacy_jax``
(``import tensorneat_legacy``) is the old side; this box serves it on the
``AMD Instinct MI300X VF`` (gfx942) through the ``jax-rocm7-pjrt`` PJRT plugin, and a row
here is that implementation's compiled generation, timed exactly as the CPU harness times
it.

The measurement is **not** re-implemented: one row is one
``benchmarks/harness/worker.py`` subprocess with ``--case <case> --impl old --backend jax
--repeat 0 --seed <s> --generations <n> --warmup <m> --threads 0 --out <row.json>``,
which already *is* the ``tensorneat-bench/1`` protocol ``GPU.md`` uses --
``jax.jit(pipeline.step).lower(state).compile()`` timed separately, ``--warmup`` untimed
generations, then per-generation ``time.perf_counter()`` around the whole call
*including* the ``jax.device_get`` host sync, median/mean/min/max, ``ru_maxrss``,
``loadavg_start``/``loadavg_end`` and the fitness curve. This module lifts that record
into its own row shape and adds the one thing the worker cannot know: proof that the
device was reached.

Device verification
-------------------
``jax.devices()`` is only evidence when it comes from a process that imported JAX, and
this tool never imports JAX itself (module scope stays stdlib-only, so ``--list`` and
``--help`` run on an interpreter without numpy). Every row therefore spawns a **separate
tiny probe process** (``python -c <probe>``) with the row's exact environment, reporting
``jax.__version__``, ``jaxlib.__version__``, ``jax.default_backend()``,
``[str(d) for d in jax.devices()]``, each device's ``device_kind``, the resolved
``xla_rocm_plugin.so``, and the environment variables the measurement pinned. A row whose
``jax.default_backend()`` is not the backend ``--platform`` promises (``rocm`` -> ``gpu``,
``cpu`` -> ``cpu``) is a named ``status="error"`` row: a silent CPU fallback would be a
fast, wrong number, so it is refused instead of measured. The plugin fingerprint comes
from :func:`gpu_targets.fingerprint_plugin`, so the JAX ROCm backend and the committed
``xla`` rows share one ``sha256`` of provenance -- the backend *is* that PJRT plugin.

Row environment (exported to the probe and the worker alike)
-----------------------------------------------------------
``JAX_PLATFORMS=<platform>`` pins the backend kind; ``XLA_FLAGS`` carries the ROCm
command-buffer workaround (:data:`gpu_targets.XLA_WORKAROUND_FLAGS`) because without it
the legacy jitted step fails with ``INTERNAL: Failed to set memcpy d2d node params:
HIP_ERROR_InvalidValue [executable_name='jit_step']`` -- XLA captures the generation as a
FUSION command buffer and cannot replay it, and ``--xla_gpu_enable_command_buffer=WHILE``
cures it (an empty value also works); and ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
allocates on demand instead of reserving 75 % of a shared 205.8 GB device up front, which
failed rows with ``HIP_ERROR_OutOfMemory``. XLA reads ``XLA_FLAGS`` at backend init, so it
is a process-environment setting and can never be a compile option; ``--xla-flags ''``
*removes* the variable, making "measured without the workaround" a real absence. Both are
recorded per row and per document, so no number is attributed to a setting it did not run
with. ``--threads 0`` is fixed and recorded, because a positive thread count makes the
worker overwrite ``XLA_FLAGS`` on the old side.

The new-side columns are never measured here
--------------------------------------------
Per case, the ``xla`` and ``iree`` per-generation medians and their ``build_s`` are read
from a committed accelerator record (``tensorneat-gpu/1``, default
``results/gpu_rocm_xla_2026-11-07.json``) at render time, and the batch-forward table
joins two ``tensorneat-forward/1`` records (``--forward-old``: the ``jax`` target on the
device, ``--forward-new``: ``results/forward.json``'s ``xla-rocm`` rows). A source
document that cannot be read is a named ``RenderError`` (exit 2, before any block is
written) and a leg with no ``ok`` row renders ``n/a`` -- never a fabricated number.

Subcommands
-----------
``run`` measures one case (exit 0 ok / 1 error); ``matrix`` plans the case list, spawns
one ``run`` per case strictly sequentially in a fresh process, writes one document
(``schema``, ``meta``, ``records``) and with ``--out REPORT`` refreshes that report's
marked block (exit 0 all ok / 1 any error / 2 unrenderable); ``render`` re-renders the
block from an existing record without measuring; ``--list`` prints the platforms, cases,
sources and defaults on a numpy-free interpreter.

    python3 benchmarks/jax_device_measure.py --list
    uv run --extra bench python benchmarks/jax_device_measure.py matrix --platform rocm \\
        --cases all --json benchmarks/results/jax_rocm_2026-11-07.json --out benchmarks/GPU.md
    uv run --extra bench python benchmarks/jax_device_measure.py render \\
        --json benchmarks/results/jax_rocm_2026-11-07.json --out benchmarks/GPU.md

The old side needs the ``bench`` extra (``jax``); the tool itself needs neither numpy nor
JAX, and every subprocess is a fresh interpreter. This file is a **documented exception to
the ~1000-line ceiling** (like ``forward_measure.py``): the measurement protocol and the
renderer that only it feeds are one deliverable, and the CLI, the record schema and the
row shape are shared by both halves, so splitting them would add a module with a single
reader.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Repository root (``<repo>/benchmarks/jax_device_measure.py``).
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
#: The harness worker every timing is delegated to (never re-implemented here).
WORKER = HERE / "harness" / "worker.py"

if str(HERE) not in sys.path:  # the sibling report/target modules are not a package
    sys.path.insert(0, str(HERE))
from gpu_report import (  # noqa: E402  (after the sys.path guard)
    RenderError,
    _git_text,
    _session_text,
    _table,
    display_path,
    last_error_line,
    splice_block,
)
from gpu_targets import (  # noqa: E402  (after the sys.path guard)
    PJRT_ENV_VAR,
    PJRT_FLAG,
    XLA_WORKAROUND_FLAGS,
    PluginError,
    fingerprint_plugin,
    resolve_plugin,
)

#: Schema of this tool's document: ``meta`` plus one record list under ``records``.
SCHEMA = "tensorneat-jax-device/1"
#: Schema of the accelerator record the new-side columns are lifted from.
NEW_SCHEMA = "tensorneat-gpu/1"
#: Schema of the two forward-pass records the batch-forward table joins.
FORWARD_SCHEMA = "tensorneat-forward/1"
#: Exact report markers whose *interior* :func:`render_block` owns.
BEGIN_MARKER = "<!-- BEGIN MEASURED (jax_device_measure.py) -->"
END_MARKER = "<!-- END MEASURED (jax_device_measure.py) -->"

#: ``--platform`` -> the ``jax.default_backend()`` value that platform must report.
PLATFORMS: dict[str, str] = {"rocm": "gpu", "cpu": "cpu"}

#: The cases the committed accelerator record covers, in its own order (data only).
DEFAULT_CASES = "xor_p100,xor_p500,xor_p1000,xor3d_p500,custom_pagie_p500,cartpole_p100"
DEFAULT_PLATFORM = "rocm"
DEFAULT_SEED, DEFAULT_WARMUP, DEFAULT_GENERATIONS = 42, 3, 20
#: Seconds one row subprocess may take before it is killed and recorded as an error.
DEFAULT_TIMEOUT = 3600
#: Seconds the device probe may take (it imports JAX and initialises the plugin).
PROBE_TIMEOUT = 600
#: The thread count every row runs with: 0 = runtime defaults, "as users run it", and the
#: only value that leaves the exported ``XLA_FLAGS`` untouched on the old side.
DEFAULT_THREADS = 0

DEFAULT_JSON = HERE / "results" / "jax_rocm.json"
DEFAULT_REPORT = HERE / "GPU.md"
#: The committed accelerator record the new-side (``xla``/``iree``) columns come from.
DEFAULT_NEW_RECORD = HERE / "results" / "gpu_rocm_xla_2026-11-07.json"
#: The forward-pass record of the frozen JAX side measured on the device.
DEFAULT_FORWARD_OLD = HERE / "results" / "forward_jax_rocm_2026-11-07.json"
#: The committed forward-pass record whose ``xla-rocm`` rows are the new side.
DEFAULT_FORWARD_NEW = HERE / "results" / "forward.json"

#: The new-side targets, in the order the per-generation table reports them.
NEW_TARGETS: tuple[str, ...] = ("xla-rocm", "iree-rocm")
#: Forward record target names: the old side is ``jax``, the new side ``xla-rocm``.
FORWARD_OLD_TARGET, FORWARD_NEW_TARGET = "jax", "xla-rocm"

#: The XLA allocator policy the old-side rows run with (see the module docstring).
JAX_PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
JAX_ALLOCATOR = "false"

DEFAULT_TITLE = "TensorNEAT accelerator comparison — frozen JAX vs evoxc on one MI300X VF"

#: The device probe: a *separate* process on purpose, because this module must never
#: import jax. It prints one JSON object as its last line; jax's chatter goes to stderr.
PROBE_SOURCE = r'''
import json
import os
import pathlib
import sys
import sysconfig
import traceback


def discover_plugin():
    """An installed PJRT plugin .so, found the way jax itself finds one (first match)."""
    bases = []
    purelib = (sysconfig.get_paths() or {}).get("purelib")
    if purelib:
        bases.append(purelib)
    bases.extend(entry for entry in sys.path if entry)
    seen = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        try:
            matches = sorted(pathlib.Path(base).glob("jax_plugins/*/*_plugin.so"))
        except OSError:
            continue
        if matches:
            return str(matches[0].resolve())
    return None


def main():
    requested = sys.argv[1] if len(sys.argv) > 1 else ""
    source = sys.argv[2] if len(sys.argv) > 2 else ""
    block = {
        "error": None,
        "python": sys.version.split()[0],
        "jax_version": None,
        "jaxlib_version": None,
        "jax_default_backend": None,
        "jax_devices": [],
        "device_kinds": [],
        "platforms_env": os.environ.get("JAX_PLATFORMS"),
        "preallocate_env": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "plugin_path": requested or None,
        "plugin_source": source or None,
    }
    if not block["plugin_path"]:
        block["plugin_path"] = discover_plugin()
        block["plugin_source"] = "jax_plugins discovery" if block["plugin_path"] else None
    try:
        import jax
        import jaxlib

        block["jax_version"] = getattr(jax, "__version__", None)
        block["jaxlib_version"] = getattr(jaxlib, "__version__", None)
        block["jax_default_backend"] = str(jax.default_backend())
        devices = list(jax.devices())
        block["jax_devices"] = [str(device) for device in devices]
        block["device_kinds"] = sorted(
            {str(getattr(device, "device_kind", "n/a")) for device in devices}
        )
    except Exception:
        block["error"] = traceback.format_exc()
    print(json.dumps(block))


if __name__ == "__main__":
    main()
'''


# --- Lazy imports (module scope stays numpy/jax/tensorneat-free) ------------------------


def _cases_module() -> Any:
    """Import the canonical case matrix (``benchmarks/cases.py``, stdlib-only)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import cases

    return cases


def _case_names() -> list[str]:
    """Every canonical case name, in declaration order."""
    return [case.name for case in _cases_module().get_cases(None)]


def _case(name: str) -> Any:
    """The registered :class:`cases.Case` for ``name`` (``ValueError`` when unknown)."""
    return _cases_module().get_cases([name])[0]


def _environment_report() -> dict[str, Any]:
    """The harness environment fingerprint (git commit/dirty, CPU, numpy, loadavg).

    Reuses :func:`harness.protocol.env_report`, so the tree under test is fingerprinted by
    the one implementation that owns that rule; a failing probe is recorded, never fatal.
    """
    try:
        from harness.protocol import env_report

        return env_report(None)
    except Exception:  # noqa: BLE001 - a degraded fingerprint must not lose a session
        return {"probe_error": traceback.format_exc(), "git_commit": None,
                "git_dirty": None, "git_dirty_paths": [],
                "python": sys.version.split()[0]}


def _select(text: str | None, table: dict[str, Any], what: str) -> list[str]:
    """Resolve a CSV / ``all`` CLI value against ``table``'s keys, keeping order."""
    names = [item.strip() for item in (text or "").split(",") if item.strip()]
    if not names:
        raise ValueError(f"{what} is empty; pass at least one of {sorted(table)} (or 'all')")
    if names == ["all"]:
        return list(table)
    unknown = [name for name in names if name not in table]
    if unknown:
        raise ValueError(f"unknown {what} {unknown!r}; valid names: {sorted(table)} (or 'all')")
    return list(dict.fromkeys(names))


def _selected_cases(text: str | None) -> list[str]:
    """The planned case names (``ValueError`` on an unknown name, naming it)."""
    return _select(text, {name: None for name in _case_names()}, "--cases")


# --- The row environment and its protocol -----------------------------------------------


def _child_environment(platform: str, xla_flags: str | None) -> dict[str, str]:
    """This process's environment plus the three settings every old-side child gets.

    ``JAX_PLATFORMS`` pins the backend kind, ``JAX_PYTHON_CLIENT_PREALLOCATE=false`` keeps
    a shared device usable, and ``xla_flags`` is exported as ``XLA_FLAGS`` -- or *removed*
    for the empty value, so "measured without the workaround" is a real absence rather
    than a value inherited from the parent.
    """
    child = dict(os.environ)
    child["JAX_PLATFORMS"] = platform
    child[JAX_PREALLOCATE_ENV] = JAX_ALLOCATOR
    if xla_flags:
        child["XLA_FLAGS"] = xla_flags
    else:
        child.pop("XLA_FLAGS", None)
    return child


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    """The exact protocol block recorded with every row (deterministic given ``args``)."""
    return {
        "implementation": "old -- the frozen pre-rewrite JAX implementation under "
        "benchmarks/legacy_jax (import tensorneat_legacy)",
        "worker": "benchmarks/harness/worker.py --impl old --backend jax, so a row is the "
        "same tensorneat-bench/1 run the CPU harness records",
        "one_configuration_per_process": True,
        "rows_strictly_sequential": True,
        "warmup_generations": args.warmup, "timed_generations": args.generations,
        "seed": args.seed, "repeat": "one repeat per row (repeat 0)",
        "threads": DEFAULT_THREADS,
        "timed_region": "one compiled generation (jax.jit(pipeline.step)), timed with "
        "time.perf_counter() around the whole call including the jax.device_get host sync",
        "compile_measured_separately": True, "compile_charged_to_a_generation": False,
        "device_verification": "a separate probe process, spawned with the row's exact "
        "environment, imports jax and reports default_backend/devices/device_kind; a row "
        "whose default_backend is not the one --platform promises is an error row",
        "row_environment": f"JAX_PLATFORMS=<platform>, {JAX_PREALLOCATE_ENV}={JAX_ALLOCATOR} "
        "and XLA_FLAGS=<the recorded workaround> are exported to every child (probe and "
        "worker); an empty --xla-flags removes XLA_FLAGS instead of inheriting it",
        "peak_rss": "resource.getrusage(RUSAGE_SELF).ru_maxrss in MiB, a process-lifetime "
        "high-water mark, which is why one row is one process",
        "logging_in_timed_region": False,
        "new_side_columns": "never re-measured here: read from a committed "
        f"{NEW_SCHEMA!r} accelerator record at render time",
        "errors": "recorded as status='error' rows carrying the traceback, never dropped",
    }


# --- The device probe -------------------------------------------------------------------


def _empty_probe_block(reason: str | None = None) -> dict[str, Any]:
    """A probe block with every key present and nothing measured."""
    return {
        "error": reason, "python": None, "jax_version": None, "jaxlib_version": None,
        "jax_default_backend": None, "jax_devices": [], "device_kinds": [],
        "platforms_env": None, "preallocate_env": None, "xla_flags": None,
        "plugin_path": None, "plugin_source": None,
    }


def _parse_probe(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """The last JSON object the probe printed, or a block carrying why there is none."""
    for line in reversed((completed.stdout or "").splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            block = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(block, dict) and "jax_default_backend" in block:
            return block
    tail = "\n".join((completed.stderr or "").splitlines()[-20:])[-4000:]
    reason = (f"the device probe exited {completed.returncode} without printing its JSON "
              "block" + (f"\n\n--- probe stderr tail ---\n{tail}" if tail else ""))
    return _empty_probe_block(reason)


def _probe_device(
    plugin_flag: str | None, environment: dict[str, str]
) -> dict[str, Any]:
    """Run the probe process and return what JAX serves there (never imports jax here).

    ``PluginError`` propagates: a path named by ``--pjrt-plugin`` or
    ``$EVOXC_PJRT_PLUGIN`` that is not a file is a configuration error, not a device
    condition, and the caller records it as a loud error row.
    """
    resolved = resolve_plugin(plugin_flag)
    source = "" if resolved is None else (PJRT_FLAG if plugin_flag else PJRT_ENV_VAR)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", PROBE_SOURCE, resolved or "", source],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT, env=environment,
        )
    except subprocess.TimeoutExpired:
        return _empty_probe_block(f"the device probe timed out after {PROBE_TIMEOUT} s")
    except OSError as exc:
        return _empty_probe_block(
            f"could not spawn the device probe: {type(exc).__name__}: {exc}")
    return _parse_probe(completed)


def _device_block(block: dict[str, Any], platform: str) -> dict[str, Any]:
    """The row's device block: the probe's facts, the platform gate and the plugin."""
    expected, actual = PLATFORMS[platform], block.get("jax_default_backend")
    plugin = None
    if block.get("plugin_path"):
        try:
            plugin = fingerprint_plugin(str(block["plugin_path"]))
        except PluginError as exc:
            plugin = {"path": str(block["plugin_path"]),
                      "basename": Path(str(block["plugin_path"])).name,
                      "size_bytes": None, "sha256": None, "error": str(exc)}
    return {
        **block, "platform": platform, "expected_backend": expected,
        "verified": bool(block.get("error") is None and actual == expected),
        "plugin": plugin,
    }


def _device_refusal(block: dict[str, Any], platform: str) -> str | None:
    """Why the device was not reached, or ``None`` when it was."""
    expected = PLATFORMS[platform]
    if block.get("error"):
        return ("the device probe failed, so this row could not be verified on the "
                f"device: {last_error_line(block['error'])}\n\n{block['error']}")
    actual = block.get("jax_default_backend")
    if actual != expected:
        return (f"jax.default_backend() is {actual!r}, not {expected!r} as --platform "
                f"{platform} promises: refusing a silent fallback (JAX_PLATFORMS="
                f"{block.get('platforms_env')!r}, devices {block.get('jax_devices')!r}, "
                f"device_kind {block.get('device_kinds')!r})")
    return None


# --- The measurement --------------------------------------------------------------------


def _worker_command(case_name: str, args: argparse.Namespace, row_path: Path) -> list[str]:
    """The harness worker argv that measures this case's old-side row."""
    return [
        sys.executable, str(WORKER), "--case", case_name, "--impl", "old",
        "--backend", "jax", "--repeat", "0", "--seed", str(args.seed),
        "--generations", str(args.generations), "--warmup", str(args.warmup),
        "--threads", str(DEFAULT_THREADS), "--out", str(row_path),
    ]


def _read_worker_row(
    row_path: Path, command: list[str], completed: subprocess.CompletedProcess[str] | None,
    reason: str | None = None,
) -> dict[str, Any]:
    """The worker's record, or a synthetic ``error`` row explaining why there is none."""
    returncode = None if completed is None else completed.returncode
    stderr_text = None if completed is None else completed.stderr
    if reason is None and not row_path.exists():
        reason = f"the worker exited {returncode} without writing a record"
    if reason is None:
        try:
            record = json.loads(row_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reason = (f"the worker's record at {row_path.name} could not be read: "
                      f"{type(exc).__name__}: {exc}")
        else:
            if isinstance(record, dict) and "status" in record:
                return record
            reason = (f"the worker's record at {row_path.name} is not a row record "
                      "(no 'status')")
    tail = "\n".join(str(stderr_text or "").splitlines()[-20:])[-4000:]
    parts = [reason] + (["", "--- worker stderr tail ---", tail] if tail else [])
    return {"status": "error", "error": "\n".join(parts),
            "child_command": shlex.join(command), "stderr_tail": tail}


def _lift(record: dict[str, Any], worker: dict[str, Any]) -> None:
    """Copy the worker's ``tensorneat-bench/1`` fields into this tool's row shape."""
    record["status"] = str(worker.get("status") or "error")
    record["error"] = worker.get("error")
    record["case_config"] = worker.get("case_config") or {}
    record["worker_env"] = worker.get("env") or {}
    record["worker_command"] = worker.get("child_command") or record.get("worker_command")
    if record["status"] != "ok":
        record["ms_per_generation"] = None
        return
    record["compile_s"] = worker.get("compile_time_s")
    record["generations_consumed_before_timing"] = worker.get(
        "generations_consumed_before_timing")
    record["ms_per_generation"] = {
        "mean": worker.get("ms_per_generation_mean"),
        "median": worker.get("ms_per_generation"),
        "min": worker.get("ms_per_generation_min"),
        "max": worker.get("ms_per_generation_max"),
        "samples": worker.get("generation_ms") or [],
    }
    record["gens_per_second"] = worker.get("gens_per_second")
    record["peak_rss_mib"] = worker.get("rss_final_mb")
    record["loadavg_start"] = worker.get("loadavg_start")
    record["loadavg_end"] = worker.get("loadavg_end")
    record["final_max_fitness"] = (worker.get("quality") or {}).get("final_best_fitness")


def measure_row(args: argparse.Namespace) -> dict[str, Any]:
    """Measure one case on the device: probe, then one worker process; never raise."""
    record: dict[str, Any] = {
        "case": args.case, "impl": "old", "backend": "jax", "platform": args.platform,
        "seed": args.seed, "warmup": args.warmup, "timed": args.generations,
        "threads": DEFAULT_THREADS, "status": "error", "error": None,
        "xla_flags": args.xla_flags,
        "jax_allocator": f"{JAX_PREALLOCATE_ENV}={JAX_ALLOCATOR}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    environment = _child_environment(args.platform, args.xla_flags)
    try:
        _case(args.case)  # fail loudly on an unknown case before spawning anything
        block = _probe_device(args.pjrt_plugin, environment)
        record["device"] = _device_block(block, args.platform)
        refusal = _device_refusal(block, args.platform)
        if refusal is not None:
            record["error"] = refusal
            return record
        with tempfile.TemporaryDirectory(prefix="jax_device_measure_") as tmp:
            row_path = Path(tmp) / "worker.json"
            command = _worker_command(args.case, args, row_path)
            record["worker_command"] = shlex.join(command)
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=args.timeout,
                    env=environment)
            except subprocess.TimeoutExpired:
                worker = _read_worker_row(
                    row_path, command, None,
                    f"the worker timed out after {args.timeout} s")
            except OSError as exc:
                worker = _read_worker_row(
                    row_path, command, None,
                    f"could not spawn the worker process: {type(exc).__name__}: {exc}")
            else:
                worker = _read_worker_row(row_path, command, completed)
        _lift(record, worker)
    except PluginError as exc:
        record["error"] = f"{exc}\n\n{traceback.format_exc()}"
    except Exception:
        record["error"] = traceback.format_exc()
    return record


def _fmt(value: Any, digits: int = 3) -> str:
    """Format one number, ``n/a`` when it is absent."""
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def render_row(record: dict[str, Any]) -> str:
    """Render one row record as the aligned table ``run`` prints."""
    head = (f"jax_device_measure: {record.get('case', '?')} / impl old (frozen JAX) / "
            f"platform {record.get('platform')!r} / seed {record.get('seed')}")
    if record.get("status") != "ok":
        device = record.get("device") or {}
        return "\n".join([
            head, "", "status: error",
            str(record.get("error") or "(no error recorded)").rstrip(),
            "", f"default_backend: {device.get('jax_default_backend')!r}; devices "
            f"{device.get('jax_devices')!r}"])
    timing, device = record.get("ms_per_generation") or {}, record.get("device") or {}
    cells = [
        ("compile (jax.jit ... .compile())", f"{_fmt(record.get('compile_s'))} s"),
        ("warmup / timed generations", f"{record.get('warmup')} / {record.get('timed')}"),
        ("ms/generation mean / median",
         f"{_fmt(timing.get('mean'))} / {_fmt(timing.get('median'))} ms"),
        ("ms/generation min / max",
         f"{_fmt(timing.get('min'))} / {_fmt(timing.get('max'))} ms"),
        ("generations/second (from median)", f"{_fmt(record.get('gens_per_second'))} /s"),
        ("final max fitness", _fmt(record.get("final_max_fitness"), 6)),
        ("peak RSS (process high-water)", f"{_fmt(record.get('peak_rss_mib'), 2)} MiB"),
        ("jax default_backend / device_kind",
         f"{device.get('jax_default_backend')!r} / "
         f"{', '.join(device.get('device_kinds') or []) or 'n/a'}"),
        ("jax devices", f"{device.get('jax_devices')!r}"),
        ("jax / jaxlib",
         f"{device.get('jax_version') or 'n/a'} / {device.get('jaxlib_version') or 'n/a'}"),
        ("JAX_PLATFORMS / XLA_FLAGS",
         f"{device.get('platforms_env')!r} / {device.get('xla_flags')!r}"),
        ("allocator policy", str(record.get("jax_allocator"))),
        ("PJRT plugin", f"{(device.get('plugin') or {}).get('basename')!r} sha256 "
                        f"{(device.get('plugin') or {}).get('sha256')!r}"),
    ]
    return "\n".join(
        [head, "", f"{'metric':<38}{'value':>32}", f"{'-' * 38} {'-' * 32:>32}"]
        + [f"{label:<38}{value:>32}" for label, value in cells]
        + ["", "compile is one-off and reported separately; peak RSS is a process-lifetime "
              "high-water mark (one worker process per row)."])


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


# --- The source records (read at render time, never re-measured) -------------------------


def _resolve_input(path: Path | str) -> Path:
    """A source path as given, or the same path under the repository root.

    A document stores its sources repo-relative (so a report never quotes another
    worktree's absolute path), which only resolves from the repository root; this makes a
    render from elsewhere read the same file.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    fallback = REPO_ROOT / candidate
    return fallback if fallback.exists() else candidate


def _read_source(path: Path | str, schema: str, what: str) -> dict[str, Any]:
    """Read one source document, naming the file and the reason it is unusable."""
    try:
        document = json.loads(_resolve_input(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RenderError(
            f"cannot read the {what} {display_path(path)}: {exc} — this block is not "
            "fabricated from a missing record") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(
            f"the {what} {display_path(path)} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != schema:
        raise RenderError(
            f"the {what} {display_path(path)} is not a {schema!r} document (schema "
            f"{(document or {}).get('schema') if isinstance(document, dict) else None!r})")
    if not isinstance(document.get("records"), list):
        raise RenderError(f"the {what} {display_path(path)} carries no 'records' list")
    return document


def _source_block(path: Path | str, document: dict[str, Any]) -> dict[str, Any]:
    """``{path, git_commit, session_utc}`` for the record one column set comes from.

    ``path`` is repo-relative when the record sits in this checkout, so the rendered
    provenance names a path a reader can open instead of a measuring worktree's absolute
    one; :func:`_resolve_input` reads it back from any working directory.
    """
    meta = document.get("meta") or {}
    return {"path": display_path(path), "git_commit": (meta.get("env") or {}).get("git_commit"),
            "session_utc": meta.get("session_utc")}


def _resolve_paths(
    new_record: Path | None, forward_old: Path | None, forward_new: Path | None,
    meta: dict[str, Any],
) -> dict[str, Path]:
    """The three source paths: the flag, else the record's stored path, else the default.

    A render handed no flags reproduces the stored paths, so a committed record re-renders
    byte-for-byte; the flags win when a caller points at newer sources.
    """
    stored_forward = meta.get("forward_sources") or {}
    return {
        "new_record": Path(new_record or (meta.get("new_source") or {}).get("path")
                           or DEFAULT_NEW_RECORD),
        "forward_old": Path(forward_old or (stored_forward.get("old") or {}).get("path")
                            or DEFAULT_FORWARD_OLD),
        "forward_new": Path(forward_new or (stored_forward.get("new") or {}).get("path")
                            or DEFAULT_FORWARD_NEW),
    }


def _read_sources(paths: dict[str, Path]) -> dict[str, Any]:
    """Read the three source records this block joins, or raise a named ``RenderError``."""
    new_document = _read_source(paths["new_record"], NEW_SCHEMA, "accelerator record")
    forward_old = _read_source(paths["forward_old"], FORWARD_SCHEMA, "forward record (old side)")
    forward_new = _read_source(paths["forward_new"], FORWARD_SCHEMA, "forward record (new side)")
    return {
        "new": new_document, "forward_old": forward_old, "forward_new": forward_new,
        "blocks": {
            "new_source": _source_block(paths["new_record"], new_document),
            "forward_sources": {
                "old": _source_block(paths["forward_old"], forward_old),
                "new": _source_block(paths["forward_new"], forward_new),
            },
        },
    }


# --- The document -----------------------------------------------------------------------


def validate_document(document: Any) -> None:
    """Reject a record the renderer cannot read, with a named error (``RenderError``)."""
    if not isinstance(document, dict):
        raise RenderError("the record is not a JSON object")
    if document.get("schema") != SCHEMA:
        raise RenderError(f"unsupported record schema {document.get('schema')!r}; this "
                          f"renderer reads {SCHEMA!r}")
    if not isinstance(document.get("records"), list):
        raise RenderError("the record carries no 'records' list")


def read_document(path: Path) -> dict[str, Any]:
    """Load and validate this tool's document (``RenderError`` on anything unusable)."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RenderError(f"cannot read the record {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(f"the record {path} is not valid JSON: {exc}") from exc
    validate_document(document)
    return document


def _records(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Every row record of a document, in stored order."""
    return [record for record in document.get("records") or [] if isinstance(record, dict)]


def _cases_of(document: dict[str, Any]) -> list[str]:
    """The planned cases of a document (``meta.cases``, else the record order)."""
    cases = (document.get("meta") or {}).get("cases")
    if isinstance(cases, list) and cases:
        return [str(case) for case in cases]
    return list(dict.fromkeys(str(record["case"]) for record in _records(document)
                              if record.get("case")))


def _find(
    document: dict[str, Any], name: str, target: str | None = None, key: str = "case"
) -> dict[str, Any] | None:
    """The row of ``(name, target)``; ``target=None`` matches this tool's own rows."""
    for record in _records(document):
        if record.get(key) != name:
            continue
        if target is None or record.get("target") == target:
            return record
    return None


def _median_ms(record: dict[str, Any] | None) -> float | None:
    """One row's median ms per generation, ``None`` unless the row is ``ok``."""
    if not record or record.get("status") != "ok":
        return None
    value = (record.get("ms_per_generation") or {}).get("median")
    return float(value) if isinstance(value, (int, float)) else None


def _new_median(new_document: dict[str, Any], case: str, target: str) -> float | None:
    """One new-side (accelerator record) median ms per generation, ``None`` if not ``ok``."""
    return _median_ms(_find(new_document, case, target))


def _spread_ms(record: dict[str, Any] | None) -> str:
    """One row's ``min – max`` per-generation range, ``n/a`` without timings."""
    timing = (record or {}).get("ms_per_generation") or {}
    low, high = timing.get("min"), timing.get("max")
    if (record or {}).get("status") != "ok" or not isinstance(low, (int, float)) \
            or not isinstance(high, (int, float)):
        return "n/a"
    return f"{low:.3f} – {high:.3f}"


def _speedup(baseline_ms: float | None, new_ms: float | None) -> str:
    """``baseline / new`` as ``N.NNx`` (``n/a`` when either leg has no timing)."""
    if baseline_ms is None or new_ms is None or new_ms <= 0:
        return "n/a"
    return f"{baseline_ms / new_ms:.2f}x"


def _forward_ms(record: dict[str, Any] | None) -> float | None:
    """One forward row's median ms per pass, ``None`` unless the row is ``ok``."""
    if not record or record.get("status") != "ok":
        return None
    value = (record.get("ms_per_forward") or {}).get("median")
    return float(value) if isinstance(value, (int, float)) else None


def _forward_workloads(old: dict[str, Any]) -> list[str]:
    """The workloads of the old-side forward record, in its own order."""
    workloads = (old.get("meta") or {}).get("workloads")
    if isinstance(workloads, list) and workloads:
        return [str(name) for name in workloads]
    return list(dict.fromkeys(str(record["workload"]) for record in _records(old)
                              if record.get("workload")))


def _document_device(document: dict[str, Any]) -> dict[str, Any]:
    """The session's device block: the first row that carries one (``{}`` if none)."""
    for record in _records(document):
        device = record.get("device")
        if isinstance(device, dict) and (device.get("jax_version") or device.get("error")):
            return device
    return {}


# --- The rendered block -----------------------------------------------------------------


def _ratio_pairs(
    document: dict[str, Any], new_document: dict[str, Any], target: str
) -> list[tuple[str, float]]:
    """``(case, old / new)`` for every case where both legs have a timing."""
    pairs = []
    for case in _cases_of(document):
        old = _median_ms(_find(document, case))
        new = _new_median(new_document, case, target)
        if old is None or new is None or new <= 0:
            continue
        pairs.append((case, old / new))
    return pairs


def _verdict_line(document: dict[str, Any], new_document: dict[str, Any]) -> str:
    """The block's one-line verdict, derived from the two documents' medians."""
    clauses, slower, counts = [], [], []
    for target in NEW_TARGETS:
        pairs = _ratio_pairs(document, new_document, target)
        if not pairs:
            clauses.append(f"`{target}` n/a (no case measured both legs)")
            continue
        counts.append(len(pairs))
        losses = [(case, ratio) for case, ratio in pairs if ratio <= 1.0]
        slower += [f"`{case}` (`{target}` {ratio:.2f}x)" for case, ratio in losses]
        low, high = min(r for _, r in pairs), max(r for _, r in pairs)
        span = f"{low:.2f}x" if abs(high - low) < 5e-3 else f"{low:.2f}x–{high:.2f}x"
        scope = "" if not losses else f" on {len(pairs) - len(losses)} of {len(pairs)} cases"
        clauses.append(f"`{target}` by {span}{scope}")
    compared = min(counts) if counts else 0
    if not compared:  # nothing comparable at all: say so instead of claiming a win
        return ("**Verdict: no case in this record has an `ok` old-side row that the "
                "accelerator record also measured, so no same-device speed-up can be "
                "computed — see `### Blocked rows`.**")
    cases = f"{compared} compared case{'' if compared == 1 else 's'}"
    faster = (f"is faster per generation than the frozen JAX implementation on the same "
              f"device on all {cases}" if not slower else
              "is faster per generation than the frozen JAX implementation on the same "
              "device on most compared cases")
    tail = "" if not slower else f", and slower on {', '.join(slower)}"
    return (f"**Verdict: the new evoxc engine {faster} — " + ", ".join(clauses)
            + f"{tail} (old ÷ new, medians of one repeat per row).**")


def _verdict_section(document: dict[str, Any], new_document: dict[str, Any]) -> list[str]:
    """Section 1: the same-device verdict plus what each column set is."""
    return [
        "## Frozen JAX on the device — old vs new per generation", "",
        _verdict_line(document, new_document),
        "The old side is `benchmarks/legacy_jax` (`import tensorneat_legacy`) — the frozen "
        "pre-rewrite implementation — compiled with `jax.jit(pipeline.step).lower(state)"
        ".compile()` and measured by one `benchmarks/harness/worker.py` process per row, "
        "exactly as the CPU matrix measures it.",
        "The new side is the committed accelerator record's own `xla` and `iree` rows on "
        "the same device, read at render time and never re-measured, so each speed-up is a "
        "per-generation ratio of two sessions on one physical device.",
    ]


def _per_generation_section(
    document: dict[str, Any], new_document: dict[str, Any]
) -> list[str]:
    """Section 2: the per-generation table over every planned case."""
    rows, missing = [], []
    for case in _cases_of(document):
        record = _find(document, case)
        if record is None:
            missing.append(case)
        old = _median_ms(record)
        xla = _new_median(new_document, case, "xla-rocm")
        iree = _new_median(new_document, case, "iree-rocm")
        build = {target: (_find(new_document, case, target) or {}).get("build_s")
                 for target in NEW_TARGETS}
        rows.append([
            f"`{case}`",
            f"{old:.3f} ({_spread_ms(record)})" if old is not None else "n/a",
            _fmt(xla), _fmt(iree), _speedup(old, xla), _speedup(old, iree),
            _fmt((record or {}).get("compile_s")) if old is not None else "n/a",
            _fmt(build["xla-rocm"]), _fmt(build["iree-rocm"]),
        ])
    lines = [
        "### Per generation", "",
        "`median` is one old-side worker process's median over its timed generations and "
        "`min – max` that row's own per-generation spread — the uncertainty measure here, "
        "because a single median is not a distribution.",
        "The `xla` and `iree` columns are the same case's medians in the accelerator record "
        "this block names under `### Provenance`; their `build s` is that record's one-off "
        "compile, reported beside the timing and never charged to a generation.",
        "`n/a` means that leg has no `ok` row (see `### Blocked rows`); it is never filled "
        "in from another session.", "",
        *_table(["Case", "old JAX ms/gen (median, min–max)", "`xla` ms/gen",
                 "`iree` ms/gen", "old ÷ `xla`", "old ÷ `iree`", "old JAX compile s",
                 "`xla` build s", "`iree` build s"], rows),
    ]
    if missing:
        lines += ["", "Planned but absent from this record (a defect, never measured): "
                      + ", ".join(f"`{case}`" for case in missing) + "."]
    return lines


def _forward_section(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Section 3: the batch forward pass, joining the two forward records."""
    rows, missing = [], []
    for workload in _forward_workloads(old):
        base = _find(old, workload, FORWARD_OLD_TARGET, key="workload")
        target = _find(new, workload, FORWARD_NEW_TARGET, key="workload")
        old_ms, new_ms = _forward_ms(base), _forward_ms(target)
        if target is None:
            missing.append(workload)
        setup, build = (base or {}).get("setup_s"), (base or {}).get("build_s")
        setup_text = (f"{setup:.3f} / {build:.3f}"
                      if isinstance(setup, (int, float)) and isinstance(build, (int, float))
                      else "n/a")
        rows.append([
            f"`{workload}`", str((base or target or {}).get("rows", "n/a")),
            _fmt(old_ms), _fmt(new_ms), _speedup(old_ms, new_ms), setup_text,
            _fmt((target or {}).get("build_s")),
        ])
    lines = [
        "### Batch forward", "",
        "One forward pass of a whole population of differently-shaped networks: the RL "
        "rollout pays this once per timestep, which is why it is reported beside the "
        "generation and not folded into it.",
        "Both columns are `forward_measure.py` records joined row for row — the old side's "
        "`jax` target and the new side's `xla-rocm` target, medians over that row's timed "
        "passes — and nothing here is re-measured or re-derived (see `### Provenance`).",
        "`setup / build` is the JAX side's one-off cost, reported beside the timing and "
        "never charged to a pass; `n/a` means a leg has no `ok` row for that workload.", "",
        *_table(["Workload", "rows", "old JAX ms/pass", "`xla` ms/pass", "speed-up",
                 "jax setup / build s", "`xla` build s"], rows),
    ]
    if missing:
        lines += ["", f"The new-side forward record has no `{FORWARD_NEW_TARGET}` row for "
                      + ", ".join(f"`{workload}`" for workload in missing) + "."]
    return lines


def _device_text(device: dict[str, Any]) -> str:
    """The probe's device strings and kinds as one cell (``n/a`` when unprobed)."""
    devices = device.get("jax_devices") or []
    kinds = device.get("device_kinds") or []
    if not devices and not kinds:
        return "n/a"
    return (f"{', '.join(f'`{name}`' for name in devices) or 'n/a'}; device_kind "
            f"{', '.join(f'`{kind}`' for kind in kinds) or 'n/a'}")


def _plugin_text(plugin: dict[str, Any] | None) -> str:
    """A PJRT plugin fingerprint as one cell: basename, size, 12-hex digest."""
    if not plugin:
        return "n/a (no installed `jax_plugins/*/*_plugin.so` was found)"
    size, digest = plugin.get("size_bytes"), plugin.get("sha256")
    mib = f"{size / (1024 * 1024):.1f} MiB" if isinstance(size, (int, float)) else "n/a"
    name = plugin.get("basename") or plugin.get("path") or "n/a"
    return f"`{name}` ({mib}, sha256 `{digest[:12] if digest else 'n/a'}…`)"


def _xla_flags_text(document: dict[str, Any]) -> str:
    """The ``XLA_FLAGS`` the old-side rows ran with -- never left implicit.

    A key present but empty means the session measured *without* the workaround
    (``--xla-flags ''``), which removes the variable instead of inheriting it.
    """
    meta = document.get("meta") or {}
    if "xla_flags" not in meta:
        return "not recorded"
    flags = meta.get("xla_flags")
    if isinstance(flags, str) and flags:
        return (f"`{flags}` (ROCm command-buffer workaround: without it the legacy jitted "
                "step fails with `HIP_ERROR_InvalidValue`)")
    return "not set (`--xla-flags ''`: measured WITHOUT the command-buffer workaround)"


def _loadavg_text(document: dict[str, Any]) -> str:
    """The ``min – max`` first load-average sample over the measured rows."""
    starts = [record["loadavg_start"][0] for record in _records(document)
              if isinstance(record.get("loadavg_start"), list) and record["loadavg_start"]]
    if not starts:
        return "n/a (not recorded)"
    return f"{min(starts):.2f} – {max(starts):.2f} (1-minute sample at each row's start)"


def _environment_section(document: dict[str, Any]) -> list[str]:
    """Section 4: the measured environment of the old side, as a two-column table."""
    meta = document.get("meta") or {}
    env = meta.get("env") or {}
    device = _document_device(document) or meta.get("device") or {}
    rows = [
        ["`--platform` → `jax.default_backend()`",
         f"`{meta.get('platform')!r}` → `{device.get('jax_default_backend')!r}` (expected "
         f"`{device.get('expected_backend')!r}`, verified "
         f"{'yes' if device.get('verified') else 'no'})"],
        ["Device(s) / `device_kind`", _device_text(device)],
        ["JAX / jaxlib",
         f"{meta.get('jax_version') or device.get('jax_version') or 'n/a'} / "
         f"{meta.get('jaxlib_version') or device.get('jaxlib_version') or 'n/a'}"],
        ["`JAX_PLATFORMS` (as the rows saw it)", f"`{device.get('platforms_env')!r}`"],
        ["`XLA_FLAGS`", _xla_flags_text(document)],
        ["XLA allocator policy",
         f"`{JAX_PREALLOCATE_ENV}={JAX_ALLOCATOR}` — allocate on demand instead of reserving "
         "75 % of a shared device up front, which OOMed rows"],
        ["PJRT plugin (the jax ROCm backend *is* this plugin)",
         _plugin_text(meta.get("pjrt_plugin") or device.get("plugin"))],
        ["Load average (start of each row)", _loadavg_text(document)],
        ["Python / numpy", f"{env.get('python') or 'n/a'} / {env.get('numpy') or 'n/a'}"],
        ["CPU cores / platform",
         f"{env.get('cpu_count') or 'n/a'} / {env.get('platform') or 'n/a'}"],
        ["Git commit (code under test)", _git_text(env)],
        ["Session (UTC)", _session_text(meta.get("session_utc"))],
    ]
    return ["### Environment", "",
            "Measured conditions, not a verdict: these numbers are comparable with another "
            "session on the same device and tree, and with nothing else.", "",
            *_table(["Item", "Value"], rows)]


def _provenance_section(
    document: dict[str, Any], sources: dict[str, Any], record_path: Path | str | None
) -> list[str]:
    """Section 5: which record owns each column, and the protocol behind it."""
    meta = document.get("meta") or {}
    env = meta.get("env") or {}
    records = _records(document)
    n_ok = sum(1 for record in records if record.get("status") == "ok")
    blocks = sources["blocks"]
    new_source, forward = blocks["new_source"], blocks["forward_sources"]
    protocol = meta.get("protocol") or {}

    def _origin(block: dict[str, Any]) -> str:
        """One source record as ``path at git X (session Y)``."""
        commit = block.get("git_commit")
        return (f"`{display_path(block.get('path'))}` at git "
                f"`{commit[:12] if commit else 'n/a'}` (session "
                f"{_session_text(block.get('session_utc'))})")

    lines = [
        "### Provenance", "",
        f"Old-side rows (this block's `old JAX` and `old ÷ …` columns): "
        f"`{display_path(record_path)}` — {len(records)} row(s), {n_ok} ok and "
        f"{len(records) - n_ok} error — measured at git {_git_text(env)} in the session "
        f"{_session_text(meta.get('session_utc'))}.",
        f"New-side columns (`xla`, `iree`, their `build s`): {_origin(new_source)}, schema "
        f"`{NEW_SCHEMA}` — read at render time, never re-measured here and never hard-coded.",
        f"Batch-forward rows: {_origin(forward['old'])} (target `{FORWARD_OLD_TARGET}`) "
        f"joined with {_origin(forward['new'])} (target `{FORWARD_NEW_TARGET}`), both schema "
        f"`{FORWARD_SCHEMA}`.",
        f"Protocol: seed {protocol.get('seed', 'n/a')}, "
        f"{protocol.get('warmup_generations', 'n/a')} warmup + "
        f"{protocol.get('timed_generations', 'n/a')} timed generations, one fresh process "
        "per row, one repeat per row.",
        "The one-off compile (`jax.jit(pipeline.step).lower(state).compile()`) is timed "
        "separately and never charged to a generation, and every quoted millisecond is a "
        "median, not a mean.",
        "Nothing here is a statement about HEAD: every number belongs to the record, tree "
        "and session it names.",
    ]
    if meta.get("report_note"):
        lines.append(str(meta["report_note"]))
    if env.get("probe_error"):
        lines.append("The environment fingerprint degraded: "
                     f"{last_error_line(env['probe_error'])} (the measurement ran).")
    return lines


def _comparability_section(document: dict[str, Any]) -> list[str]:
    """Section 6: what makes the ratios meaningful and what they cannot claim."""
    new_path = (document.get("meta") or {}).get("new_source", {}).get("path")
    return [
        "### What is and is not comparable", "",
        "- Same physical device: both legs run on the accelerator this block names, and a "
        "row whose `jax.default_backend()` did not match `--platform` is an error row "
        "rather than a fast CPU number.",
        "- Same workload shape: one seed, the same generation count and the same "
        "`benchmarks/cases.py` case configuration on both sides.",
        "- Not comparable in values: the two implementations draw different PRNGs (JAX "
        "`PRNGKey` versus evoxc splitmix64), so their search trajectories differ by design "
        "and fitness is never compared here (`test/parity/` owns numerical fidelity).",
        f"- Not a lockstep comparison: the new-side columns come from `{display_path(new_path)}`"
        " — an earlier session on the same device at a different tree — so each ratio is a "
        "per-generation ratio of two sessions, not one interleaved run.",
        "- One repeat per row: the per-row `min – max` is the uncertainty measure and the "
        "medians are point estimates, so a small ratio difference is noise, not a result.",
    ]


def _blocked_section(document: dict[str, Any]) -> list[str]:
    """Section 7: every row that could not be measured, with its own message."""
    blocked = [record for record in _records(document) if record.get("status") != "ok"]
    if not blocked:
        return ["### Blocked rows", "",
                "None: every planned old-side row measured on the device. This section is "
                "written from the record, so it reappears the moment a leg fails."]
    lines = ["### Blocked rows", "",
             "Each row below is an explicit blocked record (`error` or `skipped`) with its own "
             "message; no number is quoted for a leg that could not run.", ""]
    for record in blocked:
        lines += [f"**`{record.get('case', '?')}`** — `{record.get('impl', '?')}` impl, "
                  f"backend `{record.get('backend', '?')}`, platform "
                  f"`{record.get('platform', '?')}`, status "
                  f"`{record.get('status', '?')}`. Last line of its error:", "",
                  f"    {last_error_line(record.get('error'))}", ""]
    return lines[:-1]


def render_block(
    document: dict[str, Any], record_path: Path | str | None = None,
    paths: dict[str, Path] | None = None,
) -> str:
    """Render the report's measured block from a document (or a path to one).

    ``paths`` names the three source records; ``None`` falls back to the paths the document
    stores, then to this module's defaults. ``RenderError`` on an invalid or unreadable
    document and on an unreadable source record — a missing accelerator or forward record
    is never papered over with an invented row.
    """
    if isinstance(document, (str, Path)):
        document = read_document(Path(document))
    validate_document(document)
    resolved = paths or _resolve_paths(None, None, None, document.get("meta") or {})
    sources = _read_sources(resolved)
    lines: list[str] = []
    for section in (_verdict_section(document, sources["new"]),
                    _per_generation_section(document, sources["new"]),
                    _forward_section(sources["forward_old"], sources["forward_new"]),
                    _environment_section(document),
                    _provenance_section(document, sources, record_path),
                    _comparability_section(document),
                    _blocked_section(document)):
        lines += section + [""]
    return "\n".join(lines).strip("\n")


# --- ``render`` and ``list`` ------------------------------------------------------------


def render_command(args: argparse.Namespace) -> int:
    """``render``: re-render the measured block from a record (no measurement, no probe)."""
    try:
        document = read_document(args.json)
        paths = _resolve_paths(args.new_record, args.forward_old, args.forward_new,
                               document.get("meta") or {})
        splice_block(args.out, render_block(document, record_path=args.json, paths=paths),
                     begin_marker=BEGIN_MARKER, end_marker=END_MARKER)
    except RenderError as exc:
        print(f"jax_device_measure render: {exc}", file=sys.stderr)
        return 2
    print(f"rendered the measured block of {args.out} from {args.json} "
          f"({len(document['records'])} record(s))")
    return 0


def list_command(args: argparse.Namespace) -> int:
    """``--list``: the platforms, the cases, the source records and the defaults."""
    del args
    print("platforms (--platform -> the jax.default_backend() that proves the device):")
    for name, expected in PLATFORMS.items():
        print(f"  {name:<6} must report jax.default_backend() == {expected!r}; anything "
              "else is a named error row (never a silent fallback)")
    print()
    print(f"old side: the frozen implementation under {display_path(HERE / 'legacy_jax')} "
          "(`import tensorneat_legacy`), one benchmarks/harness/worker.py process per row")
    print(f"every child gets JAX_PLATFORMS=<platform>, {JAX_PREALLOCATE_ENV}={JAX_ALLOCATOR} "
          f"and XLA_FLAGS={XLA_WORKAROUND_FLAGS!r} (--xla-flags '' removes it)")
    print(f"new-side columns: {display_path(DEFAULT_NEW_RECORD)} "
          f"(targets {', '.join(NEW_TARGETS)}); forward rows: "
          f"{display_path(DEFAULT_FORWARD_OLD)} (target {FORWARD_OLD_TARGET}) joined with "
          f"{display_path(DEFAULT_FORWARD_NEW)} (target {FORWARD_NEW_TARGET})")
    print()
    print(f"default matrix: cases {DEFAULT_CASES}; platform {DEFAULT_PLATFORM!r}; seed "
          f"{DEFAULT_SEED}; {DEFAULT_WARMUP} warmup + {DEFAULT_GENERATIONS} timed "
          f"generations; {DEFAULT_THREADS} thread override")
    print(f"default record: {display_path(DEFAULT_JSON)}; default report: "
          f"{display_path(DEFAULT_REPORT)}")
    print()
    print("cases (from benchmarks/cases.py; `all` expands to every case):")
    for name in _case_names():
        case = _case(name)
        print(f"  {name:<20} pop={case.pop_size:<5} target={case.fitness_target!r:<8} "
              f"impls={list(case.impls)}")
    return 0


# --- Matrix driver ----------------------------------------------------------------------


def _child_command(args: argparse.Namespace, case: str, row_path: Path) -> list[str]:
    """The ``run`` argv that measures one matrix row in a fresh process."""
    command = [
        sys.executable, str(Path(__file__).resolve()), "run", "--case", case,
        "--platform", args.platform, "--warmup", str(args.warmup),
        "--generations", str(args.generations), "--seed", str(args.seed),
        "--timeout", str(args.timeout), "--json", str(row_path),
    ]
    if args.pjrt_plugin:
        command += [PJRT_FLAG, args.pjrt_plugin]
    # Forwarded explicitly so `--xla-flags ''` reaches the child as an empty value.
    return command + [f"--xla-flags={args.xla_flags}"]


def _crash_row(
    case: str, command: list[str], reason: str, stderr_text: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Synthesise an ``error`` row for a child that wrote no record."""
    tail = "\n".join(str(stderr_text or "").splitlines()[-20:])[-4000:]
    parts = [reason] + (["", "--- child stderr tail ---", tail] if tail else [])
    return {
        "case": case, "impl": "old", "backend": "jax", "platform": args.platform,
        "seed": args.seed, "warmup": args.warmup, "timed": args.generations,
        "status": "error", "error": "\n".join(parts), "returncode": None,
        "worker_command": shlex.join(command), "stderr_tail": tail,
        "xla_flags": args.xla_flags,
        "jax_allocator": f"{JAX_PREALLOCATE_ENV}={JAX_ALLOCATOR}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _read_child_row(
    row_path: Path, case: str, command: list[str],
    completed: subprocess.CompletedProcess[str], args: argparse.Namespace,
) -> dict[str, Any]:
    """Read the child's record, or synthesise an error row explaining why there is none."""
    reason = None
    if not row_path.exists():
        reason = f"the child process exited {completed.returncode} without writing a record"
    else:
        try:
            record = json.loads(row_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reason = (f"the child's record at {row_path.name} could not be read: "
                      f"{type(exc).__name__}: {exc}")
        else:
            if isinstance(record, dict) and "status" in record:
                return record
            reason = (f"the child's record at {row_path.name} is not a row record "
                      "(no 'status')")
    return _crash_row(case, command, reason, completed.stderr, args)


def _summary_line(index: int, total: int, record: dict[str, Any]) -> str:
    """One-line per-row summary printed by ``matrix``."""
    head = (f"[{index}/{total}] {record.get('case', '?')} / old (frozen JAX) / "
            f"{record.get('platform', '?')}: {str(record.get('status', 'error')).upper()}")
    if record.get("status") != "ok":
        return f"{head} — {last_error_line(record.get('error'))}"
    timing, device = record.get("ms_per_generation") or {}, record.get("device") or {}
    return (f"{head} — compile {_fmt(record.get('compile_s'))} s, "
            f"{_fmt(timing.get('median'))} ms/gen, on "
            f"{', '.join(device.get('device_kinds') or []) or 'device n/a'}")


def _document_meta(
    args: argparse.Namespace, session: dict[str, str], cases: list[str],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """The ``meta`` block of a matrix document (command, sources, env, protocol)."""
    return {
        "command": shlex.join([sys.executable, *sys.argv]),
        "report_title": args.report_title, "report_note": args.report_note,
        "created_utc": session["start"], "session_utc": session,
        "protocol": _protocol(args),
        "platform": args.platform, "xla_flags": args.xla_flags,
        "jax_allocator": f"{JAX_PREALLOCATE_ENV}={JAX_ALLOCATOR}",
        "cases": cases, "new_targets": list(NEW_TARGETS),
        "new_source": sources["blocks"]["new_source"],
        "forward_sources": sources["blocks"]["forward_sources"],
        "env": _environment_report(),
    }


def matrix_command(args: argparse.Namespace) -> int:
    """``matrix``: measure every planned case, one fresh process each, sequentially."""
    try:
        cases = _selected_cases(args.cases)
    except ValueError as exc:
        print(f"jax_device_measure matrix: {exc}", file=sys.stderr)
        return 2
    # Read the sources before the first row: a missing record would make the block
    # unrenderable after a whole session, so it costs nothing here instead.
    paths = _resolve_paths(args.new_record, args.forward_old, args.forward_new, {})
    try:
        sources = _read_sources(paths)
    except RenderError as exc:
        print(f"jax_device_measure matrix: {exc}", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="jax_device_measure_") as tmp:
        for index, case in enumerate(cases, 1):
            row_path = Path(tmp) / f"row_{index:03d}.json"
            command = _child_command(args, case, row_path)
            child_env = _child_environment(args.platform, args.xla_flags)
            print(f"[{index}/{len(cases)}] {case} / old (frozen JAX): running", flush=True)
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=args.timeout,
                    env=child_env)
            except subprocess.TimeoutExpired as exc:
                record = _crash_row(
                    case, command, f"the child timed out after {args.timeout} s",
                    exc.stderr if isinstance(exc.stderr, str) else None, args)
            except OSError as exc:
                record = _crash_row(
                    case, command,
                    f"could not spawn the child process: {type(exc).__name__}: {exc}",
                    None, args)
            else:
                record = _read_child_row(row_path, case, command, completed, args)
            records.append(record)
            print(_summary_line(index, len(cases), record), flush=True)
    session = {"start": started.isoformat(timespec="seconds"),
               "end": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    document: dict[str, Any] = {
        "schema": SCHEMA, "meta": _document_meta(args, session, cases, sources),
        "records": records,
    }
    device = _document_device(document)
    if device:
        document["meta"]["device"] = device
        document["meta"]["jax_version"] = device.get("jax_version")
        document["meta"]["jaxlib_version"] = device.get("jaxlib_version")
        if device.get("plugin"):
            document["meta"]["pjrt_plugin"] = device["plugin"]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.json}")

    blocked = [record for record in records if record.get("status") != "ok"]
    for record in blocked:
        print(f"BLOCKED: {record.get('case', '?')} / old (frozen JAX): "
              f"{last_error_line(record.get('error'))}")
    failures = [record for record in blocked if record.get("status") != "skipped"]
    print(f"{len(records) - len(blocked)}/{len(records)} row(s) ok, {len(blocked)} blocked "
          f"({len(failures)} error)")
    if args.out is not None:
        try:
            splice_block(args.out, render_block(document, record_path=args.json, paths=paths),
                         begin_marker=BEGIN_MARKER, end_marker=END_MARKER)
        except RenderError as exc:
            print(f"jax_device_measure matrix: {exc}", file=sys.stderr)
            return 2
        print(f"rendered the measured block of {args.out} from {args.json}")
    # A `skipped` row is a case the old side does not implement (an expected `n/a` column),
    # so only a real `error` makes the session fail.
    return 1 if failures else 0


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
    """The arguments ``run`` and ``matrix`` share."""
    parser.add_argument(
        "--platform", choices=tuple(PLATFORMS), default=DEFAULT_PLATFORM,
        help="JAX platform the old side is pinned to via JAX_PLATFORMS (default: "
             "%(default)s); the row is verified against the backend it must report "
             f"({', '.join(f'{k} -> {v}' for k, v in PLATFORMS.items())})")
    parser.add_argument(
        "--warmup", type=_non_negative_int, default=DEFAULT_WARMUP,
        help="untimed generations before the timed region (default: %(default)s)")
    parser.add_argument(
        "--generations", type=_positive_int, default=DEFAULT_GENERATIONS,
        help="timed generations (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--timeout", type=_positive_int, default=DEFAULT_TIMEOUT, metavar="SECONDS",
        help="per-row wall-clock budget before the child is killed (default: %(default)s)")
    parser.add_argument(
        "--pjrt-plugin", default=None, metavar="PATH",
        help="the PJRT C API plugin .so the jax ROCm backend loads; the row fingerprints "
             f"it for provenance. Resolution order: this flag, then ${PJRT_ENV_VAR}, then "
             "an installed jax_plugins/*/*_plugin.so (the jax ROCm backend is that plugin)")
    parser.add_argument(
        "--xla-flags", default=XLA_WORKAROUND_FLAGS, metavar="TEXT",
        help=f"XLA_FLAGS exported to every child (default: {XLA_WORKAROUND_FLAGS!r}, the "
             "ROCm command-buffer workaround: without it the legacy jitted step fails with "
             "HIP_ERROR_InvalidValue). XLA reads it at backend init, so it is a "
             "process-environment setting; --xla-flags '' removes the variable instead of "
             "inheriting it")


def _add_source_arguments(parser: argparse.ArgumentParser, *, defaults: bool) -> None:
    """The three source records: defaulted on ``matrix``, stored-value-or-default on render."""

    def _default(path: Path) -> Path | None:
        return path if defaults else None

    parser.add_argument(
        "--new-record", type=Path, default=_default(DEFAULT_NEW_RECORD), metavar="PATH",
        help=f"committed accelerator record (schema `{NEW_SCHEMA}`) whose `xla`/`iree` rows "
             f"supply the new-side columns (default: {display_path(DEFAULT_NEW_RECORD)})")
    parser.add_argument(
        "--forward-old", type=Path, default=_default(DEFAULT_FORWARD_OLD), metavar="PATH",
        help="`tensorneat-forward/1` record holding the frozen JAX side's forward rows "
             f"(target `{FORWARD_OLD_TARGET}`; default: {display_path(DEFAULT_FORWARD_OLD)})")
    parser.add_argument(
        "--forward-new", type=Path, default=_default(DEFAULT_FORWARD_NEW), metavar="PATH",
        help="`tensorneat-forward/1` record holding the new side's forward rows (target "
             f"`{FORWARD_NEW_TARGET}`; default: {display_path(DEFAULT_FORWARD_NEW)})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI (``--help``/``--list`` must work without numpy or jax)."""
    parser = argparse.ArgumentParser(
        prog="jax_device_measure.py",
        description=(
            "Measure the frozen pre-rewrite JAX implementation (benchmarks/legacy_jax) per "
            "generation on the accelerator device, and render the same-device old-vs-new "
            "block of benchmarks/GPU.md against a committed accelerator record."),
        epilog=(
            "Try: jax_device_measure.py --list  |  jax_device_measure.py run --case xor_p100 "
            "--platform rocm  |  jax_device_measure.py matrix --platform rocm --cases all "
            "--json benchmarks/results/jax_rocm_2026-11-07.json --out benchmarks/GPU.md  |  "
            "jax_device_measure.py render --json benchmarks/results/jax_rocm_2026-11-07.json "
            "--out benchmarks/GPU.md. 'matrix' exits 1 when any planned row is an error row "
            "(a CPU fallback on a rocm row is one), and 2 when a source record is missing or "
            "the report could not be rendered."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="print the platforms, the cases, the sources and the defaults")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="measure one case on the device, here")
    run.add_argument("--case", choices=tuple(_case_names()),
                     default=DEFAULT_CASES.split(",")[0],
                     help="case name from benchmarks/cases.py")
    _add_measurement_arguments(run)
    run.add_argument("--json", type=Path, default=None, metavar="row.json",
                     help="also write this row's record as JSON")

    matrix = subparsers.add_parser(
        "matrix", help="measure every planned case sequentially, one process each")
    matrix.add_argument("--cases", default=DEFAULT_CASES, metavar="a,b",
                        help="comma-separated case names, or 'all' (default: %(default)s)")
    _add_measurement_arguments(matrix)
    _add_source_arguments(matrix, defaults=True)
    matrix.add_argument("--json", type=Path, default=DEFAULT_JSON, metavar="out.json",
                        help="document to write (default: %(default)s)")
    matrix.add_argument("--out", type=Path, default=None, metavar="REPORT",
                        help="refresh this report's measured block (e.g. benchmarks/GPU.md)")
    matrix.add_argument("--report-title", default=DEFAULT_TITLE, metavar="TEXT",
                        help="title stored in the record's meta (default: this file's title)")
    matrix.add_argument("--report-note", default=None, metavar="TEXT",
                        help="one-line note stored in the record's meta and rendered in its "
                             "provenance")

    render = subparsers.add_parser(
        "render", help="re-render the measured block from an existing record")
    render.add_argument("--json", type=Path, default=DEFAULT_JSON, metavar="record.json",
                        help="document to render from (default: %(default)s)")
    render.add_argument("--out", type=Path, default=DEFAULT_REPORT, metavar="REPORT",
                        help="report whose marked block is replaced (default: %(default)s)")
    _add_source_arguments(render, defaults=False)

    subparsers.add_parser("list", help="print the platforms, the cases and the defaults")

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
