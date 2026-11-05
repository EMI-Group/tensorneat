"""The record schema plus the rendered measured block of ``benchmarks/GPU.md``.

This is the presentation half of ``benchmarks/gpu_measure.py``, split out for the
repository's one-responsibility file rule exactly like
``benchmarks/harness/report_format.py``/``report_sections.py`` split the harness
report from its protocol. It owns the document schema, the two report markers, the
cross-check tolerances that the report states, and every line inside the markers;
the measurement half owns the experiment. Stdlib-only at import time, so
``gpu_measure.py --list``/``--help`` work on an interpreter without numpy.

Nothing here invents a number: every cell of the block is read off the record, a
metric that does not exist prints ``n/a``, and a row the record marks ``error``
prints its error text instead of a value.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

#: Repository root (``<repo>/benchmarks/gpu_report.py``).
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

#: Schema of the aggregated matrix document (one record list under ``records``).
SCHEMA = "tensorneat-gpu/1"
#: Exact report markers whose *interior* :func:`render_block` owns.
BEGIN_MARKER = "<!-- BEGIN MEASURED (gpu_measure.py) -->"
END_MARKER = "<!-- END MEASURED (gpu_measure.py) -->"

#: Relative tolerance of the float32 comparison between two backends.
#:
#: float32 addition is not associative, so a reduction iree reassociates differs
#: from the numpy interpreter's order by about one ulp (~1.19e-7 at unit
#: magnitude) per reordering: the error grows with the reduction depth but stays
#: within a small multiple of eps. Measured on small populations over several
#: seeds, max |iree - numpy| is ~6e-8, so 1e-5 keeps ~two orders of magnitude of
#: headroom for a box- or build-dependent reordering while still failing on any
#: real divergence (a wrong operand, a dropped edge or a changed activation moves
#: values by far more than 1e-2). Same constants as
#: ``test/backends/test_iree_pipeline.py``.
FLOAT_RTOL = 1e-5
#: Absolute floor for the same comparison: attributes can sit near zero, where a
#: pure relative bound is meaningless. 1e-6 is ~8x the largest difference measured
#: there and still ~1 % of the smallest meaningful weight.
FLOAT_ATOL = 1e-6

#: Payload fields of one genome batch, in ``GenomeTensors`` order.
PAYLOAD_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: Why a final-state key mismatch is not automatically a kernel defect.
TRAJECTORY_NOTE = (
    "Both legs draw the same evoxc splitmix64 stream from one seed and one graph, "
    "so generation 1 must match exactly on the integer keys and within "
    f"rtol={FLOAT_RTOL}/atol={FLOAT_ATOL} on the float payload. Beyond it a "
    "float-reassociation difference between the backends can reorder a near-tied "
    "selection, so a final-state key mismatch is search-trajectory drift, not "
    "necessarily a kernel defect."
)

#: One-sentence reading rule recorded with every lockstep trajectory (``--trace``).
LOCKSTEP_NOTE = (
    "Both legs share one PRNG from one seed and one graph, so a first key divergence "
    "beyond generation 1 is a selection near-tie flip caused by a sub-ulp float "
    "difference between the backends, not a kernel defect."
)

__all__ = [
    "SCHEMA",
    "BEGIN_MARKER",
    "END_MARKER",
    "FLOAT_RTOL",
    "FLOAT_ATOL",
    "PAYLOAD_FIELDS",
    "TRAJECTORY_NOTE", "LOCKSTEP_NOTE",
    "RenderError",
    "fmt",
    "last_error_line",
    "snapshot",
    "compare_keys",
    "compare_float",
    "compare_snapshots", "trajectory_row",
    "read_document",
    "cross_summary",
    "environment",
    "render_row",
    "render_block",
    "summary_line",
    "splice_block",
]


class RenderError(RuntimeError):
    """The report or the record cannot be rendered (exit code 2, never a traceback)."""


# --- Small formatting helpers (shared with the measurement half) ------------------------


def fmt(value: Any, digits: int = 3) -> str:
    """Render one optional record value, ``n/a`` when it does not exist."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def last_error_line(text: Any) -> str:
    """The last non-empty line of a traceback/error string (``n/a`` when empty)."""
    if not text:
        return "n/a"
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else "n/a"


# --- Cross-check primitives (the comparison that defines "agrees") ---------------------
def _numpy() -> Any:
    """Import numpy lazily: nothing at this module's import time needs it."""
    import numpy

    return numpy


def snapshot(payload: Any, fitness: Any) -> dict[str, Any]:
    """One host numpy snapshot of a four-array genome batch plus its ``(P,)`` fitness."""
    np = _numpy()
    values = {
        name: np.asarray(value) for name, value in zip(PAYLOAD_FIELDS, tuple(payload))
    }
    values["fitness"] = np.asarray(fitness)
    return values


def compare_keys(name: str, accelerated: Any, reference: Any) -> dict[str, Any]:
    """Exact comparison of one int key payload (``node_keys``/``conn_keys``)."""
    np = _numpy()
    left = np.asarray(accelerated)
    right = np.asarray(reference)
    field: dict[str, Any] = {
        "kind": "int-keys",
        "shape_match": left.shape == right.shape,
        "mismatching_entries": None,
    }
    if not field["shape_match"]:
        return {**field, "status": "fail"}
    field["mismatching_entries"] = int(np.count_nonzero(left != right))
    field["status"] = "pass" if field["mismatching_entries"] == 0 else "fail"
    return field


def compare_float(
    accelerated: Any, reference: Any, kind: str, masks: tuple[str, ...]
) -> dict[str, Any]:
    """One float payload: identical masks plus ``FLOAT_RTOL``/``FLOAT_ATOL``.

    Args:
        accelerated: The measured leg's array.
        reference: The numpy reference's array.
        kind: ``"float-attrs"`` or ``"float-fitness"`` (recorded, never guessed).
        masks: Which non-finite masks must match exactly -- ``("nan",)`` for the
            attribute payloads, ``("nan", "neginf")`` for the fitness vector.

    Returns:
        The per-field verdict: shape agreement, every mask-mismatch count, the max
        absolute difference over entries finite on both sides, ``allclose``, the
        tolerances and the pass/fail status.
    """
    np = _numpy()
    left = np.asarray(accelerated)
    right = np.asarray(reference)
    if left.shape != right.shape:
        return {"kind": kind, "shape_match": False, "status": "fail"}
    predicates = {"nan": np.isnan, "neginf": np.isneginf}
    field: dict[str, Any] = {"kind": kind, "shape_match": True}
    for label in masks:
        field[f"{label}_mask_mismatches"] = int(
            np.count_nonzero(predicates[label](left) != predicates[label](right))
        )
    field["non_finite_mismatches"] = int(
        np.count_nonzero(np.isfinite(left) != np.isfinite(right))
    )
    both_finite = np.isfinite(left) & np.isfinite(right)
    field["max_abs_diff"] = None
    field["allclose"] = True
    if np.any(both_finite):
        field["max_abs_diff"] = float(
            np.max(np.abs(left[both_finite] - right[both_finite]))
        )
        field["allclose"] = bool(
            np.allclose(
                left[both_finite], right[both_finite], rtol=FLOAT_RTOL, atol=FLOAT_ATOL
            )
        )
    field["rtol"] = FLOAT_RTOL
    field["atol"] = FLOAT_ATOL
    mask_clean = all(field[f"{label}_mask_mismatches"] == 0 for label in masks)
    passed = mask_clean and field["non_finite_mismatches"] == 0 and field["allclose"]
    field["status"] = "pass" if passed else "fail"
    return field


def compare_snapshots(
    accelerated: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Compare one full snapshot (four payload arrays + fitness) field by field.

    Returns:
        ``{"status", "first_failing_field", "fields"}`` with a per-field verdict, the
        max absolute difference over finite entries and the mask-mismatch counts --
        the record the report's cross-check table is rendered from.
    """
    attrs = ("float-attrs", ("nan",))
    fields: dict[str, Any] = {
        "node_keys": compare_keys(*("node_keys", accelerated["node_keys"], reference["node_keys"])),
        "conn_keys": compare_keys(*("conn_keys", accelerated["conn_keys"], reference["conn_keys"])),
        "node_attrs": compare_float(accelerated["node_attrs"], reference["node_attrs"], *attrs),
        "conn_attrs": compare_float(accelerated["conn_attrs"], reference["conn_attrs"], *attrs),
        "fitness": compare_float(
            accelerated["fitness"], reference["fitness"], "float-fitness", ("nan", "neginf")),
    }
    first_failing = next(
        (name for name, field in fields.items() if field["status"] != "pass"), None
    )
    return {
        "status": "pass" if first_failing is None else "fail",
        "first_failing_field": first_failing,
        "fields": fields,
    }


def trajectory_row(
    generation: int, accelerated: dict[str, Any], reference: dict[str, Any],
    accelerated_species: int | None = None, reference_species: int | None = None,
) -> tuple[dict[str, Any], bool]:
    """One generation of the lockstep trace: compare two snapshots into one row.

    The species counts are ``None`` when a state does not expose them cheaply; the
    mask counts are :func:`compare_float`'s (both attribute NaN masks summed; fitness
    NaN and ``-inf`` masks). Returns ``(row, float_exceeds)``, the flag saying whether
    any float payload exceeded ``FLOAT_RTOL``/``FLOAT_ATOL`` in this generation.
    """
    fields = compare_snapshots(accelerated, reference)["fields"]
    node, conn, fit = fields["node_attrs"], fields["conn_attrs"], fields["fitness"]
    row = {
        "generation": int(generation),
        "node_keys_mismatches": fields["node_keys"].get("mismatching_entries"),
        "conn_keys_mismatches": fields["conn_keys"].get("mismatching_entries"),
        "node_attrs_max_abs_diff": node.get("max_abs_diff"),
        "conn_attrs_max_abs_diff": conn.get("max_abs_diff"),
        "attr_nan_mask_mismatches": int(node.get("nan_mask_mismatches") or 0)
        + int(conn.get("nan_mask_mismatches") or 0),
        "fitness_max_abs_diff": fit.get("max_abs_diff"),
        "fitness_mask_mismatches": int(fit.get("nan_mask_mismatches") or 0)
        + int(fit.get("neginf_mask_mismatches") or 0),
        "max_fitness_numpy": float(_numpy().max(reference["fitness"])),
        "max_fitness_accelerated": float(_numpy().max(accelerated["fitness"])),
        "species_count_numpy": reference_species,
        "species_count_accelerated": accelerated_species,
    }
    exceeds = any(
        fields[name].get("shape_match") is False or fields[name].get("allclose") is False
        for name in ("node_attrs", "conn_attrs", "fitness"))
    return row, exceeds


def display_path(path: Path | str | None) -> str:
    """A repo-relative path when possible, else the path as given."""
    if path is None:
        return "n/a"
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def is_committed_record(path: Path | str | None) -> bool:
    """True when ``path`` is a record the evidence ``.gitignore`` re-includes."""
    if path is None:
        return False
    try:
        relative = Path(path).resolve().relative_to(REPO_ROOT)
    except (OSError, ValueError):
        return False
    if relative.parent != Path("benchmarks/results"):
        return False
    try:
        lines = [line.strip() for line in (HERE / "results" / ".gitignore").read_text().splitlines()]
    except OSError:
        return False
    return f"!{relative.name}" in lines


# --- Environment fingerprint (session provenance, never fatal) --------------------------


#: Seconds allowed for the ``rocm-smi`` provenance probe.
ROCM_SMI_TIMEOUT = 10
#: ROCm version file; unreadable -> ``None`` (provenance is never fatal).
ROCM_VERSION_FILE = Path("/opt/rocm/.info/version")


def version(package: str) -> str | None:
    """Installed version from package metadata, or ``None`` when it is absent."""
    try:
        return metadata.version(package)
    except Exception:
        # Importing a package just to read a version would defeat the metadata probe.
        return None


def _read_text(path: Path) -> str | None:
    """File contents, or ``None`` when it cannot be read."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _parse_rocm_smi(text: str) -> dict[str, str | None]:
    """Scrape the model, gfx version and driver version out of ``rocm-smi`` output.

    Args:
        text: Combined stdout/stderr of ``rocm-smi --showproductname
            --showdriverversion``.

    Returns:
        ``{"gpu_model", "gfx_version", "driver_version"}``, each ``None`` when the
        corresponding line is absent (the tool's output is not a stable API).
    """
    found: dict[str, str | None] = {
        "gpu_model": None,
        "gfx_version": None,
        "driver_version": None,
    }
    labels = {
        "Card Series:": "gpu_model",
        "GFX Version:": "gfx_version",
        "Driver version:": "driver_version",
    }
    for line in text.splitlines():
        for label, key in labels.items():
            if label in line:
                value = line.split(label, 1)[1].strip()
                if value and found[key] is None:
                    found[key] = value
    return found


def _rocm_block() -> dict[str, Any]:
    """Best-effort ROCm/GPU provenance: the version file plus one ``rocm-smi`` call.

    Returns:
        ``{"rocm_version", "rocm_smi", "rocm_smi_error"}``; every failure is
        recorded in ``rocm_smi_error`` instead of aborting the measurement.
    """
    block: dict[str, Any] = {
        "rocm_version": _read_text(ROCM_VERSION_FILE) or None,
        "rocm_smi": None,
        "rocm_smi_error": None,
    }
    try:
        completed = subprocess.run(
            ["rocm-smi", "--showproductname", "--showdriverversion"],
            capture_output=True,
            text=True,
            timeout=ROCM_SMI_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        block["rocm_smi_error"] = f"{type(exc).__name__}: {exc}"
        return block
    block["rocm_smi"] = _parse_rocm_smi(
        (completed.stdout or "") + "\n" + (completed.stderr or "")
    )
    if completed.returncode != 0:
        block["rocm_smi_error"] = f"rocm-smi exited {completed.returncode}"
    return block


def _iree_block() -> dict[str, Any]:
    """IREE compiler/runtime metadata plus the HIP devices its runtime can see.

    Returns:
        ``{"compiler", "runtime", "hip_devices", "hip_error"}``; the device query
        degrades to ``None`` with a recorded reason, never an exception.
    """
    block: dict[str, Any] = {
        "compiler": version("iree-base-compiler"),
        "runtime": version("iree-base-runtime"),
        "hip_devices": None,
        "hip_error": None,
    }
    try:
        import iree.runtime as iree_runtime

        devices = iree_runtime.get_driver("hip").query_available_devices()
        block["hip_devices"] = [
            {"device_id": item.get("device_id"), "name": item.get("name")}
            for item in devices
        ]
    except Exception as exc:
        block["hip_error"] = f"{type(exc).__name__}: {exc}"
    return block


def environment() -> dict[str, Any]:
    """The environment fingerprint: harness bases plus this protocol's GPU probes.

    ``harness.protocol.env_report`` already embeds ``git_report`` (commit, dirty
    flag over the code under test, offending paths) and records the thread
    variables as inherited, so nothing here re-implements those probes. The
    ``gpu``/``iree`` blocks are the accelerator additions; a failing probe is
    recorded as an error string and never aborts the measurement.

    Returns:
        A JSON-safe mapping; ``probe_error`` carries the harness fingerprint's own
        traceback if that ever failed.
    """
    env: dict[str, Any] = {}
    try:
        from harness.protocol import env_report

        env.update(env_report(None, root=REPO_ROOT))
        env["probe_error"] = None
    except Exception:
        env["probe_error"] = traceback.format_exc()
        env.update(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "numpy": version("numpy"),
                "evoxc": version("evoxc"),
                "git_commit": None,
                "git_dirty": None,
                "git_dirty_paths": [],
                "loadavg": None,
            }
        )
    env["iree"] = _iree_block()
    env["gpu"] = _rocm_block()
    return env


# --- The document -----------------------------------------------------------------------


def validate_document(document: Any) -> None:
    """Reject a record the renderer cannot read, with a named error.

    Raises:
        RenderError: When the document is not an object, carries another schema, or
            has no ``records`` list.
    """
    if not isinstance(document, dict):
        raise RenderError("the record is not a JSON object")
    if document.get("schema") != SCHEMA:
        raise RenderError(
            f"unsupported record schema {document.get('schema')!r}; this renderer "
            f"reads {SCHEMA!r}"
        )
    if not isinstance(document.get("records"), list):
        raise RenderError("the record carries no 'records' list")


def read_document(path: Path) -> dict[str, Any]:
    """Load and validate a matrix document.

    Args:
        path: The JSON record to read.

    Returns:
        The validated document.

    Raises:
        RenderError: On an unreadable file, invalid JSON, a wrong schema or a
            missing ``records`` list.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise RenderError(f"cannot read the record {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(f"the record {path} is not valid JSON: {exc}") from exc
    validate_document(document)
    return document


# --- Table helpers ----------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """A markdown table from header titles and already-stringified cells."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in rows]
    return lines


def _session_text(session: dict[str, Any] | None) -> str:
    """Render a ``meta.session_utc`` block as a date/session range."""
    if not session:
        return "n/a"
    start = session.get("start")
    end = session.get("end")
    if start and end and start != end:
        return f"{start} – {end}"
    return start or end or "n/a"


def _git_text(env: dict[str, Any]) -> str:
    """Render the git commit plus the dirty flag of the code under test."""
    commit = env.get("git_commit")
    if not commit:
        return "n/a"
    dirty = env.get("git_dirty")
    if dirty is None:
        return f"`{commit[:12]}` (dirty flag n/a)"
    if not dirty:
        return f"`{commit[:12]}` (clean — generated report artifacts are not counted)"
    paths = env.get("git_dirty_paths") or []
    named = ", ".join(paths[:5]) + (f" +{len(paths) - 5} more" if len(paths) > 5 else "")
    return f"`{commit[:12]}` (dirty: {named})"


# --- The six block sections -------------------------------------------------------------


def _environment_section(document: dict[str, Any]) -> list[str]:
    """Section 1: the measured environment fingerprint as a two-column table."""
    meta = document.get("meta") or {}
    env = meta.get("env") or {}
    gpu = env.get("gpu") or {}
    rocm = gpu.get("rocm_smi") or {}
    iree = env.get("iree") or {}
    devices = iree.get("hip_devices") or []
    model = rocm.get("gpu_model") or (devices[0].get("name") if devices else None)
    gfx = rocm.get("gfx_version")
    gpu_text = f"{model} ({gfx})" if model and gfx else (model or gfx or "n/a")
    hip_text = (
        ", ".join(f"device_id={item.get('device_id')} {item.get('name')}" for item in devices)
        if devices
        else f"n/a ({iree['hip_error']})" if iree.get("hip_error") else "n/a"
    )
    memory = env.get("memory_mb")
    threads = env.get("threads")
    loadavg = env.get("loadavg")
    rows = [
        ["Date / session (UTC)", _session_text(meta.get("session_utc"))],
        ["GPU model (gfx version)", gpu_text],
        ["HIP devices (IREE runtime query)", hip_text],
        ["ROCm version", gpu.get("rocm_version") or "n/a"],
        ["Driver version", rocm.get("driver_version") or "n/a"],
        ["IREE compiler / runtime",
         f"{iree.get('compiler') or 'n/a'} / {iree.get('runtime') or 'n/a'}"],
        ["evoxc", env.get("evoxc") or "n/a"],
        ["numpy", env.get("numpy") or "n/a"],
        ["Python", env.get("python") or "n/a"],
        ["Platform", env.get("platform") or "n/a"],
        ["CPU model / count",
         f"{env.get('cpu_model') or 'n/a'} / {env.get('cpu_count') or 'n/a'} logical"],
        ["Memory", f"{memory:.0f} MiB" if isinstance(memory, (int, float)) else "n/a"],
        ["Threads",
         "runtime defaults (nothing exported; inherited environment recorded)"
         if threads is None else str(threads)],
        ["Git commit", _git_text(env)],
        ["Load average (1/5/15 min, document build)",
         " / ".join(fmt(value, 2) for value in loadavg) if loadavg else "n/a"],
    ]
    if env.get("probe_error"):
        rows.append(["Harness fingerprint error", last_error_line(env["probe_error"])])
    if gpu.get("rocm_smi_error"):
        rows.append(["rocm-smi probe", gpu["rocm_smi_error"]])
    return ["## Environment (measured, not assumed)", ""] + _table(["Item", "Value"], rows)


def cross_summary(record: dict[str, Any]) -> str:
    """Rendered cross-check verdict: generation 1 is the equality gate, later states are not.
    Generation 1 is a hard equality comparison (one seed, one graph, one PRNG); a
    later-generation difference is only a search-trajectory observation, so a stored
    ``fail`` with a passing generation 1 renders ``gen 1 pass; final-state drift
    (<field>)``; one without keeps the unsoftened ``FAIL (gen 1: <field>)``.
    """
    cross = record.get("cross_check") or {}
    status = cross.get("status")
    if status == "pass":
        return "pass (gen 1)"
    if status == "fail":
        passed = (cross.get("generation_1") or {}).get("status") == "pass"
        state = cross.get("final_state" if passed else "generation_1") or {}
        name = state.get("first_failing_field") or "unknown field"
        return f"gen 1 pass; final-state drift ({name})" if passed else f"FAIL (gen 1: {name})"
    if status == "skipped":
        return f"skipped ({cross.get('reason') or 'n/a'})"
    if status == "error":
        return f"error ({last_error_line(cross.get('error'))})"
    return "n/a"


def _measured_section(document: dict[str, Any]) -> list[str]:
    """Section 2: one row per ``(case, target)``, every planned row included."""
    rows: list[list[str]] = []
    for record in document.get("records") or []:
        timing = record.get("ms_per_generation") or {}
        rows.append(
            [
                str(record.get("case", "?")),
                f"`{record.get('target', '?')}`",
                f"`{record.get('backend', '?')}`",
                str(record.get("status", "?")),
                fmt(record.get("setup_s")),
                fmt(record.get("build_s")),
                fmt(timing.get("median")),
                fmt(timing.get("mean")),
                f"{fmt(timing.get('min'))} – {fmt(timing.get('max'))}" if timing else "n/a",
                fmt(record.get("generations_per_second")),
                fmt(record.get("peak_rss_mib"), 1),
                fmt(record.get("final_max_fitness"), 6),
                cross_summary(record),
            ]
        )
    return [
        "## Measured: numpy vs iree llvm-cpu vs iree rocm",
        "",
        "One row per planned `(case, target)`; `n/a` marks a metric the row does not "
        "have (an error row has no timings at all). `build` is the one-off "
        "generation-graph compile and is excluded from every `ms/gen` sample; `peak "
        "RSS` is the process high-water mark of that single row, which is why each row "
        "runs in its own process.",
        "",
    ] + _table(
        ["Case", "Target", "Backend", "Status", "setup (s)", "build (s)", "ms/gen median",
         "ms/gen mean", "ms/gen min–max", "gens/s", "peak RSS (MiB)", "final max fitness",
         "cross-check"],
        rows,
    )


def _fields_of(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """The per-field dict of one snapshot comparison (empty when absent)."""
    return (snapshot or {}).get("fields") or {}


def _keys_cell(snapshot: dict[str, Any] | None) -> str:
    """Render the exact-int-key verdict of one snapshot comparison."""
    fields = _fields_of(snapshot)
    parts = []
    for name in ("node_keys", "conn_keys"):
        field = fields.get(name)
        if not field:
            continue
        if field.get("status") == "pass":
            parts.append(f"{name} exact")
        else:
            parts.append(f"{name} {fmt(field.get('mismatching_entries'))} mismatch(es)")
    return "; ".join(parts) or "n/a"


def _attrs_cell(snapshot: dict[str, Any] | None) -> str:
    """Render the max absolute attribute difference of one snapshot comparison."""
    fields = _fields_of(snapshot)
    parts = []
    for name in ("node_attrs", "conn_attrs"):
        field = fields.get(name)
        if field:
            delta = field.get("max_abs_diff")
            parts.append(f"{name} {fmt(delta, 9) if delta is not None else 'n/a'}")
    return "; ".join(parts) or "n/a"


def _fitness_cell(snapshot: dict[str, Any] | None) -> str:
    """Render the max absolute fitness difference of one snapshot comparison."""
    field = _fields_of(snapshot).get("fitness")
    if not field or field.get("max_abs_diff") is None:
        return "n/a"
    return fmt(field["max_abs_diff"], 9)


def _masks_cell(snapshot: dict[str, Any] | None) -> str:
    """Render the summed mask-mismatch counts of one snapshot comparison."""
    seen = False
    total = 0
    for field in _fields_of(snapshot).values():
        for key in ("nan_mask_mismatches", "neginf_mask_mismatches", "non_finite_mismatches"):
            if key in field:
                seen = True
                total += int(field.get(key) or 0)
    return str(total) if seen else "n/a"


def _cross_check_section(document: dict[str, Any]) -> list[str]:
    """Section 3: the numerical cross-check, one row per measured non-numpy row."""
    rows: list[list[str]] = []
    for record in document.get("records") or []:
        if record.get("target") == "numpy":
            continue
        cross = record.get("cross_check") or {}
        status = cross.get("status")
        first = cross.get("generation_1")
        final = cross.get("final_state")
        head = [str(record.get("case", "?")), f"`{record.get('target', '?')}`"]
        if status == "error":
            rows.append(head + ["error", last_error_line(cross.get("error"))] + ["n/a"] * 7)
            continue
        rows.append(
            head
            + [
                str((first or {}).get("status") or status or "n/a"),
                _keys_cell(first),
                _attrs_cell(first),
                _fitness_cell(first),
                _masks_cell(first),
                str((final or {}).get("status") or status or "n/a"),
                _keys_cell(final),
                _attrs_cell(final),
                _fitness_cell(final),
                _masks_cell(final),
            ]
        )
    return [
        "## Numerical cross-check",
        "",
        "Each non-numpy row is compared against a fresh `backend=\"numpy\"` run of the "
        "same case, seed and step count in the same process. Generation 1 must match "
        "(one seed, one graph, one PRNG): integer keys exactly, attributes and fitness "
        f"to rtol {FLOAT_RTOL:g} / atol {FLOAT_ATOL:g} over finite entries with an "
        "identical NaN mask. The final state is a search-trajectory observation, not a "
        "pass/fail gate on its own: beyond generation 1 a float-reassociation "
        "difference can reorder a near-tied selection and legitimately diverge the keys. "
        "The table's `cross-check` column summarises both snapshots: generation 1 is the "
        "equality gate; a final-state-only difference renders `gen 1 pass; final-state drift (…)`.",
        "",
    ] + _table(
        ["Case", "Target", "Generation 1", "Keys (gen 1)", "Attr max abs diff (gen 1)",
         "Fitness max abs diff", "Mask mismatches", "Final state", "Keys (final)",
         "Attr max abs diff (final)", "Fitness max abs diff (final)",
         "Mask mismatches (final)"],
        rows,
    )


def _trajectory_section(document: dict[str, Any]) -> list[str]:
    """Section 4: the per-generation lockstep trace; empty without an ``ok`` trace."""
    traced = [r for r in document.get("records") or []
              if isinstance(r.get("trajectory"), dict)]
    ok = [r for r in traced if r["trajectory"].get("status") == "ok"]
    if not ok:
        return []
    headers = [
        "Generation", "Node-key mismatches", "Conn-key mismatches",
        "Node attr max abs diff", "Conn attr max abs diff", "Attr NaN-mask mismatches",
        "Fitness max abs diff", "Fitness mask mismatches",
        "Max fitness (numpy / accelerated)", "Species (numpy / accelerated)",
    ]
    lines = [
        "## Per-generation trajectory",
        "",
        "Every traced row was re-run after the timed region, with the peak-RSS sample "
        "already taken, as a lockstep pair: a fresh accelerated pipeline and a fresh "
        "`backend=\"numpy\"` pipeline of the same case and seed, each paying its own "
        "build, compared per generation with the cross-check rules above. The trace "
        "touches neither the timings nor the recorded RSS.",
        "",
    ]
    for record in ok:
        trace = record["trajectory"]
        steps = fmt(trace.get("steps"))
        none = f"none within {steps} generations"
        key, floated = trace.get("first_key_divergence"), trace.get("first_float_exceedance")
        lines += [
            f"**{record.get('case', '?')} / `{record.get('target', '?')}`** — {steps} "
            f"generation(s) compared in lockstep; first node-key divergence: "
            f"{none if key is None else key}; first float exceedance: "
            f"{none if floated is None else floated}; wall time {fmt(trace.get('trace_s'))} s.",
            "",
        ]
        lines += [f"*{trace['note']}*", ""] if trace.get("note") else []
        lines += _table(
            headers,
            [
                [
                    fmt(item.get("generation")),
                    fmt(item.get("node_keys_mismatches")),
                    fmt(item.get("conn_keys_mismatches")),
                    fmt(item.get("node_attrs_max_abs_diff"), 9),
                    fmt(item.get("conn_attrs_max_abs_diff"), 9),
                    fmt(item.get("attr_nan_mask_mismatches")),
                    fmt(item.get("fitness_max_abs_diff"), 9),
                    fmt(item.get("fitness_mask_mismatches")),
                    f"{fmt(item.get('max_fitness_numpy'), 6)} / "
                    f"{fmt(item.get('max_fitness_accelerated'), 6)}",
                    f"{fmt(item.get('species_count_numpy'))} / "
                    f"{fmt(item.get('species_count_accelerated'))}",
                ]
                for item in trace.get("generations") or []
            ],
        ) + [""]
    errors = [f"The trajectory trace of {r.get('case', '?')} / {r.get('target', '?')} "
              f"errored ({last_error_line(r['trajectory'].get('error'))}); its timings "
              "stay valid." for r in traced if r["trajectory"].get("status") == "error"]
    return lines + errors if errors else lines[:-1]  # no trailing blank before the next section


def _blocked_section(document: dict[str, Any]) -> list[str]:
    """Section 5: one row per error record, with the tail of its error."""
    rows = [
        [str(record.get("case", "?")), f"`{record.get('target', '?')}`",
         last_error_line(record.get("error"))]
        for record in document.get("records") or []
        if record.get("status") != "ok"
    ]
    if not rows:
        return ["## Blocked rows", "", "No row was blocked: every planned row is `ok`."]
    return [
        "## Blocked rows",
        "",
        "The full traceback of every row below stays in the record; the cell prints its "
        "last non-empty line.",
        "",
    ] + _table(["Case", "Target", "Error (last non-empty line)"], rows)


def _provenance_section(document: dict[str, Any], record_path: Path | str | None) -> list[str]:
    """Section 6: data-derived provenance of the block (tree, session, record)."""
    meta = document.get("meta") or {}
    env = meta.get("env") or {}
    records = document.get("records") or []
    n_ok = sum(1 for record in records if record.get("status") == "ok")
    display = display_path(record_path)
    if is_committed_record(record_path):
        status = (
            f"`{display}` is committed evidence (it is in the re-include list of "
            "`benchmarks/results/.gitignore`), so every cell above is traceable to the "
            "committed record."
        )
    else:
        status = (
            f"`{display}` is a scratch record: it is git-ignored and **not** committed "
            "evidence, so the cells above describe this session only."
        )
    lines = [
        "## Provenance of this block",
        "",
        f"This block is rendered from `{display}`: {len(records)} record(s), {n_ok} ok and "
        f"{len(records) - n_ok} error, measured in the session "
        f"{_session_text(meta.get('session_utc'))} (UTC) at git {_git_text(env)}.",
        status,
        "Nothing here is a statement about HEAD: every number belongs to that record and "
        "to the tree its fingerprint names.",
    ]
    if meta.get("report_note"):
        lines.append(str(meta["report_note"]))
    return lines


# --- The row tables (stdout) ------------------------------------------------------------


def render_row(record: dict[str, Any]) -> str:
    """Render one row record as the aligned table ``run`` prints."""
    head = (
        f"gpu_measure: {record['case']} / {record['target']} / backend "
        f"{record['backend']} / device {record['device']!r} / seed {record['seed']}"
    )
    if record.get("status") != "ok":
        return "\n".join(
            [
                head,
                "",
                "status: error",
                str(record.get("error") or "(no error recorded)").rstrip(),
                "",
                f"device_info: {record.get('device_info')!r}",
            ]
        )
    timing = record["ms_per_generation"]
    return "\n".join(
        [
            head,
            "",
            f"{'metric':<38}{'value':>24}",
            f"{'-' * 38} {'-' * 24:>24}",
            f"{'setup (Pipeline.setup)':<38}{record['setup_s']:>18.3f} s",
            f"{'evoxc build (pipeline.build_seconds)':<38}{record['build_s']:>18.3f} s",
            f"{'first step (build + 1 run)':<38}{record['first_step_s']:>18.3f} s",
            f"{'warmup generations (untimed)':<38}{record['warmup_generations']:>24d}",
            f"{'timed generations':<38}{record['timed_generations']:>24d}",
            f"{'ms/generation mean':<38}{timing['mean']:>21.3f} ms",
            f"{'ms/generation median':<38}{timing['median']:>21.3f} ms",
            f"{'ms/generation min / max':<38}"
            f"{timing['min']:>14.3f} / {timing['max']:.3f} ms",
            f"{'generations/second (from mean)':<38}"
            f"{fmt(record['generations_per_second']):>18} /s",
            f"{'peak RSS (process high-water mark)':<38}{record['peak_rss_mib']:>18.2f} MiB",
            f"{'final generation':<38}{record['final_generation']:>24d}",
            f"{'final generation max fitness':<38}{record['final_max_fitness']:>24.6f}",
            "",
            f"compile_options: {record['compile_options']!r}",
            f"cross-check vs numpy: {cross_summary(record)}",
            "build time is one-off and per shape; it is excluded from every "
            "ms/generation sample.",
            "peak RSS is a process-lifetime high-water mark -- one row per process.",
        ]
    )


def summary_line(index: int, total: int, record: dict[str, Any]) -> str:
    """One-line per-row summary printed by ``matrix``."""
    head = (
        f"[{index}/{total}] {record.get('case', '?')} / {record.get('target', '?')}: "
        f"{str(record.get('status', 'error')).upper()}"
    )
    if record.get("status") != "ok":
        return f"{head} — {last_error_line(record.get('error'))}"
    timing = record.get("ms_per_generation") or {}
    return (
        f"{head} — setup {fmt(record.get('setup_s'))} s, build "
        f"{fmt(record.get('build_s'))} s, {fmt(timing.get('median'))} ms/gen, "
        f"{fmt(record.get('peak_rss_mib'), 1)} MiB, cross-check "
        f"{cross_summary(record)}"
    )


# --- The block and the markers ----------------------------------------------------------


def render_block(
    document: dict[str, Any], record_path: Path | str | None = None
) -> str:
    """Render the measured block of ``GPU.md`` from a matrix document.

    Args:
        document: A ``schema="tensorneat-gpu/1"`` document.
        record_path: The record file the block is rendered from; the provenance
            paragraph names it and classifies it as committed evidence or scratch.

    Returns:
        The block text, without the markers themselves.

    Raises:
        RenderError: On an invalid document (or an unreadable one, when a path is
            passed instead of a mapping).
    """
    if isinstance(document, (str, Path)):
        document = read_document(Path(document))
    validate_document(document)
    env = (document.get("meta") or {}).get("env") or {}
    lines: list[str] = []
    lines += _environment_section(document) + [""]
    lines += _measured_section(document) + [""]
    lines += _cross_check_section(document) + [""]
    trajectory = _trajectory_section(document)
    if trajectory:
        lines += trajectory + [""]
    lines += _blocked_section(document) + [""]
    lines += _provenance_section(document, record_path)
    if env.get("probe_error"):
        lines.append(
            "The environment fingerprint degraded: "
            f"{last_error_line(env['probe_error'])} (the measurement itself ran)."
        )
    return "\n".join(lines).strip("\n")


def splice_block(report_path: Path | str, block: str) -> None:
    """Replace the text between the report's exact markers with ``block``.

    Args:
        report_path: The report to rewrite; every byte outside the markers is kept.
        block: The rendered block (markers are added here, never by the caller).

    Raises:
        RenderError: When the report cannot be read, or either marker is absent or
            out of order.
    """
    path = Path(report_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RenderError(f"cannot read the report {path}: {exc}") from exc
    begin = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if begin < 0 or end < 0:
        raise RenderError(
            f"the report {path} does not carry both markers ({BEGIN_MARKER!r} and "
            f"{END_MARKER!r}); refusing to touch it"
        )
    if end < begin:
        raise RenderError(
            f"the markers in {path} are out of order: the END marker precedes the BEGIN "
            "marker"
        )
    replacement = f"{BEGIN_MARKER}\n{block.strip()}\n{END_MARKER}"
    path.write_text(text[:begin] + replacement + text[end + len(END_MARKER):], encoding="utf-8")
