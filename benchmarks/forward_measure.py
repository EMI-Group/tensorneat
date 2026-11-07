#!/usr/bin/env python3
"""Measure the NEAT **batch forward pass**: new evoxc implementation vs frozen JAX.

The forward pass is one query of a whole population of differently-shaped networks, and
it is the dominant cost of a generation (and of an RL rollout, which queries the same
population once per environment timestep). Nothing else sits in the timed region -- no
mutation, no selection, no environment step.

New side (``impl="new"``): the ``benchmarks/cases.py`` case is built with
``tensorneat`` and wrapped in a ``Pipeline`` carrying the target's
``backend``/``device``/``compile_options``, whose ``Pipeline.setup()`` supplies the
population for the whole row. The timed graph is *only* the forward -- one
``evoxc.build`` of a kernel taking the four payload arrays plus the input block, whose
body is the public path the pipeline hands a problem: ``algorithm.transform`` then
``host_payload._NetworkActFn.prepare(P * S)`` (``prepare_forward`` once), whose query is
``forward_prepared`` per pass, exactly as ``Pipeline._generation_kernel`` wires the
``act_fn`` it passes to ``problem.evaluate``. The four private ``Pipeline`` members read
here (``_static_config``, ``_build_options``, ``_device_kind``, ``_placement_device``)
*are* the compile site: the package exposes no public "build me a forward graph" entry
point, and re-deriving them would measure a different graph.

Old side (``impl="jax"``): ``benchmarks/legacy_jax`` (``import tensorneat_legacy``) is
built from the same case config dicts; its population comes from ``algorithm.setup`` and
``vmap(algorithm.transform)``, and one pass is the composition the legacy
``Pipeline.step`` uses to evaluate -- ``vmap(population, vmap(samples,
genome.forward))`` -- compiled with ``jax.jit(...).lower(...).compile()``, the wiring
``benchmarks/harness/adapters/old.py`` uses. That side runs on the JAX default device
(this box: ``[RocmDevice(id=0)]``) and records ``jax.devices()`` per row.

Workloads are an explicit table (:data:`WORKLOADS`), because a forward's cost is a
function of ``(population, rows, input width, output width)``: each row names a case for
its genome geometry plus a sample count ``S`` (XOR 4, Pagie 500, RL 1). The 500-sample
rows use the XOR genome, whose config is byte-identical to ``custom_pagie_p500``'s (2
inputs, 1 output, sigmoid read-out), so they measure the Pagie shape. Targets are data
in :data:`TARGETS`: :data:`gpu_targets.TARGETS` verbatim (``numpy``, ``iree-rocm``,
``xla-rocm``) plus ``jax``. All four are in the default matrix (:data:`DEFAULT_TARGETS`);
``xla-rocm`` additionally needs the PJRT plugin named by ``--pjrt-plugin PATH`` or
``$EVOXC_PJRT_PLUGIN`` (an unresolvable plugin is a named error row, never a fallback).

The ``XLA_FLAGS`` workaround (``--xla-flags``): a *generation* graph holding a long
``while`` (the RL rollout) SIGSEGVs on this box inside
``PJRT_LoadedExecutable_Execute`` because XLA captures it as a HIP graph / FUSION
command buffer, and ``--xla_gpu_enable_command_buffer=WHILE`` in ``XLA_FLAGS`` cures
it (XLA reads that variable at backend init, so it must be an environment setting,
never a ``compile_options`` entry; see ``benchmarks/gpu_targets.py``). A forward pass
holds no ``while`` and does not crash, so the flag is not load-bearing here -- but an
``xla-rocm`` row is exported the very same ``XLA_FLAGS`` value, for an xla target
only, and records it (``record["xla_flags"]`` / ``meta["xla_flags"]``), so a
forward number is never attributed to a flag it did not run with.

Protocol, per row, one fresh process: ``setup_s`` and ``build_s`` are measured and
reported **separately, never charged to a pass**; ``--warmup`` untimed passes then
``--passes`` timed ones run under ``os.devnull``, each timed with ``perf_counter`` around
the call *including* the host materialisation of the output block, so an asynchronous
backend cannot report before the pass completed. The input block is drawn once from
``default_rng(seed)`` and pre-placed on the device, so no transfer sits inside the timed
region. A row records mean/median/min/max ms per forward, ``rows_per_second``,
``forwards_per_second``, the peak RSS high-water mark (which is why one row is one
process), the environment fingerprint, ``device_info``, the device's VRAM and an
``output`` fingerprint.

Nothing is faked: a leg that cannot run is an ``error`` row carrying its message or
traceback -- ``run`` writes a record even on a traceback and ``matrix`` synthesises one
when a child wrote none (crash, kill, timeout, non-JSON). Every new-side target runs the
same graph on the same payload, so the report holds them against the ``numpy`` leg; the
JAX side draws a different population from a different PRNG and is not value-comparable
by design (``test/parity/`` owns fidelity).

Subcommands: ``run`` measures one row (exit 0 ok / 1 error); ``matrix`` plans
``workloads x targets``, spawns one ``run`` per row strictly sequentially, writes one
document (``schema``, ``meta``, ``records``) and with ``--out REPORT`` refreshes that
report's marked block (exit 0 all ok / 1 any error / 2 unrenderable); ``render``
re-renders the block from an existing record; ``--list`` prints the tables on a
numpy-free interpreter.

    python3 benchmarks/forward_measure.py --list
    uv run --all-extras python benchmarks/forward_measure.py matrix --workloads all \\
        --targets numpy,iree-rocm,xla-rocm,jax --pjrt-plugin "$EVOXC_PJRT_PLUGIN" \\
        --json benchmarks/results/forward.json --out benchmarks/FORWARD.md
    uv run --all-extras python benchmarks/forward_measure.py render \\
        --json benchmarks/results/forward.json --out benchmarks/FORWARD.md

Requires ``tensorneat`` (installed or importable from ``../src``) and ``numpy``; the
``jax`` target needs the ``bench`` extra, ``iree-rocm`` the ``iree`` extra plus a ROCm
runtime, ``xla-rocm`` the ``xla-rocm`` extra plus a resolvable plugin.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
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

#: Repository root (``<repo>/benchmarks/forward_measure.py``).
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
#: The src-layout package, so the CLI also works from a bare checkout.
SRC_DIR = REPO_ROOT / "src"
#: The frozen pre-rewrite implementation, put on ``sys.path`` on the old side only.
LEGACY_DIR = HERE / "legacy_jax"

if str(HERE) not in sys.path:  # the sibling record/report modules are not a package
    sys.path.insert(0, str(HERE))
from gpu_report import (  # noqa: E402  (after the sys.path guard)
    FLOAT_ATOL, FLOAT_RTOL, RenderError, _table, display_path, environment,
    last_error_line, splice_block,
)
from gpu_targets import (  # noqa: E402  (after the sys.path guard)
    PJRT_ENV_VAR, PJRT_FLAG, PJRT_TARGETS, TARGETS as GPU_TARGETS,
    XLA_WORKAROUND_FLAGS, PluginError, fingerprint_plugin, plugin_for, require_plugin,
)

#: Schema of a forward-pass document (one record list under ``records``).
SCHEMA = "tensorneat-forward/1"
#: Exact report markers whose *interior* :func:`render_block` owns.
BEGIN_MARKER = "<!-- BEGIN MEASURED (forward_measure.py) -->"
END_MARKER = "<!-- END MEASURED (forward_measure.py) -->"

#: New-side targets are the accelerator protocol's own table, entry for entry; ``jax``
#: (the frozen baseline, on the JAX default device) is this file's addition. Data only.
TARGETS: dict[str, dict[str, Any]] = {
    name: {"impl": "new", **GPU_TARGETS[name]} for name in ("numpy", "iree-rocm", "xla-rocm")
}
TARGETS["jax"] = {"impl": "old", "backend": "jax", "device": None, "compile_options": None}

DEFAULT_WORKLOADS = "all"
#: The shipped matrix: every new-side leg plus the baseline. ``xla-rocm`` needs the PJRT
#: plugin (``--pjrt-plugin``/``$EVOXC_PJRT_PLUGIN``), which ``matrix`` resolves once up
#: front; a missing one is a named error, never a silently dropped row.
DEFAULT_TARGETS = "numpy,iree-rocm,xla-rocm,jax"
DEFAULT_SEED, DEFAULT_WARMUP, DEFAULT_PASSES = 42, 3, 10
#: Seconds one row subprocess may take before it is killed and recorded as an error.
DEFAULT_TIMEOUT = 3600
#: The XLA allocator policy of the JAX baseline. ``false`` allocates on demand instead of
#: reserving 75 % of the device up front: this box is a shared 205.8 GB ``MI300X VF`` and
#: the reservation failed rows with ``HIP_ERROR_OutOfMemory``. It changes neither the
#: compiled computation nor the device op count.
JAX_PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
DEFAULT_JSON = HERE / "results" / "forward.json"
DEFAULT_REPORT = HERE / "FORWARD.md"
DEFAULT_TITLE = "TensorNEAT batch forward pass — new evoxc vs frozen JAX"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Workload:
    """One forward-pass workload: a genome geometry plus the tiled row count.

    Attributes:
        name: Workload id, also the key of :data:`WORKLOADS`.
        case: Name in ``benchmarks/cases.py``; supplies the population and the genome
            geometry (input/output width, ``max_nodes``/``max_conns``, read-out).
        samples: ``S``, the tiled rows per individual (a func-fit dataset size, or 1 for
            one RL observation).
        note: How the row maps onto a shipped case, rendered in the workload table.
    """

    name: str
    case: str
    samples: int
    note: str = ""

    def __post_init__(self) -> None:
        """Reject a workload whose sample count could never be a positive tile."""
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 1:
            raise ValueError(f"workload {self.name!r}: samples must be a positive int")


WORKLOADS: tuple[Workload, ...] = (
    Workload(name="xor_p100_s4", case="xor_p100", samples=4,
             note="`xor_p100`: XOR's 4-row dataset, 2 inputs -> 1 sigmoid output"),
    Workload(name="xor_p500_s4", case="xor_p500", samples=4, note="`xor_p500`, same dataset"),
    Workload(name="xor_p1000_s4", case="xor_p1000", samples=4, note="`xor_p1000`, same dataset"),
    Workload(name="xor_p100_s500", case="xor_p100", samples=500,
             note="Pagie tiling (500 samples) on the `custom_pagie_p500` genome geometry"),
    Workload(name="xor_p500_s500", case="xor_p500", samples=500,
             note="Pagie tiling (500 samples) at the shipped Pagie population"),
    Workload(name="xor_p1000_s500", case="xor_p1000", samples=500,
             note="Pagie tiling (500 samples) at pop 1000"),
    Workload(name="cartpole_p100_s1", case="cartpole_p100", samples=1,
             note="`cartpole_p100` RL step: 4 inputs -> 2 outputs, one observation each"),
)
"""The measured workloads, in declaration order."""


# --- Lazy imports (module scope stays numpy/evoxc/jax/tensorneat-free) -----------------


def _numpy() -> Any:
    """Import numpy lazily so ``--help``/``--list`` work on a bare interpreter."""
    import numpy

    return numpy


def _evoxc() -> Any:
    """Import ``evoxc`` (the new side's only hard dependency)."""
    import evoxc

    return evoxc


def _tensorneat() -> Any:
    """Import ``tensorneat``, falling back to the in-repo ``src`` directory."""
    try:
        import tensorneat
    except ImportError:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        import tensorneat
    return tensorneat


def _legacy() -> Any:
    """Import the frozen pre-rewrite implementation (old side only)."""
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    import tensorneat_legacy

    return tensorneat_legacy


def _jax() -> Any:
    """Import ``jax`` (old side only; a new-side row must never pull it in)."""
    import jax

    return jax


def _case(name: str) -> Any:
    """The registered :class:`cases.Case` for ``name`` (``ValueError`` when unknown)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import cases

    return cases.get_cases([name])[0]


# --- The data tables -------------------------------------------------------------------


def _select(text: str | None, table: dict[str, Any], what: str) -> list[str]:
    """Resolve a CSV / ``all`` CLI value against ``table``'s keys, keeping order.

    A bare ``all`` expands to the table's own order. ``ValueError`` names the offending
    entries, so a typo fails loudly instead of silently measuring a subset.
    """
    names = [item.strip() for item in (text or "").split(",") if item.strip()]
    if not names:
        raise ValueError(f"{what} is empty; pass at least one of {sorted(table)} (or 'all')")
    if names == ["all"]:
        return list(table)
    unknown = [name for name in names if name not in table]
    if unknown:
        raise ValueError(f"unknown {what} {unknown!r}; valid names: {sorted(table)} (or 'all')")
    return list(dict.fromkeys(names))


def _workload_table() -> dict[str, Workload]:
    """``{workload.name: workload}`` (the shape :func:`_select` wants)."""
    return {workload.name: workload for workload in WORKLOADS}


def _workload(name: str) -> Workload:
    """One registered workload (``ValueError`` when unknown)."""
    return _workload_table()[name]


def workload_geometry(workload: Workload) -> dict[str, int]:
    """The static shape of one workload, read off its case (needs no numpy).

    ``rows`` is ``pop_size * samples``, the number of evaluated activations of one pass.
    ``ValueError`` when the case's genome config declares no input/output width.
    """
    case = _case(workload.case)
    try:
        num_inputs = int(case.genome_config["num_inputs"])
        num_outputs = int(case.genome_config["num_outputs"])
    except KeyError as exc:
        raise ValueError(
            f"case {case.name!r} does not declare genome_config[{exc.args[0]!r}], "
            "which a forward pass needs"
        ) from None
    return {
        "pop_size": int(case.pop_size), "samples": int(workload.samples),
        "rows": int(case.pop_size) * int(workload.samples),
        "num_inputs": num_inputs, "num_outputs": num_outputs,
    }


def _target_plan(target_name: str, plugin: str | None = None) -> dict[str, Any]:
    """Resolve one named target into a JSON-safe build plan (data, never a code path)."""
    base = TARGETS[target_name]
    options = None if base["compile_options"] is None else copy.deepcopy(base["compile_options"])
    if target_name in PJRT_TARGETS and plugin is not None:
        options = dict(options or {}, plugin_path=plugin)
    return {
        "name": target_name, "impl": base["impl"], "backend": base["backend"],
        "device": base["device"], "compile_options": options,
    }


# --- The xla ``XLA_FLAGS`` workaround (row environment, never a compile option) --------


def _effective_xla_flags(target_name: str, requested: str) -> str | None:
    """The ``XLA_FLAGS`` value this target is measured with (``None`` = leave alone).

    Only a :data:`gpu_targets.PJRT_TARGETS` target gets the ROCm command-buffer
    workaround (:data:`gpu_targets.XLA_WORKAROUND_FLAGS`); every other target --
    including ``iree-rocm`` and the ``jax`` baseline -- inherits the parent
    environment untouched. An empty ``requested`` is a request for the *absence* of
    the workaround (``--xla-flags ''``), so it stays the empty string.
    """
    return requested if target_name in PJRT_TARGETS else None


def _export_xla_flags(
    environment: MutableMapping[str, str], flags: str | None
) -> MutableMapping[str, str]:
    """Apply the workaround rule to ``environment`` in place and return it.

    ``None`` leaves the environment byte-identical; a non-empty value is exported as
    ``XLA_FLAGS``; an empty value **removes** the variable, so "measured without the
    workaround" is a real absence rather than a value inherited from the parent
    process (XLA reads ``XLA_FLAGS`` when its backend is initialised).
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
        "one_forward_configuration_per_process": True,
        "rows_strictly_sequential": True,
        "warmup_forward_passes": args.warmup, "timed_forward_passes": args.passes,
        "seed": args.seed,
        "timed_region": "one forward pass of the whole population: no mutation, no "
        "selection and no environment step inside the timed region",
        "timing_clock": "time.perf_counter() around each pass, including the host "
        "materialisation of the output block",
        "input_block": "a fixed (rows, num_inputs) float32 array drawn once from numpy "
        "default_rng(seed) and pre-placed on the target device; nothing is transferred "
        "inside the timed region",
        "payload": "Pipeline.setup() (new) / algorithm.setup() (old) initial population",
        "build_measured_separately": True, "build_charged_to_a_pass": False,
        "peak_rss": "resource.getrusage(RUSAGE_SELF).ru_maxrss in MiB -- a "
        "process-lifetime high-water mark, which is why one row is one process",
        "logging_in_timed_region": False,
        "jax_allocator": f"old-side rows pin {JAX_PREALLOCATE_ENV}=false before importing "
        "jax and record it; the default 75 % reservation fails rows on a shared device",
        "float_rtol": FLOAT_RTOL, "float_atol": FLOAT_ATOL,
        "errors": "recorded as status='error' rows carrying the traceback, never dropped",
    }

# --- Per-row environment probes --------------------------------------------------------


def _prepare_jax_allocator() -> str:
    """Pin the baseline's XLA allocator; return the policy string it resolved to.

    The variable is read when the XLA client is created, so this must run before
    ``import jax``; a process that already imported it would silently keep the default,
    which is why that case raises instead of recording a policy that did not apply.
    """
    if "jax" in sys.modules:
        raise RuntimeError(
            f"{JAX_PREALLOCATE_ENV} must be set before jax is imported; this process "
            "already imported it, so the policy would not apply"
        )
    return f"{JAX_PREALLOCATE_ENV}={os.environ.setdefault(JAX_PREALLOCATE_ENV, 'false')}"


def _vram_mib() -> dict[str, Any]:
    """Best-effort device memory of GPU 0, so a shared box is visible in the record.

    A wide func-fit forward allocates hundreds of megabytes on the device and this box
    is shared, so a co-tenant's resident VRAM belongs next to the timing it may have
    disturbed. A failing probe is recorded, never raised.
    """
    block: dict[str, Any] = {"vram_total_mib": None, "vram_used_mib": None, "error": None}
    try:
        completed = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        block["error"] = f"{type(exc).__name__}: {exc}"
        return block
    labels = {"VRAM Total Memory (B):": "vram_total_mib",
              "VRAM Total Used Memory (B):": "vram_used_mib"}
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    for line in text.splitlines():
        for label, key in labels.items():
            if label in line and block[key] is None:
                try:
                    block[key] = round(int(line.split(label, 1)[1].strip()) / (1024 * 1024), 1)
                except ValueError:
                    block["error"] = f"unparsable {label} line: {line.strip()!r}"
    if block["vram_total_mib"] is None and block["error"] is None:
        block["error"] = f"rocm-smi exited {completed.returncode} without the VRAM rows"
    return block


# --- The two sides ---------------------------------------------------------------------


def _new_query(
    tn: Any, case: Any, workload: Workload, plan: dict[str, Any],
    args: argparse.Namespace, record: dict[str, Any],
) -> Any:
    """Build the new side's forward graph and return its zero-argument query.

    Writes ``setup_s``, ``build_s``, ``device_info`` and ``jax_imported`` into
    ``record``; setup and build are reported separately, never charged to a pass.
    Anything ``Pipeline.setup``/``evoxc.build`` raises propagates to the caller.
    """
    from tensorneat.common import to_numpy
    from tensorneat.genome.utils import GenomeTensors
    from tensorneat.host_payload import _NetworkActFn

    evoxc, numpy = _evoxc(), _numpy()
    algorithm, problem = case.build(tn, "new")
    pipeline = tn.Pipeline(
        algorithm=algorithm, problem=problem, seed=args.seed,
        fitness_target=case.fitness_target, generation_limit=1,
        backend=plan["backend"], device=plan["device"],
        compile_options=plan["compile_options"], is_detailed=False,
    )
    started = time.perf_counter()
    state = pipeline.setup()
    record["setup_s"] = round(time.perf_counter() - started, 6)
    # Fairness evidence, per row: a new-side row must not have pulled JAX in.
    record["jax_imported"] = "jax" in sys.modules

    geometry = workload_geometry(workload)
    rows = geometry["rows"]
    config = pipeline._static_config(state)
    device_kind = pipeline._device_kind()

    def kernel(node_keys: Any, node_attrs: Any, conn_keys: Any, conn_attrs: Any,
               inputs: Any) -> Any:
        """The generation's forward half: transform -> prepare(P*S) -> query."""
        pop = GenomeTensors(node_keys, node_attrs, conn_keys, conn_attrs)
        current = config.register(pop=pop)
        transformed = algorithm.transform(current, pop)
        act_fn = _NetworkActFn(
            lambda: (current, transformed, pop), algorithm.forward_prepared,
            algorithm.prepare_forward, device_kind)
        return act_fn.prepare(rows)(inputs)

    specs = [evoxc.TensorSpec(tuple(field.shape), field.dtype) for field in state.pop]
    specs.append(evoxc.TensorSpec((rows, geometry["num_inputs"]), numpy.dtype("float32")))
    started = time.perf_counter()
    executable = evoxc.build(kernel, *specs, **pipeline._build_options(state))
    record["build_s"] = round(time.perf_counter() - started, 6)

    device = pipeline._placement_device()

    def place(host: Any) -> Any:
        """Copy one host array onto the target device (the only transfer, done once)."""
        tensor = evoxc.from_numpy(numpy.ascontiguousarray(host))
        return tensor.to(device) if device is not None else tensor

    payload_args = [place(to_numpy(field)) for field in state.pop]
    inputs = numpy.random.default_rng(args.seed).random(
        (rows, geometry["num_inputs"]), dtype=numpy.float32)
    inputs_arg = place(inputs)
    record["device_info"] = {
        "impl": "new", "backend": plan["backend"],
        "device": None if device is None else f"{device.kind}:{device.index}",
        "device_kind": device_kind, "compile_options": plan["compile_options"],
        "payload_device": str(state.pop.node_keys.device),
        "max_nodes": int(state.pop.node_keys.shape[1]),
        "max_conns": int(state.pop.conn_keys.shape[1]),
    }

    def query() -> Any:
        """Run one forward pass and materialise its output block on the host."""
        return to_numpy(evoxc.run(executable, *payload_args, inputs_arg))

    return query


def _old_query(workload: Workload, args: argparse.Namespace, record: dict[str, Any]) -> Any:
    """Build the frozen JAX baseline's forward and return its zero-argument query.

    Genome and algorithm come straight from the case's config dicts
    (``case._build_genome``/``_build_algorithm``) rather than ``case.build(tn, "old")``,
    because a forward pass needs no *problem*: ``Case.impls`` gates the problem, and
    ``cartpole_p100`` is new-only while its 4 -> 2 geometry is an ordinary
    ``DefaultGenome``. Such rows set ``device_info["genome_only"]``.
    """
    jax, numpy, tnl = _jax(), _numpy(), _legacy()
    case = _case(workload.case)
    genome = case._build_genome(tnl)
    algorithm = case._build_algorithm(tnl, genome)
    geometry = workload_geometry(workload)
    pop_size = geometry["pop_size"]
    if int(algorithm.pop_size) != pop_size:
        raise ValueError(
            f"workload {workload.name!r}: case {case.name!r} builds pop "
            f"{algorithm.pop_size} but the workload declares {pop_size}")
    started = time.perf_counter()
    state = tnl.common.State().register(randkey=jax.random.PRNGKey(args.seed))
    state = algorithm.setup(state)
    pop = algorithm.ask(state)
    transformed = jax.vmap(algorithm.transform, in_axes=(None, 0))(state, pop)
    jax.block_until_ready(transformed)  # the schedule derivation is graph code
    record["setup_s"] = round(time.perf_counter() - started, 6)

    inputs = numpy.random.default_rng(args.seed).random(
        (pop_size, geometry["samples"], geometry["num_inputs"]), dtype=numpy.float32)

    def batch_forward(transformed: Any, inputs: Any) -> Any:
        """The legacy evaluate composition: vmap(population) of vmap(samples)."""

        def per_individual(one: Any, sample_block: Any) -> Any:
            return jax.vmap(algorithm.forward, in_axes=(None, None, 0))(state, one, sample_block)

        return jax.vmap(per_individual)(transformed, inputs)

    started = time.perf_counter()
    compiled = jax.jit(batch_forward).lower(transformed, inputs).compile()
    record["build_s"] = round(time.perf_counter() - started, 6)
    transformed_arg = jax.device_put(transformed)
    inputs_arg = jax.device_put(inputs)
    record["device_info"] = {
        "impl": "old", "backend": "jax", "jax_version": getattr(jax, "__version__", None),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(item) for item in jax.devices()],
        "genome_only": "old" not in case.impls,
        "max_nodes": int(genome.max_nodes), "max_conns": int(genome.max_conns),
    }

    def query() -> Any:
        """Run one forward pass and materialise its output block on the host."""
        return numpy.asarray(compiled(transformed_arg, inputs_arg))

    return query


# --- The measurement -------------------------------------------------------------------


def _peak_rss_mib() -> float:
    """This process's peak resident set size in MiB (``ru_maxrss`` is KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _fingerprint(output: Any) -> dict[str, Any]:
    """``{"shape", "dtype", "mean", "finite_fraction"}`` -- a pass cannot compute nothing."""
    numpy = _numpy()
    array = numpy.asarray(output)
    finite = numpy.isfinite(array)
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "mean": float(numpy.mean(array[finite])) if array.size and bool(finite.any()) else None,
        "finite_fraction": float(numpy.mean(finite)) if array.size else None,
    }


def _timed_passes(query: Any, args: argparse.Namespace) -> list[float]:
    """Run the untimed warmup plus the timed passes; return per-pass milliseconds.

    The whole region runs with stdout/stderr on ``os.devnull`` so no logging lands
    inside a sample.
    """
    with (
        open(os.devnull, "w", encoding="utf-8") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        for _ in range(args.warmup):
            query()
        samples: list[float] = []
        for _ in range(args.passes):
            started = time.perf_counter()
            query()
            samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _measure(
    workload: Workload, plan: dict[str, Any], args: argparse.Namespace, record: dict[str, Any],
) -> None:
    """Fill ``record`` with the measured row (raises on any failure).

    Writing into the caller's dict is deliberate: an exception halfway through leaves
    every already-measured field on the ``error`` record.
    """
    case = _case(workload.case)
    record["case_config"] = case.as_dict()
    if plan["impl"] == "new":
        query = _new_query(_tensorneat(), case, workload, plan, args, record)
    elif plan["impl"] == "old":
        query = _old_query(workload, args, record)
    else:  # pragma: no cover - TARGETS is static data validated at import
        raise ValueError(f"target {plan['name']!r} has unknown impl {plan['impl']!r}")
    record["output"] = _fingerprint(query())  # one untimed pass, also warms the backend
    milliseconds = _timed_passes(query, args)
    record["peak_rss_mib"] = round(_peak_rss_mib(), 2)  # after the timed region
    median_ms = statistics.median(milliseconds)
    rows = int(record["rows"])
    record["ms_per_forward"] = {
        "mean": round(statistics.fmean(milliseconds), 4), "median": round(median_ms, 4),
        "min": round(min(milliseconds), 4), "max": round(max(milliseconds), 4),
        "samples": [round(value, 4) for value in milliseconds],
    }
    record["forwards_per_second"] = round(1000.0 / median_ms, 4) if median_ms > 0 else None
    record["rows_per_second"] = round(rows * 1000.0 / median_ms, 4) if median_ms > 0 else None


def measure_row(args: argparse.Namespace) -> dict[str, Any]:
    """Measure ``args.workload``/``args.target`` in this process and return the record.

    Never raises for a measurement failure: the ``error`` row carries the full
    traceback plus whatever was measured before the failure. A PJRT target resolves its
    plugin first -- the path goes into ``compile_options["plugin_path"]`` *and* into
    ``$EVOXC_PJRT_PLUGIN``, which pins the same ``.so`` for any option-less discovery
    call -- and an unresolvable plugin is a named ``error`` row, never a fallback to
    another backend. The row also records the ``XLA_FLAGS`` value it ran with
    (``record["xla_flags"]``; ``None`` for a target that does not use XLA) and exports
    it into :data:`os.environ` before the graph is built, because XLA reads the
    variable at backend init.
    """
    workload = _workload(args.workload)
    base = TARGETS[args.target]
    record: dict[str, Any] = {
        "workload": workload.name, "case": workload.case, "target": args.target,
        "impl": base["impl"], "backend": base["backend"], "device": base["device"],
        "compile_options": copy.deepcopy(base["compile_options"]),
        "seed": args.seed, **workload_geometry(workload),
        "warmup_passes": args.warmup, "timed_passes": args.passes,
        "xla_flags": _effective_xla_flags(args.target, args.xla_flags),
        "protocol": _protocol(args), "status": "ok", "error": None,
        "jax_imported": None, "jax_allocator": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Before any backend init: XLA reads the variable when its client is created.
    _export_xla_flags(os.environ, record["xla_flags"])
    try:
        plugin = plugin_for(args.target, args.pjrt_plugin)
    except PluginError as exc:
        return {**record, "status": "error", "error": str(exc), "plugin_error": True}
    if plugin is not None:
        os.environ[PJRT_ENV_VAR] = plugin  # pins option-less discovery to the same .so
        record["pjrt_plugin"] = fingerprint_plugin(plugin)
    if base["impl"] == "old":  # before any jax import: the client reads it at creation
        try:
            record["jax_allocator"] = _prepare_jax_allocator()
        except RuntimeError as exc:
            return {**record, "status": "error", "error": str(exc), "protocol_error": True}
    plan = _target_plan(args.target, plugin)
    record.update(backend=plan["backend"], device=plan["device"],
                  compile_options=plan["compile_options"])
    record["env"] = environment()
    record["device_memory"] = _vram_mib()
    record["device_info"] = {
        **plan, "hip_devices": record["env"]["iree"]["hip_devices"],
        "hip_error": record["env"]["iree"]["hip_error"],
    }
    try:
        _measure(workload, plan, args, record)
    except Exception:
        record["status"] = "error"
        record["error"] = traceback.format_exc()
    return record


def _fmt(value: Any, digits: int = 3) -> str:
    """Format one number, ``n/a`` when it is absent."""
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def render_row(record: dict[str, Any]) -> str:
    """Render one row record as the aligned table ``run`` prints."""
    head = (f"forward_measure: {record['workload']} / {record['target']} / impl "
            f"{record['impl']} / backend {record['backend']} / device {record['device']!r} "
            f"/ seed {record['seed']}")
    if record.get("status") != "ok":
        lines = [head, "", "status: error",
                 str(record.get("error") or "(no error recorded)").rstrip()]
        if "device_info" in record:  # a row that never built a forward has none
            lines += ["", f"device_info: {record['device_info']!r}"]
        return "\n".join(lines)
    timing, output = record["ms_per_forward"], record.get("output") or {}
    cells = [
        ("population x samples", f"{record['pop_size']} x {record['samples']}"),
        ("evaluated rows per pass", str(record["rows"])),
        ("setup (pipeline / algorithm)", f"{record['setup_s']:.3f} s"),
        ("forward graph build", f"{record['build_s']:.3f} s"),
        ("warmup / timed passes", f"{record['warmup_passes']} / {record['timed_passes']}"),
        ("ms/pass mean", f"{timing['mean']:.3f} ms"),
        ("ms/pass median", f"{timing['median']:.3f} ms"),
        ("ms/pass min / max", f"{timing['min']:.3f} / {timing['max']:.3f} ms"),
        ("passes/second (from median)", f"{_fmt(record['forwards_per_second'])} /s"),
        ("rows/second (from median)", f"{_fmt(record['rows_per_second'])} /s"),
        ("output shape / mean",
         f"{output.get('shape')} / {_fmt(output.get('mean'))}"),
        ("peak RSS (process high-water)", f"{record['peak_rss_mib']:.2f} MiB"),
    ]
    return "\n".join(
        [head, "", f"{'metric':<38}{'value':>24}", f"{'-' * 38} {'-' * 24:>24}"]
        + [f"{label:<38}{value:>24}" for label, value in cells]
        + ["", f"compile_options: {record['compile_options']!r}",
           f"jax_allocator: {record['jax_allocator']!r}",
           f"device_info: {record['device_info']!r}",
           "build and setup are one-off, reported separately; peak RSS is a "
           "process-lifetime high-water mark (one row per process)."])

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


# --- Matrix driver ---------------------------------------------------------------------


def _child_command(
    args: argparse.Namespace, script: Path, workload: str, target: str, row_path: Path,
    plugin: str | None,
) -> list[str]:
    """The ``run`` argv that measures one matrix row in a fresh process."""
    command = [
        sys.executable, str(script), "run", "--workload", workload, "--target", target,
        "--warmup", str(args.warmup), "--passes", str(args.passes),
        "--seed", str(args.seed), "--json", str(row_path),
    ]
    if plugin is not None:
        command += [PJRT_FLAG, plugin]
    # Forwarded explicitly so `--xla-flags ''` reaches the child as an empty value.
    # The `=` form is required: the default value itself begins with `--`.
    return command + [f"--xla-flags={args.xla_flags}"]


def _child_environment(plugin: str | None, xla_flags: str | None) -> dict[str, str]:
    """This process's environment, plus the plugin and the xla workaround.

    ``xla_flags`` follows :func:`_effective_xla_flags`: exported as ``XLA_FLAGS`` for
    an xla target, removed for an empty value (``--xla-flags ''``), and untouched
    (``None``) for every other target.
    """
    child = dict(os.environ)
    if plugin is not None:
        child[PJRT_ENV_VAR] = plugin
    return _export_xla_flags(child, xla_flags)


def _crash_row(
    workload: str, target: str, command: list[str], reason: str, returncode: int | None,
    stderr_text: str | None, args: argparse.Namespace,
) -> dict[str, Any]:
    """Synthesise an ``error`` row for a child that wrote no record."""
    tail = "\n".join(str(stderr_text or "").splitlines()[-20:])[-4000:]
    parts = [reason] + (["", "--- child stderr tail ---", tail] if tail else [])
    return {
        "workload": workload, "target": target, "impl": TARGETS[target]["impl"],
        "backend": TARGETS[target]["backend"], "status": "error",
        "error": "\n".join(parts), "returncode": returncode,
        "child_command": shlex.join(command), "stderr_tail": tail,
        "xla_flags": _effective_xla_flags(target, args.xla_flags),
        "protocol": _protocol(args),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _read_child_row(
    row_path: Path, workload: str, target: str, command: list[str],
    completed: subprocess.CompletedProcess[str], args: argparse.Namespace,
) -> dict[str, Any]:
    """Read the child's JSON record, or synthesise an error row explaining why not."""
    if not row_path.exists():
        return _crash_row(
            workload, target, command,
            f"the child process exited {completed.returncode} without writing a record",
            completed.returncode, completed.stderr, args)
    try:
        record = json.loads(row_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _crash_row(
            workload, target, command,
            f"the child's record at {row_path.name} could not be read: "
            f"{type(exc).__name__}: {exc}", completed.returncode, completed.stderr, args)
    if not isinstance(record, dict) or "status" not in record:
        return _crash_row(
            workload, target, command,
            f"the child's record at {row_path.name} is not a row record (no 'status')",
            completed.returncode, completed.stderr, args)
    return record


def _document_meta(
    args: argparse.Namespace, session: dict[str, str], workloads: list[Workload],
    targets: list[str], pjrt_plugin: dict[str, Any] | None,
) -> dict[str, Any]:
    """The ``meta`` block of a matrix document (command, title/note, env, protocol).

    ``xla_flags`` is the ``XLA_FLAGS`` value the ``xla`` rows ran with, and ``None``
    when no planned target is an xla target (a numpy/iree/jax matrix records no flag).
    """
    meta = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "report_title": args.report_title, "report_note": args.report_note,
        "created_utc": session["start"], "session_utc": session,
        "protocol": _protocol(args),
        "workloads": [workload.name for workload in workloads],
        "workload_geometry": {workload.name: workload_geometry(workload)
                              for workload in workloads},
        "targets": targets,
        "xla_flags": args.xla_flags if any(t in PJRT_TARGETS for t in targets) else None,
        "env": environment(),
    }
    if pjrt_plugin is not None:
        meta["pjrt_plugin"] = pjrt_plugin
    return meta


def _summary_line(index: int, total: int, record: dict[str, Any]) -> str:
    """One-line per-row summary printed by ``matrix``."""
    head = (f"[{index}/{total}] {record.get('workload', '?')} / {record.get('target', '?')}: "
            f"{str(record.get('status', 'error')).upper()}")
    if record.get("status") != "ok":
        return f"{head} — {last_error_line(record.get('error'))}"
    timing = record.get("ms_per_forward") or {}
    return (f"{head} — setup {_fmt(record.get('setup_s'))} s, build "
            f"{_fmt(record.get('build_s'))} s, {_fmt(timing.get('median'))} ms/pass, "
            f"{_fmt(record.get('rows_per_second'), 0)} rows/s")


def matrix_command(args: argparse.Namespace) -> int:
    """``matrix``: measure every planned row, one fresh process each, sequentially."""
    try:
        workloads = [_workload(name) for name in
                     _select(args.workloads, _workload_table(), "--workloads")]
        targets = _select(args.targets, TARGETS, "--targets")
    except ValueError as exc:
        print(f"forward_measure matrix: {exc}", file=sys.stderr)
        return 2
    # Resolve the plugin once, before any row: a missing plugin must not cost a session.
    try:
        plugin, plugin_fingerprint = require_plugin(targets, args.pjrt_plugin)
    except PluginError as exc:
        print(f"forward_measure matrix: {exc}", file=sys.stderr)
        return 2
    plan = [(workload.name, target) for workload in workloads for target in targets]
    script, records = Path(__file__).resolve(), []
    started = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(prefix="forward_measure_") as tmp:
        for index, (workload, target) in enumerate(plan, 1):
            row_path = Path(tmp) / f"row_{index:03d}.json"
            command = _child_command(args, script, workload, target, row_path, plugin)
            # Per row, because the workaround applies to xla targets only.
            child_env = _child_environment(
                plugin, _effective_xla_flags(target, args.xla_flags))
            print(f"[{index}/{len(plan)}] {workload} / {target}: running", flush=True)
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=args.timeout,
                    env=child_env)
            except subprocess.TimeoutExpired as exc:
                record = _crash_row(
                    workload, target, command, f"the child timed out after {args.timeout} s",
                    None, exc.stderr if isinstance(exc.stderr, str) else None, args)
            except OSError as exc:
                record = _crash_row(
                    workload, target, command,
                    f"could not spawn the child process: {type(exc).__name__}: {exc}",
                    None, None, args)
            else:
                record = _read_child_row(row_path, workload, target, command, completed, args)
            records.append(record)
            print(_summary_line(index, len(plan), record), flush=True)
    ended = datetime.now(timezone.utc)
    session = {"start": started.isoformat(timespec="seconds"),
               "end": ended.isoformat(timespec="seconds")}
    document = {"schema": SCHEMA,
                "meta": _document_meta(args, session, workloads, targets, plugin_fingerprint),
                "records": records}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.json}")

    blocked = [record for record in records if record.get("status") != "ok"]
    for record in blocked:
        print(f"BLOCKED: {record.get('workload', '?')} / {record.get('target', '?')}: "
              f"{last_error_line(record.get('error'))}")
    print(f"{len(records) - len(blocked)}/{len(records)} row(s) ok, {len(blocked)} error")
    if args.out is not None:
        try:
            splice_block(args.out, render_block(document, record_path=args.json),
                         begin_marker=BEGIN_MARKER, end_marker=END_MARKER)
        except RenderError as exc:
            print(f"forward_measure matrix: {exc}", file=sys.stderr)
            return 2
        print(f"rendered the measured block of {args.out} from {args.json}")
    return 0 if not blocked else 1


# --- The document ----------------------------------------------------------------------


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
    """Load and validate a matrix document (``RenderError`` on anything unusable)."""
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


def _find(document: dict[str, Any], workload: str, target: str) -> dict[str, Any] | None:
    """The row of ``(workload, target)``, or ``None`` when the record has no such row."""
    for record in _records(document):
        if record.get("workload") == workload and record.get("target") == target:
            return record
    return None


def _median_ms(record: dict[str, Any] | None) -> float | None:
    """One row's median ms per forward pass, ``None`` unless the row is ``ok``."""
    if not record or record.get("status") != "ok":
        return None
    value = (record.get("ms_per_forward") or {}).get("median")
    return float(value) if isinstance(value, (int, float)) else None


def _spread_ms(record: dict[str, Any] | None) -> str:
    """One row's ``min – max`` per-pass range, ``n/a`` without timings."""
    timing = (record or {}).get("ms_per_forward") or {}
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


# --- The rendered block ----------------------------------------------------------------


def _session_text(session: dict[str, Any] | None) -> str:
    """Render ``meta.session_utc`` as one date/session range."""
    start, end = (session or {}).get("start"), (session or {}).get("end")
    return f"{start} – {end}" if start and end and start != end else (start or end or "n/a")


def _git_text(env: dict[str, Any]) -> str:
    """Render the git commit plus the dirty flag of the code under test."""
    commit = env.get("git_commit")
    if not commit:
        return "n/a"
    if env.get("git_dirty") is None:
        return f"`{commit[:12]}` (dirty flag n/a)"
    if not env.get("git_dirty"):
        return f"`{commit[:12]}` (clean — generated report artifacts are not counted)"
    paths = env.get("git_dirty_paths") or []
    named = ", ".join(paths[:5]) + (f" +{len(paths) - 5} more" if len(paths) > 5 else "")
    return f"`{commit[:12]}` (dirty: {named})"


def _jax_of(document: dict[str, Any]) -> dict[str, Any]:
    """The JAX identity an old-side row recorded (``{}`` when none was measured)."""
    for record in _records(document):
        info = record.get("device_info") or {}
        if info.get("impl") == "old" and info.get("jax_version"):
            return info
    return {}


def _vram_text(document: dict[str, Any]) -> str:
    """The ``min – max`` VRAM used during the rows, with the device's total size."""
    probes = [record["device_memory"] for record in _records(document)
              if isinstance(record.get("device_memory"), dict)]
    used = [p["vram_used_mib"] for p in probes if isinstance(p.get("vram_used_mib"), (int, float))]
    total = [p["vram_total_mib"] for p in probes
             if isinstance(p.get("vram_total_mib"), (int, float))]
    if not used:
        return "n/a (rocm-smi could not be read)"
    size = f"{total[0]:.0f} MiB total" if total else "total n/a"
    return f"{min(used):.0f} – {max(used):.0f} MiB used of {size}"


def _jax_allocator_text(document: dict[str, Any]) -> str:
    """The XLA allocator policy the baseline rows ran with (``n/a`` when none)."""
    for record in _records(document):
        if record.get("impl") == "old" and record.get("jax_allocator"):
            return str(record["jax_allocator"])
    return "n/a (no old-side row)"


def _plugin_text(plugin: dict[str, Any]) -> str:
    """A PJRT plugin fingerprint as one cell: basename, size, 12-hex digest."""
    size, digest = plugin.get("size_bytes"), plugin.get("sha256")
    mib = f"{size / (1024 * 1024):.1f} MiB" if isinstance(size, (int, float)) else "n/a"
    name = plugin.get("basename") or plugin.get("path") or "n/a"
    return f"`{name}` ({mib}, sha256 `{digest[:12] if digest else 'n/a'}…`)"


def _xla_flags_text(document: dict[str, Any]) -> str:
    """The ``XLA_FLAGS`` the record's xla rows ran with -- never left implicit.

    A non-empty ``meta["xla_flags"]`` names the workaround the session exported; a
    key present but empty while the record holds xla rows says those numbers were
    measured *without* it (``--xla-flags ''``); a record with no key predates the
    workaround and is attributed to no flag at all; and a key of ``None`` with no xla
    row means no flag applied because no target uses XLA.
    """
    meta = document.get("meta") or {}
    if "xla_flags" not in meta:
        return "not recorded (record predates the workaround)"
    flags = meta["xla_flags"]
    if isinstance(flags, str) and flags:
        return (
            f"`{flags}` (ROCm command-buffer workaround: the RL rollout's long "
            "`while` SIGSEGVs under FUSION capture; a forward pass has no `while`)"
        )
    if any(record.get("target") in PJRT_TARGETS for record in _records(document)):
        return "not set (`--xla-flags ''`: measured without the command-buffer workaround)"
    return "not applicable (no `xla` row in this record)"


def _environment_section(document: dict[str, Any]) -> list[str]:
    """Section 1: the measured environment fingerprint as a two-column table."""
    meta = document.get("meta") or {}
    env = meta.get("env") or {}
    gpu = env.get("gpu") or {}
    rocm = gpu.get("rocm_smi") or {}
    iree = env.get("iree") or {}
    devices = iree.get("hip_devices") or []
    model = rocm.get("gpu_model") or (devices[0].get("name") if devices else None)
    gfx, plugin, loadavg = rocm.get("gfx_version"), meta.get("pjrt_plugin"), env.get("loadavg")
    jax_info = _jax_of(document)
    hip = ", ".join(f"device_id={item.get('device_id')} {item.get('name')}" for item in devices)
    memory = f"{env['memory_mb']} MB" if env.get("memory_mb") else "n/a"
    rows = [
        ["Date / session (UTC)", _session_text(meta.get("session_utc"))],
        ["GPU model (gfx version)",
         f"{model} ({gfx})" if model and gfx else (model or gfx or "n/a")],
        ["HIP devices (IREE runtime query)",
         hip or (f"n/a ({iree['hip_error']})" if iree.get("hip_error") else "n/a")],
        ["Device memory during the rows", _vram_text(document)],
        ["ROCm version", gpu.get("rocm_version") or "n/a"],
        ["IREE compiler / runtime",
         f"{iree.get('compiler') or 'n/a'} / {iree.get('runtime') or 'n/a'}"],
        ["JAX (old side)",
         f"{jax_info.get('jax_version') or env.get('jax') or 'n/a'}; backend "
         f"{jax_info.get('jax_default_backend') or 'n/a'}; devices "
         f"{', '.join(jax_info.get('jax_devices') or []) or 'n/a'}; allocator "
         f"{_jax_allocator_text(document)}"],
        *([["PJRT plugin (xla targets)", _plugin_text(plugin)]] if plugin else []),
        ["XLA_FLAGS (xla targets)", _xla_flags_text(document)],
        ["tensorneat / evoxc", f"{env.get('tensorneat') or 'n/a'} / {env.get('evoxc') or 'n/a'}"],
        ["numpy", env.get("numpy") or "n/a"],
        ["Python", env.get("python") or "n/a"],
        ["Platform", env.get("platform") or "n/a"],
        ["CPU cores / RAM", f"{env.get('cpu_count') or 'n/a'} / {memory}"],
        ["Threads (inherited env)", env.get("threads") or "runtime defaults"],
        ["Git commit (code under test)", _git_text(env)],
        ["Load average (1/5/15 min)",
         " / ".join(f"{value:.2f}" for value in loadavg) if loadavg else "n/a"],
    ]
    return ["## Environment", "",
            "Measured conditions, not a verdict: these numbers are comparable only with "
            "another session on the same box and tree.", "",
            *_table(["Item", "Value"], rows)]


def _measured_section(document: dict[str, Any]) -> list[str]:
    """Section 2: one row per measured ``(workload, target)`` pair."""
    meta = document.get("meta") or {}
    rows = []
    for name in meta.get("workloads") or []:
        for target in meta.get("targets") or []:
            record = _find(document, name, target)
            if record is None:  # a planned row without a record is a defect, said loudly
                rows.append([f"`{name}`", target, "n/a", "n/a", "n/a", "n/a",
                             "missing row (never measured — a defect in the record)"])
                continue
            timing, ok = record.get("ms_per_forward") or {}, record.get("status") == "ok"
            rows.append([
                f"`{name}`", target, str(record.get("rows", "n/a")),
                _fmt(timing.get("median")) if ok else "n/a", _spread_ms(record),
                _fmt(record.get("rows_per_second"), 0) if ok else "n/a",
                _fmt(record.get("peak_rss_mib"), 1) if ok else "n/a",
                "ok" if ok else f"error — {last_error_line(record.get('error'))}",
            ])
    return ["## Measured (one forward pass of the whole population)", "",
            "`median` is the median over the timed passes of one fresh process and "
            "`min – max` that row's own per-pass spread -- the uncertainty measure here (a "
            "single median is not a distribution). Build and setup are one-off: they are "
            "reported in the record, never charged to a pass.", "",
            *_table(["Workload", "Target", "Rows", "Median ms", "Min – max ms", "Rows/s",
                     "Peak RSS MiB", "Status"], rows)]


def _speedup_section(document: dict[str, Any]) -> list[str]:
    """Section 3: each new-side target against the frozen JAX baseline, per workload."""
    meta = document.get("meta") or {}
    targets = list(meta.get("targets") or [])
    new_targets = [name for name in targets if TARGETS.get(name, {}).get("impl") == "new"]
    baselines = [name for name in targets if TARGETS.get(name, {}).get("impl") == "old"]
    if not baselines:
        return ["## Speed-up against the frozen JAX baseline", "",
                "No JAX baseline row was planned in this session (`--targets` carries no "
                "`jax`), so this record holds new-side timings only and no speed-up can be "
                "computed from it."]
    baseline = baselines[0]
    header = ["Workload", f"{baseline} median ms"]
    for name in new_targets:
        header += [f"{name} median ms", f"speed-up vs {baseline}"]
    rows = []
    for workload in meta.get("workloads") or []:
        base_ms = _median_ms(_find(document, workload, baseline))
        cells = [f"`{workload}`", _fmt(base_ms)]
        for name in new_targets:
            target_ms = _median_ms(_find(document, workload, name))
            cells += [_fmt(target_ms), _speedup(base_ms, target_ms)]
        rows.append(cells)
    return ["## Speed-up against the frozen JAX baseline", "",
            "Speed-up is `jax median / new median` per forward pass (higher means the new "
            "side is faster), the same `old/new` convention as `RESULTS.md`; `n/a` means one "
            "of the two legs has no timing in this record — see `## Blocked rows`.", "",
            *_table(header, rows)]


def _agreement_section(document: dict[str, Any]) -> list[str]:
    """Section 4: the new-side targets against the numpy leg on the same payload."""
    meta = document.get("meta") or {}
    new_targets = [name for name in meta.get("targets") or []
                   if TARGETS.get(name, {}).get("impl") == "new"]
    if "numpy" not in new_targets:
        return ["## Numerical agreement (new-side targets vs the numpy leg)", "",
                "The `numpy` leg was not planned in this session, so there is no reference to "
                "compare the accelerated legs against."]
    rows = []
    for workload in meta.get("workloads") or []:
        reference = (_find(document, workload, "numpy") or {}).get("output") or {}
        ref_mean, shape = reference.get("mean"), reference.get("shape")
        for name in new_targets:
            if name == "numpy":
                continue
            output = (_find(document, workload, name) or {}).get("output") or {}
            mean, other_shape = output.get("mean"), output.get("shape")
            if not isinstance(ref_mean, (int, float)) or not isinstance(mean, (int, float)):
                rows.append([f"`{workload}`", name, _fmt(ref_mean), _fmt(mean), "n/a",
                             "skipped — a leg has no output fingerprint"])
                continue
            delta = abs(float(mean) - float(ref_mean))
            bound = FLOAT_ATOL + FLOAT_RTOL * abs(float(ref_mean))
            status = "pass" if delta <= bound and other_shape == shape else "fail"
            rows.append([f"`{workload}`", name, f"{float(ref_mean):.8f}", f"{float(mean):.8f}",
                         f"{delta:.3e}", status if other_shape == shape
                         else f"{status} — shape {other_shape} != {shape}"])
    return ["## Numerical agreement (new-side targets vs the numpy leg)", "",
            "Every new-side target runs the same graph on the same payload, so its output must "
            "match the `numpy` leg's: the absolute difference of the output means is checked "
            f"against `{FLOAT_ATOL} + {FLOAT_RTOL} x |reference|`. This is a self-consistency "
            "check that the timed pass computed the forward, not a cross-implementation "
            "comparison: the JAX side draws a different population from a different PRNG, so "
            "its values are not comparable by design (`test/parity/` owns fidelity).", "",
            *_table(["Workload", "Target", "numpy mean", "target mean", "abs delta", "Status"],
                    rows)]


def _blocked_section(document: dict[str, Any]) -> list[str]:
    """Section 5: every row that could not be measured, with its own message."""
    blocked = [record for record in _records(document) if record.get("status") != "ok"]
    if not blocked:
        return ["## Blocked rows", "",
                "None: every planned row measured. This section is written from the record, "
                "so it reappears the moment a leg fails."]
    lines = ["## Blocked rows", "",
             "Each row below is an explicit `error` record with its own message; no number is "
             "quoted for a leg that could not run.", ""]
    for record in blocked:
        lines += [f"### `{record.get('workload', '?')}` / {record.get('target', '?')}", "",
                  f"`{record.get('impl', '?')}` impl, backend `{record.get('backend', '?')}`. "
                  "Last line of the record's error:", "",
                  f"    {last_error_line(record.get('error'))}", ""]
    return lines[:-1]


def _provenance_section(document: dict[str, Any], record_path: Path | str | None) -> list[str]:
    """Section 6: data-derived provenance of the block (tree, session, record)."""
    meta = document.get("meta") or {}
    env = meta.get("env") or {}
    records = _records(document)
    n_ok = sum(1 for record in records if record.get("status") == "ok")
    lines = [
        "## Provenance of this block", "",
        f"This block is rendered from `{display_path(record_path)}`: {len(records)} row(s), "
        f"{n_ok} ok and {len(records) - n_ok} error, measured in the session "
        f"{_session_text(meta.get('session_utc'))} (UTC) at git {_git_text(env)}.",
        "Every row is one `forward_measure.py run` in its own process, so the reported peak "
        "RSS is that process's high-water mark and no JAX thread pool shared a process with "
        "the evoxc interpreter.",
        "Nothing here is a statement about HEAD: every number belongs to that record and to "
        "the tree its fingerprint names.",
    ]
    if meta.get("report_note"):
        lines.append(str(meta["report_note"]))
    if env.get("probe_error"):
        lines.append("The environment fingerprint degraded: "
                     f"{last_error_line(env['probe_error'])} (the measurement itself ran).")
    return lines


def render_block(document: dict[str, Any], record_path: Path | str | None = None) -> str:
    """Render ``FORWARD.md``'s measured block from a matrix document (or a path to one).

    ``record_path`` is the file the block is rendered from, which the provenance
    paragraph names. ``RenderError`` on an invalid or unreadable document.
    """
    if isinstance(document, (str, Path)):
        document = read_document(Path(document))
    validate_document(document)
    lines: list[str] = []
    for section in (_environment_section, _measured_section, _speedup_section,
                    _agreement_section, _blocked_section):
        lines += section(document) + [""]
    lines += _provenance_section(document, record_path)
    return "\n".join(lines).strip("\n")


# --- ``render`` and ``--list`` ---------------------------------------------------------


def render_command(args: argparse.Namespace) -> int:
    """``render``: re-render the measured block from an existing record (no measurement)."""
    try:
        document = read_document(args.json)
        splice_block(args.out, render_block(document, record_path=args.json),
                     begin_marker=BEGIN_MARKER, end_marker=END_MARKER)
    except RenderError as exc:
        print(f"forward_measure render: {exc}", file=sys.stderr)
        return 2
    print(f"rendered the measured block of {args.out} from {args.json} "
          f"({len(document['records'])} record(s))")
    return 0


def list_command(args: argparse.Namespace) -> int:
    """``--list``: the targets, the workloads and the defaults (numpy-free)."""
    del args
    print("targets (named label -> how the forward pass is built):")
    for name, base in TARGETS.items():
        plugin = f"  plugin from {PJRT_FLAG}/${PJRT_ENV_VAR}" if name in PJRT_TARGETS else ""
        print(f"  {name:<12} impl={base['impl']:<4} backend={base['backend']:<6} "
              f"device={base['device']!r:<8} compile_options="
              f"{base['compile_options']!r}{plugin}")
    print()
    print(f"old side (impl=jax): the frozen implementation under {display_path(LEGACY_DIR)}, "
          "on the JAX default device")
    print("new side (impl=new): tensorneat's Pipeline.setup() population, one evoxc "
          "forward graph")
    print()
    print(f"xla targets export XLA_FLAGS={XLA_WORKAROUND_FLAGS!r} by default "
          "(--xla-flags '' measures without it)")
    print()
    print(f"default matrix: workloads {DEFAULT_WORKLOADS}; targets {DEFAULT_TARGETS}; "
          f"seed {DEFAULT_SEED}; {DEFAULT_WARMUP} warmup + {DEFAULT_PASSES} timed passes")
    print(f"default record: {display_path(DEFAULT_JSON)}; default report: "
          f"{display_path(DEFAULT_REPORT)}")
    print()
    print("workloads (case from benchmarks/cases.py; samples = tiled rows per individual):")
    for workload in WORKLOADS:
        geometry = workload_geometry(workload)
        print(f"  {workload.name:<18} case={workload.case:<18} "
              f"pop={geometry['pop_size']:<5} samples={geometry['samples']:<4} "
              f"rows={geometry['rows']:<7} {geometry['num_inputs']} -> "
              f"{geometry['num_outputs']}")
    return 0


# --- CLI -------------------------------------------------------------------------------


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
        "--warmup", type=_non_negative_int, default=DEFAULT_WARMUP,
        help="untimed forward passes before the timed region (default: %(default)s)")
    parser.add_argument(
        "--passes", type=_positive_int, default=DEFAULT_PASSES,
        help="timed forward passes (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--pjrt-plugin", default=None, metavar="PATH",
        help=(f"PJRT C API plugin .so (exporting GetPjRtApi) the xla targets drive: it "
              f"becomes compile_options['plugin_path'] and is exported as {PJRT_ENV_VAR}, "
              "which pins it for option-less discovery calls. Resolution order: this "
              "flag, then "
              f"${PJRT_ENV_VAR}; an xla target with neither is a named error, never a "
              "fallback to another backend"))
    parser.add_argument(
        "--xla-flags", default=XLA_WORKAROUND_FLAGS, metavar="TEXT",
        help=(f"XLA_FLAGS exported for the xla targets only (default: "
              f"{XLA_WORKAROUND_FLAGS!r}, the ROCm command-buffer workaround for a "
              "captured long `while`). It is a process-environment setting because XLA "
              "reads it at backend init, so it cannot be a compile option. Pass "
              "--xla-flags '' to measure WITHOUT the workaround: the variable is then "
              "removed from the child environment, never inherited. Targets that do "
              "not use XLA keep their environment untouched"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI (``--help``/``--list`` must work without numpy)."""
    parser = argparse.ArgumentParser(
        prog="forward_measure.py",
        description=("Measure the NEAT batch forward pass -- numpy / iree rocm / xla rocm on "
                     "the new evoxc implementation against the frozen JAX baseline -- one "
                     "configuration per process, and render the measured block of "
                     "benchmarks/FORWARD.md."),
        epilog=("Try: forward_measure.py --list  |  forward_measure.py run --workload "
                "xor_p500_s4 --target iree-rocm  |  forward_measure.py matrix --targets "
                "numpy,iree-rocm,xla-rocm,jax --out benchmarks/FORWARD.md  |  "
                "forward_measure.py render --json benchmarks/results/forward.json --out "
                "benchmarks/FORWARD.md. An xla target needs --pjrt-plugin PATH or "
                f"${PJRT_ENV_VAR}. 'matrix' exits 1 when any planned row is an error row (a "
                "blocked leg is loud by construction) and 2 when the report could not be "
                "rendered."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="print the targets, the workloads and the defaults")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="measure one (workload, target) row here")
    run.add_argument("--workload", choices=tuple(_workload_table()), default=WORKLOADS[0].name,
                     help="workload name (see --list)")
    run.add_argument("--target", choices=tuple(TARGETS), default="numpy",
                     help="named target (see --list)")
    _add_measurement_arguments(run)
    run.add_argument("--json", type=Path, default=None, metavar="row.json",
                     help="also write this row's record as JSON")

    matrix = subparsers.add_parser("matrix",
                                   help="measure every planned row sequentially, one process each")
    matrix.add_argument("--workloads", default=DEFAULT_WORKLOADS, metavar="a,b",
                        help="comma-separated workload names, or 'all' (default: %(default)s)")
    matrix.add_argument("--targets", default=DEFAULT_TARGETS, metavar="a,b",
                        help="comma-separated target labels, or 'all' (default: %(default)s)")
    _add_measurement_arguments(matrix)
    matrix.add_argument("--timeout", type=_positive_int, default=DEFAULT_TIMEOUT, metavar="SECONDS",
                        help="per-row wall-clock budget before the child is killed (default: %(default)s)")
    matrix.add_argument("--json", type=Path, default=DEFAULT_JSON, metavar="out.json",
                        help="matrix document to write (default: %(default)s)")
    matrix.add_argument("--out", type=Path, default=None, metavar="REPORT",
                        help="refresh this report's measured block (e.g. benchmarks/FORWARD.md)")
    matrix.add_argument("--report-title", default=DEFAULT_TITLE, metavar="TEXT",
                        help="title stored in the record's meta (default: this file's title)")
    matrix.add_argument("--report-note", default=None, metavar="TEXT",
                        help="one-line note stored in the record's meta and rendered in its provenance")

    render = subparsers.add_parser("render", help="re-render the measured block from a record")
    render.add_argument("--json", type=Path, default=DEFAULT_JSON, metavar="record.json",
                        help="matrix document to render from (default: %(default)s)")
    render.add_argument("--out", type=Path, default=DEFAULT_REPORT, metavar="REPORT",
                        help="report whose marked block is replaced (default: %(default)s)")

    subparsers.add_parser("list", help="print the targets, the workloads and the defaults")

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
