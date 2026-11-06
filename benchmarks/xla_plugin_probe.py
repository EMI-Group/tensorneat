#!/usr/bin/env python3
"""Probe: drive evoxc's ``xla`` adapter end to end through a PJRT C API plugin.

Intent
------
This is the standalone, dependency-light smoke gate for evoxc's ``xla`` backend
on an accelerator: it proves that one concrete PJRT C API plugin ``.so`` can be
discovered, negotiated (``GetPjRtApi`` + ABI version gate), compiled and run,
and that the numbers it returns agree with the ``numpy`` interpreter. It is not
a measurement protocol -- it produces no timing and no record; it answers one
yes/no question and returns a shell exit code.

On this box that plugin is the ROCm 7 plugin bundled in the
``jax-rocm7-pjrt==0.10.2`` wheel (``jax_plugins/xla_rocm7/xla_rocm_plugin.so``,
503 453 288 bytes, sha256 ``8e8319b48fcc…``), driving an AMD Instinct MI300X
(``gfx942``) under ROCm 7.1. That wheel is the ``xla-rocm`` extra of
``pyproject.toml``, so ``uv sync --extra xla-rocm`` installs it and every later
sync that asks for the extra keeps it; the plugin is always a ``.so`` named by the
caller, because no CPU PJRT plugin is pip-installable.

Protocol
--------
Both graphs are traced, built and run twice -- once on ``backend="numpy"`` (the
reference interpreter) and once on ``backend="xla"`` with ``device=<kind>``:

1. ``elem_reduce`` -- ``sum(a * b + 1.0, axes=1)``: one elementwise multiply-add
   fused into a reduction (the ordinary graph shape).
2. ``loop_graph`` -- a ``while_loop`` accumulator with a *float32* counter
   (``i < 7``, ``acc + x * 0.5``): the runtime-control-flow shape whose export
   path is the fragile one.

The two fixtures are seeded and fixed, so every invocation compares the same
bytes. For the device leg the executable is built first (which registers the
adapter and its placement provider), the inputs are then placed explicitly with
``t.to(Device(kind, 0))``, the graph runs device-resident, and each output is
copied back with an explicit ``.to(Device("cpu", 0))`` -- evoxc never moves data
across the run boundary by itself. Agreement is the maximum absolute difference
over every output leaf; a graph *matches* when that value is ``<= --tolerance``.

The load-bearing gotcha
-----------------------
``XlaBackend.check_available()`` is called on every backend lookup and takes
**no options**: it calls ``_load_plugin()`` with no argument, so it reads only
``EVOXC_PJRT_PLUGIN`` (or a well-known path under ``/usr/local/lib`` etc.), never
``compile_options["plugin_path"]``. A build that sets only that option therefore
still raises::

    evoxc.core.errors.BackendError: the xla adapter requires a PJRT C API
    plugin library (.so exporting GetPjRtApi) and none was found. ...

This probe therefore exports ``EVOXC_PJRT_PLUGIN`` from the resolved
``--pjrt-plugin`` value *and* passes the same path as ``plugin_path`` for
compile/load. Nothing falls back to another backend, and no path is invented.

Exit codes
----------
0 -- both graphs match within ``--tolerance`` (usable as a smoke gate).
1 -- at least one graph disagrees (or failed to run without being a plugin
     ``BackendError``); the traceback is printed.
2 -- no plugin was named, the named path is not a file, or the adapter rejected
     the plugin (a ``BackendError``, e.g. the ABI gate); named error, no number.

Usage
-----
    # the plugin comes from the xla-rocm extra (`uv sync --extra xla-rocm`),
    # i.e. its jax-rocm7-pjrt wheel, or any XLA build exporting GetPjRtApi
    uv run python benchmarks/xla_plugin_probe.py \\
        --pjrt-plugin /path/to/xla_rocm7/xla_rocm_plugin.so
    EVOXC_PJRT_PLUGIN=/path/to/xla_rocm_plugin.so \\
        uv run python benchmarks/xla_plugin_probe.py

Requires ``numpy`` and ``evoxc`` only; TensorNEAT is never imported.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

import evoxc

#: The environment variable the adapter's ``check_available()`` probe reads.
PJRT_ENV_VAR = "EVOXC_PJRT_PLUGIN"
#: The flag that names the plugin for this probe (wins over the environment).
PJRT_FLAG = "--pjrt-plugin"
#: Chunk size for the plugin's SHA-256 (~0.4 s for the 500 MB ROCm plugin).
HASH_CHUNK_BYTES = 1 << 20


class PluginError(RuntimeError):
    """The PJRT plugin could not be resolved; never a fallback to another backend."""


@evoxc.defn
def elem_reduce(a, b):
    """Elementwise multiply-add fused into an axis-1 reduction."""
    return evoxc.sum(a * b + 1.0, axes=1)


@evoxc.defn
def loop_graph(x, i0):
    """A ``while_loop`` accumulator with a float32 counter (7 iterations)."""

    def cond(carry):
        i, _acc = carry
        return i < 7

    def body(carry):
        i, acc = carry
        return i + 1, acc + x * 0.5

    return evoxc.while_loop(cond, body, (i0, x * 0.0))


def resolve_plugin(
    flag: str | None, environ: dict[str, str] | None = None
) -> str:
    """Resolve the plugin path: ``--pjrt-plugin`` first, then the environment.

    Args:
        flag: The ``--pjrt-plugin`` value, if the caller passed one.
        environ: The environment to read (defaults to :data:`os.environ`).

    Returns:
        The absolute plugin path.

    Raises:
        PluginError: When neither source names a path, or when the named path is
            not a readable file -- naming the source, so a typo surfaces here
            instead of as a ctypes failure inside the adapter.
    """
    environment = os.environ if environ is None else environ
    if flag:
        candidate, source = flag, PJRT_FLAG
    elif environment.get(PJRT_ENV_VAR):
        candidate, source = environment[PJRT_ENV_VAR], f"the {PJRT_ENV_VAR} env var"
    else:
        raise PluginError(
            f"no PJRT C API plugin named: pass {PJRT_FLAG} PATH or set "
            f"{PJRT_ENV_VAR}=PATH (neither named one) -- the xla adapter needs a "
            ".so exporting GetPjRtApi (`uv sync --extra xla-rocm` installs the AMD "
            "one)"
        )
    path = Path(candidate).expanduser()
    if not path.is_file():
        raise PluginError(
            f"{source} names {candidate!r}, which is not a file: the xla adapter "
            "needs a PJRT C API plugin library (.so exporting GetPjRtApi) there"
        )
    return str(path.resolve())


def fingerprint_plugin(path: str) -> dict[str, Any]:
    """The identity of one plugin file: basename, size in bytes and SHA-256.

    Raises:
        PluginError: When the file cannot be stat'd or read, so an unreadable
            plugin is loud instead of a silently incomplete fingerprint.
    """
    resolved = Path(path)
    digest = hashlib.sha256()
    try:
        size = resolved.stat().st_size
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PluginError(
            f"cannot fingerprint the PJRT plugin at {str(resolved)!r}: {exc}"
        ) from exc
    return {"basename": resolved.name, "size_bytes": size, "sha256": digest.hexdigest()}


def _host(tensor: Any) -> np.ndarray:
    """Copy one evoxc output to a host numpy array (explicit device-to-host)."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if not isinstance(tensor, evoxc.Tensor):
        raise TypeError(f"expected an evoxc Tensor, got {type(tensor).__name__}")
    return np.asarray(tensor.to(evoxc.Device("cpu", 0)).numpy())


def _leaves(outputs: Any) -> tuple[np.ndarray, ...]:
    """The flat tuple of host arrays behind an ``evaluate``/``run`` result."""
    items = outputs if isinstance(outputs, tuple) else (outputs,)
    return tuple(_host(item) for item in items)


def evaluate_reference(fn: Any, args: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Run ``fn`` on the ``numpy`` reference interpreter and return host outputs."""
    return _leaves(evoxc.evaluate(fn, *args, backend="numpy"))


def evaluate_device(
    fn: Any, args: tuple[np.ndarray, ...], backend: str, device_kind: str, plugin: str
) -> tuple[np.ndarray, ...]:
    """Build ``fn`` on ``backend``/``device_kind`` and run it device-resident.

    Returns the outputs as host arrays. The executable is built first: that
    registers the adapter and its placement provider, which is what
    ``Tensor.to(Device(kind, 0))`` then uses to stage the inputs. The outputs stay
    on the device until the explicit copy back here.
    """
    device = evoxc.Device(device_kind, 0)
    specs = tuple(
        evoxc.core.TensorSpec(
            shape=tuple(np.asarray(arg).shape), dtype=np.asarray(arg).dtype
        )
        for arg in args
    )
    executable = evoxc.build(
        fn, *specs, backend=backend, device=device_kind, plugin_path=plugin
    )
    placed = tuple(evoxc.core.from_numpy(np.asarray(arg)).to(device) for arg in args)
    return _leaves(evoxc.run(executable, *placed))


def max_abs_difference(
    reference: tuple[np.ndarray, ...], other: tuple[np.ndarray, ...]
) -> float:
    """The largest absolute element-wise difference over every output leaf.

    Raises:
        ValueError: When the two results have a different leaf count or leaf
            shape -- a structural mismatch is a failure, never a skipped leaf.
    """
    if len(reference) != len(other):
        raise ValueError(
            f"output leaf count differs: {len(reference)} (numpy) vs "
            f"{len(other)} (device)"
        )
    worst = 0.0
    for index, (ref, got) in enumerate(zip(reference, other)):
        if ref.shape != got.shape:
            raise ValueError(
                f"output leaf {index} shape differs: {ref.shape} (numpy) vs "
                f"{got.shape} (device)"
            )
        if ref.size == 0:
            continue
        diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
        worst = max(worst, float(np.max(diff)))
    return worst


def run_probe(args: argparse.Namespace, plugin: str) -> int:
    """Run both graphs on both legs, print one line each, and return the exit code."""
    # The load-bearing wiring: the availability probe reads only this variable,
    # so it must carry the same plugin that ``plugin_path`` names.
    os.environ[PJRT_ENV_VAR] = plugin
    fingerprint = fingerprint_plugin(plugin)

    try:
        evoxc_version = version("evoxc")
    except PackageNotFoundError:  # pragma: no cover - installed in this tree
        evoxc_version = "unknown"

    print(f"xla_plugin_probe: evoxc {evoxc_version} / numpy {np.__version__}")
    print(
        f"plugin  {fingerprint['basename']}  {fingerprint['size_bytes']} bytes  "
        f"sha256 {fingerprint['sha256']}"
    )
    print(
        f"device  {args.device}  backend {args.backend}  "
        f"tolerance {args.tolerance:g}"
    )
    print()

    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(8, 16)).astype(np.float32)
    x = rng.normal(size=(4,)).astype(np.float32)
    zero = np.array(0.0, np.float32)

    graphs = (
        ("elem_reduce", elem_reduce, (a, b)),
        ("loop_graph", loop_graph, (x, zero)),
    )
    failures = 0
    for name, fn, graph_args in graphs:
        reference = evaluate_reference(fn, graph_args)
        device = evaluate_device(
            fn, graph_args, args.backend, args.device, plugin
        )
        difference = max_abs_difference(reference, device)
        match = difference <= args.tolerance
        failures += 0 if match else 1
        print(f"{name}: match={match} max_abs_diff={difference:.6e}")

    print()
    if failures:
        print(
            f"verdict: FAIL ({failures}/{len(graphs)} graph(s) exceed "
            f"tolerance {args.tolerance:g})"
        )
        return 1
    print(
        f"verdict: PASS ({len(graphs)}/{len(graphs)} graphs match within "
        f"{args.tolerance:g})"
    )
    return 0


def _non_negative_float(text: str) -> float:
    """argparse type for a float >= 0 (a zero tolerance compares exactly)."""
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI."""
    parser = argparse.ArgumentParser(
        prog="xla_plugin_probe.py",
        description=(
            "Build and run two evoxc graphs -- an elementwise+reduce and a "
            "while_loop -- on backend='numpy' and on backend='xla' through a "
            "PJRT C API plugin, and compare them. Exit 0 if both match, 1 if "
            "either does not, 2 on a missing or unusable plugin."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"the plugin is also taken from ${PJRT_ENV_VAR}; {PJRT_FLAG} wins. "
            "The adapter's availability probe reads only the environment "
            "variable, so this tool exports it from the resolved path."
        ),
    )
    parser.add_argument(
        PJRT_FLAG,
        dest="pjrt_plugin",
        default=None,
        metavar="PATH",
        help=(
            f"PJRT C API plugin .so exporting GetPjRtApi "
            f"(falls back to ${PJRT_ENV_VAR})"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("rocm", "hip", "cuda"),
        default="rocm",
        help=(
            "accelerator device kind passed through as device=<kind> "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("xla",),
        default="xla",
        help=(
            "evoxc backend under test; numpy is always the reference leg "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=_non_negative_float,
        default=1e-4,
        metavar="ABS",
        help=(
            "maximum absolute per-element difference for a match "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Resolve the plugin, probe it, and return the exit code."""
    args = parse_args(argv)
    try:
        plugin = resolve_plugin(args.pjrt_plugin)
    except PluginError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return run_probe(args, plugin)
    except evoxc.core.BackendError as exc:
        # The adapter refused the plugin (missing at load time, ABI gate,
        # compile refusal): a named plugin/backend error, never a bare traceback.
        print(
            f"error: the xla adapter could not use {plugin!r}: {exc}",
            file=sys.stderr,
        )
        return 2
    except Exception:  # noqa: BLE001 - loud, with the full traceback, exit 1
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
