"""The named accelerator targets, the PJRT plugin resolution and its fingerprint.

The data half of the accelerator protocol (``benchmarks/gpu_measure.py``): a
*target* is a named way to build the ``Pipeline`` and lives in :data:`TARGETS` as
plain data, never as a code path --

``numpy``
    the reference interpreter: backend numpy, no device, no options.
``iree-llvm-cpu``
    iree with ``target_backends=["llvm-cpu"]``.
``iree-rocm``
    iree with ``device="rocm"`` and ``target_backends=["rocm"]``.
``xla-rocm``
    evoxc's ``xla`` adapter, ``device="rocm"``, driving a PJRT C API plugin
    (``.so`` exporting ``GetPjRtApi``) named by ``--pjrt-plugin PATH`` or the
    ``EVOXC_PJRT_PLUGIN`` environment variable; the ``xla-rocm`` extra of
    ``pyproject.toml`` installs the AMD one (``uv sync --extra xla-rocm``), since
    no CPU plugin is pip-installable. Nothing is guessed: an unresolved plugin
    raises :class:`PluginError` naming both sources, so an ``xla-rocm`` row is
    either a real measurement or a loud ``error`` row, never a fallback to
    another backend.

Why the environment variable matters
------------------------------------
``evoxc.backends.registry.get("xla")`` runs ``XlaBackend.check_available()``,
which calls ``_load_plugin()`` with **no options** and therefore reads only
``EVOXC_PJRT_PLUGIN`` (and the well-known paths). The ``plugin_path`` compile
option takes precedence in every later discovery call but cannot satisfy that
first probe, so the caller must wire **both**: ``compile_options["plugin_path"]``
for the build and ``EVOXC_PJRT_PLUGIN`` in the measuring process's environment.

The ROCm command-buffer workaround (``XLA_WORKAROUND_FLAGS``)
------------------------------------------------------------
On the MI300X, an ``xla`` generation graph that contains a long ``while`` (the RL
rollout) SIGSEGVs inside ``PJRT_LoadedExecutable_Execute``: XLA captures the graph
as a HIP graph / FUSION command buffer and replays it through ``hipGraphLaunch``,
which the ROCm plugin cannot do for that loop. Excluding ``WHILE`` from the
command buffer cures it (:data:`XLA_WORKAROUND_FLAGS`,
``--xla_gpu_enable_command_buffer=WHILE``; an empty value also works, while
``=FUSION`` still crashes), and a pure-forward graph with no ``while`` does not
crash at all. XLA reads ``XLA_FLAGS`` when the backend is initialised -- before the
plugin builds the executable -- so the flag is a **process environment** setting
and cannot be a ``compile_options`` entry. Both accelerator tools export it for
their ``xla`` targets (and only for those) and record the effective value, so a
number is never attributed to a flag it did not run with.

This module is stdlib-only at import time (``gpu_measure.py --list``/``--help``
must work on an interpreter without numpy), and it owns no rendering and no
measurement: the record schema and every rendered line stay in
``benchmarks/gpu_report.py``, the experiment in ``benchmarks/gpu_measure.py``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

#: Named target -> how the ``Pipeline`` is built. Data, never code paths.
TARGETS: dict[str, dict[str, Any]] = {
    "numpy": {"backend": "numpy", "device": None, "compile_options": None},
    "iree-llvm-cpu": {
        "backend": "iree",
        "device": None,
        "compile_options": {"target_backends": ["llvm-cpu"]},
    },
    "iree-rocm": {
        "backend": "iree",
        "device": "rocm",
        "compile_options": {"target_backends": ["rocm"]},
    },
    "xla-rocm": {
        "backend": "xla",
        "device": "rocm",
        "compile_options": {"plugin_path": None},
    },
}

#: Targets whose backend is evoxc's ``xla`` adapter and therefore needs a plugin.
PJRT_TARGETS: frozenset[str] = frozenset({"xla-rocm"})

#: The CLI flag naming the PJRT C API plugin (first in the resolution order).
PJRT_FLAG = "--pjrt-plugin"
#: The environment variable the evoxc ``xla`` adapter reads itself.
PJRT_ENV_VAR = "EVOXC_PJRT_PLUGIN"

#: The ``XLA_FLAGS`` value the ``xla`` targets are measured with (the ROCm workaround).
#:
#: On the MI300X an xla generation graph holding a long ``while`` (the RL rollout)
#: SIGSEGVs inside ``PJRT_LoadedExecutable_Execute``, because XLA captures it as a
#: HIP graph / FUSION command buffer and replays it via ``hipGraphLaunch``. Removing
#: ``WHILE`` from the command buffer cures it: an empty value also works, while
#: ``=FUSION`` still crashes, and a pure-forward graph (no ``while``) never crashes.
#: XLA reads ``XLA_FLAGS`` at backend init -- before the plugin builds the executable
#: -- so this is exported into the measuring process's environment (never into
#: ``compile_options``) and recorded with the measurement.
XLA_WORKAROUND_FLAGS = "--xla_gpu_enable_command_buffer=WHILE"

#: The build plan of the ``backend="numpy"`` reference leg (cross-check and trace).
NUMPY_PLAN: dict[str, Any] = {"name": "numpy", **TARGETS["numpy"], "device_index": None}

#: Bytes read per chunk while hashing the plugin (it is a ~500 MB shared object).
HASH_CHUNK_BYTES = 1 << 20

#: Resolved path -> fingerprint, so one process hashes the plugin at most once.
_FINGERPRINTS: dict[str, dict[str, Any]] = {}

__all__ = [
    "TARGETS",
    "PJRT_TARGETS",
    "PJRT_FLAG",
    "PJRT_ENV_VAR",
    "XLA_WORKAROUND_FLAGS",
    "NUMPY_PLAN",
    "PluginError",
    "resolve_plugin",
    "plugin_for",
    "require_plugin",
    "fingerprint_plugin",
    "short_sha",
]


class PluginError(RuntimeError):
    """The PJRT C API plugin a target needs could not be resolved (never a fallback)."""


def resolve_plugin(
    flag: str | None = None, env: Mapping[str, str] | None = None
) -> str | None:
    """Resolve the plugin path: ``--pjrt-plugin`` first, then ``EVOXC_PJRT_PLUGIN``.

    Args:
        flag: The ``--pjrt-plugin`` value, if the caller passed one.
        env: The environment to read (defaults to :data:`os.environ`).

    Returns:
        The absolute plugin path, or ``None`` when neither source named one (the
        caller decides whether that is an error).

    Raises:
        PluginError: When a source named a path that is not a readable file --
            naming which source, so a typo never surfaces as a ctypes failure
            inside the adapter.
    """
    environment = os.environ if env is None else env
    if flag:
        candidate, source = flag, PJRT_FLAG
    elif environment.get(PJRT_ENV_VAR):
        candidate, source = environment[PJRT_ENV_VAR], f"the {PJRT_ENV_VAR} env var"
    else:
        return None
    path = Path(candidate).expanduser()
    if not path.is_file():
        raise PluginError(
            f"{source} names {candidate!r}, which is not a file: the xla adapter "
            "needs a PJRT C API plugin library (.so exporting GetPjRtApi) there"
        )
    return str(path.resolve())


def plugin_for(
    target_name: str, flag: str | None = None, env: Mapping[str, str] | None = None
) -> str | None:
    """The plugin path a target needs; ``None`` for a target that needs none.

    A plugin flag is ignored for a target whose backend does not use the ``xla``
    adapter (the flags apply to every row of a matrix run, so a ``numpy`` row must
    not fail because of an ``xla``-only flag).

    Raises:
        PluginError: For a PJRT target with no usable plugin, naming both
            ``--pjrt-plugin`` and ``EVOXC_PJRT_PLUGIN`` so the fix is obvious.
    """
    if target_name not in PJRT_TARGETS:
        return None
    resolved = resolve_plugin(flag, env)
    if resolved is None:
        raise PluginError(
            f"target {target_name!r} drives the xla adapter and needs a PJRT C API "
            f"plugin library (.so exporting GetPjRtApi): pass {PJRT_FLAG} PATH or set "
            f"{PJRT_ENV_VAR}=PATH (neither named one) — `uv sync --extra xla-rocm` "
            "installs the AMD plugin, and there is no fallback to another backend"
        )
    return resolved


def require_plugin(
    targets: list[str], flag: str | None = None, env: Mapping[str, str] | None = None
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve the plugin once for a whole planned target list (``matrix`` preflight).

    Args:
        targets: The selected target labels.
        flag: The ``--pjrt-plugin`` value, if any.
        env: The environment to read (defaults to :data:`os.environ`).

    Returns:
        ``(path, fingerprint)``: the resolved path and its :func:`fingerprint_plugin`
        record, or ``(None, None)`` when no planned target needs a plugin. Resolving
        before the first row means a missing plugin costs no measurement at all.

    Raises:
        PluginError: When a planned target is a :data:`PJRT_TARGETS` member and no
            usable plugin was named (naming both sources), or when the plugin file
            cannot be fingerprinted.
    """
    needed = sorted(set(targets) & PJRT_TARGETS)
    if not needed:
        return None, None
    resolved = resolve_plugin(flag, env)
    if resolved is None:
        raise PluginError(
            f"the planned targets include {needed}, which drive the xla adapter and "
            f"need a PJRT C API plugin library (.so exporting GetPjRtApi): pass "
            f"{PJRT_FLAG} PATH or set {PJRT_ENV_VAR}=PATH (neither named one) — "
            "`uv sync --extra xla-rocm` installs the AMD plugin, and there is no "
            "fallback to another backend"
        )
    return resolved, fingerprint_plugin(resolved)


def fingerprint_plugin(path: str) -> dict[str, Any]:
    """The identity of one plugin file, for the record's provenance.

    Args:
        path: The resolved plugin path (:func:`resolve_plugin`).

    Returns:
        ``{"path", "basename", "size_bytes", "sha256"}``: the absolute path, the
        file name, its size in bytes, and the SHA-256 of its bytes (hashed in
        chunks; ~0.4 s for a 500 MB plugin, cached per path in this process).

    Raises:
        PluginError: When the file cannot be read or stat, so an unreadable
            plugin is loud instead of a silently incomplete fingerprint.
    """
    resolved = str(Path(path).resolve())
    cached = _FINGERPRINTS.get(resolved)
    if cached is not None:
        return dict(cached)
    digest = hashlib.sha256()
    try:
        size = Path(resolved).stat().st_size
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PluginError(f"cannot fingerprint the PJRT plugin {resolved!r}: {exc}") from exc
    fingerprint = {
        "path": resolved,
        "basename": Path(resolved).name,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }
    _FINGERPRINTS[resolved] = fingerprint
    return dict(fingerprint)


def short_sha(digest: str | None, length: int = 12) -> str:
    """The leading ``length`` hex characters of a digest (``n/a`` when absent)."""
    if not digest:
        return "n/a"
    return digest[:length]
