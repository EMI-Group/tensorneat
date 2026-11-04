"""Unit-suite guard: the numpy-backend tests must not pull in an optional framework.

``../release_gate_no_jax.py`` checks the *package* in a clean subprocess; this
conftest checks the *tests*, inside the process that runs them, so a leak is
attributed to the module or test that caused it.  Two cooperating mechanisms,
because neither one is sufficient here.

**A session baseline.**  The forbidden modules present in ``sys.modules`` are
snapshotted once, as early as this file can observe them: at
:func:`pytest_sessionstart` when the conftest belongs to pytest's initial
conftest set, and in the session-scoped :func:`_guard_baseline` fixture
otherwise (a conftest that is only loaded when collection reaches this
directory).  The snapshot is taken exactly once — the two entry points are
idempotent — so a module in ``sys.modules`` now but not in it arrived during
this session, which is what lets the guard see a leak at *collection* time and
not only one inside a test body.

**An import recorder.**  ``builtins.__import__`` is wrapped once for the session
and records every import *statement* whose root module is forbidden, together
with the file that executed it.  This is what gives the guard teeth: it
attributes a forbidden module to the file that imported it, so the diff can tell
a test-side import from an import performed by ``src/`` or another suite.  The
recorded file also supplies the two allowances the guard needs, i.e. it answers
"whose leak is it?": an import executed from ``src/``, from ``pytest``'s own
helpers (``pytest.importorskip``) or from another suite (parity imports ``jax``
while it is being collected, so a mixed ``pytest`` run over parity and unit
together must not fail here) is not this suite's leak; an import executed from a
file below this directory is.

What the guard does not cover: a *dynamic* ``importlib.import_module("jax")`` of
a module that is already loaded touches neither ``builtins.__import__`` nor
``sys.modules`` and is therefore invisible, and anything a subprocess does is
``../release_gate_no_jax.py``'s business.  It is a per-test check on this
process, not a proof that the package is framework-free.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import FrameType

import pytest

#: Top-level modules whose presence means the suite silently acquired an optional
#: dependency (the same list ``../release_gate_no_jax.py`` enforces).
FORBIDDEN = ("jax", "jaxlib", "torch", "brax", "gymnax", "mujoco")

#: This directory: an import executed from a file below it is this suite's leak.
UNIT_DIR = Path(__file__).resolve().parent

#: Directories whose frames only relay someone else's import statement.
_MACHINERY_DIRS = (Path(pytest.__file__).resolve().parent,)

#: Forbidden modules already present when the guard started; frozen once.
_baseline: frozenset[str] | None = None

#: Forbidden imports seen since then, as ``(module name, file that executed it)``.
_pending: list[tuple[str, str | None]] = []


def _present() -> set[str]:
    """The forbidden top-level modules currently in ``sys.modules``."""
    return {name for name in FORBIDDEN if name in sys.modules}


def _freeze_baseline() -> frozenset[str]:
    """Snapshot the forbidden modules present now, the first time this is called."""
    global _baseline
    if _baseline is None:
        _baseline = frozenset(_present())
    return _baseline


def _inside_unit(filename: str) -> bool:
    """True when ``filename`` is a file of this directory or of a directory below it."""
    try:
        return Path(filename).resolve().is_relative_to(UNIT_DIR)
    except OSError:  # pragma: no cover - an unresolvable path is not a unit file
        return False


def _is_machinery(filename: str) -> bool:
    """True for pytest's own files, whose frames merely relay the caller's import."""
    try:
        path = Path(filename).resolve()
    except OSError:  # pragma: no cover - an unresolvable path is not a pytest file
        return False
    return any(path.is_relative_to(directory) for directory in _MACHINERY_DIRS)


def _executing_file(frame: FrameType | None) -> str | None:
    """The first file up the stack that is user code (not pytest, not a pseudo-file)."""
    while frame is not None:
        filename = frame.f_globals.get("__file__")
        if filename and not filename.startswith("<") and not _is_machinery(filename):
            return filename
        frame = frame.f_back
    return None


#: The ``__import__`` the wrapper delegates to (the real one, or an older wrapper).
_import = builtins.__import__


def _recording_import(
    name: str,
    globals: dict | None = None,
    locals: dict | None = None,
    fromlist: tuple = (),
    level: int = 0,
):
    """``builtins.__import__`` that notes forbidden imports before delegating."""
    if name.partition(".")[0] in FORBIDDEN:
        _pending.append((name, _executing_file(sys._getframe(1))))
    return _import(name, globals, locals, fromlist, level)


if not getattr(builtins.__import__, "__unit_guard__", False):
    _recording_import.__unit_guard__ = True  # type: ignore[attr-defined]
    builtins.__import__ = _recording_import


def _leaks(entry: set[str], baseline: frozenset[str]) -> list[str]:
    """Describe, and forget, every leak this suite is responsible for.

    Args:
        entry: the forbidden modules present when the current test started, i.e.
            everything that arrived before its body ran (collected modules, a
            conftest, another suite's collection).
        baseline: the session baseline from :func:`_freeze_baseline`.

    Returns:
        Human-readable ``"<module> (<why>)"`` strings, empty when nothing leaked.
        Recorded imports are consumed, so one leak is reported once instead of
        by every later test as well.

    Notes:
        An arrival counted by the ``sys.modules`` diff is only reported when it
        happened *inside* the test; one that happened earlier is reported by the
        recorder when a file of this suite caused it (module scope of a test
        module or conftest) and excused otherwise (``src/``, another suite).
    """
    reasons: dict[str, list[str]] = {}
    while _pending:
        module, filename = _pending.pop()
        if filename is not None and _inside_unit(filename):
            root = module.partition(".")[0]
            reasons.setdefault(root, []).append(f"imported by {filename}")
    for name in sorted(_present() - baseline - entry):
        reasons.setdefault(name, []).append("entered sys.modules while this test ran")
    return [f"{name} ({'; '.join(why)})" for name, why in sorted(reasons.items())]


def pytest_sessionstart(session: pytest.Session) -> None:
    """Freeze the baseline before any test module is imported.

    Only called when this file is part of pytest's initial conftest set (the
    common case for ``pytest test/unit``); a conftest that pytest loads when
    collection reaches this directory freezes it in :func:`_guard_baseline`.
    """
    _freeze_baseline()


@pytest.fixture(scope="session", autouse=True)
def _guard_baseline() -> frozenset[str]:
    """Freeze the session baseline, for the runs where no earlier hook could."""
    return _freeze_baseline()


@pytest.fixture(autouse=True)
def _stays_free_of_optional_frameworks(
    request: pytest.FixtureRequest, _guard_baseline: frozenset[str]
):
    """Fail the test that imports an optional framework at run time or at module scope."""
    entry = _present()
    yield
    leaked = _leaks(entry, _guard_baseline)
    assert not leaked, (
        f"{request.node.nodeid} imported {leaked}; the unit suite runs on the numpy "
        "backend only (parity tests are the only place allowed to import jax) — no "
        "test body, test module or conftest under ./unit may import an optional "
        "framework"
    )
