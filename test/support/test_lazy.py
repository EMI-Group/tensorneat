"""Tests for :mod:`support.lazy` — the skip channel must not be able to hide a real defect.

The contract under test is that a pending implementation is a *defect* unless the
caller explicitly opts in, and that even then only a `NotImplementedError` raised
by the library under test may become a skip.
Every test builds the module it imports on disk under ``tmp_path`` (added to
``sys.path`` for the duration of the test and purged from ``sys.modules`` after),
so the raising frame is under the test's control and no test depends on any
current (or future) state of ``src/tensorneat``.
The suite's own modules are never the subject: a `NotImplementedError` raised by
test-support code is exactly the case that must stay a failure.
A test that expects a raise or a clean import wraps its body in `_forbids_skip`,
because an escaped `pytest.skip` would otherwise be *reported* as a skip instead
of as the failure it is.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from support import lazy
from support.lazy import require

#: ``<repo>``, derived independently of the constant the module under test derives.
REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _forbids_skip() -> Iterator[None]:
    """Turn a ``pytest.skip`` raised by the code under test into a failure.

    ``pytest.skip.Exception`` derives from ``BaseException``, so it slips past
    ``pytest.raises(NotImplementedError)`` and would report the test as *skipped*
    — the very outcome these tests exist to forbid.
    Every test that expects a raise or a successful import wraps its body in this
    guard, so a wrongly converted skip is a red test, not a grey one.
    """
    try:
        yield
    except pytest.skip.Exception as exc:
        pytest.fail(f"a skip escaped where a raise was required: {exc}")


def _make_package(root: Path, package: str, body: str) -> Path:
    """Write ``root/<package>/__init__.py`` plus a ``mod.py`` whose module body is ``body``.

    Args:
        root: The directory to put on ``sys.path``.
        package: A unique package name, so two tests never share ``sys.modules``
            entries.
        body: Python source for ``mod.py``.

    Returns:
        The directory holding the package, i.e. what ``LIBRARY_DIR`` is patched to
        when the module is meant to look like library code.
    """
    directory = root / package
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    (directory / "mod.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return directory


@pytest.fixture
def tmp_sys_path(tmp_path: Path) -> Iterator[Path]:
    """Put ``tmp_path`` on ``sys.path`` first, and undo the import afterwards.

    Yields:
        ``tmp_path``, the directory the temporary packages are written into.

    Notes:
        Cleanup deletes every ``sys.modules`` entry whose file lives under
        ``tmp_path`` and removes the path entry, so a later test re-imports the
        same dotted name from its own package instead of reusing this one.
    """
    entry = str(tmp_path)
    sys.path.insert(0, entry)
    try:
        yield tmp_path
    finally:
        if entry in sys.path:
            sys.path.remove(entry)
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if filename is None:
                continue
            try:
                is_temporary = Path(filename).resolve().is_relative_to(tmp_path.resolve())
            except OSError:  # pragma: no cover - an unparsable synthetic path is not ours
                is_temporary = False
            if is_temporary:
                del sys.modules[name]


def test_library_dir_is_derived_from_this_file() -> None:
    """Invariant: ``LIBRARY_DIR`` is derived from this file's location, not a fixed absolute path.

    Oracle: ``Path(__file__).resolve().parents[2] / "src" / "tensorneat"``, built
    here without calling the module.

    Assertion: equality with that path, plus the directory and its ``__init__.py``
    really exist, so the containment test in :func:`require` has a real root.
    """
    expected = REPO_ROOT / "src" / "tensorneat"
    assert lazy.LIBRARY_DIR == expected
    assert lazy.LIBRARY_DIR.is_dir()
    assert (lazy.LIBRARY_DIR / "__init__.py").is_file()


def test_public_surface_is_only_require() -> None:
    """Invariant: the module exports exactly one callable, so no caller grows a second skip path.

    Oracle: the fixed list ``["require"]``.

    Assertion: ``__all__`` equals that list and every name in it is defined.
    """
    assert lazy.__all__ == ["require"]
    assert callable(require)


def test_default_mode_propagates_not_implemented_error(tmp_sys_path: Path) -> None:
    """Invariant: without ``allow_pending`` a pending implementation is a defect, not a skip.

    Oracle: a temporary module whose body raises ``NotImplementedError`` with a
    recognisable message, put on ``sys.path``.

    Assertion: :func:`require` raises that ``NotImplementedError`` itself (no
    ``Skipped``), and the message is the one the module raised.
    """
    _make_package(tmp_sys_path, "pending_default", 'raise NotImplementedError("still pending")\n')
    with _forbids_skip():
        with pytest.raises(NotImplementedError, match="still pending"):
            require("pending_default.mod")


def test_allow_pending_skips_and_names_module_file_line_and_function(
    tmp_sys_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: an opted-in pending implementation inside the library is a diagnosable skip.

    Oracle: a temporary package patched in as ``LIBRARY_DIR`` whose ``mod.py``
    raises ``NotImplementedError`` on its first line, in the module body
    (function name ``<module>``).

    Assertion: the call raises ``pytest.skip.Exception`` and its message names the
    dotted module, the absolute source file, that file's line 1 and the raising
    function — everything needed to find the pending implementation.
    """
    directory = _make_package(
        tmp_sys_path, "pending_in_library", 'raise NotImplementedError("not written yet")\n'
    )
    monkeypatch.setattr(lazy, "LIBRARY_DIR", directory)

    with pytest.raises(pytest.skip.Exception) as info:
        require("pending_in_library.mod", allow_pending=True)

    message = str(info.value)
    assert "pending_in_library.mod" in message
    assert str((directory / "mod.py").resolve()) in message
    assert f"{directory / 'mod.py'}:1" in message
    assert "in <module>()" in message
    assert "not written yet" in message


def test_allow_pending_still_propagates_outside_the_library(tmp_sys_path: Path) -> None:
    """Invariant: the opt-in can never hide a ``NotImplementedError`` raised outside the library.

    Oracle: a temporary module that is *not* under ``LIBRARY_DIR`` (the constant is
    left at its real value and the module's path is asserted not to be inside it).

    Assertion: with ``allow_pending=True`` the ``NotImplementedError`` still
    propagates, so a pending feature in test-support code, ``evoxc`` or
    site-packages fails the run instead of greening a skip.
    """
    directory = _make_package(
        tmp_sys_path, "pending_outside", 'raise NotImplementedError("support-side pending")\n'
    )
    assert not (directory / "mod.py").resolve().is_relative_to(lazy.LIBRARY_DIR)

    with _forbids_skip():
        with pytest.raises(NotImplementedError, match="support-side pending"):
            require("pending_outside.mod", allow_pending=True)


@pytest.mark.parametrize("allow_pending", [False, True])
def test_missing_module_raises_import_error(allow_pending: bool) -> None:
    """Invariant: ``ImportError`` is never converted, in either mode.

    Oracle: a dotted name that no package provides.

    Assertion: :func:`require` raises ``ModuleNotFoundError`` (an ``ImportError``)
    rather than skipping — a missing module is a broken package, not an unfinished
    one.
    """
    with _forbids_skip():
        with pytest.raises(ImportError):
            require("tensorneat_support_absent_module_xyz", allow_pending=allow_pending)


def test_allow_pending_propagates_import_error_from_the_library(
    tmp_sys_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: only ``NotImplementedError`` converts — an in-library ``ImportError`` is a bug.

    Oracle: a temporary package patched in as ``LIBRARY_DIR`` whose module body
    raises ``ImportError``.

    Assertion: with ``allow_pending=True`` the ``ImportError`` propagates with its
    own message instead of becoming a skip.
    """
    directory = _make_package(
        tmp_sys_path, "broken_in_library", 'raise ImportError("broken dependency")\n'
    )
    monkeypatch.setattr(lazy, "LIBRARY_DIR", directory)

    with _forbids_skip():
        with pytest.raises(ImportError, match="broken dependency"):
            require("broken_in_library.mod", allow_pending=True)


def test_healthy_module_imports_and_is_the_cached_module() -> None:
    """Invariant: a healthy import is a plain import — same object, no wrapping.

    Oracle: ``importlib.import_module``, the library function the module's
    docstring names as its job.

    Assertion: :func:`require` returns a ``ModuleType`` identical (``is``, not
    equal) to what ``importlib.import_module`` returns, and a second call returns
    that same object, so callers can rely on module identity and module state.
    """
    with _forbids_skip():
        module = require("support.payloads", allow_pending=True)
        assert isinstance(module, ModuleType)
        assert module is importlib.import_module("support.payloads")
        assert require("support.payloads") is module
