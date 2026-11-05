"""Deferred imports at fixture time, with an opt-in, library-scoped skip for pending work.

:func:`require` imports a module when the fixture that needs it runs, rather than
at module scope, so a failing import is attributed to the test that needed it.
By default every error propagates — a ``NotImplementedError`` included, because a
pending implementation in the library under test is a defect, not a reason to skip.

``allow_pending=True`` opts into one narrow conversion: a ``NotImplementedError``
whose raising frame is a file inside :data:`LIBRARY_DIR` (``src/tensorneat/``)
becomes :func:`pytest.skip`, and the message names the dotted module, that file
and the line and function that raised it.
A ``NotImplementedError`` raised anywhere else — a ``support`` helper, ``evoxc``,
site-packages — propagates even then, because converting it would hide a real
failure instead of a pending feature.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from types import ModuleType

import pytest

__all__ = ["require"]

#: The library under test, derived from this file (``<repo>/test/support/lazy.py``).
LIBRARY_DIR = Path(__file__).resolve().parents[2] / "src" / "tensorneat"


def _raising_frame(exc: BaseException) -> traceback.FrameSummary | None:
    """The innermost frame of ``exc``'s traceback: the file, line and function that raised it."""
    frames = traceback.extract_tb(exc.__traceback__)
    return frames[-1] if frames else None


def _raised_inside_library(exc: BaseException) -> traceback.FrameSummary | None:
    """The raising frame of ``exc`` when it is a file of :data:`LIBRARY_DIR`, else ``None``."""
    frame = _raising_frame(exc)
    if frame is None:
        return None
    if not Path(frame.filename).resolve().is_relative_to(LIBRARY_DIR.resolve()):
        return None
    return frame


def require(module: str, *, allow_pending: bool = False) -> ModuleType:
    """Import ``module``, converting a pending implementation into a skip only on request.

    Args:
        module: Dotted module path, e.g. ``"tensorneat.genome"`` or
            ``"support.payloads"``.
        allow_pending: When true, a ``NotImplementedError`` raised by a file of
            :data:`LIBRARY_DIR` becomes :func:`pytest.skip`; the default lets it
            propagate, and an ``allow_pending`` ``NotImplementedError`` raised
            outside the library propagates too.

    Returns:
        The imported module.

    Raises:
        pytest.skip.Exception: (as a skip) only with ``allow_pending=True`` and a
            ``NotImplementedError`` raised inside :data:`LIBRARY_DIR`; the message
            names the module, the source file and the raising line and function.
        NotImplementedError: Without ``allow_pending``, or when the raising frame
            is not a file of the library under test (a test-support module,
            ``evoxc``, site-packages).
        ImportError, Exception: Everything else propagates unchanged.
    """
    try:
        return __import__(module, fromlist=["_"])
    except NotImplementedError as exc:
        if not allow_pending:
            raise
        frame = _raised_inside_library(exc)
        if frame is None:
            raise
        pytest.skip(
            f"pending implementation: importing {module} raised NotImplementedError "
            f"at {frame.filename}:{frame.lineno} in {frame.name}(): {exc}"
        )
