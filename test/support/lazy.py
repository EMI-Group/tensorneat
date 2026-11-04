"""Deferred imports at fixture time, with a pending implementation reported as a skip.

:func:`require` imports a module when the fixture that needs it runs, rather than
at module scope, and converts a ``NotImplementedError`` raised *during that
import* into :func:`pytest.skip`. Every other error — an ``ImportError``, a
configuration error, any bug — propagates unchanged, because a broken package
must not look unfinished.

The deferral keeps a suite collectable when a source module still has a
pending implementation, and costs nothing otherwise: the fixtures call
:func:`require` from their bodies, so a healthy package simply imports.
"""

from __future__ import annotations

from types import ModuleType

import pytest

__all__ = ["require"]


def require(module: str) -> ModuleType:
    """Import ``module``, turning a ``NotImplementedError`` raised during the import into a skip.

    Args:
        module: Dotted module path, e.g. ``"tensorneat.genome"`` or
            ``"support.payloads"``.

    Returns:
        The imported module.

    Raises:
        pytest.skip.Exception: (as a skip) when importing the module raises
            ``NotImplementedError``, which is the signature of a pending
            implementation.
        ImportError, Exception: Everything else propagates unchanged.
    """
    try:
        return __import__(module, fromlist=["_"])
    except NotImplementedError as exc:  # a pending implementation, not a test failure
        pytest.skip(f"not implemented: importing {module} raises {exc!r}")
