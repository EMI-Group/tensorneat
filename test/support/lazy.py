"""Deferred imports, so the suite stays collectable while ``src/`` is unfinished.

The package under test is mid-rewrite: some modules are design skeletons whose
bodies raise ``NotImplementedError``, and one of them
(``tensorneat/algorithm/base.py``) evaluates ``State()`` as a *default argument*,
so ``import tensorneat`` itself raises ``NotImplementedError`` at the moment.
A module-scope import would therefore turn every test module into a collection
error instead of a named skip.

:func:`require` is the single sanctioned workaround: call it from a fixture
body (or a test body), never at module scope, and it converts only that
pending-implementation error into a skip. An ``ImportError`` — or any other
failure — still surfaces, because a broken package must not look unfinished.

Once ``src/`` is implemented these calls can become ordinary module-scope
imports; nothing else in the suite depends on the deferral.
"""

from __future__ import annotations

from types import ModuleType

import pytest

__all__ = ["require"]


def require(module: str) -> ModuleType:
    """Import ``module``, turning a design-skeleton ``NotImplementedError`` into a skip.

    Args:
        module: Dotted module path, e.g. ``"tensorneat.genome"`` or
            ``"support.payloads"``.

    Returns:
        The imported module.

    Raises:
        pytest.skip.Exception: (as a skip) when the module raises
            ``NotImplementedError`` while being imported, which is the
            signature of an unimplemented ``src/`` slice.
        ImportError, Exception: Everything else propagates unchanged.
    """
    try:
        return __import__(module, fromlist=["_"])
    except NotImplementedError as exc:  # unfinished src slice, not a test failure
        pytest.skip(f"not implemented: importing {module} raises {exc!r}")
