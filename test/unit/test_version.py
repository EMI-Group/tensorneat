"""Unit tests for the package's version-metadata surface.

What this module proves
-----------------------
``tensorneat.__version__`` is the installed distribution's version string, and an
import from a source checkout that has no installed distribution degrades to
``"0.0.0"`` instead of raising at import time.

Oracles used here (never the implementation)
--------------------------------------------
The distribution metadata itself — :func:`importlib.metadata.version` — is the
oracle for the happy path; the fallback is proved by making that same lookup
report a missing distribution, reloading the module, and restoring both the
lookup and the module afterwards.

Gotchas this module encodes
---------------------------
``tensorneat/__init__.py`` binds ``version`` at import time, so the
``PackageNotFoundError`` fallback is observable only by *reloading* the module
while the metadata function is patched; the real module is reloaded once more in
the ``finally`` block so no later test sees ``"0.0.0"``.
No ``jax``/``torch`` is imported here.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata

import tensorneat


def test_version_matches_the_installed_distribution_metadata():
    """Invariant: ``__version__`` is a non-empty string equal to the distribution version.

    Oracle: ``importlib.metadata.version("tensorneat")``, the metadata entry the
    module reads at import time.
    """
    installed = metadata.version("tensorneat")

    assert isinstance(tensorneat.__version__, str)
    assert tensorneat.__version__
    assert tensorneat.__version__ == installed


def test_missing_distribution_metadata_falls_back_to_zero(monkeypatch):
    """Invariant: ``PackageNotFoundError`` during the lookup yields ``"0.0.0"``.

    Oracle: :func:`importlib.metadata.version` patched to raise, followed by
    ``importlib.reload`` so the module re-executes its ``from importlib.metadata
    import version`` binding and calls the patched function. The real lookup and
    the real module are restored in the ``finally`` block.
    """
    installed = metadata.version("tensorneat")

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    try:
        reloaded = importlib.reload(tensorneat)
        assert reloaded.__version__ == "0.0.0"
    finally:
        monkeypatch.undo()

    restored = importlib.reload(tensorneat)
    assert restored.__version__ == installed
