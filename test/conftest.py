"""Shared pytest configuration for the TensorNEAT (evoxc) test suite.

Two jobs, both host-side and dependency-free:

1. **Import guards.** ``tensorneat`` is found under ``src/`` and
   ``tensorneat_legacy`` (the frozen parity oracle) under
   ``benchmarks/legacy_jax/``, so the suite runs from a plain checkout without
   installing the package. Both directories are prepended to ``sys.path``
   only when they exist and only when they are not already present, which
   keeps the setup idempotent under ``pytest-xdist``/nested runs. The checkout
   root is appended (never prepended) so ``test/unit/benchmarks`` can import
   the harness under test as ``benchmarks.harness.protocol``.

2. **Configuration.** Markers that ``pyproject.toml`` does not declare
   (``requires_iree``) are registered here, and
   ``release_gate_no_jax.py`` — deliberately named without a ``test_`` prefix
   because it is a release gate, not a unit test — is collected explicitly.

Nothing here imports ``tensorneat``, ``evoxc`` or ``jax`` at module scope:
collection must stay cheap and must never pull JAX into a numpy-only run.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import pytest

#: Repository root, derived from this file (``<repo>/test/conftest.py``).
REPO_ROOT = Path(__file__).resolve().parents[1]
#: The installable package (src-layout).
SRC_DIR = REPO_ROOT / "src"
#: Frozen pre-rewrite JAX implementation, imported as ``tensorneat_legacy``.
LEGACY_DIR = REPO_ROOT / "benchmarks" / "legacy_jax"
#: This directory, which must be importable so suites can ``import support``.
TEST_DIR = Path(__file__).resolve().parent

#: The filename of the release gate; it is collected by :func:`pytest_collect_file`.
RELEASE_GATE_FILENAME = "release_gate_no_jax.py"


def _prepend(path: Path) -> None:
    """Put ``path`` at the front of ``sys.path`` when it is a real, absent directory."""
    if not path.is_dir():
        return
    entry = str(path)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)


for _path in (SRC_DIR, LEGACY_DIR, TEST_DIR):
    _prepend(_path)

#: The checkout root is *appended*, not prepended: ``test/unit/benchmarks`` imports the
#: harness under test as ``benchmarks.harness.protocol``, and appending keeps the source and
#: test trees above it on the path.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers this suite uses that ``pyproject.toml`` does not."""
    config.addinivalue_line(
        "markers",
        "requires_iree: needs the optional ``evoxc[iree]`` backend; in no gate",
    )
    config.addinivalue_line(
        "markers",
        "unit: numpy-backend kernel and invariant test (default gate)",
    )


def _named_on_command_line(config: pytest.Config, path: Path) -> bool:
    """True when ``path`` was passed to pytest explicitly (``pytest test/<file>.py``).

    pytest already collects an explicitly named file whatever its name is, so
    :func:`pytest_collect_file` must decline for it — otherwise the file is
    collected twice and every subprocess in it runs twice.
    """
    for argument in config.invocation_params.args:
        candidate = str(argument).split("::", 1)[0]
        if not candidate or candidate.startswith("-"):
            continue
        try:
            if Path(candidate).resolve() == path:
                return True
        except OSError:  # pragma: no cover - an unparsable path is not our argument
            continue
    return False


def pytest_collect_file(file_path: Path, parent: pytest.Collector):
    """Collect ``release_gate_no_jax.py`` even though it lacks a ``test_`` prefix.

    It is named after the gate it implements rather than after pytest's default
    ``python_files`` patterns; rather than rename it (and lose the readable
    gate name) the collection rule is spelled out here.
    """
    if file_path.name != RELEASE_GATE_FILENAME or file_path.parent != TEST_DIR:
        return None
    if _named_on_command_line(parent.config, Path(file_path)):
        return None
    return pytest.Module.from_parent(parent, path=file_path)


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    """A deterministic per-test seed derived from the test id.

    Using the node id rather than a global counter means a test keeps its seed
    when other tests are added, removed or reordered — a failure stays
    reproducible under ``pytest -k``.
    """
    return zlib.crc32(request.node.nodeid.encode("utf-8")) & 0xFFFFFF


@pytest.fixture
def rng(seed: int):
    """A ``numpy.random.Generator`` seeded by :func:`seed` (never the global state)."""
    import numpy as np

    return np.random.default_rng(seed)


@pytest.fixture
def evoxc_key(seed: int):
    """A rank-0 ``evoxc.random`` key seeded by :func:`seed`."""
    import evoxc

    return evoxc.random.key(seed)
