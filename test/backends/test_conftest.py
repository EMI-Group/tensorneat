"""The request guard's selector logic: which ``-m`` expressions *imply* ``requires_iree``.

``conftest.py`` stops the session when the command line explicitly asks for the
compiled-backend leg and ``iree`` is absent, so the predicate behind that decision
(``implies_requires_iree``) is the one piece of the guard that must be exactly
right: a false positive turns an ordinary default run red in an environment
without the optional extra, and a false negative turns a requested-but-unavailable
leg into a green "1 skipped".

The predicate is read here directly, so this module needs no optional dependency
and runs in a bare ``uv sync`` (it is not marked ``requires_iree``).
The expected column is hand-written from the rule -- the expression selects a
test that carries only ``requires_iree``, and would not select the same test
without that marker -- never produced by calling the helper under test again.
"""

from __future__ import annotations

import pytest

# ``test/conftest.py`` puts ``test/`` on ``sys.path`` and ``test/backends`` is a
# package (``__init__.py``), so the directory's conftest is importable under its
# package name without importing ``iree``.
from backends.conftest import INSTALL_COMMAND, implies_requires_iree

#: ``(expression, implies)`` -- the whole truth table of the predicate.
CASES = [
    # A bare marker is the capability request itself.
    ("requires_iree", True),
    # A disjunction keeps the marker sufficient.
    ("requires_iree or parity", True),
    ("parity or requires_iree", True),
    # Negating the marker cannot request the leg.
    ("not requires_iree", False),
    # The conjunction cannot be satisfied by the marker alone.
    ("requires_iree and slow", False),
    # Marker-free selectors include the bare-marker test only because it lacks
    # the excluded markers -- they do not request the compiled backend.
    ("not parity and not slow", False),
    ("parity", False),
    ("not slow", False),
    # pytest reports malformed expressions; this predicate stays out of the way.
    ("(", False),
]


@pytest.mark.parametrize(("expression", "implies"), CASES)
def test_implies_requires_iree(expression, implies):
    """Only a selector that the marker alone *changes* implies the compiled backend."""
    assert implies_requires_iree(expression) is implies, (
        f"-m {expression!r} must {'imply' if implies else 'not imply'} requires_iree"
    )


def test_install_command_names_the_git_pinned_extra():
    """The announced install step must resolve the ``evoxc`` version the project pins.

    ``uv pip install "evoxc[iree]"`` resolves the PyPI release 0.1.0, which
    ``pyproject.toml`` forbids (``>= 0.1.1``), so only the project extra and the
    git-pinned pip form may be offered.
    """
    assert "uv sync --extra iree" in INSTALL_COMMAND
    assert "git+https://github.com/EMI-Group/evoxc@v0.1.1" in INSTALL_COMMAND
    assert 'uv pip install "evoxc[iree]"' not in INSTALL_COMMAND
