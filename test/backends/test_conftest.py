"""The request guard's selector logic: which ``-m`` expressions *imply* a leg's marker.

``conftest.py`` stops the session when the command line explicitly asks for a
compiled-backend leg and ``iree`` is absent, so the predicate behind that decision
(``implies_marker``, with ``implies_requires_iree`` as its llvm-cpu wrapper) is the
one piece of the guard that must be exactly right: a false positive turns an
ordinary default run red in an environment without the optional extra, and a false
negative turns a requested-but-unavailable leg into a green "1 skipped".

The predicate is read here directly, so this module needs no optional dependency
and runs in a bare ``uv sync`` (it is not marked ``requires_iree`` or
``requires_rocm``).
The expected column is hand-written from the rule -- the expression selects a test
that carries only that marker, and would not select the same test without it --
never produced by calling the helper under test again.
"""

from __future__ import annotations

import pytest

# ``test/conftest.py`` puts ``test/`` on ``sys.path`` and ``test/backends`` is a
# package (``__init__.py``), so the directory's conftest is importable under its
# package name without importing ``iree``.
from backends.conftest import (
    INSTALL_COMMAND,
    MARKER,
    ROCM_MARKER,
    implies_marker,
    implies_requires_iree,
)

#: ``(marker, expression, implies)`` -- the whole truth table of the predicate, for
#: both compiled-backend markers. The third column is the expectation read off the
#: rule, not the helper's answer.
CASES = [
    # -- the llvm-cpu leg: the original table, unchanged ----------------------
    # A bare marker is the capability request itself.
    (MARKER, "requires_iree", True),
    # A disjunction keeps the marker sufficient.
    (MARKER, "requires_iree or parity", True),
    (MARKER, "parity or requires_iree", True),
    # Negating the marker cannot request the leg.
    (MARKER, "not requires_iree", False),
    # The conjunction cannot be satisfied by the marker alone.
    (MARKER, "requires_iree and slow", False),
    # Marker-free selectors include the bare-marker test only because it lacks
    # the excluded markers -- they do not request the compiled backend.
    (MARKER, "not parity and not slow", False),
    (MARKER, "parity", False),
    (MARKER, "not slow", False),
    # pytest reports malformed expressions; this predicate stays out of the way.
    (MARKER, "(", False),
    # -- the ROCm leg: the same rule, the second marker -----------------------
    (ROCM_MARKER, "requires_rocm", True),
    (ROCM_MARKER, "requires_rocm or parity", True),
    (ROCM_MARKER, "parity or requires_rocm", True),
    (ROCM_MARKER, "not requires_rocm", False),
    (ROCM_MARKER, "requires_rocm and slow", False),
    (ROCM_MARKER, "not parity and not slow", False),
    (ROCM_MARKER, "parity", False),
    (ROCM_MARKER, "not slow", False),
    (ROCM_MARKER, "(", False),
    # -- the two markers are independent requests -----------------------------
    # Asking for one leg is not asking for the other.
    (MARKER, "requires_rocm", False),
    (ROCM_MARKER, "requires_iree", False),
    # A conjunction of the two is satisfiable only by a test carrying both, which
    # the bare-marker synthetic test does not -- so neither marker alone implies it.
    (MARKER, "requires_iree and requires_rocm", False),
    (ROCM_MARKER, "requires_iree and requires_rocm", False),
    # A disjunction of the two is implied by either one.
    (MARKER, "requires_iree or requires_rocm", True),
    (ROCM_MARKER, "requires_iree or requires_rocm", True),
    # Neither marker is named, so no leg is requested.
    (MARKER, "not requires_iree and not requires_rocm", False),
    (ROCM_MARKER, "not requires_iree and not requires_rocm", False),
]


def test_marker_names_are_the_registered_ones():
    """The guard's marker constants must be the names ``../conftest.py`` registers."""
    assert MARKER == "requires_iree"
    assert ROCM_MARKER == "requires_rocm"


@pytest.mark.parametrize(("marker", "expression", "implies"), CASES)
def test_implies_marker(marker, expression, implies):
    """Only a selector that the marker alone *changes* requests that leg."""
    assert implies_marker(expression, marker) is implies, (
        f"-m {expression!r} must {'imply' if implies else 'not imply'} {marker}"
    )


#: The llvm-cpu half of the table, for the marker-defaulting wrapper.
IREE_ONLY_CASES = [
    (expression, implies) for marker, expression, implies in CASES if marker == MARKER
]


@pytest.mark.parametrize(("expression", "implies"), IREE_ONLY_CASES)
def test_implies_requires_iree_defaults_to_the_iree_marker(expression, implies):
    """``implies_requires_iree`` stays the ``MARKER``-defaulting wrapper it was."""
    assert implies_requires_iree(expression) is implies, (
        f"-m {expression!r} must {'imply' if implies else 'not imply'} requires_iree"
    )
    assert implies_requires_iree(expression) is implies_marker(expression, MARKER)


def test_install_command_names_the_iree_313_wheels():
    """The announced install step must be the one ``pyproject.toml`` actually resolves.

    The ``iree`` extra names the IREE 3.13 compiler + runtime wheels directly; the
    older spellings (the ``evoxc[iree]`` extra, whose pin is 3.11, and the
    git-pinned ``evoxc`` tag) no longer describe what this suite needs, so offering
    either would send a user to a wheel set that cannot run this leg.
    """
    assert "uv sync --extra iree" in INSTALL_COMMAND
    assert "iree-base-compiler>=3.13,<3.14" in INSTALL_COMMAND
    assert "iree-base-runtime>=3.13,<3.14" in INSTALL_COMMAND
    assert "git+https://github.com/EMI-Group/evoxc@v0.1.1" not in INSTALL_COMMAND
    assert "evoxc[iree]" not in INSTALL_COMMAND
