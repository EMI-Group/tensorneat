"""Loud-failure guard for the optional compiled-backend (``iree``) suites.

``test_iree_pipeline.py`` (llvm-cpu) and ``test_rocm_pipeline.py`` (AMD GPU) both
need the optional ``iree`` extra -- the IREE 3.13.0 compiler and runtime wheels
-- so each skips at import when ``iree`` cannot be found.  That structural skip is
the right shape for a whole-tree run: the extra is optional and the numpy suites
must stay green without it.

It is the wrong shape when a compiled-backend leg was *explicitly requested*.
``pytest -m requires_iree`` (or ``-m requires_rocm``) in an environment without
the extra used to exit 0 with "1 skipped", indistinguishable from a green run -- a
silent degradation of exactly the leg the user asked for.  This conftest closes
that hole: when the command line selects either marker and ``iree`` is genuinely
absent, the session stops with a non-zero exit code and the install command
instead of collecting a skip.

Two intents are distinguished deliberately:

* an explicit ``-m`` selector whose expression implies the requested marker (e.g.
  ``-m requires_iree``, ``-m requires_rocm``, ``-m "requires_rocm or parity"``) is
  a *capability* request -- the compiled backend must be there, so its absence is
  a hard error;
* merely naming ``test/backends`` on the command line (``pytest test/backends``)
  is a *scope* request, not a capability request -- the modules still skip, but
  the skip is announced together with the install command so it cannot be
  mistaken for a green compiled-backend run.

Nothing here imports ``iree`` -- nor probes ``/dev/kfd`` or the HIP driver, which
would need the extension -- so loading this file never pulls in an optional
dependency: availability is decided by :func:`importlib.util.find_spec`, and the
ROCm-specific preconditions live in ``test_rocm_pipeline.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ``_pytest.mark.expression`` is pytest's own ``-m`` parser.  It is imported so
# the guard evaluates a selector with exactly the grammar and precedence pytest
# uses, instead of a hand-rolled approximation that could disagree; the module is
# private but has been stable since pytest 7 and is the only way to reuse that
# parser (``pytest.mark.expression`` is a *marker*, not the parser).
from _pytest.mark.expression import Expression

#: This directory, resolved, so a relative command-line argument can be compared.
BACKENDS_DIR = Path(__file__).resolve().parent

#: The markers that select the compiled-backend legs (registered in ``../conftest.py``).
#: ``requires_iree`` is the llvm-cpu leg; ``requires_rocm`` the GPU leg, which needs
#: the same IREE wheels *plus* a HIP device.
MARKER = "requires_iree"
ROCM_MARKER = "requires_rocm"

#: The markers this guard treats as compiled-backend capability requests.
REQUESTED_MARKERS = (MARKER, ROCM_MARKER)

#: The optional top-level package the ``iree`` extra provides.
OPTIONAL_MODULE = "iree"

#: One install step, spelled the way ``pyproject.toml``/``uv.lock`` resolve it: the
#: ``iree`` extra names the IREE 3.13 wheels directly (``iree-base-compiler`` +
#: ``iree-base-runtime``) instead of ``evoxc[iree]``, whose pin is 3.11 and whose HIP
#: runtime aborts on loop-carried ``while`` graphs.  Compiler and runtime must be the
#: same release, hence the matching ``>=3.13,<3.14`` bounds on both.
INSTALL_COMMAND = (
    "uv sync --extra iree   (equivalently: pip install "
    '"iree-base-compiler>=3.13,<3.14" "iree-base-runtime>=3.13,<3.14")'
)


def iree_available() -> bool:
    """Whether ``import iree`` would find the optional compiled backend.

    The probe uses :func:`importlib.util.find_spec` and therefore never imports
    the iree extension (which on IREE 3.13 costs ~0.14 s and a few MiB of RSS,
    but would still make every numpy-only run pay for it).  ``find_spec`` raises
    ``ValueError`` when the module is already in :data:`sys.modules` without a
    ``__spec__``; that state still means it is importable, so it counts as
    available.
    """
    try:
        return importlib.util.find_spec(OPTIONAL_MODULE) is not None
    except ValueError:  # already imported, but its __spec__ was dropped
        return OPTIONAL_MODULE in sys.modules


def explicit_marker_expressions(config: pytest.Config) -> list[str]:
    """The ``-m`` expressions that came from the command line, in order.

    Only ``config.invocation_params.args`` is read: a selector that arrives
    through ``addopts`` in ``pyproject.toml`` is configuration, not a per-run
    request, and must not turn an ordinary default run red.
    """
    args = [str(argument) for argument in config.invocation_params.args]
    expressions: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "-m":
            index += 1
            if index < len(args):
                expressions.append(args[index])
        elif argument.startswith("-m="):
            expressions.append(argument[3:])
        elif argument.startswith("-m") and not argument.startswith("--"):
            # A short option's value may be attached: ``-mrequires_iree``.
            expressions.append(argument[2:])
        index += 1
    return expressions


def implies_marker(expression: str, marker: str = MARKER) -> bool:
    """Whether ``expression`` selects a test *because* it carries ``marker``.

    The compiled ``-m`` expression is evaluated twice against the same synthetic
    test, one that carries ``marker`` and nothing else:

    * with a matcher that accepts that marker, i.e. the real selector;
    * with a matcher that accepts nothing, i.e. the same test as if it lacked
      the marker.

    The expression implies the marker only when the first selects the test and
    the second does not, so a selector that happens to include every marker-free
    test (``not parity and not slow``) is *not* misread as a capability request:
    it selects the bare-marker test twice, once with and once without the marker.

    Consequences, for either of the compiled-backend markers: ``requires_iree``,
    ``requires_rocm``, ``requires_iree or parity`` and
    ``requires_iree or requires_rocm`` do imply the leg, while
    ``not requires_iree``, ``not requires_rocm``, ``requires_iree and slow``,
    ``requires_iree and requires_rocm``, ``parity`` and ``not parity and not slow``
    do not.  The two markers are independent: a request for one is not a request
    for the other.

    A malformed expression returns ``False`` and is left to pytest's own ``-m``
    validation, which reports it with pytest's message; raising a second error
    here would only obscure that one.

    Args:
        expression: A ``-m`` expression taken from the command line.
        marker: The marker the expression might be requesting.

    Returns:
        bool: ``True`` only when the marker alone changes the selection.
    """
    try:
        compiled = Expression.compile(expression)
    except SyntaxError:
        return False
    selected_with = compiled.evaluate(lambda name, **kwargs: name == marker)
    selected_without = compiled.evaluate(lambda name, **kwargs: False)
    return selected_with and not selected_without


def implies_requires_iree(expression: str) -> bool:
    """Whether ``expression`` requests the llvm-cpu compiled-backend leg.

    The marker-defaulting wrapper around :func:`implies_marker`, kept because it
    is the predicate this directory's tests name explicitly; its behaviour is
    exactly ``implies_marker(expression, MARKER)``.

    Args:
        expression: A ``-m`` expression taken from the command line.

    Returns:
        bool: ``True`` only when ``requires_iree`` alone changes the selection.
    """
    return implies_marker(expression, MARKER)


def names_the_backends_directory(config: pytest.Config) -> bool:
    """Whether the command line names this directory or a test file inside it.

    Command-line arguments are compared as resolved paths, so option *values*
    (``-o addopts=``, ``-k expr``) are not mistaken for paths: only a token that
    really resolves to :data:`BACKENDS_DIR` or a path below it counts.  A broad
    scope such as ``pytest test`` or the implicit whole-tree collection does not.
    """
    for argument in config.invocation_params.args:
        token = str(argument).split("::", 1)[0]
        if not token or token.startswith("-"):
            continue
        try:
            candidate = Path(token).resolve()
        except OSError:  # an unparsable argument is not a path we claim
            continue
        if candidate == BACKENDS_DIR or BACKENDS_DIR in candidate.parents:
            return True
    return False


def pytest_configure(config: pytest.Config) -> None:
    """Stop a requested-but-unavailable compiled-backend run instead of skipping it.

    ``pytest_configure`` is a historic hook, so it also runs for this
    sub-directory conftest when pytest reaches ``test/backends`` during
    collection -- before the module-level ``importorskip`` could turn either leg
    into a single silent skip.
    """
    if iree_available():
        return

    requested = [
        (marker, expression)
        for marker in REQUESTED_MARKERS
        for expression in explicit_marker_expressions(config)
        if implies_marker(expression, marker)
    ]
    if requested:
        marker, expression = requested[-1]
        pytest.exit(
            "test/backends: this run explicitly requests the compiled-backend leg "
            f"(-m {expression!r}, which selects the {marker!r} marker) but the "
            "optional iree backend is not installed.\n"
            "Its absence must not read as a green '1 skipped' run, so the session "
            "stops here instead of collecting the skip.\n"
            f"Install it with: {INSTALL_COMMAND}",
            returncode=1,
        )

    if names_the_backends_directory(config):
        config.issue_config_time_warning(
            UserWarning(
                "test/backends was named on the command line but the optional iree "
                "backend is not installed: its tests are skipped as a structural "
                "optional-dependency skip, so this run proves nothing about the "
                f"compiled backend. Install it with: {INSTALL_COMMAND}. "
                "An explicit `pytest -m requires_iree` (or `-m requires_rocm`) run "
                "fails instead of skipping."
            ),
            stacklevel=1,
        )
