"""Loud-failure guard for the optional compiled-backend (``iree``) suite.

``test_iree_pipeline.py`` needs the optional ``evoxc[iree]`` extra, so it skips
at import when ``iree`` cannot be found.  That structural skip is the right shape
for a whole-tree run: the extra is optional and the numpy suites must stay green
without it.

It is the wrong shape when the compiled-backend leg was *explicitly requested*.
``pytest -m requires_iree`` in an environment without the extra used to exit 0
with "1 skipped", indistinguishable from a green run -- a silent degradation of
exactly the leg the user asked for.  This conftest closes that hole: when the
command line selects the ``requires_iree`` marker and ``iree`` is genuinely
absent, the session stops with a non-zero exit code and the install command
instead of collecting a skip.

Two intents are distinguished deliberately:

* an explicit ``-m`` selector whose expression implies ``requires_iree`` (e.g.
  ``-m requires_iree``, ``-m "requires_iree or parity"``) is a *capability*
  request -- the compiled backend must be there, so its absence is a hard error;
* merely naming ``test/backends`` on the command line (``pytest test/backends``)
  is a *scope* request, not a capability request -- the module still skips, but
  the skip is announced together with the install command so it cannot be
  mistaken for a green compiled-backend run.

Nothing here imports ``iree``: availability is probed with
:func:`importlib.util.find_spec`, so loading this file never pulls in an optional
dependency (and never pays the ~80 MB iree import).
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

#: The marker that selects the compiled-backend leg (registered in ``../conftest.py``).
MARKER = "requires_iree"

#: The optional top-level package the ``evoxc[iree]`` extra provides.
OPTIONAL_MODULE = "iree"

#: One install step: the project extra, which pins ``evoxc`` from git rather than PyPI.
#: ``uv pip install "evoxc[iree]"`` is deliberately *not* offered: it resolves the PyPI
#: release 0.1.0, which ``pyproject.toml`` forbids (the frontend work landed in 0.1.1).
INSTALL_COMMAND = (
    "uv sync --extra iree   (equivalently: pip install "
    '"evoxc[iree] @ git+https://github.com/EMI-Group/evoxc@v0.1.1")'
)


def iree_available() -> bool:
    """Whether ``import iree`` would find the optional compiled backend.

    The probe uses :func:`importlib.util.find_spec` and therefore never imports
    the iree extension.  ``find_spec`` raises ``ValueError`` when the module is
    already in :data:`sys.modules` without a ``__spec__``; that state still means
    it is importable, so it counts as available.
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


def implies_requires_iree(expression: str) -> bool:
    """Whether ``expression`` selects a test *because* it carries the ``requires_iree`` marker.

    The compiled ``-m`` expression is evaluated twice against the same synthetic
    test, one that carries the ``requires_iree`` marker and nothing else:

    * with a matcher that accepts that marker, i.e. the real selector;
    * with a matcher that accepts nothing, i.e. the same test as if it lacked
      the marker.

    The expression implies the marker only when the first selects the test and
    the second does not, so a selector that happens to include every marker-free
    test (``not parity and not slow``) is *not* misread as a capability request:
    it selects the bare-``requires_iree`` test twice, once with and once without
    the marker.

    Consequences: ``requires_iree`` and ``requires_iree or parity`` do imply the
    compiled-backend leg, while ``not requires_iree``, ``requires_iree and slow``,
    ``parity`` and ``not parity and not slow`` do not.

    A malformed expression returns ``False`` and is left to pytest's own ``-m``
    validation, which reports it with pytest's message; raising a second error
    here would only obscure that one.
    """
    try:
        compiled = Expression.compile(expression)
    except SyntaxError:
        return False
    selected_with = compiled.evaluate(lambda name, **kwargs: name == MARKER)
    selected_without = compiled.evaluate(lambda name, **kwargs: False)
    return selected_with and not selected_without


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
    collection -- before the module-level ``importorskip`` could turn the leg
    into a single silent skip.
    """
    if iree_available():
        return

    requested = [
        expression
        for expression in explicit_marker_expressions(config)
        if implies_requires_iree(expression)
    ]
    if requested:
        pytest.exit(
            "test/backends: this run explicitly requests the compiled-backend leg "
            f"(-m {requested[-1]!r}) but the optional iree backend is not "
            "installed.\n"
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
                "An explicit `pytest -m requires_iree` run fails instead of skipping."
            ),
            stacklevel=1,
        )
