"""Integrity of the differential evidence: the oracle is frozen, independent and actually collected.

Why this module exists
----------------------
Every comparison in ``./`` rests on two assumptions that no value assertion can
check for itself:

1. ``benchmarks/legacy_jax/tensorneat_legacy`` is a *genuinely separate* JAX
   implementation, not a copy or re-export of ``src/tensorneat``.  If the oracle
   were re-exported from the code under test, every parity test would compare the
   rewrite against itself and stay green.
2. the four oracle modules are *collected*.  ``test_forward`` /
   ``test_distance`` / ``test_fitness`` / ``test_grad`` call
   ``pytest.importorskip("jax", ...)`` at module scope, so in an environment
   without the ``bench`` extra they vanish at collection and ``pytest -m parity``
   exits 0 with only the library-free tests — an empty differential suite that
   looks like a pass.  :func:`test_the_four_oracle_modules_import` fails when such
   a skip happens *although* jax is installed, and :func:`test_collected_parity_tests_are_ratcheted`
   pins the directory-wide collected count so "53 became 18" cannot pass unnoticed.

No module here imports ``jax`` at module scope, and the independence tests run in
any environment: they read the oracle *sources* with :mod:`ast` instead of
importing them (importing would be the one thing an integrity check must not do to
prove an import exists).  Only the two guards that need a live oracle skip when
jax is absent, each with a reason naming what is therefore not collected.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.parity]

#: Repo root, from ``test/parity/test_oracle_integrity.py`` (``parents[2]``).
REPO_ROOT = Path(__file__).resolve().parents[2]
#: The frozen oracle tree; read-only input of every parity comparison.
LEGACY_ROOT = REPO_ROOT / "benchmarks" / "legacy_jax"
#: The oracle's own top-level package name (what the payload converters import).
LEGACY_PACKAGE_NAME = "tensorneat_legacy"
#: The top-level import name of the code *under test*, which the oracle must not use.
UNDER_TEST_TOP_NAME = "tensorneat"
#: The four modules whose module-scope ``importorskip`` can hide the oracle.
ORACLE_MODULES = ("test_forward", "test_distance", "test_fitness", "test_grad")
#: This directory, i.e. the tree the ratchet counts.
PARITY_DIR = Path(__file__).resolve().parent
#: Lower bound on the oracle's source-file count: the frozen library is a full
#: genome/algorithm/problem package (69 ``.py`` files today); a collapse to a few
#: shim modules would mean the oracle was replaced rather than frozen.
MIN_LEGACY_SOURCE_FILES = 20
#: The exact number of tests ``pytest --collect-only`` must find under ``test/parity``.
#: Update this constant when a parity test is added or removed — that is the point.
EXPECTED_COLLECTED_TESTS = 60


def _legacy_sources() -> list[Path]:
    """Every ``.py`` file of the frozen oracle tree, sorted for stable messages."""
    return sorted(path for path in LEGACY_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _imported_top_names(node: ast.AST) -> list[str]:
    """Top-level module names an ``import``/``from ... import`` node pulls in.

    ``import jax.numpy as jnp`` contributes ``"jax"``; ``from jax import grad``
    contributes ``"jax"``; ``from .. import base`` contributes nothing (a relative
    import has no absolute top name, so it cannot smuggle in the package under
    test from outside).
    """
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level or node.module is None:
            return []
        return [node.module.split(".")[0]]
    return []


def _scan_imports() -> list[tuple[Path, int, str]]:
    """``(path, lineno, top_name)`` for every absolute import in the oracle tree.

    Raises:
        AssertionError: If a source file of the oracle tree does not parse — an
            unparseable oracle is an integrity defect, not something to skip.
    """
    records: list[tuple[Path, int, str]] = []
    for path in _legacy_sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            pytest.fail(f"frozen oracle source does not parse: {path}:{exc.lineno}: {exc.msg}")
        for node in ast.walk(tree):
            for top_name in _imported_top_names(node):
                records.append((path, node.lineno, top_name))
    return records


# --------------------------------------------------------------------------- (a) independence
def test_the_frozen_oracle_tree_is_present():
    """Invariant: the oracle is a real package on disk, not a thin stub.

    Oracle: the tree itself.  ``benchmarks/legacy_jax/tensorneat_legacy`` is the
    frozen pre-rewrite JAX library every parity module imports as
    ``tensorneat_legacy`` (``./conftest.py``); a missing tree, a missing
    ``__init__.py`` or a handful of shim files would make every comparison
    vacuous while still importing successfully.

    Assertion: the directory exists, ``tensorneat_legacy/__init__.py`` exists, and
    the tree holds at least :data:`MIN_LEGACY_SOURCE_FILES` ``.py`` files.  The
    failure message names the path and the count actually found.
    """
    package_root = LEGACY_ROOT / LEGACY_PACKAGE_NAME
    assert package_root.is_dir(), (
        f"frozen oracle package is missing: expected directory {package_root}"
    )
    init_file = package_root / "__init__.py"
    assert init_file.is_file(), f"frozen oracle package has no __init__.py: {init_file}"

    sources = _legacy_sources()
    assert len(sources) >= MIN_LEGACY_SOURCE_FILES, (
        f"frozen oracle tree {LEGACY_ROOT} holds {len(sources)} .py files, "
        f"expected at least {MIN_LEGACY_SOURCE_FILES}; files found: "
        f"{[str(path.relative_to(REPO_ROOT)) for path in sources]}"
    )


def test_the_frozen_oracle_really_imports_jax():
    """Invariant: the oracle is the *JAX* implementation, so it is an independent engine.

    Oracle: the oracle sources, read with :mod:`ast` — never imported here, since
    the probe must work in an environment without jax and must not execute code it
    is only supposed to inspect.  A non-JAX oracle (e.g. a rewrite of the rewrite)
    would compare evoxc against evoxc and prove nothing.

    Assertion: at least one absolute import of top-level module ``jax`` (plain
    ``import jax`` or any ``from jax... import``) exists in the tree, and *every*
    hit is reported as ``path:line`` in the failure message otherwise.  Non-``.py``
    files and ``__pycache__`` are ignored; a file that does not parse fails.
    """
    sources = _legacy_sources()
    jax_imports = [
        (path, lineno)
        for path, lineno, top_name in _scan_imports()
        if top_name == "jax"
    ]
    assert jax_imports, (
        f"no import of `jax` in the frozen oracle tree {LEGACY_ROOT}; "
        f"scanned {len(sources)} .py files "
        f"({[str(path.relative_to(REPO_ROOT)) for path in sources]}) "
        f"and found no `import jax` / `from jax... import` statement"
    )
    # Name concrete witnesses so the evidence is in the report, not implied.
    witnesses = [f"{path.relative_to(REPO_ROOT)}:{lineno}" for path, lineno in jax_imports[:5]]
    assert all(path.is_relative_to(LEGACY_ROOT) for path, _ in jax_imports), (
        f"jax imports outside the oracle tree: {witnesses}"
    )


def test_the_oracle_never_imports_the_package_under_test():
    """Invariant: the oracle cannot be a re-export of ``src/tensorneat``.

    Oracle: the oracle sources, read with :mod:`ast`.  Its top-level package is
    ``tensorneat_legacy``; a statement importing the top-level name
    ``tensorneat`` would make the "oracle" load the code under test (and, in an
    editable install, the very worktree that is being graded).  ``tensorneat_legacy``
    is a *different* top name and is allowed everywhere.

    Assertion: no absolute import in the tree has top-level name ``tensorneat``;
    the failure message lists every offending ``path:line``.  Also asserts no
    top-level ``benchmarks/legacy_jax/tensorneat`` directory exists, because it
    would shadow the real package on ``sys.path`` (``../conftest.py`` inserts
    ``benchmarks/legacy_jax``).
    """
    shadow = LEGACY_ROOT / UNDER_TEST_TOP_NAME
    assert not shadow.exists(), (
        f"{shadow} would shadow the package under test on sys.path "
        f"(test/conftest.py inserts {LEGACY_ROOT}); the oracle's own package must be "
        f"the only top-level name there"
    )

    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path, lineno, top_name in _scan_imports()
        if top_name == UNDER_TEST_TOP_NAME
    ]
    assert not offenders, (
        f"the frozen oracle imports the package under test (`{UNDER_TEST_TOP_NAME}`) at "
        f"{offenders}; the oracle must be independent of src/{UNDER_TEST_TOP_NAME}"
    )


def test_the_package_under_test_is_not_the_oracle_copy():
    """Invariant: ``import tensorneat`` resolves to ``src/tensorneat``, not into the oracle tree.

    Oracle: :func:`importlib.util.find_spec`, so the check reads the resolved
    import origin without importing the package (and without needing jax).  An
    installed copy that pointed at ``benchmarks/legacy_jax`` would make
    ``tensorneat`` and the oracle the same code.

    Assertion: the spec has an origin, the origin is not inside
    ``benchmarks/legacy_jax``, and it is inside this repo's ``src/tensorneat``
    (editable install of the tree under test).  The failure message names the
    resolved origin and the forbidden prefix.
    """
    spec = importlib.util.find_spec(UNDER_TEST_TOP_NAME)
    assert spec is not None and spec.origin is not None, (
        f"`import {UNDER_TEST_TOP_NAME}` does not resolve to a file; "
        f"find_spec returned {spec!r}"
    )
    origin = Path(spec.origin).resolve()
    legacy_root = LEGACY_ROOT.resolve()
    assert not origin.is_relative_to(legacy_root), (
        f"`{UNDER_TEST_TOP_NAME}` resolves into the frozen oracle tree: origin={origin}, "
        f"forbidden prefix={legacy_root}"
    )
    expected_root = (REPO_ROOT / "src" / UNDER_TEST_TOP_NAME).resolve()
    assert origin.is_relative_to(expected_root), (
        f"`{UNDER_TEST_TOP_NAME}` resolves outside the tree under test: origin={origin}, "
        f"expected prefix={expected_root}"
    )


# --------------------------------------------------------------------------- (b) collection guard
def _jax_installed() -> bool:
    """Whether ``import jax`` succeeds, mirroring what ``importorskip`` tests.

    Imported inside the call (never at module scope) and only in this directory,
    the one place allowed to import ``jax`` (``../CONTEXT.md``).  An absent jax is
    a skip, not a failure; a *broken* jax surfaces as its own error.
    """
    try:
        importlib.import_module("jax")
    except ImportError:
        return False
    return True


def _not_collected_reason() -> str:
    """The shared skip reason naming exactly what is missing without jax."""
    modules = ", ".join(f"./{name}.py" for name in ORACLE_MODULES)
    return (
        "jax is not installed, so the four oracle modules "
        f"({modules}) call pytest.importorskip at module scope and are NOT "
        "collected: only the library-free tests (./test_convert.py and the "
        "independence checks in ./test_oracle_integrity.py) run, and no kernel is "
        "compared against the frozen implementation. Install the `bench` extra for "
        "the full differential evidence."
    )


def test_the_four_oracle_modules_import():
    """Invariant: with jax installed, no oracle module may skip itself at import.

    Oracle: the modules' own module-scope ``pytest.importorskip("jax", ...)``.
    Their skip is the mechanism that silently removes the differential evidence,
    so it must be observable *as a failure* whenever jax is importable — a module
    whose import raises :class:`pytest.skip.Exception` in that environment is a
    defect, not a legitimate skip.

    Assertion: each of the four modules is imported fresh from its file (no
    ``sys.modules`` reuse, so the module-scope guard really executes) and the
    result is a module object carrying the ``parity`` mark and at least one
    ``test_*`` callable; a raised ``pytest.skip.Exception`` is collected and turned
    into a failure naming the module, its path and the skip reason.  Without jax
    the test skips with :func:`_not_collected_reason` and never fails.
    """
    if not _jax_installed():
        pytest.skip(_not_collected_reason())

    failures: list[str] = []
    for name in ORACLE_MODULES:
        path = PARITY_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_parity_oracle_probe_{name}", path)
        assert spec is not None and spec.loader is not None, (
            f"cannot build an import spec for oracle module {path}"
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except pytest.skip.Exception as exc:
            failures.append(
                f"{path}: import raised pytest.skip({exc}) although jax is installed — "
                f"the module's module-scope importorskip dropped the oracle evidence"
            )
            continue
        marks = getattr(module, "pytestmark", [])
        marks = marks if isinstance(marks, list) else [marks]
        assert [getattr(mark, "name", mark) for mark in marks].count("parity") == 1, (
            f"{path}: imported but its module-level pytestmark does not carry the `parity` "
            f"marker (found {[getattr(mark, 'name', mark) for mark in marks]})"
        )
        test_names = [key for key in vars(module) if key.startswith("test_")]
        assert test_names, (
            f"{path}: imported but defines no `test_*` callable, so the module contributes "
            f"no parity evidence even though it was collected"
        )

    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------- (c) directory ratchet
def test_collected_parity_tests_are_ratcheted():
    """Invariant: ``test/parity`` still collects :data:`EXPECTED_COLLECTED_TESTS` tests.

    Oracle: a real ``pytest --collect-only`` subprocess over this directory, run
    with the current interpreter from the repo root with the project ``addopts``
    cleared (the repo sets ``-q``; doubling it suppresses the summary line).  Every
    module-scope skip or collection error shows up here as a missing item or a
    ``skipped``/``error`` line, so a silent degradation of the oracle evidence
    fails the suite even when the surviving tests are green.

    Assertion: exit status 0, the ``N tests collected`` summary equals
    :data:`EXPECTED_COLLECTED_TESTS`, the number of ``path::test`` ids equals that
    count, and no output line reports ``skipped``/``error``.  Failure messages
    embed the subprocess output.  Without jax the four oracle modules legitimately
    drop out, so the test skips with :func:`_not_collected_reason` instead.
    """
    if not _jax_installed():
        pytest.skip(_not_collected_reason())

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-o",
        "addopts=",
        "test/parity",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, (
        f"`{' '.join(command)}` failed with exit {completed.returncode}:\n{output}"
    )

    summary = re.search(r"^(\d+) tests? collected", completed.stdout, re.MULTILINE)
    assert summary is not None, (
        f"`{' '.join(command)}` printed no `N tests collected` summary:\n{output}"
    )
    collected = int(summary.group(1))
    assert collected == EXPECTED_COLLECTED_TESTS, (
        f"test/parity collects {collected} tests, expected {EXPECTED_COLLECTED_TESTS} "
        f"(update EXPECTED_COLLECTED_TESTS in ./test_oracle_integrity.py when a parity "
        f"test is added or removed):\n{output}"
    )

    ids = [line for line in completed.stdout.splitlines() if "::" in line]
    assert len(ids) == collected, (
        f"summary reports {collected} collected tests but {len(ids)} node ids were listed:\n{output}"
    )

    degraded = [
        line
        for line in output.splitlines()
        if re.search(r"\b(skipped|error|errors)\b", line, re.IGNORECASE)
    ]
    assert not degraded, (
        f"test/parity collection reported skipped/error lines although jax is installed: "
        f"{degraded}\n{output}"
    )
