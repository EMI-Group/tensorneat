"""Release gate: the package *surface* — every module imports, every ``__all__`` resolves.

What this module proves
-----------------------
``release_gate_no_jax.py`` proves that ``import tensorneat`` binds none of the
forbidden frameworks. That is a statement about the package *root*. A rewrite
can pass it and still ship a subtree that no longer imports at all: a module
whose body raises ``NotImplementedError`` (a port that was started and not
finished), or a package that advertises names in ``__all__`` which no longer
exist. Three statements, the third being the one the other gate could not make:

1. **Import surface** — in a fresh subprocess, every module under
   ``src/tensorneat`` (``pkgutil.walk_packages`` over the imported package) is
   imported by name. No module may raise ``NotImplementedError``: an unfinished
   port must fail *here*, not be turned into a skip by a lazy-import fixture (the
   ``test/parity`` ``support.lazy.require`` helper deliberately converts a
   ``NotImplementedError`` raised at import into a named skip, so this gate is
   the only place such a module is red). Every name in a module's ``__all__``
   must also resolve via ``getattr``. The walk must find a plausible number of
   modules, so an empty walk cannot pass vacuously.
2. **Static source** — in-process, no imports: every ``.py`` file under
   ``src/tensorneat`` is parsed with :mod:`ast` and the top-level roots of its
   ``import``/``from ... import`` statements, plus any string literal passed to
   ``__import__(...)`` or ``<anything>.import_module(...)``, are intersected
   with the forbidden set. ``file:line:name`` for each offender goes into the
   failure message.
3. **The forbidden set is enforced, not merely uninstalled** — the import hook
   shared with ``release_gate_no_jax.py`` must refuse *each* forbidden name, so
   a green import gate cannot be an artefact of ``jax``/``torch``/... simply not
   being importable on the box.

Where the shared state comes from
---------------------------------
The forbidden tuple and the subprocess plumbing (controlled environment,
``PYTHONPATH=<repo>/src``, ``PYTHONHASHSEED=0``, the timeout, the ``KEY=value``
report parser and the ``sys.meta_path`` hook) are imported from
``release_gate_no_jax`` — one list and one environment for both gates, so the
two cannot drift. ``test/conftest.py`` puts this directory on ``sys.path``, and
this module asserts the paths it derives from ``__file__`` are the ones that
gate uses.

Current state: green
--------------------
All 87 modules of ``src/tensorneat`` import cleanly, every ``__all__`` entry
resolves, no file under ``src/tensorneat`` mentions a forbidden framework, and
the hook refuses all eight names. Nothing here is skipped or xfailed, no
optional dependency is used (the subprocess imports only the standard library
plus the package under test), and every failure message names the offender.

The file needs no collection hook: ``test_*.py`` is pytest's default pattern.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: This directory, which holds the shared gate module (put on ``sys.path`` by
#: ``conftest.py``).
TEST_DIR = Path(__file__).resolve().parent
#: Repository root, derived from this file (``<repo>/test/...``).
REPO_ROOT = TEST_DIR.parent
#: The ``src/`` directory the subprocess puts on ``PYTHONPATH``.
SRC_DIR = REPO_ROOT / "src"
#: The package directory the AST walk covers and the subprocess must import.
PACKAGE_DIR = SRC_DIR / "tensorneat"

#: One list, one environment for both release gates: reusing the import-hygiene
#: gate's plumbing is what keeps ``FORBIDDEN``, the subprocess environment and the
#: ``sys.meta_path`` hook identical between them.
import release_gate_no_jax as gate  # noqa: E402  (resolved via ``conftest.py``'s ``sys.path``)

#: Importing any of these (or a submodule of one) fails the gate.
FORBIDDEN = gate.FORBIDDEN

#: A lower bound on the walked module count: ``src/tensorneat`` ships 87 modules,
#: so a walk that reports fewer than 80 has silently lost the tree (and an empty
#: walk must never pass a gate).
MIN_MODULES = 80

#: A lower bound on the scanned file count, in the same spirit.
MIN_FILES = 80

#: The subprocess script of gate 1: import every module, check every ``__all__``.
_IMPORT_SURFACE_BODY = '''\
import importlib
import pkgutil

import tensorneat

MODULES = sorted(
    info.name for info in pkgutil.walk_packages(tensorneat.__path__, "tensorneat.")
)

bad = []
missing = []
for name in MODULES:
    try:
        module = importlib.import_module(name)
    except NotImplementedError as exc:
        reason = f"{type(exc).__name__}: {exc}".replace(",", ";").replace("\\n", " ")
        bad.append(f"{name}:{reason}")
        continue
    for attribute in getattr(module, "__all__", ()) or ():
        try:
            getattr(module, attribute)
        except AttributeError:
            missing.append(f"{name}:{attribute}")

print("MODULES=" + str(len(MODULES)))
print("ROOT=" + str(list(tensorneat.__path__)[0]))
print("BAD=" + ",".join(bad))
print("MISSING=" + ",".join(missing))
'''

#: The sentence ``release_gate_no_jax.py``'s ``sys.meta_path`` finder puts in its
#: ``ImportError``; a probe that sees it knows the *hook* refused the name and not
#: the import machinery (a module that is merely not installed raises
#: ``ModuleNotFoundError`` instead).
_HOOK_MARKER = "is forbidden by the TensorNEAT release gate"

#: The subprocess script of gate 3: ask the hook for each forbidden name.
_HOOK_PROBE_BODY = '''\
MARKER = %r
refused = []
not_refused = []

for name in FORBIDDEN:
    try:
        __import__(name)
    except ImportError as exc:
        if getattr(exc, "name", None) == name and MARKER in str(exc):
            refused.append(name)
        else:
            detail = f"{type(exc).__name__}: {exc}".replace(",", ";")
            not_refused.append(f"{name}:{detail}")
    else:
        not_refused.append(f"{name}:imported")

print("REFUSED=" + ",".join(refused))
print("NOTREFUSED=" + ",".join(not_refused))
''' % (_HOOK_MARKER,)

#: A positive control for the AST scanner: every spelling the scanner claims to
#: understand, one line each, so a broken walk cannot pass gate 2 vacuously.
_POSITIVE_CONTROL_SOURCE = '''\
import jax
import jax.numpy as jnp
import flax.linen as linen

from optax import adam
from torch import nn
from brax.envs import Env
from gymnax import EnvParams

import mujoco

dynamic_one = __import__("jax")
dynamic_two = importlib.import_module("optax")
dynamic_three = loader.import_module("torch.distributed")
'''

#: The names the positive control must yield, in file order.
_POSITIVE_CONTROL_EXPECTED = (
    "jax",
    "jax",
    "flax",
    "optax",
    "torch",
    "brax",
    "gymnax",
    "mujoco",
    "jax",
    "optax",
    "torch",
)


# --------------------------------------------------------------------------- #
# gate 1: every module imports, every ``__all__`` entry resolves
# --------------------------------------------------------------------------- #


def test_every_module_imports_and_every_all_resolves() -> None:
    """Invariant: all of ``src/tensorneat`` imports, and every ``__all__`` name exists.

    The walk runs in a fresh interpreter so a module the parent pytest process
    already imported (or poisoned) cannot make it green, and it imports each
    module *by name* rather than trusting the import graph reachable from
    ``tensorneat``: a subtree that nothing imports at package-import time is
    exactly the one that rots unnoticed.
    """
    completed = gate._run(gate._script(_IMPORT_SURFACE_BODY))
    stdout = gate._succeeded(
        completed,
        "walking and importing every module under src/tensorneat in a subprocess",
    )

    bad = gate._report(stdout, "BAD")
    assert not bad, (
        "these modules raise NotImplementedError on import, i.e. they are "
        f"unfinished ports rather than a skip: {bad}. The walk reported "
        f"{gate._line(stdout, 'MODULES')} modules from {gate._line(stdout, 'ROOT')}."
    )

    missing = gate._report(stdout, "MISSING")
    assert not missing, (
        f"these `__all__` entries do not resolve via getattr (module:name): {missing}"
    )

    count = int(gate._line(stdout, "MODULES"))
    assert count >= MIN_MODULES, (
        f"the import walk found only {count} modules, below the {MIN_MODULES} this "
        f"package ships; the walk did not reach the tree at {PACKAGE_DIR}"
    )

    root = Path(gate._line(stdout, "ROOT")).resolve()
    assert root == PACKAGE_DIR.resolve(), (
        f"the import walk started at {root}, not the checkout's "
        f"{PACKAGE_DIR.resolve()}: it exercised an installed copy of the package"
    )


def test_the_walk_roots_at_this_checkout() -> None:
    """Invariant: the gate's own paths are the ones both gates use and they exist.

    Cheap, in-process, no subprocess: the package directory is derived from this
    file, it really is a directory with an ``__init__.py``, and the shared gate
    module (whose ``PYTHONPATH`` and ``FORBIDDEN`` this file borrows) resolves to
    the same place. A mismatch would make gate 1 inspect a different tree than
    gate 2 and than ``release_gate_no_jax.py``.
    """
    assert (PACKAGE_DIR / "__init__.py").is_file(), (
        f"{PACKAGE_DIR} is not an importable package directory"
    )
    assert gate.SRC_DIR.resolve() == SRC_DIR.resolve(), (
        f"the shared gate puts {gate.SRC_DIR.resolve()} on PYTHONPATH, this module "
        f"expects {SRC_DIR.resolve()}"
    )
    assert Path(gate.__file__).resolve().parent == TEST_DIR, (
        f"the shared gate module was imported from {gate.__file__}, not {TEST_DIR}"
    )


# --------------------------------------------------------------------------- #
# gate 2: the source tree imports no forbidden framework
# --------------------------------------------------------------------------- #


def _package_sources() -> list[Path]:
    """Every ``.py`` file under the checkout's ``src/tensorneat``, sorted."""
    return sorted(PACKAGE_DIR.rglob("*.py"))


def _imported_roots(source: str, filename: str = "<string>") -> list[tuple[str, int, str]]:
    """Parse ``source`` and return ``(filename, line, top-level module)`` per import.

    Covers static imports (``import a.b``, ``from a.b import c``) and the two
    dynamic spellings that appear in real code (``__import__("a")`` and
    ``something.import_module("a")``) when the module name is a plain string
    literal. Relative imports (``from .jax import x``) are skipped: their root is
    a module of this package, not an installed framework.

    Raises:
        SyntaxError: If ``source`` is not valid Python — a broken file is a real
            defect and must not be mistaken for "no forbidden import".
    """
    tree = ast.parse(source, filename=filename)
    found: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((filename, node.lineno, alias.name.partition(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            found.append((filename, node.lineno, node.module.partition(".")[0]))
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            is_dunder_import = (
                isinstance(function, ast.Name) and function.id == "__import__"
            )
            is_import_module = (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
            )
            if not (is_dunder_import or is_import_module):
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                found.append((filename, node.lineno, argument.value.partition(".")[0]))
    return found


def _forbidden_imports(source: str, filename: str = "<string>") -> list[str]:
    """``file:line:name`` for every forbidden import found in ``source``."""
    return [
        f"{path}:{line}:{name}"
        for path, line, name in _imported_roots(source, filename)
        if name in FORBIDDEN
    ]


def test_source_tree_imports_no_forbidden_framework() -> None:
    """Invariant: no file under ``src/tensorneat`` imports a forbidden framework.

    The ``sys.modules`` check in ``release_gate_no_jax.py`` only sees imports that
    actually execute on the platform the gate runs on: an import guarded by
    ``if TYPE_CHECKING``, sitting behind a conditional, or inside a function that
    was never called is invisible to it but visible here. This is a static read of
    every file, so it also covers the modules nothing imports.
    """
    files = _package_sources()
    assert len(files) >= MIN_FILES, (
        f"the AST scan found only {len(files)} files under {PACKAGE_DIR}, below the "
        f"{MIN_FILES} the package ships: the scan did not reach the tree"
    )

    offenders: list[str] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        offenders.extend(_forbidden_imports(source, str(path.relative_to(REPO_ROOT))))

    assert not offenders, (
        "these source files import a forbidden framework "
        f"({', '.join(FORBIDDEN)}): {offenders}"
    )


def test_scanner_detects_a_known_offense() -> None:
    """Positive control: every spelling the scanner claims to read is caught.

    Without this, a scanner that silently returned nothing — a walk that never
    visits a node, an attribute name that drifted, a filter that swallowed the
    comparison — would make the gate above pass on a leaked tree. The synthetic
    source exercises static imports, dotted imports, the ``from`` form,
    ``__import__`` and ``<anything>.import_module``; the assertion pins both the
    names and the order (i.e. that line numbers are real, not fabricated).
    """
    detected = _imported_roots(_POSITIVE_CONTROL_SOURCE, "control.py")
    assert tuple(name for _, _, name in detected) == _POSITIVE_CONTROL_EXPECTED, (
        "the AST scanner read "
        f"{[name for _, _, name in detected]}, expected {list(_POSITIVE_CONTROL_EXPECTED)}"
    )
    lines = [line for _, line, _ in detected]
    assert lines == sorted(lines) and len(set(lines)) == len(lines), (
        f"the AST scanner reported implausible line numbers {lines}"
    )
    assert {path for path, _, _ in detected} == {"control.py"}, (
        f"the AST scanner mislabelled its source: {[path for path, _, _ in detected]}"
    )


# --------------------------------------------------------------------------- #
# gate 3: the hook really refuses each forbidden name
# --------------------------------------------------------------------------- #


def test_import_hook_refuses_every_forbidden_name() -> None:
    """Invariant: the shared import hook raises ``ImportError`` for all eight names.

    ``test_import_is_guarded_by_an_import_hook`` in ``release_gate_no_jax.py``
    can only prove that ``import tensorneat`` did not *reach for* a forbidden
    module. If a framework were simply not installed, that test would be green
    for the wrong reason. Here the ``sys.meta_path`` finder is installed and each
    forbidden name is imported on purpose: every one has to be refused by the
    hook itself — an ``ImportError`` carrying that name and the hook's own
    message, not a ``ModuleNotFoundError`` from an absent installation — because
    only then does the earlier green mean "nothing tried" rather than "nothing
    could".
    """
    assert _HOOK_MARKER in gate._IMPORT_HOOK, (
        f"the shared import hook no longer says {_HOOK_MARKER!r}, so this probe "
        "cannot tell a hook refusal from an unavailable module"
    )
    completed = gate._run(gate._script(_HOOK_PROBE_BODY, hook=True))
    stdout = gate._succeeded(
        completed,
        "probing the sys.meta_path import hook with every forbidden module name",
    )

    refused = gate._report(stdout, "REFUSED")
    assert refused == list(FORBIDDEN), (
        f"the import hook refused {refused} but must refuse {list(FORBIDDEN)}"
    )
    not_refused = gate._report(stdout, "NOTREFUSED")
    assert not not_refused, (
        f"the import hook did not refuse {not_refused}: a green import gate would "
        "then rest on these modules merely being unavailable"
    )
