"""Release gate: ``import tensorneat`` must pull in none of the legacy frameworks.

What this module proves
-----------------------
The rewrite replaces the computation layer (``jax.jit``/``vmap``) with the EvoX
Compiler, so *the package* — not the tests, not the benchmarks — has to be free
of ``jax``, ``jaxlib``, ``torch``, ``brax``, ``gymnax`` and ``mujoco``. Four
separate statements, from weakest to strongest:

1. ``import tensorneat`` leaves every forbidden top-level module (and any
   submodule of one) out of ``sys.modules``.
2. The same import survives an import hook that raises ``ImportError`` for each
   forbidden top-level module: a green (1) can still hide a *deferred* import,
   which only fires later; a red (2) names the module that was reached for.
3. The documented public surface resolves — ``Pipeline``, ``algorithm.NEAT``/
   ``HyperNEAT``/``HyperNEATFeedForward``, ``genome.DefaultGenome``/
   ``RecurrentGenome``, ``problem.XOR``/``XOR3d``/``CustomFuncFit``/``CartPole``,
   ``common.ACT``/``AGG``/``State`` —
   and ``tensorneat.__file__`` lives under ``src/tensorneat/``, so the gate is
   exercising *this checkout* and not an unrelated installed copy.
4. A whole (tiny) XOR run still evolves on the numpy backend with the hook
   installed: freedom from JAX must not have been bought by breaking the
   pipeline.

Why every check is a subprocess
-------------------------------
The gate must not be able to hide a leak behind the parent interpreter's own
imports: ``pytest`` itself may have imported ``jax`` for ``./parity``, and a
module already in ``sys.modules`` is never re-imported. So each check runs
``[sys.executable, "-c", <script>]`` with ``capture_output=True``,
``PYTHONPATH`` set to the checkout's ``src/`` directory (derived from this file,
never hardcoded), ``PYTHONHASHSEED=0`` and a timeout, so a hung import fails the
gate instead of hanging CI. The subprocess is the oracle: its exit status, its
``LEAKED=``/``MISSING=`` report lines and its traceback are the only evidence a
test here is allowed to use.

Current state: green
--------------------
``src/tensorneat/pipeline.py`` is evoxc-native and ``import tensorneat`` succeeds
without binding ``jax``/``torch``/``brax``/``gymnax``/``mujoco``, so every test
here passes. Nothing here is skipped or xfailed — a skip would make the ratchet
invisible — and every failure message carries the offending module list or the
subprocess traceback, so it names what leaked or what raised.

No ``jax`` import happens in this module (the *subprocess script strings* may
mention module names), there is no network and no fixture. The file is collected
by ``conftest.py::pytest_collect_file`` because it is deliberately named without
a ``test_`` prefix; naming it explicitly on the command line collects each test
twice (that hook plus pytest's explicit-path handling), which is harmless but
why the repo-wide run is the canonical invocation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

#: Repository root, derived from this file (``<repo>/test/release_gate_no_jax.py``).
REPO_ROOT = Path(__file__).resolve().parents[1]
#: The ``src/`` directory put on the subprocess ``PYTHONPATH``.
SRC_DIR = REPO_ROOT / "src"
#: The package directory ``tensorneat.__file__`` must live under.
PACKAGE_DIR = SRC_DIR / "tensorneat"

#: Seconds a subprocess may take before the gate fails instead of hanging CI.
TIMEOUT_S = 300

#: Importing any of these (or a submodule of one) fails the gate.
FORBIDDEN = ("jax", "jaxlib", "torch", "brax", "gymnax", "mujoco")

#: The documented public surface, written as dotted paths so both the script and
#: the assertion message come from this one list.
PUBLIC_API = (
    "tensorneat.Pipeline",
    "tensorneat.algorithm.NEAT",
    "tensorneat.algorithm.HyperNEAT",
    "tensorneat.algorithm.HyperNEATFeedForward",
    "tensorneat.genome.DefaultGenome",
    "tensorneat.genome.RecurrentGenome",
    "tensorneat.problem.XOR",
    "tensorneat.problem.XOR3d",
    "tensorneat.problem.CustomFuncFit",
    "tensorneat.problem.CartPole",
    "tensorneat.common.ACT",
    "tensorneat.common.AGG",
    "tensorneat.common.State",
)


# --------------------------------------------------------------------------- #
# subprocess plumbing (the only way a check is allowed to observe the package)
# --------------------------------------------------------------------------- #


def _environment() -> dict[str, str]:
    """The controlled environment of every subprocess.

    The inherited environment is kept (the interpreter's own ``site-packages``
    must stay reachable — ``evoxc``, ``numpy`` and ``sympy`` live there), but
    ``PYTHONPATH`` is *replaced* so the checkout's ``src/`` wins over anything
    the caller injected, and ``PYTHONHASHSEED`` is pinned so a failure is
    reproducible.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)
    env["PYTHONHASHSEED"] = "0"
    return env


def _run(script: str, timeout: float = TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a fresh interpreter rooted at the repository.

    Args:
        script: Python source executed as ``python -c``; it must print its
            findings as ``KEY=value`` lines so this side can parse them.
        timeout: Seconds before the run is abandoned.

    Returns:
        The completed process, output captured as text.

    Raises:
        pytest.fail.Exception: If the script hangs, so a stuck import is a red
            gate with a readable message rather than a stuck CI job.
    """
    try:
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"the release-gate subprocess did not finish within {timeout} s "
            f"(interpreter {sys.executable}, PYTHONPATH={SRC_DIR}).\n"
            f"--- script ---\n{script}\n"
            f"--- partial stdout ---\n{exc.stdout}\n"
            f"--- partial stderr ---\n{exc.stderr}"
        )


def _script(body: str, *, hook: bool = False) -> str:
    """Assemble a subprocess script: forbidden list, optional hook, ``body``.

    Args:
        body: The check itself; it must ``import tensorneat`` after the hook.
        hook: Whether to install the ``sys.meta_path`` finder *before* the body
            runs, so a deferred import of a forbidden module raises ``ImportError``.
            The modules the check reached for are reported after it.
    """
    parts = ["import sys", f"FORBIDDEN = {FORBIDDEN!r}"]
    if hook:
        parts.append(_IMPORT_HOOK)
    parts.append(textwrap.dedent(body))
    if hook:
        parts.append(_BLOCKED_REPORT)
    return "\n\n".join(parts)


def _succeeded(completed: subprocess.CompletedProcess[str], what: str) -> str:
    """Return the subprocess stdout, failing with its full output if it did not exit 0.

    The traceback is the whole point: when ``import tensorneat`` raises, the
    message has to name the module and the error, not just say "non-zero exit".
    """
    assert completed.returncode == 0, (
        f"{what} failed (exit code {completed.returncode}).\n"
        f"--- stdout ---\n{completed.stdout}\n"
        f"--- stderr ---\n{completed.stderr}\n"
        f"--- environment ---\nPYTHONPATH={SRC_DIR} PYTHONHASHSEED=0 "
        f"interpreter={sys.executable}"
    )
    return completed.stdout


def _line(stdout: str, key: str) -> str:
    """Read the raw value of a ``KEY=value`` line a script printed.

    Args:
        stdout: The subprocess output.
        key: The report key, e.g. ``"PACKAGE"``.

    Returns:
        Everything after ``KEY=``, verbatim (commas and all, unlike :func:`_report`).
    """
    prefix = f"{key}="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise AssertionError(
        f"the release-gate subprocess never printed a {key}= line; its output was:\n{stdout}"
    )


def _report(stdout: str, key: str) -> list[str]:
    """Read the ``KEY=a,b`` line a script printed back as a list of names.

    Args:
        stdout: The subprocess output.
        key: The report key, e.g. ``"LEAKED"`` or ``"MISSING"``.

    Returns:
        The comma-separated values, empty when the line is ``KEY=``.
    """
    return [item for item in _line(stdout, key).split(",") if item]


#: A ``sys.meta_path`` finder that refuses every forbidden top-level module.
#: It is installed at position 0 *before* ``tensorneat`` is imported, so an
#: import that only happens lazily (at first use, not at import time) fails
#: loudly instead of slipping past the ``sys.modules`` check.
_IMPORT_HOOK = '''\
class _ForbidLegacyFramework:
    """A ``sys.meta_path`` finder that raises for each forbidden top-level module."""

    def __init__(self, forbidden):
        self.forbidden = frozenset(forbidden)
        self.blocked = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in self.forbidden:
            self.blocked.append(fullname)
            raise ImportError(
                f"{fullname} is forbidden by the TensorNEAT release gate", name=fullname
            )
        return None


BLOCKER = _ForbidLegacyFramework(FORBIDDEN)
sys.meta_path.insert(0, BLOCKER)'''

#: Prints every forbidden module (or submodule of one) that reached ``sys.modules``.
_LEAK_REPORT = '''\
leaked = sorted(name for name in sys.modules if name.partition(".")[0] in FORBIDDEN)
print("LEAKED=" + ",".join(leaked))'''

#: Prints what the import hook refused, so a red gate names the deferred import.
_BLOCKED_REPORT = 'print("BLOCKED=" + ",".join(BLOCKER.blocked))'

_IMPORT_BODY = f"""\
import tensorneat  # the only import under test

{_LEAK_REPORT}
print("PACKAGE=" + str(tensorneat.__file__))
"""

_PUBLIC_API_BODY = f"""\
import tensorneat

PUBLIC_API = {PUBLIC_API!r}


def resolve(dotted):
    obj = tensorneat
    for part in dotted.split(".")[1:]:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return False
    return True


missing = [dotted for dotted in PUBLIC_API if not resolve(dotted)]
print("MISSING=" + ",".join(missing))
print("PACKAGE=" + str(tensorneat.__file__))
"""

_XOR_RUN_BODY = f"""\
import tensorneat

pipeline = tensorneat.Pipeline(
    algorithm=tensorneat.algorithm.NEAT(
        genome=tensorneat.genome.DefaultGenome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=8,
            max_conns=16,
        ),
        pop_size=8,
        species_size=2,
        min_species_size=1,
    ),
    problem=tensorneat.problem.XOR(),
    seed=42,
    fitness_target=4.0,
    generation_limit=2,
    backend="numpy",
    is_detailed=False,
)

state = pipeline.setup()
result = pipeline.auto_run(state)
if not isinstance(result, tuple):
    result = (result,)
assert result[0] is not None, "Pipeline.auto_run returned no state"

print("RUN=ok")
{_LEAK_REPORT}
"""


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def test_import_pulls_no_legacy_framework() -> None:
    """Invariant: ``import tensorneat`` loads no JAX/torch/brax/gymnax/mujoco module.

    Checked in a clean interpreter, so an import made anywhere else in the
    pytest session cannot mask the leak (a module already in ``sys.modules`` is
    never re-imported). The report covers submodules too: ``jax.numpy``
    implies ``jax``, and a leaked submodule of a forbidden framework is the
    same defect.
    """
    completed = _run(_script(_IMPORT_BODY))
    stdout = _succeeded(completed, "`import tensorneat` in a clean subprocess")

    leaked = _report(stdout, "LEAKED")
    assert not leaked, (
        f"`import tensorneat` pulled in {leaked} (from {_line(stdout, 'PACKAGE')}); "
        "the evoxc rewrite must not depend on any legacy framework "
        f"({', '.join(FORBIDDEN)}). Offending modules: {', '.join(leaked)}"
    )


def test_import_is_guarded_by_an_import_hook() -> None:
    """Invariant: ``import tensorneat`` succeeds with a finder that refuses JAX/torch/...

    The module-list check above can only see imports that already happened. A
    *deferred* import — one that fires on first use, inside a function — passes
    it and leaks later, in a user's process. So a ``sys.meta_path`` finder is
    installed before ``tensorneat`` is imported and raises ``ImportError`` for
    every forbidden top-level module: any deferred import reached during the
    import now fails here. If the import fails, the traceback in the message
    names the module that was reached for.
    """
    completed = _run(_script(_IMPORT_BODY, hook=True))
    stdout = _succeeded(
        completed,
        "`import tensorneat` with a sys.meta_path finder that raises ImportError "
        f"for {', '.join(FORBIDDEN)}",
    )

    leaked = _report(stdout, "LEAKED")
    assert not leaked, (
        f"`import tensorneat` loaded {leaked} despite the import hook "
        f"(sender package: {_line(stdout, 'PACKAGE')})"
    )


def test_public_api_is_present() -> None:
    """Invariant: the documented entry points resolve, and they come from this checkout.

    ``Pipeline``, the three algorithms, the two genomes, the four problems and
    ``common.ACT``/``AGG``/``State`` are what ``README``/``docs`` promise, so a
    green release must actually export them. ``tensorneat.__file__`` is asserted
    to sit under ``src/tensorneat/``: a gate that silently imported an installed
    wheel would say nothing about the code under review.
    """
    completed = _run(_script(_PUBLIC_API_BODY))
    stdout = _succeeded(completed, "resolving the public API in a subprocess")

    missing = _report(stdout, "MISSING")
    assert not missing, (
        f"the public API is incomplete: {missing} did not resolve on "
        f"{_line(stdout, 'PACKAGE')}"
    )

    package_file = Path(_line(stdout, "PACKAGE")).resolve()
    assert package_file.is_relative_to(PACKAGE_DIR.resolve()), (
        f"`import tensorneat` resolved to {package_file}, which is outside "
        f"{PACKAGE_DIR.resolve()}: the gate must exercise the checkout, not an "
        "installed copy of the package"
    )


@pytest.mark.slow
def test_xor_generation_runs_without_legacy_frameworks() -> None:
    """Invariant: a small XOR evolution completes on the numpy backend, JAX-free.

    Being JAX-free is worthless if the pipeline no longer runs, so this is the
    end-to-end counterweight to the three import checks: ``NEAT`` over
    ``DefaultGenome`` evolves ``XOR`` for two generations with ``pop_size=8``
    under the same import hook, and afterwards ``sys.modules`` is inspected
    again — a leak that only appears once a graph is built or run shows up here.
    """
    completed = _run(_script(_XOR_RUN_BODY, hook=True))
    stdout = _succeeded(
        completed,
        "a two-generation XOR run on the numpy backend under the import hook",
    )

    assert _report(stdout, "RUN") == ["ok"], (
        f"the XOR subprocess never reported a completed run; its output was:\n{stdout}"
    )
    leaked = _report(stdout, "LEAKED")
    assert not leaked, (
        f"evolving XOR pulled in {leaked}; the numpy backend must stay free of "
        f"{', '.join(FORBIDDEN)}"
    )
