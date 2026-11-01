"""Fixtures shared by the ``./unit/common`` suites: the three lazily resolved inputs.

Every module in this directory needs one of three things that is not importable
at *collection* time, so each is resolved through a fixture instead of a
module-scope import:

* ``tensorneat.common`` (:func:`common`). ``import tensorneat`` currently raises
  ``NotImplementedError``: ``tensorneat/algorithm/base.py`` evaluates ``State()``
  as a default argument while the class body is being created, and ``State`` is
  a design skeleton. Deferring the import to fixture time keeps every module
  collectable and turns the missing implementation into a named skip instead of
  a collection error. Once ``src/`` is implemented, the import can move back to
  module scope and this fixture becomes a one-line accessor.
* ``support.payloads`` (:func:`payloads`) and ``support.oracle``
  (:func:`oracle`). They are the hand-built gene payloads and the independent
  numpy reference oracles documented in ``../../support/CONTEXT.md`` and land
  with the support slice, so they are resolved with
  :func:`pytest.importorskip` and reported as a skip while absent.

Nothing here imports ``jax``/``torch``, ``numpy`` or ``evoxc``: the fixtures
return modules, and the suites import what they need from them. That keeps the
per-test guard in ``../conftest.py`` (no optional framework may enter
``sys.modules``) honest for every module in this directory.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def common():
    """The ``tensorneat.common`` package under test.

    Returns:
        The imported package, so a test reads ``common.tools.flat_gather`` /
        ``common.State`` / ``common.tensorization.validate_schedule``.

    Notes:
        The import is deferred (see the module docstring): ``import tensorneat``
        raises ``NotImplementedError`` from the ``src/`` design skeletons today,
        and only that pending-implementation error is turned into a skip —
        an ``ImportError`` or any other failure still surfaces as an error,
        because those mean the package is broken, not unfinished.
    """
    try:
        import tensorneat.common as package
    except NotImplementedError as exc:  # src/ design skeleton, not a test failure
        pytest.skip(f"not implemented: importing tensorneat.common raises {exc!r}")
    return package


@pytest.fixture(scope="module")
def payloads():
    """The hand-built gene payloads of ``../../support/payloads.py``.

    Returns:
        The ``support.payloads`` module: ``new_payload``, ``set_node``,
        ``set_conn``, ``chain_payload``, ``skip_payload``, ``isolated_payload``,
        ``ring_payload``, ``default_genome``, ``genome_state`` and ``I_INF``.

    Notes:
        ``importorskip`` (not a plain import) because the support slice is a
        sibling deliverable: while it is missing the dependent tests skip with
        this reason instead of failing collection.
    """
    return pytest.importorskip(
        "support.payloads",
        reason="shared payload builders (./support) are not in this checkout yet",
    )


@pytest.fixture(scope="module")
def oracle():
    """The independent numpy reference oracles of ``../../support/oracle.py``.

    Returns:
        The ``support.oracle`` module: ``schedule`` (the reference tensorization
        derivation) and, for the other suites, ``forward``/``func_fit_loss``/
        ``cartpole_rollout``.

    Notes:
        Oracle code is written from the documented algorithm, never by calling
        the kernel it checks — see ``../../support/CONTEXT.md``. Resolved with
        ``importorskip`` for the same reason as :func:`payloads`.
    """
    return pytest.importorskip(
        "support.oracle",
        reason="independent oracles (./support) are not in this checkout yet",
    )
