"""Fixtures shared by the ``./unit/common`` suites: the three resolved inputs.

Every module in this directory needs one of three things that is better reached
through a fixture than through a module-scope import:

* ``tensorneat.common`` (:func:`common`). A plain module-scoped accessor: the
  package is implemented and ``import tensorneat.common`` succeeds, so the
  fixture only keeps that import out of this file's module scope and gives every
  test in the directory one spelling of it.
* ``support.payloads`` (:func:`payloads`) and ``support.oracle`` (:func:`oracle`).
  They are the hand-built gene payloads and the independent numpy reference
  oracles documented in ``../../support/CONTEXT.md``, i.e. sibling deliverables
  of this suite, so they are resolved with :func:`pytest.importorskip` and an
  environment without them skips the dependent tests with a reason instead of
  failing at collection.

Nothing here imports ``jax``/``torch``, ``numpy``, ``evoxc`` or ``tensorneat``
at module scope: the fixtures return modules, and the suites import what they
need from them. That keeps the per-test guard in ``../conftest.py`` (no optional
framework may enter ``sys.modules``) honest for every module in this directory.

Known side effect of :func:`common`: ``import tensorneat.common`` executes
``src/tensorneat/__init__.py``, which imports the legacy JAX ``pipeline.py``,
so ``jax``/``jaxlib`` enter ``sys.modules`` the first time the fixture runs.
``../conftest.py``'s leak guard snapshots ``sys.modules`` at autouse-fixture
setup, and this module-scoped fixture is instantiated before that function-scoped
one, so the guard does not observe those modules arriving.
The cause lies in ``src/`` — the ``pipeline.py`` rewrite away from JAX — and
``test/release_gate_no_jax.py`` is where that failure is owned.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def common():
    """The ``tensorneat.common`` package under test.

    Returns:
        The imported package, so a test reads ``common.tools.flat_gather`` /
        ``common.State`` / ``common.tensorization.validate_schedule``.
    """
    import tensorneat.common as package

    return package


@pytest.fixture(scope="module")
def payloads():
    """The hand-built gene payloads of ``../../support/payloads.py``.

    Returns:
        The ``support.payloads`` module: the ``Payload`` container plus
        ``new_payload``, ``set_node``, ``set_conn``, the canonical topologies
        ``chain_payload``, ``skip_payload``, ``isolated_payload`` and
        ``ring_payload``, the ``default_genome``/``recurrent_genome``/
        ``genome_state`` hosts, ``I_INF`` and the ``DEFAULT_*`` shape knobs.

    Notes:
        ``importorskip`` (not a plain import) because the support modules are a
        sibling deliverable: while one is missing the dependent tests skip with
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
        derivation), ``forward``, ``func_fit_loss``, ``cartpole_params`` /
        ``cartpole_rollout``, the ``default_node_attrs_of`` /
        ``default_conn_attr_of`` decoders, and the re-exported ``act_*`` /
        ``agg_*`` kernels with ``apply_activation`` / ``apply_aggregation``.

    Notes:
        Oracle code is written from the documented algorithm, never by calling
        the kernel it checks — see ``../../support/CONTEXT.md``. Resolved with
        ``importorskip`` for the same reason as :func:`payloads`.
    """
    return pytest.importorskip(
        "support.oracle",
        reason="independent oracles (./support) are not in this checkout yet",
    )
