"""Fixtures shared by the ``./unit/genome`` suites: five modules and three graph runners.

Two kinds of fixture live here, and both exist for the same reason — a test
module in this directory must be *collectable* before ``src/`` is finished.

**Inputs, resolved lazily.** ``import tensorneat`` still raises
``NotImplementedError``: ``tensorneat/algorithm/base.py`` evaluates ``State()``
as a default argument while its class body is being created, and ``State`` is a
design skeleton. A module-scope ``from tensorneat.genome import DefaultGenome``
would therefore turn every module here into a collection error instead of a
named skip. So ``tensorneat.genome``, ``tensorneat.common``, ``support.payloads``,
``support.oracle`` and ``support.runner`` are each resolved through
:func:`support.lazy.require`, which converts *only* the pending-implementation
``NotImplementedError`` into a skip; an ``ImportError`` (the package is broken)
still surfaces. The import sits inside the fixture body, never at module scope.
The ``import support.lazy`` itself is deferred for the same reason
``../conftest.py`` prepends ``test/`` to ``sys.path``: it must happen after
collection has set the path up, not while this file is being imported.

**Graph glue, one place.** ``support/runner.py`` is the only module allowed to
call ``evoxc.build``/``run``/``lower``/``compile``/``load``
(``../../support/CONTEXT.md``), and it takes *top-level* numpy leaves as data.
That single rule has two consequences these fixtures encode so no test module
re-derives them:

* a method must be executed through a closure —
  ``run_defn(lambda *arrays: obj.method(state, *arrays), ...)`` — because two
  nested ``@evoxc.defn`` functions may not call each other (:func:`run_kernel`);
* a ``Transformed`` schedule is a *namedtuple*, and the runner classifies only
  top-level array leaves as data, so passing the schedule as one argument would
  bake it into the trace as a static value whose fields are plain numpy arrays —
  every ``transformed.in_conn`` inside the kernel would then be an ndarray, not
  a symbolic tensor, and the trace fails with a ``TypeError``. The schedule is
  therefore passed **field by field** and rebuilt inside the traced closure,
  where constructing a namedtuple is a pure-Python operation
  (:func:`run_transform` produces the fields, :func:`run_forward` consumes them).
  ``evoxc`` itself accepts a pytree of tensor arguments, so this is a property
  of the shared runner, not of the compiler; see the report to the parent.

Nothing here imports ``jax``/``torch``, ``evoxc`` or ``numpy`` at module scope:
the fixtures return modules and closures, and the suites import what they need
from them. That keeps the per-test guard in ``../conftest.py`` (no optional
framework may enter ``sys.modules``) honest for every module in this directory.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def genome():
    """The ``tensorneat.genome`` package under test.

    Returns:
        The imported package, so a test reads ``genome.DefaultGenome`` /
        ``genome.RecurrentGenome`` / ``genome.DefaultNode`` /
        ``genome.GenomeTensors`` / ``genome.DefaultMutation`` and the rest of the
        public surface listed in ``tensorneat/genome/__init__.py``.

    Notes:
        The import is deferred (see the module docstring). Only the
        pending-implementation ``NotImplementedError`` becomes a skip; today the
        package does not even get that far — ``import tensorneat`` raises from
        ``common/state.py`` first — and ``support.lazy.require`` catches that
        same error whichever module raised it.
    """
    from support.lazy import require

    return require("tensorneat.genome")


@pytest.fixture(scope="module")
def common():
    """The ``tensorneat.common`` package (``State``, ``I_INF``, ``Transformed``, errors).

    Returns:
        The imported package: ``common.State`` to build the host container the
        genome registers its configuration on, ``common.I_INF`` for the ``int32``
        padding sentinel, ``common.Transformed`` to rebuild a schedule inside a
        trace, ``common.validate_schedule`` for the schedule oracle and
        ``common.TensorNEATError`` for the configuration-error assertions.

    Notes:
        Resolved separately from :func:`genome` so a test that only needs the
        sentinel or the error class does not depend on the genome subtree.
    """
    from support.lazy import require

    return require("tensorneat.common")


@pytest.fixture(scope="module")
def payloads():
    """The hand-built gene payloads of ``../../support/payloads.py``.

    Returns:
        The ``support.payloads`` module: ``new_payload``, ``set_node``,
        ``set_conn``, ``chain_payload``, ``skip_payload``, ``isolated_payload``,
        ``ring_payload``, ``default_genome``, ``recurrent_genome``,
        ``genome_state`` and ``I_INF``.

    Notes:
        Every topology in this directory is written by hand, never drawn, so a
        test asserts a value it can print. ``support.payloads`` imports
        ``tensorneat.common.tools`` at module scope, so while ``src/`` is
        unfinished this fixture skips too — with the payload module named in the
        reason, which is what tells a reader which slice is missing.
    """
    from support.lazy import require

    return require("support.payloads")


@pytest.fixture(scope="module")
def oracle():
    """The independent numpy reference oracles of ``../../support/oracle.py``.

    Returns:
        The ``support.oracle`` module: ``schedule`` (the reference tensorization
        derivation), ``forward`` (the NaN-aware level-synchronous / tick-based
        rollout), ``default_node_attrs_of``/``default_conn_attr_of`` (the column
        readers a non-default gene layout overrides) and the activation /
        aggregation kernels re-exported from ``support.functions_ref``.

    Notes:
        The oracle is written from the documented algorithm, never by calling the
        kernel it checks — otherwise the assertion is a tautology
        (``../../support/CONTEXT.md``).
    """
    from support.lazy import require

    return require("support.oracle")


@pytest.fixture(scope="module")
def runner():
    """The one sanctioned bridge to ``evoxc`` (``../../support/runner.py``).

    Returns:
        The ``support.runner`` module: ``run_defn`` (build + run one call),
        ``build_defn`` (a reusable executable), ``run_graph`` (lower → compile →
        load → run, for the ``evoxc.Graph`` that ``genome.grad`` returns) and
        ``to_numpy``.

    Notes:
        Tests call these through the module rather than importing the three names
        at module scope, so a support-slice regression is reported once, as a
        skip naming ``support.runner``.
    """
    from support.lazy import require

    return require("support.runner")


@pytest.fixture(scope="module")
def run_kernel(runner):
    """Execute one genome kernel (a bound method) inside a single traced closure.

    Returns:
        ``run(method, state, *args) -> host numpy``: ``method`` is any genome
        kernel that takes the host ``state`` first and tensors afterwards
        (``genome.forward``, ``genome.execute_mutation``,
        ``genome.execute_crossover``, ``genome.execute_distance``,
        ``genome.transform`` ...), ``state`` is static configuration, ``*args``
        are the tensors/statics of the call in the method's own order.

    Notes:
        A *method* cannot be traced directly: it is not an ``@evoxc.defn``
        function, and nesting two defns is not allowed. Closing over the bound
        method and the host objects, and passing only tensors/statics, is the
        documented workaround (``../../support/CONTEXT.md``). A ``Transformed``
        argument must be flattened by the caller — see :func:`run_forward` and
        the module docstring.
    """

    def run(method, state, *args):
        return runner.run_defn(lambda *arrays: method(state, *arrays), *args)

    return run


@pytest.fixture(scope="module")
def run_transform(runner):
    """Derive the execution schedule of a whole payload through ``genome.transform``.

    Returns:
        ``run(genome_object, state, payload) -> Transformed`` of numpy arrays
        (``support.runner.to_numpy`` rebuilds the namedtuple, so the result keeps
        its field names and can be compared field by field against
        ``support.oracle.schedule``).

    Notes:
        ``transform`` is the public entry point, so the tests never name a
        ``Transformed`` field that the derivation might rename; the schedule's
        *contents* are checked against the oracle instead. Deriving it is a
        separate graph from the forward pass because ``argsort`` has no AD rule
        and because the trace is then cached across the population.
    """

    def run(genome_object, state, payload):
        return runner.run_defn(
            lambda node_keys, conn_keys: genome_object.transform(state, node_keys, conn_keys),
            payload.node_keys,
            payload.conn_keys,
        )

    return run


@pytest.fixture(scope="module")
def run_forward(common, runner):
    """Evaluate one payload's networks with an already-derived schedule.

    Returns:
        ``run(genome_object, state, transformed, payload, inputs) -> (P,
        num_outputs)`` numpy output of ``genome_object.forward``.

    Notes:
        The schedule is passed **field by field** and the ``Transformed`` is
        rebuilt inside the traced closure, for the reason spelled out in the
        module docstring: the shared runner hands only top-level arrays to
        ``evoxc.build`` as data, so a namedtuple argument would become a static
        value whose fields are numpy arrays. ``len(common.Transformed._fields)``
        is read instead of a literal, so adding a field to the schedule does not
        silently shift the payload arguments. The same helper serves
        ``DefaultGenome.forward`` (one step per level) and
        ``RecurrentGenome.forward`` (one step per tick); only the schedule they
        are given differs.
    """
    width = len(common.Transformed._fields)

    def run(genome_object, state, transformed, payload, inputs):
        fields = tuple(transformed)
        if len(fields) != width:
            raise ValueError(
                f"expected the {width} fields of Transformed, got {len(fields)}"
            )
        return runner.run_defn(
            lambda *arrays: genome_object.forward(
                state, common.Transformed(*arrays[:width]), *arrays[width:]
            ),
            *fields,
            *payload.args(),
            inputs,
        )

    return run
