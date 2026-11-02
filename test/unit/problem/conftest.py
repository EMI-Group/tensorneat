"""Fixtures shared by the ``./unit/problem`` suites: the resolved inputs, one
fake execution schedule and the two graph runners.

**Inputs, resolved inside a fixture body.** ``tensorneat.problem``,
``tensorneat.common``, ``tensorneat.algorithm``, ``support.payloads``,
``support.oracle``, ``support.runner``, ``evoxc`` and ``evoxc.numpy`` are
imported in the fixture body rather than at module scope, only because
``../conftest.py`` prepends ``src/`` and ``test/`` to ``sys.path`` at collection
time -- an import while this file is loaded would run too early. Every import is
a plain statement: a broken or regressed module must surface as an error, never
be converted into a skip, so a green run in this directory really executed every
kernel it claims to.

**Graph glue, one place.** ``support/runner.py`` is the only module allowed to
call ``evoxc.build``/``run``/``lower``/``compile``/``load``
(``../../support/CONTEXT.md``), and it classifies only *top-level* numpy leaves
as data. Three consequences are encoded here so no test module re-derives them:

* a problem method is executed through a closure
  (``run_defn(lambda *arrays: problem.evaluate(state, key, act_fn, sched), ...)``)
  because two nested ``@evoxc.defn`` functions may not call each other;
* the execution schedule is passed **field by field** and rebuilt inside the
  traced closure. Handing the namedtuple over as one argument would bake it
  into the trace as a static value whose fields are *numpy* arrays, so a
  kernel reading ``transformed.<field>`` would get an ndarray instead of a
  symbolic tensor;
* an evoxc RNG key is rebuilt in the trace from an integer ``key_seed`` rather
  than passed as data: a rank-0 ``int64`` key has no ``__array__`` and the
  shared runner's ``spec_like`` refuses it (``../../unit/common/test_tools.py``
  takes the same route).

**Why a fake schedule.** A problem may read exactly one thing from
``transformed`` -- ``static_batch_size(transformed)``, the static leading
dimension of its leaves -- so a problem test needs *some* batch-first pytree,
not a real ``Transformed``. Using the real one would drag the genome subtree
into every assertion and make a problem failure indistinguishable from a
genome failure. :data:`FakeSchedule` is that pytree: two leaves, both carrying
the population on axis 0, and nothing else.

Nothing here imports ``jax``/``torch``, ``numpy`` or ``evoxc`` at module scope:
the fixtures return modules and closures, and the suites import what they need
from them. That keeps the per-test guard in ``../conftest.py`` (no optional
framework may enter ``sys.modules``) honest for every module in this directory.
"""

from __future__ import annotations

from collections import namedtuple

import pytest


#: Stand-in for ``common.Transformed``: any batch-first pytree will do, because
#: ``static_batch_size`` may read nothing but the static leading dimension.
#: Field names are deliberately un-schedule-like (``weights``/``scores``): a
#: problem must not depend on the real schedule's schema, so a test that named a
#: real field would be testing the genome, not the problem.
FakeSchedule = namedtuple("FakeSchedule", ["weights", "scores"])


@pytest.fixture(scope="module")
def problem():
    """The ``tensorneat.problem`` package under test.

    Returns:
        The imported package, so a test reads ``problem.XOR``,
        ``problem.XOR3d``, ``problem.FuncFit``, ``problem.CustomFuncFit``,
        ``problem.CartPole`` and ``problem.BaseProblem`` -- the surface listed
        in ``tensorneat/problem/__init__.py``.

    Notes:
        Imported in the body because ``../conftest.py`` sets ``sys.path`` at
        collection time (see the module docstring).
    """
    import tensorneat.problem

    return tensorneat.problem


@pytest.fixture(scope="module")
def common():
    """The ``tensorneat.common`` package (``State``, errors, `TensorNEATError`).

    Returns:
        The imported package: ``common.State`` for the host container every
        ``evaluate``/``show`` signature takes, and ``common.TensorNEATError``
        (with ``common.ConfigError``) for the configuration-error assertions.

    Notes:
        Resolved separately from :func:`problem` so a test that only needs the
        error class does not depend on the problem subtree being importable.
    """
    import tensorneat.common

    return tensorneat.common


@pytest.fixture(scope="module")
def algorithm():
    """The ``tensorneat.algorithm`` package, used only for the arity cross-check.

    Returns:
        The imported package, so a test can build ``algorithm.NEAT(genome,
        pop_size)`` and compare ``neat.num_inputs``/``neat.num_outputs`` with
        ``problem.input_shape[-1]``/``problem.output_shape[-1]`` -- the same
        invariant ``Pipeline`` asserts between a problem and an algorithm.

    Notes:
        One test in ``test_problem_api.py`` needs it; every other test in this
        directory works on a genome-free ``act_fn``.
    """
    import tensorneat.algorithm

    return tensorneat.algorithm


@pytest.fixture(scope="module")
def payloads():
    """The hand-built gene payloads and genome factories of ``../../support/payloads.py``.

    Returns:
        The ``support.payloads`` module; this directory only uses
        ``default_genome`` (a ``DefaultGenome`` over ``num_inputs=2``,
        ``num_outputs=1``) as the genome that pairs with ``XOR``, plus ``I_INF``
        when a payload is involved.

    Notes:
        Only ``test_problem_api.py::test_the_input_and_output_arity_match_the_algorithm``
        needs it; every other test in this directory works on a genome-free
        ``act_fn``.
    """
    import support.payloads

    return support.payloads


@pytest.fixture(scope="module")
def oracle():
    """The independent numpy references of ``../../support/oracle.py``.

    Returns:
        The ``support.oracle`` module: ``func_fit_loss(pred, targets,
        error_method)`` (the reduction this directory checks),
        ``cartpole_params(**knobs)`` and ``cartpole_rollout(actions, obs0,
        max_step=...)`` (the rollout that is CartPole's only oracle, since the
        legacy RL problems are gone), and ``FUNC_FIT_ERROR_METHODS``.

    Notes:
        Written from the documented algorithm, never by calling the kernel it
        checks (``../../support/CONTEXT.md``).
    """
    import support.oracle

    return support.oracle


@pytest.fixture(scope="module")
def runner():
    """The one sanctioned bridge to ``evoxc`` (``../../support/runner.py``).

    Returns:
        The ``support.runner`` module: ``run_defn`` (build + run one call),
        ``build_defn`` (a reusable executable), ``run_graph`` and ``to_numpy``.

    Notes:
        Suites call it through this fixture (and the two runners below) rather
        than importing the names at module scope, so ``support.runner`` stays the
        single place that touches ``evoxc.build``/``run``/``lower``.
    """
    import support.runner

    return support.runner


@pytest.fixture(scope="module")
def evoxc():
    """The ``evoxc`` package: the ``float32`` dtype and ``random.key``.

    Returns:
        The imported package. ``evoxc.float32`` is what ``enp.astype`` wants
        (``evoxc.numpy`` has no dtype names of its own), and ``evoxc.random``
        is how a test spells "one key" -- evoxc has no batched key type.

    Notes:
        ``enp`` (the traced-op namespace) is a separate fixture: ops and
        dtypes/keys are different concerns, and a dummy ``act_fn`` usually
        needs both.
    """
    import evoxc

    return evoxc


@pytest.fixture(scope="module")
def enp():
    """``evoxc.numpy`` -- the only op namespace a traced kernel may use.

    Returns:
        The module, so a genome-free ``act_fn`` is written with ``enp.full`` /
        ``enp.arange`` / ``enp.split`` / ``enp.stack`` / ``enp.where``.

    Notes:
        A traced tensor has no ``__getitem__`` and there is no eager mode, so a
        dummy predictor is built from ``enp`` ops only -- ``np.*`` on a traced
        value would silently produce a host array at trace time.
    """
    from evoxc import numpy as enp

    return enp


@pytest.fixture(scope="module")
def state(common):
    """An empty host ``State``, the first argument of every problem protocol call.

    Returns:
        ``common.State()`` -- no keys today, since no shipped problem registers
        any (``setup`` exists for protocol uniformity and for user problems that
        add normalization statistics).

    Notes:
        Module-scoped because ``State`` is immutable: sharing one instance
        across a module cannot leak state into a test.
    """
    return common.State()


@pytest.fixture(scope="module")
def schedule_type():
    """The fake batch-first execution schedule used in place of ``Transformed``.

    Returns:
        :data:`FakeSchedule`, a two-field namedtuple. Call it with the traced
        arrays (``schedule_type(*arrays)``) inside a kernel, or with numpy
        arrays on the host.

    Notes:
        A namedtuple (not a dict) so that a problem implementation walking
        ``_fields`` or tuple positions sees the same *shape* of object the real
        schedule is; the field names are fake on purpose (see the module
        docstring).
    """
    return FakeSchedule


@pytest.fixture(scope="module")
def fake_schedule(schedule_type):
    """Build a fake execution schedule whose leaves carry ``pop_size`` on axis 0.

    Returns:
        ``make(pop_size=4, *, scores_pop_size=None, feature_width=3)`` ->
        :data:`FakeSchedule` of numpy leaves: ``weights`` of shape ``(pop_size,
        feature_width) float32`` and ``scores`` of shape ``(scores_pop_size,)
        int32``, the latter defaulting to ``pop_size``.

    Notes:
        ``scores_pop_size`` is the hook for the "leaves disagree about the
        population axis" test: :func:`~tensorneat.problem.base.static_batch_size`
        must refuse such a schedule with a ``TensorNEATError`` instead of
        picking one of the two sizes. Degenerate schedules (rank-0 leaves, no
        leaves) are built by hand from :func:`schedule_type` in the test that
        needs them, where the shape under test is visible in the test body.
    """

    def make(pop_size=4, *, scores_pop_size=None, feature_width=3):
        import numpy as np

        scores_size = pop_size if scores_pop_size is None else scores_pop_size
        return schedule_type(
            np.zeros((pop_size, feature_width), dtype=np.float32),
            np.zeros((scores_size,), dtype=np.int32),
        )

    return make


@pytest.fixture(scope="module")
def run_evaluate(runner, schedule_type):
    """Evaluate a whole population through ``problem.evaluate`` in one graph.

    Returns:
        ``run(problem_object, state, act_fn, schedule, key_seed=0) -> (P,)``
        host numpy fitness. ``problem_object`` is any ``BaseProblem`` instance,
        ``state`` the host ``State`` from :func:`state` (closed over, never
        passed as data), ``act_fn`` a plain Python closure baked into the trace,
        ``schedule`` a :data:`FakeSchedule` whose leaves fix ``P``, and
        ``key_seed`` the integer a fresh ``evoxc.random.key`` is rebuilt from
        inside the trace.

    Notes:
        Everything a problem reads from the schedule is
        ``static_batch_size(transformed)``, so the fake schedule is enough to
        run the real kernel; the assertion is always on the returned ``(P,)``
        fitness, which is the only thing a problem's contract promises.
        ``act_fn`` must be pure ``enp``: it is inlined into the same graph.
    """

    def run(problem_object, state, act_fn, schedule, key_seed=0):
        import evoxc

        def kernel(*arrays):
            return problem_object.evaluate(
                state,
                evoxc.random.key(key_seed),
                act_fn,
                schedule_type(*arrays),
            )

        return runner.run_defn(kernel, *tuple(schedule))

    return run


@pytest.fixture(scope="module")
def run_in_graph(runner):
    """Trace an arbitrary closure over the given arrays and return host numpy.

    Returns:
        ``run(fn, *arrays)``: ``fn`` takes the arrays positionally and must
        return graph values; the result comes back as numpy
        (``runner.to_numpy``), namedtuples included.

    Notes:
        This is the escape hatch for the *helper* kernels that are not the whole
        ``evaluate`` protocol: ``finalize_fitness(fitness)``,
        ``static_batch_size(schedule)`` and CartPole's ``_terminated(obs,
        _constants())`` are all plain in-trace helpers, so a test hands the
        closure in verbatim and asserts on the host side. Statics are closed
        over, never passed: the runner bakes non-array arguments into the trace.
    """

    def run(fn, *arrays):
        return runner.run_defn(fn, *arrays)

    return run
