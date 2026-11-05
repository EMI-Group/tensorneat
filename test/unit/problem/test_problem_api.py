"""Unit tests for the problem protocol itself (``problem/base.py``) and the
per-sample shape declarations of the shipped problems.

What this module proves
-----------------------
Every problem is *host-side configuration plus plain kernel helpers*: there is
no traced state of its own, and the protocol is small enough to check clause by
clause.

1. ``input_shape``/``output_shape`` are **per-sample** shapes (``XOR`` →
   ``(2,)``/``(1,)``, ``CartPole`` → ``(4,)``/``(2,)``), so ``input_shape[-1]``
   is the network's input arity and not the dataset length. The legacy
   dataset-shaped tuples are gone.
2. ``setup(state)`` returns a ``State`` -- the same container when one is
   handed in, a fresh empty one for ``None`` -- and adds no keys: no shipped
   problem registers any today.
3. ``static_batch_size(transformed)`` is the only thing a problem reads from
   the execution schedule: the static leading dimension of its leaves. A
   schedule whose leaves disagree, or one that exposes no leading dimension at
   all (rank-0 leaves, an empty schedule, a leaf that is not a tensor), is a
   ``TensorNEATError``.
4. ``finalize_fitness`` maps ``NaN`` to ``-inf`` and touches nothing else --
   importantly ``+inf`` stays ``+inf``, so the helper is a NaN rule and not a
   general "make it finite" rule.
5. ``show``/``show_details`` are host-side: they are called outside a trace
   (no ``@evoxc.defn`` in sight) and return ``None``; ``show_details``'s default
   is a no-op.
6. The legacy ``jitable`` flag is gone: a problem is always traced, and whether
   a kernel compiles is the backend's business. ``problem.jitable`` must raise
   ``AttributeError`` rather than answer ``True``.

Oracles used here (never the kernel under test)
-----------------------------------------------
* the shape literals in the class docstrings, and ``inputs.shape == (S,) +
  input_shape`` / ``targets.shape == (S,) + output_shape`` -- an identity that
  pins "per sample" against "dataset shaped" without hard-coding ``S``;
* ``algorithm.NEAT``'s own ``num_inputs``/``num_outputs`` (forwarded from a
  genome built by ``support.payloads``), which is the pairing ``Pipeline``
  itself asserts;
* float values chosen so each entry of the ``finalize_fitness`` mapping is
  identifiable (``-inf``, a finite negative, zero, a finite positive, ``+inf``,
  ``NaN``).

Gotchas this module encodes
---------------------------
* ``finalize_fitness`` and ``static_batch_size`` are module-level helpers of
  ``problem/base.py``, not ``BaseProblem`` methods, so a test reaches them
  through ``problem.base`` (the package re-exports only ``BaseProblem`` and the
  concrete problems).
* ``finalize_fitness`` is in-trace only: it uses ``evoxc.isnan`` /
  ``evoxc.select``, so it is exercised through the shared runner.
  ``static_batch_size`` only walks pytree shapes, so it answers on host numpy
  arrays too -- both routes are checked below, and the malformed schedules are
  rejected identically on either side of the trace boundary.
* ``show`` builds and runs its own graph of one individual, so the test only
  needs a fake schedule of ``P = 1`` and a genome-free ``act_fn``.

Backend is ``"numpy"`` throughout and every graph goes through the shared
runner; the host-side protocol calls (``setup``, ``show``, ``show_details``,
the malformed-schedule checks) need no backend at all. No ``jax``/``torch``, no
network, no GPU.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest


def _constant_predictor(enp, value):
    """An ``act_fn`` whose whole batch predicts ``value``.

    Args:
        enp: The ``evoxc.numpy`` op namespace.
        value: The float the ``(B, 1)`` prediction block is filled with.

    Returns:
        ``act_fn(batch) -> (B, 1) float32``.
    """

    def act_fn(batch):
        return enp.full((batch.shape[0], 1), value)

    return act_fn


def _two_logits(enp):
    """An ``act_fn`` emitting two constant logits (CartPole's output shape).

    Returns:
        ``act_fn(batch) -> (B, 2) float32``, always "push right".
    """

    def act_fn(batch):
        n = batch.shape[0]
        return enp.stack([enp.full((n,), 0.0), enp.full((n,), 1.0)], axis=1)

    return act_fn


def _single_logit(enp):
    """An ``act_fn`` emitting one constant logit (the two continuous RL policies).

    Returns:
        ``act_fn(batch) -> (B, 1) float32``, always a zero torque / zero force.
    """

    def act_fn(batch):
        return enp.full((batch.shape[0], 1), 0.0)

    return act_fn


def _positive_grid_problem(problem):
    """A ``CustomFuncFit`` on ``[0, 1]^2`` with a deterministic four-row dataset.

    Returns:
        ``CustomFuncFit(lambda point: point[0] + 1.5, low_bounds=[0, 0],
        upper_bounds=[2, 2], method="grid", step_size=[1, 1])``; ``I = 2`` and
        ``O = 1`` because the function returns a scalar per point.
    """
    return problem.CustomFuncFit(
        lambda point: point[0] + 1.5,
        low_bounds=[0.0, 0.0],
        upper_bounds=[2.0, 2.0],
        method="grid",
        step_size=[1.0, 1.0],
    )


# --------------------------------------------------------------------------- #
# shapes: per sample, and consistent with the dataset
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "factory, input_shape, output_shape",
    [
        pytest.param(lambda problem: problem.XOR(), (2,), (1,), id="xor"),
        pytest.param(lambda problem: problem.XOR3d(), (3,), (1,), id="xor3d"),
        pytest.param(lambda problem: problem.CartPole(max_step=20), (4,), (2,), id="cartpole"),
        pytest.param(_positive_grid_problem, (2,), (1,), id="custom_func_fit"),
    ],
)
def test_shapes_are_per_sample_and_match_the_dataset(
    problem, factory, input_shape, output_shape
):
    """Invariant: the declared shapes describe one sample, never the dataset.

    Oracle: the docstring literals (``XOR`` → ``(2,)``/``(1,)``, ``XOR3d`` →
    ``(3,)``/``(1,)``, ``CartPole`` → ``(4,)``/``(2,)``, ``CustomFuncFit`` →
    the box dimension and ``func``'s per-point output), plus the identity
    ``inputs.shape == (S,) + input_shape`` and
    ``targets.shape == (S,) + output_shape`` for the func-fit problems. The
    identity is the real guard: it fails for a legacy dataset-shaped
    declaration (``(4, 2)`` for XOR) even though ``S`` is never hard-coded here.
    """
    subject = factory(problem)
    assert tuple(subject.input_shape) == input_shape
    assert tuple(subject.output_shape) == output_shape

    if not hasattr(subject, "inputs"):  # CartPole has no dataset
        assert not hasattr(subject, "targets")
        return

    inputs = np.asarray(subject.inputs)
    targets = np.asarray(subject.targets)
    assert inputs.ndim == len(input_shape) + 1
    assert targets.ndim == len(output_shape) + 1
    assert inputs.shape[1:] == input_shape
    assert targets.shape[1:] == output_shape
    assert inputs.shape[0] == targets.shape[0]  # one target row per sample
    # "per sample, never the dataset": the declaration is strictly narrower.
    assert inputs.shape != input_shape
    assert targets.shape != output_shape


def test_the_input_and_output_arity_match_the_algorithm(problem, algorithm, payloads):
    """Invariant: ``problem.input_shape[-1]`` is the genome's input width, and likewise for outputs.

    Oracle: ``algorithm.NEAT(genome, pop_size).num_inputs`` / ``num_outputs``,
    which are forwarded from the genome -- the same values ``Pipeline`` checks
    the problem against. A genome built by ``support.payloads`` is used for
    ``XOR`` (2 in, 1 out) and ``CartPole`` (4 in, 2 out), so both directions of
    the pairing are covered.

    Notes:
        This is a cross-subtree check on purpose: the invariant belongs to the
        *pair* of configuration objects, and a problem whose shape declaration
        drifted by one (the classic ``input_shape`` dataset-shaped mistake)
        passes every single-problem test and fails here.
    """
    cases = [
        (
            problem.XOR(),
            payloads.default_genome(num_inputs=2, num_outputs=1),
            (2, 1),
        ),
        (
            problem.CartPole(max_step=20),
            payloads.default_genome(num_inputs=4, num_outputs=2),
            (4, 2),
        ),
    ]
    for subject, genome, arity in cases:
        neat = algorithm.NEAT(genome, 10)  # 10 >= species_size * min_species_size
        assert (neat.num_inputs, neat.num_outputs) == arity
        assert subject.input_shape[-1] == neat.num_inputs
        assert subject.output_shape[-1] == neat.num_outputs


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda problem: problem.XOR(), id="xor"),
        pytest.param(lambda problem: problem.CartPole(max_step=20), id="cartpole"),
    ],
)
def test_setup_returns_a_state_and_registers_no_keys(common, problem, factory):
    """Invariant: ``setup(state)`` returns the *same* ``State`` and adds no keys.

    Oracle: ``State`` is immutable and no shipped problem registers anything, so
    the returned container must be a ``common.State`` that is the very object
    handed in, with its keys -- and their values -- unchanged: not a superset,
    not a rebuilt copy that dropped an entry. ``setup`` exists for protocol
    uniformity (and for user problems that add normalization statistics), which
    is why the assertion is "no new keys" rather than "a new object".

    Notes:
        The incoming state deliberately registers three unrelated keys
        (``backend``, ``pop``, ``seed``) with non-default values, so the
        comparison is against a **non-empty** key set: an empty-state check
        (``[] == []``) cannot notice a ``setup`` that dropped or replaced keys,
        which is exactly the regression this test has to be able to fail on.
        ``setup(None)`` -- the documented default -- must start from a genuinely
        empty ``State``, so its key list is empty rather than "the same as the
        input's".
    """
    incoming = common.State(backend="numpy", pop=3, seed=11)
    original_keys = set(incoming.registered_keys())
    assert original_keys == {"backend", "pop", "seed"}
    subject = factory(problem)

    returned = subject.setup(incoming)

    assert isinstance(returned, common.State)
    assert returned is incoming
    assert set(returned.registered_keys()) == original_keys
    assert (returned.backend, returned.pop, returned.seed) == ("numpy", 3, 11)
    # The argument is an immutable container: reading it back is unchanged.
    assert (incoming.backend, incoming.pop, incoming.seed) == ("numpy", 3, 11)

    fresh = subject.setup(None)  # documented default: start from an empty State
    assert isinstance(fresh, common.State)
    assert list(fresh.registered_keys()) == []


# --------------------------------------------------------------------------- #
# static_batch_size
# --------------------------------------------------------------------------- #


def test_static_batch_size_reads_the_static_leading_dimension(
    problem, enp, run_in_graph, fake_schedule
):
    """Invariant: the population size is the static leading dim of every leaf.

    Oracle: the leaves themselves. A fake schedule of ``P`` individuals is
    passed through the runner and the kernel returns the size it read back, so
    the value is observed *inside* a trace (where ``shape[0]`` is a plain
    ``int``), and the helper's host-side answer is required to agree.

    Notes:
        ``P = 1`` and ``P = 4`` are both checked, and the second field of the
        schedule carries the same leading dimension as the first, so an
        implementation that only reads the first leaf cannot pass. The feature
        width is deliberately different from ``P`` (``7``), so reading the
        *second* axis instead of the leading one is caught as well.
    """
    schedule_type = type(fake_schedule(1))

    def read_size(*arrays):
        """Return the size the kernel reads, as a rank-0 graph value."""
        size = problem.base.static_batch_size(schedule_type(*arrays))
        return enp.full((), float(size))

    for pop_size in (1, 4):
        schedule = fake_schedule(pop_size, feature_width=7)
        observed = run_in_graph(read_size, *tuple(schedule))
        assert int(observed) == pop_size

    # The host-side answer must agree with the traced one (same helper, no trace).
    assert problem.base.static_batch_size(fake_schedule(4, feature_width=7)) == 4


def test_static_batch_size_rejects_leaves_that_disagree(
    problem, common, enp, run_in_graph, schedule_type
):
    """Invariant: leaves with different leading dimensions raise ``TensorNEATError``.

    Oracle: the documented ``Raises:`` clause -- "if no leaf exposes a static
    leading dimension, or if the leaves disagree about it". A schedule of
    ``(3, 2)`` and ``(4,)`` leaves must raise instead of silently picking one of
    the two sizes, because a wrong ``P`` would corrupt the ``(P, S, O)`` reshape
    somewhere far from the mistake. Both routes are checked: on the host and
    inside a trace, where the same ``TensorNEATError`` escapes the build.
    """
    disagreeing = schedule_type(
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((4,), dtype=np.int32),
    )

    with pytest.raises(common.TensorNEATError, match="disagree"):
        problem.base.static_batch_size(disagreeing)

    def read_size(*arrays):
        """Ask the kernel for the population size of the traced schedule."""
        return enp.full(
            (), float(problem.base.static_batch_size(schedule_type(*arrays)))
        )

    with pytest.raises(common.TensorNEATError, match="disagree"):
        run_in_graph(read_size, *tuple(disagreeing))


def test_static_batch_size_rejects_a_schedule_without_a_leading_dimension(
    problem, common, enp, run_in_graph, schedule_type
):
    """Invariant: a schedule exposing no static leading dimension raises ``TensorNEATError``.

    Oracle: same ``Raises:`` clause from the other side -- "no leaf exposes a
    static leading dimension". A population cannot be recovered from a scalar
    schedule, so the failure must be a configuration error naming the problem's
    only input, never a silent ``P = 1``. Every shape that exposes no leading
    dimension is covered: rank-0 leaves (in either position), an empty
    schedule, and a leaf that is not a tensor at all. The runner widens a rank-0
    numpy input to ``(1,)``, so these degenerate schedules are built host-side,
    where the helper's contract is exercised without a trace.
    """
    size = problem.base.static_batch_size

    with pytest.raises(common.TensorNEATError, match="rank-0"):
        size(schedule_type(np.zeros((), np.float32), np.zeros((), np.float32)))

    # ... and a rank-0 leaf is rejected wherever it sits in the pytree.
    with pytest.raises(common.TensorNEATError, match="rank-0"):
        size(schedule_type(np.zeros((4, 3), np.float32), np.zeros((), np.float32)))

    with pytest.raises(common.TensorNEATError, match="no leaves"):
        size(())

    with pytest.raises(common.TensorNEATError, match="has no ``shape``"):
        size(schedule_type(4, 4))


# --------------------------------------------------------------------------- #
# finalize_fitness
# --------------------------------------------------------------------------- #


def test_finalize_fitness_maps_only_nan_to_negative_infinity(problem, run_in_graph):
    """Invariant: ``NaN -> -inf``; every other value passes through unchanged.

    Oracle: closed form. The input ``[-inf, -1.5, 0.0, 3.0, +inf, nan]`` must map
    to ``[-inf, -1.5, 0.0, 3.0, +inf, -inf]``: ``-inf`` stays ``-inf`` (already
    the worst), ``+inf`` stays ``+inf`` (a legitimate perfect score must not be
    demoted to ``-inf``), and only ``NaN`` moves. The asserted dtype is float32,
    because the algorithm's selection step reads this batch.

    Notes:
        ``+inf`` is the discriminating case: ``evoxc`` has no ``isfinite``, so a
        "make it finite" implementation built from ``f - f`` would be NaN for
        both infinities and turn a perfect individual into the worst one.
    """
    fitness = np.array([-np.inf, -1.5, 0.0, 3.0, np.inf, np.nan], dtype=np.float32)
    finalized = run_in_graph(problem.base.finalize_fitness, fitness)

    assert np.asarray(finalized).dtype == np.float32
    expected = np.array([-np.inf, -1.5, 0.0, 3.0, np.inf, -np.inf], dtype=np.float32)
    assert np.array_equal(np.asarray(finalized), expected)
    assert not np.isnan(np.asarray(finalized)).any()


# --------------------------------------------------------------------------- #
# host-side reporting
# --------------------------------------------------------------------------- #


def test_show_is_host_side_and_returns_nothing(
    state, problem, enp, fake_schedule, evoxc_key
):
    """Invariant: ``show`` runs outside a trace, prints, and returns ``None``.

    Oracle: the protocol docstring -- "host-side only: never called from a
    trace; it evaluates a batch of one and prints a readable summary (for func
    fit one row per sample plus the loss)". The test calls it directly with a
    ``P = 1`` schedule (no runner, no graph of its own), captures stdout, and
    asserts the return value is ``None`` and that at least one row per sample
    plus a summary line was printed -- so an implementation that computed
    silently would be caught. The func-fit half is closed-form: a constant
    ``0.5`` predictor on XOR prints four ``input: / target: / predict:`` rows
    and ``loss: 0.25``. The CartPole half asserts the other documented report
    (the survival length and the return) on a deterministic rollout
    (``reset_scale = 0.0``, ``max_step = 10``, always-push-right), whose
    ``9/10`` survival is the analytic value.

    Notes:
        The call is made outside any ``@evoxc.defn``; whatever graph ``show``
        needs internally is its own business (and is the ``src``'s only place to
        do so).
    """
    xor = problem.XOR()
    schedule = fake_schedule(pop_size=1)

    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        returned = xor.show(
            state, evoxc_key, _constant_predictor(enp, 0.5), schedule
        )
    assert returned is None

    lines = report.getvalue().splitlines()
    rows = [line for line in lines if line.startswith("input:")]
    assert len(rows) == int(np.asarray(xor.inputs).shape[0])
    assert all("target:" in line and "predict:" in line for line in rows)
    loss_lines = [line for line in lines if line.startswith("loss:")]
    assert len(loss_lines) == 1
    assert float(loss_lines[0].split(":", 1)[1]) == pytest.approx(0.25)

    cartpole = problem.CartPole(max_step=10, reset_scale=0.0)
    rollout = io.StringIO()
    with contextlib.redirect_stdout(rollout):
        returned = cartpole.show(state, evoxc_key, _two_logits(enp), schedule)
    assert returned is None
    assert "survived 9/10 steps" in rollout.getvalue()


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda problem: problem.XOR(), id="xor"),
        pytest.param(lambda problem: problem.CartPole(max_step=20), id="cartpole"),
    ],
)
def test_show_details_is_a_no_op_hook(state, problem, fake_schedule, evoxc_key, capsys, factory):
    """Invariant: the default ``show_details`` prints nothing and returns ``None``.

    Oracle: the docstring -- "the default is a no-op; a problem that wants to
    publish running statistics overrides it", and ``Pipeline.auto_run`` calls it
    once per generation when ``show_problem_details=True``. A silent, non-raising
    no-op is exactly what is asserted (empty stdout, ``None`` return), so a
    subclass that started printing to stderr or returned a value would fail on
    the shared base.

    Notes:
        ``act_fn`` is ``None`` on purpose: a no-op hook must not touch it, so an
        implementation that started evaluating the population would raise here.
    """
    subject = factory(problem)
    returned = subject.show_details(state, evoxc_key, None, fake_schedule())

    assert returned is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --------------------------------------------------------------------------- #
# the reporting backend: build_options and show's "no silent fallback"
# --------------------------------------------------------------------------- #


def test_build_options_reads_the_registered_selection(problem, common):
    """Invariant: ``build_options`` forwards ``backend``/``device``/``compile_options`` off ``state``.

    Oracle: the helper's own documented mapping -- ``backend`` from the state or
    ``"numpy"`` when none is registered, ``device`` only when registered and not
    ``None``, and ``compile_options`` expanded into ``**options`` (not nested
    under its own key, because ``evoxc.build`` reads them as keyword arguments).

    Notes:
        Every branch is asserted on a hand-built ``State``, so the fallback
        (``{"backend": "numpy"}``) is distinguished from a forwarded selection:
        the two differ only in the value, and a helper that always answered
        ``"numpy"`` would pass only the first case.
    """
    build_options = problem.base.build_options

    assert build_options(common.State()) == {"backend": "numpy"}
    assert build_options(common.State(backend="iree")) == {"backend": "iree"}
    # A registered ``device=None`` is dropped rather than forwarded.
    assert build_options(common.State(backend="numpy", device=None)) == {
        "backend": "numpy"
    }
    assert build_options(common.State(device="cpu")) == {
        "backend": "numpy",
        "device": "cpu",
    }
    # ``compile_options`` is expanded, never nested.
    assert build_options(common.State(compile_options={"foo": 1})) == {
        "backend": "numpy",
        "foo": 1,
    }
    assert build_options(
        common.State(backend="iree", device="cpu", compile_options={"bar": 2})
    ) == {"backend": "iree", "device": "cpu", "bar": 2}


@pytest.mark.parametrize(
    "factory, policy",
    [
        pytest.param(
            lambda problem: problem.XOR(),
            "constant",
            id="xor",
        ),
        pytest.param(
            lambda problem: problem.CartPole(max_step=10),
            "two_logits",
            id="cartpole",
        ),
        pytest.param(
            lambda problem: problem.Pendulum(max_step=10),
            "single_logit",
            id="pendulum",
        ),
        pytest.param(
            lambda problem: problem.MountainCarContinuous(max_step=10),
            "single_logit",
            id="mountaincar",
        ),
    ],
)
def test_show_honours_the_registered_backend_instead_of_falling_back_to_numpy(
    problem, common, enp, evoxc, evoxc_key, fake_schedule, capsys, factory, policy
):
    """Invariant: a bogus *registered* backend makes ``show`` raise, not silently use numpy.

    Oracle: the documented rule on every ``show`` -- "the report runs on the
    backend the run was configured with ... never a silent numpy fallback behind
    a compiled run", and ``evoxc``'s registry, which raises ``BackendError``
    ("unknown backend '<name>'; registered backends: ...") for a name it does not
    know. The name is asserted verbatim in the message, so the failure is
    attributed to the registered selection and not to some later stage.

    Notes:
        An *unregistered* name is the strongest available probe on a box without
        every compiled backend: it exercises the same ``backend`` keyword that
        ``build_options`` forwards, and a ``show`` that ignored the state (the
        numpy fallback) would print a report and pass instead of raising. Nothing
        is printed before the build, so stdout must be empty afterwards.
    """
    subject = factory(problem)
    state = common.State(backend="definitely-not-a-registered-backend")
    assert problem.base.build_options(state) == {
        "backend": "definitely-not-a-registered-backend"
    }

    if policy == "constant":
        act_fn = _constant_predictor(enp, 0.5)
    elif policy == "two_logits":
        act_fn = _two_logits(enp)
    else:
        act_fn = _single_logit(enp)

    with pytest.raises(evoxc.BackendError, match="definitely-not-a-registered-backend"):
        subject.show(state, evoxc_key, act_fn, fake_schedule(pop_size=1))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_the_base_problem_skeleton_raises_not_implemented(
    problem, evoxc_key, fake_schedule
):
    """Invariant: the abstract protocol raises ``NotImplementedError``; ``show_details`` does not.

    Oracle: the ``Raises:`` clauses of the skeleton -- ``evaluate``, ``input_shape``,
    ``output_shape`` and ``show`` all raise ``NotImplementedError`` "always, in the
    skeleton", while ``show_details`` is the documented no-op hook ("the default
    is a no-op") that ``Pipeline.auto_run`` may call on any problem.

    Notes:
        A missing override must be loud: ``BaseProblem`` is what a user problem
        subclasses, and a silent default (a zero fitness or an empty shape) would
        be inherited by mistake. The no-op half is asserted in the same test so a
        skeleton that started raising from ``show_details`` is caught too.
    """
    base = problem.BaseProblem()
    schedule = fake_schedule(pop_size=1)

    with pytest.raises(NotImplementedError):
        base.evaluate(None, evoxc_key, None, schedule)
    with pytest.raises(NotImplementedError):
        base.input_shape
    with pytest.raises(NotImplementedError):
        base.output_shape
    with pytest.raises(NotImplementedError):
        base.show(None, evoxc_key, None, schedule)

    # ``None`` for the state/key/act_fn is legal for the no-op hook: it must not
    # touch any of them.
    assert base.show_details(None, evoxc_key, None, schedule) is None


# --------------------------------------------------------------------------- #
# the legacy flag
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda problem: problem.XOR(), id="xor"),
        pytest.param(lambda problem: problem.XOR3d(), id="xor3d"),
        pytest.param(_positive_grid_problem, id="custom_func_fit"),
        pytest.param(lambda problem: problem.CartPole(max_step=20), id="cartpole"),
    ],
)
def test_the_legacy_jitable_flag_is_gone(problem, factory):
    """Invariant: ``problem.jitable`` raises ``AttributeError`` on every shipped problem.

    Oracle: the documented removal -- "the legacy ``jitable`` flag is
    intentionally absent: a problem is always traced, and whether a kernel
    compiles is the backend's business". Asserted twice: ``hasattr`` is false,
    and reading the attribute raises ``AttributeError`` (``hasattr`` alone would
    also swallow a ``KeyError``-carrying ``__getattr__``).

    Notes:
        The legacy ``Pipeline.__init__`` asserted ``problem.jitable``; that
        assertion has no counterpart in the rewrite, so a stale attribute would
        be dead weight that could silently re-enable a host-side fallback.
    """
    subject = factory(problem)

    assert not hasattr(subject, "jitable")
    assert "jitable" not in dir(type(subject))
    assert "jitable" not in dir(problem.BaseProblem)
    with pytest.raises(AttributeError):
        subject.jitable
