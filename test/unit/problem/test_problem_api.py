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
2. ``setup(state)`` returns a ``State`` -- a *new* one, since ``State`` is
   immutable -- and adds no keys: no shipped problem registers any today.
3. ``static_batch_size(transformed)`` is the only thing a problem reads from
   the execution schedule: the static leading dimension of its leaves. A
   schedule whose leaves disagree, or one with no leading dimension at all, is
   a ``TensorNEATError``.
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
* Both helpers are in-trace code: ``finalize_fitness`` uses ``evoxc.isnan`` /
  ``evoxc.select``, and ``static_batch_size`` relies on
  ``SymbolicTensor.shape[0]`` being a plain ``int``. They are therefore called
  through the shared runner, never on host numpy arrays where an ``np.*``
  shortcut would accidentally work.
* ``show`` builds and runs its own graph of one individual, so the test only
  needs a fake schedule of ``P = 1`` and a genome-free ``act_fn``.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

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
    pytest.skip(
        "not implemented: input_shape/output_shape are per-sample tuples "
        "(owner: problem/*/*.input_shape/output_shape)"
    )


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
    pytest.skip(
        "not implemented: problem.input_shape[-1] == algorithm.num_inputs "
        "(owner: problem/*/*.input_shape + algorithm/neat/neat.num_inputs)"
    )


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
def test_setup_returns_a_state_and_registers_no_keys(common, problem, state, factory):
    """Invariant: ``setup(state)`` returns a ``State`` holding the same keys.

    Oracle: ``State`` is immutable and no shipped problem registers anything, so
    the returned container must be a ``common.State`` whose
    ``registered_keys()`` equal the input's -- not a superset. ``setup`` exists
    for protocol uniformity (and for user problems that add normalization
    statistics), which is why the assertion is "no new keys" rather than
    "a new object".
    """
    pytest.skip(
        "not implemented: setup(state) returns a State with the same keys "
        "(owner: common/stateful_class.StatefulBaseClass.setup)"
    )


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
    ``int``) rather than on host numpy arrays where any ``len()`` would do.

    Notes:
        ``P = 1`` and ``P = 4`` are both checked, and the second field of the
        schedule carries the same leading dimension as the first, so an
        implementation that only reads the first leaf cannot pass.
    """
    pytest.skip(
        "not implemented: static_batch_size reads the static leading dim of the leaves "
        "(owner: problem/base.static_batch_size)"
    )


def test_static_batch_size_rejects_leaves_that_disagree(
    problem, common, enp, run_in_graph, schedule_type
):
    """Invariant: leaves with different leading dimensions raise ``TensorNEATError``.

    Oracle: the documented ``Raises:`` clause -- "if no leaf exposes a static
    leading dimension, or if the leaves disagree about it". A schedule of
    ``(3, 2)`` and ``(4,)`` leaves must raise instead of silently picking one of
    the two sizes, because a wrong ``P`` would corrupt the ``(P, S, O)`` reshape
    somewhere far from the mistake.
    """
    pytest.skip(
        "not implemented: disagreeing leaf sizes raise TensorNEATError "
        "(owner: problem/base.static_batch_size)"
    )


def test_static_batch_size_rejects_a_schedule_without_a_leading_dimension(
    problem, common, enp, run_in_graph, schedule_type
):
    """Invariant: a schedule made of rank-0 leaves raises ``TensorNEATError``.

    Oracle: same ``Raises:`` clause from the other side -- "no leaf exposes a
    static leading dimension". A population cannot be recovered from a scalar
    schedule, so the failure must be a configuration error naming the problem's
    only input, never a silent ``P = 1``.
    """
    pytest.skip(
        "not implemented: a schedule with no leading dimension raises TensorNEATError "
        "(owner: problem/base.static_batch_size)"
    )


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
    pytest.skip(
        "not implemented: finalize_fitness maps NaN to -inf and leaves +inf alone "
        "(owner: problem/base.finalize_fitness)"
    )


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
    silently would be caught.

    Notes:
        The call is made outside any ``@evoxc.defn``; whatever graph ``show``
        needs internally is its own business (and is the ``src``'s only place to
        do so).
    """
    pytest.skip(
        "not implemented: show is host-side, prints a report and returns None "
        "(owner: problem/*/*.show)"
    )


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
    """
    pytest.skip(
        "not implemented: show_details is a host-side no-op returning None "
        "(owner: problem/base.BaseProblem.show_details)"
    )


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
    pytest.skip(
        "not implemented: the jitable flag is gone (AttributeError) "
        "(owner: problem/base.BaseProblem)"
    )
