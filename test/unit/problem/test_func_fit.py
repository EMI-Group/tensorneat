"""Unit tests for the function-fitting algebra of ``problem/func_fit/``.

What this module proves
-----------------------
``FuncFit`` owns the only non-trivial part of the family: how a small fixed
dataset becomes **one** batched evaluation of the whole population, and how the
resulting error tensor becomes one scalar per individual. The documented
pipeline (``src/tensorneat/problem/func_fit/func_fit.py``) is

1. ``P = static_batch_size(transformed)`` -- the only thing read from the
   schedule, so no genome is needed to test it;
2. ``tiled = reshape(broadcast_to(inputs, (P, S, I)), (P * S, I))``, so row
   ``p * S + s`` is sample ``s`` of individual ``p`` (individual-major);
3. **one** ``act_fn`` call over that tiled batch, reshaped back to ``(P, S, O)``;
4. ``err = pred - expand_dims(targets, 0)`` and a reduction over the sample
   **and** output axes, negated for maximization, ``NaN -> -inf``.

Each test below fixes one clause of that list with a genome-free ``act_fn``
(a plain Python closure over ``enp`` ops, inlined into the graph by the shared
runner), so nothing here depends on the genome subtree at all.

``CustomFuncFit``'s own dataset generation and box validation live in
``test_custom.py`` (one module per source module).

Oracles used here (never the kernel under test)
-----------------------------------------------
* closed forms -- a constant ``0.5`` predictor on XOR has ``mse = 0.25``,
  ``rmse = mae = 0.5``; a zero predictor on a strictly positive dataset has
  ``mape = 1``; the two-output nonzero case has ``mse = 91/8``,
  ``mae = 22/8`` and ``mape = (467/60)/8``;
* :func:`support.oracle.func_fit_loss` -- the independent numpy reduction over
  both axes, written from the docstring, used wherever a closed form would be
  tedious (the row-index tiling test, the four-method parametrization);
* the dataset literals themselves (``XOR`` is four rows, ``XOR3d`` is eight),
  which are data in the API, not an implementation detail;
* the byte arithmetic of the baked constants (``(S, I)`` plus ``(S, O)``
  float32), which is what ``MAX_DATASET_NBYTES`` is compared against.

Gotchas this module encodes
---------------------------
* ``mape`` divides by the target, so a **zero target** with a non-zero
  prediction yields ``+inf`` (legacy behaviour, kept) and a zero *prediction* on
  a zero target yields ``0/0 = NaN``. Both reach ``-inf`` fitness, through
  different arithmetic, and **both** are asserted: the guard-free NaN is what a
  "fix" that replaced ``0/0`` with ``0`` would silently change.
* The reduction denominators matter: with ``O = 1`` every plausible reduction
  (over both axes, over samples only, mean-of-means) coincides, which is why the
  axis tests use ``O = 2``, where a factor-``O`` mistake is visible.
* The dataset is a compile-time constant bounded by
  ``FuncFit.MAX_DATASET_NBYTES`` (1 MiB), checked at construction, in ``setup``
  and again at bake time -- a direct ``evaluate`` is guarded too.
* ``S`` stays small: every case here uses ``S <= 8`` and populations of at most
  four individuals.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# genome-free predictors: plain ``enp`` closures, inlined into the trace
# --------------------------------------------------------------------------- #


def _constant_predictor(enp, value):
    """An ``act_fn`` whose whole batch predicts ``value``.

    Args:
        enp: The ``evoxc.numpy`` op namespace (a traced tensor has no
            ``__getitem__`` and there is no eager mode, so the constant is a
            graph constructor).
        value: The float the ``(B, 1)`` prediction block is filled with; ``nan``
            is the poisoning case.

    Returns:
        ``act_fn(batch) -> (B, 1) float32``, independent of ``batch``'s values.
    """

    def act_fn(batch):
        return enp.full((batch.shape[0], 1), value)

    return act_fn


def _row_index_predictor(enp, evoxc):
    """An ``act_fn`` that predicts the row's *index* in the tiled batch.

    Args:
        enp: The ``evoxc.numpy`` op namespace.
        evoxc: The ``evoxc`` module (``enp.astype`` needs ``evoxc.float32``).

    Returns:
        ``act_fn(batch) -> (B, 1) float32`` with entry ``i`` equal to ``i``.

    Notes:
        The index *is* the tiling: under the documented individual-major layout
        row ``i = p * S + s`` belongs to individual ``p`` and sample ``s``, so
        the per-individual loss is a closed form that a sample-major layout
        cannot reproduce (see :func:`test_the_tiled_batch_is_individual_major`).
    """

    def act_fn(batch):
        n = batch.shape[0]
        return enp.reshape(enp.astype(enp.arange(n), evoxc.float32), (n, 1))

    return act_fn


def _nan_on_first_individual(enp, evoxc, first_rows):
    """An ``act_fn`` that poisons exactly one individual with ``NaN``.

    Args:
        enp: The ``evoxc.numpy`` op namespace.
        evoxc: The ``evoxc`` module (``enp.astype`` needs ``evoxc.float32``).
        first_rows: How many leading rows of the tiled batch belong to the
            poisoned individual, i.e. the dataset size ``S``.

    Returns:
        ``act_fn(batch) -> (B, 1) float32`` that is ``NaN`` on rows
        ``[0, first_rows)`` and ``0`` everywhere else.

    Notes:
        A *mixed* population is the case that proves the ``NaN -> -inf`` mapping
        is per individual: one poisoned network is demoted, its neighbours keep
        their finite fitness. The mask is built from the row index because a
        traced tensor has no ``__getitem__``.
    """

    def act_fn(batch):
        n = batch.shape[0]
        row = enp.astype(enp.arange(n), evoxc.float32)
        poisoned = enp.where(
            enp.less(row, float(first_rows)),
            enp.full((n,), float("nan")),
            enp.full((n,), 0.0),
        )
        return enp.reshape(poisoned, (n, 1))

    return act_fn


def _two_output_constant(enp, left, right):
    """An ``act_fn`` emitting two constant logits -- the ``(B, 2)`` output case.

    Args:
        enp: The ``evoxc.numpy`` op namespace.
        left: Value of output column 0.
        right: Value of output column 1.

    Returns:
        ``act_fn(batch) -> (B, 2) float32``.
    """

    def act_fn(batch):
        n = batch.shape[0]
        return enp.stack(
            [enp.full((n,), left), enp.full((n,), right)],
            axis=1,
        )

    return act_fn


def _positive_grid_problem(problem, error_method="mse"):
    """A ``CustomFuncFit`` on the box ``[0, 2]^2`` with strictly positive targets.

    Args:
        problem: The ``tensorneat.problem`` package.
        error_method: The reduction under test; forwarded to the constructor.

    Returns:
        ``CustomFuncFit(lambda point: point[0] + 1.5, low_bounds=[0, 0],
        upper_bounds=[2, 2], method="grid", step_size=[1, 1])``: a deterministic
        ``S = 4`` dataset whose grid points are ``{0, 1}^2`` and whose targets
        are ``[1.5, 1.5, 2.5, 2.5]``.

    Notes:
        Strictly positive targets are what makes ``mape`` finite, so the same
        problem can carry the four-method parametrization without special-casing
        the division by zero that XOR's ``[0, 1, 1, 0]`` targets have.
    """
    return problem.CustomFuncFit(
        lambda point: point[0] + 1.5,
        low_bounds=[0.0, 0.0],
        upper_bounds=[2.0, 2.0],
        method="grid",
        step_size=[1.0, 1.0],
        error_method=error_method,
    )


def _two_output_grid_problem(problem, error_method="mse"):
    """A ``CustomFuncFit`` on ``[0, 2]^2`` whose ``func`` returns ``(O,) = (2,)``.

    Args:
        problem: The ``tensorneat.problem`` package.
        error_method: The reduction under test; forwarded to the constructor.

    Returns:
        A problem with ``S = 4`` grid rows ``{0, 1}^2`` and targets
        ``[[1, 2], [1, 4], [2, 3], [2, 5]]`` -- every entry positive and
        non-zero, so a reduction over the wrong axis cannot coincide with the
        right one by accident.
    """
    return problem.CustomFuncFit(
        lambda point: np.array([point[0] + 1.0, point[0] + 2.0 * (point[1] + 1.0)]),
        low_bounds=[0.0, 0.0],
        upper_bounds=[2.0, 2.0],
        method="grid",
        step_size=[1.0, 1.0],
        error_method=error_method,
    )


#: Closed-form loss of the **zero** predictor on :func:`_positive_grid_problem`'s
#: targets ``[1.5, 1.5, 2.5, 2.5]``: ``mse = mean(t**2) = 17 / 4``,
#: ``rmse = sqrt(mse)``, ``mae = mean|t| = 8 / 4`` and ``mape = mean|t / t| = 1``.
_POSITIVE_GRID_LOSSES = {
    "mse": 4.25,
    "rmse": 4.25**0.5,
    "mae": 2.0,
    "mape": 1.0,
}


# --------------------------------------------------------------------------- #
# the tiling: row p * S + s belongs to individual p, sample s
# --------------------------------------------------------------------------- #


def test_the_tiled_batch_is_individual_major(
    state, problem, oracle, enp, evoxc, run_evaluate, fake_schedule
):
    """Invariant: tiled row ``p * S + s`` is sample ``s`` of individual ``p``.

    Oracle: closed form + :func:`support.oracle.func_fit_loss`. The predictor
    returns its row index, so individual ``p``'s error is
    ``(p * S + s) - target[s]`` and its loss is
    ``mean_s ((p * S + s) - target[s]) ** 2`` -- a value that *depends on the
    layout*. The test also computes the sample-major alternative
    (``s * P + p``) and asserts the observed fitness is the individual-major one
    and **not** the sample-major one, so a transposed tiling cannot pass.
    """
    xor = problem.XOR()
    pop_size = 4
    observed = run_evaluate(
        xor, state, _row_index_predictor(enp, evoxc), fake_schedule(pop_size)
    )

    num_samples = int(np.asarray(xor.inputs).shape[0])
    targets = np.asarray(xor.targets, dtype=np.float64)
    rows = np.arange(pop_size * num_samples, dtype=np.float64)
    # Both candidate layouts, as the ``(P, S, O)`` block ``act_fn`` would see.
    individual_major = rows.reshape(pop_size, num_samples, 1)
    sample_major = rows.reshape(num_samples, pop_size, 1).transpose(1, 0, 2)
    assert not np.allclose(individual_major, sample_major), "the two layouts must differ"

    expected = oracle.func_fit_loss(individual_major, targets, "mse")
    transposed = oracle.func_fit_loss(sample_major, targets, "mse")

    assert observed.shape == (pop_size,)
    np.testing.assert_allclose(observed, -expected, rtol=1e-6)
    # Individual 0 predicts its own rows 0..3, so ``err = [0, 0, 1, 3]`` and
    # ``mse = (0 + 0 + 1 + 9) / 4 = 2.5`` -- the closed form of the first entry.
    assert np.isclose(observed[0], -2.5, rtol=1e-6)
    assert not np.allclose(observed, -transposed, rtol=1e-6)


# --------------------------------------------------------------------------- #
# the reduction: closed forms, the oracle, and the four error methods
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "error_method, fitness",
    [("mse", -0.3125), ("rmse", -(0.3125**0.5)), ("mae", -0.5)],
)
def test_error_method_matches_its_hand_computed_scalar(
    state, problem, enp, run_evaluate, fake_schedule, error_method, fitness
):
    """Invariant: ``mse``/``rmse``/``mae`` equal their hand-computed value on XOR.

    Oracle: closed form. XOR's targets are ``[0, 1, 1, 0]`` and the predictor is
    the constant ``0.25``, so ``err = [0.25, -0.75, -0.75, 0.25]`` and

    - ``mse  = mean(err**2) = 1.25 / 4 = 0.3125``
    - ``rmse = sqrt(mse)    = 0.5590...``
    - ``mae  = mean(|err|)  = 2.0 / 4 = 0.5``

    Fitness is the negation, so the asserted values are ``-0.3125``,
    ``-sqrt(0.3125)`` and ``-0.5``: three *distinct* numbers, which is what lets
    one parametrized test tell the three reductions apart (a constant ``0.5``
    predictor would not, since it makes ``rmse == mae``).
    """
    xor = problem.XOR(error_method=error_method)
    observed = run_evaluate(
        xor, state, _constant_predictor(enp, 0.25), fake_schedule(4)
    )

    assert observed.shape == (4,)
    np.testing.assert_allclose(observed, np.full(4, fitness), rtol=1e-6)

    # The three parametrized values are pairwise distinct, and the kernel really
    # returns distinguishable fitness for them: the *same predictor* on the
    # *same dataset* through another reduction is asserted to differ, so
    # matching one value cannot be a coincidence of a shared float32 rounding,
    # and a reduction aliased onto another one fails here.
    for other_method in {"mse", "rmse", "mae"} - {error_method}:
        other = run_evaluate(
            problem.XOR(error_method=other_method),
            state,
            _constant_predictor(enp, 0.25),
            fake_schedule(4),
        )
        assert not np.allclose(observed, other, rtol=1e-6), (
            error_method,
            other_method,
        )


@pytest.mark.parametrize("error_method", ["mse", "rmse", "mae", "mape"])
def test_every_error_method_matches_the_numpy_oracle(
    state, problem, oracle, enp, run_evaluate, fake_schedule, error_method
):
    """Invariant: each of the four reductions equals ``support.oracle.func_fit_loss``.

    Oracle: ``func_fit_loss`` on the reference prediction block
    (``np.zeros((P, S, O))``, the zero predictor the ``act_fn`` emits) and the
    problem's own ``targets``, for a ``CustomFuncFit`` with strictly positive
    targets (``[1.5, 1.5, 2.5, 2.5]``) so ``mape`` stays finite. The zero
    predictor's four losses are then hand-checkable: ``mse = mean(t**2) = 4.25``,
    ``rmse = 2.0616...``, ``mae = mean(t) = 2.0`` and
    ``mape = mean|(0 - t) / t| = 1``.

    Notes:
        This is the test that would catch a reduction over one axis only or a
        sum instead of a mean: the four methods differ from each other on this
        dataset, and the tolerance is ``rtol=1e-6`` (float32 arithmetic against a
        float64 oracle).
    """
    custom = _positive_grid_problem(problem, error_method=error_method)
    pop_size = 4
    observed = run_evaluate(
        custom, state, _constant_predictor(enp, 0.0), fake_schedule(pop_size)
    )

    targets = np.asarray(custom.targets, dtype=np.float64)
    assert targets.shape == (4, 1)
    zero_prediction = np.zeros((pop_size,) + targets.shape, dtype=np.float64)
    expected = oracle.func_fit_loss(zero_prediction, targets, error_method)

    assert observed.shape == (pop_size,)
    np.testing.assert_allclose(observed, -expected, rtol=1e-6)
    np.testing.assert_allclose(
        observed, np.full(pop_size, -_POSITIVE_GRID_LOSSES[error_method]), rtol=1e-6
    )

    # The four closed forms differ, and the kernel really distinguishes them:
    # every *other* reduction on the same dataset is asserted to move the
    # observed fitness, so an implementation that aliased two methods (e.g.
    # ``rmse`` onto ``mse``) cannot pass by matching a literal table.
    for other_method in set(problem.FuncFit.ERROR_METHODS) - {error_method}:
        other = run_evaluate(
            _positive_grid_problem(problem, error_method=other_method),
            state,
            _constant_predictor(enp, 0.0),
            fake_schedule(pop_size),
        )
        assert not np.allclose(observed, other, rtol=1e-6), (
            error_method,
            other_method,
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mape_with_a_zero_target_is_infinite_and_so_is_the_negated_fitness(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: ``mape`` divides by the target, so a zero target gives ``-inf`` fitness.

    Oracle: closed form on XOR. With the constant ``0.5`` predictor,
    ``err = [0.5, -0.5, -0.5, 0.5]``, so ``|err / target|`` is ``0.5 / 0 = +inf``
    for the two samples whose target is ``0`` and ``0.5`` for the other two; the
    mean over both axes is ``+inf`` and the negated fitness is ``-inf`` for
    **every** individual.

    Notes:
        Legacy behaviour, deliberately kept (``problem/func_fit/CONTEXT.md``).
        Not the same arithmetic as the ``NaN`` case: a *zero prediction* on a
        zero target is ``0 / 0 = NaN``, which reaches ``-inf`` through
        ``finalize_fitness`` instead. Both are asserted, in this test and in
        :func:`test_a_nan_prediction_scores_negative_infinity`.
    """
    pop_size = 4
    observed = run_evaluate(
        problem.XOR(error_method="mape"),
        state,
        _constant_predictor(enp, 0.5),
        fake_schedule(pop_size),
    )

    assert observed.shape == (pop_size,)
    assert np.all(np.isneginf(observed))
    assert not np.any(np.isnan(observed))

    # Same predictor, same dataset, a reduction that does not divide by the
    # target: finite. The ``-inf`` therefore comes from the ``mape`` division and
    # not from a poisoned batch.
    finite = run_evaluate(
        problem.XOR(error_method="mse"),
        state,
        _constant_predictor(enp, 0.5),
        fake_schedule(pop_size),
    )
    assert np.all(np.isfinite(finite))
    np.testing.assert_allclose(finite, np.full(pop_size, -0.25), rtol=1e-6)


@pytest.mark.parametrize(
    "error_method, expected_loss",
    [
        # err = pred - targets on [[1, 2], [1, 4], [2, 3], [2, 5]] with the
        # constant prediction (0.5, -1) is
        # [[-0.5, -3], [-0.5, -5], [-1.5, -4], [-1.5, -6]], so over all S*O = 8
        # entries:  sum(err**2) = 0.25+9+0.25+25+2.25+16+2.25+36 = 91,
        #           sum(|err|)  = 0.5+3+0.5+5+1.5+4+1.5+6 = 22, and
        #           sum(|err/t|) = 1/2+3/2+1/2+5/4+3/4+4/3+3/4+6/5 = 467/60.
        ("mse", 91.0 / 8.0),
        ("rmse", (91.0 / 8.0) ** 0.5),
        ("mae", 22.0 / 8.0),
        ("mape", (467.0 / 60.0) / 8.0),
    ],
)
def test_every_reduction_reduces_over_the_sample_and_the_output_axes(
    state, problem, enp, run_evaluate, fake_schedule, error_method, expected_loss
):
    """Invariant: all four reductions average over ``S`` *and* ``O``.

    Oracle: the closed forms above, derived from the documented reductions on a
    two-output ``CustomFuncFit`` and a *non-zero* predictor. ``O = 2`` is what
    makes the axis question observable -- with one output a mean over the
    samples and a mean over both axes coincide -- and a non-zero predictor keeps
    ``mape`` away from its ``|0 / t| = 1`` fixed point, where every dataset would
    agree.

    Notes:
        The zero predictor is avoided on purpose: at a zero prediction every
        ``mape`` dataset sits at its ``|0 / t| = 1`` fixed point, where the axis
        question is invisible. The factor checks (``O`` and ``S*O`` too large)
        reject the two classic denominator mistakes explicitly, and the ``(P,)``
        shape rejects a reduction that averaged over the sample axis only (that
        would leave a ``(P, O)`` block, which is not a fitness).
    """
    custom = _two_output_grid_problem(problem, error_method=error_method)
    pop_size = 4
    observed = run_evaluate(
        custom, state, _two_output_constant(enp, 0.5, -1.0), fake_schedule(pop_size)
    )

    targets = np.asarray(custom.targets, dtype=np.float64)
    prediction = np.tile(np.array([[0.5, -1.0]], dtype=np.float64), (4, 1))
    errors = prediction - targets
    # The same closed forms, re-derived with plain numpy arithmetic.
    independent = {
        "mse": np.mean(errors**2),
        "rmse": np.sqrt(np.mean(errors**2)),
        "mae": np.mean(np.abs(errors)),
        "mape": np.mean(np.abs(errors / targets)),
    }
    assert np.isclose(independent[error_method], expected_loss, rtol=1e-12)

    assert observed.shape == (pop_size,)
    np.testing.assert_allclose(observed, np.full(pop_size, -expected_loss), rtol=1e-6)
    # A mean over the sample axis only (a factor ``O`` too large) ...
    assert not np.isclose(observed[0], -expected_loss * 2, rtol=1e-6)
    # ... and a sum instead of a mean (a factor ``S * O`` too large) both differ.
    assert not np.isclose(observed[0], -expected_loss * 8, rtol=1e-6)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mape_with_a_zero_prediction_on_a_zero_target_is_nan_and_negative_infinite(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: the ``0/0`` corner of ``mape`` poisons the loss to ``-inf``.

    Oracle: closed form on XOR (targets ``[0, 1, 1, 0]``) with a **zero**
    predictor: the two zero-target rows give ``|0 / 0| = NaN`` while the two
    one-target rows give ``0``, so the mean over both axes is ``NaN`` and
    ``finalize_fitness`` maps it to ``-inf``. That is the second corner of the
    documented ``mape`` behaviour -- the first, a *non-zero* prediction on a zero
    target (``+inf``), is
    :func:`test_mape_with_a_zero_target_is_infinite_and_so_is_the_negated_fitness`.

    Notes:
        The poison really is the division: the same zero predictor on the same
        dataset under ``mse`` is finite (``-0.5``). An implementation that
        guarded the division (mapping ``0/0`` to ``0``) would return ``0.0`` here
        instead of ``-inf``, so the assertion distinguishes a silent guard from
        the documented NaN propagation.
    """
    pop_size = 4
    targets = np.asarray(problem.XOR().targets, dtype=np.float64)
    zero_prediction = np.zeros_like(targets)
    with np.errstate(invalid="ignore"):
        ratio = np.abs(zero_prediction / targets)
    # The documented corner, stated on the host: half the rows are ``0/0``.
    assert np.isnan(ratio).any()

    observed = run_evaluate(
        problem.XOR(error_method="mape"),
        state,
        _constant_predictor(enp, 0.0),
        fake_schedule(pop_size),
    )
    assert observed.shape == (pop_size,)
    assert np.all(np.isneginf(observed))
    assert not np.any(np.isnan(observed))

    finite = run_evaluate(
        problem.XOR(error_method="mse"),
        state,
        _constant_predictor(enp, 0.0),
        fake_schedule(pop_size),
    )
    np.testing.assert_allclose(finite, np.full(pop_size, -0.5), rtol=1e-6)


def test_a_nan_prediction_scores_negative_infinity(
    state, problem, enp, evoxc, run_evaluate, fake_schedule
):
    """Invariant: a NaN prediction makes the fitness ``-inf``, never a survivor.

    Oracle: the documented rule ("a NaN prediction -- an unconnected or dead
    topology -- yields a NaN loss and therefore ``-inf`` fitness"). Every
    individual of a population whose ``act_fn`` emits NaN must come back as
    ``-inf``, which is what keeps a broken topology from winning selection. The
    mapping is per individual: a *mixed* population -- one poisoned individual,
    three clean ones -- must demote only the poisoned one and leave the others at
    their finite ``mse`` fitness of ``-0.5``.
    """
    pop_size = 4
    poisoned = run_evaluate(
        problem.XOR(), state, _constant_predictor(enp, float("nan")), fake_schedule(pop_size)
    )
    assert poisoned.shape == (pop_size,)
    assert np.all(np.isneginf(poisoned))

    num_samples = int(np.asarray(problem.XOR().inputs).shape[0])
    mixed = run_evaluate(
        problem.XOR(),
        state,
        _nan_on_first_individual(enp, evoxc, num_samples),
        fake_schedule(pop_size),
    )
    assert mixed.shape == (pop_size,)
    assert np.all(np.isneginf(mixed[:1]))
    # The clean individuals predict 0 on targets [0, 1, 1, 0]: mse = 2 / 4.
    assert np.all(np.isfinite(mixed[1:]))
    np.testing.assert_allclose(mixed[1:], np.full(pop_size - 1, -0.5), rtol=1e-6)


# --------------------------------------------------------------------------- #
# the datasets
# --------------------------------------------------------------------------- #


def test_xor_dataset_is_the_four_row_literal(problem):
    """Invariant: ``XOR`` is the literal ``S = 4``, ``I = 2`` dataset.

    Oracle: the literal in the class docstring --
    inputs ``[[0, 0], [0, 1], [1, 0], [1, 1]]`` and targets ``[[0], [1], [1],
    [0]]``, both ``float32``, with ``input_shape == (2,)`` and
    ``output_shape == (1,)`` (per sample, not dataset-shaped). The target column
    is the XOR truth table, so the assertion on the values is also the assertion
    that the *rows* are in the documented order.
    """
    xor = problem.XOR()
    inputs = np.asarray(xor.inputs)
    targets = np.asarray(xor.targets)

    assert inputs.shape == (4, 2)
    assert targets.shape == (4, 1)
    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32
    np.testing.assert_array_equal(
        inputs, np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        targets, np.array([[0], [1], [1], [0]], dtype=np.float32)
    )
    # "the target column is XOR" as a computed property of the input rows.
    np.testing.assert_array_equal(
        targets[:, 0], np.not_equal(inputs[:, 0], inputs[:, 1]).astype(np.float32)
    )
    assert xor.input_shape == (2,)
    assert xor.output_shape == (1,)


def test_xor3d_dataset_is_the_eight_row_parity_literal(problem):
    """Invariant: ``XOR3d`` is the literal ``S = 8``, ``I = 3`` parity dataset.

    Oracle: the literal in the class docstring -- all three-bit patterns in
    counting order, targets ``[[0], [1], [1], [0], [1], [0], [0], [1]]`` (the
    3-bit parity, i.e. ``sum(bits) % 2``), ``input_shape == (3,)`` and
    ``output_shape == (1,)``. The input rows are also checked against
    ``np.arange(8)`` written out in base 2, so "counting order" is a computed
    property, not a copied literal.
    """
    xor3d = problem.XOR3d()
    inputs = np.asarray(xor3d.inputs)
    targets = np.asarray(xor3d.targets)

    assert inputs.shape == (8, 3)
    assert targets.shape == (8, 1)
    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32

    # ``np.arange(8)`` in base 2, most significant bit first.
    counting = ((np.arange(8)[:, None] >> np.arange(2, -1, -1)) & 1).astype(np.float32)
    np.testing.assert_array_equal(inputs, counting)
    np.testing.assert_array_equal(inputs @ np.array([4.0, 2.0, 1.0]), np.arange(8.0))

    np.testing.assert_array_equal(
        targets, np.array([[0], [1], [1], [0], [1], [0], [0], [1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(targets[:, 0], (np.sum(inputs, axis=1) % 2).astype(np.float32))
    assert xor3d.input_shape == (3,)
    assert xor3d.output_shape == (1,)


# --------------------------------------------------------------------------- #
# the baked-dataset budget: MAX_DATASET_NBYTES, validate_dataset, setup, bake
# --------------------------------------------------------------------------- #


def test_the_dataset_budget_is_the_two_float32_constants(problem):
    """Invariant: ``dataset_nbytes`` is exactly ``inputs.nbytes + targets.nbytes``.

    Oracle: the documented arithmetic -- "measured on the float32 host arrays
    that ``_dataset_constants`` bakes", never on the population, because the
    dataset is baked untiled. XOR is ``(4, 2)`` float32 inputs plus ``(4, 1)``
    float32 targets, i.e. ``32 + 16 = 48`` bytes, and the class budget is the
    documented 1 MiB (``EVOXC_LARGE_CONSTANT_BYTES``).

    Notes:
        Recomputing the footprint from the two arrays in the test is what pins
        "float32, untiled": a ``float64`` coercion (96 bytes) or a pre-tiled
        constant (scaled by ``P``) would both disagree here.
    """
    xor = problem.XOR()
    inputs = np.asarray(xor.inputs, dtype=np.float32)
    targets = np.asarray(xor.targets, dtype=np.float32)

    assert (inputs.shape, targets.shape) == ((4, 2), (4, 1))
    assert problem.FuncFit.MAX_DATASET_NBYTES == 1 << 20
    assert xor.dataset_nbytes == inputs.nbytes + targets.nbytes == 48
    assert xor.validate_dataset() == 48


def test_validate_dataset_and_setup_reject_a_dataset_over_budget(
    problem, common, state
):
    """Invariant: a footprint above ``MAX_DATASET_NBYTES`` raises from both host entry points.

    Oracle: the documented ``Raises:`` clauses of :meth:`validate_dataset` and
    :meth:`setup`, driven by lowering the **instance** attribute to one byte
    below XOR's 48-byte footprint (:meth:`setup` calls ``validate_dataset``
    first, so a budget check that lived only in one of them would show up as one
    failing half). The exact boundary is probed from both sides: ``48`` is
    accepted and anything smaller is not, which pins ``>`` against ``>=``.

    Notes:
        The budget is a *configuration* error, so it is ``TensorNEATError`` and
        the message names the attribute a user can raise. ``setup`` registers no
        keys, so the accepted call is asserted to hand the very same ``State``
        back, and a broken dataset must not leave a partially-built state behind.
    """
    xor = problem.XOR()
    assert xor.dataset_nbytes == 48

    xor.MAX_DATASET_NBYTES = 48  # the boundary itself is allowed
    assert xor.validate_dataset() == 48
    assert xor.setup(state) is state

    xor.MAX_DATASET_NBYTES = 47
    with pytest.raises(common.TensorNEATError) as excinfo:
        xor.validate_dataset()
    assert "MAX_DATASET_NBYTES" in str(excinfo.value)
    assert "48" in str(excinfo.value)

    with pytest.raises(common.TensorNEATError, match="MAX_DATASET_NBYTES"):
        xor.setup(None)


def test_the_bake_time_check_rejects_an_oversized_dataset(
    state, problem, common, enp, run_evaluate, fake_schedule
):
    """Invariant: a direct ``evaluate`` is guarded too -- the error escapes ``evoxc.build``.

    Oracle: the documented re-check in ``_dataset_constants`` ("the budget is
    enforced here as well as on the host, so an ``evaluate`` called directly ...
    is guarded too; the ``TensorNEATError`` propagates unchanged through
    ``evoxc.build``"). The instance budget is lowered *after* construction, so
    the constructor's own ``validate_dataset`` has already passed and the only
    remaining guard is the bake-time one.

    Notes:
        This is the path the parity harness takes (a direct ``evaluate``), so the
        failure has to be the library's ``TensorNEATError`` and not a bare
        trace-time error surfacing from inside evoxc.
    """
    xor = problem.XOR()
    xor.MAX_DATASET_NBYTES = 1

    with pytest.raises(common.TensorNEATError, match="MAX_DATASET_NBYTES"):
        run_evaluate(xor, state, _constant_predictor(enp, 0.0), fake_schedule(2))


# --------------------------------------------------------------------------- #
# the host-side report
# --------------------------------------------------------------------------- #


def test_show_prints_one_block_per_individual(problem, state, enp, evoxc_key, fake_schedule, capsys):
    """Invariant: ``show`` accepts any static ``P`` and labels each individual's block.

    Oracle: the documented report -- "one row per sample plus the loss ... the
    report prints one block per individual, so any static ``P`` is accepted".
    With ``P = 2`` the output must carry two ``individual: <i>`` headers, ``S``
    ``input:`` rows and one ``loss:`` line under each, and the closed-form loss of
    the constant ``0.5`` predictor (``0.25``) in both blocks -- so a report that
    silently evaluated only the first individual, or reused one loss, fails.

    Notes:
        ``P = 1`` omits the header (asserted by the previous test's absence of
        ``individual:``), so this is the branch a multi-individual schedule
        reaches and the one a user calling ``show`` on ``state.pop`` depends on.
    """
    xor = problem.XOR()
    returned = xor.show(state, evoxc_key, _constant_predictor(enp, 0.5), fake_schedule(2))
    report = capsys.readouterr().out

    assert returned is None
    lines = report.splitlines()
    assert [line for line in lines if line.startswith("individual:")] == [
        "individual: 0",
        "individual: 1",
    ]
    num_samples = int(np.asarray(xor.inputs).shape[0])
    assert len([line for line in lines if line.startswith("input:")]) == 2 * num_samples
    losses = [float(line.split(":", 1)[1]) for line in lines if line.startswith("loss:")]
    assert losses == pytest.approx([0.25, 0.25], rel=1e-6)
    # The rows of individual 1 follow the ``individual: 1`` header, not before it.
    second_block = lines[lines.index("individual: 1"):]
    assert sum(line.startswith("input:") for line in second_block) == num_samples


# --------------------------------------------------------------------------- #
# configuration errors
# --------------------------------------------------------------------------- #


def test_an_unknown_error_method_is_a_configuration_error(problem, common, oracle):
    """Invariant: an unsupported ``error_method`` raises ``TensorNEATError`` at construction.

    Oracle: the class attribute ``ERROR_METHODS == ("mse", "rmse", "mae",
    "mape")`` (also compared with ``support.oracle.FUNC_FIT_ERROR_METHODS``, so
    the test and the oracle cannot drift), and ``TensorNEATError`` for any other
    string -- on ``FuncFit`` subclasses alike (``XOR``, ``XOR3d``,
    ``CustomFuncFit``). The check is at construction time, on the host, before
    any graph is traced.
    """
    assert problem.FuncFit.ERROR_METHODS == ("mse", "rmse", "mae", "mape")
    assert problem.FuncFit.ERROR_METHODS == oracle.FUNC_FIT_ERROR_METHODS

    constructors = {
        "FuncFit": lambda method: problem.FuncFit(error_method=method),
        "XOR": lambda method: problem.XOR(error_method=method),
        "XOR3d": lambda method: problem.XOR3d(error_method=method),
        "CustomFuncFit": lambda method: problem.CustomFuncFit(
            lambda point: point[0],
            low_bounds=[0.0, 0.0],
            upper_bounds=[1.0, 1.0],
            method="grid",
            step_size=[1.0, 1.0],
            error_method=method,
        ),
    }
    for name, construct in constructors.items():
        for method in problem.FuncFit.ERROR_METHODS:
            construct(method)  # every documented method is accepted
        with pytest.raises(common.TensorNEATError) as excinfo:
            construct("bogus")
        assert "bogus" in str(excinfo.value), name
        assert isinstance(excinfo.value, common.TensorNEATError), name

