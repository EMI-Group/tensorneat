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

Oracles used here (never the kernel under test)
-----------------------------------------------
* closed forms -- a constant ``0.5`` predictor on XOR has ``mse = 0.25``,
  ``rmse = mae = 0.5``; a zero predictor on a strictly positive dataset has
  ``mape = 1``;
* :func:`support.oracle.func_fit_loss` -- the independent numpy reduction over
  both axes, written from the docstring, used wherever a closed form would be
  tedious (the row-index tiling test, the four-method parametrization);
* the dataset literals themselves (``XOR`` is four rows, ``XOR3d`` is eight),
  which are data in the API, not an implementation detail.

Gotchas this module encodes
---------------------------
* ``mape`` divides by the target, so a **zero target** with a non-zero
  prediction yields ``+inf`` (legacy behaviour, kept). A zero *prediction* on a
  zero target yields ``0/0 = NaN`` instead -- both reach ``-inf`` fitness, but
  through different arithmetic, so the test states which one it asserts.
* The reduction denominators matter: with ``O = 1`` every plausible reduction
  (over both axes, over samples only, mean-of-means) coincides, which is why the
  axis test uses ``O = 2``, where a factor-``O`` mistake is visible.
* ``CustomFuncFit`` samples from ``np.random.default_rng(seed)`` on the host, so
  a dataset is reproducible from the seed alone and the global ``np.random``
  state is never touched (the legacy global-seed coupling is gone).
* The dataset is a compile-time constant, so ``S`` stays small: every case here
  uses ``S <= 8`` and populations of at most four individuals.

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


def _two_output_grid_problem(problem):
    """A ``CustomFuncFit`` on ``[0, 2]^2`` whose ``func`` returns ``(O,) = (2,)``.

    Args:
        problem: The ``tensorneat.problem`` package.

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

    # The three parametrized values are pairwise distinct, so matching one is
    # not a coincidence of a shared float32 rounding.
    others = {"mse": -0.3125, "rmse": -(0.3125**0.5), "mae": -0.5}
    for other_method, other_fitness in others.items():
        if other_method != error_method:
            assert not np.isclose(fitness, other_fitness), (error_method, other_method)


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

    # The four closed forms differ, so this dataset really does distinguish them.
    for other_method, other_loss in _POSITIVE_GRID_LOSSES.items():
        if other_method != error_method:
            assert not np.isclose(
                _POSITIVE_GRID_LOSSES[error_method], other_loss, rtol=1e-3
            ), (error_method, other_method)


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


def test_the_loss_reduces_over_the_sample_and_the_output_axes(
    state, problem, oracle, enp, run_evaluate, fake_schedule
):
    """Invariant: the per-individual loss is the mean over ``S`` *and* ``O`` elements.

    Oracle: closed form on a two-output ``CustomFuncFit`` (``O = 2``) with the
    zero predictor: ``mse = mean(targets ** 2)`` over all ``S * O`` entries,
    compared with ``np.mean`` of the whole target array and cross-checked against
    ``support.oracle.func_fit_loss``.

    Notes:
        ``O = 2`` is the whole point: with one output, "mean over both axes",
        "mean over the samples" and "mean of the per-sample means" are the same
        number, so only a two-output problem exposes a reduction that misses an
        axis (a factor ``O``) or sums instead of averaging (a factor ``S * O``).
        The result is also asserted to be ``(P,)``: a reduction over the sample
        axis alone would leave a ``(P, O)`` block, which is not a fitness.
    """
    custom = _two_output_grid_problem(problem)
    pop_size = 4
    observed = run_evaluate(
        custom, state, _two_output_constant(enp, 0.0, 0.0), fake_schedule(pop_size)
    )

    targets = np.asarray(custom.targets, dtype=np.float64)
    num_samples, num_outputs = targets.shape
    assert (num_samples, num_outputs) == (4, 2)
    assert custom.output_shape == (2,)

    # mean over the S * O = 8 squared entries of [[1, 2], [1, 4], [2, 3], [2, 5]].
    expected = float(np.mean(targets**2))
    assert np.isclose(expected, 8.0)
    np.testing.assert_allclose(
        oracle.func_fit_loss(
            np.zeros((pop_size, num_samples, num_outputs)), targets, "mse"
        ),
        np.full(pop_size, expected),
        rtol=1e-12,
    )

    assert observed.shape == (pop_size,)
    np.testing.assert_allclose(observed, np.full(pop_size, -expected), rtol=1e-6)
    # A mean over the sample axis only (a factor ``O`` too large) ...
    assert not np.isclose(observed[0], -expected * num_outputs)
    # ... and a sum instead of a mean (a factor ``S * O`` too large) both differ.
    assert not np.isclose(observed[0], -expected * num_samples * num_outputs)


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


def test_custom_func_fit_grid_builds_the_cartesian_product_in_order(problem):
    """Invariant: ``method="grid"`` is the row-major cartesian product of the per-axis grids.

    Oracle: ``np.arange(low[i], high[i], step[i])`` per axis, combined
    row-major (first axis slowest) -- the order ``cartesian_product`` documents.
    With ``low = [0, 0]``, ``high = [2, 3]`` and ``step = [1, 1]`` the six rows
    are ``[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]``, and the targets
    are ``np.apply_along_axis(func, -1, inputs)``.

    Notes:
        The order matters twice: the tiling test assumes sample order, and a
        user reading a printed report expects the first axis to move slowest.
        ``S`` is the product of the per-axis grid lengths, never ``num_samples``
        (left at its default ``100`` here).
    """

    def func(point):
        return point[0] * 10.0 + point[1]

    custom = problem.CustomFuncFit(
        func,
        low_bounds=[0.0, 0.0],
        upper_bounds=[2.0, 3.0],
        method="grid",
        step_size=[1.0, 1.0],
    )
    inputs = np.asarray(custom.inputs)
    targets = np.asarray(custom.targets)

    expected_inputs = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=np.float32
    )
    assert inputs.shape == (6, 2)
    np.testing.assert_array_equal(inputs, expected_inputs)

    # "first axis slowest" as a computed property: the grid is the ``indexing="ij"``
    # meshgrid of the two per-axis ``arange`` grids, flattened row-major.
    axis_grids = [np.arange(0, 2, 1), np.arange(0, 3, 1)]
    mesh = np.stack(np.meshgrid(*axis_grids, indexing="ij"), axis=-1).reshape(-1, 2)
    np.testing.assert_array_equal(inputs.astype(np.int64), mesh)

    assert targets.shape == (6, 1)
    np.testing.assert_array_equal(
        targets,
        np.apply_along_axis(func, -1, expected_inputs).astype(np.float32)[:, None],
    )
    np.testing.assert_array_equal(targets[:, 0], np.array([0, 1, 2, 10, 11, 12], dtype=np.float32))

    assert custom.num_samples == 100
    assert custom.inputs.shape[0] != custom.num_samples
    assert custom.input_shape == (2,)
    assert custom.output_shape == (1,)


def test_cartesian_product_pairs_every_row_of_both_inputs_major_first_slowest(problem):
    """Invariant: ``custom.cartesian_product`` repeats ``arr1`` rows for each ``arr2`` row.

    Oracle: closed form. ``arr1 = [[0], [1]]`` and ``arr2 = [[10], [11], [12]]``
    must produce ``[[0, 10], [0, 11], [0, 12], [1, 10], [1, 11], [1, 12]]`` --
    shape ``(n1 * n2, k1 + k2)``, first input slowest -- and an ``AssertionError``
    for inputs of different rank or above rank 2.
    """
    # ``cartesian_product`` is exported by ``problem.func_fit.custom`` only; the
    # package re-exports the problems, not the grid helper.
    cartesian_product = problem.func_fit.custom.cartesian_product

    first = np.array([[0], [1]], dtype=np.int32)
    second = np.array([[10], [11], [12]], dtype=np.int32)
    product = cartesian_product(first, second)

    expected = np.array(
        [[0, 10], [0, 11], [0, 12], [1, 10], [1, 11], [1, 12]], dtype=np.int32
    )
    assert product.shape == (first.shape[0] * second.shape[0], 2)
    np.testing.assert_array_equal(product, expected)

    # "first input slowest": column 0 repeats each ``first`` row ``n2`` times and
    # column 1 repeats the whole of ``second`` row by row.
    np.testing.assert_array_equal(product[:, 0], np.repeat(first[:, 0], second.shape[0]))
    np.testing.assert_array_equal(product[:, 1], np.tile(second[:, 0], first.shape[0]))

    # Rank-1 inputs are single columns, so the result is the same ``(6, 2)`` block.
    np.testing.assert_array_equal(
        cartesian_product(np.array([0, 1]), np.array([10, 11, 12])), expected
    )

    with pytest.raises(AssertionError):
        cartesian_product(np.array([[0], [1]]), np.array([1, 2, 3]))
    with pytest.raises(AssertionError):
        cartesian_product(np.zeros((1, 1, 1)), np.zeros((2, 1, 1)))


def test_custom_func_fit_sample_method_is_reproducible_from_its_seed(problem):
    """Invariant: ``method="sample"`` datasets are reproducible from ``seed`` alone.

    Oracle: two ``CustomFuncFit`` instances built with the same seed must have
    bit-identical ``inputs``/``targets``; a different seed must give different
    inputs (asserted on the arrays, not on a hash). Every row must lie inside
    ``[low_bounds, upper_bounds]``, ``S == num_samples`` as documented, and the
    targets must equal ``np.apply_along_axis(func, -1, inputs)`` re-derived in
    the test -- so a dataset that ignored ``func`` cannot pass.

    Notes:
        ``np.random.default_rng(seed)`` is host-side only; the test also asserts
        that constructing a problem leaves the *global* ``np.random`` state
        untouched (``np.random.get_state()`` before/after), which is the legacy
        coupling this rewrite removes.
    """
    num_samples = 8

    def func(point):
        return point[0] - point[1]

    def build(seed):
        return problem.CustomFuncFit(
            func,
            low_bounds=[0.0, 0.0],
            upper_bounds=[1.0, 1.0],
            method="sample",
            num_samples=num_samples,
            seed=seed,
        )

    global_before = np.random.get_state()
    first = build(7)
    second = build(7)
    other = build(8)
    global_after = np.random.get_state()

    inputs = np.asarray(first.inputs)
    targets = np.asarray(first.targets)
    assert inputs.shape == (num_samples, 2) == (first.num_samples, 2)
    assert targets.shape == (num_samples, 1)
    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32
    assert first.input_shape == (2,)
    assert first.output_shape == (1,)

    # Reproducible bit for bit from the seed ...
    assert inputs.tobytes() == np.asarray(second.inputs).tobytes()
    assert targets.tobytes() == np.asarray(second.targets).tobytes()
    # ... and a different seed really moves the draw.
    assert not np.array_equal(inputs, np.asarray(other.inputs))

    assert np.all(inputs >= np.asarray(first.low_bounds))
    assert np.all(inputs <= np.asarray(first.upper_bounds))

    np.testing.assert_allclose(
        targets[:, 0], np.apply_along_axis(func, -1, inputs), rtol=0, atol=1e-6
    )

    # The global legacy state is untouched.
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]


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


def test_custom_func_fit_validates_its_box_and_sampling_configuration(problem, common):
    """Invariant: every documented misuse of ``CustomFuncFit`` raises ``TensorNEATError``.

    Oracle: the ``Raises:`` clause of the constructor docstring --
    ``method`` outside ``{"sample", "grid"}``, ``num_samples <= 0``, a missing
    ``step_size`` for ``method="grid"``, a ``step_size`` that does not match the
    bounds' shape or is not strictly positive,
    ``low_bounds.shape != upper_bounds.shape``, and ``func(low_bounds)`` raising.
    Each case is asserted separately, so a partially-validating constructor is
    caught.
    """

    def func(point):
        return point[0] + point[1]

    valid = dict(
        func=func,
        low_bounds=[0.0, 0.0],
        upper_bounds=[2.0, 2.0],
        method="grid",
        step_size=[1.0, 1.0],
        num_samples=4,
    )
    problem.CustomFuncFit(**valid)  # the baseline really is accepted

    cases = [
        ("method outside {'sample', 'grid'}", dict(method="quadrature")),
        ("num_samples == 0", dict(num_samples=0)),
        ("num_samples < 0", dict(num_samples=-5)),
        ("method='grid' without step_size", dict(step_size=None)),
        ("step_size of the wrong shape", dict(step_size=[1.0, 1.0, 1.0])),
        ("step_size with a zero entry", dict(step_size=[1.0, 0.0])),
        ("step_size with a negative entry", dict(step_size=[-1.0, 1.0])),
        ("method='sample' with num_samples == 0", dict(method="sample", num_samples=0)),
        (
            "low_bounds.shape != upper_bounds.shape",
            dict(low_bounds=[0.0, 0.0], upper_bounds=[2.0, 2.0, 2.0]),
        ),
    ]
    for label, overrides in cases:
        kwargs = dict(valid)
        kwargs.update(overrides)
        with pytest.raises(common.TensorNEATError):
            problem.CustomFuncFit(**kwargs)

    # ``step_size`` is a grid-only requirement: ``method="sample"`` ignores it.
    problem.CustomFuncFit(
        func,
        low_bounds=[0.0, 0.0],
        upper_bounds=[2.0, 2.0],
        method="sample",
        num_samples=4,
        step_size=None,
        seed=0,
    )

    def broken(point):
        raise ValueError("boom")

    with pytest.raises(common.TensorNEATError) as excinfo:
        problem.CustomFuncFit(
            broken,
            low_bounds=[0.0, 0.0],
            upper_bounds=[2.0, 2.0],
            method="grid",
            step_size=[1.0, 1.0],
        )
    assert "boom" in str(excinfo.value)
