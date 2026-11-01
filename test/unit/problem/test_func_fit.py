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


def _positive_grid_problem(problem):
    """A ``CustomFuncFit`` on ``[0, 1]^2`` with strictly positive targets.

    Returns:
        ``CustomFuncFit(lambda point: point[0] + 1.5, low_bounds=[0, 0],
        upper_bounds=[2, 2], method="grid", step_size=[1, 1])``: a deterministic
        ``S = 4`` dataset whose targets lie in ``[1.5, 2.5]``.

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
    )


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
    pytest.skip(
        "not implemented: the dataset tiling is individual-major, row p*S + s "
        "(owner: problem/func_fit/func_fit.FuncFit.evaluate)"
    )


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
    pytest.skip(
        f"not implemented: {error_method} of a constant 0.25 predictor equals the "
        "hand-computed scalar (owner: problem/func_fit/func_fit.FuncFit._loss)"
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
    pytest.skip(
        f"not implemented: {error_method} equals the numpy oracle over both axes "
        "(owner: problem/func_fit/func_fit.FuncFit._loss)"
    )


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
    pytest.skip(
        "not implemented: mape divides by the target, so a zero target -> -inf fitness "
        "(owner: problem/func_fit/func_fit.FuncFit._loss)"
    )


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
    pytest.skip(
        "not implemented: the loss is a mean over both the sample and output axes "
        "(owner: problem/func_fit/func_fit.FuncFit._loss)"
    )


def test_a_nan_prediction_scores_negative_infinity(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: a NaN prediction makes the fitness ``-inf``, never a survivor.

    Oracle: the documented rule ("a NaN prediction -- an unconnected or dead
    topology -- yields a NaN loss and therefore ``-inf`` fitness"). Every
    individual of a population whose ``act_fn`` emits NaN must come back as
    ``-inf``, which is what keeps a broken topology from winning selection.
    """
    pytest.skip(
        "not implemented: a NaN prediction gives a NaN loss and -inf fitness "
        "(owner: problem/func_fit/func_fit.FuncFit.evaluate + problem/base.finalize_fitness)"
    )


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
    pytest.skip(
        "not implemented: XOR's four-row literal dataset "
        "(owner: problem/func_fit/xor.XOR.inputs/targets)"
    )


def test_xor3d_dataset_is_the_eight_row_parity_literal(problem):
    """Invariant: ``XOR3d`` is the literal ``S = 8``, ``I = 3`` parity dataset.

    Oracle: the literal in the class docstring -- all three-bit patterns in
    counting order, targets ``[[0], [1], [1], [0], [1], [0], [0], [1]]`` (the
    3-bit parity, i.e. ``sum(bits) % 2``), ``input_shape == (3,)`` and
    ``output_shape == (1,)``. The input rows are also checked against
    ``np.arange(8)`` written out in base 2, so "counting order" is a computed
    property, not a copied literal.
    """
    pytest.skip(
        "not implemented: XOR3d's eight-row parity literal dataset "
        "(owner: problem/func_fit/xor3d.XOR3d.inputs/targets)"
    )


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
    """
    pytest.skip(
        "not implemented: method='grid' is the row-major cartesian product of the "
        "per-axis arange grids (owner: problem/func_fit/custom.CustomFuncFit.generate_dataset)"
    )


def test_cartesian_product_pairs_every_row_of_both_inputs_major_first_slowest(problem):
    """Invariant: ``custom.cartesian_product`` repeats ``arr1`` rows for each ``arr2`` row.

    Oracle: closed form. ``arr1 = [[0], [1]]`` and ``arr2 = [[10], [11], [12]]``
    must produce ``[[0, 10], [0, 11], [0, 12], [1, 10], [1, 11], [1, 12]]`` --
    shape ``(n1 * n2, k1 + k2)``, first input slowest -- and an ``AssertionError``
    for inputs of different rank or above rank 2.
    """
    pytest.skip(
        "not implemented: cartesian_product is row-major, first input slowest "
        "(owner: problem/func_fit/custom.cartesian_product)"
    )


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
    pytest.skip(
        "not implemented: method='sample' draws from np.random.default_rng(seed), "
        "not the global state (owner: problem/func_fit/custom.CustomFuncFit.generate_dataset)"
    )


# --------------------------------------------------------------------------- #
# configuration errors
# --------------------------------------------------------------------------- #


def test_an_unknown_error_method_is_a_configuration_error(problem, common):
    """Invariant: an unsupported ``error_method`` raises ``TensorNEATError`` at construction.

    Oracle: the class attribute ``ERROR_METHODS == ("mse", "rmse", "mae",
    "mape")`` (also compared with ``support.oracle.FUNC_FIT_ERROR_METHODS``, so
    the test and the oracle cannot drift), and ``TensorNEATError`` for any other
    string -- on ``FuncFit`` subclasses alike (``XOR``, ``XOR3d``,
    ``CustomFuncFit``). The check is at construction time, on the host, before
    any graph is traced.
    """
    pytest.skip(
        "not implemented: an unknown error_method raises TensorNEATError "
        "(owner: problem/func_fit/func_fit.FuncFit.__init__)"
    )


def test_custom_func_fit_validates_its_box_and_sampling_configuration(problem, common):
    """Invariant: every documented misuse of ``CustomFuncFit`` raises ``TensorNEATError``.

    Oracle: the ``Raises:`` clause of the constructor docstring --
    ``method`` outside ``{"sample", "grid"}``, ``num_samples <= 0``, a missing
    ``step_size`` for ``method="grid"``, a ``step_size`` that does not match the
    bounds' shape or is not strictly positive, and
    ``low_bounds.shape != upper_bounds.shape``. Each case is asserted
    separately, so a partially-validating constructor is caught.
    """
    pytest.skip(
        "not implemented: CustomFuncFit rejects a bad box/step/sample configuration "
        "(owner: problem/func_fit/custom.CustomFuncFit.__init__)"
    )
