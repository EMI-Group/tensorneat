"""Unit tests for ``CustomFuncFit`` -- the user-supplied-function problem.

What this module proves
-----------------------
``CustomFuncFit`` (``src/tensorneat/problem/func_fit/custom.py``) is the only
func-fit subclass whose dataset is *generated* rather than a literal, so it owns
two separable contracts:

1. **Dataset generation.** ``method="grid"`` is the row-major cartesian product
   of the per-axis ``np.arange(low[i], high[i], step[i])`` grids (first axis
   slowest, ``S`` = the product of the grid lengths, unrelated to
   ``num_samples``); ``method="sample"`` draws ``num_samples`` rows from
   ``np.random.default_rng(seed)`` and so is reproducible from the seed alone.
   Targets are always ``np.apply_along_axis(func, -1, inputs)``.
2. **Construction-time validation.** Every documented misuse is a
   ``TensorNEATError`` raised on the host, before anything is traced: a bad box
   (mismatched shapes, a non-finite corner, a zero-width or reversed axis), a
   bad ``method``/``num_samples``/``step_size``, a ``func`` that raises, and a
   grid whose baked constants exceed ``FuncFit.MAX_DATASET_NBYTES``.

Oracles used here (never the kernel under test)
-----------------------------------------------
* ``np.arange`` / ``np.meshgrid`` / ``np.apply_along_axis`` re-derived inside
  the test -- the independent spelling of the documented generation rule;
* the documented error messages, matched per case, so each rejection is
  attributed to *its own* validation path instead of to any error at all.

The loss reductions themselves are not this module's subject: they belong to
``test_func_fit.py``, which drives ``FuncFit.evaluate``.

Backend is irrelevant here (every assertion is host-side dataset algebra), and
no graph is built at all; no ``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# dataset generation
# --------------------------------------------------------------------------- #


def test_grid_builds_the_cartesian_product_in_order(problem):
    """Invariant: ``method="grid"`` is the row-major cartesian product of the per-axis grids.

    Oracle: ``np.arange(low[i], high[i], step[i])`` per axis, combined
    row-major (first axis slowest) -- the order ``cartesian_product`` documents.
    With ``low = [0, 0]``, ``high = [2, 3]`` and ``step = [1, 1]`` the six rows
    are ``[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]``, and the targets
    are ``np.apply_along_axis(func, -1, inputs)``.

    Notes:
        The order matters twice: the tiling test in ``test_func_fit.py`` assumes
        sample order, and a user reading a printed report expects the first axis
        to move slowest. ``S`` is the product of the per-axis grid lengths, never
        ``num_samples`` (left at its default ``100`` here).
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
    np.testing.assert_array_equal(
        targets[:, 0], np.array([0, 1, 2, 10, 11, 12], dtype=np.float32)
    )

    assert custom.num_samples == 100
    assert custom.inputs.shape[0] != custom.num_samples
    assert custom.input_shape == (2,)
    assert custom.output_shape == (1,)


@pytest.mark.parametrize(
    "bounds, step, expected_rows",
    [
        # One axis: the grid is just ``arange(low, high, step)`` as a column.
        (([0.0], [1.0]), [0.25], [[0.0], [0.25], [0.5], [0.75]]),
        # Three axes: S = 2*2*2 = 8, first axis slowest.
        (
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            [0.5, 0.5, 0.5],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
            ],
        ),
    ],
    ids=["one_axis", "three_axes"],
)
def test_grid_spans_one_and_three_axes(problem, bounds, step, expected_rows):
    """Invariant: the grid generation is dimension-agnostic (1-D and 3-D boxes).

    Oracle: ``np.arange`` per axis combined with ``indexing="ij"`` meshgrid and
    flattened row-major, re-derived in the test. A one-axis box must not collapse
    to an empty or single-row dataset, and a three-axis box must expose ``S = 8``
    rows with the first axis slowest -- the two shapes a 2-D-only implementation
    (the only one the 2-D test above pins) could get wrong.

    Notes:
        ``input_shape`` is the number of bounds, and every target row is
        ``func(point)`` on the matching input row, so a grid that ordered rows
        differently would disagree in the target column too.
    """
    low, high = bounds
    num_axes = len(low)

    def func(point):
        return point[0] + 2.0 * point[1] if num_axes > 1 else point[0]

    custom = problem.CustomFuncFit(
        func,
        low_bounds=low,
        upper_bounds=high,
        method="grid",
        step_size=step,
    )
    inputs = np.asarray(custom.inputs)
    expected = np.array(expected_rows, dtype=np.float32)
    assert inputs.shape == expected.shape
    np.testing.assert_array_equal(inputs, expected)
    assert custom.input_shape == (num_axes,)
    assert custom.output_shape == (1,)
    np.testing.assert_allclose(
        np.asarray(custom.targets)[:, 0],
        np.apply_along_axis(func, -1, expected),
        rtol=0,
        atol=1e-6,
    )


def test_cartesian_product_pairs_every_row_of_both_inputs_major_first_slowest(
    problem, common
):
    """Invariant: ``custom.cartesian_product`` repeats ``arr1`` rows for each ``arr2`` row.

    Oracle: closed form. ``arr1 = [[0], [1]]`` and ``arr2 = [[10], [11], [12]]``
    must produce ``[[0, 10], [0, 11], [0, 12], [1, 10], [1, 11], [1, 12]]`` --
    shape ``(n1 * n2, k1 + k2)``, first input slowest -- and a
    ``TensorNEATError`` (the library's single configuration-error type, never a
    bare ``AssertionError``) for inputs of different rank or above rank 2.
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
    np.testing.assert_array_equal(
        product[:, 0], np.repeat(first[:, 0], second.shape[0])
    )
    np.testing.assert_array_equal(
        product[:, 1], np.tile(second[:, 0], first.shape[0])
    )

    # Rank-1 inputs are single columns, so the result is the same ``(6, 2)`` block.
    np.testing.assert_array_equal(
        cartesian_product(np.array([0, 1]), np.array([10, 11, 12])), expected
    )

    with pytest.raises(common.TensorNEATError, match="same number of dimensions"):
        cartesian_product(np.array([[0], [1]]), np.array([1, 2, 3]))
    with pytest.raises(common.TensorNEATError, match="at most 2 dimensions"):
        cartesian_product(np.zeros((1, 1, 1)), np.zeros((2, 1, 1)))


def test_sample_method_is_reproducible_from_its_seed(problem):
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
# box validation: one test per documented rejection path
# --------------------------------------------------------------------------- #


def _func(point):
    """The reference target function: ``point[0] + point[1]``."""
    return point[0] + point[1]


def _build(problem, **overrides):
    """Construct a valid ``CustomFuncFit``, overriding individual keywords."""
    kwargs = dict(
        func=_func,
        low_bounds=[0.0, 0.0],
        upper_bounds=[2.0, 2.0],
        method="grid",
        step_size=[1.0, 1.0],
        num_samples=4,
    )
    kwargs.update(overrides)
    return problem.CustomFuncFit(**kwargs)


def test_a_zero_width_box_is_a_configuration_error(problem, common):
    """Invariant: every axis must satisfy ``low < high``; equality is rejected.

    Oracle: the constructor's ``Raises:`` clause -- "if any axis is not a proper
    interval (``low_bounds[i] < upper_bounds[i]``)", with the reason spelled out
    in the message ("a zero-width or reversed box yields an empty dataset
    (S = 0)"). An equal corner is the interesting half: ``np.arange(0, 0, 1)``
    is empty, so the dataset would have ``S = 0`` and every loss would be a
    ``0/0`` NaN that silently demotes the whole population.

    Notes:
        The same path covers a *reversed* box (``low > high``); both are asserted
        separately so a ``<=``-typo cannot pass one and fail the other by
        accident. ``axis 0`` and ``axis 1`` are checked, so the validator is not
        only inspecting the first axis.
    """
    equal_first = dict(low_bounds=[0.0, 0.0], upper_bounds=[0.0, 2.0])
    equal_second = dict(low_bounds=[0.0, 0.0], upper_bounds=[2.0, 0.0])
    reversed_box = dict(low_bounds=[2.0, 0.0], upper_bounds=[0.0, 2.0])

    for label, overrides in {
        "equal first axis": equal_first,
        "equal second axis": equal_second,
        "reversed axis": reversed_box,
    }.items():
        with pytest.raises(common.TensorNEATError) as excinfo:
            _build(problem, **overrides)
        message = str(excinfo.value)
        assert "strictly below" in message, label
        assert "low_bounds" in message and "upper_bounds" in message, label


def test_a_non_finite_bound_is_a_configuration_error(problem, common):
    """Invariant: ``NaN``/``+inf`` in either corner is rejected before any ``<`` test.

    Oracle: the constructor's ``Raises:`` clause -- "if a bound is not finite",
    from the explicit ``np.isfinite`` census. That census must run **first**:
    every comparison against ``NaN`` is false, so a ``low < high``-only check
    would accept ``low = NaN`` and then sample from an undefined box.

    Notes:
        All four corners (``NaN``/``+inf`` on the low side and the high side) are
        asserted, and the message names the offending axis, so the finiteness
        rule is proven on both arguments rather than on one.
    """
    cases = {
        "nan low": dict(low_bounds=[float("nan"), 0.0], upper_bounds=[1.0, 1.0]),
        "inf low": dict(low_bounds=[float("-inf"), 0.0], upper_bounds=[1.0, 1.0]),
        "nan high": dict(low_bounds=[0.0, 0.0], upper_bounds=[float("nan"), 1.0]),
        "inf high": dict(low_bounds=[0.0, 0.0], upper_bounds=[float("inf"), 1.0]),
    }
    for label, overrides in cases.items():
        with pytest.raises(common.TensorNEATError) as excinfo:
            _build(problem, **overrides)
        message = str(excinfo.value)
        assert "bounds must be finite" in message, label
        assert "axis" in message, label


# --------------------------------------------------------------------------- #
# method / sampling validation
# --------------------------------------------------------------------------- #


def test_validates_its_box_and_sampling_configuration(problem, common):
    """Invariant: every documented misuse raises ``TensorNEATError``, attributed to its own path.

    Oracle: the ``Raises:`` clause of the constructor docstring --
    ``method`` outside ``{"sample", "grid"}``, ``num_samples <= 0``, a missing
    ``step_size`` for ``method="grid"``, a ``step_size`` that does not match the
    bounds' shape or is not strictly positive,
    ``low_bounds.shape != upper_bounds.shape``, and ``func(low_bounds)`` raising.

    Notes:
        Each case matches a message fragment unique to its validator, so a
        constructor that rejected everything through one catch-all branch (or
        through the wrong branch) cannot pass: an unmatched ``pytest.raises``
        would hide exactly the kind of regression this test exists for.
    """
    _build(problem)  # the baseline really is accepted

    cases = [
        ("method outside {'sample', 'grid'}", dict(method="quadrature"), "method must be"),
        ("num_samples == 0", dict(num_samples=0), "num_samples must be positive"),
        ("num_samples < 0", dict(num_samples=-5), "num_samples must be positive"),
        ("method='grid' without step_size", dict(step_size=None), "step_size must be provided"),
        (
            "step_size of the wrong shape",
            dict(step_size=[1.0, 1.0, 1.0]),
            "same shape as the bounds",
        ),
        ("step_size with a zero entry", dict(step_size=[1.0, 0.0]), "strictly positive"),
        ("step_size with a negative entry", dict(step_size=[-1.0, 1.0]), "strictly positive"),
        (
            "method='sample' with num_samples == 0",
            dict(method="sample", num_samples=0),
            "num_samples must be positive",
        ),
        (
            "low_bounds.shape != upper_bounds.shape",
            dict(low_bounds=[0.0, 0.0], upper_bounds=[2.0, 2.0, 2.0]),
            "must equal upper_bounds.shape",
        ),
    ]
    for label, overrides, fragment in cases:
        with pytest.raises(common.TensorNEATError) as excinfo:
            _build(problem, **overrides)
        assert fragment in str(excinfo.value), label

    # ``step_size`` is a grid-only requirement: ``method="sample"`` ignores it.
    _build(problem, method="sample", step_size=None, seed=0)

    def broken(point):
        raise ValueError("boom")

    with pytest.raises(common.TensorNEATError) as excinfo:
        _build(problem, func=broken)
    assert "func(low_bounds) raised ValueError" in str(excinfo.value)
    assert "boom" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the baked-dataset budget, from the CustomFuncFit side
# --------------------------------------------------------------------------- #


def test_an_oversized_grid_exceeds_the_dataset_budget(problem, common):
    """Invariant: a grid whose baked constants exceed ``MAX_DATASET_NBYTES`` is rejected.

    Oracle: the documented budget arithmetic. The grid ``[0, 400)^2`` at step
    ``1`` has ``S = 400 * 400 = 160_000`` rows, so the two float32 constants cost
    ``160_000 * 2 * 4 + 160_000 * 1 * 4 = 1_920_000`` bytes against the 1 MiB
    (``1_048_576``) default -- computed in the test from the shapes, never read
    off the exception. The message must name both the measured size and the
    attribute, because the budget is the *documented* knob a user raises.

    Notes:
        The budget is enforced at construction by ``FuncFit.__init__`` (which
        calls ``validate_dataset`` after the dataset exists), so no graph is ever
        built and the failure is a configuration error rather than an evoxc
        warning mid-trace. A subclass that deliberately raises the attribute
        accepts the same grid, which proves the constant -- not a hard-coded
        literal -- is what the check compares against.
    """

    def func(point):
        return point[0] + point[1]

    grid = dict(
        func=func,
        low_bounds=[0.0, 0.0],
        upper_bounds=[400.0, 400.0],
        method="grid",
        step_size=[1.0, 1.0],
    )
    samples = 400 * 400
    expected_nbytes = samples * 2 * 4 + samples * 1 * 4
    assert expected_nbytes > problem.FuncFit.MAX_DATASET_NBYTES

    with pytest.raises(common.TensorNEATError) as excinfo:
        problem.CustomFuncFit(**grid)
    message = str(excinfo.value)
    assert str(expected_nbytes) in message
    assert "MAX_DATASET_NBYTES" in message

    class _RoomierCustomFuncFit(problem.CustomFuncFit):
        """The documented escape hatch: a subclass with a deliberately raised budget."""

        MAX_DATASET_NBYTES = 8 << 20

    roomy = _RoomierCustomFuncFit(**grid)
    assert roomy.dataset_nbytes == expected_nbytes
    assert np.asarray(roomy.inputs).shape == (samples, 2)
