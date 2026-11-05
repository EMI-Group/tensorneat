"""Unit tests for ``HyperNEAT._weights_from_query``: the CPPN output -> weight map.

What this module proves
-----------------------
``_weights_from_query`` is the one arithmetic step that turns a CPPN's raw output into a
substrate connection's weight, and it is the only place the constructor's
``weight_threshold`` / ``max_weight`` knobs are consumed.  The documented map is

    ``sign(w) * max(|w| - weight_threshold, 0) * max_weight / (1 - weight_threshold)``

with a *strict* dead zone (``|w| <= weight_threshold`` prunes the edge, i.e. the weight is
exactly ``0`` and, unlike the legacy ``-t < w < t`` test, the boundary ties are inside it)
and NaN propagated rather than masked (``sign(NaN)`` is NaN and ``0 * NaN`` is NaN).

Every assertion compares the traced kernel against an oracle written here, in this file:
:func:`_documented_map` is a hand transcription of the formula evaluated in float32 steps
(so the comparison can be bit-exact) and :func:`_closed_form` states the same value in
float64 straight from the formula (so the float32 transcription is never the only
witness).  The implementation is never called to compute an expectation.

The harness itself (``./hyperneat_kernel.py``) is verified by the last two tests: a
``State`` from ``make_state`` threaded through a real decode kernel, and the documented
rejection of a state-taking method called without its state.

No network, no GPU, ``backend="numpy"``; ``P = 4`` and ``Q = 4``, so every table below is
readable by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from .hyperneat_kernel import (
    CONNS_CNT,
    POP_SIZE,
    cppn_payload,
    make_hyper,
    make_state,
    run_method,
)

#: ``HyperNEAT``'s documented default dead-zone half-width, restated here so the oracle
#: does not read it off the object under test.
DEFAULT_THRESHOLD = 0.3

#: ``HyperNEAT``'s documented default saturating magnitude, for the same reason.
DEFAULT_MAX_WEIGHT = 5.0


def _documented_map(
    raw,
    threshold: float = DEFAULT_THRESHOLD,
    max_weight: float = DEFAULT_MAX_WEIGHT,
):
    """Hand transcription of the documented map, evaluated in float32 steps.

    ``abs`` -> ``greater`` -> ``select`` -> ``subtract`` -> ``sign`` -> two multiplies, the
    order the documentation describes and the order the kernel uses, in ``float32`` so the
    result is bit-exact and can be compared with ``rtol=0, atol=0``.

    Args:
        raw: A scalar or an ``(P, Q)`` array of raw CPPN outputs.
        threshold: Dead-zone half-width.
        max_weight: Saturating magnitude.

    Returns:
        ``float32`` array (or 0-d array) of the same shape: ``0`` where ``|w| <= threshold``,
        ``sign(w) * (|w| - threshold) * max_weight / (1 - threshold)`` elsewhere, and NaN
        where ``raw`` is NaN (``np.where`` evaluates both branches, so the dead zone does
        not mask NaN -- ``sign(NaN) * 0`` is NaN -- exactly as documented).
    """
    x = np.asarray(raw, dtype=np.float32)
    threshold_f32 = np.float32(threshold)
    scale = np.float32(float(max_weight) / (1.0 - float(threshold)))
    magnitude = np.abs(x)
    shifted = np.where(
        magnitude > threshold_f32, magnitude - threshold_f32, np.float32(0.0)
    ).astype(np.float32)
    return (np.sign(x) * shifted) * scale


def _closed_form(
    w: float,
    threshold: float = DEFAULT_THRESHOLD,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> float:
    """The same map in float64, straight from the formula (no operation ordering).

    Args:
        w: One raw CPPN output.
        threshold: Dead-zone half-width.
        max_weight: Saturating magnitude.

    Returns:
        ``0.0`` inside the dead zone, NaN for a NaN input, and
        ``sign(w) * (|w| - threshold) * max_weight / (1 - threshold)`` outside it.
    """
    w = float(w)
    if np.isnan(w):
        return float("nan")
    excess = abs(w) - threshold
    if excess <= 0.0:
        return 0.0
    return float(np.sign(w)) * excess * max_weight / (1.0 - threshold)


def _mapped(hyper, raw):
    """Run the real kernel on ``raw`` and return its host-numpy output."""
    return run_method(hyper, "_weights_from_query", np.asarray(raw, dtype=np.float32))


def _mapped_scalar(hyper, w: float) -> np.float32:
    """Run the real kernel on the single value ``w`` and return that value as float32."""
    out = _mapped(hyper, np.array([[w]], dtype=np.float32))
    assert out.shape == (1, 1), (
        f"_weights_from_query on a (1, 1) input must return a (1, 1) tensor, got {out.shape}"
    )
    return np.float32(out[0, 0])


@pytest.fixture(scope="module")
def hyper():
    """The default ``HyperNEAT`` of the harness (``weight_threshold=0.3``, ``max_weight=5``)."""
    return make_hyper()


def test_exactly_zero_maps_to_zero(hyper):
    """A CPPN output of ``0`` prunes its edge: ``sign(0) == 0`` and ``|0| <= threshold``."""
    got = _mapped_scalar(hyper, 0.0)
    assert got == np.float32(0.0), (
        f"_weights_from_query(0.0) must be exactly 0 (both terms of the product are 0), "
        f"got {got!r}"
    )
    assert got == np.float32(_documented_map(0.0)), (
        f"_weights_from_query(0.0) disagrees with the documented map, which is 0: got {got!r}"
    )


def test_threshold_tie_is_pruned(hyper):
    """``|w| == threshold`` is *inside* the dead zone, for both signs.

    The legacy test was ``-t < w < t``; the documented split of this rewrite is strict
    (``|w| <= t`` -> 0), so a tie must prune.  A value just below the threshold
    (``0.299``) is inside it too.
    """
    below = np.float32(0.299)
    for w, why in (
        (DEFAULT_THRESHOLD, "the positive tie"),
        (-DEFAULT_THRESHOLD, "the negative tie"),
        (float(below), "a value just below the threshold"),
        (-float(below), "a value just below the negative threshold"),
    ):
        got = _mapped_scalar(hyper, w)
        expected = np.float32(_documented_map(w))
        assert got == expected, (
            f"_weights_from_query({w}) ({why}) must prune to the documented dead-zone "
            f"value {expected!r}, got {got!r}"
        )
        assert float(got) == 0.0, (
            f"_weights_from_query({w}) ({why}) must be 0, got {got!r}; the dead zone is "
            f"|w| <= weight_threshold == {DEFAULT_THRESHOLD}"
        )


def test_just_above_the_threshold_is_shifted(hyper):
    """``threshold + eps`` maps to ``max_weight * eps / (1 - threshold)``, sign preserved.

    With ``eps = 0.05`` the documented value is ``5 * 0.05 / 0.7 == 5/14``; float32
    represents it as ``0.35714272``, so the kernel is compared bit-exactly against
    :func:`_documented_map` and within ``1e-6`` relative of the float64 closed form.
    The ``1e-6`` (rather than one-ULP ``1e-7``) is what the *inputs* cost: float32 has no
    ``0.35`` and no ``0.3``, and ``float32(0.35) - float32(0.3) == 0.049999982`` rather
    than ``0.05`` -- a 3.7e-7 relative error in ``eps`` that the map carries through.
    """
    eps = 0.05
    w = DEFAULT_THRESHOLD + eps
    positive = _mapped_scalar(hyper, w)
    exact_closed = _closed_form(w)
    assert exact_closed == pytest.approx(DEFAULT_MAX_WEIGHT * eps / (1.0 - DEFAULT_THRESHOLD)), (
        "the oracle's own closed form must equal max_weight * eps / (1 - threshold) = "
        f"5/14, got {exact_closed!r}"
    )
    assert positive == np.float32(_documented_map(w)), (
        f"_weights_from_query({w}) must equal the documented map "
        f"{np.float32(_documented_map(w))!r}, got {positive!r}"
    )
    np.testing.assert_allclose(
        positive,
        exact_closed,
        rtol=1e-6,
        atol=0,
        err_msg=(
            f"_weights_from_query({w}) must be the documented max_weight * eps / "
            f"(1 - threshold) = {exact_closed!r} up to the float32 cost of eps, got "
            f"{positive!r}"
        ),
    )

    negative = _mapped_scalar(hyper, -w)
    assert negative == np.float32(-positive), (
        f"_weights_from_query({-w}) must be the exact negative of "
        f"_weights_from_query({w}) == {positive!r} (abs/sign are symmetric), got {negative!r}"
    )
    np.testing.assert_allclose(
        negative,
        -exact_closed,
        rtol=1e-6,
        atol=0,
        err_msg=(
            f"_weights_from_query({-w}) must be -{exact_closed!r} up to the float32 cost "
            f"of eps, got {negative!r}"
        ),
    )


def test_large_magnitude_scales_linearly(hyper):
    """``w = 1.0`` saturates at ``max_weight``; ``w = 0.65`` lands on ``2.5``.

    ``5.0 * (1.0 - 0.3) / 0.7 == 5.0`` and ``5.0 * (0.65 - 0.3) / 0.7 == 2.5`` in exact
    arithmetic.  In float32 the first is exact and the second is ``2.4999998`` because
    ``float32(0.65) - float32(0.3)`` is ``0.34999996`` -- so the closed-form comparison is
    ``rtol=1e-6`` while the transcription comparison stays bit-exact.
    """
    saturated = _mapped_scalar(hyper, 1.0)
    assert saturated == np.float32(DEFAULT_MAX_WEIGHT), (
        f"_weights_from_query(1.0) must saturate at max_weight == {DEFAULT_MAX_WEIGHT}, "
        f"got {saturated!r}"
    )
    assert saturated == np.float32(_documented_map(1.0)), (
        f"_weights_from_query(1.0) disagrees with the documented map, which is "
        f"{DEFAULT_MAX_WEIGHT}, got {saturated!r}"
    )

    partial = _mapped_scalar(hyper, 0.65)
    assert partial == np.float32(_documented_map(0.65)), (
        f"_weights_from_query(0.65) must equal the float32 transcription of "
        f"max_weight * (0.65 - threshold) / (1 - threshold) == {np.float32(_documented_map(0.65))!r}, "
        f"got {partial!r}"
    )
    np.testing.assert_allclose(
        partial,
        _closed_form(0.65),
        rtol=1e-6,
        atol=0,
        err_msg=(
            "_weights_from_query(0.65) must be max_weight * 0.35 / 0.7 == 2.5 up to "
            f"float32 rounding, got {partial!r}"
        ),
    )
    np.testing.assert_allclose(
        _mapped_scalar(hyper, -0.65),
        -_closed_form(0.65),
        rtol=1e-6,
        atol=0,
        err_msg=(
            "_weights_from_query(-0.65) must be -2.5 up to float32 rounding (the map is "
            "odd)"
        ),
    )


def test_nan_in_is_nan_out(hyper):
    """A NaN CPPN output keeps a NaN weight; the dead-zone ``select`` does not mask it."""
    got = _mapped_scalar(hyper, float("nan"))
    assert np.isnan(got), (
        "_weights_from_query(NaN) must be NaN: the dead zone selects 0 but sign(NaN) * 0 "
        f"is NaN and the final multiply carries it, got {got!r}"
    )
    nan_reference = _documented_map(float("nan"))
    assert np.isnan(np.float32(nan_reference)), (
        f"the oracle's own transcription must be NaN for a NaN input, got {nan_reference!r}"
    )
    assert np.isnan(_closed_form(float("nan")))


def test_batch_rows_map_every_entry_elementwise(hyper):
    """A ``(P, Q)`` batch maps entry by entry, each row independently.

    The grid covers all five regimes -- exactly zero, both dead-zone ties, a value just
    outside it with both signs, a saturating magnitude and a NaN -- and every cell is
    compared *individually* so a failure names its coordinate.
    """
    grid = np.array(
        [
            [0.0, 0.3, -0.3, 0.35],
            [-0.35, 1.0, -1.0, 0.65],
            [-0.65, 0.5, -0.5, 0.299],
            [np.nan, -0.299, 0.7, -0.7],
        ],
        dtype=np.float32,
    )
    assert grid.shape == (POP_SIZE, CONNS_CNT), grid.shape

    got = _mapped(hyper, grid)
    assert got.shape == grid.shape, (
        f"_weights_from_query must preserve the (P, Q) shape {grid.shape}, got {got.shape}"
    )
    assert got.dtype == np.float32, (
        f"_weights_from_query must return float32 (the gene attribute dtype), got {got.dtype}"
    )

    reference = _documented_map(grid)
    for p in range(grid.shape[0]):
        for q in range(grid.shape[1]):
            np.testing.assert_allclose(
                got[p, q],
                reference[p, q],
                rtol=0,
                atol=0,
                equal_nan=True,
                err_msg=(
                    f"row {p}, column {q}: _weights_from_query({grid[p, q]!r}) must equal "
                    f"the documented map of that entry ({reference[p, q]!r}), got "
                    f"{got[p, q]!r}"
                ),
            )

    # A per-row statement of the same fact: row p is the map of row p alone, so re-running
    # one row through the kernel reproduces exactly that row of the batch.
    for p in range(grid.shape[0]):
        row = _mapped(hyper, grid[p : p + 1])
        np.testing.assert_allclose(
            row[0],
            got[p],
            rtol=0,
            atol=0,
            equal_nan=True,
            err_msg=(
                f"row {p} must not depend on the other rows: mapping it alone must "
                f"reproduce the batch's row, got {row[0]!r} vs {got[p]!r}"
            ),
        )


def test_output_shape_always_matches_the_input_shape(hyper):
    """The map is elementwise, so shape and dtype survive any ``(P, Q)`` -- including wide rows."""
    for shape in ((3, 4), (2, 1), (5, 2), (1, 6)):
        raw = np.linspace(-1.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
        got = _mapped(hyper, raw)
        assert got.shape == shape, (
            f"_weights_from_query must return the input shape {shape}, got {got.shape}"
        )
        assert got.dtype == np.float32, (
            f"_weights_from_query must return float32, got {got.dtype} for shape {shape}"
        )


def test_constructor_arguments_reach_the_map(hyper):
    """``weight_threshold`` / ``max_weight`` are actually threaded into the traced kernel.

    The custom instance (``threshold=0.5``, ``max_weight=2.0``) prunes ``0.4`` that the
    default instance keeps, and maps the same ``0.6`` to ``2.0 * 0.1 / 0.5 == 0.4``
    instead of ``5.0 * 0.3 / 0.7``.  Both instances are compared against their own
    transcription of the formula, so a constructor that silently dropped either argument
    fails on the values, not only on an attribute.
    """
    custom = make_hyper(weight_threshold=0.5, max_weight=2.0)
    assert hyper.weight_threshold == DEFAULT_THRESHOLD, (
        f"the default HyperNEAT must keep the documented weight_threshold "
        f"{DEFAULT_THRESHOLD}, got {hyper.weight_threshold!r}"
    )
    assert hyper.max_weight == DEFAULT_MAX_WEIGHT, (
        f"the default HyperNEAT must keep the documented max_weight "
        f"{DEFAULT_MAX_WEIGHT}, got {hyper.max_weight!r}"
    )
    assert (custom.weight_threshold, custom.max_weight) == (0.5, 2.0), (
        "the constructor must store the arguments it was given, got "
        f"({custom.weight_threshold!r}, {custom.max_weight!r})"
    )

    raw = np.array([[0.4, -0.4, 0.6, -1.0]], dtype=np.float32)
    got_default = _mapped(hyper, raw)
    got_custom = _mapped(custom, raw)

    np.testing.assert_allclose(
        got_default,
        _documented_map(raw, DEFAULT_THRESHOLD, DEFAULT_MAX_WEIGHT),
        rtol=0,
        atol=0,
        err_msg=(
            "the default map must use weight_threshold=0.3 / max_weight=5.0"
        ),
    )
    np.testing.assert_allclose(
        got_custom,
        _documented_map(raw, 0.5, 2.0),
        rtol=0,
        atol=0,
        err_msg=(
            "a HyperNEAT constructed with weight_threshold=0.5 / max_weight=2.0 must use "
            "those values in the map"
        ),
    )

    # Hand-stated consequences, independent of both transcriptions.
    assert got_custom[0, 0] == np.float32(0.0), (
        f"0.4 is inside the custom dead zone (|0.4| <= 0.5) so it must prune, got "
        f"{got_custom[0, 0]!r}"
    )
    assert got_default[0, 0] > 0.0, (
        f"0.4 is outside the default dead zone (|0.4| > 0.3) so it must survive, got "
        f"{got_default[0, 0]!r}"
    )
    np.testing.assert_allclose(
        got_custom[0, 2],
        0.4,
        rtol=1e-6,
        atol=0,
        err_msg=(
            "the custom map sends 0.6 to max_weight * (0.6 - 0.5) / 0.5 == 0.4, got "
            f"{got_custom[0, 2]!r}"
        ),
    )
    np.testing.assert_allclose(
        got_custom[0, 3],
        -2.0,
        rtol=1e-6,
        atol=0,
        err_msg=(
            "the custom map sends -1.0 to -max_weight == -2.0, got "
            f"{got_custom[0, 3]!r}"
        ),
    )
    assert got_default[0, 3] != got_custom[0, 3], (
        "the two instances must disagree on -1.0 (5.0 vs 2.0), so the constructor "
        f"arguments reach the map; both returned {got_default[0, 3]!r}"
    )


def test_harness_threads_a_state_into_a_decode_kernel(hyper):
    """``make_state`` + ``run_method(..., state=...)`` produce the ``(P, Q)`` query tensor.

    This is the harness's own smoke test (the module docstring's three rules): the state
    is a static host object, the payload and its leaves cross as graph inputs, and the
    result comes back as host numpy with the static ``(P, Q)`` shape.  Feeding that tensor
    through the weight map then agrees with the documented map, which is the chain the
    later decode tests need.
    """
    state = make_state(hyper, seed=0)
    payload = cppn_payload(state)
    assert payload.node_keys.shape[0] == hyper.pop_size == POP_SIZE, (
        "the CPPN payload must be batch-first at the algorithm's pop_size: expected "
        f"{POP_SIZE} rows, got {payload.node_keys.shape}"
    )

    query = run_method(hyper, "_query_substrate", payload, state=state)
    assert query.shape == (POP_SIZE, CONNS_CNT), (
        f"_query_substrate must return one query per (individual, substrate connection): "
        f"expected {(POP_SIZE, CONNS_CNT)}, got {query.shape}"
    )
    assert query.dtype == np.float32, (
        f"_query_substrate must return float32 raw CPPN outputs, got {query.dtype}"
    )

    weights = run_method(hyper, "_weights_from_query", query)
    np.testing.assert_allclose(
        weights,
        _documented_map(query),
        rtol=0,
        atol=0,
        equal_nan=True,
        err_msg=(
            "the weights decoded from a real CPPN population must be the documented map "
            "of the raw query tensor"
        ),
    )


def test_run_method_rejects_a_state_taking_method_without_a_state(hyper):
    """``run_method`` refuses to shift a state-taking method's arguments silently."""
    with pytest.raises(TypeError, match="takes 'state' as its first argument"):
        run_method(hyper, "transform")
