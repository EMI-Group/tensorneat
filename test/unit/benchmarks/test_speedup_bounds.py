"""Regression tests for ``protocol.speedup`` / ``protocol.speedup_bounds``.

``speedup_bounds`` is the range the report prints as ``Speed-up range (worst–best)``, and the
two bounds are deliberately not symmetric mirror images of one number:

* ``best`` is the best case *for the old side* — the fastest old repeat over the slowest new
  repeat (``old_min / new_max``), i.e. the pessimistic bound for the new side;
* ``worst`` is the slowest old repeat over the fastest new repeat
  (``old_max / new_min``), i.e. the optimistic bound for the new side.

Both are read from the per-repeat extremes ``aggregate`` recorded, never from the medians, and
both are ``None`` when either side lacks the extreme they need. ``speedup`` itself is
undefined (``None``) unless both values are finite and ``new > 0`` — the guard is
``new <= 0``, so a zero-scaled new side can never be reported as an infinite win.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.harness import protocol


def _side(low: float | None, median: float, high: float | None) -> dict:
    """One aggregated side: the per-repeat extremes plus an unrelated median."""
    return {
        "ms_per_generation": median,
        "ms_per_generation_min": low,
        "ms_per_generation_max": high,
    }


OLD = _side(50.0, 52.0, 54.0)
NEW = _side(20.0, 22.0, 25.0)


def test_best_is_the_fastest_old_repeat_over_the_slowest_new_repeat() -> None:
    """``best = old_min / new_max`` — 50 ms over 25 ms — not the median pair (52/22)."""
    bounds = protocol.speedup_bounds(OLD, NEW)
    assert bounds["best"] == pytest.approx(50.0 / 25.0)
    assert bounds["best"] != pytest.approx(52.0 / 22.0)


def test_worst_is_the_slowest_old_repeat_over_the_fastest_new_repeat() -> None:
    """``worst = old_max / new_min`` — 54 ms over 20 ms — the optimistic bound for the new side."""
    bounds = protocol.speedup_bounds(OLD, NEW)
    assert bounds["worst"] == pytest.approx(54.0 / 20.0)
    assert bounds["worst"] != pytest.approx(52.0 / 22.0)


def test_the_bounds_bracket_every_mixed_ratio() -> None:
    """A ratio built from one old and one new extreme lies between the two bounds."""
    bounds = protocol.speedup_bounds(OLD, NEW)
    assert bounds["best"] < bounds["worst"]
    assert bounds["best"] <= protocol.speedup(50.0, 20.0) <= bounds["worst"]
    assert bounds["best"] <= protocol.speedup(54.0, 25.0) <= bounds["worst"]


def test_the_median_never_feeds_a_bound() -> None:
    """A wild median changes nothing: only ``*_min``/``*_max`` are read."""
    huge_median = _side(50.0, 10_000.0, 54.0)
    assert protocol.speedup_bounds(huge_median, NEW) == protocol.speedup_bounds(OLD, NEW)


def test_a_single_repeat_collapses_both_bounds_to_the_same_ratio() -> None:
    """With no spread ``min == max`` on both sides, so best and worst coincide."""
    bounds = protocol.speedup_bounds(_side(50.0, 50.0, 50.0), _side(20.0, 20.0, 20.0))
    assert bounds["best"] == pytest.approx(2.5)
    assert bounds["worst"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("old", "expected"),
    [
        # Only the maximum: `worst` is computable, `best` (which needs the minimum) is not.
        ({"ms_per_generation_max": 54.0}, {"best": None, "worst": 2.7}),
        ({"ms_per_generation_min": 50.0}, {"best": 2.0, "worst": None}),
        ({}, {"best": None, "worst": None}),
        (None, {"best": None, "worst": None}),
    ],
)
def test_a_missing_old_extreme_kills_only_the_bound_that_reads_it(old, expected) -> None:
    """No recorded old extreme means no bound from it — and never a guessed one."""
    assert protocol.speedup_bounds(old, NEW) == {
        "best": None if expected["best"] is None else pytest.approx(expected["best"]),
        "worst": None if expected["worst"] is None else pytest.approx(expected["worst"]),
    }


@pytest.mark.parametrize(
    ("new", "expected"),
    [
        ({"ms_per_generation_max": 25.0}, {"best": 2.0, "worst": None}),
        ({"ms_per_generation_min": 20.0}, {"best": None, "worst": 2.7}),
        ({}, {"best": None, "worst": None}),
        (None, {"best": None, "worst": None}),
    ],
)
def test_a_missing_new_extreme_kills_only_the_bound_that_reads_it(new, expected) -> None:
    """The same on the new side: half a spread is not a range."""
    assert protocol.speedup_bounds(OLD, new) == {
        "best": None if expected["best"] is None else pytest.approx(expected["best"]),
        "worst": None if expected["worst"] is None else pytest.approx(expected["worst"]),
    }


@pytest.mark.parametrize(
    ("old_ms", "new_ms", "expected"),
    [
        (50.0, 25.0, 2.0),
        (50.0, 50.0, 1.0),
        (50.0, 0.0, None),
        (50.0, -1.0, None),
        (0.0, 25.0, 0.0),
        (None, 25.0, None),
        (50.0, None, None),
        (math.inf, 25.0, None),
        (50.0, math.nan, None),
    ],
)
def test_speedup_is_undefined_without_two_positive_finite_measurements(
    old_ms, new_ms, expected
) -> None:
    """``None`` unless both are finite and ``new > 0``; a zero new side is not an infinity."""
    assert protocol.speedup(old_ms, new_ms) == expected


def test_a_zero_new_extreme_kills_only_the_bound_that_reads_it() -> None:
    """``new <= 0`` poisons the bound it feeds and leaves the other one intact."""
    assert protocol.speedup_bounds(OLD, _side(0.0, 22.0, 25.0))["worst"] is None
    assert protocol.speedup_bounds(OLD, _side(0.0, 22.0, 25.0))["best"] == pytest.approx(2.0)
    assert protocol.speedup_bounds(OLD, _side(20.0, 22.0, 0.0))["best"] is None
    assert protocol.speedup_bounds(OLD, _side(20.0, 22.0, 0.0))["worst"] == pytest.approx(2.7)


def test_the_bounds_read_the_aggregated_row_it_produced() -> None:
    """End to end on fabricated repeats: the row ``aggregate`` wrote feeds the two bounds."""
    runs = [
        {"case": "xor_p100", "impl": "old", "backend": "jax", "seed": s, "status": "ok",
         "ms_per_generation": ms}
        for s, ms in ((42, 50.0), (43, 52.0), (44, 54.0))
    ] + [
        {"case": "xor_p100", "impl": "new", "backend": "numpy", "seed": s, "status": "ok",
         "ms_per_generation": ms}
        for s, ms in ((42, 20.0), (43, 22.0), (44, 25.0))
    ]
    old, new = protocol.aggregate(runs)
    assert protocol.speedup_bounds(old, new) == {
        "best": pytest.approx(50.0 / 25.0),
        "worst": pytest.approx(54.0 / 20.0),
    }
