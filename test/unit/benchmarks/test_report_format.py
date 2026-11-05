"""Regression tests for the generations-to-target cell (``report_format.reach``).

The cell has to carry three facts at once and never invent a fourth:

* a number when every repeat arrived at the target (``78``) or a ``>N`` bound when none did
  (``152`` recorded generations, none of them at the target);
* the reached/attempted annotation when only some repeats arrived (``78 (2/3)``, ``>152 (0/3)``);
* ``n/a (no informative target)`` for a case measured without a target — never a ``>N`` bound,
  because nothing was compared against a target at all.

The legacy form (a record without ``gens_to_target_all``/``target``) must render exactly as it
always did, which is why ``target`` defaults to the :data:`TARGET_UNKNOWN` sentinel rather than
to ``None``; :func:`reach` also rebuilds the median from the per-repeat values when the
aggregate is missing, so a cell never degrades to ``>N`` while a repeat did reach.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.harness import report_format

ALL_THREE = [78, 78, 78]
TWO_OF_THREE = [78, 78, None]
NONE_OF_THREE = [None, None, None]


def test_all_repeats_reaching_the_target_render_a_plain_number() -> None:
    """No annotation when nothing is missing: ``78``."""
    assert report_format.reach(78, ALL_THREE, 150) == "78"


@pytest.mark.parametrize(
    ("value", "all_values", "length", "target", "expected"),
    [
        # Some repeats arrived: the median plus how many of how many.
        (78, TWO_OF_THREE, 150, report_format.TARGET_UNKNOWN, "78 (2/3)"),
        # None arrived: the bound plus the zero tally.
        (None, NONE_OF_THREE, 152, report_format.TARGET_UNKNOWN, ">152 (0/3)"),
        # A target recorded as null suppresses the metric instead of bounding it.
        (None, NONE_OF_THREE, 152, None, "n/a (no informative target)"),
        # ... even when a numeric value survived in the record.
        (78, TWO_OF_THREE, 150, None, "n/a (no informative target)"),
        # Legacy record: no per-repeat list, no target key.
        (78, None, 150, report_format.TARGET_UNKNOWN, "78"),
        (None, None, 150, report_format.TARGET_UNKNOWN, ">150"),
        # A missing aggregate is rebuilt from the repeats that did arrive (median of 60, 78).
        (None, [60, 78, None], 150, report_format.TARGET_UNKNOWN, "69 (2/3)"),
        # One repeat is not a spread: no annotation at all.
        (78, [78], 150, report_format.TARGET_UNKNOWN, "78"),
        (None, [None], 150, report_format.TARGET_UNKNOWN, ">150"),
    ],
)
def test_reach_renders_the_documented_forms(value, all_values, length, target, expected) -> None:
    """Every cell shape the report can print, verbatim."""
    assert report_format.reach(value, all_values, length, target=target) == expected


def test_the_absent_target_sentinel_is_not_a_null_target() -> None:
    """``TARGET_UNKNOWN`` (key absent) keeps the legacy bound; ``None`` (measured without one) does not."""
    legacy = report_format.reach(78, [78, None, None], 150)
    suppressed = report_format.reach(78, [78, None, None], 150, target=None)
    assert legacy == "78 (1/3)"
    assert suppressed == "n/a (no informative target)"


def test_partial_reach_rounds_half_up_before_annotating() -> None:
    """The fractional median 77.5 renders as 78 (not 77) and keeps its tally."""
    assert report_format.reach(77.5, [77, 78, None], 150) == "78 (2/3)"
    assert report_format.gens_to_target(77.5, 150) == "78"


def test_a_rebuilt_median_of_an_even_number_of_repeats_averages_the_middle_pair() -> None:
    """Two arrived values 60/70 give 65; the cell must not fall back to the ``>N`` bound."""
    assert report_format.reach(None, [60, 70, None, None], 150) == "65 (2/4)"


@pytest.mark.parametrize(
    ("value", "length", "expected"),
    [
        (78, 150, "78"),
        (77.4, 150, "77"),
        (77.5, 150, "78"),
        (0.5, 150, "1"),
        (None, 152, ">152"),
        (math.inf, 152, ">152"),
        (-math.inf, 152, ">152"),
        (None, None, "n/a"),
        (None, 0, "n/a"),
        ("78", 150, "78"),
        # A non-numeric value with a known budget is the "nothing reached" case.
        ("n/a", 150, ">150"),
    ],
)
def test_gens_to_target_bounds_and_rounds_as_documented(value, length, expected) -> None:
    """Half-up rounding of a median, ``>N`` when nothing was reached, ``n/a`` without an N."""
    assert report_format.gens_to_target(value, length) == expected
