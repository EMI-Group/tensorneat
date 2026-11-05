"""Regression tests for ``protocol.aggregate``'s generations-to-target bookkeeping.

``aggregate`` collapses the per-repeat run records into the rows the report renders, and four
keys of that row answer a question the median alone cannot: ``target`` (the fitness target the
case was measured against, ``None`` when the case was measured without one),
``gens_to_target_all`` (one entry per repeat, ``None`` for a repeat that never arrived),
``gens_to_target_reached`` (how many repeats arrived) and ``gens_to_target_repeats`` (how many
were attempted). Those are what let a cell say ``78 (2/3)`` instead of only the median of the
repeats that happened to arrive.
``target`` comes from the first *ok* repeat in seed order and is never read from an
error/skipped record, so a failed run cannot lend the report a target the case did not use;
the medians stay over the ok repeats only, while ``gens_to_target_all`` keeps the per-repeat
spread of the group as recorded.
"""

from __future__ import annotations

from benchmarks.harness import protocol

TARGET = -0.1


def _run(
    seed: int,
    *,
    gens=-1,
    target: float | None = TARGET,
    status: str = protocol.STATUS_OK,
    ms: float = 52.0,
    case: str = "xor_p100",
    impl: str = "old",
    backend: str = "jax",
) -> dict:
    """One per-repeat run record with only the fields ``aggregate`` reads."""
    return {
        "case": case,
        "impl": impl,
        "backend": backend,
        "seed": seed,
        "status": status,
        "ms_per_generation": ms,
        "quality": {"target": target, "generations_to_target": None if gens < 0 else gens},
    }


def test_aggregate_writes_the_target_and_the_reached_repeat_count() -> None:
    """Two of three repeats arriving is ``target`` plus ``reached 2`` of ``repeats 3``."""
    row = protocol.aggregate([_run(42, gens=78), _run(43, gens=78), _run(44, gens=-1)])[0]
    assert row["target"] == TARGET
    assert row["gens_to_target_all"] == [78, 78, None]
    assert row["gens_to_target_reached"] == 2
    assert row["gens_to_target_repeats"] == 3
    # The keys the report read before this bookkeeping existed are unchanged.
    assert row["generations_to_target"] == 78.0
    assert row["n_repeats"] == 3


def test_gens_to_target_all_is_one_entry_per_repeat_in_seed_order() -> None:
    """The per-repeat list follows the seed order, not the order the records arrived in."""
    row = protocol.aggregate(
        [_run(44, gens=80), _run(42, gens=78), _run(43, gens=-1)]
    )[0]
    assert row["seeds"] == [42, 43, 44]
    assert row["gens_to_target_all"] == [78, None, 80]
    assert row["gens_to_target_reached"] == 2
    assert row["gens_to_target_repeats"] == len(row["gens_to_target_all"]) == 3


def test_the_median_stays_over_the_repeats_that_reached_the_target() -> None:
    """``generations_to_target`` is the median of the arrived repeats (84 from 78 and 90)."""
    row = protocol.aggregate([_run(42, gens=78), _run(43, gens=90), _run(44, gens=-1)])[0]
    assert row["generations_to_target"] == 84.0
    assert row["gens_to_target_all"] == [78, 90, None]
    assert row["gens_to_target_reached"] == 2


def test_a_non_ok_repeat_never_supplies_the_target() -> None:
    """An errored repeat's stored target is ignored; the first ok repeat supplies it."""
    row = protocol.aggregate(
        [
            _run(42, status=protocol.STATUS_ERROR, gens=-1, target=999.0),
            _run(43, gens=78, target=-0.05),
        ]
    )[0]
    assert row["target"] == -0.05
    assert row["n_errors"] == 1
    # The medians exclude the errored repeat, but it still counts as an attempted repeat.
    assert row["generations_to_target"] == 78.0
    assert row["gens_to_target_repeats"] == 2
    assert row["gens_to_target_reached"] == 1


def test_a_case_measured_without_a_target_records_target_none() -> None:
    """``None`` is a recorded answer ("no target"), distinct from a missing key."""
    row = protocol.aggregate(
        [
            _run(42, gens=-1, target=None),
            _run(43, gens=-1, target=None),
            _run(44, gens=-1, target=None),
        ]
    )[0]
    assert "target" in row
    assert row["target"] is None
    assert row["gens_to_target_all"] == [None, None, None]
    assert row["gens_to_target_reached"] == 0
    assert row["gens_to_target_repeats"] == 3
    assert row["generations_to_target"] is None


def test_gens_to_target_all_keeps_every_repeat_including_non_ok_ones() -> None:
    """``gens_to_target_all`` is the group's spread as recorded, not the ok-only median view.

    A repeat that errored after its quality curve existed still occupies its slot, so the
    ``(reached/total)`` annotation counts it in the denominator; only ``target`` and the
    medians are gated on ``status``.
    """
    row = protocol.aggregate(
        [
            _run(42, status=protocol.STATUS_ERROR, gens=3),
            _run(43, gens=-1),
        ]
    )[0]
    assert row["gens_to_target_all"] == [3, None]
    assert row["gens_to_target_reached"] == 1
    assert row["gens_to_target_repeats"] == 2
    assert row["generations_to_target"] is None


def test_every_group_carries_the_target_keys() -> None:
    """The keys exist per ``(case, impl, backend)`` group, including an all-error group."""
    rows = protocol.aggregate(
        [
            _run(42, case="xor_p100", impl="old", backend="jax"),
            _run(42, case="xor_p100", impl="new", backend="numpy", ms=30.0),
            _run(42, case="cartpole_p100", impl="new", backend="numpy", status=protocol.STATUS_ERROR),
        ]
    )
    assert [(row["case"], row["impl"]) for row in rows] == [
        ("xor_p100", "old"),
        ("xor_p100", "new"),
        ("cartpole_p100", "new"),
    ]
    for row in rows:
        assert {"target", "gens_to_target_reached", "gens_to_target_repeats"} <= set(row)
