"""The ``validate_schedule`` half of the ``tensorization`` suites — the host checker.

This module is the *second* of a two-module pair covering one source module,
``tensorneat.common.tensorization``: :mod:`unit.common.test_tensorization` keeps
the ``derive_schedule`` semantics (the field/shape/dtype contract, level order,
incoming-edge windows, ``edge_overflow``, the leaf key helpers and the
schedule-vs-oracle comparison) and owns every fixture used here, so the schedule
handed to the checker is always the kernel's own output, never a hand-built one.
Splitting the pair keeps each file under the repo's ~1000-line baseline without
dropping or weakening a single assertion.

What this half proves: ``validate_schedule`` accepts what ``derive_schedule``
produces — ``I_INF`` levels (a recurrent ring), absent padding rows and a lossy
window all included — and rejects every documented tampered invariant with a
``ConfigError`` naming the rule it broke.

Oracles (never the checker itself): the kernel's own schedules (the acceptance
half, plus the documented ``None`` return so a *repairing* checker cannot pass),
hand-copied fields edited through ``Transformed._replace`` with one cell/field
tampered so no other invariant can fire first, and the ``Raises`` list of
``validate_schedule``'s docstring as the expected message for each case.

Gotchas: ``edge_overflow`` is data, a negative count is however impossible and
must raise; ``input_slot`` may hold a sentinel but ``N`` itself is out of range
(the first illegal index an off-by-one derivation makes); ``in_conn`` and
``in_src`` must share their sentinel mask cell for cell. Tampered arrays are
copies — the module-scoped schedules are shared with the sibling module, so a
test never edits one in place.

Backend is ``"numpy"``, no ``jax``/``torch`` import, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

# Shared fixtures: defined (and documented) in the derive-schedule half, imported
# here so both modules exercise one schedule per topology rather than two builds.
# The payload fixtures come along because pytest resolves a fixture's dependencies
# in the namespace of the module that uses it, not of the module that defines it.
from .test_tensorization import (
    chain_payload,
    chain_schedule,
    chain_schedule_short_window,
    isolated_payload,
    isolated_schedule,
    ring_payload,
    ring_schedule,
    skip_payload,
    skip_schedule,
    sparse_payload,
    sparse_schedule,
)


# ---- validate_schedule: the host checker -----------------------------------------------------


def test_validate_schedule_accepts_the_kernel_output(
    common,
    chain_schedule,
    skip_schedule,
    isolated_schedule,
    ring_schedule,
    chain_schedule_short_window,
):
    """Invariant: ``validate_schedule`` returns ``None`` for every kernel schedule.

    Oracle: the kernel itself (the acceptance half) plus the documented ``None``
    return — a checker that returned a *repaired* schedule would hide the
    violation it was asked to report. The sample covers ``I_INF`` levels (the
    ring), absent rows (the isolated topology) and a lossy window (``K == 1``),
    because all three are valid schedules.
    """
    for schedule in (
        chain_schedule,
        skip_schedule,
        isolated_schedule,
        ring_schedule,
        chain_schedule_short_window,
    ):
        assert common.tensorization.validate_schedule(schedule) is None


def test_validate_schedule_rejects_in_conn_in_src_disagreement(common, chain_schedule):
    """Invariant: a padded cell in one table and a real index in the other is a ``ConfigError``.

    Oracle: a schedule tampered with ``Transformed._replace`` — one ``in_src``
    padding cell set to a real slot while ``in_conn`` keeps ``I_INF``. The tables
    must agree on every sentinel, so this must raise, naming the invariant.
    """
    schedule = chain_schedule
    assert schedule.in_conn[0, 0, 0] == common.I_INF  # node 0 has no incoming edge

    tampered = schedule.in_src.copy()
    tampered[0, 0, 0] = 0  # a real slot where in_conn says I_INF
    with pytest.raises(common.ConfigError, match="in_conn and in_src"):
        common.tensorization.validate_schedule(schedule._replace(in_src=tampered))


def test_validate_schedule_rejects_i_inf_level_with_a_finite_topo_key(common, sparse_schedule):
    """Invariant: ``level == I_INF`` implies ``topo_key == I_INF``; a finite key raises.

    Oracle: the documented pair ``topo_key = level * N + slot`` plus its sentinel —
    an unavailable node cannot be given a place in the sweep. Tampered by giving
    one absent row ``topo_key == 0``, a place in the very first level.
    """
    schedule = sparse_schedule
    dead_rows = np.flatnonzero(schedule.node_level[0] == common.I_INF)
    assert dead_rows.size, "the sparse fixture must have an I_INF level to tamper with"

    tampered = schedule.topo_key.copy()
    tampered[0, int(dead_rows[0])] = 0
    with pytest.raises(
        common.ConfigError, match="I_INF node_level must pair with an I_INF topo_key"
    ):
        common.tensorization.validate_schedule(schedule._replace(topo_key=tampered))


def test_validate_schedule_rejects_negative_edge_overflow(common, chain_schedule):
    """Invariant: a negative ``edge_overflow`` is a ``ConfigError``.

    Oracle: the field's meaning (a *count* of dropped edges); a negative count is
    impossible for any derivation, so it always means a corrupted or hand-built
    schedule, which must not reach a rebuild decision.
    """
    schedule = chain_schedule
    assert np.all(schedule.edge_overflow >= 0)

    tampered = schedule.edge_overflow.copy()
    tampered[0] = -1
    with pytest.raises(common.ConfigError, match="cannot be negative"):
        common.tensorization.validate_schedule(schedule._replace(edge_overflow=tampered))


def test_validate_schedule_rejects_an_out_of_range_input_slot(common, chain_schedule):
    """Invariant: an ``input_slot`` outside ``[0, N)`` — and not ``I_INF`` — is a ``ConfigError``.

    Oracle: the documented "slots point inside ``[0, N)`` or are the sentinel";
    ``N`` itself is out of range (the first illegal index), the mistake an
    off-by-one derivation makes.
    """
    schedule = chain_schedule
    tampered = schedule.input_slot.copy()
    tampered[0, 0] = schedule.node_level.shape[1]
    with pytest.raises(common.ConfigError, match="input_slot must hold slots"):
        common.tensorization.validate_schedule(schedule._replace(input_slot=tampered))


def _tamper_in_valid(schedule, i_inf):
    """Mark a padded window cell live while ``in_conn`` keeps ``I_INF``."""
    assert not schedule.in_valid[0, 0, 0]
    tampered = schedule.in_valid.copy()
    tampered[0, 0, 0] = True
    return schedule._replace(in_valid=tampered)


def _tamper_live_level_with_i_inf_key(schedule, i_inf):
    """Blank the ``topo_key`` of a node that *does* have a finite level."""
    node = int(np.flatnonzero(schedule.node_level[0] != i_inf)[-1])
    tampered = schedule.topo_key.copy()
    tampered[0, node] = i_inf
    return schedule._replace(topo_key=tampered)


def _tamper_topo_key_formula(schedule, i_inf):
    """Shift one live ``topo_key`` off ``node_level * N + slot``."""
    node = int(np.flatnonzero(schedule.node_level[0] != i_inf)[-1])
    assert schedule.topo_key[0, node] != i_inf
    tampered = schedule.topo_key.copy()
    tampered[0, node] = int(tampered[0, node]) + 1
    return schedule._replace(topo_key=tampered)


def _tamper_field_shape(schedule, i_inf):
    """Give the ``(P, N)`` node level the wrong node count."""
    return schedule._replace(node_level=schedule.node_level[:, :1])


def _tamper_field_dtype(schedule, i_inf):
    """Turn an integer field into a float array."""
    return schedule._replace(edge_overflow=schedule.edge_overflow.astype(np.float64))


@pytest.mark.parametrize(
    "tamper, message",
    [
        (_tamper_in_valid, "in_valid must be exactly in_conn != I_INF"),
        (
            _tamper_live_level_with_i_inf_key,
            "a finite node_level must pair with a finite topo_key",
        ),
        (_tamper_topo_key_formula, "topo_key must equal node_level \\* N \\+ slot"),
        (_tamper_field_shape, "node_level has shape"),
        (_tamper_field_dtype, "edge_overflow must be an integer array"),
    ],
    ids=["in_valid", "live-level-dead-key", "topo-key-formula", "shape", "dtype"],
)
def test_validate_schedule_rejects_each_remaining_tampered_invariant(
    common, chain_schedule, tamper, message
):
    """Invariant: each remaining documented rejection is a ``ConfigError`` naming its rule.

    Oracle: the ``Raises`` list of :func:`validate_schedule` — ``in_valid``
    disagreeing with ``in_conn != I_INF``, a live level paired with a dead
    ``topo_key``, a live ``topo_key`` off ``node_level * N + slot``, a field of the
    wrong shape, and a non-integer field. Each case tampers with one cell/field of
    a kernel-produced schedule, so no other invariant can fire first.
    """
    tampered = tamper(chain_schedule, common.I_INF)
    with pytest.raises(common.ConfigError, match=message):
        common.tensorization.validate_schedule(tampered)
