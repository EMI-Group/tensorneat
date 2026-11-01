"""Unit tests for spawn allocation in ``SpeciesController`` — the species budget.

What this module proves
-----------------------
Spawn allocation is the one place where a float score becomes an *integer* number
of children, so the whole module is about the documented algebra of that
conversion rather than about "nice" numbers:

* ``sum(spawn) == pop_size`` **exactly**, in both allocators
  (``_cal_spawn_numbers_by_rank`` / ``_cal_spawn_numbers_by_fitness``), for every
  damping rate and for 1, 2, 3 and 4 live slots out of 4 — the residual
  correction is the only reason the rounding gap cannot silently shrink (or
  inflate) the population;
* a dead slot (``isnan(species_keys)``) gets exactly ``0`` in both modes, even
  when it carries a stale (finite, large) ``member_count`` or a stale
  ``best_fitness``, so a killed species cannot buy its way back in through the
  damping term or through a fitness reduction;
* every live slot is floored at ``min_species_size`` when the budget allows it;
* damping is the documented mixture ``previous + (target - previous) * rate``
  with ``previous = member_count``: ``rate = 0`` reproduces the previous
  allocation, ``rate = 1`` the raw target, and ``0.5`` is strictly between them;
* linear ranking is monotone over the live slots of a *sorted* table (slot order
  is the fitness rank, a precondition established by ``update_species``);
* the integerisation step never produces a negative or non-finite spawn, from
  ``0``/``NaN`` ``member_count`` inputs, in ``int32``;
* the two building blocks are pinned: ``_prefix_sum_exclusive`` against
  ``np.cumsum`` with a ``0`` head, and ``_block_layout`` as the bijection from
  population slots onto ``(block, offset)`` pairs.

Oracles used here (never the implementation)
--------------------------------------------
:func:`spawn_target` is a plain-numpy transcription of the documented target of
each allocator, :func:`damped_values` the documented damping/floor/mask pipeline,
and :func:`expected_spawn` the two composed into the documented spawn vector.
They are formulas — no gather/mask machinery, no triangular ``dot``, no
``argmax`` over a reversed mask — so they cannot mimic a data-movement bug in the
kernel they check.
:func:`expected_block_layout` is a Python loop over the documented
``loc(p) = the last i with spawn[i] > 0 and E[i] <= p`` definition.
:func:`exclusive_prefix` pins the prefix sum's own definition.

Gotchas this module encodes
---------------------------
* The residual ``pop_size - sum(spawn)`` is added to the **first live slot**
  (``argmax`` of the liveness mask) by one ``scatter``; after the descending
  species sort of ``update_species`` that is slot ``0``.  The leading-dead table
  of :func:`test_the_residual_lands_on_the_first_live_slot` is the case where the
  two differ, so "first live" cannot degrade into "slot ``0``".
* The residual can be *negative*: the ``min_species_size`` floor is applied
  before the total is known, so a four-live table with ``min_species_size = 2``
  overflows and the correction subtracts from the head slot.  Conservation, not
  "the residual is positive", is the invariant.
* An all-dead table (every key ``NaN``) still returns ``sum(spawn) == pop_size``:
  the residual has nowhere else to go and lands on slot ``0``, which the kernel
  documents explicitly.  Only the *mixed* tables are asserted to keep dead slots
  at ``0``.
* ``_cal_spawn_numbers_by_rank`` does **not** sort: ``rank_score = n - i`` reads
  the slot index as the rank.  A rank-mode table is therefore built with its live
  slots first (the post-sort layout); a mid-table dead slot is still exercised, as
  an interleaved table, but only against the oracle — never with a monotonicity
  claim.
* ``spawn`` is ``int32``, so "finite" cannot be checked on the output; it is
  checked on the oracle's pre-cast vector plus an ``int32`` dtype check on the
  kernel's.

Backend is ``"numpy"`` throughout; no optional dependency, no ``jax``/``torch``,
no network.
"""

from __future__ import annotations

import numpy as np
import pytest

#: Method name of each allocator, keyed by ``species_number_calculate_by``.
ALLOCATORS = {
    "rank": "_cal_spawn_numbers_by_rank",
    "fitness": "_cal_spawn_numbers_by_fitness",
}

#: ``(species_keys, best_fitness, member_count)`` per live count over ``S = 4``.
#: Live slots come first in ``species_keys`` (the rank allocator's sorted
#: precondition), dead slots carry ``NaN`` in the two float payloads, and the live
#: ``member_count`` always sums to ``pop_size = 8`` — the real invariant of a
#: partition of the population.
LIVE_TABLES = {
    1: (
        [0.0, np.nan, np.nan, np.nan],
        [0.5, np.nan, np.nan, np.nan],
        [8.0, np.nan, np.nan, np.nan],
    ),
    2: (
        [0.0, 1.0, np.nan, np.nan],
        [0.5, 0.2, np.nan, np.nan],
        [5.0, 3.0, np.nan, np.nan],
    ),
    3: (
        [0.0, 1.0, 2.0, np.nan],
        [0.5, 0.2, 0.9, np.nan],
        [3.0, 3.0, 2.0, np.nan],
    ),
    4: (
        [0.0, 1.0, 2.0, 3.0],
        [0.5, 0.2, 0.9, 0.1],
        [2.0, 2.0, 2.0, 2.0],
    ),
}


# --------------------------------------------------------------------------- #
# numpy oracles
# --------------------------------------------------------------------------- #


def live_mask(species_keys):
    """``(S,)`` bool: a slot is live iff its species key is not ``NaN``.

    Args:
        species_keys: ``(S,)`` float array of species keys.

    Returns:
        ``(S,)`` bool, the only dead-slot predicate the algorithm uses.
    """
    return ~np.isnan(np.asarray(species_keys, dtype=np.float32))


def spawn_target(species_keys, best_fitness, *, mode, pop_size):
    """Transcribe the documented *pre-damping* spawn target into plain numpy.

    ``"rank"`` (linear ranking over the live slots of a sorted table): with
    ``n = sum(live)`` the rank score of slot ``i`` is ``n - i``, the denominator is
    the triangular number ``(n + 1) * n / 2`` floored at ``1`` (the legacy code
    divided by zero on an all-dead table) and ``target = floor(rank_score /
    denominator * pop_size)``.  ``best_fitness`` is not consulted at all.

    ``"fitness"`` (proportional): ``f' = f - min_live(f) + 1`` so the worst live
    species still has a positive weight, non-finite weights are cleaned to ``0``
    (a freshly created species starts at ``-inf`` best fitness), ``rate = weight /
    max(sum(weight), 1)`` and ``target = floor(rate * pop_size)``.  Dead slots get
    weight ``0`` *without* reading their ``best_fitness``, which is where the
    masked ``min``/``sum`` fix lives.

    Args:
        species_keys: ``(S,)`` float keys; ``NaN`` marks a dead slot.
        best_fitness: ``(S,)`` float, read by ``"fitness"`` mode only.
        mode: ``"rank"`` or ``"fitness"``.
        pop_size: Static population size the targets are scaled to.

    Returns:
        ``(S,)`` float32, the raw target per slot.

    Raises:
        AssertionError: If ``mode`` is neither known allocator.

    Notes:
        The arithmetic is float32 exactly like the kernel's, so a target that
        lands on an integer cannot be floored differently by one ULP of oracle
        precision.  The formula still contains none of the kernel's machinery.
    """
    keys = np.asarray(species_keys, dtype=np.float32)
    live = live_mask(keys)
    n_live = int(live.sum())

    if mode == "rank":
        rank_score = (n_live - np.arange(keys.shape[0])).astype(np.float32)
        denominator = np.float32(max((n_live + 1) * n_live / 2.0, 1.0))
        return np.floor(rank_score / denominator * np.float32(pop_size))

    assert mode == "fitness", f"unknown allocator mode {mode!r}"
    fitness = np.asarray(best_fitness, dtype=np.float32)
    lowest = np.min(np.where(live, fitness, np.float32(np.inf)))
    shifted = fitness - lowest + np.float32(1.0)
    cleaned = np.where(np.isfinite(shifted), shifted, np.float32(0.0))
    weight = np.where(live, cleaned, np.float32(0.0)).astype(np.float32)
    total = np.float32(max(float(weight.sum()), 1.0))
    return np.floor(weight / total * np.float32(pop_size))


def damped_values(species_keys, member_count, target, *, change_rate, min_species_size):
    """Transcribe the documented damping / floor / mask pipeline (pre-cast).

    The order is the kernel's and matters:

    1. ``previous = where(isnan(member_count), 0, member_count)`` — a dead slot's
       ``NaN`` count must not create a ``NaN`` here;
    2. ``damped = previous + (target - previous) * change_rate``;
    3. ``damped = where(isnan(damped), 0, damped)`` (the kernel's NaN guard runs
       *before* the floor, so a NaN damped value ends up at the floor, not at
       ``0``);
    4. ``damped = maximum(damped, min_species_size)`` — the floor is applied while
       the slot is still unmasked;
    5. ``damped = where(live, damped, 0)`` — *then* dead slots are forced to
       ``0.0``, which is what keeps them out of the budget without relying on
       float32 → int32 cast semantics.

    Args:
        species_keys: ``(S,)`` float keys; ``NaN`` marks a dead slot.
        member_count: ``(S,)`` float, the previous allocation (``NaN`` on dead slots).
        target: ``(S,)`` float from :func:`spawn_target`.
        change_rate: ``spawn_number_change_rate`` in ``[0, 1]``.
        min_species_size: Lower bound of a live slot's damped value.

    Returns:
        ``(S,)`` float32, the pre-cast spawn vector.
    """
    keys = np.asarray(species_keys, dtype=np.float32)
    live = live_mask(keys)
    previous = np.where(
        np.isnan(np.asarray(member_count, dtype=np.float32)),
        np.float32(0.0),
        np.asarray(member_count, dtype=np.float32),
    )
    damped = previous + (np.asarray(target, dtype=np.float32) - previous) * np.float32(
        change_rate
    )
    damped = np.where(np.isnan(damped), np.float32(0.0), damped)
    damped = np.maximum(damped, np.float32(min_species_size))
    return np.where(live, damped, np.float32(0.0)).astype(np.float32)


def integerise(damped, species_keys, *, pop_size):
    """Floor, cast to ``int32`` and correct the total on the first live slot.

    Args:
        damped: ``(S,)`` float from :func:`damped_values`.
        species_keys: ``(S,)`` float keys; ``NaN`` marks a dead slot.
        pop_size: Static population size the spawn must sum to.

    Returns:
        ``(S,)`` ``int32`` with ``sum == pop_size``; the residual lands on the
        first live slot (slot ``0`` when the whole table is dead).

    Notes:
        ``np.argmax`` on a bool array yields the first ``True`` and ``0`` for an
        all-``False`` mask, which is exactly the kernel's ``head_slot`` rule.
    """
    spawn = np.floor(np.asarray(damped, dtype=np.float32)).astype(np.int32)
    head = int(np.argmax(live_mask(species_keys)))
    spawn[head] += np.int32(pop_size - int(spawn.sum()))
    return spawn


def expected_spawn(
    species_keys,
    best_fitness,
    member_count,
    *,
    mode,
    pop_size,
    change_rate,
    min_species_size,
):
    """The documented spawn vector of one allocator, as a formula.

    Args:
        species_keys: ``(S,)`` float keys; ``NaN`` marks a dead slot.
        best_fitness: ``(S,)`` float, read by ``"fitness"`` mode only.
        member_count: ``(S,)`` float, the previous allocation.
        mode: ``"rank"`` or ``"fitness"``.
        pop_size: Static population size the spawn must sum to.
        change_rate: ``spawn_number_change_rate`` in ``[0, 1]``.
        min_species_size: Lower bound of a live slot's spawn count.

    Returns:
        ``(S,) int32`` equal to what the controller method must return.
    """
    target = spawn_target(species_keys, best_fitness, mode=mode, pop_size=pop_size)
    damped = damped_values(
        species_keys,
        member_count,
        target,
        change_rate=change_rate,
        min_species_size=min_species_size,
    )
    return integerise(damped, species_keys, pop_size=pop_size)


def exclusive_prefix(values):
    """``out[i] = sum(values[:i])``, with ``out[0] == 0`` — the definition.

    Args:
        values: ``(n,)`` float array.

    Returns:
        ``(n,)`` float array, ``np.cumsum`` shifted right by one slot.
    """
    cumulative = np.cumsum(np.asarray(values, dtype=np.float32))
    return np.concatenate((np.zeros(1, dtype=np.float32), cumulative[:-1]))


def expected_block_layout(spawn):
    """Compute ``(loc, idx_in_block)`` from the documented rule, by hand.

    The authoritative rule is ``loc(p) = the **last** i with ``spawn[i] > 0`` and
    ``E[i] <= p``, and ``idx_in_block(p) = p - E[loc(p)]``, where ``E`` is the
    exclusive prefix sum of ``spawn``.  "Last qualifying block" — *not* "number of
    qualifying blocks": a live species can legitimately receive ``spawn == 0`` in
    the middle of the table, and a count would then point one block too early.

    Args:
        spawn: ``(S,)`` int array summing to ``P``; non-positive entries are absent
            blocks and consume no population slot.

    Returns:
        tuple: ``(loc, idx_in_block)``, both ``(P,)`` ``int64`` arrays.

    Notes:
        ``./test_parent_selection.py`` carries a local copy of this function (a test
        module must stay importable on its own); both copies state the same formula
        and change together.
    """
    spawn = np.asarray(spawn, dtype=np.int64)
    exclusive = np.concatenate(([0], np.cumsum(spawn)[:-1]))
    total = int(spawn.sum())
    loc = np.zeros(total, dtype=np.int64)
    offset = np.zeros(total, dtype=np.int64)
    for p in range(total):
        blocks = [i for i in range(spawn.shape[0]) if spawn[i] > 0 and exclusive[i] <= p]
        loc[p] = blocks[-1]
        offset[p] = p - exclusive[loc[p]]
    return loc, offset


# --------------------------------------------------------------------------- #
# execution helpers
# --------------------------------------------------------------------------- #


def allocate(species_kernel, mode, bookkeeping, controller):
    """Run one allocator over ``bookkeeping`` and return the host ``(S,) int32`` spawn.

    Args:
        species_kernel: The graph runner fixture.
        mode: ``"rank"`` or ``"fitness"``, the key of :data:`ALLOCATORS`.
        bookkeeping: The ``SpeciesState`` table (also the oracle's input).
        controller: The ``SpeciesController`` whose configuration the oracle reads.

    Returns:
        Host numpy ``(S,) int32``.

    Notes:
        The controller is passed as a host-side closure value; the table travels as
        graph inputs, so the oracle and the kernel consume byte-identical tables.
    """
    return species_kernel(ALLOCATORS[mode], bookkeeping, controller=controller)


def oracle(mode, bookkeeping, controller):
    """:func:`expected_spawn` for ``bookkeeping`` under ``controller``'s configuration.

    Args:
        mode: ``"rank"`` or ``"fitness"``.
        bookkeeping: The ``SpeciesState`` table.
        controller: The controller supplying ``pop_size``, the damping rate and
            ``min_species_size``.

    Returns:
        ``(S,) int32``.
    """
    return expected_spawn(
        bookkeeping.species_keys,
        bookkeeping.best_fitness,
        bookkeeping.member_count,
        mode=mode,
        pop_size=controller.pop_size,
        change_rate=controller.spawn_number_change_rate,
        min_species_size=controller.min_species_size,
    )


def damped_for(mode, bookkeeping, controller):
    """:func:`damped_values` for ``bookkeeping`` under ``controller``'s configuration.

    Args:
        mode: ``"rank"`` or ``"fitness"``.
        bookkeeping: The ``SpeciesState`` table.
        controller: The controller supplying ``pop_size``, the damping rate and
            ``min_species_size``.

    Returns:
        ``(S,)`` float32, the pre-cast spawn vector the kernel floors.

    Notes:
        Exposed separately from :func:`oracle` so a test can assert on the *float*
        value before the ``int32`` cast (a spawn vector cannot carry ``NaN``) and on
        the pre-correction total (which is what makes the residual negative).
    """
    target = spawn_target(
        bookkeeping.species_keys,
        bookkeeping.best_fitness,
        mode=mode,
        pop_size=controller.pop_size,
    )
    return damped_values(
        bookkeeping.species_keys,
        bookkeeping.member_count,
        target,
        change_rate=controller.spawn_number_change_rate,
        min_species_size=controller.min_species_size,
    )


def table_from(species_state, live_count):
    """Build the canonical ``live_count``-live table (see :data:`LIVE_TABLES`).

    Args:
        species_state: The ``SpeciesState`` factory fixture.
        live_count: Number of live slots, ``1..4``.

    Returns:
        A ``SpeciesState`` with live slots first and ``NaN`` in every dead slot.
    """
    keys, best_fitness, members = LIVE_TABLES[live_count]
    return species_state(
        species_keys=np.array(keys, dtype=np.float32),
        best_fitness=np.array(best_fitness, dtype=np.float32),
        member_count=np.array(members, dtype=np.float32),
        last_improved=np.where(
            np.isnan(np.array(keys, dtype=np.float32)), np.nan, 0.0
        ).astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# conservation of the population budget
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
@pytest.mark.parametrize("change_rate", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("live_count", sorted(LIVE_TABLES))
def test_allocation_corrects_the_total_to_pop_size_exactly(
    species_state, controller, species_kernel, mode, change_rate, live_count
):
    """Invariant: ``sum(spawn) == pop_size`` exactly, for every rate, mode and live count.

    Oracle: ``int(spawn.sum()) == pop_size``, ``spawn.dtype == np.int32`` and
    equality with :func:`expected_spawn` — over 1, 2, 3 and 4 live slots out of 4
    and the three damping rates, so the correction is exercised whenever rounding
    down (plus the ``min_species_size`` floor) leaves budget on the table.  The
    opposite sign — a floor that overflows ``pop_size``, so the residual has to be
    subtracted — is the subject of
    :func:`test_a_live_species_receives_at_least_min_species_size`.
    A missing correction shrinks or inflates the population every generation,
    which is a silent divergence from the pre-rewrite library, not cosmetic
    rounding.
    """
    ctrl = controller(spawn_number_change_rate=change_rate)
    bookkeeping = table_from(species_state, live_count)

    spawn = allocate(species_kernel, mode, bookkeeping, ctrl)

    assert spawn.dtype == np.int32, spawn.dtype
    assert int(spawn.sum()) == ctrl.pop_size, spawn
    np.testing.assert_array_equal(oracle(mode, bookkeeping, ctrl), spawn)


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
@pytest.mark.parametrize("dead_member_count", [np.nan, 7.0, 1e6])
@pytest.mark.parametrize("layout", ["trailing", "interleaved"])
def test_dead_slots_receive_no_spawn(
    species_state, controller, species_kernel, mode, dead_member_count, layout
):
    """Invariant: a slot whose key is ``NaN`` gets exactly ``0``, in both allocators.

    Oracle: the dead slots must be ``0`` and the live ones must sum to ``pop_size``,
    over two layouts (dead slots trailing — the post-sort shape — and dead slots
    interleaved with live ones) and three stale dead-slot ``member_count`` values
    (``NaN``, ``7.0``, ``1e6``).  The large value is the interesting one: the damping
    term ``previous + (target - previous) * rate`` is computed *before* the liveness
    mask, so only the explicit ``where(live, damped, 0)`` keeps a killed species
    from inheriting population slots through a stale count.  The full vector is
    compared against :func:`expected_spawn` as well, so "0 for the dead slot" cannot
    be satisfied by stealing slots from a live one.
    """
    if layout == "trailing":
        keys = [0.0, 1.0, np.nan, np.nan]
        best_fitness = [0.5, 0.2, np.nan, np.nan]
        members = [5.0, 3.0, dead_member_count, dead_member_count]
    else:
        keys = [0.0, np.nan, 1.0, np.nan]
        best_fitness = [0.5, np.nan, 0.2, np.nan]
        members = [5.0, dead_member_count, 3.0, dead_member_count]

    ctrl = controller()
    bookkeeping = species_state(
        species_keys=np.array(keys, dtype=np.float32),
        best_fitness=np.array(best_fitness, dtype=np.float32),
        member_count=np.array(members, dtype=np.float32),
    )

    spawn = allocate(species_kernel, mode, bookkeeping, ctrl)

    dead = ~live_mask(keys)
    np.testing.assert_array_equal(
        spawn[dead], np.zeros(int(dead.sum()), np.int32), err_msg=f"spawn={spawn}"
    )
    assert int(spawn.sum()) == ctrl.pop_size, spawn
    np.testing.assert_array_equal(oracle(mode, bookkeeping, ctrl), spawn)


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
def test_the_residual_lands_on_the_first_live_slot(
    species_state, controller, species_kernel, mode
):
    """Invariant: with slot ``0`` dead, the residual goes to the first *live* slot.

    Oracle: a leading-dead table (``keys = [NaN, 0, 1, NaN]``) at ``rate = 1``,
    where the pre-correction total is ``7 < 8`` and the correction is ``+1``.  It
    must land on slot ``1``: both dead slots stay ``0`` and the vector matches
    :func:`expected_spawn`, whose :func:`integerise` applies the same ``argmax``
    rule.  An ``argmax``-free "always slot ``0``" correction would hand the budget
    to a killed species, which is exactly the corruption the liveness mask prevents.
    """
    ctrl = controller(spawn_number_change_rate=1.0)
    bookkeeping = species_state(
        species_keys=np.array([np.nan, 0.0, 1.0, np.nan], dtype=np.float32),
        best_fitness=np.array([np.nan, 0.5, 0.2, np.nan], dtype=np.float32),
        member_count=np.array([np.nan, 0.0, 0.0, np.nan], dtype=np.float32),
    )
    assert np.floor(damped_for(mode, bookkeeping, ctrl)).sum() < ctrl.pop_size, (
        "fixture must leave budget over so the residual is non-zero"
    )

    spawn = allocate(species_kernel, mode, bookkeeping, ctrl)

    np.testing.assert_array_equal(oracle(mode, bookkeeping, ctrl), spawn)
    np.testing.assert_array_equal(
        spawn[[0, 3]], np.zeros(2, np.int32), err_msg=f"spawn={spawn}"
    )
    assert int(spawn.sum()) == ctrl.pop_size, spawn


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
def test_a_live_slot_with_a_nan_member_count_starts_from_zero(
    species_state, controller, species_kernel, mode
):
    """Invariant: ``previous = 0`` for a ``NaN`` ``member_count``, live slot or not.

    Oracle: slot ``1`` is live with ``member_count = NaN`` at ``rate = 1``, so its
    spawn is the raw target and not the ``min_species_size`` floor that a ``NaN``
    damped value falls through to (the kernel's ``isnan(damped)`` guard runs *before*
    the floor).  The two differ, so this pins the ``previous`` guard that protects
    the damping term from a ``NaN`` count — the count is read before the liveness
    mask, which is why only this table can observe it.
    """
    ctrl = controller(spawn_number_change_rate=1.0)
    bookkeeping = species_state(
        species_keys=np.array([0.0, 1.0, np.nan, np.nan], dtype=np.float32),
        best_fitness=np.array([0.5, 0.2, np.nan, np.nan], dtype=np.float32),
        member_count=np.array([2.0, np.nan, np.nan, np.nan], dtype=np.float32),
    )
    target = np.floor(damped_for(mode, bookkeeping, ctrl))
    assert target[1] > ctrl.min_species_size, target

    spawn = allocate(species_kernel, mode, bookkeeping, ctrl)

    np.testing.assert_array_equal(oracle(mode, bookkeeping, ctrl), spawn)
    assert int(spawn[1]) == int(target[1]), (
        f"slot 1 must start from 0 and land on its target {target[1]}, got {spawn[1]}"
    )


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
@pytest.mark.parametrize("live_count", [3, 4])
def test_a_live_species_receives_at_least_min_species_size(
    species_state, controller, species_kernel, mode, live_count
):
    """Invariant: every live slot is floored at ``min_species_size`` when the budget allows.

    Oracle: ``min_species_size = 2`` at ``rate = 1`` (the raw target, so nothing but
    the floor stands between the target and the spawn) with ``3 * 2 <= 8`` (three
    live slots, residual positive) and with four live slots, where the floor
    *overflows* the budget — the pre-cast total is ``3 + 2 + 2 + 2 = 9 > 8``, so the
    correction has to subtract from the head slot and the result must be
    ``[2, 2, 2, 2]``.  Both tables start from ``member_count = 0`` and a best fitness
    spread (``[10, 5, 0, -5]``) whose targets round to ``0`` for the worst slot, so
    the floor — not the target — is what supplies the spawn.  Every live slot must
    come out ``>= 2`` and the sum must still be exactly ``pop_size``: the floor is
    what keeps a small but successful species alive, and the two cases together show
    it can only do so by pulling the head slot down.
    """
    keys = [0.0, 1.0, 2.0, 3.0][:live_count] + [np.nan] * (4 - live_count)
    best_fitness = [10.0, 5.0, 0.0, -5.0][:live_count] + [np.nan] * (4 - live_count)
    members = [0.0] * live_count + [np.nan] * (4 - live_count)
    ctrl = controller(min_species_size=2, spawn_number_change_rate=1.0)
    bookkeeping = species_state(
        species_keys=np.array(keys, dtype=np.float32),
        best_fitness=np.array(best_fitness, dtype=np.float32),
        member_count=np.array(members, dtype=np.float32),
    )
    damped = damped_for(mode, bookkeeping, ctrl)
    if live_count == 4:
        assert np.floor(damped).sum() > ctrl.pop_size, (
            "fixture must overflow the budget so the residual is negative, got "
            f"{np.floor(damped)}"
        )

    spawn = allocate(species_kernel, mode, bookkeeping, ctrl)

    live = live_mask(bookkeeping.species_keys)
    assert np.all(spawn[live] >= ctrl.min_species_size), spawn
    assert int(spawn.sum()) == ctrl.pop_size, spawn
    np.testing.assert_array_equal(oracle(mode, bookkeeping, ctrl), spawn)


# --------------------------------------------------------------------------- #
# the damping rate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
@pytest.mark.parametrize("change_rate", [0.0, 0.5, 1.0])
def test_damping_follows_previous_plus_the_rate_times_the_gap(
    species_state, controller, species_kernel, mode, change_rate
):
    """Invariant: ``damped = previous + (target - previous) * rate``, per live slot.

    Oracle: the law applied slot by slot.  ``rate = 0`` must reproduce
    ``member_count``, ``rate = 1`` the raw target from :func:`spawn_target`, and
    ``rate = 0.5`` the mixture of the two — each floored at
    ``min_species_size``, masked on dead slots, and corrected on the head slot by
    the residual.  The fixture ``member_count = [2, 6]`` against the targets
    ``[5, 2]`` (rank) and ``[4, 3]`` (fitness) is chosen so that no two of the three
    rates collapse onto the same integer vector, and so that the two mixed cases
    leave a slot of budget over (``[3, 4]`` and ``[5, 2]`` / ``[4, 3]`` sum to 7)
    while ``rate = 0`` needs no correction at all.

    Notes:
        ``change_rate`` is read from the controller, so the oracle and the kernel
        cannot disagree about which rate was under test.
    """
    ctrl = controller(spawn_number_change_rate=change_rate)
    bookkeeping = species_state(
        species_keys=np.array([0.0, 1.0, np.nan, np.nan], dtype=np.float32),
        best_fitness=np.array([0.5, 0.2, np.nan, np.nan], dtype=np.float32),
        member_count=np.array([2.0, 6.0, np.nan, np.nan], dtype=np.float32),
    )
    keys = bookkeeping.species_keys
    live = live_mask(keys)
    target = spawn_target(keys, bookkeeping.best_fitness, mode=mode, pop_size=ctrl.pop_size)
    previous = np.where(
        np.isnan(bookkeeping.member_count), np.float32(0.0), bookkeeping.member_count
    )

    if change_rate == 0.0:
        law = previous
    elif change_rate == 1.0:
        law = target
    else:
        law = damped_values(
            keys,
            bookkeeping.member_count,
            target,
            change_rate=ctrl.spawn_number_change_rate,
            min_species_size=ctrl.min_species_size,
        )

    floored = np.where(live, np.maximum(law, np.float32(ctrl.min_species_size)), 0.0)
    base = np.floor(floored).astype(np.int32)
    head = int(np.argmax(live))
    base[head] += np.int32(ctrl.pop_size - int(base.sum()))

    spawn = allocate(species_kernel, mode, bookkeeping, ctrl)

    np.testing.assert_array_equal(base, spawn)
    np.testing.assert_array_equal(oracle(mode, bookkeeping, ctrl), spawn)


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
def test_damping_midpoint_strictly_between_the_previous_size_and_the_target(
    species_state, controller, species_kernel, mode
):
    """Invariant: ``rate = 0.5`` lands strictly between the two extremes.

    Oracle: the three allocations of the same table at rates ``0``, ``0.5`` and
    ``1`` must be pairwise distinct, and the midpoint's live slots must lie between
    the ``rate = 0`` and ``rate = 1`` integer vectors slot by slot.  This is the
    property that stops a species from doubling in one generation: it rules out
    "the rate is ignored", "the rate is rounded to 0/1" and an inverted mixture
    (``previous + (previous - target) * rate``), all of which still conserve the
    population and would pass a sum-only assertion.
    """
    bookkeeping = species_state(
        species_keys=np.array([0.0, 1.0, np.nan, np.nan], dtype=np.float32),
        best_fitness=np.array([0.5, 0.2, np.nan, np.nan], dtype=np.float32),
        member_count=np.array([2.0, 6.0, np.nan, np.nan], dtype=np.float32),
    )
    live = live_mask(bookkeeping.species_keys)
    spawns = {}
    for rate in (0.0, 0.5, 1.0):
        ctrl = controller(spawn_number_change_rate=rate)
        spawns[rate] = allocate(species_kernel, mode, bookkeeping, ctrl)
        np.testing.assert_array_equal(oracle(mode, bookkeeping, ctrl), spawns[rate])

    assert not np.array_equal(spawns[0.0], spawns[1.0]), spawns
    assert not np.array_equal(spawns[0.5], spawns[1.0]), spawns
    assert not np.array_equal(spawns[0.0], spawns[0.5]), spawns
    low = np.minimum(spawns[0.0], spawns[1.0])
    high = np.maximum(spawns[0.0], spawns[1.0])
    assert np.all(spawns[0.5][live] >= low[live]), spawns
    assert np.all(spawns[0.5][live] <= high[live]), spawns


# --------------------------------------------------------------------------- #
# the two allocators' own properties
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("live_count", [2, 3, 4])
def test_linear_rank_allocation_is_monotone_in_species_rank(
    species_state, controller, species_kernel, live_count
):
    """Invariant: over the live slots of a sorted table, spawn is non-increasing.

    Oracle: ``spawn[i] >= spawn[i + 1]`` for consecutive live slots.  The rank
    allocator documents the sorted state as a *precondition* (slot 0 is the
    fittest, established by ``update_species``), so this is the promise that a
    better species never gets fewer children than a worse one.  ``rate = 1`` (pure
    target) is the configuration where the property belongs to the allocator
    itself: at ``rate = 0`` the result is the previous allocation by definition and
    is not expected to be ranked.
    """
    ctrl = controller(spawn_number_change_rate=1.0)
    bookkeeping = table_from(species_state, live_count)

    spawn = allocate(species_kernel, "rank", bookkeeping, ctrl)

    live_spawn = spawn[live_mask(bookkeeping.species_keys)]
    assert np.all(np.diff(live_spawn) <= 0), spawn
    np.testing.assert_array_equal(oracle("rank", bookkeeping, ctrl), spawn)


def test_fitness_allocation_masks_the_nan_best_fitness_of_dead_slots(
    species_state, controller, species_kernel
):
    """Invariant: a dead slot's ``best_fitness`` cannot reach ``min``/``sum``.

    Oracle: the same table three times with the dead slots' ``best_fitness`` set to
    ``NaN``, ``+1e9`` and ``-1e9``.  The returned spawn must be *identical* in all
    three, dead slots at ``0``, and equal to :func:`expected_spawn`, whose
    ``"fitness"`` target reads only live slots.  Varying the value is what proves
    masking rather than "the ``NaN`` happened to cancel": with an unmasked
    reduction, ``+1e9`` swamps every live weight and ``-1e9`` shifts ``min_live``
    by a billion, both turning the live rates into garbage while a residual
    correction could still restore ``sum == pop_size``.
    """
    ctrl = controller()
    spawns = {}
    for dead_best_fitness in (np.nan, 1e9, -1e9):
        bookkeeping = species_state(
            species_keys=np.array([0.0, 1.0, np.nan, np.nan], dtype=np.float32),
            best_fitness=np.array(
                [0.5, 0.2, dead_best_fitness, dead_best_fitness], np.float32
            ),
            member_count=np.array([5.0, 3.0, np.nan, np.nan], dtype=np.float32),
        )
        spawns[dead_best_fitness] = allocate(
            species_kernel, "fitness", bookkeeping, ctrl
        )
        dead = ~live_mask(bookkeeping.species_keys)
        np.testing.assert_array_equal(
            spawns[dead_best_fitness][dead], np.zeros(int(dead.sum()), np.int32)
        )
        np.testing.assert_array_equal(
            oracle("fitness", bookkeeping, ctrl), spawns[dead_best_fitness]
        )

    np.testing.assert_array_equal(spawns[np.nan], spawns[1e9])
    np.testing.assert_array_equal(spawns[np.nan], spawns[-1e9])


# --------------------------------------------------------------------------- #
# integerisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
@pytest.mark.parametrize(
    "member_count", [[0.0, 0.0, np.nan, np.nan], [np.nan] * 4], ids=["zero", "nan"]
)
def test_integerisation_never_yields_a_negative_spawn(
    species_state, controller, species_kernel, mode, member_count
):
    """Invariant: the float → ``int32`` step yields no negative and no non-finite spawn.

    Oracle: ``spawn.dtype == np.int32`` and ``spawn >= 0`` on tables whose
    ``member_count`` is ``0`` and ``NaN`` — the values the pre-rewrite code fed
    straight into ``.astype(int)``, where a ``NaN`` becomes the platform's
    ``INT_MIN``.  The float vector of :func:`damped_values` is additionally checked
    to be finite, so a negative or ``NaN`` spawn is caught before it can wrap, and
    the whole vector must still match :func:`expected_spawn` (which forces
    ``sum == pop_size``, so "all zeros" is not an escape).
    """
    ctrl = controller()
    bookkeeping = species_state(
        species_keys=np.array([0.0, 1.0, np.nan, np.nan], dtype=np.float32),
        best_fitness=np.array([0.5, 0.2, np.nan, np.nan], dtype=np.float32),
        member_count=np.array(member_count, dtype=np.float32),
    )
    damped = damped_for(mode, bookkeeping, ctrl)

    spawn = allocate(species_kernel, mode, bookkeeping, ctrl)

    assert np.all(np.isfinite(damped)), damped
    assert np.all(damped >= 0.0), damped
    assert spawn.dtype == np.int32, spawn.dtype
    assert np.all(spawn >= 0), spawn
    np.testing.assert_array_equal(oracle(mode, bookkeeping, ctrl), spawn)


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
def test_an_all_dead_table_still_conserves_the_population(
    species_state, controller, species_kernel, mode
):
    """Invariant: with no live slot left, ``sum(spawn) == pop_size`` still holds.

    Oracle: every key is ``NaN`` (every species killed in one generation), so the
    residual has no live slot to land on; the kernel documents that it then goes to
    slot ``0`` (``argmax`` of an all-``False`` liveness mask).  The assertion is
    therefore only conservation, ``int32``, non-negativity and agreement with
    :func:`expected_spawn` — *not* "dead slots are zero", which is the mixed-table
    invariant of
    :func:`test_dead_slots_receive_no_spawn`.  The pre-rewrite rank denominator
    divided by zero here and produced ``NaN``; the rewrite floors it at ``1``.
    """
    ctrl = controller()
    bookkeeping = species_state(
        species_keys=np.array([np.nan] * 4, dtype=np.float32),
        best_fitness=np.array([np.nan] * 4, dtype=np.float32),
        member_count=np.array([np.nan] * 4, dtype=np.float32),
    )

    spawn = allocate(species_kernel, mode, bookkeeping, ctrl)

    assert spawn.dtype == np.int32, spawn.dtype
    assert int(spawn.sum()) == ctrl.pop_size, spawn
    assert np.all(spawn >= 0), spawn
    np.testing.assert_array_equal(oracle(mode, bookkeeping, ctrl), spawn)


# --------------------------------------------------------------------------- #
# the spawn algebra's two building blocks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [-1.5, 2.5, 0.0, 7.0],
    ],
    ids=["ascending", "zeros", "mixed-sign"],
)
def test_exclusive_prefix_sum_matches_the_numpy_reference(raw_kernel, values):
    """Invariant: ``_prefix_sum_exclusive(v)[i] == sum(v[:i])``, with ``out[0] == 0``.

    Oracle: ``np.concatenate(([0], np.cumsum(v)[:-1]))`` — the definition, not a
    cumsum kernel (``evoxc.cumsum`` does not export on the compiler backends, so the
    implementation is a triangular ``dot``; the test pins the *result*, not the
    strategy).  The mixed-sign vector catches a triangular matrix with an off-by-one
    diagonal (a ``-1.5`` that lands in the wrong column), and the all-zero vector
    catches a kernel that forgets to zero the head.
    """
    expected = exclusive_prefix(values)

    result = raw_kernel("_prefix_sum_exclusive", np.array(values, dtype=np.float32))

    assert result.dtype == np.float32, result.dtype
    np.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "spawn",
    [
        [8, 0, 0, 0],
        [3, 0, 5, 0],
        [2, 3, 3, 0],
        [0, 4, 4, 0],
    ],
    ids=["single-block", "gap-in-the-middle", "three-blocks", "leading-gap"],
)
def test_block_layout_maps_every_population_slot_to_exactly_one_block(
    controller, raw_kernel, spawn
):
    """Invariant: ``(loc, idx_in_block)`` is a bijection from population slots to blocks.

    Oracle: :func:`expected_block_layout` (a Python loop over the documented
    "last qualifying block" rule) plus the algebraic restatement for every slot
    ``p``: ``E[loc[p]] + idx_in_block[p] == p`` and
    ``0 <= idx_in_block[p] < spawn[loc[p]]``, which together say that each block is
    filled left to right and that no population slot is left over or handed out
    twice.

    Notes:
        The ``[3, 0, 5, 0]`` case is the regression for the *count* form of the
        rule (``#{i : spawn[i] > 0 and E[i] <= p} - 1``): at ``p = 3`` the count is
        ``2`` while the answer is block ``2``, so a count-based ``loc`` fails here
        and nowhere else in this parameter list.  ``[0, 4, 4, 0]`` pins the
        leading zero-spawn block (the first qualifying block need not be slot 0).
    """
    ctrl = controller()
    spawn_vector = np.array(spawn, dtype=np.int32)
    exclusive = exclusive_prefix(spawn_vector)

    loc, idx_in_block = raw_kernel("_block_layout", spawn_vector)

    assert loc.dtype == np.int32, loc.dtype
    assert idx_in_block.dtype == np.int32, idx_in_block.dtype
    expected_loc, expected_idx = expected_block_layout(spawn)
    np.testing.assert_array_equal(expected_loc, loc)
    np.testing.assert_array_equal(expected_idx, idx_in_block)

    slots = np.arange(ctrl.pop_size)
    np.testing.assert_array_equal(
        exclusive.astype(np.int32)[loc] + idx_in_block, slots.astype(np.int32)
    )
    assert np.all(idx_in_block >= 0), idx_in_block
    assert np.all(idx_in_block < spawn_vector[loc]), (idx_in_block, spawn_vector[loc])

    pairs = {(int(block), int(offset)) for block, offset in zip(loc, idx_in_block)}
    assert len(pairs) == ctrl.pop_size, pairs
    assert pairs == {
        (block, offset)
        for block, count in enumerate(spawn)
        for offset in range(int(count))
    }
