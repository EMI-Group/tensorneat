"""Unit tests for the spawn allocation of ``SpeciesController`` — the species budget.

What this module proves
-----------------------
Spawn allocation decides how many children each species contributes to the next
generation.  It is the one place where a float computation turns into an integer
budget, so the whole suite is about *conservation and hygiene* rather than exact
values:

* ``sum(spawn) == pop_size`` **exactly**, in both allocators
  (``_cal_spawn_numbers_by_rank`` and ``_cal_spawn_numbers_by_fitness``) and for
  every damping rate — the error-correction step is the only reason a rounding
  residual cannot silently shrink the population;
* a dead slot (``isnan(species_keys)``) receives ``0`` and no live slot is starved
  below ``min_species_size`` when the budget allows it;
* ``spawn_number_change_rate`` is a *damping* factor: ``0`` reproduces the previous
  allocation, ``1`` jumps straight to the target, and anything in between stays
  between the two — so a species cannot double in one generation;
* linear-rank allocation is monotone in species rank (slot order after the
  descending sort), and fitness-proportional allocation never divides by the NaN
  ``best_fitness`` of a dead slot;
* the float → ``int32`` integerisation never produces a negative or non-finite
  spawn — the legacy code relied on "casting NaN to int32 gives 0", which is
  exactly the platform-dependent behaviour this rewrite removed.

Oracles used here (never the implementation)
--------------------------------------------
* :func:`expected_spawn` — a numpy transcription of the documented formula
  (``n`` live slots, ``rank_score = n - i``, triangular denominator,
  ``f' = f - min_live(f) + 1``, damping against ``member_count``, floor at
  ``min_species_size``, mask the dead slots, cast, add the residual to slot ``0``).
  It is a formula, not a second implementation: no gather/mask machinery, so it
  cannot mimic a bug in the kernel's data movement.
* :func:`expected_block_layout` — a Python loop over the documented
  ``loc(p) = #{i : spawn[i] > 0 and E[i] <= p} - 1`` / ``idx_in_block(p) =
  p - E[loc(p)]`` definition, compared against the triangular-``dot`` kernel.
* ``np.cumsum`` — the exclusive prefix sum's definition, with the ``0`` head.

Gotchas this module encodes
---------------------------
* The residual is added to **slot 0** by convention, and the constructor rejects
  ``species_size * min_species_size > pop_size`` upstream, so every fixture state
  here keeps slot ``0`` live.  A state whose first slot is dead would have the
  residual land on a dead species; that is excluded by construction, not asserted.
* ``spawn`` is ``int32``; a test that wants to check "no NaN" must check the *float*
  value **before** the cast (an int32 simply cannot carry NaN), so the assertion is
  written as "finite and ``>= 0``" on the numpy oracle's pre-cast vector plus an
  ``int32`` dtype check on the kernel's output.
* ``_cal_spawn_numbers_by_rank`` assumes the species state is already sorted
  (slot order == fitness rank): "slot 0 is the fittest" is a *precondition* of the
  allocator, established by ``update_species``, not something it recomputes.

Backend is ``"numpy"`` throughout; no optional dependency, no ``jax``/``torch``, no
network.
"""

from __future__ import annotations

import numpy as np
import pytest

from support.runner import run_defn, to_numpy

#: Method name of each allocator, keyed by ``species_number_calculate_by``.
ALLOCATORS = {
    "rank": "_cal_spawn_numbers_by_rank",
    "fitness": "_cal_spawn_numbers_by_fitness",
}


# --------------------------------------------------------------------------- #
# numpy oracles
# --------------------------------------------------------------------------- #


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
    """Transcribe the documented spawn formula into plain numpy.

    Args:
        species_keys: ``(S,)`` float keys; ``NaN`` marks a dead slot.
        best_fitness: ``(S,)`` float, the fitness-proportional weight
            (``"fitness"`` mode only; ``NaN`` on dead slots).
        member_count: ``(S,)`` float, the previous allocation the new one is damped
            against.
        mode: ``"rank"`` or ``"fitness"``.
        pop_size: Static population size the allocation must sum to.
        change_rate: ``spawn_number_change_rate`` in ``[0, 1]``.
        min_species_size: Lower bound of a live species' spawn count.

    Returns:
        ``(S,) int32``, the expected spawn counts summing to ``pop_size``.

    Notes:
        The order of operations is the documented one and matters: floor at
        ``min_species_size``, *then* force dead/NaN slots to ``0``, *then* cast,
        *then* correct the total on slot ``0``.
    """
    keys = np.asarray(species_keys, dtype=np.float64)
    previous = np.asarray(member_count, dtype=np.float64)
    live = ~np.isnan(keys)
    slots = keys.shape[0]
    n_live = int(live.sum())

    if mode == "rank":
        rank_score = n_live - np.arange(slots, dtype=np.float64)
        denominator = (n_live + 1) * n_live / 2.0
        target = np.floor(rank_score / denominator * pop_size)
    else:
        fitness = np.asarray(best_fitness, dtype=np.float64)
        shifted = np.where(live, fitness - np.nanmin(fitness[live]) + 1.0, np.nan)
        total = np.nansum(shifted)
        target = np.floor(shifted / total * pop_size)

    damped = previous + (target - previous) * change_rate
    damped = np.where(live, np.maximum(damped, float(min_species_size)), 0.0)
    damped = np.where(np.isfinite(damped), damped, 0.0)
    spawn = damped.astype(np.int32)
    spawn[0] += np.int32(pop_size - spawn.sum())
    return spawn


def expected_block_layout(spawn):
    """Compute ``(loc, idx_in_block)`` with the documented definition, by hand.

    Args:
        spawn: ``(S,)`` int, summing to ``P``; non-positive entries are absent
            blocks.

    Returns:
        tuple: ``(loc, idx_in_block)``, both ``(P,)`` ``int`` arrays.

    Notes:
        ``loc(p)`` is the *last* block whose exclusive prefix start is ``<= p`` and
        whose spawn is positive, which is what makes a zero-spawn species consume no
        population slot.  ``./test_parent_selection.py`` keeps a local copy of this
        function (a test module must stay importable on its own); the two copies
        define the same formula and change together.
    """
    spawn = np.asarray(spawn)
    exclusive = np.concatenate(([0], np.cumsum(spawn)[:-1]))
    total = int(spawn.sum())
    loc = np.zeros(total, dtype=np.int64)
    offset = np.zeros(total, dtype=np.int64)
    for p in range(total):
        blocks = [
            i for i in range(spawn.shape[0]) if spawn[i] > 0 and exclusive[i] <= p
        ]
        loc[p] = blocks[-1]
        offset[p] = p - exclusive[loc[p]]
    return loc, offset


# --------------------------------------------------------------------------- #
# execution helpers
# --------------------------------------------------------------------------- #


def run_static(species, method_name, *arrays):
    """Run a ``SpeciesController`` static helper (no controller instance) in a graph.

    Args:
        species: The ``tensorneat.algorithm.neat.species`` module fixture.
        method_name: ``"_prefix_sum_exclusive"`` or ``"_block_layout"``.
        *arrays: One numpy array per graph input.

    Returns:
        Host numpy output, ``to_numpy``-unwrapped (a tuple for ``_block_layout``).

    Notes:
        The helper is reached on the class and takes no ``self``-like configuration,
        so it needs neither a controller nor a ``SpeciesState``.
    """
    method = getattr(species.SpeciesController, method_name)
    return to_numpy(run_defn(lambda array: method(array), *arrays))


# --------------------------------------------------------------------------- #
# conservation of the population budget
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
@pytest.mark.parametrize("change_rate", [0.0, 0.5, 1.0])
def test_allocation_corrects_the_total_to_pop_size_exactly(
    species, controller, species_state, species_kernel, mode, change_rate
):
    """Invariant: ``sum(spawn) == pop_size`` exactly, for every damping rate and both allocators.

    Oracle: ``int(spawn.sum()) == pop_size`` plus ``spawn.dtype == np.int32``,
    asserted over several live counts (1, 2 and 3 live slots out of 4) so the
    residual correction is exercised when the floor/rounding leaves a positive *and*
    a negative gap.  A missing correction shrinks the population every generation,
    which is a silent divergence from the legacy library, not a cosmetic rounding.
    """
    pytest.skip(
        "not implemented: sum(spawn) == pop_size after error correction "
        "(owner: algorithm/neat/species.SpeciesController._cal_spawn_numbers_by_rank "
        "/ _cal_spawn_numbers_by_fitness)"
    )


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
def test_dead_slots_receive_no_spawn(species, controller, species_state, species_kernel, mode):
    """Invariant: a slot with ``NaN`` species key gets exactly ``0`` spawn, in both allocators.

    Oracle: the dead slots of the fixture table (2 and 3) must be ``0`` while the
    live ones sum to ``pop_size``; the check runs with a dead slot whose
    ``best_fitness`` is ``NaN`` (fitness mode) **and** with ``member_count`` left at
    a stale large value, so a slot cannot sneak in through the damping term.
    """
    pytest.skip(
        "not implemented: dead species get zero spawn "
        "(owner: algorithm/neat/species.SpeciesController._cal_spawn_numbers_by_rank "
        "/ _cal_spawn_numbers_by_fitness)"
    )


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
def test_a_live_species_receives_at_least_min_species_size(
    species, controller, species_state, species_kernel, mode
):
    """Invariant: every live slot is floored at ``min_species_size`` when the budget allows it.

    Oracle: ``min_species_size = 2`` with three live slots out of ``pop_size = 8``
    (``3 * 2 <= 8``, the constructor's own precondition), with a species whose raw
    target rounds to ``0`` and whose previous size is ``0``; assert each live slot is
    ``>= 2`` after the residual correction, and that the sum is still exactly
    ``pop_size``.  The floor is what keeps a small but successful species alive.
    """
    pytest.skip(
        "not implemented: min_species_size floor on live slots "
        "(owner: algorithm/neat/species.SpeciesController._cal_spawn_numbers_by_rank "
        "/ _cal_spawn_numbers_by_fitness)"
    )


# --------------------------------------------------------------------------- #
# the damping rate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
@pytest.mark.parametrize("change_rate", [0.0, 1.0])
def test_damping_rate_moves_between_the_previous_size_and_the_target(
    species, controller, species_state, species_kernel, mode, change_rate
):
    """Invariant: ``rate = 0`` reproduces ``member_count``, ``rate = 1`` jumps to the target.

    Oracle: :func:`expected_spawn` at both extremes.  ``rate = 0`` must reproduce the
    integerised ``member_count`` (masked and floored) and ``rate = 1`` must reproduce
    the raw target; the ``0.5`` midpoint is exercised by
    :func:`test_allocation_corrects_the_total_to_pop_size_exactly`, which only
    requires the conservation law.  The damping term is the contract that stops a
    species from doubling in one generation, and the documented formula is
    ``previous + (target - previous) * rate`` — an off-by-one in either direction
    changes every run of a seeded experiment.
    """
    pytest.skip(
        "not implemented: spawn_number_change_rate damping against member_count "
        "(owner: algorithm/neat/species.SpeciesController._cal_spawn_numbers_by_rank "
        "/ _cal_spawn_numbers_by_fitness)"
    )


# --------------------------------------------------------------------------- #
# the two allocators' own properties
# --------------------------------------------------------------------------- #


def test_linear_rank_allocation_is_monotone_in_species_rank(
    species, controller, species_state, species_kernel
):
    """Invariant: with the sorted species state, spawn is non-increasing over live slots.

    Oracle: the fixture's live slots are slot ``0`` and ``1`` (rank 0 and rank 1, the
    fittest first, because ``_cal_spawn_numbers_by_rank`` documents the sort as a
    *precondition*); assert ``spawn[0] >= spawn[1]`` and that the same holds for a
    four-live-slot table.  Linear ranking is the promise that a better species never
    gets fewer children than a worse one.
    """
    pytest.skip(
        "not implemented: monotone linear-rank allocation "
        "(owner: algorithm/neat/species.SpeciesController._cal_spawn_numbers_by_rank)"
    )


def test_fitness_allocation_masks_the_nan_best_fitness_of_dead_slots(
    species, controller, species_state, species_kernel
):
    """Invariant: the NaN ``best_fitness`` of dead slots cannot reach ``min``/``sum``.

    Oracle: the same state three times with the dead slots' ``best_fitness`` set to
    ``NaN``, ``+1e9`` and ``-1e9`` — the returned spawn must be *identical* in all
    three, finite, and equal to :func:`expected_spawn` over the live slots only.
    Replacing the value proves masking rather than "NaN happened to cancel out";
    the legacy bug was exactly that a ``-1e9``/NaN mix turned every rate into NaN.
    """
    pytest.skip(
        "not implemented: fitness mode masks dead slots before min/sum "
        "(owner: algorithm/neat/species.SpeciesController._cal_spawn_numbers_by_fitness)"
    )


# --------------------------------------------------------------------------- #
# integerisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", sorted(ALLOCATORS))
@pytest.mark.parametrize("member_count", [[0.0, 0.0, np.nan, np.nan], [np.nan] * 4])
def test_integerisation_never_yields_a_negative_spawn(
    species, controller, species_state, species_kernel, mode, member_count
):
    """Invariant: the float → ``int32`` step produces no negative and no non-finite spawn.

    Oracle: ``np.all(spawn >= 0)`` and ``spawn.dtype == np.int32`` on states whose
    ``member_count`` contains ``0`` and ``NaN`` (the values the legacy code fed
    straight into ``.astype(int)``, where NaN maps to the platform's ``INT_MIN``).
    The pre-cast vector of :func:`expected_spawn` is checked to be finite as well, so
    a negative residual would be caught before it can wrap.
    """
    pytest.skip(
        "not implemented: non-negative int32 spawn from NaN/zero inputs "
        "(owner: algorithm/neat/species.SpeciesController._cal_spawn_numbers_by_rank "
        "/ _cal_spawn_numbers_by_fitness)"
    )


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
)
def test_exclusive_prefix_sum_matches_the_numpy_reference(species, values):
    """Invariant: ``_prefix_sum_exclusive(v)[i] == sum(v[:i])``, with ``out[0] == 0``.

    Oracle: ``np.concatenate(([0.0], np.cumsum(v)[:-1]))`` — the definition, not a
    cumsum kernel (``evoxc.cumsum`` does not export on the compiler backends, so the
    implementation is a triangular ``dot``; the test pins the *result*, not the
    strategy).  Negative entries are included so a triangular matrix with the wrong
    diagonal offset is caught.
    """
    pytest.skip(
        "not implemented: exclusive prefix sum equals cumsum with a zero head "
        "(owner: algorithm/neat/species.SpeciesController._prefix_sum_exclusive)"
    )


def test_block_layout_maps_every_population_slot_to_exactly_one_block(species):
    """Invariant: ``(loc, idx_in_block)`` is a bijection from population slots to blocks.

    Oracle: :func:`expected_block_layout` on the same spawn vector (a Python loop
    over the documented definition) plus the algebraic restatement
    ``E[loc[p]] + idx_in_block[p] == p`` and ``0 <= idx_in_block[p] < spawn[loc[p]]``
    for every ``p``.  The spawn vector mixes a zero-spawn species (which must consume
    no slot) with several positive blocks, because "count only blocks with
    ``spawn > 0``" is the whole point of the formula.
    """
    pytest.skip(
        "not implemented: block flattening is a bijection onto the population axis "
        "(owner: algorithm/neat/species.SpeciesController._block_layout)"
    )
