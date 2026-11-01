"""Unit tests for ``SpeciesController._pair_parents`` — who becomes a parent, and of whom.

What this module proves
-----------------------
Parent pairing is the only step of a generation that *draws* from the population, so
its contract is a mixture of algebra and randomness, and the tests check the algebra
exactly and the randomness only through the bounds it must respect:

* the survivor count is ``clamp(floor(survival_threshold * member_count), 1,
  min(member_count, pop_size))`` — floored at one so a one-member species is legal,
  and capped at the member count so the non-member tail of the ranking (all
  ``-inf``) can never be drawn (the implementation spells the cap as
  ``max(min(member_count, pop_size), 1)``, which is the same value for every live
  species with at least one member — the floor covers the empty slot);
* a one-member species yields **two identical parents** (self-crossover, which is the
  identity) instead of the legacy zero-survivor division that produced a NaN
  probability vector and made ``random.choice`` fail;
* elite slots — the slots with ``idx_in_block < genome_elitism`` of every species
  block — are pinned to that species' **best** member, so an elite child is the
  unmutated crossover of the champion with itself;
* the winner/loser decision always makes the **fitter** genome parent 1 (the
  order-sensitive argument of ``execute_crossover``: parent 1's structure is kept),
  with ties going to the first draw (the father, ``>=``) — pinned against the
  reproduced draw by :func:`test_a_tie_hands_the_slot_to_the_father`, not only by
  the ``fitness[winner] >= fitness[loser]`` inequality both tie rules satisfy;
* every population slot draws its parents from **its own** species block, and the
  block layout assigns each slot to exactly one block;
* the draw depends on the supplied ``evoxc.random`` key and on nothing else — the
  numpy global RNG has no influence, which is what makes ``(seed, config, shape)``
  reproducible.

Oracles used here (never the implementation)
--------------------------------------------
* :func:`species_ranking` — the documented ranking
  (``argsort`` of the masked fitness, stable ascending, then reversed to reproduce
  the legacy tie order, i.e. exactly what ``_sort_desc_legacy`` defines) computed on
  the host from ``fitness``/``idx2species``.
* :func:`survivor_count` — the documented clamp formula, written out.
* :func:`expected_block_layout` — the ``loc``/``idx_in_block`` mapping by its
  definition, so "draws from its own block" is checked against the block a slot
  *should* belong to.
* :func:`survive_bound`, :func:`ranking_matrix` and :func:`reproduce_draws` — the
  documented ``randint`` call and its ``flat_gather`` re-run on the host from the
  same key, shape, argument order and survivor bounds, so winner/loser is checked
  against the *draw* itself rather than against an inequality both tie rules
  satisfy.
* ``np.array_equal`` — determinism and RNG independence.

Gotchas this module encodes
---------------------------
* ``_pair_parents`` returns **row indices into the current population**, not ranks:
  a test compares against ``species_ranking(...)[rank]``, never against the rank.
* The ranking is over ``select(membership, fitness, -inf)``, so the tail of the
  ranking holds non-members.  The survivor cap is what makes that tail unreachable;
  at ``survival_threshold = 1.0`` the bound degenerates to "any member", which is
  why the parametrised table keeps the ``0.1``/``0.5`` regimes next to it (those
  make the survivor prefix strictly smaller than the member count).
* Ties are ordered by the legacy reversal, so the fitness tables in the elite and
  prefix tests use **distinct** values within a species; the tie order itself is
  pinned separately by :func:`test_the_fitter_parent_is_always_parent_one`, which
  also runs on a table with a tie at the top of both species.
* The ``>=``/``>`` difference is invisible to ``fitness[winner] >= fitness[loser]``:
  both rules satisfy it.  ``test_a_tie_hands_the_slot_to_the_father`` therefore runs
  on a table where *every* member of a species has the same fitness, reproduces the
  draw and asserts ``winner``/``loser`` are the father/mother row of each slot.
* ``elite_mask`` is per population slot and spans *every* block, so "the elites are
  pinned" is asserted block by block (``loc(p) == i`` and ``idx_in_block(p) <
  genome_elitism``), not as a prefix of the whole population.
* A live species may own a block of ``spawn == 0`` slots in the middle of the table
  (only *dead* slots are guaranteed to trail after the descending sort); the block
  layout has to skip such a species, which is why one table below has a live,
  empty species between two populated ones.

Backend is ``"numpy"`` throughout; no optional dependency, no ``jax``/``torch``, no
network.
"""

from __future__ import annotations

import evoxc
import numpy as np
import pytest

from support.runner import run_defn

#: Population size of the hand-built tables — the ``POP_SIZE`` of `./conftest.py`,
#: repeated here so the oracles below can be called on their own.
POP_SIZE = 8
#: Species-slot capacity of the same tables (two live, two dead by default).
SPECIES_SIZE = 4

#: The fixture population: rows ``0..4`` are species key ``0.0``, rows ``5..7`` are
#: key ``1.0``.  Fitness is distinct inside each species and the species maxima are
#: ``0.9`` (row 1) and ``0.8`` (row 6), which is what the elite assertions compare
#: against.
FITNESS = np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.2, 0.8, 0.4], dtype=np.float32)
#: The same table with a two-way tie at the top of *both* species (rows ``0``/``1``
#: at ``0.9`` and rows ``6``/``7`` at ``0.8``).
TIED_FITNESS = np.array([0.9, 0.9, 0.5, 0.7, 0.3, 0.2, 0.8, 0.8], dtype=np.float32)
#: A table where **every** member of a species has the same fitness (``0.4`` for
#: species ``0``, ``0.2`` for species ``1``), so every drawn pair is a tie.  The
#: species maxima differ so the table cannot be mistaken for a globally flat one.
TIED_SPECIES_FITNESS = np.array(
    [0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2], dtype=np.float32
)
#: The spawn allocation matching :data:`FITNESS`/``GROUPS_DEFAULT`` (sums to ``8``).
SPAWN = np.array([5, 3, 0, 0], dtype=np.int32)
#: A spawn vector with a live, **empty** species in the middle, so the block index
#: and the species slot index deliberately differ (sums to ``8``).
MIDDLE_EMPTY_SPAWN = np.array([3, 0, 5, 0], dtype=np.int32)

#: ``(key, member rows)`` per species slot, for the ``FITNESS``/``SPAWN`` table.
GROUPS_DEFAULT = ((0.0, (0, 1, 2, 3, 4)), (1.0, (5, 6, 7)))
#: The same for :data:`MIDDLE_EMPTY_SPAWN`: slot ``1`` is live but owns no row.
GROUPS_MIDDLE_EMPTY = ((0.0, (0, 1, 2)), (1.0, ()), (2.0, (3, 4, 5, 6, 7)))


# --------------------------------------------------------------------------- #
# numpy oracles
# --------------------------------------------------------------------------- #


def survivor_count(survival_threshold, members):
    """``clamp(floor(survival_threshold * members), 1, min(members, pop_size))``.

    Args:
        survival_threshold: The controller's ``survival_threshold``.
        members: A live species' ``member_count`` (``1 <= members <= pop_size``).

    Returns:
        int: The number of best-ranked members eligible as parents.
    """
    floored = int(np.floor(survival_threshold * members))
    return int(min(max(floored, 1), min(int(members), POP_SIZE)))


def survive_bound(member_count, survival_threshold):
    """``randint``'s per-slot upper bound: the documented clamp over the raw counts.

    Args:
        member_count: ``(S,)`` ``SpeciesState.member_count``; a dead slot holds ``NaN``.
        survival_threshold: The controller's ``survival_threshold``.

    Returns:
        ``(S,)`` int32, ``clamp(floor(threshold * members), 1, max(min(members,
        pop_size), 1))`` with a ``NaN`` count read as ``0``.

    Notes:
        This is the bound ``_pair_parents`` hands to ``evoxc.random.randint``, so the
        draw can be rerun on the host.  For a live slot the value equals
        :func:`survivor_count`; a dead slot's ``NaN`` count floors to the ``1`` of the
        clamp's *lower* bound (the legacy zero-bound fix), which is why the clamp is
        written out here instead of reused.  The difference for dead slots is
        unobservable: a dead slot owns no population slot, and the numpy backend
        derives each draw element from the key, the shape and that element's own
        bound, so another slot's bound cannot shift it.
    """
    previous = np.where(
        np.isnan(member_count), np.float32(0.0), np.asarray(member_count, np.float32)
    ).astype(np.float32)
    floored = np.floor(previous * np.float32(survival_threshold))
    ceiling = np.maximum(np.minimum(previous, np.float32(POP_SIZE)), np.float32(1.0))
    return np.clip(floored, np.float32(1.0), ceiling).astype(np.int32)


def species_ranking(fitness, idx2species, key):
    """Population row indices of one species, fittest first, in the legacy tie order.

    Args:
        fitness: ``(P,)`` float, sanitised and maximised.
        idx2species: ``(P,)`` species *keys*, one per population slot.
        key: The species key to rank.

    Returns:
        ``(members,)`` int row indices: a stable ascending ``argsort`` of the member
        fitness followed by a reversal, which is how ``_sort_desc_legacy`` reproduces
        the pre-rewrite ``argsort(...)[::-1]`` tie order.

    Notes:
        Population slots that are not members of ``key`` never appear — the kernel
        ranks the whole ``(P,)`` axis with ``-inf`` for non-members, and unreachable
        entries are simply not part of the expectation.
    """
    rows = np.where(np.asarray(idx2species) == key)[0]
    order = np.argsort(np.asarray(fitness)[rows], kind="stable")[::-1]
    return rows[order]


def ranking_matrix(fitness, idx2species, species_keys):
    """The kernel's ``(S, P)`` ranking matrix, assembled from :func:`species_ranking`.

    Args:
        fitness: ``(P,)`` float, sanitised and maximised.
        idx2species: ``(P,)`` species *keys*, one per population slot.
        species_keys: ``(S,)`` float, ``NaN`` for a dead slot.

    Returns:
        ``(S, P)`` int32: row ``s`` is the population rows of species slot ``s``,
        fittest first, in the legacy tie order.

    Notes:
        Only the prefix ``[:member_count]`` of a live row is reachable — every draw is
        ``< survive_size <= member_count`` — so the tail is filled with the remaining
        population rows only to keep the matrix well formed and is never asserted
        against.  A dead slot owns no block, so its identity row is equally
        unreachable.
    """
    every_row = np.arange(POP_SIZE, dtype=np.int32)
    ranking = np.zeros((SPECIES_SIZE, POP_SIZE), dtype=np.int32)
    for slot in range(SPECIES_SIZE):
        key = float(species_keys[slot])
        if np.isnan(key):
            ranking[slot] = every_row
            continue
        members = species_ranking(fitness, idx2species, key).astype(np.int32)
        ranking[slot, : members.size] = members
        ranking[slot, members.size :] = np.setdiff1d(every_row, members)
    return ranking


def expected_block_layout(spawn):
    """``(loc, idx_in_block)`` from the spawn definition, by hand.

    Args:
        spawn: ``(S,)`` int summing to ``P``.

    Returns:
        tuple: ``(loc, idx_in_block)``, both ``(P,)`` int arrays.

    Notes:
        ``loc(p)`` is the **last** ``i`` with ``spawn[i] > 0`` and ``E[i] <= p``
        (``E`` the exclusive prefix sum), and ``idx_in_block(p) = p - E[loc(p)]``.
        It is the *last qualifying block* rather than the *count* of qualifying
        blocks, because a live species may own no slot in the middle of the table
        (only dead slots are guaranteed to trail after the sort).

        Deliberately a local copy of :func:`test_spawn.expected_block_layout` rather
        than an import from the sibling module: a test module must stay importable on
        its own, and importing one test module from another would turn a change in
        ``test_spawn`` into a collection error here.  The two copies define the same
        documented formula, so they have to be changed together.
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
# host helpers
# --------------------------------------------------------------------------- #


def species_table(species_state, groups):
    """Build a sorted ``SpeciesState`` whose live species own the given rows.

    Args:
        species_state: The factory fixture of `./conftest.py`.
        groups: ``(species_key, rows)`` pairs, one per species *slot*, in slot order.
            ``rows`` may be empty — the slot is still live (its key is finite), which
            is how the "live but zero-spawn species in the middle" table is built.

    Returns:
        A ``SpeciesState`` with ``member_count`` = the row count per slot and
        ``idx2species`` mapping every listed row to its key (``NaN`` elsewhere).

    Notes:
        ``best_fitness``/``last_improved`` are zero-filled on live slots and ``NaN``
        on dead ones: ``_pair_parents`` never reads either, and a dead slot is dead
        *only* because ``species_keys`` is ``NaN``.  The centre payload arrays keep
        the fixture's zero fill for the same reason.
    """
    keys = np.full(SPECIES_SIZE, np.nan, np.float32)
    best = np.full(SPECIES_SIZE, np.nan, np.float32)
    improved = np.full(SPECIES_SIZE, np.nan, np.float32)
    counts = np.full(SPECIES_SIZE, np.nan, np.float32)
    idx2species = np.full(POP_SIZE, np.nan, np.float32)
    for slot, (key, rows) in enumerate(groups):
        keys[slot] = np.float32(key)
        best[slot] = np.float32(0.0)
        improved[slot] = np.float32(0.0)
        counts[slot] = np.float32(len(rows))
        idx2species[list(rows)] = np.float32(key)
    return species_state(
        species_keys=keys,
        best_fitness=best,
        last_improved=improved,
        member_count=counts,
        idx2species=idx2species,
    )


def rank_in_species(parent, fitness, idx2species, key):
    """Rank of population row ``parent`` inside its species (``0`` = fittest).

    Raises:
        AssertionError: when ``parent`` is not a member of ``key`` at all — that is
            the interesting failure of a parent draw, so it is reported with the
            member list instead of an empty-array ``IndexError``.
    """
    ranking = species_ranking(fitness, idx2species, key)
    matches = np.flatnonzero(ranking == parent)
    assert matches.size == 1, (
        f"population row {parent} is not a member of species key {key}; "
        f"that species owns rows {ranking.tolist()}"
    )
    return int(matches[0])


def assert_parent_indices(winner, loser):
    """Both parent arrays are ``(P,)`` integer row indices into the population."""
    for name, parent in (("winner", winner), ("loser", loser)):
        assert isinstance(parent, np.ndarray), f"{name} is not an ndarray: {type(parent)!r}"
        assert parent.shape == (POP_SIZE,), f"{name} has shape {parent.shape}"
        assert np.issubdtype(parent.dtype, np.integer), (
            f"{name} dtype {parent.dtype} is not integral"
        )
        assert not np.any(np.isnan(parent)), f"{name} contains NaN: {parent}"
        assert np.all((parent >= 0) & (parent < POP_SIZE)), (
            f"{name} is outside the population: {parent}"
        )


def pair(species_kernel, controller, bookkeeping, key, spawn=SPAWN, fitness=FITNESS):
    """Run ``_pair_parents`` through the shared runner and return ``(winner, loser, elite)``.

    Args:
        species_kernel: The conftest factory that rebuilds the ``SpeciesState`` inside
            a trace and calls one controller method.
        controller: A ``SpeciesController`` from the ``controller`` factory fixture.
        bookkeeping: The ``SpeciesState`` table under test.
        key: A rank-0 ``evoxc.random`` key (the ``evoxc_key`` fixture).
        spawn: ``(S,) int32`` allocation summing to ``pop_size``.
        fitness: ``(P,) float32`` sanitised fitness.

    Returns:
        tuple: ``(winner, loser, elite_mask)`` as host numpy arrays, ``(P,)`` each.
    """
    return species_kernel(
        "_pair_parents",
        bookkeeping,
        key,
        np.asarray(spawn),
        np.asarray(fitness),
        controller=controller,
    )


def reproduce_draws(common, selection_key, survive, ranking):
    """Re-run ``_pair_parents``'s single RNG call and its ``flat_gather`` on the host.

    Args:
        common: The ``tensorneat.common`` package fixture, for its ``flat_gather``.
        selection_key: The rank-0 ``evoxc.random`` key handed to ``_pair_parents``.
        survive: ``(S,)`` int32 ``randint`` upper bounds, i.e. :func:`survive_bound`.
        ranking: ``(S, P)`` int32 row indices, i.e. :func:`ranking_matrix`.

    Returns:
        tuple: ``(draws, father, mother)`` — ``(S, 2, P)`` int32 ranks, and the two
        ``(S * P,)`` int32 parent rows in ``(species_slot, population_slot)`` order.

    Notes:
        The call is the documented one — same key, same shape, same argument order,
        same per-slot bound — because the numpy backend's ``randint`` derives each
        element from the key, the shape and that element's own bound: a different
        shape, a different bound or a swapped ``father``/``mother`` axis changes the
        stream and breaks the match against the kernel.  ``flat_gather`` is the
        kernel's own helper rather than a re-implementation of it: it is the per-row
        gather whose flattened index ``species_slot * pop_size + rank`` a plain
        ``evoxc.gather`` cannot express.
    """

    def kernel(*values):
        key, bounds, rows = values
        draws = evoxc.random.randint(
            key,
            (SPECIES_SIZE, 2, POP_SIZE),
            0,
            evoxc.reshape(bounds, (SPECIES_SIZE, 1, 1)),
            dtype=evoxc.int32,
        )
        parent = common.flat_gather(rows, draws)
        width = SPECIES_SIZE * POP_SIZE
        father = evoxc.reshape(evoxc.gather(parent, 0, axis=1), (width,))
        mother = evoxc.reshape(evoxc.gather(parent, 1, axis=1), (width,))
        return draws, father, mother

    return run_defn(kernel, selection_key, np.asarray(survive), np.asarray(ranking))


def slot_parents(father, mother, loc, idx_in_block):
    """Map the ``(S * P,)`` species-space parents onto the ``(P,)`` population slots.

    Args:
        father: ``(S * P,)`` int32 rows in ``(species_slot, population_slot)`` order.
        mother: ``(S * P,)`` int32 mother rows, the same order.
        loc: ``(P,)`` block index per population slot (:func:`expected_block_layout`).
        idx_in_block: ``(P,)`` offset inside that block.

    Returns:
        tuple: ``(slot_father, slot_mother)``, both ``(P,)`` int32 parent rows.

    Notes:
        The flattening is the kernel's ``flat = loc * pop_size + idx_in_block``; it is
        what makes the *same* species-space draw serve every population slot of a
        block, so a slot's parent is decided by its position inside the block and not
        by the block index alone.
    """
    flat = np.asarray(loc, dtype=np.int64) * POP_SIZE + np.asarray(
        idx_in_block, dtype=np.int64
    )
    return np.asarray(father)[flat], np.asarray(mother)[flat]


# --------------------------------------------------------------------------- #
# the survivor set
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("survival_threshold", "members", "expected"),
    [
        (0.1, 5, 1),  # floor(0.5) == 0 -> floored at one member
        (0.5, 5, 2),  # floor(2.5) == 2 members
        (1.0, 5, 5),  # capped at the member count
        (0.1, 1, 1),  # the one-member species: floor(0.1) == 0 -> one member
        (0.9, 3, 2),  # floor(2.7) == 2 of a three-member species
    ],
)
def test_parents_come_only_from_the_survivor_prefix(
    species, controller, species_state, species_kernel, evoxc_key,
    survival_threshold, members, expected,
):
    """Invariant: both parents of a slot rank below ``clamp(floor(threshold * members), 1, members)``.

    Oracle: :func:`survivor_count` and :func:`species_ranking`; the drawn rows are
    mapped back to their rank inside the species and asserted ``< expected``.  The
    table pins the three interesting regimes — the floor at one (``0.1 * 5`` and the
    one-member species), the plain floor (``0.5 * 5``, ``0.9 * 3``), and the cap at
    the member count (``1.0``, which must not read past the block).  Species ``0``
    owns ``members`` rows and species ``1`` the rest, so the bound is checked against
    both species: the population-axis ranking is ``(S, P)`` wide and a stale
    survivor count could just as well read past *another* block.

    A threshold above ``1`` is not a pairing case any more: ``SpeciesController``
    validates ``survival_threshold`` as a ``(0, 1]`` unit interval and raises
    ``TensorNEATError``, so the cap is exercised with ``1.0`` here and the rejection
    belongs to the constructor suite.
    """
    groups = ((0.0, tuple(range(members))), (1.0, tuple(range(members, POP_SIZE))))
    bookkeeping = species_table(species_state, groups)
    # The spawn vector is ``(S,)`` wide like every species field: the two trailing
    # dead slots receive zero population slots.
    spawn = np.array([members, POP_SIZE - members, 0, 0], dtype=np.int32)
    # The parametrisation is a statement about the *documented* clamp; assert it so a
    # wrong expectation cannot silently weaken the bound below.
    counts = bookkeeping.member_count
    assert int(counts[0]) == members and int(counts[1]) == POP_SIZE - members
    assert survivor_count(survival_threshold, members) == expected

    winner, loser, elite = pair(
        species_kernel,
        controller(survival_threshold=survival_threshold),
        bookkeeping,
        evoxc_key,
        spawn=spawn,
    )
    assert_parent_indices(winner, loser)
    assert elite.shape == (POP_SIZE,)

    loc, _idx_in_block = expected_block_layout(spawn)
    for slot in range(POP_SIZE):
        block = int(loc[slot])
        key = float(bookkeeping.species_keys[block])
        survivors = survivor_count(survival_threshold, int(counts[block]))
        for parent in (int(winner[slot]), int(loser[slot])):
            rank = rank_in_species(parent, FITNESS, bookkeeping.idx2species, key)
            assert rank < survivors, (
                f"slot {slot} drew population row {parent} at rank {rank} of species "
                f"{key}, but only the best {survivors} members are parents"
            )


def test_a_one_member_species_yields_two_identical_parents(
    species, controller, species_state, species_kernel, evoxc_key
):
    """Invariant: a species with one member produces ``winner == loser ==`` that member.

    Oracle: a state whose second species owns exactly one row; both parent vectors
    must equal that row for the slot in its block, and every value must be a real
    population index.  This is the documented fix of the legacy zero-survivor
    division (``high = max(survive_size, 1)``): the old probability vector divided by
    zero, produced NaN and made ``random.choice`` fail, whereas a self-crossover is
    the identity and is therefore the correct outcome.  ``genome_elitism = 0`` keeps
    the elite pinning out of the way, so the assertion is about the **draw** and not
    about the elite shortcut (which would give the same answer for the wrong reason).
    """
    # The lone member of species ``1.0``, which owns the only slot of its block.
    lone = POP_SIZE - 1
    bookkeeping = species_table(
        species_state, ((0.0, tuple(range(lone))), (1.0, (lone,)))
    )
    spawn = np.array([lone, 1, 0, 0], dtype=np.int32)
    winner, loser, elite = pair(
        species_kernel,
        controller(genome_elitism=0),
        bookkeeping,
        evoxc_key,
        spawn=spawn,
    )

    assert_parent_indices(winner, loser)
    assert not np.any(elite), "genome_elitism = 0 must not mark any slot"
    assert winner[lone] == lone and loser[lone] == lone, (
        "the one-member species must draw its only member twice, but drew "
        f"winner={winner[lone]}, loser={loser[lone]}"
    )
    assert winner[lone] == loser[lone]
    # ``survival_threshold = 0.1`` leaves every live species with exactly one
    # survivor, so block ``0`` must repeat its own best member in every slot as well;
    # a NaN from the legacy division would show up as an out-of-range or
    # non-integral index in the checks above.
    assert np.all(winner[:lone] == winner[0]) and np.all(loser[:lone] == loser[0])


# --------------------------------------------------------------------------- #
# elites
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("genome_elitism", [1, 2, 3])
def test_elite_slots_are_pinned_to_the_species_best_member(
    species, controller, species_state, species_kernel, evoxc_key, genome_elitism
):
    """Invariant: for an elite slot, both parents are the best member of its own species.

    Oracle: :func:`expected_block_layout` gives ``loc``/``idx_in_block``; every slot
    with ``idx_in_block < genome_elitism`` must satisfy ``winner == loser ==
    species_ranking(...)[0]`` for its own block.  With the fixture table that is row
    ``1`` for the elite slots of block ``0`` and row ``6`` for those of block ``1``,
    so a mis-sliced block or a wrong elite width shows up as an exact row mismatch
    rather than a fuzzy set difference.  The widths cover a partial first block
    (``1``), the suite default (``2``) and a width that swallows a whole three-slot
    block (``3``).  Non-elite slots are still members of their own species, so the
    pin cannot be "achieved" by drawing garbage elsewhere.

    ``survival_threshold = 1.0`` is what gives the assertion teeth: with the suite
    default of ``0.1`` every live species has a survivor count of one, so the plain
    draw would hand back the species best member anyway and *removing* the elite
    pinning would go unnoticed.  At ``1.0`` the survivor set is the whole species and
    only the pin can put the best member in both parent slots.
    """
    bookkeeping = species_state()
    winner, loser, elite = pair(
        species_kernel,
        controller(genome_elitism=genome_elitism, survival_threshold=1.0),
        bookkeeping,
        evoxc_key,
    )

    assert_parent_indices(winner, loser)
    loc, idx_in_block = expected_block_layout(SPAWN)
    keys = bookkeeping.species_keys
    idx2species = bookkeeping.idx2species

    elite_slots = np.flatnonzero(idx_in_block < genome_elitism)
    assert elite_slots.size, "the parametrised width must select at least one slot"
    for slot in elite_slots:
        key = float(keys[int(loc[slot])])
        best = int(species_ranking(FITNESS, idx2species, key)[0])
        assert int(winner[slot]) == best, (
            f"elite slot {slot} of species {key} drew winner {int(winner[slot])}, "
            f"expected the species best member {best}"
        )
        assert int(loser[slot]) == best, (
            f"elite slot {slot} of species {key} drew loser {int(loser[slot])}, "
            f"expected the species best member {best}"
        )
    # Membership only: an out-of-species parent fails inside ``rank_in_species``.
    for slot in range(POP_SIZE):
        key = float(keys[int(loc[slot])])
        for parent in (int(winner[slot]), int(loser[slot])):
            rank_in_species(parent, FITNESS, idx2species, key)


@pytest.mark.parametrize(
    ("spawn", "groups"),
    [
        (SPAWN, GROUPS_DEFAULT),
        (MIDDLE_EMPTY_SPAWN, GROUPS_MIDDLE_EMPTY),
    ],
    ids=["dense", "middle-empty"],
)
@pytest.mark.parametrize("genome_elitism", [0, 1, 2])
def test_elite_mask_covers_the_leading_slots_of_every_block(
    species, controller, species_state, species_kernel, evoxc_key,
    spawn, groups, genome_elitism,
):
    """Invariant: ``elite_mask`` is True exactly for ``idx_in_block(p) < genome_elitism``.

    Oracle: :func:`expected_block_layout` on the spawn vector, compared elementwise
    against the returned mask.  The mask is what ``NEAT.tell`` uses to keep the
    crossover child unmutated, so an off-by-one here would silently mutate the
    champion — and a mask that covered only the first block would let the second
    species lose its elites.  The second table has a live, zero-spawn species in the
    middle: a mask computed from the species *slot* instead of the block index would
    mark the wrong slots there.  ``genome_elitism = 0`` pins the empty-mask edge.
    """
    bookkeeping = species_table(species_state, groups)
    winner, loser, elite = pair(
        species_kernel,
        controller(genome_elitism=genome_elitism),
        bookkeeping,
        evoxc_key,
        spawn=spawn,
    )

    assert_parent_indices(winner, loser)
    assert elite.shape == (POP_SIZE,)
    assert elite.dtype == np.bool_, f"elite_mask dtype is {elite.dtype}"
    loc, idx_in_block = expected_block_layout(spawn)
    np.testing.assert_array_equal(elite, idx_in_block < genome_elitism)

    # The same statement block by block, so a failure names the block that is off.
    for block in range(SPECIES_SIZE):
        if int(spawn[block]) <= 0:
            assert not np.any(loc == block), (
                f"zero-spawn species slot {block} owns population slots"
            )
            continue
        slots = np.flatnonzero(loc == block)
        assert slots.size == int(spawn[block])
        np.testing.assert_array_equal(
            elite[slots], np.arange(int(spawn[block])) < genome_elitism
        )


# --------------------------------------------------------------------------- #
# winner / loser and the block layout
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("survival_threshold", "fitness"),
    [
        (1.0, FITNESS),
        (0.5, FITNESS),
        (1.0, TIED_FITNESS),
    ],
    ids=["all-members", "two-survivors", "tied-at-the-top"],
)
def test_the_fitter_parent_is_always_parent_one(
    species, controller, species_state, species_kernel, evoxc_key,
    survival_threshold, fitness,
):
    """Invariant: ``fitness[winner] >= fitness[loser]`` for every population slot.

    Oracle: the two drawn rows compared through the fitness vector — the very
    comparison the kernel documents (``>=``).  ``execute_crossover`` is
    order-sensitive (parent 1's structure is kept), so a flipped comparison does not
    merely swap two equal-quality parents: it changes the topology of every child.

    This inequality cannot tell ``>=`` from ``>`` (both satisfy it), so the tie
    *direction* is pinned separately by
    :func:`test_a_tie_hands_the_slot_to_the_father`.  ``genome_elitism = 0`` keeps
    the pinned slots (where both parents are the same row by construction) out of
    the comparison, so the draw is what is measured.  The three cases make the
    inequality strict somewhere for the tie-free table (survivor sets of five and
    three, and of two and one), while the tied table at least proves that equal
    fitness is not re-ordered into a "fitter loser".
    """
    bookkeeping = species_state()
    winner, loser, elite = pair(
        species_kernel,
        controller(genome_elitism=0, survival_threshold=survival_threshold),
        bookkeeping,
        evoxc_key,
        fitness=fitness,
    )

    assert_parent_indices(winner, loser)
    assert not np.any(elite), "genome_elitism = 0 must not mark any slot"
    drawn = np.asarray(fitness)
    assert np.all(drawn[winner] >= drawn[loser]), (
        "a slot made the less fit genome parent 1: "
        f"winner fitness {drawn[winner].tolist()} vs loser fitness {drawn[loser].tolist()}"
    )
    if np.unique(drawn).size == POP_SIZE:
        # Teeth: on a tie-free table at least one slot must have drawn two
        # differently-fitnessed parents, otherwise the >= above is vacuous.
        assert np.any(drawn[winner] > drawn[loser]), (
            "no slot drew two differently-fitnessed parents: "
            f"winner={winner.tolist()} loser={loser.tolist()}"
        )


def test_a_tie_hands_the_slot_to_the_father(
    common, controller, species_state, species_kernel, evoxc_key
):
    """Invariant: on a tie, ``winner`` is the father draw and ``loser`` the mother draw.

    Oracle: :func:`survive_bound`, :func:`ranking_matrix` and
    :func:`reproduce_draws` re-derive the kernel's single ``evoxc.random.randint``
    call and its ``flat_gather`` from the same key, shape, argument order and
    survivor bounds, and :func:`expected_block_layout` maps ``(species_slot,
    population_slot)`` onto the population slot.  The documented rule is ``>=``: a tie
    keeps the *first* draw as parent 1, and ``execute_crossover`` is order-sensitive,
    so ``>`` would silently rewrite the topology of every tied child.

    Every member of a species carries the same fitness here
    (:data:`TIED_SPECIES_FITNESS`), so **every** drawn pair is a tie and the
    ``winner != loser`` / ``np.any(slot_father != slot_mother)`` assertions keep the
    table from collapsing into self-crossovers, where both tie rules agree and the
    test would be vacuous.  ``genome_elitism = 0`` keeps the elite pinning (which
    makes both parents the same row) out of the comparison, and
    ``survival_threshold = 1.0`` makes each species' whole ranking drawable so the
    two draws of a pair can differ.
    """
    bookkeeping = species_table(species_state, GROUPS_DEFAULT)
    winner, loser, elite = pair(
        species_kernel,
        controller(genome_elitism=0, survival_threshold=1.0),
        bookkeeping,
        evoxc_key,
        fitness=TIED_SPECIES_FITNESS,
    )

    assert_parent_indices(winner, loser)
    assert not np.any(elite), "genome_elitism = 0 must not mark any slot"

    survive = survive_bound(bookkeeping.member_count, 1.0)
    # The bound is the documented clamp, cross-checked against its live-slot oracle
    # (the dead slots differ, see :func:`survive_bound`, and own no population slot).
    for slot in (0, 1):
        members = int(bookkeeping.member_count[slot])
        assert int(survive[slot]) == survivor_count(1.0, members)
    ranking = ranking_matrix(
        TIED_SPECIES_FITNESS, bookkeeping.idx2species, bookkeeping.species_keys
    )
    draws, father, mother = reproduce_draws(common, evoxc_key, survive, ranking)

    assert draws.shape == (SPECIES_SIZE, 2, POP_SIZE), draws.shape
    assert np.all(draws >= 0)
    assert np.all(draws < survive.reshape(SPECIES_SIZE, 1, 1))

    loc, idx_in_block = expected_block_layout(SPAWN)
    slot_father, slot_mother = slot_parents(father, mother, loc, idx_in_block)

    # The premise, asserted rather than assumed: the drawn pairs tie, and at least one
    # of them is a pair of *different* rows (otherwise >= and > agree everywhere).
    np.testing.assert_array_equal(
        TIED_SPECIES_FITNESS[winner],
        TIED_SPECIES_FITNESS[loser],
        err_msg="the drawn pairs are not a tie, so this table tests nothing about ties",
    )
    assert np.any(slot_father != slot_mother), (
        "no slot drew two different rows, so the tie rule is untested here: "
        f"father={slot_father.tolist()} mother={slot_mother.tolist()}"
    )

    for slot in range(POP_SIZE):
        block = int(loc[slot])
        assert int(winner[slot]) == int(slot_father[slot]), (
            f"slot {slot} (species {float(bookkeeping.species_keys[block])}) must keep "
            f"the father draw {int(slot_father[slot])} as parent 1 on a tie, but "
            f"returned {int(winner[slot])} (mother draw {int(slot_mother[slot])})"
        )
        assert int(loser[slot]) == int(slot_mother[slot]), (
            f"slot {slot} (species {float(bookkeeping.species_keys[block])}) must keep "
            f"the mother draw {int(slot_mother[slot])} as parent 2 on a tie, but "
            f"returned {int(loser[slot])} (father draw {int(slot_father[slot])})"
        )


def test_the_fitter_parent_wins_regardless_of_draw_order(
    common, controller, species_state, species_kernel, evoxc_key
):
    """Invariant: whichever draw is fitter becomes parent 1, in both draw orders.

    Oracle: :func:`reproduce_draws` and the comparison written out per slot —
    ``father`` if ``fitness[father] >= fitness[mother]``, the mother otherwise.  The
    tied table of :func:`test_a_tie_hands_the_slot_to_the_father` pins the tie
    direction; this one pins the *inequality*, on the distinct-fitness table at
    ``survival_threshold = 1.0``, where the reproduced draw makes the father the
    fitter parent in some slots and the mother in others.  An implementation that
    always took the father, or that swapped the two ``select`` arms, satisfies a bare
    ``fitness[winner] >= fitness[loser]`` assertion only when all draws agree — which
    they do not, and the ``difference > 0`` / ``difference < 0`` assertions below keep
    that premise from silently eroding.
    """
    bookkeeping = species_state()
    winner, loser, elite = pair(
        species_kernel,
        controller(genome_elitism=0, survival_threshold=1.0),
        bookkeeping,
        evoxc_key,
    )

    assert_parent_indices(winner, loser)
    assert not np.any(elite), "genome_elitism = 0 must not mark any slot"

    survive = survive_bound(bookkeeping.member_count, 1.0)
    ranking = ranking_matrix(FITNESS, bookkeeping.idx2species, bookkeeping.species_keys)
    _draws, father, mother = reproduce_draws(common, evoxc_key, survive, ranking)
    loc, idx_in_block = expected_block_layout(SPAWN)
    slot_father, slot_mother = slot_parents(father, mother, loc, idx_in_block)

    difference = FITNESS[slot_father] - FITNESS[slot_mother]
    assert np.any(difference > 0), (
        "no slot drew a fitter father, so 'the mother can win' is untested: "
        f"father={slot_father.tolist()} mother={slot_mother.tolist()}"
    )
    assert np.any(difference < 0), (
        "no slot drew a fitter mother, so 'the father can win' is untested: "
        f"father={slot_father.tolist()} mother={slot_mother.tolist()}"
    )

    father_wins = difference >= 0
    np.testing.assert_array_equal(
        winner,
        np.where(father_wins, slot_father, slot_mother),
        err_msg=(
            "winner is not the fitter of the two reproduced draws: "
            f"father={slot_father.tolist()} mother={slot_mother.tolist()}"
        ),
    )
    np.testing.assert_array_equal(
        loser,
        np.where(father_wins, slot_mother, slot_father),
        err_msg=(
            "loser is not the less fit of the two reproduced draws: "
            f"father={slot_father.tolist()} mother={slot_mother.tolist()}"
        ),
    )


def test_every_slot_draws_from_its_own_species_block(
    species, controller, species_state, species_kernel, raw_kernel, evoxc_key
):
    """Invariant: both parents of slot ``p`` are members of the species of block ``loc(p)``.

    Oracle: :func:`expected_block_layout` for ``loc`` (cross-checked against
    ``_block_layout`` itself, so a disagreement names which side is wrong) and an
    ``idx2species[parent] == species_keys[loc(p)]`` membership test on both parents,
    over a spawn vector with a live, zero-spawn species in the middle
    (``[3, 0, 5, 0]``) so the block index and the species slot index deliberately
    differ.  This is the ``flat = loc * pop_size + idx_in_block`` gather: an
    implementation that confused the flattened index with the species slot would
    cross two species' parents, and one that treated the zero-spawn species as a
    block would hand population slots to a species with no members.
    """
    spawn = MIDDLE_EMPTY_SPAWN
    bookkeeping = species_table(species_state, GROUPS_MIDDLE_EMPTY)
    winner, loser, _elite = pair(
        species_kernel, controller(genome_elitism=0), bookkeeping, evoxc_key, spawn=spawn
    )

    assert_parent_indices(winner, loser)
    loc, idx_in_block = expected_block_layout(spawn)
    kernel_loc, kernel_idx = raw_kernel("_block_layout", spawn)
    np.testing.assert_array_equal(kernel_loc, loc)
    np.testing.assert_array_equal(kernel_idx, idx_in_block)
    # The table is what makes this test bite: block 0 -> species 0, block 1 is empty,
    # block 2 -> species 2.  A species-slot/block-index mix-up lands on species 1.
    assert loc.tolist() == [0, 0, 0, 2, 2, 2, 2, 2]
    assert not np.any(loc == 1), "the empty species must own no population slot"

    keys = bookkeeping.species_keys
    idx2species = bookkeeping.idx2species
    for slot in range(POP_SIZE):
        owner = float(keys[int(loc[slot])])
        for parent in (int(winner[slot]), int(loser[slot])):
            assert float(idx2species[parent]) == owner, (
                f"slot {slot} (owned by species {owner}) drew population row "
                f"{parent} of species {float(idx2species[parent])}"
            )


# --------------------------------------------------------------------------- #
# randomness hygiene
# --------------------------------------------------------------------------- #


def test_draws_do_not_depend_on_the_numpy_global_rng(
    species, controller, species_state, species_kernel, evoxc_key
):
    """Invariant: the same key gives the same parents before and after reseeding numpy.

    Oracle: run ``_pair_parents`` twice with the same ``evoxc`` key while the numpy
    global RNG is at two different states (``np.random.seed(0)`` then
    ``np.random.seed(123456)``); all three outputs must be ``np.array_equal``, and the
    drawn parents must still be members of their own species.  Reproducibility is
    defined on ``(seed, config, shape)``, so a host RNG draw — or a host
    ``random.choice`` — would break every seeded experiment while a single-process
    "same key, same result" check still passed.
    """
    bookkeeping = species_state()
    settings = controller(genome_elitism=0, survival_threshold=1.0)
    saved = np.random.get_state()
    try:
        np.random.seed(0)
        first = pair(species_kernel, settings, bookkeeping, evoxc_key)
        np.random.seed(123456)
        second = pair(species_kernel, settings, bookkeeping, evoxc_key)
    finally:
        np.random.set_state(saved)

    for name, left, right in zip(("winner", "loser", "elite"), first, second):
        np.testing.assert_array_equal(
            np.asarray(left),
            np.asarray(right),
            err_msg=f"{name} changed when only the numpy global RNG state changed",
        )

    winner, loser, elite = first
    assert_parent_indices(winner, loser)
    assert not np.any(elite)
    # The draws are members of the slot's own species, not just deterministic.
    loc, _idx_in_block = expected_block_layout(SPAWN)
    keys = bookkeeping.species_keys
    for slot in range(POP_SIZE):
        owner = float(keys[int(loc[slot])])
        for parent in (int(winner[slot]), int(loser[slot])):
            rank_in_species(parent, FITNESS, bookkeeping.idx2species, owner)
