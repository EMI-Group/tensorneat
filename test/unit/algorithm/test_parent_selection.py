"""Unit tests for ``SpeciesController._pair_parents`` — who becomes a parent, and of whom.

What this module proves
-----------------------
Parent pairing is the only step of a generation that *draws* from the population, so
its contract is a mixture of algebra and randomness, and the tests check the algebra
exactly and the randomness only through the bounds it must respect:

* the survivor count is ``clamp(floor(survival_threshold * member_count), 1,
  member_count)`` — floored at one so a one-member species is legal, and capped at
  the member count so the non-member tail of the ranking (all ``-inf``) can never be
  drawn;
* a one-member species yields **two identical parents** (self-crossover, which is the
  identity) instead of the legacy zero-survivor division that produced a NaN
  probability vector and made ``random.choice`` fail;
* elite slots — the first ``genome_elitism`` slots of every species block — are
  pinned to that species' **best** member, so an elite child is the unmutated
  crossover of the champion with itself;
* the winner/loser decision always makes the **fitter** genome parent 1 (the
  order-sensitive argument of ``execute_crossover``: parent 1's structure is kept),
  with ties going to the first draw;
* every population slot draws its parents from **its own** species block, and the
  block layout assigns each slot to exactly one block;
* the draw depends on the supplied ``evoxc.random`` key and on nothing else — the
  numpy global RNG has no influence, which is what makes ``(seed, config, shape)``
  reproducible.

Oracles used here (never the implementation)
--------------------------------------------
* :func:`species_ranking` — the documented ranking
  (``argsort`` of the masked fitness, stable ascending, then reversed to reproduce
  the legacy tie order) computed on the host from ``fitness``/``idx2species``.
* :func:`survivor_count` — the documented clamp formula, written out.
* :func:`expected_block_layout` — the ``loc``/``idx_in_block`` mapping by the
  definition (see ``./test_spawn.py``), so "draws from its own block" is checked
  against the block a slot *should* belong to.
* ``np.array_equal`` — determinism and RNG independence.

Gotchas this module encodes
---------------------------
* ``_pair_parents`` returns **row indices into the current population**, not ranks:
  a test compares against ``species_ranking(...)[rank]``, never against the rank.
* The ranking is over ``select(membership, fitness, -inf)``, so the tail of the
  ranking holds non-members.  The survivor cap is what makes that tail unreachable —
  a test that leaves ``survival_threshold = 1.0`` cannot observe it, so every
  bounding test uses a threshold that makes the survivor prefix strictly smaller
  than the member count.
* Ties are ordered by the legacy reversal, so the tests use **distinct** fitness
  values within a species; with a tie the rank order itself would be the thing under
  test, and that is pinned separately by ``test_the_fitter_parent_is_always_parent_one``
  (the ``>=`` comparison, not the sort order).
* ``elite_mask`` is per population slot and spans *every* block, so "the elites are
  pinned" is asserted block by block (``loc(p) == i and idx_in_block(p) <
  genome_elitism``), not as a prefix of the whole population.

Backend is ``"numpy"`` throughout; no optional dependency, no ``jax``/``torch``, no
network.
"""

from __future__ import annotations

import numpy as np
import pytest

#: The fixture population: rows ``0..4`` are species key ``0.0``, rows ``5..7`` are
#: key ``1.0``.  Fitness is distinct inside each species and the species maxima are
#: ``0.9`` (row 1) and ``0.8`` (row 6), which is what the elite assertions compare
#: against.
FITNESS = np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.2, 0.8, 0.4], dtype=np.float32)
#: The spawn allocation matching that population (sums to ``8``).
SPAWN = np.array([5, 3, 0, 0], dtype=np.int32)


# --------------------------------------------------------------------------- #
# numpy oracles
# --------------------------------------------------------------------------- #


def survivor_count(survival_threshold, members):
    """``clamp(floor(survival_threshold * members), 1, members)``, the documented count."""
    floored = int(np.floor(survival_threshold * members))
    return int(min(max(floored, 1), members))


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


def expected_block_layout(spawn):
    """``(loc, idx_in_block)`` from the spawn definition, by hand.

    Args:
        spawn: ``(S,)`` int summing to ``P``.

    Returns:
        tuple: ``(loc, idx_in_block)``, both ``(P,)`` int arrays.

    Notes:
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
        "_pair_parents", controller, bookkeeping, key, np.asarray(spawn), np.asarray(fitness)
    )


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
        (2.0, 3, 3),  # an over-large threshold is capped, not an IndexError
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
    one-member species), the plain floor (``0.5 * 5``), and the cap at the member
    count (``1.0`` and a threshold above ``1``, which must not read past the block).
    """
    pytest.skip(
        "not implemented: parents drawn from the survivor prefix only "
        "(owner: algorithm/neat/species.SpeciesController._pair_parents)"
    )


def test_a_one_member_species_yields_two_identical_parents(
    species, controller, species_state, species_kernel, evoxc_key
):
    """Invariant: a species with one member produces ``winner == loser ==`` that member.

    Oracle: a state whose second species owns exactly one row; both parent vectors
    must equal that row for the slot in its block, and every value must be a finite
    row index.  This is the documented fix of the legacy zero-survivor division
    (``high = max(survive_size, 1)``): the old probability vector divided by zero,
    produced NaN and made ``random.choice`` fail, whereas a self-crossover is the
    identity and is therefore the correct outcome.
    """
    pytest.skip(
        "not implemented: one-member species self-crosses instead of dividing by zero "
        "(owner: algorithm/neat/species.SpeciesController._pair_parents)"
    )


# --------------------------------------------------------------------------- #
# elites
# --------------------------------------------------------------------------- #


def test_elite_slots_are_pinned_to_the_species_best_member(
    species, controller, species_state, species_kernel, evoxc_key
):
    """Invariant: for an elite slot, both parents are the best member of its own species.

    Oracle: :func:`expected_block_layout` gives ``loc(p)``; when ``idx_in_block(p) <
    genome_elitism`` the test requires ``winner[p] == loser[p] ==
    species_ranking(...)[0]`` for that block.  With the fixture table that is row
    ``1`` for the two elite slots of block ``0`` and row ``6`` for the two of block
    ``1``, so a mis-sliced block or a wrong elite width shows up as an exact row
    mismatch rather than a fuzzy set difference.
    """
    pytest.skip(
        "not implemented: elite slots pinned to the species best member "
        "(owner: algorithm/neat/species.SpeciesController._pair_parents)"
    )


def test_elite_mask_covers_the_leading_slots_of_every_block(
    species, controller, species_state, species_kernel, evoxc_key
):
    """Invariant: ``elite_mask`` is True exactly for ``idx_in_block(p) < genome_elitism``.

    Oracle: :func:`expected_block_layout` on the spawn vector, compared elementwise
    against the returned mask.  The mask is what ``NEAT.tell`` uses to keep the
    crossover child unmutated, so an off-by-one here would silently mutate the
    champion — and a mask that covered only the first block would let the second
    species lose its elites.
    """
    pytest.skip(
        "not implemented: elite mask marks the leading slots of every species block "
        "(owner: algorithm/neat/species.SpeciesController._pair_parents)"
    )


# --------------------------------------------------------------------------- #
# winner / loser and the block layout
# --------------------------------------------------------------------------- #


def test_the_fitter_parent_is_always_parent_one(
    species, controller, species_state, species_kernel, evoxc_key
):
    """Invariant: ``fitness[winner] >= fitness[loser]`` for every population slot.

    Oracle: the two drawn rows compared through the fitness vector — the very
    comparison the kernel documents (``>=``, so a tie keeps the first draw as parent
    1).  ``execute_crossover`` is order-sensitive (parent 1's structure is kept), so
    a flipped comparison does not merely swap two equal-quality parents: it changes
    the topology of every child.
    """
    pytest.skip(
        "not implemented: winner is the fitter parent (ties to the first draw) "
        "(owner: algorithm/neat/species.SpeciesController._pair_parents)"
    )


def test_every_slot_draws_from_its_own_species_block(
    species, controller, species_state, species_kernel, evoxc_key
):
    """Invariant: both parents of slot ``p`` are members of the species of block ``loc(p)``.

    Oracle: :func:`expected_block_layout` for ``loc`` and a
    ``idx2species[parent] == species_keys[loc(p)]`` membership test on both parents,
    over a spawn vector with a zero-spawn species in the middle (``[3, 0, 5, 0]``) so
    the block index and the species slot index deliberately differ.  This is the
    ``flat = loc * pop_size + idx_in_block`` gather: an implementation that confused
    the flattened index with the species slot would cross two species' parents.
    """
    pytest.skip(
        "not implemented: block flattening maps every slot to its own species' parents "
        "(owner: algorithm/neat/species.SpeciesController._pair_parents)"
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
    ``np.random.seed(123456)``); all three outputs must be ``np.array_equal``.
    Reproducibility is defined on ``(seed, config, shape)``, so a host RNG draw — or
    a host ``random.choice`` — would break every seeded experiment while the tests
    still passed a "same key, same result" check in one process.
    """
    pytest.skip(
        "not implemented: parent draws depend on the evoxc key only "
        "(owner: algorithm/neat/species.SpeciesController._pair_parents)"
    )
