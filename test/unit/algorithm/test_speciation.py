"""Unit tests for ``SpeciesController.speciate`` — the two fixed-length scans.

What this module proves
-----------------------
Speciation is what turns "a population of networks" into "a population of species":
it writes one species key per individual into ``idx2species``, re-elects the centre
genome of every species and creates new species when a group of genomes is far
from every existing centre.  It is written as *two* ``evoxc.scan``s over the static
``species_size`` axis instead of the pre-rewrite data-dependent ``while_loop``, so
the replacement must be observationally identical:

* after the call **every** individual carries exactly one **live** species key —
  no ``NaN`` (the in-progress "unassigned" sentinel) and no stale key of a slot
  that died this generation;
* a genome whose distance to a centre is **strictly** below
  ``compatibility_threshold`` joins that species, one at exactly the threshold does
  not (it falls through to the leftover rule);
* a genome that is *closer* to a later species than to its current one is
  reassigned (scan 2's ``d < o2c``), so the scan order can never freeze a bad
  first claim;
* a one-member species keeps a **finite** ``member_count`` of ``1`` — the species
  algebra divides by it in spawn allocation and in the survivor count;
* a dead slot is recognisable by ``isnan(species_keys)`` alone, and its centre rows
  are never read: they hold stale integers (``int32`` cannot carry ``NaN``) and
  must come out of the call exactly as they went in;
* ``next_species_key`` advances by exactly one per species that is actually
  created — never per dead slot that had nothing to seed;
* the result is a pure function of ``(state, distances)``: running it twice yields
  identical arrays, and permuting the population permutes ``idx2species`` while
  ``member_count`` and the elected centres stay the same.

Oracles used here (never the implementation)
--------------------------------------------
* **A controlled distance table.** ``speciate`` takes ``distance_fn`` as an
  argument, so the tests hand it a fake genome kernel (:func:`table_distance`)
  that ignores the real compatibility metric and reads a static ``(P, P)`` host
  matrix instead. ``table[centre_identity, member_identity]`` is written out
  literally in each test, which is what makes "closer than the threshold",
  "strictly closer than the owner" and the ``argmin`` order testable at all.
* **An independent host re-derivation.** :func:`assign_by_hand` applies the two
  documented phases (nearest unassigned genome per live centre, then threshold
  membership with the ``d < o2c`` rule, then the leftover rule and the
  compare-sum member count) with plain Python loops over the same distance table,
  and the kernel's ``idx2species``/``member_count`` are compared field by field —
  a different algorithm (loops vs. scans), so a scan that writes one slot too many
  or masks one iteration too few shows up.
* **Identity, on the structural claims.** Determinism, the untouched centre rows
  and the permutation law are checked with ``np.array_equal`` against the inputs.

Gotchas this module encodes
---------------------------
* ``idx2species`` holds species **keys**, not species slot indices.  Every live key
  is compared against ``species_keys`` (``NaN`` excluded) rather than against
  ``np.arange(species_size)``.
* Scan 1 always claims the *nearest unassigned* genome for a live slot; it does
  **not** apply the threshold.  So the threshold rule is only observable when
  ``pop_size > live slots``, i.e. when a genome survives scan 1 unclaimed — every
  threshold test builds ``S < P`` for that reason.
* The post-pass gives every still-unassigned genome the **last live species key**.
  That is what makes the strict ``<`` observable: a genome at exactly the threshold
  ends up in the last species, a genome just below it in the nearest one.
* A dead slot's centre rows are integers; "never read" is checked by filling them
  with a sentinel no real payload would contain and asserting the sentinel
  survives, not by checking for ``NaN``.
* The permutation law is asserted on a table with **distinct** distances: with a
  tie in the ``argmin`` the winner is index-dependent by definition, and a
  permutation legitimately reorders it.

Backend is ``"numpy"`` throughout; no optional dependency, no ``jax``/``torch``,
no network.  ``tensorneat.algorithm.neat.species`` is reached through the
``./conftest.py`` fixtures because ``import tensorneat`` still raises from the
``src/`` design skeletons.
"""

from __future__ import annotations

import dataclasses

import evoxc
import numpy as np
import pytest

from support.runner import run_defn, to_numpy

#: Population, node and connection sizes of the tables below.  They mirror the
#: defaults of ``./conftest.py`` (``POP_SIZE``/``MAX_NODES``/``MAX_CONNS``) because
#: a species table and the population it is derived from must agree; the values are
#: repeated here only because ``conftest`` is not an importable module.
POP_SIZE = 8
MAX_NODES = 8
MAX_CONNS = 16
#: Node row that carries the per-individual *identity key* in these tests.
#: ``support.payloads`` writes inputs to rows ``0..num_inputs-1`` and outputs to the
#: last ``num_outputs`` rows, so row ``2`` is the first hidden row a two-input,
#: one-output payload leaves free.
IDENTITY_ROW = 2


# --------------------------------------------------------------------------- #
# distance oracle, host re-derivation and execution helper
# --------------------------------------------------------------------------- #


def table_distance(table):
    """Build a trace-time ``distance_fn`` that reads a static ``(P, P)`` matrix.

    Args:
        table: ``(P, P) float32`` host array, ``table[c, m]`` = distance from the
            centre whose identity key is ``c`` to the individual whose identity key
            is ``m``.  Every test writes it out literally.

    Returns:
        A callable with the genome-owned signature
        ``(state, center_nodes_i, center_conns_i, pop_nodes, pop_conns) -> (P,)``:
        it reads the centre's identity out of the centre gene row and returns the
        matching table row.  The lookup is a ``gather`` against a graph constant,
        never a host read of a traced value.

    Notes:
        This is deliberately *not* the compatibility metric: ``speciate`` only
        consumes distances, so controlling them directly is what makes the scan
        semantics (threshold, stealing, ``argmin`` order) observable.
        ``center_nodes_i`` is the ``(N, NL)`` gene-row slice the controller hands
        over; column ``0`` is the node key (``BaseGene.length`` puts the key
        columns first).
    """
    lookup = evoxc.constant(np.asarray(table, dtype=np.float32))

    def distance(state, center_nodes_i, center_conns_i, pop_nodes, pop_conns):
        centre_identity = center_nodes_i[IDENTITY_ROW, 0]
        return evoxc.gather(lookup, centre_identity, axis=0)

    return distance


def assign_by_hand(
    distance_table,
    species_keys,
    centre_identities,
    member_identities,
    next_species_key,
    compatibility_threshold,
):
    """Re-derive both scan phases with plain Python loops over the distance table.

    Args:
        distance_table: ``(P, P)`` host matrix, indexed by identity key.
        species_keys: ``(S,)`` float keys of the slots handed to ``speciate``
            (``NaN`` = dead slot).
        centre_identities: ``(S,)`` identity key of each slot's centre genome; the
            value of a dead slot is never read.
        member_identities: ``(P,)`` identity key of each population slot, in
            population order.
        next_species_key: 0-d float, the counter before the call.
        compatibility_threshold: The controller's threshold.

    Returns:
        dict: ``idx2species`` (``(P,)``), ``member_count`` (``(S,)``) and
        ``next_species_key``, computed by the documented rules only.

    Notes:
        Phase 1 reserves the nearest *unassigned* genome for every live slot, in
        slot order and without a threshold; a slot with nothing left to claim
        writes nothing.  Phase 2 applies
        ``d < compatibility_threshold and (unassigned or d < o2c)``, and a dead slot
        first creates a species from the first unassigned genome (its key is the
        counter, which then advances by one).  The leftover rule finally gives the
        remainder the last live key, and ``member_count`` is a compare-sum.
    """
    table = np.asarray(distance_table, dtype=np.float64)
    keys = np.array(species_keys, dtype=np.float64)
    centres = np.array(centre_identities, dtype=np.int64)
    identities = np.asarray(member_identities, dtype=np.int64)
    population = identities.shape[0]

    def distances(slot):
        """Distances from slot's centre to every individual, in population order."""
        return table[int(centres[slot])][identities]

    assigned = np.full(population, np.nan)  # population index -> species key
    closest_so_far = np.full(population, np.inf)  # the scan's `o2c`
    counter = float(next_species_key)

    # Scan 1: centre reservation, live slots only, nearest-unassigned argmin.
    for slot in range(keys.shape[0]):
        if np.isnan(keys[slot]):
            continue
        pending = np.isnan(assigned)
        if not pending.any():
            continue
        d = distances(slot)
        winner = int(np.argmin(np.where(pending, d, np.inf)))
        assigned[winner] = keys[slot]
        centres[slot] = identities[winner]
        closest_so_far[winner] = 0.0

    # Scan 2: threshold membership; a dead slot is materialised from the first
    # unassigned genome (which joins it as the centre, `o2c = 0`) and then assigns
    # against its own new centre.
    for slot in range(keys.shape[0]):
        if np.isnan(keys[slot]):
            pending = np.where(np.isnan(assigned))[0]
            if pending.size == 0:
                continue
            keys[slot] = counter
            counter += 1.0
            seed = pending[0]
            centres[slot] = identities[seed]
            assigned[seed] = keys[slot]
            closest_so_far[seed] = 0.0
        d = distances(slot)
        take = (d < compatibility_threshold) & (
            np.isnan(assigned) | (d < closest_so_far)
        )
        assigned = np.where(take, keys[slot], assigned)
        closest_so_far = np.where(take, d, closest_so_far)

    live_keys = keys[~np.isnan(keys)]
    if np.isnan(assigned).any():
        if live_keys.size == 0:  # no live species at all: nothing can adopt them
            raise AssertionError("speciation left genomes unassigned with no live species")
        assigned[np.isnan(assigned)] = live_keys[-1]

    members = np.array(
        [np.sum(assigned == key) if not np.isnan(key) else np.nan for key in keys],
        dtype=np.float32,
    )
    return dict(
        idx2species=assigned.astype(np.float32),
        member_count=members,
        next_species_key=np.float32(counter),
    )


def run_speciate(species, controller, state, distance_fn):
    """Run ``SpeciesController.speciate`` in one graph and return host numpy fields.

    ``evoxc.build`` only turns a ``numpy.ndarray`` argument into a graph input and
    bakes everything else as a constant, so the bookkeeping fields **and** the
    population travel positionally; the ``State`` / ``SpeciesState`` containers are
    rebuilt inside the traced closure.

    Args:
        species: The ``tensorneat.algorithm.neat.species`` module fixture.
        controller: A ``SpeciesController`` from the ``controller`` factory fixture.
        state: The ``State`` to speciate, from the ``algorithm_state`` factory (its
            ``species``/``pop_nodes``/``pop_conns`` may be numpy arrays — they are
            replaced by the graph inputs below).
        distance_fn: The trace-time distance callable (see :func:`table_distance`).

    Returns:
        dict: field name → host numpy array, for the ``SpeciesState`` the call
        returned.  A ``dict`` rather than the dataclass because ``to_numpy``
        rebuilds tuples/lists/dicts/namedtuples only.

    Notes:
        The field order comes from ``dataclasses.fields``, i.e. the dataclass
        declaration order — one place to change if ``SpeciesState`` grows a field.
    """
    names = [field.name for field in dataclasses.fields(species.SpeciesState)]
    arrays = [np.asarray(getattr(state.species, name)) for name in names]
    pop_arrays = [np.asarray(state.pop_nodes), np.asarray(state.pop_conns)]

    def kernel(*values):
        bookkeeping = species.SpeciesState(*values[: len(names)])
        pop_nodes, pop_conns = values[len(names) :]
        inner = state.update(
            species=bookkeeping, pop_nodes=pop_nodes, pop_conns=pop_conns
        )
        updated = controller.speciate(inner, distance_fn)
        return tuple(getattr(updated.species, name) for name in names)

    result = to_numpy(run_defn(kernel, *(arrays + pop_arrays)))
    return dict(zip(names, result))


# --------------------------------------------------------------------------- #
# fixtures: a population whose identity keys are its population indices
# --------------------------------------------------------------------------- #


@pytest.fixture
def population(payloads):
    """A payload ``(P, N)`` node keys / ``(P, C, CK)`` conn keys with identity keys.

    Every individual keeps the two input rows and the output row that a genome's
    row/key invariant requires, and stores its own population index in
    ``IDENTITY_ROW`` — so a test can address a genome in the distance table by its
    identity instead of by its (mutable) row position.
    """
    payload = payloads.new_payload(
        pop_size=POP_SIZE, max_nodes=MAX_NODES, max_conns=MAX_CONNS
    )
    for p in range(payload.pop_size):
        payloads.set_node(payload, p, 0, 0)
        payloads.set_node(payload, p, 1, 1)
        payloads.set_node(payload, p, IDENTITY_ROW, p)
        payloads.set_node(payload, p, payload.max_nodes - 1, payload.max_nodes - 1)
    return payload


# --------------------------------------------------------------------------- #
# the post-condition of both scans
# --------------------------------------------------------------------------- #


def test_every_genome_is_assigned_to_exactly_one_live_species(
    species, controller, species_state, algorithm_state, population
):
    """Invariant: after ``speciate`` no ``NaN`` survives and every key is a live slot's key.

    Oracle: ``np.isnan(idx2species).sum() == 0``, plus "every assigned value is in
    ``species_keys[~isnan]``" (a set test, not a range test — the assignment holds
    *keys*, not slot indices), plus the per-key counts equal ``member_count`` and
    sum to ``pop_size``.  A scan that skips one iteration leaves ``NaN`` behind; a
    scan that writes the slot index instead of the key leaves a value outside the
    live key set.
    """
    pytest.skip(
        "not implemented: idx2species fully assigned to live species keys after speciate "
        "(owner: algorithm/neat/species.SpeciesController.speciate)"
    )


def test_a_genome_closer_than_the_threshold_joins_the_centre(
    species, controller, species_state, algorithm_state, population
):
    """Invariant: ``d < compatibility_threshold`` joins, ``d == threshold`` does not (strict ``<``).

    Oracle: a distance table with two live slots (keys ``0.0``, ``1.0``; centre
    identities ``0`` and ``2``) and one genome (identity ``1``) that scan 1 leaves
    unclaimed.  ``species_size = 2`` so there is **no dead slot** that could seed a
    new species — the leftover rule (last live key ``1.0``) is therefore the only
    alternative owner.  ``table[0, 1]`` is parametrised over ``threshold - 1e-3``,
    ``threshold`` and ``threshold + 1e-3`` while ``table[2, 1]`` stays above the
    threshold.  Expected: the first case assigns key ``0.0``, the other two key
    ``1.0``.  Asserting the exact key is what distinguishes the scan-2 threshold
    write from the post-pass, since both are legal owners of the same genome.
    """
    pytest.skip(
        "not implemented: strict compatibility_threshold membership in scan 2 "
        "(owner: algorithm/neat/species.SpeciesController.speciate)"
    )


def test_a_member_closer_than_its_owner_steals_the_slot(
    species, controller, species_state, algorithm_state, population
):
    """Invariant: a later slot with ``d < o2c`` reassigns the genome, so scan order cannot freeze a claim.

    Oracle: two live slots (keys ``0.0``, ``1.0``) and a genome that scan 1 leaves
    unassigned; ``table[0, g] = threshold - 0.1`` (slot ``0`` claims it first) and
    ``table[1, g] = threshold - 1.0`` (strictly closer).  Expected key:
    ``species_keys[1]``.  The mirror case with ``table[1, g]`` *larger* keeps
    ``species_keys[0]``, which pins that the rule is ``d < o2c`` and not "the last
    slot always wins".
    """
    pytest.skip(
        "not implemented: scan-2 d < o2c reassignment between species "
        "(owner: algorithm/neat/species.SpeciesController.speciate)"
    )


def test_a_one_member_species_keeps_a_finite_member_count(
    species, controller, species_state, algorithm_state, population
):
    """Invariant: a live slot with exactly one member reports ``member_count == 1.0``, never ``NaN``.

    Oracle: a ``(species_size, pop_size)`` compare-sum recomputed on the host from
    the returned ``idx2species`` and ``species_keys`` — the same primitive the
    post-pass documents.  A species that ends up with one member is the case that
    made the legacy ``floor(threshold * members) / survive_size`` divide by zero, so
    the count has to stay a real number for spawn allocation to work.
    """
    pytest.skip(
        "not implemented: finite member_count for a one-member species "
        "(owner: algorithm/neat/species.SpeciesController.speciate)"
    )


def test_dead_slots_carry_nan_keys_and_their_centre_rows_are_never_read(
    species, controller, species_state, algorithm_state, population
):
    """Invariant: a dead slot stays dead, and its ``center_nodes``/``center_conns`` rows are untouched.

    Oracle: the centre rows of the dead slots are pre-filled with a sentinel
    (``-7``) that no gene row could legitimately contain, and the call must return
    them byte-identical; meanwhile every live slot's centre equals the gene rows of
    the population slot that ended up in it (``center_nodes[i] == pop[first member
    of i]``).  "Never read" is therefore checked as "never written from a stale
    row", and "dead" is checked to be the ``isnan(species_keys)`` mask alone — an
    implementation keying off ``member_count`` or a slot index fails one half.
    """
    pytest.skip(
        "not implemented: dead slots keep NaN keys and untouched centre rows "
        "(owner: algorithm/neat/species.SpeciesController.speciate)"
    )


def test_each_created_species_consumes_exactly_one_species_key(
    species, controller, species_state, algorithm_state, population
):
    """Invariant: ``next_species_key`` grows by exactly one per species actually created.

    Oracle: seed a state with ``next_species_key = k`` and a population that forces
    ``n`` new species; assert the returned counter is ``k + n``, that the ``n`` new
    keys are ``k .. k + n - 1`` (in creation order, no gaps, no reuse of ``0``) and
    that all live keys are distinct.  A dead slot that finds nothing to seed must
    **not** consume a key, which is why the test also covers a population that
    needs fewer species than there are dead slots.
    """
    pytest.skip(
        "not implemented: one new species key per created species, no gaps "
        "(owner: algorithm/neat/species.SpeciesController.speciate)"
    )


def test_speciation_is_deterministic_for_the_same_state(
    species, controller, species_state, algorithm_state, population
):
    """Invariant: the same ``(state, distances)`` yields identical arrays on every run.

    Oracle: ``np.array_equal`` over all ``SpeciesState`` fields of two independent
    calls.  Speciation must not touch ``randkey`` or any host RNG — it is a pure
    function of the distance table, and the pipeline relies on that to make a
    ``(seed, config, shape)`` triple reproducible.
    """
    pytest.skip(
        "not implemented: speciate is a pure function of (state, distances) "
        "(owner: algorithm/neat/species.SpeciesController.speciate)"
    )


def test_speciation_permuting_the_population_only_permutes_the_keys(
    species, controller, species_state, algorithm_state, population
):
    """Invariant: a population permutation permutes ``idx2species`` and changes nothing else.

    Oracle: run once on the payload and once on ``payload[perm]`` with the same
    species state and the permuted distance table; assert
    ``idx2species_permuted == idx2species[perm]`` (a genome keeps the species of the
    genome it *is*, not of the row it sat in), ``member_count`` unchanged, the
    ``next_species_key`` unchanged, and the multiset of live centre rows unchanged.
    Distinct distances are used so no ``argmin`` tie makes the result
    index-dependent by definition.
    """
    pytest.skip(
        "not implemented: permutation law of speciate "
        "(owner: algorithm/neat/species.SpeciesController.speciate)"
    )


def test_leftover_genomes_adopt_the_last_species_key_when_the_cap_is_hit(
    species, controller, species_state, algorithm_state, population
):
    """Invariant: when the species cap is reached, leftovers join the last live key, never ``NaN``.

    Oracle: a population with more clusters than live slots (``S = 2``, three
    well-separated groups) so the second scan runs out of dead slots; assert the
    uncapped group carries exactly ``species_keys[last live slot]`` and that
    ``member_count`` therefore exceeds the number of genomes near that centre.  The
    post-pass exists precisely so the "cap reached" path cannot leak the in-progress
    ``NaN`` sentinel into selection.
    """
    pytest.skip(
        "not implemented: leftover genomes adopt the last species key "
        "(owner: algorithm/neat/species.SpeciesController.speciate)"
    )
