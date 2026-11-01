"""Unit tests for ``SpeciesController.speciate`` — the two fixed-length scans.

What this module proves
-----------------------
Speciation turns "a population of networks" into "a population of species": it
writes one species key per individual into ``idx2species``, re-elects the centre
genome of every slot and materialises new species from dead slots when a group of
genomes is far from every existing centre.  It is written as *two*
``evoxc.scan``s over the static ``species_size`` axis instead of the pre-rewrite
data-dependent ``while_loop``, so the replacement must be observationally
identical:

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
  ``member_count`` and the elected centres stay the same;
* the centre genes keep the **split payload** contract of
  ``tensorneat/CONTEXT.md`` reconciliation #2: four arrays of shapes ``(S, N)``,
  ``(S, N, NA)``, ``(S, E, CK)`` and ``(S, E, CA)``, integers and floats never
  packed into one array.

Oracles used here (never the implementation)
--------------------------------------------
* **A controlled distance table.** ``speciate`` takes ``distance_fn`` as an
  argument, so the tests hand it a fake genome kernel (:func:`table_distance`) that
  ignores the real compatibility metric and reads a static ``(P, P)`` host matrix
  instead.  ``table[centre_identity, member_identity]`` is written out literally in
  each test, which is what makes "closer than the threshold", "strictly closer than
  the owner" and the ``argmin`` order testable at all.
* **An independent host re-derivation.** :func:`assign_by_hand` applies the two
  documented phases (nearest unassigned genome per live slot, then threshold
  membership with the ``d < o2c`` rule, then the leftover rule and the compare-sum
  member count) with plain Python loops over the same distance table, and the
  kernel's fields are compared to it — loops vs. scans, so a scan that writes one
  slot too many or masks one iteration too few shows up.
* **Identity, on the structural claims.** Determinism, the untouched centre rows
  and the permutation law are checked with ``np.testing.assert_array_equal``
  against the inputs (NaN-aware, so a ``NaN`` padding value compares equal).

Gotchas this module encodes
---------------------------
* ``idx2species`` holds species **keys**, not species slot indices.  Every live key
  is compared against ``species_keys`` (``NaN`` excluded) rather than against
  ``np.arange(species_size)``.
* ``distance_fn`` is genome-owned and population-shaped:
  ``distance_fn(state, centre, pop) -> (P,)`` with **both** payloads shaped
  ``(P, ...)``.  ``speciate`` slices the centre row out of the ``(S, ...)`` fields
  and broadcasts it, so :func:`table_distance` reads its identity key from
  ``centre.node_keys[0, IDENTITY_ROW]`` and would raise a shape error if the
  broadcast were missing.
* Scan 1 always claims the *nearest unassigned* genome for a live slot and does
  **not** apply the threshold, so the threshold rule is only observable when a
  genome survives scan 1 unclaimed — every threshold test builds ``S < P``.
* The post-pass gives every still-unassigned genome the **last live species key**,
  which is what makes the strict ``<`` observable: a genome at exactly the
  threshold ends up in the last species, a genome just below it in the nearest one.
* Scan 2 evaluates ``distance_fn`` for a **dead** slot too (the trip count is
  static and the centre rows are only masked out afterwards), so a stale centre row
  must still hold a valid table index.  "Never read" is therefore checked as
  "never written from a stale row": the sentinel rows come out byte-identical.

Backend is ``"numpy"`` throughout; no optional dependency, no ``jax``/``torch``,
no network.  The modules under test and the support helpers are reached through
the ``./conftest.py`` fixtures, which import ``tensorneat`` once at collection
time; this module builds nothing on its own.
"""

from __future__ import annotations

import dataclasses
from collections import namedtuple

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
SPECIES_SIZE = 4
#: Gene widths of the default gene stack: ``DefaultNode`` has four attribute
#: columns, ``DefaultConn`` two key columns and one ``weight`` attribute.
NODE_ATTR_WIDTH = 4
CONN_KEY_WIDTH = 2
CONN_ATTR_WIDTH = 1
#: Node row that carries the per-individual *identity key* in these tests.
#: ``support.payloads`` writes inputs to rows ``0..num_inputs-1`` and outputs to the
#: last ``num_outputs`` rows, so row ``2`` is the first hidden row a two-input,
#: one-output payload leaves free.
IDENTITY_ROW = 2
#: The controller's ``compatibility_threshold`` in every table below.
THRESHOLD = 2.0
#: A distance safely above ``compatibility_threshold``: "not a member of this
#: centre, and not close enough to steal into it".
FAR = 9.0
#: The four split payload entries ``speciate`` reads off ``state``.
SPLIT_POP_KEYS = ("pop_node_keys", "pop_node_attrs", "pop_conn_keys", "pop_conn_attrs")
#: Host-side population payload: the same four split arrays as ``GenomeTensors``,
#: as a namedtuple so ``algorithm_state(pop=...)`` registers the split entries and
#: the runner recurses into the arrays leaf by leaf.
PayloadTensors = namedtuple(
    "PayloadTensors", ("node_keys", "node_attrs", "conn_keys", "conn_attrs")
)


# --------------------------------------------------------------------------- #
# distance oracle, host re-derivation and execution helper
# --------------------------------------------------------------------------- #


def table_distance(table, seen=None):
    """Build a trace-time ``distance_fn`` that reads a static ``(P, P)`` matrix.

    Args:
        table: ``(P, P) float32`` host array, ``table[c, m]`` = distance from the
            centre whose identity key is ``c`` to the individual whose identity key
            is ``m``.  Every test writes it out literally (see
            :func:`distance_table`).
        seen: Optional list; every call appends the four shapes of the centre
            payload it received, which is how the shape test observes the broadcast.

    Returns:
        A callable with the genome-owned signature
        ``(state, centre: GenomeTensors, pop: GenomeTensors) -> (P,)``.  It reads
        both identities from ``IDENTITY_ROW`` of the node keys (the centre's at
        index ``0`` — the broadcast makes every row equal — and the members' in
        population order) and gathers the table twice, so a distance follows a
        genome through a permutation of the payload instead of following its row.

    Notes:
        This is deliberately *not* the compatibility metric: ``speciate`` only
        consumes distances, so controlling them directly is what makes the scan
        semantics (threshold, stealing, ``argmin`` order) observable.  The constant
        is built *inside* the callable because ``evoxc.constant`` is only legal
        during an active trace.
    """
    array = np.asarray(table, dtype=np.float32)

    def distance(state, centre, pop):
        if seen is not None:
            seen.append(
                (
                    tuple(centre.node_keys.shape),
                    tuple(centre.node_attrs.shape),
                    tuple(centre.conn_keys.shape),
                    tuple(centre.conn_attrs.shape),
                )
            )
        identity = centre.node_keys[0, IDENTITY_ROW]
        lookup = evoxc.constant(evoxc.from_numpy(array))
        row = evoxc.gather(lookup, identity, axis=0)
        members = pop.node_keys[:, IDENTITY_ROW]
        return evoxc.gather(row, members, axis=0)

    return distance


def distance_table(rows):
    """Build a ``(P, P)`` distance matrix from per-centre rows.

    Args:
        rows: ``{centre_identity: {member_identity: distance}}``.  Every cell not
            mentioned is ``FAR`` (above the threshold), and the diagonal defaults
            to ``0.0`` — a centre is always its own closest genome, which is what
            makes "the first unassigned genome becomes the new centre" observable.

    Returns:
        ``(P, P) float32``.  Distinct off-diagonal values are the caller's
        responsibility: a tie would make the ``argmin`` winner index-dependent.
    """
    table = np.full((POP_SIZE, POP_SIZE), FAR, dtype=np.float32)
    np.fill_diagonal(table, 0.0)
    for centre, row in rows.items():
        for member, value in row.items():
            table[centre, member] = np.float32(value)
    return table


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
        slot order and without a threshold; a dead slot claims nothing.  Phase 2
        applies ``d < compatibility_threshold and (unassigned or d < o2c)``; a dead
        slot with anything still unassigned first materialises a species (its key is
        the counter, which then advances by one, and its centre becomes the first
        unassigned genome) and the same threshold rule then runs against that new
        centre.  The leftover rule finally gives the remainder the last live key,
        and ``member_count`` is a compare-sum with ``NaN`` for dead slots.
    """
    table = np.asarray(distance_table, dtype=np.float64)
    keys = np.array(species_keys, dtype=np.float64)
    centres = np.array(centre_identities, dtype=np.int64)
    identities = np.asarray(member_identities, dtype=np.int64)
    population = identities.shape[0]

    def distances(slot):
        """Distances from slot's current centre to every individual, in pop order."""
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
        centres[slot] = identities[winner]
        assigned[winner] = keys[slot]
        closest_so_far[winner] = 0.0

    # Scan 2: threshold membership; a dead slot is materialised from the first
    # unassigned genome, which then competes like any other centre (`o2c = 0` is
    # not written here: it comes from the self-distance of the new centre).
    for slot in range(keys.shape[0]):
        if np.isnan(keys[slot]):
            pending = np.where(np.isnan(assigned))[0]
            if pending.size == 0:
                continue
            keys[slot] = counter
            counter += 1.0
            centres[slot] = identities[pending[0]]
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
    bakes everything else as a constant, so the bookkeeping fields, ``generation``
    **and** the four split population arrays travel positionally; the ``State`` /
    ``SpeciesState`` containers are rebuilt inside the traced closure.

    Args:
        species: The ``tensorneat.algorithm.neat.species`` module fixture.
        controller: A ``SpeciesController`` from the ``controller`` factory fixture.
        state: The ``State`` to speciate, from :func:`algorithm_state`.
        distance_fn: The trace-time distance callable (see :func:`table_distance`).

    Returns:
        dict: field name → host numpy array, for the ``SpeciesState`` the call
        returned (a ``dict`` because ``to_numpy`` only rebuilds containers).

    Notes:
        ``generation`` is a graph *input*, not a baked constant: the post-pass writes
        it into ``last_improved`` through ``scalar_f32``, which rejects a numpy
        scalar.  The field order is ``dataclasses.fields`` order, so a new
        ``SpeciesState`` field costs no change here.
    """
    names = [field.name for field in dataclasses.fields(species.SpeciesState)]
    bookkeeping = [np.asarray(getattr(state.species, name)) for name in names]
    generation = np.asarray(state.generation, dtype=np.float32)
    payload = [np.asarray(getattr(state, name)) for name in SPLIT_POP_KEYS]

    def kernel(*values):
        width = len(names)
        inner = state.update(
            species=species.SpeciesState(*values[:width]),
            generation=values[width],
            **dict(zip(SPLIT_POP_KEYS, values[width + 1 :])),
        )
        updated = controller.speciate(inner, distance_fn)
        return tuple(getattr(updated.species, name) for name in names)

    result = to_numpy(run_defn(kernel, *(bookkeeping + [generation] + payload)))
    return dict(zip(names, result))


def centre_fields(payloads, identities):
    """Build the four split centre arrays, one identity key per species slot.

    Args:
        payloads: The ``support.payloads`` module fixture (``I_INF``).
        identities: ``(S,)`` sequence of table indices.  A **live** slot's row is
            overwritten by the elected genome during scan 1, so only a *dead*
            slot's row survives into the output — and there the value is exactly the
            "stale integer no kernel may read" the dead-slot test asserts on.

    Returns:
        dict: the four ``SpeciesState`` centre fields
        (``center_node_keys``/``center_node_attrs``/``center_conn_keys``/
        ``center_conn_attrs``) with the documented shapes and dtypes.

    Notes:
        Every row is padded with the sentinel of *its own* array plus a per-slot
        marker, so "the dead rows came back unchanged" is an assertion on the whole
        row.  The identity cell must stay a valid table index: ``speciate`` evaluates
        ``distance_fn`` for a dead slot too (its result is masked out afterwards).
    """
    slot_count = len(identities)
    node_keys = np.full((slot_count, MAX_NODES), payloads.I_INF, dtype=np.int32)
    node_attrs = np.full(
        (slot_count, MAX_NODES, NODE_ATTR_WIDTH), np.nan, dtype=np.float32
    )
    conn_keys = np.full(
        (slot_count, MAX_CONNS, CONN_KEY_WIDTH), payloads.I_INF, dtype=np.int32
    )
    conn_attrs = np.full(
        (slot_count, MAX_CONNS, CONN_ATTR_WIDTH), np.nan, dtype=np.float32
    )
    for slot, identity in enumerate(identities):
        node_keys[slot, 0] = -1 - slot  # a stale integer in a row nothing reads
        node_keys[slot, IDENTITY_ROW] = identity
        node_attrs[slot, IDENTITY_ROW, 0] = float(slot)  # per-slot stale marker
        conn_keys[slot, 0] = (100 + slot, 200 + slot)
        conn_attrs[slot, 0, 0] = float(slot) + 0.5
    return dict(
        center_node_keys=node_keys,
        center_node_attrs=node_attrs,
        center_conn_keys=conn_keys,
        center_conn_attrs=conn_attrs,
    )


def bookkeeping(species_state, payloads, *, keys, identities, next_species_key=2.0,
                member_count=None, best_fitness=None, last_improved=None):
    """Build the ``SpeciesState`` a test hands to ``speciate``.

    Args:
        species_state: The ``species_state`` factory fixture.
        payloads: The ``support.payloads`` module fixture.
        keys: ``(S,)`` species keys; ``NaN`` marks a dead slot.
        identities: ``(S,)`` centre identity per slot (see :func:`centre_fields`).
        next_species_key: 0-d counter before the call.
        member_count: Optional ``(S,)`` override — the dead-slot test sets a finite
            count there to pin that "dead" is ``isnan`` of the key alone.
        best_fitness: Optional ``(S,)`` override.
        last_improved: Optional ``(S,)`` override.

    Returns:
        The ``SpeciesState``, ``idx2species`` at the in-progress ``NaN`` sentinel.
    """
    slots = np.asarray(keys, dtype=np.float32)
    dead = np.isnan(slots)
    if member_count is None:
        member_count = np.where(dead, np.nan, 0.0).astype(np.float32)
    if best_fitness is None:
        best_fitness = np.where(dead, np.nan, 0.5).astype(np.float32)
    if last_improved is None:
        last_improved = np.where(dead, np.nan, 0.0).astype(np.float32)
    return species_state(
        species_keys=slots,
        best_fitness=np.asarray(best_fitness, dtype=np.float32),
        last_improved=np.asarray(last_improved, dtype=np.float32),
        member_count=np.asarray(member_count, dtype=np.float32),
        idx2species=np.full(POP_SIZE, np.nan, dtype=np.float32),
        next_species_key=np.float32(next_species_key),
        **centre_fields(payloads, identities),
    )


def permuted(population, order):
    """Return the payload with its population axis permuted by ``order``.

    ``order[i]`` is the *old* row that ends up at new row ``i``, so a genome's
    identity key (written into its own payload row) travels with it.
    """
    return PayloadTensors(*(np.asarray(array)[order] for array in population))


# --------------------------------------------------------------------------- #
# fixtures: a population whose identity keys are its population indices
# --------------------------------------------------------------------------- #


@pytest.fixture
def population(payloads):
    """A four-array payload whose ``IDENTITY_ROW`` key is the population index.

    Every individual keeps the two input rows that a genome's row/key invariant
    requires and stores its own population index in ``IDENTITY_ROW`` — so a test can
    address a genome in the distance table by its identity instead of by its
    (mutable) row position, and permuting the payload moves the identity with the
    genome.

    Returns:
        A :class:`PayloadTensors` of host numpy arrays, ready for
        ``algorithm_state(..., pop=population)``.
    """
    payload = payloads.new_payload(
        pop_size=POP_SIZE, max_nodes=MAX_NODES, max_conns=MAX_CONNS
    )
    for p in range(payload.pop_size):
        payloads.set_node(payload, p, 0, 0)
        payloads.set_node(payload, p, 1, 1)
        payloads.set_node(payload, p, IDENTITY_ROW, p)
    return PayloadTensors(
        payload.node_keys, payload.node_attrs, payload.conn_keys, payload.conn_attrs
    )


# --------------------------------------------------------------------------- #
# the split-payload contract of the centre genes
# --------------------------------------------------------------------------- #


def test_centre_genes_keep_the_split_payload_contract(
    species, controller, species_state, algorithm_state, payloads, population
):
    """Invariant: the centre genes are four split arrays of the documented shapes.

    Oracle: the field names of ``SpeciesState`` plus the shape/dtype of every
    returned centre array — ``(S, N) int32``, ``(S, N, NA) float32``,
    ``(S, E, CK) int32``, ``(S, E, CA) float32`` — and the shapes the fake
    ``distance_fn`` actually received, which must be the *population-shaped*
    broadcast ``(P, ...)`` of one centre row (``speciate`` slices and broadcasts;
    the genome never does).

    This pins ``src/tensorneat/CONTEXT.md`` reconciliation #2: a centre is the split
    ``GenomeTensors`` payload, never a packed integer gene row, and a packed form
    would change at least one of the ranks asserted here.
    """
    identities = [0, 2, 5, 6]
    state = algorithm_state(
        bookkeeping(
            species_state,
            payloads,
            keys=[0.0, 1.0, np.nan, np.nan],
            identities=identities,
        ),
        pop=population,
        generation=4.0,
    )
    seen = []
    result = run_speciate(
        species, controller(), state, table_distance(distance_table({}), seen)
    )

    names = {field.name for field in dataclasses.fields(species.SpeciesState)}
    assert {"center_node_keys", "center_node_attrs", "center_conn_keys",
            "center_conn_attrs"} <= names
    assert not {"center_nodes", "center_conns"} & names, (
        "the centre genes are split; a packed gene row violates the payload contract"
    )

    expected = {
        "center_node_keys": ((SPECIES_SIZE, MAX_NODES), np.int32),
        "center_node_attrs": (
            (SPECIES_SIZE, MAX_NODES, NODE_ATTR_WIDTH),
            np.float32,
        ),
        "center_conn_keys": (
            (SPECIES_SIZE, MAX_CONNS, CONN_KEY_WIDTH),
            np.int32,
        ),
        "center_conn_attrs": (
            (SPECIES_SIZE, MAX_CONNS, CONN_ATTR_WIDTH),
            np.float32,
        ),
    }
    for name, (shape, dtype) in expected.items():
        assert result[name].shape == shape, f"{name}: {result[name].shape} != {shape}"
        assert result[name].dtype == dtype, f"{name}: {result[name].dtype} != {dtype}"

    broadcast = (
        (POP_SIZE, MAX_NODES),
        (POP_SIZE, MAX_NODES, NODE_ATTR_WIDTH),
        (POP_SIZE, MAX_CONNS, CONN_KEY_WIDTH),
        (POP_SIZE, MAX_CONNS, CONN_ATTR_WIDTH),
    )
    assert seen, "speciate never called distance_fn"
    assert all(shapes == broadcast for shapes in seen), (
        f"distance_fn must receive population-shaped centre payloads {broadcast}, "
        f"got {sorted(set(seen))}"
    )


# --------------------------------------------------------------------------- #
# the post-condition of both scans
# --------------------------------------------------------------------------- #


def test_every_genome_is_assigned_to_exactly_one_live_species(
    species, controller, species_state, algorithm_state, payloads, population
):
    """Invariant: after ``speciate`` no ``NaN`` survives and every key is a live slot's key.

    Oracle: ``np.isnan(idx2species).sum() == 0``, plus "every assigned value is in
    ``species_keys[~isnan]``" (a set test, not a range test — the assignment holds
    *keys*, not slot indices), plus the per-key counts equal ``member_count`` and
    sum to ``pop_size``, plus :func:`assign_by_hand` field by field.  A scan that
    skips one iteration leaves ``NaN`` behind; a scan that writes the slot index
    instead of the key leaves a value outside the live key set.

    The table produces two claimed genomes, two fresh species (one per dead slot)
    and two leftovers, so the post-pass, both scans and the compare-sum are all
    exercised by the same assertion.
    """
    table = distance_table({0: {1: 0.5}, 2: {3: 1.0}})
    keys = [0.0, 1.0, np.nan, np.nan]
    identities = [0, 2, 5, 6]
    state = algorithm_state(
        bookkeeping(species_state, payloads, keys=keys, identities=identities),
        pop=population,
        generation=4.0,
    )
    result = run_speciate(species, controller(), state, table_distance(table))
    expected = assign_by_hand(
        table, keys, identities, np.arange(POP_SIZE), 2.0, THRESHOLD
    )

    live = ~np.isnan(result["species_keys"])
    live_keys = np.sort(result["species_keys"][live])
    assigned = result["idx2species"]
    assert not np.isnan(assigned).any(), f"unassigned genomes: {assigned}"
    assert set(assigned.tolist()) <= set(live_keys.tolist()), (
        f"{sorted(set(assigned.tolist()))} is not a subset of the live keys "
        f"{live_keys.tolist()} — idx2species holds keys, not slot indices"
    )
    # Per-key counts, matched by key value rather than by slot position.
    for key, count in zip(result["species_keys"][live], result["member_count"][live]):
        assert np.sum(assigned == key) == count, (
            f"member_count says {count} for key {key}, the assignment says "
            f"{np.sum(assigned == key)}"
        )
    assert int(np.sum(result["member_count"][live])) == POP_SIZE

    np.testing.assert_array_equal(assigned, expected["idx2species"])
    np.testing.assert_array_equal(result["member_count"], expected["member_count"])
    assert float(result["next_species_key"]) == float(expected["next_species_key"])


@pytest.mark.parametrize(
    "offset, expected_key",
    [(-1e-3, 0.0), (0.0, 1.0), (1e-3, 1.0)],
    ids=["below-threshold", "exactly-at-threshold", "above-threshold"],
)
def test_a_genome_closer_than_the_threshold_joins_the_centre(
    species,
    controller,
    species_state,
    algorithm_state,
    payloads,
    population,
    offset,
    expected_key,
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
    table = distance_table({0: {1: THRESHOLD + offset}})
    keys = [0.0, 1.0]
    identities = [0, 2]
    state = algorithm_state(
        bookkeeping(species_state, payloads, keys=keys, identities=identities),
        pop=population,
        generation=4.0,
    )
    result = run_speciate(
        species, controller(species_size=2), state, table_distance(table)
    )
    expected = assign_by_hand(
        table, keys, identities, np.arange(POP_SIZE), 2.0, THRESHOLD
    )

    assignment = result["idx2species"]
    assert assignment[1] == expected_key, (
        f"genome 1 sits {offset:+g} from the threshold and landed in "
        f"{assignment[1]}, expected key {expected_key}"
    )
    # genomes 0 and 2 were claimed by scan 1, and 3..7 fall through to the leftover
    # rule (the last live key, 1.0), so the whole vector is checkable.
    np.testing.assert_array_equal(
        assignment, [0.0, expected_key, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(assignment, expected["idx2species"])
    np.testing.assert_array_equal(result["member_count"], expected["member_count"])


@pytest.mark.parametrize(
    "later, expected_key",
    [(1.0, 1.0), (1.95, 0.0)],
    ids=["closer-later-slot-steals", "farther-later-slot-keeps"],
)
def test_a_member_closer_than_its_owner_steals_the_slot(
    species,
    controller,
    species_state,
    algorithm_state,
    payloads,
    population,
    later,
    expected_key,
):
    """Invariant: a later slot with ``d < o2c`` reassigns the genome, so scan order cannot freeze a claim.

    Oracle: two live slots (keys ``0.0``, ``1.0``; centre identities ``0`` and
    ``2``) and a genome (identity ``1``) that scan 1 leaves unassigned.
    ``table[0, 1] = threshold - 0.1`` (slot ``0`` claims it first in scan 2) and
    ``table[2, 1]`` is parametrised: ``1.0`` is strictly closer, so the genome must
    end in ``species_keys[1]``; ``1.95`` is farther, so it must stay in
    ``species_keys[0]``.  The mirror case pins that the rule is ``d < o2c`` and not
    "the last slot always wins".
    """
    table = distance_table({0: {1: 1.9}, 2: {1: later}})
    keys = [0.0, 1.0]
    identities = [0, 2]
    state = algorithm_state(
        bookkeeping(species_state, payloads, keys=keys, identities=identities),
        pop=population,
        generation=4.0,
    )
    result = run_speciate(
        species, controller(species_size=2), state, table_distance(table)
    )
    expected = assign_by_hand(
        table, keys, identities, np.arange(POP_SIZE), 2.0, THRESHOLD
    )

    assignment = result["idx2species"]
    assert assignment[1] == expected_key, (
        f"slot 1 is at distance {later} from genome 1 (slot 0 claimed it at 1.9); "
        f"expected key {expected_key}, got {assignment[1]}"
    )
    np.testing.assert_array_equal(
        assignment, [0.0, expected_key, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(assignment, expected["idx2species"])


def test_a_one_member_species_keeps_a_finite_member_count(
    species, controller, species_state, algorithm_state, payloads, population
):
    """Invariant: a live slot with exactly one member reports ``member_count == 1.0``, never ``NaN``.

    Oracle: a ``(species_size, pop_size)`` compare-sum recomputed on the host from
    the returned ``idx2species`` and ``species_keys`` — the same primitive the
    post-pass documents.  Slot ``0`` is set up to absorb everything except the
    genome that scan 1 already elected as slot ``1``'s centre, so slot ``1`` ends
    with one member: the case that made the legacy
    ``floor(threshold * members) / survive_size`` divide by zero, and the reason the
    count has to stay a real number for spawn allocation to work.
    """
    table = distance_table({0: {}, 2: {}})
    table[0, :] = 0.0  # every genome is at distance 0 from slot 0's centre
    table[2, :] = FAR  # ... except the one genome at slot 1's centre
    table[2, 2] = 0.0
    keys = [0.0, 1.0]
    identities = [0, 2]
    state = algorithm_state(
        bookkeeping(species_state, payloads, keys=keys, identities=identities),
        pop=population,
        generation=4.0,
    )
    result = run_speciate(
        species, controller(species_size=2), state, table_distance(table)
    )

    assignment = result["idx2species"]
    counts = np.array(
        [np.sum(assignment == key) for key in result["species_keys"]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result["member_count"], counts)
    assert counts[1] == 1.0, f"expected a one-member species, got {counts.tolist()}"
    assert np.isfinite(result["member_count"][1])
    np.testing.assert_array_equal(result["idx2species"], [0, 0, 1, 0, 0, 0, 0, 0])


def test_dead_slots_carry_nan_keys_and_their_centre_rows_are_never_read(
    species, controller, species_state, algorithm_state, payloads, population
):
    """Invariant: a dead slot stays dead, and its four centre rows come back byte-identical.

    Oracle: the centre rows of the dead slots are pre-filled with per-slot sentinel
    integers (:func:`centre_fields`) that no elected genome produces, and the call
    must return them unchanged; meanwhile each live slot's centre equals the whole
    population row of the genome scan 1 elected (``center_*[i] == pop[identity]``).
    "Never read" is therefore checked as "never written from a stale row".

    Deadness is pinned to ``isnan(species_keys)`` by filling the dead slots'
    ``member_count``/``best_fitness``/``last_improved`` with **finite** values first:
    an implementation keying off ``member_count`` would treat them as live and
    either claim a genome or rewrite their centre rows.  The table assigns every
    genome before the dead slots are reached, so no species is created and the
    counter must not move.
    """
    table = distance_table({0: {}, 2: {2: 0.0}})
    table[0, :] = 0.0  # slot 0 absorbs everyone ...
    table[2, :] = FAR  # ... except genome 2, which slot 1 already claimed
    table[2, 2] = 0.0
    keys = [0.0, 1.0, np.nan, np.nan]
    identities = [0, 2, 5, 6]
    book = bookkeeping(
        species_state,
        payloads,
        keys=keys,
        identities=identities,
        member_count=[5.0, 3.0, 3.0, 3.0],  # finite on the dead slots on purpose
        best_fitness=[0.5, 0.2, 0.1, 0.1],
        last_improved=[0.0, 3.0, 5.0, 5.0],
    )
    dead_before = {
        field.name: np.array(getattr(book, field.name))
        for field in dataclasses.fields(species.SpeciesState)
        if field.name.startswith("center_")
    }
    state = algorithm_state(book, pop=population, generation=4.0)
    result = run_speciate(species, controller(), state, table_distance(table))

    dead = np.isnan(result["species_keys"])
    np.testing.assert_array_equal(dead, [False, False, True, True])
    assert np.isnan(result["member_count"][dead]).all(), (
        "a dead slot's member_count must be NaN after the post-pass"
    )
    assert not np.isnan(result["member_count"][~dead]).any()
    assert float(result["next_species_key"]) == 2.0, (
        "a dead slot that had nothing to seed must not consume a species key"
    )

    for name, before in dead_before.items():
        np.testing.assert_array_equal(
            result[name][dead], before[dead], err_msg=f"{name}: dead rows were written"
        )
        assert result[name].dtype == before.dtype

    # The live slots were re-elected from the population: the whole row, not only
    # the identity cell, comes from the genome that won the slot.
    for slot, genome in enumerate([0, 2]):
        np.testing.assert_array_equal(
            result["center_node_keys"][slot], population.node_keys[genome]
        )
        np.testing.assert_array_equal(
            result["center_node_attrs"][slot], population.node_attrs[genome]
        )
        np.testing.assert_array_equal(
            result["center_conn_keys"][slot], population.conn_keys[genome]
        )
        np.testing.assert_array_equal(
            result["center_conn_attrs"][slot], population.conn_attrs[genome]
        )
    # A dead slot is not a created species: its bookkeeping is left alone.
    np.testing.assert_array_equal(
        result["best_fitness"][dead], np.float32([0.1, 0.1])
    )


@pytest.mark.parametrize(
    "rows, expected_keys, expected_assignment, expected_members, expected_next_key",
    [
        (
            {0: {0: 0.0, 1: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0},
             2: {2: 0.0}},
            [0.0, 1.0, np.nan, np.nan],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7.0, 1.0, np.nan, np.nan],
            2.0,
        ),
        (
            {0: {0: 0.0}, 2: {2: 0.0},
             1: {1: 0.0, 3: 0.5, 4: 1.0, 5: 1.5, 6: 1.9, 7: 1.9}},
            [0.0, 1.0, 2.0, np.nan],
            [0.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 6.0, np.nan],
            3.0,
        ),
        (
            {0: {0: 0.0}, 2: {2: 0.0}, 1: {1: 0.0},
             3: {3: 0.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}},
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 2.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [1.0, 1.0, 1.0, 5.0],
            4.0,
        ),
    ],
    ids=["no-new-species", "one-new-species", "two-new-species"],
)
def test_each_created_species_consumes_exactly_one_species_key(
    species,
    controller,
    species_state,
    algorithm_state,
    payloads,
    population,
    rows,
    expected_keys,
    expected_assignment,
    expected_members,
    expected_next_key,
):
    """Invariant: ``next_species_key`` grows by exactly one per species actually created.

    Oracle: the state starts at ``next_species_key = 2.0`` with two live slots and
    two dead slots; the table forces ``n`` new species (``n = 0, 1, 2``).  The
    returned counter must be ``2 + n``, the ``n`` created slots must carry exactly
    the keys ``2 .. 2 + n - 1`` in creation order (no gaps, no reuse of the initial
    ``0``), and a dead slot that finds nothing to seed must **not** consume a key —
    that is the ``n = 0`` case, where two dead slots stay ``NaN`` and the counter
    does not move.
    """
    table = distance_table(rows)
    keys = [0.0, 1.0, np.nan, np.nan]
    identities = [0, 2, 5, 6]
    state = algorithm_state(
        bookkeeping(species_state, payloads, keys=keys, identities=identities),
        pop=population,
        generation=4.0,
    )
    result = run_speciate(species, controller(), state, table_distance(table))

    created = int(np.sum(~np.isnan(np.asarray(expected_keys, np.float32))[2:]))
    assert float(result["next_species_key"]) == expected_next_key == 2.0 + created
    np.testing.assert_array_equal(
        result["species_keys"], np.asarray(expected_keys, np.float32)
    )
    np.testing.assert_array_equal(
        result["idx2species"], np.asarray(expected_assignment, np.float32)
    )
    np.testing.assert_array_equal(
        result["member_count"], np.asarray(expected_members, np.float32)
    )

    live_keys = result["species_keys"][~np.isnan(result["species_keys"])]
    assert len(set(live_keys.tolist())) == len(live_keys), (
        f"live species keys are not distinct: {live_keys.tolist()}"
    )
    new_keys = live_keys[live_keys >= 2.0]
    np.testing.assert_array_equal(new_keys, np.arange(2.0, 2.0 + created, dtype=np.float32))


def test_speciation_is_deterministic_for_the_same_state(
    species, controller, species_state, algorithm_state, payloads, population
):
    """Invariant: the same ``(state, distances)`` yields identical arrays on every run.

    Oracle: ``np.testing.assert_array_equal`` over all ``SpeciesState`` fields of
    two independent calls (a fresh ``distance_fn`` each time, so no mutable state
    hides in the fake).  Speciation must not touch ``randkey`` or any host RNG — it
    is a pure function of the distance table, and the pipeline relies on that to
    make a ``(seed, config, shape)`` triple reproducible.
    """
    table = distance_table({0: {1: 0.5}, 2: {3: 1.0}})
    keys = [0.0, 1.0, np.nan, np.nan]
    identities = [0, 2, 5, 6]
    state = algorithm_state(
        bookkeeping(species_state, payloads, keys=keys, identities=identities),
        pop=population,
        generation=4.0,
    )
    first = run_speciate(species, controller(), state, table_distance(table))
    second = run_speciate(species, controller(), state, table_distance(table))

    assert "randkey" not in state, (
        "this state carries no RNG key on purpose: speciate must not need one"
    )
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)
    assert not np.isnan(first["idx2species"]).any()


def test_speciation_permuting_the_population_only_permutes_the_keys(
    species, controller, species_state, algorithm_state, payloads, population
):
    """Invariant: a population permutation permutes ``idx2species`` and changes nothing else.

    Oracle: run once on the payload and once on ``payload[order]`` with the same
    species state and the **same** distance table (the identity of a genome is
    written inside its own payload row, so the table is addressed by genome, not by
    row).  Assert ``idx2species_permuted == idx2species[order]`` — a genome keeps
    the species of the genome it *is*, not of the row it sat in — plus unchanged
    ``member_count``, unchanged ``next_species_key`` and unchanged centre arrays.
    Distinct distances are used so no ``argmin`` tie makes the result
    index-dependent by definition.
    """
    rows = {
        0: {1: 0.5, 2: 0.7, 3: 1.1, 4: 2.5, 5: 2.7, 6: 2.9, 7: 3.1},
        4: {0: 3.3, 1: 2.2, 2: 2.6, 3: 1.9, 5: 0.4, 6: 0.8, 7: 1.4},
    }
    table = distance_table(rows)
    order = np.array([3, 0, 7, 2, 5, 1, 6, 4])
    keys = [0.0, 1.0]
    identities = [0, 4]
    book = bookkeeping(species_state, payloads, keys=keys, identities=identities)

    straight = algorithm_state(book, pop=population, generation=4.0)
    shuffled = algorithm_state(book, pop=permuted(population, order), generation=4.0)
    base = run_speciate(species, controller(species_size=2), straight,
                        table_distance(table))
    moved = run_speciate(species, controller(species_size=2), shuffled,
                         table_distance(table))

    np.testing.assert_array_equal(base["idx2species"], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(moved["idx2species"], base["idx2species"][order])
    np.testing.assert_array_equal(moved["member_count"], base["member_count"])
    assert float(moved["next_species_key"]) == float(base["next_species_key"])
    for name in ("center_node_keys", "center_node_attrs",
                 "center_conn_keys", "center_conn_attrs"):
        np.testing.assert_array_equal(moved[name], base[name], err_msg=name)
    np.testing.assert_array_equal(
        moved["species_keys"], base["species_keys"]
    )


def test_leftover_genomes_adopt_the_last_species_key_when_the_cap_is_hit(
    species, controller, species_state, algorithm_state, payloads, population
):
    """Invariant: when the species cap is reached, leftovers join the last live key, never ``NaN``.

    Oracle: a population with three clusters but only ``species_size = 2`` live
    slots, so the second scan runs out of dead slots; the uncapped group must carry
    exactly ``species_keys[last live slot]`` and ``member_count`` therefore exceeds
    the number of genomes near that centre.  The post-pass exists precisely so the
    "cap reached" path cannot leak the in-progress ``NaN`` sentinel into selection.

    The assertion is on the *key value*, not on the slot index, and it is
    cross-checked against :func:`assign_by_hand`.
    """
    rows = {0: {1: 0.5}, 4: {5: 0.5}}
    table = distance_table(rows)
    keys = [0.0, 1.0]
    identities = [0, 4]
    state = algorithm_state(
        bookkeeping(species_state, payloads, keys=keys, identities=identities),
        pop=population,
        generation=4.0,
    )
    result = run_speciate(
        species, controller(species_size=2), state, table_distance(table)
    )
    expected = assign_by_hand(
        table, keys, identities, np.arange(POP_SIZE), 2.0, THRESHOLD
    )

    assignment = result["idx2species"]
    live_keys = result["species_keys"][~np.isnan(result["species_keys"])]
    assert not np.isnan(assignment).any()
    np.testing.assert_array_equal(assignment, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    for genome in (2, 3, 6, 7):  # the cluster with no slot left for it
        assert assignment[genome] == live_keys[-1], (
            f"leftover genome {genome} got {assignment[genome]}, expected the last "
            f"live key {live_keys[-1]}"
        )
    np.testing.assert_array_equal(result["member_count"], [2.0, 6.0])
    assert result["member_count"][1] > 2, (
        "the leftover genomes must be counted into the species that adopted them"
    )
    np.testing.assert_array_equal(assignment, expected["idx2species"])
    np.testing.assert_array_equal(result["member_count"], expected["member_count"])
    assert float(result["next_species_key"]) == float(expected["next_species_key"])
