"""Unit tests for the gated/dense seam of the speciation distance table.

What this module proves
-----------------------
``SpeciesController.speciate`` has two distance paths, selected at trace time by
the static predicate ``_kernels.gated(species_size, pop_size)``:

* **gated** (``species_size * pop_size <= _kernels.GATED_ROWS``, 6000): each scan
  iteration computes its own ``(pop_size,)`` row behind an ``evoxc.cond``, so a
  slot that cannot write never runs the compatibility kernel;
* **dense** (above the gate): scan 1 measures its whole centre table up front
  through ``distance_fn.matrix`` (``_distance.DistanceRows.matrix``) and every
  iteration reads its own row with a ``gather``.

Every population in the rest of the suite is far below the gate (``S * P`` tops
out around ``10 * 100`` in ``./integration``), so the dense branch -- the one
every real pop-500/1000 run takes -- is dead code in CI unless a test forces it.
``GATED_ROWS`` is only ever compared against static Python ints, so forcing it
is a module-attribute flip, not a graph change:

* :func:`test_dense_and_gated_tell_agree_on_the_same_state` drives a whole
  ``NEAT.tell`` twice from one immutable starting state, once with
  ``GATED_ROWS = 0`` and once with ``GATED_ROWS = 10**9``, and asserts the two
  results are identical field for field: species assignment, centre gene rows,
  the species bookkeeping, the next population, ``generation`` and ``randkey``.
* :func:`test_dense_branch_calls_matrix_once_and_the_gated_branch_never` wraps
  ``DistanceRows.matrix`` in a counting spy, so a branch that silently falls
  back to the other path fails instead of producing equal numbers.
* :func:`test_speciate_assigns_members_as_documented` calls ``_kernels.speciate``
  directly with ``gate`` passed explicitly (never read from ``GATED_ROWS``) on a
  hand-written ``(P + S, P)`` distance table, and compares the whole
  ``idx2species``, the re-elected centre rows, ``member_count``, the created
  keys and the created-slot bookkeeping with values derived by hand from the
  documented scan rules.  Both gate values are exercised on the same table.
* :func:`test_distance_rows_matrix_measures_one_centre_per_row` pins the real
  ``DistanceRows.matrix`` against ``DistanceRows.__call__`` centre-by-centre on
  a four-genome payload, which is the safety net for a future seam change that
  removes the module attribute the first three tests flip.

Oracles used here (never the implementation)
--------------------------------------------
The dense/gated comparison reads both runs off the *same* starting state, so
the oracle is the other branch (plus an independent spy on the call count).
The direct test's expectation is a table written from the documented scan rules
in ``SpeciesController.speciate``: scan 1 claims the nearest still-unassigned
genome per live slot, scan 2 assigns everything within
``compatibility_threshold`` that is unassigned or closer, a dead slot
materialises a species seeded by the first unassigned genome, and leftovers
adopt the last live species key.  The ``matrix`` test compares two spellings of
the same measurement, which must be bit-identical because the compatibility
kernel is row-independent.

Gotchas
-------
* ``GATED_ROWS`` is read through ``_kernels`` at trace time; patch the module
  attribute, not a local import of ``gated``.
* ``evoxc.build`` turns only an ``ndarray``/``Tensor`` argument into a graph
  input, so every ``State``/``SpeciesState`` is rebuilt *inside* the traced
  closure from positional tensor leaves.
* ``evoxc.Tensor`` has no ``__array__``; unwrap with ``support.runner.to_numpy``.
* pytest hangs when an exception escapes a trace under ``--tb=auto``; use
  ``--tb=short`` while debugging.

Backend is ``"numpy"`` throughout: no optional dependency, no ``jax``/``torch``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import evoxc

from support import payloads as payloads_module
from support.runner import run_defn, to_numpy

import tensorneat.algorithm.neat._distance as distance_module
import tensorneat.algorithm.neat._kernels as kernels_module
import tensorneat.algorithm.neat.species as species_module
import tensorneat.common as common_module
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome.utils import GenomeTensors

#: Population size of the whole-``tell`` comparison (the suite's small default).
POP_SIZE = 8
#: Species-slot capacity of the whole-``tell`` comparison.
SPECIES_SIZE = 4
#: Gene widths the tagged hand-built payloads use (the ``DefaultGenome`` widths).
NODE_ROWS = payloads_module.DEFAULT_MAX_NODES
CONN_ROWS = payloads_module.DEFAULT_MAX_CONNS
NODE_ATTR_WIDTH = payloads_module.DEFAULT_NODE_ATTR_WIDTH
CONN_KEY_WIDTH = payloads_module.DEFAULT_CONN_KEY_WIDTH
CONN_ATTR_WIDTH = payloads_module.DEFAULT_CONN_ATTR_WIDTH

#: The four split payload fields of ``genome.utils.GenomeTensors``, in order.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")
#: ``State`` entries that are runtime tensors; everything else is static config.
STATE_TENSOR_KEYS = ("pop", "generation", "species", "randkey")
#: ``compatibility_threshold`` of every hand-computed speciation table.
COMPATIBILITY_THRESHOLD = 2.0
#: ``generation`` handed to the hand-computed speciation tables (created slots
#: record it in ``last_improved``, so a distinctive value is easy to read).
GENERATION = 7.0


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #


def _same(left: Any, right: Any) -> bool:
    """True when two arrays agree bit for bit, with ``NaN == NaN``.

    ``np.array_equal(..., equal_nan=True)`` rejects integer dtypes (``np.isnan``
    has no integer loop), so the NaN-awareness is applied only to floats; bytes
    would also work, but the shape/dtype check has to happen first either way.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def _first_difference(left: Any, right: Any) -> str:
    """A short, informative description of where two arrays disagree."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return f"shape {left.shape} != {right.shape}"
    if left.dtype != right.dtype:
        return f"dtype {left.dtype} != {right.dtype}"
    if _same(left, right):
        return "identical"
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    for index, (a, b) in enumerate(zip(flat_left, flat_right)):
        if not _same(np.asarray([a]), np.asarray([b])):
            return f"first difference at flat index {index}: {a!r} != {b!r}"
    return "differ (NaN pattern)"


def _assert_identical(label: str, expected: Any, actual: Any) -> None:
    """Fail naming the field, both shapes/dtypes and the first differing cell."""
    assert _same(expected, actual), (
        f"{label} mismatch: {_first_difference(expected, actual)}"
    )


def _members_and_ranks(idx2species: Any) -> dict:
    """Per-species ``{member_row: rank}`` derived from ``idx2species`` alone.

    ``idx2species`` is the only species membership the generation carries, so the
    oracle for "same member sets and ranks" is this grouping of the array itself
    (never a second kernel): members keep population order and rank ``0`` is the
    first member, which is what parent pairing pins elites to.
    """
    values = np.asarray(idx2species)
    assert not np.isnan(values).any(), (
        f"idx2species still has unassigned rows after the post-pass: {values}"
    )
    table = {}
    for key in np.unique(values):
        members = [int(row) for row in np.flatnonzero(values == key)]
        table[float(key)] = {member: rank for rank, member in enumerate(members)}
    return table


# --------------------------------------------------------------------------- #
# whole-``NEAT.tell`` comparison (the seam in its production wiring)
# --------------------------------------------------------------------------- #


def _state_leaves(state: Any, field_names: tuple) -> list:
    """The ordered tensor leaves of ``state``: payload, generation, species, key."""
    return [
        *(getattr(state.pop, name) for name in POP_FIELDS),
        state.generation,
        *(getattr(state.species, name) for name in field_names),
        state.randkey,
    ]


def _fitness(pop_size: int) -> np.ndarray:
    """A distinct fitness per row, so selection and speciation see no ties."""
    return np.arange(pop_size, dtype=np.float32) / np.float32(pop_size)


def run_tell(algorithm: Any, state: Any, fitness: Any) -> dict:
    """Run one ``NEAT.tell`` in a fresh graph and return its host result.

    Args:
        algorithm: The ``NEAT`` instance under test.
        state: Starting ``State``; it is never mutated (``State`` is immutable).
        fitness: ``(pop_size,)`` raw fitness, passed as a graph input.

    Returns:
        ``{"pop": {field: array}, "generation": array, "species": {field:
        array}, "randkey": array}`` -- the four payload fields, the advanced
        counter, all ten ``SpeciesState`` fields and the next random key.

    Notes:
        A fresh ``evoxc.build`` runs per call, so a trace-time branch on
        ``GATED_ROWS`` is re-evaluated; that is why the comparison builds the
        graph twice instead of replaying one executable.
    """
    field_names = tuple(
        field.name for field in dataclasses.fields(type(state.species))
    )
    static = {
        name: getattr(state, name)
        for name in state.registered_keys()
        if name not in STATE_TENSOR_KEYS
    }
    state_type = type(state)
    payload_type = type(state.pop)
    species_type = type(state.species)

    def kernel(*values: Any) -> tuple:
        payload = payload_type(*values[: len(POP_FIELDS)])
        cursor = len(POP_FIELDS)
        generation = values[cursor]
        cursor += 1
        species_state = species_type(*values[cursor : cursor + len(field_names)])
        cursor += len(field_names)
        randkey = values[cursor]
        cursor += 1
        current = state_type(**static).register(
            pop=payload,
            generation=generation,
            species=species_state,
            randkey=randkey,
        )
        next_state = algorithm.tell(current, values[cursor])
        return (
            tuple(getattr(next_state.pop, name) for name in POP_FIELDS),
            next_state.generation,
            tuple(getattr(next_state.species, name) for name in field_names),
            next_state.randkey,
        )

    pop, generation, species_values, randkey = to_numpy(
        run_defn(
            kernel,
            *_state_leaves(state, field_names),
            np.asarray(fitness, np.float32),
        )
    )
    return {
        "pop": dict(zip(POP_FIELDS, pop)),
        "generation": generation,
        "species": dict(zip(field_names, species_values)),
        "randkey": randkey,
    }


@pytest.fixture
def algorithm(neat) -> NEAT:
    """The ``NEAT`` under test: ``P = 8``, ``S = 4``, no batch chunking."""
    return neat(pop_size=POP_SIZE, species_size=SPECIES_SIZE)


@pytest.fixture
def start_state(algorithm, common, evoxc_key):
    """Generation-0 state built from the test-id-seeded RNG key."""
    return algorithm.setup(common.State(randkey=evoxc_key))


@pytest.fixture
def forced_gate(monkeypatch):
    """Set ``_kernels.GATED_ROWS`` so ``gated(S, P)`` answers exactly ``value``.

    ``0`` forces the dense path (``S * P > 0``) and ``10**9`` the lazy path
    (``S * P <= 10**9``) for every realistic table; the factory asserts the
    predicate really answered ``False``/``True`` so a future change to ``gated``
    cannot silently make a test exercise the wrong branch.  ``monkeypatch``
    restores the module attribute after the test, and each call inside a test
    simply overrides the previous one.
    """

    def force(value: int) -> None:
        monkeypatch.setattr(kernels_module, "GATED_ROWS", value)
        expected = 0 < SPECIES_SIZE * POP_SIZE <= value
        assert kernels_module.gated(SPECIES_SIZE, POP_SIZE) is expected, (
            f"GATED_ROWS={value} did not make gated({SPECIES_SIZE}, "
            f"{POP_SIZE}) answer {expected}"
        )

    return force


def test_dense_and_gated_tell_agree_on_the_same_state(
    algorithm, start_state, forced_gate
):
    """Both distance paths yield the same next generation from one state.

    The oracle is the *other branch*: the same population, the same ``randkey``,
    the same ``generation`` and the same fitness are handed to two fresh graphs,
    one with ``GATED_ROWS = 0`` (dense) and one with ``GATED_ROWS = 10**9``
    (gated).  Every value ``tell`` returns must match bit for bit -- species
    assignment, centre gene rows, bookkeeping, the next population, the counter
    and the advanced key -- which is the documented "bit-identical across the
    two paths" contract.
    """
    fitness = _fitness(POP_SIZE)

    forced_gate(0)
    dense = run_tell(algorithm, start_state, fitness)

    forced_gate(10**9)
    gated = run_tell(algorithm, start_state, fitness)

    # Species assignment and the four re-elected centre tables.
    for name in (
        "species_keys",
        "best_fitness",
        "last_improved",
        "member_count",
        "idx2species",
        "center_node_keys",
        "center_node_attrs",
        "center_conn_keys",
        "center_conn_attrs",
        "next_species_key",
    ):
        _assert_identical(
            f"species.{name} (dense vs gated)",
            dense["species"][name],
            gated["species"][name],
        )

    # The next population and the run bookkeeping.
    for name in POP_FIELDS:
        _assert_identical(
            f"pop.{name} (dense vs gated)", dense["pop"][name], gated["pop"][name]
        )
    _assert_identical(
        "generation (dense vs gated)", dense["generation"], gated["generation"]
    )
    _assert_identical("randkey (dense vs gated)", dense["randkey"], gated["randkey"])

    # Member sets and ranks, derived from ``idx2species`` alone (never a kernel).
    dense_members = _members_and_ranks(dense["species"]["idx2species"])
    gated_members = _members_and_ranks(gated["species"]["idx2species"])
    assert dense_members == gated_members, (
        "per-species member sets/ranks differ: "
        f"dense={dense_members}, gated={gated_members}"
    )
    rows = sorted(row for ranks in dense_members.values() for row in ranks)
    assert rows == list(range(POP_SIZE)), (
        f"the species assignment is not a partition of the population: {rows}"
    )


def test_dense_branch_calls_matrix_once_and_the_gated_branch_never(
    algorithm, start_state, forced_gate, monkeypatch
):
    """The forced branch is the branch that ran, proved by a call-count spy.

    ``DistanceRows.matrix`` exists for exactly one caller (``_kernels.speciate``'s
    dense scan 1), so a counting trampoline around it separates the two paths:
    the dense run must call it **once** (one ``(S, P)`` table) and the gated run
    **zero** times.  A path that silently fell back to the other spelling would
    leave the count at the wrong value even though the numbers agree.
    """
    fitness = _fitness(POP_SIZE)
    original = distance_module.DistanceRows.matrix
    measured = []

    def spy(self, state, centres):
        measured.append(int(centres[0].shape[0]))
        return original(self, state, centres)

    monkeypatch.setattr(distance_module.DistanceRows, "matrix", spy)

    forced_gate(0)
    run_tell(algorithm, start_state, fitness)
    assert measured == [SPECIES_SIZE], (
        "the dense path must measure the whole centre table once, with "
        f"species_size={SPECIES_SIZE} centres; DistanceRows.matrix saw {measured}"
    )

    measured.clear()
    forced_gate(10**9)
    run_tell(algorithm, start_state, fitness)
    assert measured == [], (
        "the gated path must not build a dense table; DistanceRows.matrix saw "
        f"{measured} call(s)"
    )


# --------------------------------------------------------------------------- #
# direct dense-branch test on a hand-written distance table
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class SpeciationCase:
    """One hand-written ``(P + S, P)`` distance table plus its expected output.

    Row ``r`` of ``table`` is the distance row of the genome tagged ``r``: rows
    ``0 .. P-1`` are population members, rows ``P .. P+S-1`` the initial species
    centre rows.  ``_kernels.speciate`` receives the table through a fake
    ``distance_fn`` whose ``matrix`` returns rows ``P .. P+S-1`` (scan 1 measures
    the pre-scan centre table) and whose ``__call__`` gathers the row of whatever
    centre it is handed (scan 2 measures the selected, possibly created centre).
    """

    name: str
    pop_size: int
    species_size: int
    table: np.ndarray
    species_keys: tuple
    best_fitness: tuple
    last_improved: tuple
    member_count: tuple
    next_species_key: float
    expected_idx2species: tuple
    expected_keys: tuple
    expected_member_count: tuple
    expected_centre_tags: tuple
    expected_created_slots: tuple
    expected_next_species_key: float


def _steal_and_create_case() -> SpeciationCase:
    """Two live slots, a created species, and a declined steal in scan 2.

    ``P = 6``, ``S = 4``, threshold ``2.0``.  Scan 1 claims population rows
    ``0`` and ``1`` for the two live slots (keys ``0.0`` and ``1.0``).  Scan 2
    then:

    * slot ``0`` (centre ``pop[0]``) takes row ``2`` only (rows ``3..5`` are out
      of threshold);
    * slot ``1`` (centre ``pop[1]``) takes row ``3``.  Row ``2`` is *within* its
      threshold (``0.5 < 2.0``) but only a strictly closer row may steal it, and
      its distance equals the ``o2c`` scan 2 stored for row ``2``, so
      ``d < o2c`` is false and the row stays with slot ``0`` -- the regression
      that pins ``<`` rather than ``<=``;
    * dead slot ``2`` has two unassigned rows left, creates species key ``2.0``
      seeded by row ``4`` (the first unassigned genome) and takes rows ``4`` and
      ``5``;
    * dead slot ``3`` sees nothing unassigned and does not create.
    """
    pop_size = 6
    species_size = 4
    table = np.full((pop_size + species_size, pop_size), 5.0, np.float32)
    # Initial centre rows (tags P, P+1, P+2, P+3 = 6, 7, 8, 9).
    table[6] = [0.1, 5.0, 5.0, 5.0, 5.0, 5.0]
    table[7] = [5.0, 0.1, 5.0, 5.0, 5.0, 5.0]
    # Population-tagged rows used by scan 2.
    table[0] = [0.0, 0.5, 0.5, 2.5, 2.5, 2.5]
    table[1] = [0.5, 0.0, 0.5, 0.5, 2.5, 2.5]
    table[4] = [5.0, 5.0, 5.0, 5.0, 0.0, 0.4]
    return SpeciationCase(
        name="steal-and-create",
        pop_size=pop_size,
        species_size=species_size,
        table=table,
        species_keys=(0.0, 1.0, np.nan, np.nan),
        best_fitness=(0.5, 0.2, np.nan, np.nan),
        last_improved=(0.0, 3.0, np.nan, np.nan),
        member_count=(1.0, 1.0, np.nan, np.nan),
        next_species_key=2.0,
        expected_idx2species=(0.0, 1.0, 0.0, 1.0, 2.0, 2.0),
        expected_keys=(0.0, 1.0, 2.0, np.nan),
        expected_member_count=(2.0, 2.0, 2.0, np.nan),
        # Centre rows after both scans: pop[0], pop[1], created pop[4], and the
        # untouched initial row of dead slot 3 (tag P + 3 = 9).
        expected_centre_tags=(0, 1, 4, 9),
        expected_created_slots=(2,),
        expected_next_species_key=3.0,
    )


def _leftover_case() -> SpeciationCase:
    """Every slot is live, so the post-pass adopts the leftovers.

    ``P = 4``, ``S = 2``, threshold ``2.0``.  Scan 1 claims rows ``0`` and ``1``
    for the two live slots; scan 2 finds no genome inside the threshold of either
    centre, so rows ``2`` and ``3`` stay unassigned and the post-pass writes the
    last live species key (slot ``1``) into them.  No species is created, so the
    key counter does not move.
    """
    pop_size = 4
    species_size = 2
    table = np.full((pop_size + species_size, pop_size), 9.0, np.float32)
    # Initial centre rows (tags P, P+1 = 4, 5).
    table[4] = [0.0, 1.0, 9.0, 9.0]
    table[5] = [9.0, 0.0, 9.0, 9.0]
    # Population-tagged rows used by scan 2.
    table[0] = [0.0, 5.0, 5.0, 5.0]
    table[1] = [5.0, 0.0, 5.0, 5.0]
    return SpeciationCase(
        name="leftovers",
        pop_size=pop_size,
        species_size=species_size,
        table=table,
        species_keys=(0.0, 1.0),
        best_fitness=(0.5, 0.2),
        last_improved=(0.0, 3.0),
        member_count=(1.0, 1.0),
        next_species_key=2.0,
        expected_idx2species=(0.0, 1.0, 1.0, 1.0),
        expected_keys=(0.0, 1.0),
        expected_member_count=(1.0, 3.0),
        expected_centre_tags=(0, 1),
        expected_created_slots=(),
        expected_next_species_key=2.0,
    )


class _TaggedDistanceTable:
    """A ``distance_fn`` whose table is indexed by the centre's gene tag.

    The tagged payloads below write ``tag`` into every element of a genome's
    leading row, so a centre payload carries its own row index.  Both forms the
    speciation scans use are implemented on that identity:

    * :meth:`matrix` returns the rows of the *initial* centre table (tags
      ``P .. P+S-1``), which is what the dense scan 1 measures before the scan;
    * :meth:`__call__` gathers the row of the centre it is handed -- a
      population row for a re-elected or created centre, an initial centre row
      for a live slot whose scan-1 claim was masked off.

    Counters record which spelling was reached, so a test can assert the forced
    branch is the branch that ran.
    """

    def __init__(self, case: SpeciationCase) -> None:
        self.table = np.asarray(case.table, np.float32)
        self.pop_size = case.pop_size
        self.species_size = case.species_size
        self.matrix_calls = 0
        self.call_count = 0

    def _constant(self) -> Any:
        return evoxc.constant(evoxc.from_numpy(self.table))

    def matrix(self, state: Any, centres: Any) -> Any:
        """``(S, P)`` rows of the initial centre table (scan 1, dense path)."""
        self.matrix_calls += 1
        count = int(centres[0].shape[0])
        assert count == self.species_size, (
            f"dense scan 1 measured {count} centre rows, expected {self.species_size}"
        )
        rows = self.table[self.pop_size : self.pop_size + count]
        return evoxc.constant(evoxc.from_numpy(rows))

    def __call__(self, state: Any, centre: Any, pop: Any = None) -> Any:
        """``(P,)`` row of the centre's tag (scan 2, and scan 1 when gated)."""
        self.call_count += 1
        tag = evoxc.reshape(evoxc.slice(centre[0], (0, 0), (1, 1)), (1,))
        row = evoxc.gather(self._constant(), tag, axis=0)
        return evoxc.reshape(row, (self.pop_size,))


def _tagged_arrays(pop_size: int, species_size: int) -> tuple:
    """Tagged population and initial-centre payloads for the fake distance table.

    Genome ``p`` is tagged ``p`` and initial centre ``s`` is tagged ``P + s``;
    the tag is repeated across every element of the leading gene row, so the
    fake ``distance_fn`` can read a centre's identity with one gather.  The
    remaining attribute columns stay neutral, which keeps every expected centre
    row a verbatim copy of the genome it came from.
    """
    pop_tags = np.arange(pop_size, dtype=np.int32)[:, None]
    centre_tags = np.arange(pop_size, pop_size + species_size, dtype=np.int32)[:, None]

    def build(tags: np.ndarray) -> tuple:
        count = tags.shape[0]
        node_keys = np.repeat(tags, NODE_ROWS, axis=1).astype(np.int32)
        node_attrs = np.zeros((count, NODE_ROWS, NODE_ATTR_WIDTH), np.float32)
        node_attrs[:, :, 0] = tags + 0.5
        conn_keys = np.zeros((count, CONN_ROWS, CONN_KEY_WIDTH), np.int32)
        conn_keys[:, 0, 0] = 100 + tags[:, 0]
        conn_attrs = np.zeros((count, CONN_ROWS, CONN_ATTR_WIDTH), np.float32)
        conn_attrs[:, 0, 0] = tags[:, 0] + 0.25
        return node_keys, node_attrs, conn_keys, conn_attrs

    population = build(pop_tags)
    centres = build(centre_tags)
    return population, centres


def _species_state(case: SpeciationCase, centres: tuple):
    """Build the starting ``SpeciesState`` of a case (idx2species all NaN)."""
    return species_module.SpeciesState(
        species_keys=np.asarray(case.species_keys, np.float32),
        best_fitness=np.asarray(case.best_fitness, np.float32),
        last_improved=np.asarray(case.last_improved, np.float32),
        member_count=np.asarray(case.member_count, np.float32),
        idx2species=np.full((case.pop_size,), np.nan, np.float32),
        center_node_keys=centres[0],
        center_node_attrs=centres[1],
        center_conn_keys=centres[2],
        center_conn_attrs=centres[3],
        next_species_key=np.float32(case.next_species_key),
    )


def _run_speciation(case: SpeciationCase, gate: bool) -> tuple:
    """Run ``_kernels.speciate`` once with the gate passed explicitly.

    ``gate`` travels as a static Python ``bool`` (as ``SpeciesController.speciate``
    passes it), never read from ``_kernels.GATED_ROWS``, so the test survives a
    change to the module attribute or to ``gated`` itself.

    Returns:
        ``(species_fields, distance_fn)``: the ten output arrays keyed by
        ``SpeciesState`` field name, and the fake table (with its call counters).
    """
    population, centres = _tagged_arrays(case.pop_size, case.species_size)
    starting = _species_state(case, centres)
    field_names = tuple(
        field.name for field in dataclasses.fields(species_module.SpeciesState)
    )
    table = _TaggedDistanceTable(case)
    generation = np.float32(GENERATION)

    def kernel(*values: Any) -> tuple:
        state = common_module.State(generation=values[0])
        payload = GenomeTensors(*values[1:5])
        species_state = species_module.SpeciesState(
            *values[5 : 5 + len(field_names)]
        )
        return tuple(
            kernels_module.speciate(
                state,
                species_state,
                table,
                payload,
                case.pop_size,
                case.species_size,
                COMPATIBILITY_THRESHOLD,
                gate,
            )
        )

    result = to_numpy(
        run_defn(
            kernel,
            generation,
            *population,
            *(np.asarray(getattr(starting, name)) for name in field_names),
        )
    )
    return dict(zip(field_names, result)), table


@pytest.mark.parametrize("gate", [False, True], ids=["dense", "gated"])
@pytest.mark.parametrize(
    "case_factory",
    [_steal_and_create_case, _leftover_case],
    ids=lambda factory: factory().name,
)
def test_speciate_assigns_members_as_documented(case_factory, gate):
    """The whole speciation output matches the rules written out by hand.

    The oracle is the documented scan algebra applied to a hand-written
    ``(P + S, P)`` distance table -- not a second kernel:

    1. scan 1, per live slot: claim the nearest still-unassigned genome
       (``argmin`` with dead/unassigned rows masked to ``+inf``) and re-elect it
       as the slot's centre;
    2. scan 2, per live slot: assign every genome with ``d < threshold`` that is
       unassigned or closer than its current centre (``d < o2c``);
    3. scan 2, per dead slot with anything still unassigned: materialise the next
       species key, seed it with the first unassigned genome, run the threshold
       assignment against that new centre;
    4. post-pass: leftovers adopt the last live species key, ``member_count`` is
       the masked compare-sum, created slots record ``-inf`` /
       ``generation``.

    Both ``gate`` values are exercised on the same table, so the assertion also
    states that the two spellings agree.
    """
    case = case_factory()
    result, table = _run_speciation(case, gate)

    assert table.matrix_calls == (0 if gate else 1), (
        f"gate={gate} reached DistanceRows-equivalent matrix "
        f"{table.matrix_calls} time(s)"
    )

    _assert_identical(
        "idx2species",
        np.asarray(case.expected_idx2species, np.float32),
        result["idx2species"],
    )
    _assert_identical(
        "species_keys",
        np.asarray(case.expected_keys, np.float32),
        result["species_keys"],
    )
    _assert_identical(
        "member_count",
        np.asarray(case.expected_member_count, np.float32),
        result["member_count"],
    )
    _assert_identical(
        "next_species_key",
        np.float32(case.expected_next_species_key),
        result["next_species_key"],
    )

    # The centre rows are verbatim copies of the genome that owns them: a
    # population row when scan 1 (or a creation) re-elected it, the untouched
    # initial centre row otherwise.  All four gene arrays, every element.
    population, centres = _tagged_arrays(case.pop_size, case.species_size)
    for name, pop_field, centre_field in zip(POP_FIELDS, population, centres):
        field_name = f"center_{name}"
        actual = np.asarray(result[field_name])
        assert actual.shape == centre_field.shape, (
            f"{field_name} has shape {actual.shape}, expected {centre_field.shape}"
        )
        for slot, tag in enumerate(case.expected_centre_tags):
            expected = (
                pop_field[tag]
                if tag < case.pop_size
                else centre_field[tag - case.pop_size]
            )
            _assert_identical(
                f"{field_name}[{slot}] (tag {tag})", expected, actual[slot]
            )

    # Created slots get fresh stagnation bookkeeping; untouched slots keep theirs.
    best_fitness = np.asarray(result["best_fitness"])
    last_improved = np.asarray(result["last_improved"])
    for slot in range(case.species_size):
        if slot in case.expected_created_slots:
            assert np.isneginf(best_fitness[slot]), (
                f"created slot {slot} must report -inf best_fitness, got "
                f"{best_fitness[slot]}"
            )
            assert last_improved[slot] == np.float32(GENERATION), (
                f"created slot {slot} must record generation {GENERATION}, got "
                f"{last_improved[slot]}"
            )
        else:
            assert best_fitness[slot] == np.float32(case.best_fitness[slot]) or (
                np.isnan(best_fitness[slot]) and np.isnan(case.best_fitness[slot])
            ), f"slot {slot} best_fitness changed"
            assert last_improved[slot] == np.float32(case.last_improved[slot]) or (
                np.isnan(last_improved[slot]) and np.isnan(case.last_improved[slot])
            ), f"slot {slot} last_improved changed"


# --------------------------------------------------------------------------- #
# the real DistanceRows table against its own per-centre spelling
# --------------------------------------------------------------------------- #


def test_distance_rows_matrix_measures_one_centre_per_row(payloads):
    """``DistanceRows.matrix`` is the per-centre row, stacked (bit for bit).

    The safety net for a seam change: even if ``GATED_ROWS`` disappears, the
    whole-table spelling must keep agreeing with the single-centre call that
    both scans use.  The population is four *different* small genomes (dense
    chain, skip, isolated node, recurrent ring) at the ``DefaultGenome`` widths;
    three of them are measured as centres, once through :meth:`matrix` and once
    through :meth:`__call__` per centre.  The compatibility kernel is
    row-independent, so every row must be byte-identical.
    """
    genome = payloads.default_genome()
    parts = [
        payloads.chain_payload(pop_size=1),
        payloads.skip_payload(pop_size=1),
        payloads.isolated_payload(pop_size=1),
        payloads.ring_payload(pop_size=1),
    ]
    population = tuple(
        np.concatenate([getattr(part, name) for part in parts], axis=0)
        for name in POP_FIELDS
    )
    pop_size = int(population[0].shape[0])
    assert pop_size == 4
    centre_rows = (0, 1, 3)
    centres = tuple(field[list(centre_rows)] for field in population)
    centre_count = len(centre_rows)
    state = common_module.State()

    def kernel(*values: Any) -> tuple:
        payload = GenomeTensors(*values[:4])
        centre = GenomeTensors(*values[4:8])
        digest = genome.prepare_distance(state, payload)
        rows = distance_module.DistanceRows(
            genome, digest, payload, pop_size=pop_size
        )
        table = rows.matrix(state, centre)
        singles = tuple(
            rows(
                state,
                GenomeTensors(*(field[i : i + 1] for field in centre)),
                payload,
            )
            for i in range(centre_count)
        )
        return table, singles

    table, singles = to_numpy(
        run_defn(kernel, *population, *centres)
    )

    assert table.shape == (centre_count, pop_size), (
        f"matrix returned {table.shape}, expected {(centre_count, pop_size)}"
    )
    # A measurement actually happened: four different topologies cannot all be
    # at distance zero, and the diagonal is a genome against itself.
    assert np.any(table > 0), f"matrix is all zeros, nothing was measured: {table}"
    for index, single in enumerate(singles):
        _assert_identical(f"matrix row {index} vs __call__", table[index], single)
