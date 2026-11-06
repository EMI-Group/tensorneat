"""Unit tests for the device-gated distance formulations (numpy reference leg).

What this module proves
-----------------------
Speciation's compatibility distance is computed under three *orthogonal* trace-time
statics, each a plain host string on the configuration ``State``:

* ``kernel_mode`` in ``("host", "accelerator")`` (``common/schedule_accelerator.py``)
  -- the only static that makes the **grouped** distance route selectable:
  ``grouped.group_size`` answers ``1`` for a host state and ``ACCELERATOR_GROUP`` (2)
  for an accelerator one, and ``_kernels.gated`` never gates an accelerator table
  however small it is.
* ``distance_centre`` in ``("tiled", "single")`` (``algorithm/neat/_validation.py``)
  -- the width of every centre payload handed to the genome.
* ``distance_formulation`` in ``("merge", "index_free")``
  (``genome/operations/distance/index_free.py``) -- one host branch inside the
  ``index_free.alignment`` seam.

Every combination is documented to compute the same record from the same kernel and
payload, i.e. to be **bit-identical** to the shipped ``host``/``tiled``/``merge``
route.  ``./test_distance_dense.py`` pins the fourth, orthogonal seam
(``_kernels.GATED_ROWS``); this module pins the device-gated formulations on the CPU
numpy leg that CI runs:

* :func:`test_distance_table_is_identical_across_the_eight_formulation_statics`
  measures one real speciation centre table (``P = 100``, ``S = 8``) with
  ``DistanceRows.matrix`` once per point of the ``2 x 2 x 2`` grid.
* :func:`test_tell_is_identical_across_the_three_distance_routes` drives a whole
  ``NEAT.tell`` from one immutable starting state under ``host``/``tiled``/``merge``,
  ``accelerator``/``tiled`` (the grouped leg) and ``accelerator``/``single``.
* :func:`test_the_grouped_leg_is_the_route_the_accelerator_static_selects` proves the
  route was *selected*, by trace-time spies rather than by equal numbers.
* :func:`test_the_index_free_formulation_seam_is_the_branch_the_static_selects` proves
  ``distance_formulation`` reaches its arm of the alignment seam under both legs.
* :func:`test_the_reference_route_is_the_default_spelling` pins that the reference
  point *is* the absent-entry default the three readers resolve.

Every table-level comparison is byte-identical in shape, dtype, values and ``NaN``s,
and the ``tell`` comparison covers the next population's four arrays, ``generation``,
``randkey`` and all ten ``SpeciesState`` fields.

Oracles used here (never a second copy of the kernel)
-----------------------------------------------------
The oracle for every formulation is the *other* route: the same population, centre
rows and digest go into one fresh graph per route, and the shipped
``host``/``tiled``/``merge`` result is what the other seven must reproduce.  Route
*selection* is not inferred from equal numbers but from the spies (the ``for_state``
answer, the merged batch widths, the two alignment-arm counts) plus the documented
static predicates (``grouped.group_size``, ``_kernels.gated``), called directly.

Cost
----
A whole ``tell`` trace is the expensive part (about 0.25 s per graph at this geometry),
so exactly **three** tell graphs are built -- one per compared route -- and every other
combination stays at the ``(S, P)`` matrix level (about 0.07 s per graph).  The whole
module runs in about 2.5 s.

Gotchas
-------
* Every spy counts **trace-time** calls (``run_defn`` builds a fresh graph per call and
  ``evoxc.scan`` peels step 0 and traces the body again), so assertions read the
  *shape* of the record -- which routes recorded anything, every merge carrying ``k``
  payloads -- never a literal dispatch count.
* ``_distance.py`` does ``from ...grouped import centre_batch, for_state``, so patching
  ``grouped.for_state`` would not be seen: patch the ``_distance`` module attribute.
* ``index_free.alignment`` is the seam, not a branch -- it runs on both formulations --
  so its arms (``index_free.resolve`` / ``digest.measure_alignment``) discriminate.
* The host leg at ``S * P = 800 <= GATED_ROWS`` is the lazy gated path and never asks
  for a table: its ``matrix`` record is *empty*, not "one ungrouped call".
* ``State.register`` refuses an existing key, so :meth:`Route.apply` uses ``update`` for
  the entries a state already carries.  ``evoxc.Tensor`` has no ``__array__``; unwrap
  with ``support.runner.to_numpy``.

Backend is ``"numpy"`` throughout: no GPU, no ``iree``, no optional extra, no
``requires_iree`` marker.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

# The helpers and the whole-``tell`` runner are shared with the gated/dense seam
# suite rather than copied: ``run_tell`` rebuilds the ``State`` *inside* the trace
# from positional leaves and forwards every non-tensor registered entry (which is
# how the route statics reach the graph), and ``_assert_identical`` is the one
# spelling of "bit for bit".
from .test_distance_dense import (
    POP_FIELDS,
    _assert_identical,
    _fitness,
    _members_and_ranks,
    run_tell,
)

from support.runner import run_defn, to_numpy

import tensorneat.algorithm.neat._distance as distance_module
import tensorneat.algorithm.neat._kernels as kernels_module
import tensorneat.algorithm.neat._validation as validation_module
import tensorneat.common as common_module
import tensorneat.common.schedule_accelerator as accelerator_module
import tensorneat.genome.operations.distance.digest as digest_module
import tensorneat.genome.operations.distance.grouped as grouped_module
import tensorneat.genome.operations.distance.index_free as index_free_module
from tensorneat.genome.utils import GenomeTensors

#: Population batch of every graph here: the geometry the module is about, and the
#: smallest one where the grouped leg and a realistic speciation table meet.
POP_SIZE = 100
#: Species-slot capacity of the speciation centre table (``S * P = 800``, still
#: inside ``_kernels.GATED_ROWS`` -- that is the point of the gated interaction).
SPECIES_SIZE = 8
#: Individuals each of the four canonical topologies contributes (``4 * 25 = 100``).
TOPOLOGY_ROWS = POP_SIZE // 4
#: The ten ``SpeciesState`` fields ``run_tell`` returns, in ``fields`` order.
SPECIES_FIELDS = (
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
)


# --- the three-static grid ---


@dataclasses.dataclass(frozen=True)
class Route:
    """One point of the ``kernel_mode x distance_centre x formulation`` grid.

    The three entries are the plain host strings read at trace time, so they travel
    into a graph as statics and change only which route is built -- never what a route
    computes: ``kernel_mode`` is ``"host"`` (shipped scatter forms, ungrouped) or
    ``"accelerator"`` (scatter-free forms plus the grouped table), ``distance_centre``
    is ``"tiled"`` (population-shaped centre payload) or ``"single"`` (one-individual
    payload), and ``distance_formulation`` is ``"merge"`` (sort-based successor test)
    or ``"index_free"`` (equality table).
    """

    kernel_mode: str = "host"
    distance_centre: str = "tiled"
    distance_formulation: str = "merge"

    @property
    def label(self) -> str:
        """``kernel_mode/distance_centre/distance_formulation``, for messages."""
        return f"{self.kernel_mode}/{self.distance_centre}/{self.distance_formulation}"

    def statics(self) -> dict:
        """The three entries as a ``State`` keyword mapping."""
        return {
            "kernel_mode": self.kernel_mode,
            "distance_centre": self.distance_centre,
            "distance_formulation": self.distance_formulation,
        }

    def apply(self, state: Any) -> Any:
        """Return ``state`` carrying this route's three statics.

        ``State`` is immutable and ``register`` refuses an existing key, so entries the
        state already holds are replaced with ``update`` and the missing ones created
        with ``register`` -- honest whether the caller starts from a bare ``State`` or
        from ``NEAT.setup``'s output (a future ``setup`` that registers a device-derived
        default must not make these tests error out).
        """
        entries = self.statics()
        present = {
            name: value
            for name, value in entries.items()
            if name in state.registered_keys()
        }
        missing = {
            name: value for name, value in entries.items() if name not in present
        }
        if present:
            state = state.update(**present)
        if missing:
            state = state.register(**missing)
        return state


#: The shipped route every other point is compared against; it is also the spelling a
#: bare ``State`` resolves to (pinned by the last test below).
REFERENCE = Route()
#: The eight points of the grid, the reference first.
ALL_ROUTES = tuple(
    Route(kernel_mode, distance_centre, distance_formulation)
    for kernel_mode in accelerator_module.KERNEL_MODES
    for distance_centre in validation_module.DISTANCE_CENTRE_MODES
    for distance_formulation in index_free_module.DISTANCE_FORMULATION_MODES
)


# --- payloads ---


def population(payloads: Any) -> tuple:
    """A ``P = 100`` payload over the four canonical topologies.

    Rows ``0 .. 24`` are the dense chain, ``25 .. 49`` the skipped-node genome,
    ``50 .. 74`` the isolated-node one and ``75 .. 99`` the recurrent ring, so the
    population holds four different key sets and in-degrees -- the merge has real work
    to do in both halves (keys *and* attributes).  Every row also gets its own
    connection-weight offset (``row * 0.01``); ``NaN + x`` keeps padding rows NaN.

    Args:
        payloads: The ``support.payloads`` module fixture.

    Returns:
        tuple: The four payload arrays in ``GenomeTensors`` order, ``(P, ...)``.
    """
    parts = [
        payloads.chain_payload(pop_size=TOPOLOGY_ROWS),
        payloads.skip_payload(pop_size=TOPOLOGY_ROWS),
        payloads.isolated_payload(pop_size=TOPOLOGY_ROWS),
        payloads.ring_payload(pop_size=TOPOLOGY_ROWS),
    ]
    arrays = {
        name: np.concatenate([getattr(part, name) for part in parts], axis=0)
        for name in POP_FIELDS
    }
    offset = (np.arange(POP_SIZE, dtype=np.float32) * np.float32(0.01))[:, None, None]
    arrays["conn_attrs"] = arrays["conn_attrs"] + offset
    return tuple(arrays[name] for name in POP_FIELDS)


def centre_rows(payload: tuple) -> tuple:
    """``S = 8`` centre rows spread over all four topologies.

    Slot ``p`` is row ``(p % 4) * 25 + p // 4`` (``0, 25, 50, 75, 1, 26, 51, 76``), so
    the centre table is not one contiguous block and a centre whose key set matches
    only part of the population makes the disjoint/excess halves non-zero.

    Args:
        payload: The arrays :func:`population` returns.

    Returns:
        tuple: The four centre arrays in ``GenomeTensors`` order, ``(S, ...)``.
    """
    rows = [(slot % 4) * TOPOLOGY_ROWS + slot // 4 for slot in range(SPECIES_SIZE)]
    return tuple(field[rows] for field in payload)


def measure_table(
    population_payload: tuple,
    centres: tuple,
    route: Route,
    genome: Any,
) -> np.ndarray:
    """Build and run one ``DistanceRows.matrix`` graph under ``route``'s statics.

    ``evoxc.build`` turns only an ``ndarray`` argument into a graph input, so the host
    objects (``State``, both payloads, the digest, ``DistanceRows``) are rebuilt
    *inside* the traced closure from positional leaves, with the route's three strings
    closed over as statics.

    Args:
        population_payload: The ``(P, ...)`` population arrays.
        centres: The ``(S, ...)`` centre arrays.
        route (Route): The statics to build the graph with.
        genome: The genome under test; its ``prepare_distance``/``measure_distance``
            provide every number in the table.

    Returns:
        np.ndarray: The ``(S, P)`` float32 distance table.
    """
    pop_size = int(population_payload[0].shape[0])
    statics = route.statics()

    def kernel(*values: Any) -> Any:
        """One speciation centre table, measured through ``DistanceRows.matrix``."""
        state = common_module.State(**statics)
        pop = GenomeTensors(*values[: len(POP_FIELDS)])
        centre = GenomeTensors(*values[len(POP_FIELDS) : 2 * len(POP_FIELDS)])
        digest = genome.prepare_distance(state, pop)
        rows = distance_module.DistanceRows(genome, digest, pop, pop_size=pop_size)
        return rows.matrix(state, centre)

    return to_numpy(run_defn(kernel, *population_payload, *centres))


# --- distance-table level: the whole 2 x 2 x 2 grid ---


def test_distance_table_is_identical_across_the_eight_formulation_statics(payloads):
    """All eight static combinations produce the same ``(S, P)`` table, bit for bit.

    The reference is the shipped ``host``/``tiled``/``merge`` route; every other point
    of the grid is its own graph built from the same payload, centre rows and digest.
    """
    genome = payloads.default_genome()
    payload = population(payloads)
    centres = centre_rows(payload)
    assert (payload[0].shape[0], centres[0].shape[0]) == (POP_SIZE, SPECIES_SIZE)

    reference = measure_table(payload, centres, REFERENCE, genome)
    assert reference.shape == (SPECIES_SIZE, POP_SIZE), (
        f"matrix returned {reference.shape}, expected {(SPECIES_SIZE, POP_SIZE)}"
    )
    assert reference.dtype == np.float32, (
        f"matrix returned {reference.dtype}, expected float32"
    )
    # Non-vacuity: four topologies at 100 distinct weight offsets cannot all sit at
    # distance zero, so eight routes agreeing on zeros (or on one repeated column)
    # fails here instead of passing as "identical".
    assert np.any(reference > 0), f"every distance is zero: {reference}"
    assert len(np.unique(reference)) > SPECIES_SIZE, (
        "the table is degenerate (at most one value per centre row): "
        f"{np.unique(reference)}"
    )

    for route in ALL_ROUTES:
        table = measure_table(payload, centres, route, genome)
        _assert_identical(f"distance table ({route.label})", reference, table)


# --- whole-``NEAT.tell`` level: the three routes a generation can take ---


def _tell_routes() -> tuple:
    """The three routes a whole ``NEAT.tell`` is compared under.

    ``merge`` everywhere, because the formulation is pinned at the table level; the
    question for a whole generation is whether the leg (which decides grouping) and the
    centre width move any other field ``tell`` writes.
    """
    return (
        REFERENCE,
        Route(kernel_mode="accelerator", distance_centre="tiled"),
        Route(kernel_mode="accelerator", distance_centre="single"),
    )


def test_tell_is_identical_across_the_three_distance_routes(neat, common, evoxc_key):
    """One ``NEAT.tell``, three routes, identical next generations.

    The same immutable starting state (population, generation counter, species table,
    RNG key from ``NEAT.setup``) and the same fitness go into three fresh graphs that
    differ only in the route statics, so the oracle for the two accelerator routes is
    the shipped one.  Every field ``tell`` writes must agree, and the per-species member
    sets and ranks are re-derived from ``idx2species`` alone, so a reordered species
    table cannot pass on "same numbers, different rows".
    """
    algorithm = neat(pop_size=POP_SIZE, species_size=SPECIES_SIZE)
    start = algorithm.setup(common.State(randkey=evoxc_key))
    fitness = _fitness(POP_SIZE)

    results = {
        route.label: run_tell(algorithm, route.apply(start), fitness)
        for route in _tell_routes()
    }
    reference = results[REFERENCE.label]

    # ``tell`` consumed the generation: the counter advanced and a new RNG key was
    # split, so a "nothing happened" comparison cannot pass.  ``NEAT.setup`` hands back
    # ``evoxc.Tensor`` leaves, which have no ``__array__``.
    started_at = float(to_numpy(start.generation))
    assert float(reference["generation"]) == started_at + 1.0, (
        "tell must advance the generation counter by one, got "
        f"{float(reference['generation'])} from {started_at}"
    )
    assert not np.array_equal(
        np.asarray(reference["randkey"]), np.asarray(to_numpy(start.randkey))
    ), "tell must advance the RNG key"

    reference_members = _members_and_ranks(reference["species"]["idx2species"])
    rows = sorted(row for ranks in reference_members.values() for row in ranks)
    assert rows == list(range(POP_SIZE)), (
        f"the species assignment is not a partition of the population: {rows}"
    )

    for route in _tell_routes()[1:]:
        other = results[route.label]
        for group, names in (("species", SPECIES_FIELDS), ("pop", POP_FIELDS)):
            for name in names:
                _assert_identical(
                    f"{group}.{name} ({route.label} vs {REFERENCE.label})",
                    reference[group][name],
                    other[group][name],
                )
        for name in ("generation", "randkey"):
            _assert_identical(
                f"{name} ({route.label} vs {REFERENCE.label})",
                reference[name],
                other[name],
            )
        assert reference_members == _members_and_ranks(
            other["species"]["idx2species"]
        ), f"per-species member sets/ranks differ under {route.label}"


# --- route-selected proof: the spies ---


def test_the_grouped_leg_is_the_route_the_accelerator_static_selects(
    neat, common, evoxc_key, monkeypatch
):
    """The accelerator+tiled run is the grouped leg, proved by trace-time spies.

    ``_distance.for_state`` (read by ``DistanceRows.matrix``) records the slot count it
    was asked for and the ``k`` it answered, ``DistanceRows.matrix`` records how many
    centres one whole-table call carried, and ``_distance.centre_batch`` records the
    width of every merged centre batch.  The host reference at ``S * P = 800 <=
    GATED_ROWS`` gates its scans and never asks for a table, so its three records are
    empty; ``accelerator``/``single`` still *resolves* ``k = 2`` but the ``not tiled``
    guard keeps the per-row scan, so only its merge record is empty.
    ``grouped.group_size`` and ``_kernels.gated`` are then called directly, which pins
    the two documented static predicates without another graph.
    """
    algorithm = neat(pop_size=POP_SIZE, species_size=SPECIES_SIZE)
    start = algorithm.setup(common.State(randkey=evoxc_key))
    fitness = _fitness(POP_SIZE)

    groups: list = []
    tables: list = []
    merges: list = []
    original_for_state = distance_module.for_state
    original_matrix = distance_module.DistanceRows.matrix
    original_centre_batch = distance_module.centre_batch

    def for_state_spy(state, digest, **kwargs):
        """Record the slot count asked for and the group ``for_state`` answers."""
        answer = original_for_state(state, digest, **kwargs)
        groups.append((kwargs.get("species_size"), answer[1]))
        return answer

    def matrix_spy(self, state, centres):
        """Record the centre count of every whole-table call."""
        tables.append(int(centres[0].shape[0]))
        return original_matrix(self, state, centres)

    def centre_batch_spy(centres):
        """Record the width of every grouped centre merge."""
        merges.append(len(centres))
        return original_centre_batch(centres)

    monkeypatch.setattr(distance_module, "for_state", for_state_spy)
    monkeypatch.setattr(distance_module.DistanceRows, "matrix", matrix_spy)
    monkeypatch.setattr(distance_module, "centre_batch", centre_batch_spy)

    recorded = {}
    for route in _tell_routes():
        groups.clear()
        tables.clear()
        merges.clear()
        run_tell(algorithm, route.apply(start), fitness)
        recorded[route.label] = (list(groups), list(tables), list(merges))

    tiled = Route(kernel_mode="accelerator", distance_centre="tiled").label
    single = Route(kernel_mode="accelerator", distance_centre="single").label
    group = grouped_module.ACCELERATOR_GROUP

    # (i) the gated host path: no table, no group, no merge.
    assert recorded[REFERENCE.label] == ([], [], []), (
        "the gated host leg must not build a distance table, saw "
        f"{recorded[REFERENCE.label]}"
    )

    # (ii)+(iii) both accelerator runs measure the whole slot table in one call and
    # both resolve k = 2; only the "tiled" spelling ever merges a group.
    for label in (tiled, single):
        route_groups, route_tables, _merges = recorded[label]
        assert [slots for slots, _answer in route_groups] == [SPECIES_SIZE], (
            "the accelerator leg must ask for a group once, for the whole slot "
            f"table; for_state saw {route_groups} under {label}"
        )
        assert [answer for _slots, answer in route_groups] == [group], (
            "the accelerator leg must resolve k = ACCELERATOR_GROUP, saw "
            f"{route_groups} under {label}"
        )
        assert route_tables == [SPECIES_SIZE], (
            "the dense path measures the whole slot table in one matrix call; "
            f"matrix saw {route_tables} under {label}"
        )
    assert set(recorded[tiled][2]) == {group}, (
        "every grouped centre merge must carry exactly ACCELERATOR_GROUP payloads; "
        f"centre_batch saw {recorded[tiled][2]}"
    )
    assert recorded[single][2] == [], (
        "the single-centre spelling keeps the per-row scan and must not merge a "
        f"grouped centre batch; centre_batch saw {recorded[single][2]}"
    )

    # The two documented static predicates, called directly.
    assert (
        grouped_module.GROUP_ROW_BUDGET // POP_SIZE >= group
    ), "the row budget must not be the binding group cap at this population"
    assert grouped_module.group_size(common.State(), POP_SIZE, SPECIES_SIZE) == 1, (
        "the host leg groups at k = 1 (the plain route)"
    )
    assert (
        grouped_module.group_size(
            common.State(kernel_mode="accelerator"), POP_SIZE, SPECIES_SIZE
        )
        == group
    ), "the accelerator leg must group at ACCELERATOR_GROUP"
    assert kernels_module.gated(SPECIES_SIZE, POP_SIZE, "host") is True, (
        "the host leg's table is inside GATED_ROWS, so it must gate"
    )
    assert kernels_module.gated(SPECIES_SIZE, POP_SIZE, "accelerator") is False, (
        "the accelerator leg never gates, however small the table is"
    )


def test_the_index_free_formulation_seam_is_the_branch_the_static_selects(
    payloads, monkeypatch
):
    """``distance_formulation`` really switches the branch inside the alignment seam.

    The seam itself (``index_free.alignment``) runs on *both* formulations, so the
    discriminating call sites are its two arms: ``index_free.resolve``
    (``"index_free"``) and ``digest.measure_alignment`` (``"merge"``).  Each is spied at
    trace time under both ``kernel_mode`` values, and the table each arm produces is
    compared with the reference table, so "the branch changed but the numbers did not"
    is asserted rather than assumed.
    """
    genome = payloads.default_genome()
    payload = population(payloads)
    centres = centre_rows(payload)
    reference = measure_table(payload, centres, REFERENCE, genome)

    seam_calls: list = []
    resolve_calls: list = []
    merge_calls: list = []
    original_alignment = index_free_module.alignment
    original_resolve = index_free_module.resolve
    original_measure_alignment = digest_module.measure_alignment

    def alignment_spy(*args, **kwargs):
        """The seam itself: reached by both formulations, hence not discriminating."""
        seam_calls.append(True)
        return original_alignment(*args, **kwargs)

    def resolve_spy(*args, **kwargs):
        """The ``index_free`` arm: an equality table instead of a merge."""
        resolve_calls.append(True)
        return original_resolve(*args, **kwargs)

    def measure_alignment_spy(*args, **kwargs):
        """The ``merge`` arm: the shipped sort-based successor test."""
        merge_calls.append(True)
        return original_measure_alignment(*args, **kwargs)

    monkeypatch.setattr(index_free_module, "alignment", alignment_spy)
    monkeypatch.setattr(index_free_module, "resolve", resolve_spy)
    monkeypatch.setattr(digest_module, "measure_alignment", measure_alignment_spy)

    recorded = {}
    for kernel_mode in accelerator_module.KERNEL_MODES:
        for formulation in index_free_module.DISTANCE_FORMULATION_MODES:
            seam_calls.clear()
            resolve_calls.clear()
            merge_calls.clear()
            route = Route(kernel_mode=kernel_mode, distance_formulation=formulation)
            table = measure_table(payload, centres, route, genome)
            _assert_identical(f"distance table ({route.label})", reference, table)
            recorded[route.label] = (
                len(seam_calls),
                len(resolve_calls),
                len(merge_calls),
            )

    for kernel_mode in accelerator_module.KERNEL_MODES:
        equality = recorded[
            Route(kernel_mode=kernel_mode, distance_formulation="index_free").label
        ]
        merged = recorded[
            Route(kernel_mode=kernel_mode, distance_formulation="merge").label
        ]
        # The seam runs on both formulations with the same trace volume, which is what
        # keeps the switch a single host branch over one shared record; then exactly
        # one arm of it ran, the one the static names.
        assert equality[0] == merged[0] > 0, (
            "the alignment seam must be reached by both formulations with the same "
            f"call volume: {equality} vs {merged}"
        )
        assert equality[1] > 0 and equality[2] == 0, (
            "index_free must take the equality-table arm only, got "
            f"(seam, resolve, merge) = {equality}"
        )
        assert merged[1] == 0 and merged[2] > 0, (
            "merge must take the shipped merge arm only, got "
            f"(seam, resolve, merge) = {merged}"
        )


# --- the reference route is the absent-entry default ---


def test_the_reference_route_is_the_default_spelling(common):
    """The reference point is what a bare ``State`` resolves to, entry by entry.

    Three independent readers own the three fallbacks, and :data:`REFERENCE` spells them
    out explicitly; comparing their answers on a bare ``State`` and on the registered
    one keeps "the shipped route" from drifting away from "the default route", which
    would make every comparison above use a non-default reference.
    """
    bare = common.State()
    explicit = REFERENCE.apply(bare)
    readers = (
        accelerator_module.kernel_mode,
        validation_module.distance_centre_mode,
        index_free_module.formulation_mode,
    )
    for reader in readers:
        assert reader(bare) == reader(explicit), (
            f"{reader.__name__} answers {reader(bare)!r} on a bare State but "
            f"{reader(explicit)!r} on the reference route"
        )
    assert (
        accelerator_module.kernel_mode(bare),
        validation_module.distance_centre_mode(bare),
        index_free_module.formulation_mode(bare),
    ) == (
        REFERENCE.kernel_mode,
        REFERENCE.distance_centre,
        REFERENCE.distance_formulation,
    ), "REFERENCE must be the absent-entry default of all three statics"
    assert grouped_module.group_size(explicit, POP_SIZE, SPECIES_SIZE) == 1, (
        "the reference route is the ungrouped plain route"
    )
