"""Unit tests for ``NEAT`` — the generation algebra of ``tell`` and the constructor contract.

What this module proves
-----------------------
``tell`` closes one generation and opens the next; the species-level pieces live in
``./test_spawn.py``/``./test_parent_selection.py``/``./test_speciation.py``, and this
module pins what only ``NEAT`` owns: ``_sanitize_fitness`` maps every non-finite entry
to ``-inf`` *before* selection, ``generation`` is incremented at the **start** of
``tell`` (so a created species records the new value), stagnation spares the top
``species_elitism`` ranked species, the population and species shapes survive a
generation, ``tell`` is deterministic for ``(state, fitness)``, the ``randkey``
advances, ``ask`` costs no op, ``pop_batch_size`` chunking is a pure reshape, the
constructor rejects every out-of-domain argument, and ``ask`` → ``transform`` →
``forward`` wire up on the default genome.

Strict xfails: a source defect
------------------------------
``NEAT.tell`` raises ``evoxc.ShapeError`` for any ``pop_size > 1``: ``_new_gene_keys``
reshapes the genome's ``(P,)`` ``max_node_key`` result to ``()`` (``neat.py:574``).
``pop_size == 1`` runs to completion, which pins the cause, so the four tests that
drive a real ``tell`` keep their full assertions under ``xfail(strict=True)``: they
flip to XPASS once the source is fixed.

Oracles used here (never the implementation)
--------------------------------------------
:func:`sanitize_reference` and :func:`stagnation_reference` are closed forms, the
generation/shape/dtype/key claims are read back from the state itself, and the
chunking test passes ``lambda key, x: x + 1`` to ``_batched``.

How a traced method is run
--------------------------
``evoxc.build`` turns only an ``ndarray``/``Tensor`` argument into a graph input, so
:func:`run_tell` passes the state's *tensor leaves* positionally and rebuilds the
``State`` (and its dataclasses) **inside** the traced closure.

Gotchas
-------
* ``generation`` is a 0-d **float32**, not an int (it is written into the float32
  ``last_improved`` inside a scan carry), so comparisons cast explicitly.
* **Chunking and RNG.** a chunk draws ``pop_batch_size`` rows of randomness from its
  own subkey, so numerical equality between a chunked and an unchunked *mutation* is
  not a contract and is not asserted; only the population axis and the shapes are.
* The hand-written species tables are built here rather than with the shared
  ``species_state`` fixture, whose centre-array field names are stale.

Backend is ``"numpy"`` throughout: no optional dependency, no ``jax``/``torch``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from support.runner import run_defn, to_numpy

#: Population size of every table in this module (the default gate's small sizes).
POP_SIZE = 8
#: The four split payload arrays of ``genome.utils.GenomeTensors``, in field order.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")
#: State entries that are *runtime tensors*; everything else is static configuration.
STATE_TENSOR_KEYS = ("pop", "generation", "species", "randkey")

#: One-line cause shared by every strict xfail below (see the module docstring).
TELL_DEFECT = (
    "src defect: NEAT._new_gene_keys reshapes genome.max_node_key's (P,) result "
    "to () -> evoxc.ShapeError for pop_size > 1"
)


# --------------------------------------------------------------------------- #
# oracles and helpers
# --------------------------------------------------------------------------- #


def sanitize_reference(fitness):
    """
    ``np.where(np.isfinite(f), f, -inf)`` — the documented sanitising map.
    ``evoxc`` has ``isnan`` but no ``isfinite``, and the map is idempotent (``-inf``
    stays ``-inf``), not ``-f``-based.
    """
    values = np.asarray(fitness, dtype=np.float64)
    return np.where(np.isfinite(values), values, -np.inf).astype(np.float32)


def stagnation_reference(
    best_fitness,
    last_improved,
    species_fitness,
    generation,
    max_stagnation,
    species_elitism,
):
    """
    ``(f_s <= best) & (generation - last_improved > max_stagnation)``, minus elite ranks.
    The rank exemption comes from the descending stable ``argsort`` of ``species_fitness``
    (ties keep their order), so the oracle never depends on the kernel's own order.
    """
    best = np.asarray(best_fitness, dtype=np.float64)
    improved_at = np.asarray(last_improved, dtype=np.float64)
    current = np.asarray(species_fitness, dtype=np.float64)
    rank = np.empty_like(current)
    rank[np.argsort(-current, kind="stable")] = np.arange(current.shape[0])
    stagnant = (current <= best) & ((generation - improved_at) > max_stagnation)
    return stagnant & ~(rank < species_elitism)


def same_bytes(left, right):
    """True when two arrays share shape, dtype and content byte for byte.

    Bytes make ``NaN`` equal to ``NaN`` and keep ``-0.0`` distinct from ``0.0``.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return left.tobytes() == right.tobytes()


def species_field_names(species_module):
    """Names of every ``SpeciesState`` field, in the dataclass's declaration order."""
    return tuple(field.name for field in dataclasses.fields(species_module.SpeciesState))


def species_arrays(species_module, species_state):
    """Read a ``SpeciesState`` back as a ``{field: numpy array}`` mapping.

    ``to_numpy`` rebuilds tuples/lists/dicts/namedtuples only, so the dataclass
    crosses the graph boundary with ``evoxc.Tensor`` leaves; this unwraps them.
    """
    return {
        name: to_numpy(getattr(species_state, name))
        for name in species_field_names(species_module)
    }


def species_table(
    species_module,
    *,
    species_keys,
    best_fitness,
    last_improved,
    member_count,
    idx2species,
    next_species_key=4.0,
    max_nodes=8,
    max_conns=16,
):
    """
    Build a :class:`SpeciesState` from the scalar fields, with empty centre rows.
    The three kernels under test read only those scalars, so the centre arrays are
    zero-filled at the ``DefaultGenome`` widths and never inspected.
    """
    size = len(species_keys)
    return species_module.SpeciesState(
        species_keys=np.asarray(species_keys, dtype=np.float32),
        best_fitness=np.asarray(best_fitness, dtype=np.float32),
        last_improved=np.asarray(last_improved, dtype=np.float32),
        member_count=np.asarray(member_count, dtype=np.float32),
        idx2species=np.asarray(idx2species, dtype=np.float32),
        center_node_keys=np.zeros((size, max_nodes), dtype=np.int32),
        center_node_attrs=np.zeros((size, max_nodes, 4), dtype=np.float32),
        center_conn_keys=np.zeros((size, max_conns, 2), dtype=np.int32),
        center_conn_attrs=np.zeros((size, max_conns, 1), dtype=np.float32),
        next_species_key=np.float32(next_species_key),
    )


def run_sanitize(neat_module, fitness, pop_size=None):
    """Run ``_sanitize_fitness`` in one graph and return the host numpy result.

    ``pop_size`` is a static Python argument baked into the trace (it defaults to
    ``fitness.shape[0]``), so a wrong shape surfaces from ``run_defn`` and not from
    a constructor.
    """
    values = np.asarray(fitness, dtype=np.float32)
    if pop_size is None:
        pop_size = values.shape[0]
    return to_numpy(
        run_defn(lambda x: neat_module._sanitize_fitness(x, pop_size), values)
    )


def _static_entries(state):
    """The host configuration of ``state``: every entry that is not a runtime tensor."""
    return {
        key: getattr(state, key)
        for key in state.registered_keys()
        if key not in STATE_TENSOR_KEYS
    }


def _state_leaves(state, field_names):
    """The ordered tensor leaves of ``state``: payload, generation, species, key."""
    return [
        *(getattr(state.pop, name) for name in POP_FIELDS),
        state.generation,
        *(getattr(state.species, name) for name in field_names),
        state.randkey,
    ]


def _trace_context(state):
    """The static parts of a traced call: container types, host entries, field names."""
    return {
        "field_names": tuple(
            field.name for field in dataclasses.fields(type(state.species))
        ),
        "species_type": type(state.species),
        "payload_type": type(state.pop),
        "state_type": type(state),
        "static": _static_entries(state),
    }


def _rebuild_state(values, context):
    """Rebuild the ``State`` from traced leaves; return it and the next argument index."""
    payload = context["payload_type"](*values[: len(POP_FIELDS)])
    cursor = len(POP_FIELDS)
    generation = values[cursor]
    cursor += 1
    species_state = context["species_type"](
        *values[cursor : cursor + len(context["field_names"])]
    )
    cursor += len(context["field_names"])
    randkey = values[cursor]
    cursor += 1
    current = context["state_type"](**context["static"]).register(
        pop=payload,
        generation=generation,
        species=species_state,
        randkey=randkey,
    )
    return current, cursor


def run_tell(algorithm, state, fitness):
    """
    Run one ``NEAT.tell`` inside a graph; ``(population, generation, species, randkey)``.
    The ``State`` is rebuilt *inside* the traced closure, because a container passed as
    an argument would be baked in as a static constant.
    """
    context = _trace_context(state)

    def kernel(*values):
        current, cursor = _rebuild_state(values, context)
        next_state = algorithm.tell(current, values[cursor])
        return (
            tuple(getattr(next_state.pop, name) for name in POP_FIELDS),
            next_state.generation,
            tuple(
                getattr(next_state.species, name) for name in context["field_names"]
            ),
            next_state.randkey,
        )

    return to_numpy(
        run_defn(
            kernel,
            *_state_leaves(state, context["field_names"]),
            np.asarray(fitness, np.float32),
        )
    )


def run_decode_and_forward(algorithm, state, inputs):
    """
    Run ``transform`` + ``forward`` twice on one population, inside one graph.
    """
    context = _trace_context(state)

    def kernel(*values):
        current, cursor = _rebuild_state(values, context)
        payload = current.pop
        transformed = algorithm.transform(current, payload)
        first = algorithm.forward(current, transformed, values[cursor], payload)
        second = algorithm.forward(current, transformed, values[cursor], payload)
        return transformed, first, second

    return to_numpy(
        run_defn(
            kernel,
            *_state_leaves(state, context["field_names"]),
            np.asarray(inputs, np.float32),
        )
    )


# --------------------------------------------------------------------------- #
# fixtures: an algorithm, its generation-0 state and the hand-built tables
# --------------------------------------------------------------------------- #


@pytest.fixture
def algorithm(neat):
    """The ``NEAT`` instance under test: ``P = 8``, ``S = 4``, no batch chunking."""
    return neat()


@pytest.fixture
def neat_state(algorithm, common, evoxc_key):
    """
    Generation-0 state: ``algorithm.setup(State(randkey=evoxc_key))``.
    ``Pipeline.setup`` normally creates ``randkey``; doing it here keeps a run
    reproducible from the test-id-seeded key.
    """
    return algorithm.setup(common.State(randkey=evoxc_key))


@pytest.fixture
def table(species):
    """Factory: :func:`species_table` bound to the ``species`` module fixture."""
    return lambda **overrides: species_table(species, **overrides)


# --------------------------------------------------------------------------- #
# sanitising fitness
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract")
@pytest.mark.parametrize(
    "raw",
    [
        [np.nan, 0.5, 1.0, 2.0],
        [np.inf, 0.5, 1.0, 2.0],
        [-np.inf, 0.5, 1.0, 2.0],
        [np.nan, np.inf, -np.inf, 2.0],
    ],
)
def test_sanitize_fitness_maps_every_non_finite_entry_to_negative_infinity(
    neat_module, raw
):
    """Invariant: ``NaN``, ``+inf`` and ``-inf`` all become ``-inf``; nothing else changes.

    Oracle: :func:`sanitize_reference`, i.e. ``np.where(np.isfinite(f), f, -inf)``.
    The ``+inf`` case is the one that matters for selection — an infinite fitness
    would otherwise win every ``>=`` comparison forever — and the ``-inf`` case pins
    that the map is idempotent rather than ``-f``-based.
    """
    expected = sanitize_reference(raw)
    sanitized = run_sanitize(neat_module, raw, pop_size=len(raw))

    assert sanitized.shape == (len(raw),)
    assert sanitized.dtype == np.float32
    assert same_bytes(sanitized, expected)
    # Spelled out independently of the oracle: only the non-finite entries moved.
    raw_values = np.asarray(raw, dtype=np.float32)
    assert np.all(np.isneginf(sanitized[~np.isfinite(raw_values)]))
    assert same_bytes(sanitized[np.isfinite(raw_values)], raw_values[np.isfinite(raw_values)])


def test_sanitize_fitness_preserves_every_finite_value(neat_module):
    """Invariant: a fully finite fitness vector comes back bit-identical (and stays float32).

    Oracle: ``same_bytes`` against the input, including the extreme magnitudes, the
    ``±0.0`` pair and a fully finite ordering.  A sanitiser written with a comparison
    against a sentinel (``f < -1e30``) would quietly clamp a legitimate extreme
    fitness; the ``f - f`` trick cannot.
    """
    raw = np.array(
        [-0.0, 0.0, 1.5, -1.5, 1e30, -1e30, 3.0e38, -3.0e38], dtype=np.float32
    )
    sanitized = run_sanitize(neat_module, raw, pop_size=raw.size)

    assert sanitized.dtype == np.float32
    assert same_bytes(sanitized, raw)
    # ``same_bytes`` already covers this; the explicit sign check documents that
    # ``-0.0`` is *not* canonicalised to ``0.0``.
    assert np.signbit(sanitized[0]) and not np.signbit(sanitized[1])


@pytest.mark.parametrize("shape", [(8, 1), (2, 4), (8, 8, 8)])
def test_sanitize_fitness_rejects_a_non_population_vector(neat_module, errors, shape):
    """Invariant: a fitness tensor that is not ``(P,)`` raises ``TensorNEATError``.

    Oracle: the documented ``Raises`` clause; ``pop_size`` is a static argument, so
    the shape check happens at trace time and must be an explicit named error rather
    than a broadcast that silently accepts a column vector (which would make every
    later ``(P,)`` gather read the wrong axis).
    """
    fitness = np.zeros(shape, dtype=np.float32)

    with pytest.raises(errors.TensorNEATError) as excinfo:
        run_sanitize(neat_module, fitness, pop_size=POP_SIZE)

    message = str(excinfo.value)
    assert "fitness must have shape" in message
    assert f"({POP_SIZE},)" in message


# --------------------------------------------------------------------------- #
# the generation counter
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason=TELL_DEFECT)
def test_tell_increments_the_generation_before_the_species_update(
    algorithm, neat_state, species
):
    """Invariant: ``generation`` grows by exactly one, and new species record the *new* value.

    Oracle: ``state.generation == previous + 1`` (as ``float32``), every species
    created during the re-speciation (``best_fitness == -inf``, key live) has
    ``last_improved == state.generation``, and the initial species — which improved
    for the first time on this finite fitness vector — records the incremented
    generation too.  The ordering inside ``tell`` is the observable: if the counter
    advanced *after* the species update, a freshly created species would claim the
    previous generation and get one extra generation of grace before stagnation could
    kill it.
    """
    previous = float(to_numpy(neat_state.generation))
    fitness = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)

    population, generation, species_values, randkey = run_tell(
        algorithm, neat_state, fitness
    )
    fields = dict(zip(species_field_names(species), species_values))

    assert float(generation) == np.float32(previous + 1.0)
    # The initial species improved (any finite fitness beats its -inf seed)...
    assert fields["last_improved"][0] == np.float32(previous + 1.0)
    # ...and every species the re-speciation created carries the same value.
    live = ~np.isnan(fields["species_keys"])
    created = live & (fields["best_fitness"] == -np.inf)
    assert np.all(fields["last_improved"][created] == np.float32(previous + 1.0))
    assert np.all(fields["last_improved"][live] <= np.float32(previous + 1.0))



# --------------------------------------------------------------------------- #
# the generation counter, through a whole ``tell`` (``pop_size = 1``)
# --------------------------------------------------------------------------- #


def test_tell_increments_the_generation_before_the_stagnation_bookkeeping(
    neat, common, evoxc_key, species
):
    """Invariant: ``tell`` advances ``generation`` *before* the species update.

    Oracle: ``pop_size = 1`` — the one size the ``_new_gene_keys`` reshape defect does
    not break — starts at generation ``0`` with the seeded species at
    ``best_fitness = -inf`` and ``last_improved = 0``.  The first finite fitness
    improves it, so the recorded generation must be the already-incremented ``1``, not
    the generation that produced the fitness (which would grant a stale species one
    extra generation of grace before stagnation can kill it).
    """
    algorithm = neat(pop_size=1, species_size=1, min_species_size=1)
    state = algorithm.setup(common.State(randkey=evoxc_key))
    previous = float(to_numpy(state.generation))
    assert previous == 0.0

    _, generation, species_values, _ = run_tell(
        algorithm, state, np.array([0.3], dtype=np.float32)
    )
    fields = dict(zip(species_field_names(species), species_values))

    assert float(generation) == np.float32(previous + 1.0)
    assert fields["last_improved"][0] == np.float32(previous + 1.0)
    assert fields["best_fitness"][0] == np.float32(0.3)


# --------------------------------------------------------------------------- #
# selection never sees a raw non-finite fitness
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract")
def test_a_non_finite_fitness_can_never_become_a_parent(
    species, controller, table, species_kernel, neat_module, evoxc_key
):
    """
    Invariant: sanitised before selection, so a NaN/inf genome loses to any finite one.
    Oracle: :func:`sanitize_reference` then the documented selection steps on a table
    whose species ``0`` holds ``[NaN, +inf, 0.9, -inf]``: the masked ``max`` scores the
    finite ``0.9`` (and ``-inf`` when every member is non-finite), so no non-finite row
    is ever elite-pinned or drawn as the fitter parent.  Parent rows are a random draw,
    so the assertable claim is structural: finite winners, never worse than the loser.
    """
    raw = np.array([np.nan, np.inf, 0.9, -np.inf, 0.4, 0.1, 0.6, 0.2], np.float32)
    sanitized = sanitize_reference(raw)
    assert same_bytes(run_sanitize(neat_module, raw, pop_size=POP_SIZE), sanitized)

    bookkeeping = table(
        species_keys=[0.0, 1.0, np.nan, np.nan],
        best_fitness=[-np.inf, -np.inf, np.nan, np.nan],
        last_improved=[0.0, 0.0, np.nan, np.nan],
        member_count=[4.0, 4.0, np.nan, np.nan],
        idx2species=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    species_fitness = species_kernel(
        "_species_fitness",
        bookkeeping,
        sanitized,
        controller=controller(species_elitism=0),
    )
    assert not np.isnan(species_fitness).any()
    assert species_fitness[0] == np.float32(0.9)
    assert species_fitness[1] == np.float32(0.6)
    assert np.all(np.isneginf(species_fitness[2:]))

    # A species made only of non-finite genomes reduces to ``-inf``, never ``NaN``:
    # that keeps ``species_number_calculate_by="fitness"``'s reductions finite.  The
    # input must be the *sanitised* vector -- ``tell`` sanitises before this
    # reduction, and a raw ``NaN`` would poison ``max``.
    broken_raw = np.array(
        [np.nan, np.inf, -np.inf, np.nan, 0.4, 0.1, 0.6, 0.2], np.float32
    )
    broken_fitness = species_kernel(
        "_species_fitness",
        bookkeeping,
        np.asarray(sanitize_reference(broken_raw)),
        controller=controller(species_elitism=0),
    )
    assert np.isneginf(broken_fitness[0])
    assert broken_fitness[1] == np.float32(0.6)
    assert not np.isnan(broken_fitness).any()

    spawn = np.array([4, 4, 0, 0], dtype=np.int32)
    winner, loser, elite_mask = species_kernel(
        "_pair_parents",
        bookkeeping,
        evoxc_key,
        spawn,
        np.asarray(sanitized),
        controller=controller(species_elitism=0),
    )

    assert winner.shape == (POP_SIZE,)
    assert np.isfinite(sanitized[winner]).all()
    assert np.isfinite(sanitized[loser]).all()
    # Winner never worse than loser, and never a raw non-finite value.
    assert np.all(sanitized[winner] >= sanitized[loser])
    # The survivors of each species (survival_threshold * 4 floors to 1) are its
    # single best member: the non-finite rows can never be drawn or elite-pinned.
    assert np.all(winner[:4] == 2) and np.all(winner[4:] == 6)
    assert np.all(elite_mask == np.array([True, True, False, False] * 2))


# --------------------------------------------------------------------------- #
# species membership is a key match, not a slot match
# --------------------------------------------------------------------------- #


def test_species_membership_matches_keys_by_value_not_by_slot(
    controller, table, species_kernel
):
    """Invariant: ``idx2species`` holds species *keys*, so membership is a value match.

    Oracle: keys ``[5.0, 2.0]`` — neither equal to its slot index — with
    ``idx2species`` carrying those same keys.  ``_species_fitness`` reduces each
    species over its own rows (``0.9`` for key ``5.0``, ``0.7`` for key ``2.0``),
    which a mask built from ``arange(species_size)`` cannot reproduce: it matches
    nothing and reports ``-inf`` for both live species.
    """
    fitness = np.array([0.4, 0.9, 0.1, 0.2, 0.7, 0.1, 0.5, 0.3], np.float32)
    bookkeeping = table(
        species_keys=[5.0, 2.0, np.nan, np.nan],
        best_fitness=[-np.inf, -np.inf, np.nan, np.nan],
        last_improved=[0.0, 0.0, np.nan, np.nan],
        member_count=[4.0, 4.0, np.nan, np.nan],
        idx2species=[5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0],
    )

    species_fitness = species_kernel(
        "_species_fitness",
        bookkeeping,
        fitness,
        controller=controller(species_elitism=0),
    )

    np.testing.assert_array_equal(
        species_fitness, np.array([0.9, 0.7, -np.inf, -np.inf], np.float32)
    )


# --------------------------------------------------------------------------- #
# stagnation
# --------------------------------------------------------------------------- #


def test_stagnation_kills_a_species_that_stops_improving(
    species, controller, table, species_kernel
):
    """
    Invariant: ``(f_s <= best_fitness) & (generation - last_improved > max_stagnation)``.
    Oracle: :func:`stagnation_reference` over three live slots — improved, exactly at
    ``max_stagnation`` (survives, the comparison is ``>``) and one past it (dies) — plus
    a dead slot no comparison touches; a killed slot is ``NaN`` everywhere and ``-inf``
    in its species fitness.
    """
    generation = 10.0
    max_stagnation = 5
    best = np.array([0.2, 0.9, 0.8, np.nan], dtype=np.float32)
    last_improved = np.array([7.0, 5.0, 4.0, np.nan], dtype=np.float32)
    species_fitness = np.array([0.5, 0.4, 0.3, -np.inf], dtype=np.float32)
    bookkeeping = table(
        species_keys=[0.0, 1.0, 2.0, np.nan],
        best_fitness=best,
        last_improved=last_improved,
        member_count=[3.0, 3.0, 2.0, np.nan],
        idx2species=[0, 0, 0, 1, 1, 1, 2, 2],
    )

    next_state, next_fitness = species_kernel(
        "_stagnation",
        bookkeeping,
        np.asarray(species_fitness),
        np.float32(generation),
        controller=controller(max_stagnation=max_stagnation, species_elitism=0),
    )
    after = species_arrays(species, next_state)

    already_dead = np.isnan(np.asarray(bookkeeping.species_keys))
    killed = stagnation_reference(
        best,
        last_improved,
        species_fitness,
        generation,
        max_stagnation,
        species_elitism=0,
    )
    expected_killed = already_dead | killed
    assert np.array_equal(np.isnan(after["species_keys"]), expected_killed)
    # The two boundary slots specifically: exactly ``max_stagnation`` survives.
    assert not np.isnan(after["species_keys"][1])
    assert np.isnan(after["species_keys"][2])

    # Killed rows are blanked in every float field and pushed last in the sort.
    for name in ("best_fitness", "last_improved", "member_count"):
        assert np.all(np.isnan(after[name][killed]))
    assert np.all(np.isneginf(next_fitness[killed]))
    # Improving slots refresh both fields; a merely stale one keeps them.
    assert after["best_fitness"][0] == np.float32(0.5)
    assert after["last_improved"][0] == np.float32(generation)
    assert after["best_fitness"][1] == np.float32(0.9)
    assert after["last_improved"][1] == np.float32(5.0)
    # A dead slot is never touched by the NaN comparisons.
    assert np.isnan(after["best_fitness"][3])
    assert np.isnan(after["last_improved"][3])


def test_stagnation_exempts_the_top_species_elitism(
    species, controller, table, species_kernel
):
    """
    Invariant: the ``species_elitism`` best-ranked species are never killed.
    Oracle: :func:`stagnation_reference` where the rank-0 species is maximally stagnant;
    the same table with ``species_elitism = 0`` kills it too, so the test isolates the
    exemption.  Slot 2 pins ``<=``: equal fitness is not an improvement.
    """
    generation = 10.0
    max_stagnation = 2
    best = np.array([0.9, 0.6, 0.5, 0.3], dtype=np.float32)
    last_improved = np.array([1.0, 1.0, 7.0, 9.0], dtype=np.float32)
    species_fitness = np.array([0.5, 0.4, 0.5, 0.35], dtype=np.float32)
    bookkeeping = table(
        species_keys=[0.0, 1.0, 2.0, 3.0],
        best_fitness=best,
        last_improved=last_improved,
        member_count=[2.0, 2.0, 2.0, 2.0],
        idx2species=[0, 0, 1, 1, 2, 2, 3, 3],
    )

    def kill_mask(species_elitism):
        next_state, next_fitness = species_kernel(
            "_stagnation",
            bookkeeping,
            np.asarray(species_fitness),
            np.float32(generation),
            controller=controller(
                max_stagnation=max_stagnation, species_elitism=species_elitism
            ),
        )
        after = species_arrays(species, next_state)
        return np.isnan(after["species_keys"]), after, next_fitness

    killed, after, next_fitness = kill_mask(species_elitism=1)
    expected = stagnation_reference(
        best,
        last_improved,
        species_fitness,
        generation,
        max_stagnation,
        species_elitism=1,
    )
    assert np.array_equal(killed, expected)
    assert not killed[0], "the rank-0 species is exempt from stagnation"
    assert killed[1] and killed[2], "stagnant lower-ranked species die"
    assert not killed[3], "an improving species survives"
    # The exempt species keeps its bookkeeping untouched (it did not improve).
    assert after["best_fitness"][0] == np.float32(0.9)
    assert after["last_improved"][0] == np.float32(1.0)
    assert np.all(np.isneginf(next_fitness[killed]))

    # Control: without the exemption the same table kills rank 0 as well.
    without_exemption, _, _ = kill_mask(species_elitism=0)
    assert without_exemption[0]
    expected_without = stagnation_reference(
        best,
        last_improved,
        species_fitness,
        generation,
        max_stagnation,
        species_elitism=0,
    )
    assert np.array_equal(without_exemption, expected_without)


# --------------------------------------------------------------------------- #
def test_stagnation_exempts_the_best_ranked_species_not_the_first_slot(
    species, controller, table, species_kernel
):
    """Invariant: the ``species_elitism`` exemption follows the fitness rank, not the slot.

    Oracle: three maximally stagnant species whose best-rated one sits in slot ``1``
    (``species_fitness = [0.3, 0.9, 0.2]``).  With ``species_elitism = 1`` only slot
    ``1`` survives, so an exemption keyed on the slot index would instead spare slot
    ``0`` and kill the fittest species in the table.
    """
    generation, max_stagnation = 10.0, 2
    best = np.array([0.9, 0.9, 0.9, np.nan], np.float32)
    last_improved = np.array([0.0, 0.0, 0.0, np.nan], np.float32)
    species_fitness = np.array([0.3, 0.9, 0.2, -np.inf], np.float32)
    bookkeeping = table(
        species_keys=[0.0, 1.0, 2.0, np.nan],
        best_fitness=best,
        last_improved=last_improved,
        member_count=[3.0, 3.0, 2.0, np.nan],
        idx2species=[0, 0, 0, 1, 1, 1, 2, 2],
    )

    next_state, next_fitness = species_kernel(
        "_stagnation",
        bookkeeping,
        np.asarray(species_fitness),
        np.float32(generation),
        controller=controller(max_stagnation=max_stagnation, species_elitism=1),
    )
    killed = np.isnan(species_arrays(species, next_state)["species_keys"])
    expected = stagnation_reference(
        best,
        last_improved,
        species_fitness,
        generation,
        max_stagnation,
        species_elitism=1,
    ) | np.isnan(np.asarray(bookkeeping.species_keys))

    assert np.array_equal(killed, expected)
    assert not killed[1], "the best-ranked species (slot 1) is exempt"
    assert killed[0] and killed[2], "the equally stagnant lower ranks die"
    assert np.all(np.isneginf(next_fitness[killed]))


# --------------------------------------------------------------------------- #
# the population survives a generation
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason=TELL_DEFECT)
def test_tell_preserves_the_population_size_and_payload_shapes(
    algorithm, neat_state, species
):
    """
    Invariant: after ``tell`` every payload and species tensor keeps its shape and dtype.
    """
    before_population = {
        name: to_numpy(getattr(neat_state.pop, name)) for name in POP_FIELDS
    }
    before_species = species_arrays(species, neat_state.species)
    fitness = np.array([0.8, 0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.4], dtype=np.float32)

    population, generation, species_values, randkey = run_tell(
        algorithm, neat_state, fitness
    )

    assert algorithm.pop_size == POP_SIZE
    for name, values in zip(POP_FIELDS, population):
        assert values.shape == before_population[name].shape
        assert values.dtype == before_population[name].dtype

    after_species = dict(zip(species_field_names(species), species_values))
    for name, values in after_species.items():
        assert values.shape == before_species[name].shape
    assert np.isnan(after_species["idx2species"]).sum() == 0
    assert np.all(np.isin(after_species["idx2species"], [0.0, 1.0, 2.0, 3.0]))
    live = ~np.isnan(after_species["species_keys"])
    assert live.any()
    assert not np.isnan(after_species["member_count"][live]).any()
    assert float(generation) >= 1.0
    assert np.asarray(randkey).dtype == np.int64


def test_ask_costs_no_ops_and_is_idempotent(algorithm, neat_state):
    """
    Invariant: ``ask`` returns the stored population and consumes no key.
    """
    key_before = to_numpy(neat_state.randkey)
    first = algorithm.ask(neat_state)
    second = algorithm.ask(neat_state)

    assert first is neat_state.pop
    assert second is neat_state.pop
    for name in POP_FIELDS:
        assert same_bytes(
            to_numpy(getattr(first, name)), to_numpy(getattr(neat_state.pop, name))
        )
        assert same_bytes(
            to_numpy(getattr(first, name)), to_numpy(getattr(second, name))
        )
    assert same_bytes(to_numpy(neat_state.randkey), key_before)


def test_setup_requires_the_pipeline_to_have_created_the_randkey(algorithm, common, errors):
    """
    Invariant: ``setup`` without a ``randkey`` raises ``TensorNEATError``.
    Oracle: the documented ``Raises`` clause, in both spellings (explicit empty ``State``
    and the ``None`` default), since a silent default seed breaks reproducibility.
    """
    with pytest.raises(errors.TensorNEATError) as excinfo:
        algorithm.setup(common.State())
    assert "randkey" in str(excinfo.value)

    with pytest.raises(errors.TensorNEATError) as excinfo:
        algorithm.setup()
    assert "randkey" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason=TELL_DEFECT)
def test_tell_is_deterministic_for_the_same_state_and_fitness(algorithm, neat_state, species):
    """
    Invariant: ``tell(state, fitness)`` twice yields byte-identical next states.
    """
    fitness = np.array([0.8, 0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.4], dtype=np.float32)

    first = run_tell(algorithm, neat_state, fitness)
    second = run_tell(algorithm, neat_state, fitness)

    first_population, first_generation, first_species, first_key = first
    second_population, second_generation, second_species, second_key = second

    for left, right in zip(first_population, second_population):
        assert same_bytes(left, right)
    assert same_bytes(first_generation, second_generation)
    for left, right in zip(first_species, second_species):
        assert same_bytes(left, right)
    assert same_bytes(first_key, second_key)


@pytest.mark.xfail(strict=True, reason=TELL_DEFECT)
def test_tell_advances_the_randkey(algorithm, neat_state):
    """
    Invariant: the next state's ``randkey`` differs from the one it consumed.
    """
    fitness = np.array([0.8, 0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.4], dtype=np.float32)

    _, generation, _, next_key = run_tell(algorithm, neat_state, fitness)

    assert not same_bytes(next_key, to_numpy(neat_state.randkey))
    assert np.asarray(next_key).dtype == np.asarray(to_numpy(neat_state.randkey)).dtype


# --------------------------------------------------------------------------- #
# pop_batch_size chunking
# --------------------------------------------------------------------------- #


def test_batched_chunking_preserves_the_population_axis(neat, evoxc_key):
    """
    Invariant: ``_batched`` is a pure reshape — chunked and unchunked agree for a
    deterministic kernel, and the sequence branch restores both leaves.  Numerical
    equality of two real mutation runs is not a contract (each chunk draws its own
    subkey), so only the population axis and the shapes are asserted.
    """
    values = np.arange(POP_SIZE * 3, dtype=np.float32).reshape(POP_SIZE, 3)

    def run(algorithm):
        def kernel(key, tensor):
            return algorithm._batched(lambda step_key, step: step + 1, key, tensor)

        return run_defn(kernel, evoxc_key, values)

    whole = run(neat())
    assert whole.shape == values.shape
    assert whole.dtype == np.float32
    assert same_bytes(whole, values + 1)

    for batch in (4, 1):
        chunked = run(neat(pop_batch_size=batch))
        assert chunked.shape == values.shape
        assert chunked.dtype == values.dtype
        assert same_bytes(chunked, whole)

    # The sequence branch of ``_batched`` restores the population axis on each leaf.
    def run_sequence(algorithm):
        def kernel(key, tensor):
            first, second = algorithm._batched(
                lambda step_key, step: (step + 1, step * 2), key, tensor
            )
            return first, second

        return run_defn(kernel, evoxc_key, values)

    first, second = run_sequence(neat(pop_batch_size=4))
    assert same_bytes(first, values + 1)
    assert same_bytes(second, values * 2)


# --------------------------------------------------------------------------- #
# ask → transform → forward wiring
# --------------------------------------------------------------------------- #

#: The documented ``Transformed`` fields of ``common.tensorization``.
TRANSFORMED_FIELDS = (
    "src", "dst", "edge_valid", "in_conn", "in_src", "in_valid", "node_level",
    "node_exists", "topo_key", "input_slot", "output_slot", "edge_overflow",
)


def test_ask_transform_forward_wire_up_on_the_default_genome(algorithm, neat_state, rng):
    """
    Invariant: one generation-0 payload decodes to the documented ``Transformed`` fields,
    batch-first, and runs to a finite ``(P, num_outputs)`` output.
    """
    payload = algorithm.ask(neat_state)
    inputs = rng.uniform(-1.0, 1.0, size=(POP_SIZE, algorithm.num_inputs)).astype(
        np.float32
    )

    transformed, first, second = run_decode_and_forward(
        algorithm, neat_state, inputs
    )

    assert type(transformed).__name__ == "Transformed"
    assert tuple(transformed._fields) == TRANSFORMED_FIELDS
    for name in TRANSFORMED_FIELDS:
        values = getattr(transformed, name)
        assert values.shape[0] == POP_SIZE, f"{name} lost the population axis"
    assert transformed.node_exists.shape == (POP_SIZE, algorithm.genome.max_nodes)
    assert transformed.input_slot.shape == (POP_SIZE, algorithm.num_inputs)
    assert transformed.output_slot.shape == (POP_SIZE, algorithm.num_outputs)
    assert transformed.node_level.dtype == np.int32
    assert transformed.node_exists.dtype == np.bool_
    assert transformed.edge_valid.dtype == np.bool_

    assert first.shape == (POP_SIZE, algorithm.num_outputs)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert same_bytes(first, second)
    # ``ask`` is the payload the schedule was built from; shapes must agree.
    assert payload.node_keys.shape[0] == POP_SIZE


# --------------------------------------------------------------------------- #
# constructor validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("overrides", "parameter"),
    [
        ({"species_number_calculate_by": "linear"}, "species_number_calculate_by"),
        ({"pop_batch_size": 3}, "pop_batch_size"),
        ({"pop_batch_size": 0}, "pop_batch_size"),
        ({"pop_size": 0}, "pop_size"),
        ({"species_size": 4, "min_species_size": 3}, "min_species_size"),
        ({"min_species_size": 0}, "min_species_size"),
        ({"max_stagnation": -1}, "max_stagnation"),
        ({"species_elitism": -1}, "species_elitism"),
        ({"genome_elitism": -1}, "genome_elitism"),
        ({"survival_threshold": 0.0}, "survival_threshold"),
        ({"survival_threshold": 1.5}, "survival_threshold"),
        ({"spawn_number_change_rate": -0.1}, "spawn_number_change_rate"),
        ({"spawn_number_change_rate": 1.5}, "spawn_number_change_rate"),
        ({"compatibility_threshold": -1.0}, "compatibility_threshold"),
    ],
)
def test_constructor_rejects_each_out_of_domain_argument(neat, errors, overrides, parameter):
    """
    Invariant: every out-of-domain constructor argument raises ``TensorNEATError`` naming
    the offending parameter, one case per documented rule.
    """
    with pytest.raises(errors.TensorNEATError) as excinfo:
        neat(**overrides)

    message = str(excinfo.value)
    assert parameter in message, f"error message does not name {parameter!r}: {message}"


def test_constructor_accepts_a_dividing_pop_batch_size(neat):
    """
    Invariant: a ``pop_batch_size`` dividing ``pop_size`` is accepted and stored.
    """
    algorithm = neat(pop_batch_size=4)

    assert algorithm.pop_size == POP_SIZE
    assert algorithm._pop_batch_size == 4
    assert algorithm.pop_size // algorithm._pop_batch_size == 2
