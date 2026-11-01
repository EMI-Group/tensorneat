"""Unit tests for ``NEAT`` — the generation algebra of ``tell`` and the constructor contract.

What this module proves
-----------------------
``tell`` closes one generation and opens the next: sanitise the fitness, advance the
counter, update stagnation and spawn, cross and mutate, apply elitism, re-speciate.
The species-level pieces live in ``./test_spawn.py``/``./test_parent_selection.py``/
``./test_speciation.py``; this module pins what only ``NEAT`` owns:

* ``_sanitize_fitness`` maps ``NaN`` **and both infinities** to ``-inf`` *before*
  selection, so an unevaluable topology can never win and a species'
  ``best_fitness`` can never become ``NaN``;
* ``generation`` is incremented at the **start** of ``tell`` (the legacy order): a
  species created by the re-speciation records the *incremented* generation in
  ``last_improved``;
* stagnation kills a species that fails to improve within ``max_stagnation`` and
  never kills one of the top ``species_elitism`` ranked species;
* the population size and every payload/species shape and dtype survive a full
  ``tell``, the ``randkey`` advances, and ``ask`` neither changes state nor costs an op;
* ``tell`` is deterministic for ``(state, fitness)``;
* ``pop_batch_size`` chunking is a pure reshape (see the RNG caveat below);
* the constructor rejects every documented out-of-domain argument with ``TensorNEATError``;
* ``ask`` → ``transform`` → ``forward`` wire up on the default genome, down to the
  documented ``Transformed`` fields and a finite ``(P, num_outputs)`` output.

Strict xfails: a source defect
------------------------------
``NEAT.tell`` raises ``evoxc.ShapeError`` for any ``pop_size > 1``.  The
structural-key allocation in ``_new_gene_keys`` does
``evoxc.reshape(self.genome.max_node_key(state, node_keys), ())``, but the genome's
documented and implemented ``max_node_key`` returns ``(P,)`` (one maximum per
individual), so the reshape has ``P`` elements and a ``()`` target: ``ShapeError:
reshape: element counts differ: 8 vs 1`` at ``neat.py:574``.  ``pop_size == 1``
runs to completion, which pins the cause to that reshape.  The four tests that
drive a real ``tell`` carry their full assertions but are marked
``xfail(strict=True)`` with that reason: they flip to XPASS (a failure) once the
source is fixed, and nothing else here needs ``tell``.

Oracles used here (never the implementation)
--------------------------------------------
:func:`sanitize_reference` and :func:`stagnation_reference` are closed forms, the
generation/shape/dtype/key claims are read back from the state itself, and the
chunking test passes ``lambda key, x: x + 1`` to ``_batched`` so only the reshape
and the gather can differ between the chunked and the unchunked call.

How a traced method is run
--------------------------
``evoxc.build`` turns only an ``ndarray``/``Tensor`` argument into a graph input and
bakes the rest in as statics, so :func:`run_tell` passes the state's *tensor leaves*
positionally and rebuilds the ``State`` (and its dataclasses) **inside** the traced
closure; the host configuration travels as statics.  The graph returns nested tuples,
which ``support.runner.to_numpy`` turns into plain numpy arrays.

Gotchas
-------
* ``generation`` is a 0-d **float32**, not an int (it is written into the float32
  ``last_improved`` inside a scan carry), so comparisons cast explicitly.
* ``_sanitize_fitness`` is a module-level function reached as
  ``neat_module._sanitize_fitness(fitness, pop_size)``; ``pop_size`` exists only to
  validate the shape at trace time.
* **Chunking and RNG.** a chunk draws ``pop_batch_size`` rows of randomness from its
  own subkey, so numerical equality between a chunked and an unchunked *mutation* is
  not a contract and is not asserted; only the population axis and the shapes are.
* ``weight_threshold``/``max_weight`` are ``HyperNEAT.__init__`` arguments
  (``./hyperneat/test_validation.py``), and a ``species_fitness_func`` without
  ``axes=`` fails at trace time, not in the constructor.
* The hand-written species tables are built here rather than with the shared
  ``species_state`` fixture: that factory still spells the centre arrays
  ``center_nodes``/``center_conns``, which the implemented ``SpeciesState``
  (``center_node_keys``/``center_node_attrs``/``center_conn_keys``/
  ``center_conn_attrs``) does not accept.

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
    """``np.where(np.isfinite(f), f, -inf)`` — the documented sanitising map.

    ``evoxc`` has ``isnan`` but no ``isfinite``, so the kernel's
    ``finite = ~isnan(f - f)`` must agree with ``np.isfinite`` on every entry,
    including ``-inf`` staying ``-inf`` (the map is idempotent, not ``-f``-based).
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
    """``(f_s <= best) & (generation - last_improved > max_stagnation)``, minus the elite ranks.

    Args:
        best_fitness: ``(S,)`` float, ``NaN`` on dead slots.
        last_improved: ``(S,)`` float generation of the last improvement.
        species_fitness: ``(S,)`` float, this generation's per-species maximum.
        generation: float, the generation that produced ``species_fitness``.
        max_stagnation: int, the allowed number of unimproved generations.
        species_elitism: int, how many best-ranked species are exempt.

    Returns:
        ``(S,) bool``: True where the species must be killed.

    Notes:
        The rank exemption is derived from ``species_fitness`` itself (descending
        stable ``argsort``, i.e. ``rank_elements``' inverse permutation: ties keep
        their original order), so the oracle never depends on the order the kernel
        happens to produce.
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
    """Build a :class:`SpeciesState` from the scalar fields, with empty centre rows.

    Args:
        species_module: The ``tensorneat.algorithm.neat.species`` module fixture.
        species_keys: ``(S,)`` keys; ``NaN`` marks a dead slot.
        best_fitness: ``(S,)`` best per-species fitness (``NaN`` where dead).
        last_improved: ``(S,)`` generation of the last improvement.
        member_count: ``(S,)`` member counts (``NaN`` where dead).
        idx2species: ``(P,)`` species key of each individual.
        next_species_key: 0-d key handed to the next created species.
        max_nodes: Static node capacity of the (unused) centre rows.
        max_conns: Static connection capacity of the (unused) centre rows.

    Returns:
        The ``SpeciesState``.

    Notes:
        ``_species_fitness``, ``_stagnation`` and ``_pair_parents`` read only the
        five scalar fields, so the centre arrays are zero-filled at the
        ``DefaultGenome`` widths and never inspected — exactly like a dead slot's
        centre rows, which the contract says are never read.
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
    """Run one ``NEAT.tell`` inside a graph and return the next state's numpy arrays.

    Returns:
        tuple: ``(population, generation, species, randkey)`` — the payload and the
        species fields as tuples in dataclass order, then the 0-d generation
        (``float32``) and ``randkey`` (``int64``).

    Notes:
        The ``State`` and both dataclasses are rebuilt *inside* the traced closure:
        a container passed as an argument would be baked in as a static constant
        whose leaves are plain numpy.  ``pop_size``, the gene configuration and the
        operators genuinely are static.
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
    """Run ``transform`` + ``forward`` twice on one population, inside one graph.

    Returns:
        tuple: ``(transformed, first, second)`` — the genome's ``Transformed``
        pytree and two ``(P, num_outputs)`` outputs that must agree.
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
    """Generation-0 state: ``algorithm.setup(State(randkey=evoxc_key))``.

    ``Pipeline.setup`` is normally the one that creates ``randkey``; the suites do it
    directly so a run is reproducible from the ``evoxc_key`` fixture (which is seeded
    from the test id, see ``../conftest.py``).
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
# selection never sees a raw non-finite fitness
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract")
def test_a_non_finite_fitness_can_never_become_a_parent(
    species, controller, table, species_kernel, neat_module, evoxc_key
):
    """Invariant: sanitised before selection, so a NaN/inf genome loses to any finite one.

    Oracle: :func:`sanitize_reference` on the raw fitness, then the documented
    selection steps on a hand-written table with two live species:

    * species ``0`` holds rows ``0..3`` with raw fitness ``[NaN, +inf, 0.9, -inf]``,
      species ``1`` holds rows ``4..7`` with finite fitness;
    * ``_species_fitness`` reduces with a masked maximum, so the broken species
      scores the finite ``0.9`` of its one good member — never ``NaN`` — while a
      species whose members are *all* non-finite scores ``-inf``;
    * ``_pair_parents`` ranks each species' members by the sanitised fitness, so the
      broken members are never the species' best (never elite-pinned) and never the
      fitter parent of a pair.  The exact parent *rows* are a random draw, so the
      assertable claim is the structural one: every winner's fitness is finite and no
      winner is worse than its loser.
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
        "_species_fitness", controller(species_elitism=0), bookkeeping, sanitized
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
        controller(species_elitism=0),
        bookkeeping,
        np.asarray(sanitize_reference(broken_raw)),
    )
    assert np.isneginf(broken_fitness[0])
    assert broken_fitness[1] == np.float32(0.6)
    assert not np.isnan(broken_fitness).any()

    spawn = np.array([4, 4, 0, 0], dtype=np.int32)
    winner, loser, elite_mask = species_kernel(
        "_pair_parents",
        controller(species_elitism=0),
        bookkeeping,
        evoxc_key,
        spawn,
        np.asarray(sanitized),
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
# stagnation
# --------------------------------------------------------------------------- #


def test_stagnation_kills_a_species_that_stops_improving(
    species, controller, table, species_kernel
):
    """Invariant: ``(f_s <= best_fitness) & (generation - last_improved > max_stagnation)`` kills.

    Oracle: :func:`stagnation_reference` on a table with three live slots — one that
    improved this generation, one at exactly ``max_stagnation`` unimproved
    generations (must **survive**, the comparison is ``>``) and one just past it
    (must die) — plus an already-dead slot no comparison touches.  A killed slot is
    ``NaN`` in every float field and ``-inf`` in its species fitness, so the
    descending sort pushes it last.
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
        controller(max_stagnation=max_stagnation, species_elitism=0),
        bookkeeping,
        np.asarray(species_fitness),
        np.float32(generation),
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
    """Invariant: the ``species_elitism`` best-ranked species are never killed by stagnation.

    Oracle: :func:`stagnation_reference` with ``species_elitism = 1`` over a table
    where the *rank 0* species is maximally stagnant; it survives while a
    lower-ranked equally stagnant species dies.  The same table re-run with
    ``species_elitism = 0`` kills rank 0 too, which makes this a test of the
    exemption rather than of the predicate.  Slot 2 pins the ``<=``: its fitness
    equals its ``best_fitness``, so it is not an improvement and it dies.
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
            controller(
                max_stagnation=max_stagnation, species_elitism=species_elitism
            ),
            bookkeeping,
            np.asarray(species_fitness),
            np.float32(generation),
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
# the population survives a generation
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason=TELL_DEFECT)
def test_tell_preserves_the_population_size_and_payload_shapes(
    algorithm, neat_state, species
):
    """Invariant: after ``tell`` the population has the same static shapes, dtypes and size.

    Oracle: the generation-0 payload read before the call — every tensor keeps its
    shape and dtype, every species field keeps its ``(S, ...)`` shape,
    ``idx2species`` is fully assigned (no ``NaN``) and no live slot has a ``NaN``
    member count; shapes are static at build time, so a mismatch breaks the cache.
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
    """Invariant: ``ask`` returns the stored population unchanged and never consumes a key.

    Oracle: both calls return ``state.pop`` itself (and byte-equal tensors), and
    ``state.randkey`` is byte-identical before and after — ``ask`` is the documented
    zero-op hand-off, and a stray ``random.split`` would desynchronise the RNG budget
    and make a seeded run irreproducible.
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
    """Invariant: ``setup`` on a state without ``randkey`` raises ``TensorNEATError``.

    Oracle: the documented ``Raises`` clause — ``Pipeline.setup`` is the only place
    that creates the key, and a silent fallback to a default seed would break the
    "same seed reproduces the run" contract.  Both spellings are checked: an explicit
    empty ``State`` and the ``None`` default that builds one in the body.
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
    """Invariant: ``tell(state, fitness)`` twice from the same state yields identical next states.

    Oracle: byte equality over the population pytree, every species field and the
    advanced ``randkey``.  Both calls start from the same immutable state and the
    same raw fitness, so a host RNG, a dict order or a mutable default would show up
    as a divergence.
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
    """Invariant: the next state carries a different ``randkey`` than the one it consumed.

    Oracle: ``not same_bytes(next_state.randkey, state.randkey)`` after a full
    ``tell`` (a 0-d ``int64``, compared byte-exactly).  Two generations consuming the
    same key would produce identical mutations — a bug that still passes a
    single-generation test.
    """
    fitness = np.array([0.8, 0.1, 0.6, 0.2, 0.9, 0.3, 0.5, 0.4], dtype=np.float32)

    _, generation, _, next_key = run_tell(algorithm, neat_state, fitness)

    assert not same_bytes(next_key, to_numpy(neat_state.randkey))
    assert np.asarray(next_key).dtype == np.asarray(to_numpy(neat_state.randkey)).dtype


# --------------------------------------------------------------------------- #
# pop_batch_size chunking
# --------------------------------------------------------------------------- #


def test_batched_chunking_preserves_the_population_axis(neat, evoxc_key):
    """Invariant: ``_batched`` is a pure reshape — chunked and unchunked agree for a deterministic kernel.

    Oracle: ``lambda key, x: x + 1`` run through ``_batched`` on an ``(P, 3)`` tensor
    with ``pop_batch_size`` ``None``, ``4`` and ``1``; the outputs must be
    ``same_bytes`` and keep the input's shape/dtype, and the sequence-returning
    branch must restore both leaves.  Numerical equality of two *real* mutation runs
    is deliberately not asserted: a chunk draws its own subkey's randomness, so the
    streams differ by design and chunking is part of the reproducibility config.
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
    """Invariant: one generation-0 payload decodes and runs as a batch-first schedule.

    Oracle: the documented ``Transformed`` field list and its static shapes — the
    population axis on every field, ``(P, max_nodes)`` node tables,
    ``(P, num_inputs)``/``(P, num_outputs)`` slots — plus a finite ``float32``
    ``(P, num_outputs)`` output that two identical ``forward`` calls reproduce.
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
    """Invariant: every documented out-of-domain constructor argument raises ``TensorNEATError``.

    Oracle: the ``Raises`` clause of ``NEAT.__init__`` plus each parameter's
    documented domain, one case per rule.  The rewrite replaced the legacy
    ``assert``/``ValueError`` with ``TensorNEATError`` so a caller can catch
    configuration errors without catching programming errors, and the message must
    name the offending parameter so the failure is actionable.
    """
    with pytest.raises(errors.TensorNEATError) as excinfo:
        neat(**overrides)

    message = str(excinfo.value)
    assert parameter in message, f"error message does not name {parameter!r}: {message}"


def test_constructor_accepts_a_dividing_pop_batch_size(neat):
    """Invariant: a ``pop_batch_size`` that divides ``pop_size`` is accepted and stored.

    Oracle: ``NEAT(pop_batch_size=4)`` does not raise, keeps ``pop_size == 8`` and
    stores ``4`` (two chunks).  Pairing this with the rejection table keeps the
    validation from being a blanket "reject any int".
    """
    algorithm = neat(pop_batch_size=4)

    assert algorithm.pop_size == POP_SIZE
    assert algorithm._pop_batch_size == 4
    assert algorithm.pop_size // algorithm._pop_batch_size == 2
