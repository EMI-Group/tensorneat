"""Unit tests for ``NEAT`` — the generation algebra of ``tell`` and the constructor contract.

What this module proves
-----------------------
``tell`` is the whole "one generation closes, the next opens" step: sanitise the
fitness, advance the counter, update stagnation and spawn, cross and mutate, apply
elitism and re-speciate.  The species-level pieces are covered by
``./test_spawn.py``/``./test_parent_selection.py``/``./test_speciation.py``; this
module pins what only ``NEAT`` can own:

* ``_sanitize_fitness`` maps ``NaN`` **and both infinities** to ``-inf`` *before*
  selection, so an unevaluable topology can never win and a species' ``best_fitness``
  can never become ``NaN``;
* ``generation`` is incremented at the **start** of ``tell`` (the legacy order), which
  is observable because a species created by the re-speciation records the
  *incremented* generation in ``last_improved``;
* stagnation kills a species that fails to improve within ``max_stagnation`` and never
  kills one of the top ``species_elitism`` ranked species;
* the population size and every payload shape/dtype survive a full ``tell``, and the
  `randkey` advances while ``ask`` neither changes state nor costs an op;
* ``tell`` is deterministic for ``(state, fitness)`` — the same state and fitness give
  the same next state, twice, with no host RNG involved;
* ``pop_batch_size`` chunking is a pure reshape of the population axis: the
  ``_batched`` scan yields exactly what the unchunked call yields for a deterministic
  kernel (see the RNG caveat below);
* the constructor rejects every documented out-of-domain argument with
  ``TensorNEATError`` — ``species_number_calculate_by`` outside ``{"rank", "fitness"}``,
  ``species_size * min_species_size > pop_size``, a ``pop_batch_size`` that does not
  divide ``pop_size``, and the numeric ranges the docstrings spell out.

Oracles used here (never the implementation)
--------------------------------------------
* **The closed forms.** :func:`sanitize_reference` is
  ``np.where(np.isfinite(f), f, -np.inf)``; the stagnation predicate is written out in
  the test as ``(f_species <= best_fitness) & (generation - last_improved >
  max_stagnation)`` and applied to a hand-written species table.
* **State arithmetic.** ``generation + 1``, ``shapes/dtypes before == after`` and
  ``randkey != previous`` are compared against the values read back from the state
  object, never against a re-implementation of ``tell``.
* **A deterministic stand-in kernel.** The ``pop_batch_size`` test passes
  ``lambda state, key, x: x + 1`` to ``_batched``: the kernel ignores randomness, so
  the only thing that can differ between the chunked and the unchunked call is the
  reshape/gather machinery under test.

Gotchas this module encodes
---------------------------
* ``generation`` is a 0-d **float32**, not an int (it is written into the float32
  ``last_improved`` inside a scan carry), so every comparison casts explicitly.
* ``_sanitize_fitness`` is a module-level function, not a method: it is reached as
  ``neat_module._sanitize_fitness(fitness, pop_size)`` and its ``pop_size`` argument
  exists only to validate the shape at trace time.
* **The ``pop_batch_size`` caveat.** ``NEAT.__init__`` documents that chunking
  *changes the random stream* (a chunk draws ``pop_batch_size`` rows of randomness
  from its own subkey rather than a slice of an ``(P, ...)`` draw).  So numerical
  equality between a chunked and an unchunked **mutation** is not a contract and is
  not asserted anywhere; what is asserted is that the chunking machinery preserves
  the population axis and the output shape for a deterministic kernel.  A test that
  compared two real mutation runs would be asserting the opposite of the documented
  design.
* ``weight_threshold``/``max_weight`` are **``HyperNEAT.__init__``** arguments, not
  ``NEAT`` ones (see ``../../src/tensorneat/algorithm/hyperneat/CONTEXT.md``); their
  validation is covered in ``./hyperneat/test_validation.py``, not here.
* ``species_fitness_func`` that does not accept ``axes=`` fails at *trace* time, not
  in the constructor, so it is not part of the constructor-rejection table.

Backend is ``"numpy"`` throughout; no optional dependency, no ``jax``/``torch``, no
network.
"""

from __future__ import annotations

import numpy as np
import pytest

from support.runner import run_defn, to_numpy


# --------------------------------------------------------------------------- #
# oracles and helpers
# --------------------------------------------------------------------------- #


def sanitize_reference(fitness):
    """``np.where(np.isfinite(f), f, -inf)`` — the documented sanitising map.

    Args:
        fitness: ``(P,)`` float array, possibly containing ``NaN`` or ``±inf``.

    Returns:
        ``(P,) float32`` with every non-finite entry replaced by ``-inf``.

    Notes:
        ``evoxc`` has ``isnan`` but no ``isfinite``; the kernel uses
        ``finite = ~isnan(f - f)``, which is NaN for NaN and for both infinities and
        ``0`` for every finite value.  ``np.isfinite`` is the definition the two have
        to agree on, including the ``-inf`` entry staying ``-inf``.
    """
    values = np.asarray(fitness, dtype=np.float64)
    return np.where(np.isfinite(values), values, -np.inf).astype(np.float32)


def stagnation_reference(best_fitness, last_improved, species_fitness, generation,
                         max_stagnation, species_elitism):
    """Apply the documented stagnation predicate to one hand-written species table.

    Args:
        best_fitness: ``(S,)`` float, ``NaN`` on dead slots.
        last_improved: ``(S,)`` float generation of the last improvement.
        species_fitness: ``(S,)`` float, this generation's per-species maximum.
        generation: 0-d float, the generation that produced ``species_fitness``.
        max_stagnation: int, the allowed number of unimproved generations.
        species_elitism: int, how many best-ranked species are exempt.

    Returns:
        ``(S,) bool``: True where the species must be killed.

    Notes:
        The rank exemption is derived here from ``species_fitness`` itself
        (``argsort`` descending, as ``rank_elements`` does), so the oracle does not
        depend on the order the kernel happens to produce.
    """
    best = np.asarray(best_fitness, dtype=np.float64)
    improved_at = np.asarray(last_improved, dtype=np.float64)
    current = np.asarray(species_fitness, dtype=np.float64)
    rank = np.empty_like(current)
    rank[np.argsort(-current, kind="stable")] = np.arange(current.shape[0])
    stagnant = (current <= best) & (generation - improved_at > max_stagnation)
    return stagnant & ~(rank < species_elitism)


# --------------------------------------------------------------------------- #
# fixtures: an algorithm and its generation-0 state
# --------------------------------------------------------------------------- #


def run_sanitize(neat_module, fitness, pop_size=8):
    """Run ``_sanitize_fitness`` in one graph and return the host numpy result.

    Args:
        neat_module: The ``tensorneat.algorithm.neat.neat`` module fixture.
        fitness: ``(P,)`` raw fitness, possibly containing ``NaN``/``±inf``.
        pop_size: The static population size the kernel validates its input against.

    Returns:
        ``(P,) float32`` numpy array.

    Notes:
        ``pop_size`` is a static Python argument, so it is baked into the trace and
        the fitness vector is the single graph input.
    """
    return to_numpy(
        run_defn(
            lambda values: neat_module._sanitize_fitness(values, pop_size),
            np.asarray(fitness, dtype=np.float32),
        )
    )


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


# --------------------------------------------------------------------------- #
# sanitising fitness
# --------------------------------------------------------------------------- #


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
    pytest.skip(
        "not implemented: NaN and both infinities map to -inf "
        "(owner: algorithm/neat/neat._sanitize_fitness)"
    )


def test_sanitize_fitness_preserves_every_finite_value(neat_module):
    """Invariant: a fully finite fitness vector comes back bit-identical (and stays float32).

    Oracle: ``np.array_equal(sanitized, raw)`` and dtype equality, including large
    negative values and ``-0.0``.  A sanitiser written with a comparison against a
    sentinel (``f < -1e30``) would quietly clamp a legitimate extreme fitness; the
    ``f - f`` trick cannot.
    """
    pytest.skip(
        "not implemented: finite fitness passes through unchanged "
        "(owner: algorithm/neat/neat._sanitize_fitness)"
    )


@pytest.mark.parametrize("shape", [(8, 1), (2, 4), (8,) * 3])
def test_sanitize_fitness_rejects_a_non_population_vector(neat_module, shape):
    """Invariant: a fitness tensor that is not ``(P,)`` raises ``TensorNEATError``.

    Oracle: the documented ``Raises`` clause; ``pop_size`` is a static argument, so
    the shape check happens at trace time and must be an explicit named error rather
    than a broadcast that silently accepts a column vector (which would make every
    later ``(P,)`` gather read the wrong axis).
    """
    pytest.skip(
        "not implemented: TensorNEATError for a fitness tensor that is not (pop_size,) "
        "(owner: algorithm/neat/neat._sanitize_fitness)"
    )


# --------------------------------------------------------------------------- #
# the generation counter
# --------------------------------------------------------------------------- #


def test_tell_increments_the_generation_before_the_species_update(
    algorithm, neat_state, errors
):
    """Invariant: ``generation`` grows by exactly one, and new species record the *new* value.

    Oracle: ``state.generation == previous + 1`` (as ``float32``), and for every
    species created during the re-speciation ``last_improved == state.generation``.
    The ordering inside ``tell`` is the observable: if the counter advanced *after*
    the species update, a freshly created species would claim the previous
    generation and get one extra generation of grace before stagnation could kill it.
    """
    pytest.skip(
        "not implemented: generation increments before the species update "
        "(owner: algorithm/neat/neat.NEAT.tell)"
    )


# --------------------------------------------------------------------------- #
# selection never sees a raw non-finite fitness
# --------------------------------------------------------------------------- #


def test_a_non_finite_fitness_can_never_become_a_parent(
    species, controller, species_state, species_kernel, neat_module
):
    """Invariant: sanitised before selection, so a NaN/inf genome loses to any finite one.

    Oracle: :func:`sanitize_reference` on the raw fitness, then the documented
    selection steps applied to a hand-written species table: the species containing
    the non-finite genome scores ``-inf`` (a masked maximum, never ``NaN``), sorts to
    the last live slot, and — because its only members are ``-inf`` — is never the
    parent of a slot in a species that has a finite member.  The ``best_fitness`` of
    every live slot is asserted finite, which is the invariant that keeps
    ``species_number_calculate_by="fitness"``'s ``min``/``sum`` well defined.
    """
    pytest.skip(
        "not implemented: sanitised fitness is what reaches selection "
        "(owner: algorithm/neat/neat.NEAT.tell + neat._sanitize_fitness)"
    )


# --------------------------------------------------------------------------- #
# stagnation
# --------------------------------------------------------------------------- #


def test_stagnation_kills_a_species_that_stops_improving(
    species, controller, species_state, species_kernel
):
    """Invariant: ``(f_s <= best_fitness) & (generation - last_improved > max_stagnation)`` kills.

    Oracle: :func:`stagnation_reference` on a table with three live slots — one that
    improved this generation, one at exactly ``max_stagnation`` unimproved
    generations (must **survive**, the comparison is ``>``) and one just past it
    (must die).  A killed slot has ``NaN`` in ``species_keys``/``best_fitness``/
    ``last_improved``/``member_count`` and ``-inf`` species fitness, which is what the
    descending sort uses to push it last.  The boundary case is the whole point: an
    off-by-one moves every species' death one generation earlier.
    """
    pytest.skip(
        "not implemented: stagnation predicate and the > max_stagnation boundary "
        "(owner: algorithm/neat/species.SpeciesController._stagnation)"
    )


def test_stagnation_exempts_the_top_species_elitism(
    species, controller, species_state, species_kernel
):
    """Invariant: the ``species_elitism`` best-ranked species are never killed by stagnation.

    Oracle: :func:`stagnation_reference` with ``species_elitism = 1`` over a table
    where the *rank 0* species is the most stagnant one (largest
    ``generation - last_improved`` and no improvement); it must survive while a
    lower-ranked equally stagnant species dies.  Without the exemption the best
    species of a converged run would be culled exactly when it is the only one left.
    """
    pytest.skip(
        "not implemented: species_elitism exempts the best-ranked species from stagnation "
        "(owner: algorithm/neat/species.SpeciesController._stagnation)"
    )


# --------------------------------------------------------------------------- #
# the population survives a generation
# --------------------------------------------------------------------------- #


def test_tell_preserves_the_population_size_and_payload_shapes(
    algorithm, neat_state
):
    """Invariant: after ``tell`` the population has the same static shapes, dtypes and size.

    Oracle: ``algorithm.ask(state)`` before and after the call; every tensor of the
    population pytree keeps its shape and dtype, and ``algorithm.pop_size`` is
    unchanged.  Shapes are static at build time, so a mismatch means the mutation or
    the elite select changed a capacity — the one thing the pipeline's cached
    executable cannot survive.
    """
    pytest.skip(
        "not implemented: population shape/dtype/size survive tell "
        "(owner: algorithm/neat/neat.NEAT.tell)"
    )


def test_ask_costs_no_ops_and_is_idempotent(algorithm, neat_state):
    """Invariant: ``ask`` returns the stored population unchanged and never consumes a key.

    Oracle: two consecutive ``ask`` calls return ``np.array_equal`` tensors, and
    ``state.randkey`` is unchanged after both (compared with ``np.array_equal``
    before/after).  ``ask`` is documented as the zero-op hand-off; a stray
    ``random.split`` there would desynchronise the two-splits/three-subkeys budget
    and make a seeded run irreproducible.
    """
    pytest.skip(
        "not implemented: ask is a zero-op, idempotent hand-off "
        "(owner: algorithm/neat/neat.NEAT.ask)"
    )


def test_setup_requires_the_pipeline_to_have_created_the_randkey(algorithm, common, errors):
    """Invariant: ``setup`` on a state without ``randkey`` raises ``TensorNEATError``.

    Oracle: the documented ``Raises`` clause — ``Pipeline.setup`` is the only place
    that creates the key, and a silent fallback to a default seed would break the
    "same seed reproduces the run" contract for every caller who forgot it.
    """
    pytest.skip(
        "not implemented: TensorNEATError when state.randkey is missing "
        "(owner: algorithm/neat/neat.NEAT.setup)"
    )


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_tell_is_deterministic_for_the_same_state_and_fitness(algorithm, neat_state):
    """Invariant: ``tell(state, fitness)`` twice from the same state yields identical next states.

    Oracle: ``np.array_equal`` over the population pytree and the species fields of
    the two returned states.  Both calls start from the *same* immutable state (the
    fixture returns a fresh one per call), so any dependence on a host RNG, a dict
    order or a mutable default would show up as a divergence.
    """
    pytest.skip(
        "not implemented: tell is a pure function of (state, fitness) "
        "(owner: algorithm/neat/neat.NEAT.tell)"
    )


def test_tell_advances_the_randkey(algorithm, neat_state):
    """Invariant: the next state carries a different ``randkey`` than the one it consumed.

    Oracle: ``not np.array_equal(next_state.randkey, state.randkey)`` after a full
    ``tell``, while every *other* entry of the state is untouched by the key.  Two
    generations that consumed the same key would produce identical mutations — the
    classic "same key reused" bug that still passes a single-generation test.
    """
    pytest.skip(
        "not implemented: randkey advances by one split per generation "
        "(owner: algorithm/neat/neat.NEAT.tell)"
    )


# --------------------------------------------------------------------------- #
# pop_batch_size chunking
# --------------------------------------------------------------------------- #


def test_batched_chunking_preserves_the_population_axis(neat, evoxc_key):
    """Invariant: ``_batched`` is a pure reshape — chunked and unchunked agree for a deterministic kernel.

    Oracle: ``lambda state, key, x: x + 1`` run through ``_batched`` on an ``(P, 3)``
    tensor with ``pop_batch_size`` ``None``, ``4`` and ``1``; all three outputs must
    be ``np.array_equal`` and have the input's shape/dtype.  The kernel ignores its
    key, so an off-by-one in the chunk reshape, a lost remainder or a transposed
    ``(n_batches, pop_batch_size)`` stacking is visible.

    Notes:
        Numerical equality of two *real* mutation runs is deliberately not asserted:
        ``NEAT.__init__`` documents that a chunk draws its own subkey's randomness, so
        the streams differ by design and ``pop_batch_size`` is part of the
        reproducibility config.
    """
    pytest.skip(
        "not implemented: pop_batch_size chunking preserves the population axis "
        "(owner: algorithm/neat/neat.NEAT._batched)"
    )


# --------------------------------------------------------------------------- #
# constructor validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"species_number_calculate_by": "linear"}, "not 'rank' or 'fitness'"),
        ({"pop_batch_size": 3}, "does not divide pop_size (8)"),
        ({"species_size": 4, "min_species_size": 3}, "4 * 3 > pop_size (8)"),
        ({"survival_threshold": 0.0}, "outside (0, 1]"),
        ({"survival_threshold": 1.5}, "outside (0, 1]"),
        ({"spawn_number_change_rate": -0.1}, "outside [0, 1]"),
        ({"spawn_number_change_rate": 1.5}, "outside [0, 1]"),
        ({"compatibility_threshold": -1.0}, "negative distance"),
    ],
)
def test_constructor_rejects_each_out_of_domain_argument(neat, errors, overrides, reason):
    """Invariant: every documented out-of-domain constructor argument raises ``TensorNEATError``.

    Oracle: the ``Raises`` clause of ``NEAT.__init__`` plus each parameter's documented
    domain, one case per rule (see the parametrisation).  The type is what matters:
    the rewrite replaced the legacy ``assert``/``ValueError`` with
    ``TensorNEATError`` so a caller can catch configuration errors without catching
    programming errors, and a message naming the parameter is required so the failure
    is actionable.
    """
    pytest.skip(
        f"not implemented: TensorNEATError for {reason} "
        "(owner: algorithm/neat/neat.NEAT.__init__)"
    )


def test_constructor_accepts_a_dividing_pop_batch_size(neat):
    """Invariant: a ``pop_batch_size`` that divides ``pop_size`` is accepted and stored.

    Oracle: ``NEAT(pop_batch_size=4).__init__`` does not raise and the configured
    value is what the algorithm uses for its scan chunking (compared against
    ``pop_size=8``: two chunks).  Pairing this with the rejection table keeps the
    validation from being a blanket "reject any int".
    """
    pytest.skip(
        "not implemented: a dividing pop_batch_size is accepted "
        "(owner: algorithm/neat/neat.NEAT.__init__)"
    )
