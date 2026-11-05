"""``auto_run``'s early stop reads the *sanitized* fitness, so a non-finite problem never stops it (marker ``slow``).

What this module proves
-----------------------
``Pipeline.auto_run`` stops when ``max(_finite_fitness(fitnesses)) >= fitness_target``
(``src/tensorneat/pipeline.py``:773). ``_finite_fitness`` maps ``NaN`` and ``+inf`` to
``-inf`` and leaves a finite value alone, so the only thing that can satisfy a finite
target is a finite fitness. Three properties follow and are pinned here with a
problem chosen to make the sanitization observable:

* **``+inf`` never satisfies the target.** A naive ``max(fitnesses) >= target`` would
  stop after generation 1 (``inf >= 0.0`` is ``True``); the sanitized comparison sees
  ``-inf`` and consumes the whole ``generation_limit``. The test asserts *both*
  halves: the run advanced to generation 3, and the raw ``(P,)`` fitness of a
  one-step pipeline with the same configuration is ``+inf`` at every entry.
* **``NaN`` never satisfies the target** for the same reason (and is not merely
  ``>=``-false by accident: the raw vector is asserted ``all-NaN``).
* **``-inf`` is the control**: it is the only non-finite value selection ever sees,
  and it must not stop the run either.

Because an all-invalid generation has no ``Max`` to report, ``fitness_history``
records ``(generation, nan)`` per generation (``_analyse`` writes ``/``nan`` when no
finite value survives), ``best_fitness`` stays ``-inf`` and ``best_genome`` stays
``None`` — the documented "no generation produced a finite fitness" outcome.

The fourth case is the opposite control: a **finite** constant fitness ``0.5`` with
``fitness_target=0.0`` stops after exactly one generation, so "the sanitized path
runs to the limit" is not the vacuous statement "``auto_run`` never stops".

The oracle
----------
The problem is the test's own: ``evaluate`` returns a constant ``(P,)`` vector, so
every assertion compares the pipeline's bookkeeping against a value the test chose
and against the closed form of ``_finite_fitness`` — never against a recorded run.
``P`` comes from ``tensorneat.problem.base.static_batch_size(transformed)``, the one
thing a problem may read off the schedule.

Budgets (measured, never asserted)
----------------------------------
``pop_size=8``, ``max_nodes=8``, ``generation_limit=3``, one compile per pipeline;
``is_detailed=False`` keeps every run silent and nothing here reads a clock.

Why this file has no ``tensorneat`` import at module scope
---------------------------------------------------------
The package arrives through the ``tn`` fixture, and ``evoxc``/``static_batch_size``
are imported inside ``evaluate`` (which runs in a trace) or inside a test. The
directory-wide autouse probe (:func:`~.conftest.require_evoxc_pipeline`) is the only
guard.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from .conftest import TINY_CONFIG

pytestmark = [pytest.mark.slow]

#: The genome knobs every pipeline in this module shares, taken from the suite's one
#: configuration. ``num_inputs=2``/``num_outputs=1`` is what the constant problem
#: declares below, so the pipeline's own arity check passes.
GENOME_KNOBS = {
    name: TINY_CONFIG[name]
    for name in (
        "num_inputs",
        "num_outputs",
        "max_nodes",
        "max_conns",
        "max_in_degree",
        "max_depth",
    )
}

#: The constant problem's per-sample input/output shapes; they must match the
#: genome's ``num_inputs``/``num_outputs`` because ``Pipeline._validate_arity`` checks it.
INPUT_SHAPE = (2,)
OUTPUT_SHAPE = (1,)

#: The population of every pipeline here.
POP_SIZE = 8

#: ``species_size * min_species_size <= pop_size`` must hold: ``2 * 1 <= 8``.
SPECIES_SIZE = 2

#: One seed, so the raw-value witness and the run describe the same start.
SEED = 0

#: The generation budget a non-finite fitness must consume in full.
GENERATION_LIMIT = 3

#: A target no non-finite fitness may reach: ``_finite_fitness`` maps ``NaN``/``+inf``
#: to ``-inf``, which is ``< 0.0``.
FITNESS_TARGET = 0.0

#: The constant a finite control run reports.
FINITE_FITNESS = 0.5


def _constant_problem_class(tn):
    """Build a ``BaseProblem`` whose ``evaluate`` returns one constant fitness value.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture); its
            ``problem.BaseProblem`` is the protocol every problem implements.

    Returns:
        A subclass whose ``__init__(value)`` stores the constant, and whose
        ``evaluate`` returns ``full((P,), value, dtype=float32)``.

    Notes:
        ``P`` is recovered with ``static_batch_size(transformed)`` — inside a trace
        the leading dimension is a plain ``int``, so the problem needs no other
        introspection of the schedule. The problem deliberately ignores ``act_fn``:
        this module tests the *stopping rule*, not the network, and a constant
        vector is the cleanest way to make a non-finite fitness reach ``auto_run``.
        ``show``/``show_details`` keep the base implementations (``show`` is never
        called here, ``show_details`` is a no-op).
    """

    class ConstantFitnessProblem(tn.problem.BaseProblem):
        """Every individual scores the same constant; the value may be non-finite."""

        def __init__(self, value: float) -> None:
            self.value = float(value)

        @property
        def input_shape(self) -> tuple[int, ...]:
            """Per-sample input shape, matching the genome's ``num_inputs``."""
            return INPUT_SHAPE

        @property
        def output_shape(self) -> tuple[int, ...]:
            """Per-sample output shape, matching the genome's ``num_outputs``."""
            return OUTPUT_SHAPE

        def evaluate(self, state, keys, act_fn, transformed):
            """Return the constant ``(P,)`` fitness vector of the whole population.

            Args:
                state: The host container (unused: nothing is registered).
                keys: One evoxc key (unused: the problem is deterministic).
                act_fn: The network policy (unused by construction).
                transformed: The batch-first schedule, read only for its static ``P``.

            Returns:
                ``(P,)`` float32 fitness, every entry ``self.value``.
            """
            import evoxc
            import evoxc.numpy as enp
            from tensorneat.problem.base import static_batch_size

            batch = static_batch_size(transformed)
            return enp.full((batch,), self.value, dtype=evoxc.float32)

    return ConstantFitnessProblem


def _matches(value: float, fitness: np.ndarray) -> bool:
    """Whether every entry of ``fitness`` is ``value``, infinity-sign aware.

    Args:
        value: The constant the problem was built with (``+inf``, ``-inf``, ``NaN``
            or a finite number).
        fitness: The ``(P,)`` vector ``Pipeline.step`` returned, as numpy.

    Returns:
        ``True`` when every entry matches ``value`` exactly; ``NaN`` only matches
        ``NaN`` and each infinity only its own sign.
    """
    if math.isnan(value):
        return bool(np.all(np.isnan(fitness)))
    if value == math.inf:
        return bool(np.all(np.isposinf(fitness)))
    if value == -math.inf:
        return bool(np.all(np.isneginf(fitness)))
    return bool(np.all(fitness == np.float32(value)))


@pytest.fixture(scope="module")
def stopping_pipeline(tn):
    """Factory for a ``NEAT`` + constant-fitness ``Pipeline`` of :data:`POP_SIZE`.

    Returns:
        ``make(value, *, generation_limit=GENERATION_LIMIT,
        fitness_target=FITNESS_TARGET) -> Pipeline``: ``DefaultGenome`` with
        :data:`GENOME_KNOBS`, ``NEAT(pop_size=8, species_size=2)``, the constant
        problem built for ``value``, and a silent numpy-backend pipeline at
        :data:`SEED`. Nothing is run: the caller calls ``setup()`` and drives
        ``step``/``auto_run`` itself.

    Notes:
        The genome's read-out is left as the ``DefaultGenome`` default: the
        constant problem never calls ``act_fn``, so the network's values cannot
        influence the fitness and the module stays about the stopping rule.
    """

    def make(value, *, generation_limit=GENERATION_LIMIT, fitness_target=FITNESS_TARGET):
        genome = tn.genome.DefaultGenome(**GENOME_KNOBS)
        algorithm = tn.algorithm.NEAT(
            genome, pop_size=POP_SIZE, species_size=SPECIES_SIZE
        )
        problem = _constant_problem_class(tn)(value)
        return tn.Pipeline(
            algorithm,
            problem,
            seed=SEED,
            fitness_target=fitness_target,
            generation_limit=generation_limit,
            backend="numpy",
            device=None,
            compile_options=None,
            is_save=False,
            save_dir=None,
            is_detailed=False,
            show_problem_details=False,
        )

    return make


def _generation(tn, state) -> int:
    """Read ``state.generation`` (a 0-d float32 evoxc tensor) as a Python ``int``.

    Args:
        tn: The imported ``tensorneat`` package, for ``common.tools.to_numpy``.
        state: Any ``State`` the pipeline returned.

    Returns:
        The generation counter.
    """
    return int(tn.common.tools.to_numpy(state.generation))


@pytest.mark.parametrize(
    "value, label, naive_would_stop",
    [
        (math.inf, "positive infinity", True),
        (math.nan, "NaN", False),
        (-math.inf, "negative infinity", False),
    ],
    ids=["posinf", "nan", "neginf"],
)
def test_a_non_finite_fitness_never_satisfies_the_target(
    tn, stopping_pipeline, value, label, naive_would_stop
):
    """``+inf``/``NaN``/``-inf`` each consume the whole budget instead of stopping the run."""
    pipeline = stopping_pipeline(value)
    state = pipeline.setup()

    state, best = pipeline.auto_run(state)

    assert _generation(tn, state) == GENERATION_LIMIT, (
        f"{label} must never satisfy fitness_target={FITNESS_TARGET}, so the run has "
        f"to consume all {GENERATION_LIMIT} generations, got "
        f"{_generation(tn, state)}"
    )
    assert pipeline.best_fitness == float("-inf"), (
        f"no {label} generation produced a finite fitness, so best_fitness must stay "
        f"-inf, got {pipeline.best_fitness!r}"
    )
    assert best is None, (
        f"auto_run must report no best genome when no generation produced a finite "
        f"fitness ({label}), got {type(best).__name__}"
    )

    history = pipeline.fitness_history
    assert [generation for generation, _ in history] == [
        1,
        2,
        GENERATION_LIMIT,
    ], f"one history entry per consumed generation, got {history!r}"
    assert all(math.isnan(max_fitness) for _, max_fitness in history), (
        f"an all-{label} generation has no valid fitness, so every recorded Max is "
        f"NaN, got {[max_fitness for _, max_fitness in history]!r}"
    )

    # The sanitization is what stopped nothing here: prove the raw vector really is
    # non-finite, so the run above was not passing merely because the problem was
    # finite all along.
    witness = stopping_pipeline(value)
    witness_state = witness.setup()
    _witness_state, _witness_pop, raw_fitness = witness.step(witness_state)
    raw = np.asarray(raw_fitness)
    assert raw.shape == (POP_SIZE,), (
        f"the witness step must report one fitness per individual, got {raw.shape}"
    )
    assert _matches(value, raw), (
        f"the constant problem must emit {label} for every individual before "
        f"auto_run sanitizes it, got {raw!r}"
    )
    if naive_would_stop:
        assert float(np.max(raw)) >= FITNESS_TARGET, (
            "the raw maximum must satisfy the target here, so a naive "
            f"max(fitnesses) >= target early stop would have stopped at generation "
            f"1 instead of {GENERATION_LIMIT}"
        )


def test_a_finite_fitness_stops_the_run_after_one_generation(tn, stopping_pipeline):
    """The control: a finite fitness above the target stops ``auto_run`` immediately.

    Without this, "a non-finite fitness never stops the run" would be consistent with
    an ``auto_run`` that never stops at all.
    """
    pipeline = stopping_pipeline(FINITE_FITNESS)
    state = pipeline.setup()

    state, best = pipeline.auto_run(state)

    assert _generation(tn, state) == 1, (
        f"a constant {FINITE_FITNESS} fitness meets fitness_target={FITNESS_TARGET} in "
        f"the first generation, so the run must stop there, got "
        f"{_generation(tn, state)} generations"
    )
    assert pipeline.best_fitness == pytest.approx(FINITE_FITNESS), (
        f"best_fitness must be the finite {FINITE_FITNESS}, got "
        f"{pipeline.best_fitness!r}"
    )
    assert best is not None, "a finite best fitness must be reported as a genome"
    assert tuple(best.node_keys.shape) == (1, GENOME_KNOBS["max_nodes"]), (
        "auto_run's best genome is a one-individual batch, so node_keys must be "
        f"(1, max_nodes)={(1, GENOME_KNOBS['max_nodes'])}, got {tuple(best.node_keys.shape)}"
    )
    assert [generation for generation, _ in pipeline.fitness_history] == [1], (
        f"exactly one generation must be recorded, got {pipeline.fitness_history!r}"
    )
    assert pipeline.fitness_history[0][1] == pytest.approx(FINITE_FITNESS), (
        "the recorded Max of the single generation must be the finite "
        f"{FINITE_FITNESS}, got {pipeline.fitness_history[0][1]!r}"
    )
