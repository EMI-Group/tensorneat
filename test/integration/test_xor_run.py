"""End-to-end XOR evolution through the shipped ``Pipeline`` (marker ``slow``).

What this module proves
-----------------------
Not that a kernel computes something, but that the *assembled* library does what ``README``/``examples/func_fit/xor.py`` promises: a fresh ``Pipeline(NEAT(DefaultGenome(2, 1, ...), pop_size=16), XOR(), seed=0, fitness_target=-0.01, generation_limit=60, backend="numpy")`` goes from ``setup()`` through ``step()``/``auto_run()`` to a population whose best network solves XOR.
It also pins the two halves of ``auto_run``'s contract separately — the documented ``(state, best_genome)`` pair with its static payload shapes, and the claim that the returned best genome is the individual that produced the reported fitness — plus the generation bookkeeping of ``step`` and the elitism invariant the whole search rests on.

The five assertions and their oracles
-------------------------------------
* **The target is reached inside the budget.** ``XOR`` fitness is the negated ``mse`` over its four samples and single output, so ``fitness_target=-0.05`` means "mean squared error <= 0.05" against the literal targets ``[[0], [1], [1], [0]]`` — a closed form a reader can check by hand, not a number copied out of a log.
  The real oracle is the returned best genome re-scored through the documented kernel triple (``algorithm.transform`` → replicated ``algorithm.forward`` → ``problem.evaluate``), never ``pipeline.best_fitness``: a logged number that was never recomputed cannot satisfy this test.
* **The return value is the documented pair with the static signature.** ``auto_run`` returns ``(state, best_genome)``; the state still exposes the algorithm-registered payload ``state.pop`` (a ``genome.utils.GenomeTensors``) with the population on axis 0, and the best genome's rows are the configured static signature (``max_nodes`` node rows, ``max_conns`` connection rows), which is what makes it feedable as a one-individual batch of the same shapes.
* **The reported best is the evaluated best.** Re-scoring the returned genome must reproduce ``max`` over a manually stepped trajectory of the same seed, so a stale ``best_genome`` (one generation behind, or a payload of the previous shape signature) shows up as a mismatch.
* **``step`` advances exactly one generation and reports one fitness per individual.** ``state.generation`` goes 0 → 1, the returned fitness is a ``(pop_size,)`` array of finite-or-``-inf`` values, and it agrees with the host kernel triple re-run on the population the step consumed — the pipeline's own evaluation must match the documented kernels element for element.
* **The population maximum never drops.** ``NEAT``'s species elitism copies the best member unmutated, so the per-generation maximum read off ``pipeline.step`` is non-decreasing; a drop means the elite slot is not being preserved.

Budgets (measured, never asserted)
----------------------------------
:func:`test_auto_run_reaches_the_xor_fitness_target` is the only heavy test: ``pop_size=100`` × ``generation_limit=250``, calibrated from the legacy JAX curve in ``benchmarks/RESULTS.md`` (``mse <= 0.1`` at generation 77 and ``mse ≈ 0.0099`` by generation 150 at ``pop_size=100``).
That curve is an order-of-magnitude reference, because the evoxc rewrite draws different random numbers; the budget must be re-measured — never the assertion weakened — once ``pipeline.py`` lands.
Everything else runs at ``pop_size=16`` for 1–10 generations: one generation for the shape tests, five for the manual-vs-``auto_run`` cross-check, ten for the elitism check.
No test reads a clock (``test/CONTEXT.md``: a flaky timing test is a bug in the test); a run is judged by generations, fitness and payload shapes.

Why this file has no skip of its own
------------------------------------
``src/tensorneat/pipeline.py`` is the legacy JAX module until the evoxc compile site lands, so ``./conftest.py``'s autouse probe (:func:`~.conftest.require_evoxc_pipeline`) skips every test here with the exact list of missing evoxc features.
Nothing in this module hardcodes that skip, so the tests start running — unmodified — the moment the evoxc compile site lands.
``tensorneat`` is still never imported at module scope: the package arrives through the ``tn`` fixture, exactly as in every other module of this directory.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import TINY_CONFIG, step_loop, step_once

pytestmark = [pytest.mark.slow]

#: The static shape signature of :data:`TINY_CONFIG` (:func:`tiny_pipeline`'s defaults).
#:
#: Every pipeline built without an override carries these widths, so a payload read
#: back from it must show them padded to the static signature.
POP_SIZE = TINY_CONFIG["pop_size"]  # 16
MAX_NODES = TINY_CONFIG["max_nodes"]  # 8
MAX_CONNS = TINY_CONFIG["max_conns"]  # 16

#: The four field names of ``genome.utils.GenomeTensors``, in payload order.
#:
#: A "GenomeTensors-shaped" payload is one whose ``_fields`` equal this tuple; the
#: row invariants below are asserted on the same four arrays.
GENOME_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: A configuration whose run always consumes the whole generation budget.
#:
#: ``XOR`` fitness is ``-mse <= 0``, so ``fitness_target=1.0`` is unreachable and
#: ``auto_run``/``step_loop`` advance exactly ``generation_limit`` generations.
FIXED_LENGTH = {"fitness_target": 1.0}


def _generation(tn, state) -> int:
    """Read ``state.generation`` — a 0-d float32 evoxc tensor — as a Python int.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture).
        state: Any ``State`` the pipeline returned.

    Returns:
        The generation counter as a Python ``int``, through the package's documented
        host-side accessor (``evoxc.Tensor`` has no ``__array__``, so a bare
        ``int(state.generation)`` raises instead of unwrapping).
    """
    return int(tn.common.tools.to_numpy(state.generation))


def _payload_shapes(payload) -> tuple[tuple[int, ...], ...]:
    """Return the host-visible shapes of a four-leaf gene payload, in payload order.

    Args:
        payload: A ``genome.utils.GenomeTensors`` (or any 4-field namedtuple whose
            leaves are ``evoxc.Tensor``/``numpy.ndarray``).

    Returns:
        The four ``tuple`` shapes, so a test can compare against the configured
        static signature without reaching into ``evoxc``.

    Raises:
        AssertionError: If ``payload`` does not carry exactly the four documented
            gene arrays; guessing an order would assert on the wrong arrays.
    """
    assert getattr(payload, "_fields", None) == GENOME_FIELDS, (
        "expected a GenomeTensors-shaped payload with fields "
        f"{GENOME_FIELDS}, got {type(payload).__name__} with fields "
        f"{getattr(payload, '_fields', None)!r}"
    )
    return tuple(tuple(int(size) for size in leaf.shape) for leaf in payload)


def test_auto_run_reaches_the_xor_fitness_target(
    tn, tiny_pipeline, run_pipeline, evaluate_population
):
    """Invariant: ``auto_run`` solves XOR within the generation budget and stops by itself.

    Oracle: the closed form of ``XOR``'s ``mse`` — four samples
    ``[[0,0], [0,1], [1,0], [1,1]]`` against ``[[0], [1], [1], [0]]``, reduced
    over sample *and* output axes and negated into a fitness — re-executed on the
    returned best genome by ``./conftest.evaluate_population`` (``algorithm.transform``
    → replicated ``algorithm.forward`` → ``problem.evaluate``).

    Assertion: on ``pop_size=100``, ``species_size=10``, ``generation_limit=250``,
    ``seed=0``, ``fitness_target=-0.05``, ``auto_run`` returns with
    ``1 <= int(state.generation) <= 250`` **and** the returned best genome
    re-scores to ``>= -0.05`` (``mse <= 0.05``, RMS error ``<= ~0.224``). A run
    that exhausted the budget without solving XOR fails the fitness clause; a run
    that "reached the target" without advancing a generation fails the other.
    Neither clause is conditional and neither reads ``pipeline.best_fitness``:
    the genome is re-scored from scratch.

    Notes:
        ``species_size * min_species_size <= pop_size`` holds for this
        configuration (``10 * 1 <= 100``, ``min_species_size`` defaulting to ``1``),
        which is the invariant ``NEAT`` validates before it builds anything.
        The budget is calibrated from ``benchmarks/RESULTS.md``: the frozen legacy
        JAX implementation reached ``mse <= 0.1`` at generation 77 and
        ``mse ≈ 0.0099`` by generation 150 at ``pop_size=100``, so 250 generations
        at the looser ``mse <= 0.05`` is an order-of-magnitude hedge, not a
        guarantee for this seed — the evoxc rewrite draws different random numbers.
        Re-measure the budget when ``pipeline.py`` lands; do not weaken the
        assertion to make it pass.

    Owner: ``pipeline.Pipeline.auto_run`` / ``pipeline.Pipeline.step``,
    ``problem.func_fit.func_fit.FuncFit.evaluate`` (the re-scoring oracle).
    """
    pipeline = tiny_pipeline(
        pop_size=100,
        species_size=10,
        generation_limit=250,
        fitness_target=-0.05,
    )
    state, best = run_pipeline(pipeline)

    generation = _generation(tn, state)
    assert 1 <= generation <= 250, (
        "auto_run must advance at least one generation and stop within the "
        f"250-generation budget; state.generation is {generation}"
    )

    fitness = evaluate_population(pipeline, state, best)
    assert fitness.shape == (1,), (
        "re-scoring the reported best genome must yield one fitness value, got "
        f"shape {fitness.shape}"
    )
    assert np.all(np.isfinite(fitness)), (
        f"the reported best genome re-scores to a non-finite fitness {fitness!r}; "
        "a run that reached the target cannot have a NaN/-inf best genome"
    )
    assert float(fitness[0]) >= -0.05, (
        f"auto_run stopped at generation {generation} but the returned best "
        f"genome re-scores to {float(fitness[0])}, below the -0.05 target "
        "(mse <= 0.05): the run did not solve XOR within the budget"
    )


def test_auto_run_returns_the_documented_pair_with_a_payload_shaped_best(
    tn, tiny_pipeline
):
    """Invariant: ``auto_run`` returns ``(state, best_genome)`` and the best carries the static shapes.

    Oracle: the configured shape signature — ``Pipeline`` documents that shapes
    are static at build time, so the reported best genome must have exactly
    ``max_nodes`` node rows and ``max_conns`` connection rows, whatever the
    optimum turned out to be. With ``generation_limit=1`` and an unreachable
    target (``fitness_target=1.0``; ``mse >= 0`` makes any fitness ``<= 0``) the
    run is required to consume exactly one generation, which pins the loop's
    boundary instead of its luck.

    Assertion: ``auto_run`` returns a 2-tuple; ``int(state.generation) == 1``;
    ``pipeline.pop_size == 16``; ``state.pop`` is still a ``GenomeTensors``-shaped
    payload whose node rows are ``(16, 8)``; and the returned best is a 4-field
    ``GenomeTensors``-shaped payload with a one-individual population axis —
    ``node_keys`` of shape ``(1, 8)``, ``conn_keys`` of shape ``(1, 16, ...)`` —
    i.e. directly usable as a one-individual batch of the same signature.

    Notes:
        ``auto_run`` is called directly rather than through ``run_pipeline``: the
        helper folds whatever ``auto_run`` returned into a 2-tuple itself, so it
        could not catch "the loop returned the population instead of the best" or
        "the best was flattened to a bare pair of numpy arrays". The four leaves
        are compared through their field names, so a payload rebuilt in a
        different order fails here instead of scoring the wrong arrays later.

    Owner: ``pipeline.Pipeline.auto_run``, ``pipeline.Pipeline.setup``,
    ``genome.utils.GenomeTensors``.
    """
    pipeline = tiny_pipeline(generation_limit=1, **FIXED_LENGTH)
    result = pipeline.auto_run(pipeline.setup())

    assert isinstance(result, tuple) and len(result) == 2, (
        "auto_run must return the documented (state, best_genome) 2-tuple, got "
        f"{type(result).__name__} of length {len(result) if hasattr(result, '__len__') else '?'}"
    )
    state, best = result

    assert _generation(tn, state) == 1, (
        "one generation was requested and the target is unreachable, so "
        f"state.generation must be 1, got {_generation(tn, state)}"
    )
    assert pipeline.pop_size == POP_SIZE, (
        f"pipeline.pop_size must be the configured {POP_SIZE}, got "
        f"{pipeline.pop_size!r}"
    )

    pop_shapes = _payload_shapes(state.pop)
    assert pop_shapes[0] == (POP_SIZE, MAX_NODES), (
        "the final state's node payload must keep the population on axis 0 of the "
        f"static ({POP_SIZE}, {MAX_NODES}) signature, got {pop_shapes[0]}"
    )

    best_shapes = _payload_shapes(best)
    assert best_shapes[0] == (1, MAX_NODES), (
        "the reported best genome must be a one-individual batch of "
        f"max_nodes={MAX_NODES} node rows, got node_keys {best_shapes[0]}"
    )
    assert best_shapes[2][:2] == (1, MAX_CONNS), (
        "the reported best genome must be a one-individual batch of "
        f"max_conns={MAX_CONNS} connection rows, got conn_keys {best_shapes[2]}"
    )
    assert best_shapes[1][:2] == (1, MAX_NODES) and best_shapes[3][:2] == (1, MAX_CONNS), (
        "the attribute arrays must be padded to the same static rows as the key "
        f"arrays, got node_attrs {best_shapes[1]} and conn_attrs {best_shapes[3]}"
    )


def test_reported_best_genome_reproduces_its_fitness(
    tn, tiny_pipeline, run_pipeline, evaluate_population
):
    """Invariant: the returned best genome, scored again, reproduces the run's best fitness.

    Oracle: a *manual* run of the same seed — ``step_loop`` drives the documented
    ``pipeline.step`` for exactly five generations with ``target=None`` (so no
    early stop can make the two runs different lengths), and ``max(history)`` is
    the best fitness that run ever observed. The second oracle is the documented
    kernel triple, re-executed on the host by ``./conftest.evaluate_population``:
    ``algorithm.transform`` → ``algorithm.forward`` (partial-applied into the
    problem's ``act_fn``) → ``problem.evaluate`` on a **one-individual** batch of
    the genome ``auto_run`` returned.

    Assertion: after ``auto_run`` on a fresh pipeline of the same configuration
    reaches generation 5, the recomputed fitness of the returned best genome
    equals ``max(history)`` within float32 tolerance (``rtol=1e-5, atol=1e-6``)
    and is finite. Re-scoring a genome the pipeline never selected — an ancestor,
    a different individual, or a payload of the previous shape signature — is
    exactly the failure a ``best_fitness`` attribute alone cannot see.

    Notes:
        The comparison is against the manually stepped trajectory, not against
        ``pipeline.best_fitness``, because the attribute is an implementation
        detail while "the best of the generations it stepped" is the documented
        outcome. ``NEAT`` elitism keeps the population maximum non-decreasing, so
        the last generation's maximum is the trajectory's maximum whether the run
        stopped on the target or on the generation limit.

    Owner: ``pipeline.Pipeline.auto_run`` (stale best),
    ``problem.func_fit.func_fit.FuncFit.evaluate`` (the re-scoring oracle).
    """
    config = {"generation_limit": 5, **FIXED_LENGTH}

    stepped = tiny_pipeline(**config)
    _, history = step_loop(stepped, stepped.setup(), generations=5, target=None)
    assert len(history) == 5, (
        "step_loop must consume the five requested generations on an unreachable "
        f"target, got {len(history)}"
    )
    recorded_best = max(history)
    assert np.isfinite(recorded_best), (
        f"no generation of the manual run produced a valid genome (best {recorded_best!r})"
    )

    pipeline = tiny_pipeline(**config)
    state, best = run_pipeline(pipeline)
    assert _generation(tn, state) == 5, (
        "the auto_run side must consume the same five generations as the manual "
        f"run, got generation {_generation(tn, state)}"
    )

    recomputed = evaluate_population(pipeline, state, best)
    assert recomputed.shape == (1,), (
        "re-scoring the reported best genome must yield one fitness value, got "
        f"shape {recomputed.shape}"
    )
    assert np.all(np.isfinite(recomputed)), (
        "the reported best genome must re-score to a finite fitness, got "
        f"{recomputed!r}"
    )
    np.testing.assert_allclose(
        recomputed[0],
        recorded_best,
        rtol=1e-5,
        atol=1e-6,
        err_msg=(
            "the best genome auto_run returned does not re-score to the best "
            f"fitness the same seed reached when stepped manually "
            f"(recomputed {float(recomputed[0])}, trajectory max {recorded_best})"
        ),
    )


def test_step_advances_one_generation_and_reports_one_fitness_per_individual(
    tn, tiny_pipeline, evaluate_population
):
    """Invariant: one ``step`` is one generation and one fitness value per individual.

    Oracle: the documented ``pipeline.step`` contract (``[0]`` is the new
    ``State``, ``[-1]`` is a length-``pop_size`` fitness array) and the host
    kernel triple — ``./conftest.evaluate_population`` re-scores the *population
    the step consumed* (``state_before.pop``), so the pipeline's own evaluation is
    compared element for element against ``algorithm.transform`` →
    ``algorithm.forward`` → ``problem.evaluate``.

    Assertion: ``setup()`` yields generation 0; after one ``step_once`` the
    returned fitness has shape ``(16,)`` and contains only finite or ``-inf``
    values (``finalize_fitness`` maps NaN to ``-inf``); ``state.generation`` is 1;
    and that fitness vector equals the host recomputation within float32
    tolerance (``rtol=1e-5, atol=1e-6``).

    Notes:
        This is the cheap structural test of the generation algebra: no
        convergence is claimed, only that a step advances the counter by exactly
        one and scores the individuals it was handed. ``fitness_target=1.0`` and
        ``generation_limit=1`` are passed so the configuration states "one
        generation, no early stop" even though ``step`` never consults either.

    Owner: ``pipeline.Pipeline.step``, ``algorithm.neat.neat.NEAT.tell``.
    """
    pipeline = tiny_pipeline(generation_limit=1, **FIXED_LENGTH)
    state_before = pipeline.setup()
    assert _generation(tn, state_before) == 0, (
        "setup() must start the run at generation 0, got "
        f"{_generation(tn, state_before)}"
    )

    state, fitness = step_once(pipeline, state_before)

    assert fitness.shape == (POP_SIZE,), (
        f"step must report one fitness per individual, i.e. shape ({POP_SIZE},), "
        f"got {fitness.shape}"
    )
    assert np.all(np.isfinite(fitness) | np.isneginf(fitness)), (
        "step's fitness array must be finite or -inf (NaN is finalized to -inf by "
        f"the problem), got {fitness!r}"
    )
    assert _generation(tn, state) == 1, (
        f"one step must advance exactly one generation, got {_generation(tn, state)}"
    )

    recomputed = evaluate_population(pipeline, state_before, state_before.pop)
    assert recomputed.shape == fitness.shape, (
        "step's fitness and the host re-scoring must describe the same population: "
        f"{fitness.shape} vs {recomputed.shape}"
    )
    np.testing.assert_allclose(
        fitness,
        recomputed,
        rtol=1e-5,
        atol=1e-6,
        err_msg=(
            "pipeline.step's fitness disagrees with the documented kernel triple "
            "re-run on the population it consumed"
        ),
    )


def test_the_best_fitness_is_non_decreasing_across_generations(tiny_pipeline):
    """Invariant: ``NEAT`` elitism keeps the per-generation population maximum from dropping.

    Oracle: the trajectory itself — ``./conftest.step_loop`` records
    ``max(pipeline.step(...)[-1])`` per generation, which is the same comparison
    ``auto_run`` uses to decide whether the target was reached.

    Assertion: over ten generations at the tiny configuration
    (``generation_limit=10``, ``fitness_target=1.0`` so the budget is always
    consumed) no generation's maximum is below the previous one, and
    ``max(history) >= history[0]``. A drop means the elite slot is not being
    preserved; the failure message says so instead of only quoting the numbers.
    Every entry must also be non-NaN, since a NaN would make the ``<`` comparison
    silently false and let a broken run pass.

    Notes:
        Ten generations, not one: a single comparison cannot observe a drop, and
        the invariant is a property of the whole loop. No convergence is claimed —
        the fitness target is unreachable on purpose, so the run always consumes
        exactly the ten generations.

    Owner: ``algorithm.neat.neat.NEAT.tell`` (the elite copy),
    ``algorithm.species`` (``species_elitism``/``genome_elitism``).
    """
    generations = 10
    pipeline = tiny_pipeline(generation_limit=generations, **FIXED_LENGTH)
    _, history = step_loop(pipeline, pipeline.setup())

    assert len(history) == generations, (
        f"a {generations}-generation budget on an unreachable target must be "
        f"consumed, got {len(history)} generation(s)"
    )
    assert all(not np.isnan(value) for value in history), (
        f"the per-generation maxima must never be NaN, got {history!r}"
    )

    drops = [
        (generation, history[generation - 1], history[generation])
        for generation in range(1, len(history))
        if history[generation] < history[generation - 1]
    ]
    assert not drops, (
        "NEAT elitism must preserve the population maximum; a drop means the "
        "elite slot is not being preserved: "
        + ", ".join(
            f"generation {generation} fell from {previous} to {current}"
            for generation, previous, current in drops
        )
    )
    assert max(history) >= history[0], (
        "the best fitness of the run must be at least the first generation's "
        f"maximum, got max {max(history)} < first {history[0]}"
    )
