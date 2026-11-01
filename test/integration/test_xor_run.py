"""End-to-end XOR evolution through the shipped ``Pipeline`` (marker ``slow``).

What this module proves
-----------------------
Not that a kernel computes something, but that the *assembled* library does what
``README``/``examples/func_fit/xor.py`` promises: a fresh
``Pipeline(NEAT(DefaultGenome(2, 1, ...), pop_size=16), XOR(), seed=0,
fitness_target=-0.01, generation_limit=60, backend="numpy")`` goes from
``setup()`` through ``auto_run()`` to a population whose best network solves XOR,
returns ``(state, best_genome)``, and reports a best genome that really is the
one that scored.

The three assertions, and the oracles behind them
-------------------------------------------------
* **The target is reached inside the budget.** ``XOR`` fitness is the negated
  ``mse`` over its four samples and single output, so ``fitness_target=-0.01``
  means "mean squared error <= 0.01" against the literal targets
  ``[[0], [1], [1], [0]]`` — a closed form a reader can check by hand, not a
  number copied out of a log. The run is required to stop *on its own*
  (``state.generation <= generation_limit``), so a pipeline that merely ran out of
  generations fails here.
* **The return value is the documented pair.** ``auto_run`` returns
  ``(state, best_genome)``; the state still exposes the algorithm-registered
  tensors (``pop_nodes``/``pop_conns``/``generation``) and the best genome's row
  counts are the static shape signature (``max_nodes`` rows, ``max_conns`` rows),
  which is what makes it feedable back into ``algorithm.transform``.
* **The reported best is the evaluated best.** Re-scoring the returned genome
  through ``algorithm.transform`` → ``algorithm.forward`` → ``problem.evaluate``
  must reproduce the run's best recorded fitness. A stale ``best_genome`` (one
  generation behind, or the best of a previous shape) shows up as a mismatch; a
  logged number that was never recomputed does not.

Budget (measured, not asserted)
-------------------------------
``pop_size=16`` × ``generation_limit=60`` on the numpy reference interpreter: the
whole module is budgeted to run in ~10–20 s, i.e. one compile per shape plus 60
generation steps of a 16-individual population. The measured wall time is to be
recorded here in Phase 2, once ``pipeline.py`` exists; this suite never asserts
on wall-clock time (``test/CONTEXT.md``), only on generations and fitness.

Why nothing here is a module-scope import
-----------------------------------------
``import tensorneat`` still raises ``NotImplementedError`` while ``src/`` is a
design skeleton, so every input arrives as a fixture resolved through
``support.lazy.require`` (``./conftest.py``) and the stub body reports the owner.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow]


def test_auto_run_reaches_the_xor_fitness_target(tiny_pipeline, run_pipeline):
    """Invariant: ``auto_run`` solves XOR within the generation budget and stops early by itself.

    Oracle: the closed form of ``XOR``'s ``mse`` — four samples
    ``[[0,0], [0,1], [1,0], [1,1]]`` against ``[[0], [1], [1], [0]]``, reduced
    over sample *and* output axes and negated into a fitness, so "solved" is
    ``fitness >= -0.01``, i.e. ``mse <= 0.01`` (RMS error ``<= 0.1``). Nothing
    here reads a number back from the pipeline to define the bar.

    Assertion: on the default tiny configuration (``pop_size=16``,
    ``generation_limit=60``, ``seed=0``, ``fitness_target=-0.01``) the run
    terminates with ``max(history max) >= -0.01`` **and**
    ``0 < int(state.generation) <= 60``, and ``len(history) == int(state.generation)``
    — one logged line per generation, so the loop count and the trajectory agree.
    A run that hit the generation limit without solving XOR fails the first
    clause; a run that "reached the target" without advancing a generation fails
    the second.

    Notes:
        The default tiny configuration applies the ``sigmoid`` output transform
        (``./conftest.py``), so the read-out is bounded in ``(0, 1)`` like the
        shipped XOR example; a raw (unbounded) output can, in principle, reach
        ``mse <= 0.01`` too, but not for a population of 16 within 60
        generations, and the test would then be flaky rather than strict.

    Owner: ``pipeline.Pipeline.auto_run`` / ``pipeline.Pipeline.step``.
    """
    pytest.skip(
        "not implemented: auto_run evolves XOR to fitness >= -0.01 inside the "
        "generation budget (owner: pipeline.Pipeline.auto_run, pipeline.Pipeline.step)"
    )


def test_auto_run_returns_the_documented_pair_with_a_payload_shaped_best(
    tiny_pipeline, run_pipeline
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
    ``len(history) == 1``; ``state.pop_nodes``/``state.pop_conns`` still exist on
    the final state (the algorithm registers them, ``src/tensorneat/algorithm/CONTEXT.md``)
    and their population axis is ``pop_size == 16``; the best genome is a payload
    pytree whose node and connection tensors have ``max_nodes``/``max_conns`` rows,
    i.e. it is directly usable as a one-individual batch of the same signature.

    Notes:
        This is the test that catches "the loop returned the population instead of
        the best" and "the best was flattened to a bare pair of numpy arrays".

    Owner: ``pipeline.Pipeline.auto_run``, ``pipeline.Pipeline.setup``.
    """
    pytest.skip(
        "not implemented: auto_run returns (state, best_genome) with the configured "
        "payload shapes (owner: pipeline.Pipeline.auto_run, genome.utils.GenomeTensors)"
    )


def test_reported_best_genome_reproduces_its_fitness(
    tiny_pipeline, run_pipeline, evaluate_population
):
    """Invariant: the returned best genome, scored again, reproduces the run's best fitness.

    Oracle: the run's own trajectory — ``max`` over ``history``'s per-generation
    maxima, which is the number the pipeline logged as the best of the run — and
    the documented kernel triple, re-executed on the host by
    ``./conftest.evaluate_population``: ``algorithm.transform`` →
    ``algorithm.forward`` (partial-applied into the problem's ``act_fn``) →
    ``problem.evaluate`` on a **one-individual** batch of the returned genome.

    Assertion: the recomputed fitness of the returned best genome equals the run's
    recorded best within float32 tolerance (``rtol=1e-5, atol=1e-6``) and is
    finite (``np.isfinite``), so the reported genome is the one that produced the
    reported number and not an ancestor, a different individual, or a payload of
    the *previous* shape signature. Re-scoring a genome that the pipeline never
    selected is exactly the failure a "best_fitness" attribute alone cannot see.

    Notes:
        The comparison is against the trajectory, not against
        ``pipeline.best_fitness``, because the attribute is an implementation
        detail while "the best of the generations it logged" is the documented
        outcome. If a run breaks out on the fitness target, the best genome is the
        last generation's best, whose value is the trajectory's maximum — the same
        identity holds for a run that hits the generation limit.

    Owner: ``pipeline.Pipeline.auto_run`` (stale best) and
    ``problem.func_fit.func_fit.FuncFit.evaluate`` (the re-scoring oracle).
    """
    pytest.skip(
        "not implemented: the reported best genome re-scores to the run's best "
        "fitness (owner: pipeline.Pipeline.auto_run, "
        "problem.func_fit.func_fit.FuncFit.evaluate)"
    )
