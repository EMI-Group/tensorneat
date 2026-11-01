"""Reproducibility of a whole run: ``(seed, config, shapes)`` determines every number (marker ``slow``).

What this module proves
-----------------------
``State`` is documented as the only store the ``Pipeline`` writes, and every
random draw is supposed to flow through ``evoxc.random``'s stateless keys instead
of the global ``numpy.random`` state. Together those two rules give one testable
claim: **two runs of the same ``Pipeline`` configuration with the same ``seed``
are bit-identical, and a different seed is a different run** — while *anything*
happening to the global numpy generator in between must make no difference at all.

Each test builds two independent ``Pipeline`` instances (fresh objects, fresh
``setup()``), so the comparison cannot pass on shared mutable state, and forces a
fixed-length run (``fitness_target=1.0`` is unreachable because ``mse >= 0``, so
``generation_limit=10`` is always consumed) — otherwise an early stop would make
the sequences differ in *length* for a reason unrelated to the seed.

The oracles
-----------
* The **trajectory**: ``./conftest.fitness_history``'s ``(generation, max_fitness)``
  pairs — one per generation, from the pipeline's own in-memory record or its
  ``log.txt``. Comparing whole sequences (not just the final fitness) is what
  makes a run reproducible rather than merely similarly lucky.
* The **final population**: ``state.pop_nodes``/``state.pop_conns``, which
  ``src/tensorneat/algorithm/CONTEXT.md`` documents as algorithm-registered keys
  on the shared ``State``. Payloads are compared with ``np.array_equal`` — gene
  keys are ``int32`` and gene attributes are ``float32``, so "identical run" has
  an exact meaning here and no tolerance is involved.
* **The RNG key**: ``state.randkey`` after ``setup()`` and after ``auto_run``. The
  advance is compared between two same-seed runs (equal) and against the setup
  key (different), which is the documented "the key advances, the run is
  reproducible" pair of facts.

Budget (measured, not asserted)
-------------------------------
``generation_limit=10`` at ``pop_size=16`` per run, and at most two runs per test:
the module is budgeted at a few seconds on the numpy reference interpreter. The
measured wall time is to be recorded here in Phase 2. No test reads a clock
(``test/CONTEXT.md``: a flaky determinism test is a bug in the test).

Why nothing here is a module-scope import
-----------------------------------------
``import tensorneat`` still raises ``NotImplementedError`` while ``src/`` is a
design skeleton, so every input arrives as a fixture resolved through
``support.lazy.require`` (``./conftest.py``) and the stub body reports the owner.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow]

#: A configuration whose run always consumes the whole generation budget.
#:
#: ``fitness_target=1.0`` cannot be reached (XOR fitness is ``-mse <= 0``), so
#: both runs of a pair have exactly ``generation_limit`` generations and the
#: sequences are comparable position by position.
FIXED_LENGTH = {"generation_limit": 10, "fitness_target": 1.0}


def test_the_same_seed_reproduces_the_whole_run(tiny_pipeline, run_pipeline):
    """Invariant: two fresh pipelines at ``seed=0`` give the same trajectory and final population.

    Oracle: the first run itself — the per-generation ``(generation, max_fitness)``
    sequence and the final ``state.pop_nodes``/``state.pop_conns`` payloads. No
    re-implementation is involved: the claim is equality between two runs of the
    same documented configuration, which is what "``(seed, config, shapes)``
    reproduces a run" means.

    Assertion: ``history_a == history_b`` exactly (same length, same generations,
    same float32 maxima bit for bit); ``np.array_equal`` on both final payloads;
    ``int(state_a.generation) == int(state_b.generation) == 10``. A run that
    differs in a single attribute bit fails — a tolerance here would let a real
    RNG leak through.

    Notes:
        Two *independent* ``Pipeline`` objects are built (the factory is called
        twice), so a cached executable, a module-level RNG or a class attribute
        cannot carry state from one run to the other.

    Owner: ``pipeline.Pipeline.setup`` (``evoxc.random.key(seed)``),
    ``algorithm.neat.neat.NEAT.tell`` (the per-generation key split).
    """
    pytest.skip(
        "not implemented: two same-seed runs are bit-identical in trajectory and "
        "final population (owner: pipeline.Pipeline.setup, "
        "algorithm.neat.neat.NEAT.tell)"
    )


def test_a_different_seed_is_a_different_run(tiny_pipeline, run_pipeline):
    """Invariant: ``seed=0`` and ``seed=1`` do not produce the same trajectory or the same population.

    Oracle: the same two runnables as
    :func:`test_the_same_seed_reproduces_the_whole_run`; the assertion is the
    complement, which is what catches a seed that is accepted and then ignored
    (a hard-coded key, a constant initialisation, a cached executable whose
    population was never re-drawn).

    Assertion: ``history_0 != history_1`` (the sequences differ in at least one
    generation — reported as the differing positions, because "the last fitness
    happens to coincide" must not fail the test) and at least one of the final
    payloads differs under ``np.array_equal``.

    Notes:
        The comparison is deliberately *not* "every generation differs": two
        NEAT runs may agree early (the same initial topology family) and diverge
        later, and pinning per-generation difference would make the test flaky.
        What must differ is the run, not each generation.

    Owner: ``pipeline.Pipeline.setup`` (``evoxc.random.key(seed)``),
    ``genome.base.BaseGenome.initialize``.
    """
    pytest.skip(
        "not implemented: a different seed produces a different run "
        "(owner: pipeline.Pipeline.setup, genome.base.BaseGenome.initialize)"
    )


def test_a_run_ignores_the_global_numpy_random_state(tiny_pipeline, run_pipeline):
    """Invariant: disturbing the global ``numpy.random`` state between two runs changes nothing.

    Oracle: the two same-seed runs of
    :func:`test_the_same_seed_reproduces_the_whole_run`, with the global generator
    deliberately disturbed in between — ``np.random.seed(999)`` followed by a
    handful of draws — so a pipeline that still reads the global stream (the
    pre-rewrite library seeded it in ``Pipeline.__init__``) produces a different
    trajectory and fails here.

    Assertion: ``history_a == history_b`` and ``np.array_equal`` on the final
    payloads, exactly as in the same-seed test — the disturbance must be invisible.

    Notes:
        The disturbance is applied *between* the runs rather than before both, so
        it exercises the second run's initialisation (where a global reseed used
        to happen) as well as its mutation draws. ``numpy`` is imported inside the
        test body on purpose: this directory must stay importable with no
        dependency beyond ``pytest`` (``test/CONTEXT.md``).

    Owner: ``pipeline.Pipeline.setup`` — the seeding goes through
    ``evoxc.random.key`` and never through ``numpy.random``.
    """
    pytest.skip(
        "not implemented: a run is unaffected by the global numpy.random state "
        "(owner: pipeline.Pipeline.setup)"
    )


def test_a_run_leaves_the_global_numpy_generator_untouched(tiny_pipeline, run_pipeline):
    """Invariant: running a pipeline neither reseeds nor consumes the global ``numpy.random`` stream.

    Oracle: a reference draw — seed the global generator with a fixed value, draw
    two numbers, and compare them with the two numbers drawn from a generator
    seeded identically but run *around* a full ``setup()`` + ``auto_run()``. If the
    pipeline reseeded the stream (the legacy ``np.random.seed(self.seed)``) or
    consumed draws from it, the second number shifts and the two sequences differ.

    Assertion: the number drawn before the run equals the reference's first number
    and the number drawn after equals the reference's second, bit for bit.

    Notes:
        This is the leak-side complement of
        :func:`test_a_run_ignores_the_global_numpy_random_state`: that test proves
        the run does not *read* the global stream, this one proves it does not
        *write* it either. Together they pin the documented rule that all
        randomness flows through ``evoxc.random``'s stateless keys.

    Owner: ``pipeline.Pipeline.setup`` (stateless key creation).
    """
    pytest.skip(
        "not implemented: a run neither reseeds nor consumes the global numpy "
        "generator (owner: pipeline.Pipeline.setup)"
    )


def test_the_state_randkey_advances_deterministically(tiny_pipeline, run_pipeline):
    """Invariant: ``state.randkey`` advances once per generation, and only as a function of the seed.

    Oracle: the keys themselves — ``state.randkey`` observed after ``setup()`` and
    after ``auto_run()`` of two same-seed runs, unwrapped with
    ``support.runner.to_numpy`` (an ``evoxc`` key is not an ``ndarray``; wrapping
    it with ``np.asarray`` would compare objects, not values).

    Assertion: for both runs ``randkey_after != randkey_after_setup`` (the run
    consumed and advanced the key), and the two runs' ``randkey_after`` are
    ``np.array_equal``; a third run with ``seed=1`` yields a *different*
    ``randkey_after``, so the key is a function of the seed and not of a constant
    or of the global state.

    Notes:
        ``randkey`` is the only RNG state the pipeline is documented to carry
        (``Pipeline.setup`` creates it with ``evoxc.random.key(seed)``, and ``NEAT``'s
        per-generation split stores the advanced key back), so its trajectory is
        the cheapest complete witness that the RNG stream is reproducible without
        comparing populations.

    Owner: ``pipeline.Pipeline.setup``, ``algorithm.neat.neat.NEAT.tell``.
    """
    pytest.skip(
        "not implemented: state.randkey advances deterministically with the seed "
        "(owner: pipeline.Pipeline.setup, algorithm.neat.neat.NEAT.tell)"
    )
