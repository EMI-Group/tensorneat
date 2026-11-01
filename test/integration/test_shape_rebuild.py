"""Static shapes: a signature change rebuilds, and a mismatched shape is refused (marker ``slow``).

What this module proves
-----------------------
``Pipeline.step`` is the single compile site and shapes are static at build time
(``src/tensorneat/CONTEXT.md``, "Constraints"), so the pipeline must cache one
executable *per shape signature* — ``pop_size``, ``max_nodes``, ``max_conns``,
``max_in_degree``, ``max_depth``. Three consequences are tested here:

* **A new signature still runs and is really new.** A pipeline built for
  ``pop_size=8`` and a *second* pipeline built for ``pop_size=16`` both evolve,
  and the second's population/fitness arrays have 16 entries rather than a reused
  8-wide result. The first shape is then run once more, so a stale cache shared
  between the two would surface as a wrong length or a corrupted population in
  the third run.
* **A larger genome is a different signature, not an error.** ``max_nodes`` 8 → 16
  with ``max_in_degree`` 4 → 8 is a legal configuration change (the initial
  network still fits), so both must run and the second's payload must carry the
  wider row count.
* **An incompatible shape is refused, never reused.** Feeding a 16-individual
  population to a pipeline built for 8 must raise the backend's own error.
  ``src/tensorneat/CONTEXT.md`` is explicit: "never silently fall back" — a
  return value here would mean a cached executable quietly reshaped its batch.

The oracles
-----------
The **configured signature** itself: every assertion compares against
``pop_size``/``max_nodes`` as the test configured them, never against a shape read
back from the pipeline. The population axis of ``state.pop_nodes`` (documented as
an ``NEAT``-registered ``State`` key) and the re-scored ``(P,)`` fitness from
``./conftest.evaluate_population`` are the two host-visible sizes; the payload row
count is the third.

What is *not* asserted here
---------------------------
Which attribute holds the cache, and that two same-shape runs reuse one entry
bit-for-bit. The cache is documented behaviour but its attribute name belongs to
``pipeline.py``; :func:`test_the_same_shape_reuses_the_cached_executable` is
written against that behaviour and skipped, naming the owner, so it starts
passing — instead of being forgotten — when the module lands.

Budget (measured, not asserted)
-------------------------------
Each run is ``generation_limit=1`` (``fitness_target=1.0`` is unreachable, so the
generation always happens) at ``pop_size`` 8/16: four short runs plus one
deliberate failure, budgeted at a few seconds on the numpy reference interpreter.
The measured wall time is recorded here in Phase 2.

Why nothing here is a module-scope import
-----------------------------------------
``import tensorneat`` still raises ``NotImplementedError`` while ``src/`` is a
design skeleton, so every input arrives as a fixture resolved through
``support.lazy.require`` (``./conftest.py``) and the stub body reports the owner.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow]

#: One generation, always consumed (XOR fitness is ``-mse <= 0 < 1.0``).
ONE_GENERATION = {"generation_limit": 1, "fitness_target": 1.0}


def test_a_population_size_change_runs_and_yields_the_new_length(
    tiny_pipeline, run_pipeline, evaluate_population
):
    """Invariant: ``pop_size`` 8 and 16 each evolve at their own width, and 8-wide still works afterwards.

    Oracle: the configured ``pop_size`` on each side. ``state.pop_nodes`` carries
    the population on axis 0 (batch-first is the package-wide contract) and
    ``evaluate_population`` turns that state's population into a ``(P,)`` fitness
    array, so "the new length" is observable without reading any pipeline
    internals.

    Assertion: the 8-wide run and the 16-wide run both complete (``history`` has
    one entry, ``int(state.generation) == 1``); the 8-wide state's fitness array
    has shape ``(8,)`` and the 16-wide state's ``(16,)``; and a third run of a
    freshly built 8-wide pipeline — executed *after* the 16-wide one — again
    yields ``(8,)`` with a finite best fitness. A cache keyed by anything other
    than the shape signature produces a 16-long array (or an error) in that third
    run.

    Notes:
        Each run gets its own ``Pipeline`` object: ``pop_size`` is a constructor
        argument of ``NEAT``, so a shape-signature change is a configuration
        change, and the test is about the *cache*, not about mutating one pipeline.

    Owner: ``pipeline.Pipeline.auto_run`` (per-shape executable cache),
    ``algorithm.neat.neat.NEAT.setup``.
    """
    pytest.skip(
        "not implemented: a pop_size change runs at the new width without reusing "
        "the old executable (owner: pipeline.Pipeline.auto_run, "
        "algorithm.neat.neat.NEAT.setup)"
    )


def test_a_larger_genome_shape_runs(tiny_pipeline, run_pipeline):
    """Invariant: widening the genome (``max_nodes`` 8 → 16, ``max_in_degree`` 4 → 8) is a legal rebuild.

    Oracle: the configured genome shape. ``max_nodes`` and ``max_conns`` are the
    static row counts of the gene payload (``src/tensorneat/genome/CONTEXT.md``),
    and ``max_in_degree`` is the width ``K`` of the schedule's incoming-edge
    window, so a rebuilt pipeline must produce payloads with the new row count and
    a schedule without ``edge_overflow``.

    Assertion: both configurations run to ``generation_limit=1``; the wider run's
    ``state.pop_nodes``/``state.pop_conns`` carry 16 node rows / 32 connection rows
    (the configured ``max_conns``) instead of the 8/16 of the narrow one; the
    narrower run is unaffected. The schedule check is the documented diagnostic:
    ``Transformed.overflow`` is ``False`` for every individual, so no connection
    was silently dropped into a window that is too small.

    Notes:
        ``max_conns`` is widened together with ``max_nodes`` because a 16-row node
        payload of a denser initial network needs the room; widening only
        ``max_nodes`` would be a legal but pointless rebuild, and the point here
        is that the *signature*, not the population, is what changed.

    Owner: ``genome.base.BaseGenome.__init__`` (static shape validation),
    ``pipeline.Pipeline.auto_run`` (rebuilt executable),
    ``common.tensorization.build_schedule`` (the ``K``-wide window).
    """
    pytest.skip(
        "not implemented: a wider genome shape rebuilds and yields the new payload "
        "rows without edge overflow (owner: genome.base.BaseGenome.__init__, "
        "pipeline.Pipeline.auto_run)"
    )


def test_a_shape_mismatch_is_refused_instead_of_reused(
    tiny_pipeline, run_pipeline, payloads, evaluate_population
):
    """Invariant: a population wider than the built signature raises; it never returns a truncated result.

    Oracle: the documented rule that shapes are static at build time and that a
    backend/shape error propagates unchanged (``src/tensorneat/CONTEXT.md``: "never
    silently fall back"). The mismatched input is hand-built —
    ``payloads.new_payload(pop_size=16, max_nodes=8, max_conns=16, ...)`` — so the
    *only* thing wrong with it is its population axis, which is what a stale
    executable would be tempted to accept.

    Assertion: after a successful 8-wide generation, scoring the 16-individual
    payload through the same transform → forward → evaluate path raises. The
    ``pytest.raises`` is deliberately broad: pinning one ``evoxc`` exception class
    would couple this test to an internals name that the rewrite may change, while
    what the contract forbids is a *return value*. The failing case is re-checked
    against the correct 8-wide payload on the same state, which must still score —
    so the error is attributable to the shape and not to a broken pipeline.

    Notes:
        The genome payload built here is otherwise identical to the pipeline's own
        (same widths), so a kernel that quietly accepts a wrong batch size is the
        only way this test can pass while the contract is violated.

    Owner: ``pipeline.Pipeline`` (the per-shape executable cache) and
    ``common.tensorization.build_schedule`` (static ``P``).
    """
    pytest.skip(
        "not implemented: a mismatched population axis raises instead of reusing "
        "the cached executable (owner: pipeline.Pipeline, "
        "common.tensorization.build_schedule)"
    )


def test_the_same_shape_reuses_the_cached_executable(tiny_pipeline, run_pipeline):
    """Invariant: two generations of the same shape signature compile once, not twice.

    Oracle: the cache itself — the documented behaviour is "``auto_run`` caches the
    executable and loops on the host" (``src/tensorneat/CONTEXT.md``), so the
    second ``step`` of the same signature must reuse the object the first one
    compiled. The cleanest witness is the cache mapping: after two steps it holds
    exactly one entry, keyed by the shape signature, and the executable behind that
    key is the same object (``is``) before and after the second step. A callable
    that recompiles on every step would leave two entries or a new object.

    Assertion (to be enabled once ``pipeline.py`` is implemented): the pipeline's
    per-shape cache attribute — the name lives in ``pipeline.py`` and is read from
    there, not guessed here — has ``len(cache) == 1`` after two same-shape steps,
    and the value under the single key is identical (``is``) across them. A
    *different* signature reaching the same pipeline then makes it two entries
    with two distinct objects.

    Notes:
        This is the only test in the suite written against behaviour whose
        accessor is not yet public. It is skipped rather than dropped so the
        contract is pinned: when ``pipeline.py`` lands, the skip text names the
        module whose attribute name it needs, and nothing here has to be
        redesigned.

    Owner: ``pipeline.Pipeline`` (per-shape executable cache: attribute name owned
    by ``src/tensorneat/pipeline.py``).
    """
    pytest.skip(
        "not implemented: the per-shape executable cache is reused for a repeated "
        "shape signature (owner: pipeline.Pipeline — cache attribute, "
        "src/tensorneat/pipeline.py)"
    )
