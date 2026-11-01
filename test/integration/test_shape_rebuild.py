"""Static shapes: a signature change rebuilds, a repeated signature reuses, a rerun stays consistent (marker ``slow``).
What this module proves
-----------------------
``Pipeline.step`` is the single compile site and every kernel shape is static at build time, so the pipeline caches one executable per shape signature ``(P, max_nodes, max_conns, max_in_degree, key_span)`` and builds a new one when that signature changes.
Three consequences are tested here, each through the documented ``Pipeline`` surface only:
* **A new population width is a new signature.** A ``pop_size=8`` and a ``pop_size=16`` pipeline both evolve one generation at their own width, and a third, freshly built 8-wide pipeline run *after* the 16-wide one still produces 8-individual payloads and 8 fitness values, so a cache keyed by anything but the shape (or by insertion order) shows up as a wrong length.
* **A wider genome is a legal rebuild, not an error.** ``max_nodes`` 8 → 16 together with ``max_conns`` 16 → 32 and ``max_in_degree`` 4 → 8 runs, produces the wider payload rows, and derives a schedule whose ``edge_overflow`` is zero for every individual — no connection was silently dropped into a window that is too narrow.
* **A repeated signature neither recompiles nor corrupts the population.** A second generation of the same shape adds no compile call after the first one, a different ``pop_size`` does, and one extra generation on an already-run pipeline keeps the population re-scorable at the same width.
The oracles
-----------
The **configured signature** itself: every assertion compares against the ``pop_size``/``max_nodes``/``max_conns``/``max_in_degree`` the test passed to ``tiny_pipeline``, never against a shape read back from the pipeline.
The host-visible witnesses are ``state.pop``'s leading axes (population on axis 0, ``N`` node rows, ``C`` connection rows), the ``(P,)`` fitness ``pipeline.step`` returns, the same-length fitness re-scored by ``./conftest.evaluate_population`` through the documented kernel triple ``algorithm.transform`` → ``algorithm.forward`` → ``problem.evaluate``, and ``Transformed.edge_overflow`` from ``./conftest.transform_of``.
Compile reuse is observed with ``./conftest.record_compiles``, which counts calls into ``evoxc.build/build_cached/lower/compile`` — no pipeline attribute is named, so the module cannot go stale when ``pipeline.py``'s cache representation changes.
What is deliberately *not* asserted
-----------------------------------
Which attribute holds the cache, and that an off-shape population fed to a built pipeline raises: neither is observable from outside today, because the executable lives inside ``pipeline.py`` and a payload never reaches a cached executable without going through the pipeline's own entry points.
Instead of a ``pytest.raises(Exception)`` that could only restate ``evoxc``'s internals, :func:`test_a_same_shape_rerun_leaves_the_pipeline_consistent` pins what the contract actually protects — a reused or rebuilt executable leaves a population that re-scores to a finite ``(P,)`` vector.
The one remaining gap is documented, not hidden: if no compile call is visible at all, :func:`test_a_second_generation_of_the_same_shape_does_not_compile_again` skips and names the entry points the compile site has to go through.
Budget (measured, not asserted)
-------------------------------
Every run is ``generation_limit=1`` with ``fitness_target=1.0`` (XOR fitness is ``-mse <= 0``, so one generation is always consumed), except the last test which steps one extra generation at ``pop_size=8``.
``pop_size`` is 8 or 16 and ``max_nodes`` at most 16, which keeps the module at a few seconds on the numpy reference interpreter; compiles are counted, never timed.
Why nothing here is a module-scope ``tensorneat`` import
-------------------------------------------------------
``tensorneat`` arrives through the ``tn`` fixture and the payload helpers through ``./conftest``, which resolves everything lazily with ``support.lazy.require``; a module-scope import would turn a named skip into a collection error while ``src/`` is mid-rewrite.
The directory-wide autouse probe ``require_evoxc_pipeline`` is the only guard: while ``pipeline.py`` is still the legacy JAX module every test here skips with the exact list of missing evoxc features, and the skip disappears by itself once the evoxc compile site lands.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import step_once, transform_of

pytestmark = [pytest.mark.slow]

#: One generation, always consumed (XOR fitness is ``-mse <= 0 < 1.0``).
ONE_GENERATION = {"generation_limit": 1, "fitness_target": 1.0}

#: The ``state.pop`` field names, used when a payload container carries no ``_fields``.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: The narrow genome signature of :data:`integration.conftest.TINY_CONFIG`.
NARROW_GENOME = {"max_nodes": 8, "max_conns": 16, "max_in_degree": 4}

#: The widened genome signature: ``max_conns``/``max_in_degree`` grow with ``max_nodes``.
#:
#: Widening ``max_nodes`` alone would be a legal but pointless rebuild: the static
#: connection rows (``max_conns``) and the ``K``-wide incoming-edge window
#: (``max_in_degree``) are what a 16-row node payload actually needs room in, so the
#: whole signature moves together and the rebuild is exercised where it matters.
WIDE_GENOME = {"max_nodes": 16, "max_conns": 32, "max_in_degree": 8}


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


def _payload_shapes(state) -> dict[str, tuple[int, ...]]:
    """Map the four ``state.pop`` field names to their host-visible static shapes.

    Args:
        state: A ``State`` whose ``pop`` is a ``GenomeTensors`` payload (population
            on axis 0).

    Returns:
        ``{field_name: shape}`` in documented order, each shape as a tuple of Python
        ``int``. A container without ``_fields`` is named positionally instead of
        guessed, so a failure message still says which array had the wrong shape.

    Notes:
        Shapes are static, so reading ``.shape`` needs no unwrapping of values; that
        keeps the check identical for an ``evoxc.Tensor`` leaf and a numpy one.
    """
    pop = state.pop
    names = getattr(pop, "_fields", None) or tuple(
        f"pop[{index}]" for index in range(len(pop))
    )
    return {name: tuple(int(size) for size in field.shape) for name, field in zip(names, pop)}


def _assert_one_generation_run(tn, evaluate_population, pipeline, state, pop_size: int) -> None:
    """Assert a finished run advanced exactly one generation and is re-scorable at its width.

    Args:
        tn: The imported ``tensorneat`` package.
        evaluate_population: The ``./conftest`` host re-scoring helper.
        pipeline: The ``Pipeline`` the state came from (its ``pop_size`` is the
            configured width, and the documented ``pipeline.pop_size`` attribute).
        state: The final ``State`` of the run.
        pop_size: The width the test configured.

    Raises:
        AssertionError: If the run did not stop at generation 1, if ``pipeline.pop_size``
            disagrees with the configuration, if the population's leading axes are not
            ``pop_size``, or if the re-scored fitness is not a finite ``(pop_size,)``
            vector.
    """
    generation = _generation(tn, state)
    assert generation == 1, (
        "the target is unreachable and one generation was requested, so the run must "
        f"stop at generation 1 for pop_size={pop_size}, got {generation}"
    )
    assert pipeline.pop_size == pop_size, (
        f"pipeline.pop_size must be the configured {pop_size}, got "
        f"{pipeline.pop_size!r}"
    )

    shapes = _payload_shapes(state)
    assert tuple(shapes) == POP_FIELDS, (
        f"state.pop must carry the {len(POP_FIELDS)} GenomeTensors fields "
        f"{POP_FIELDS}, got {tuple(shapes)}"
    )
    rows = shapes["node_keys"][0]
    assert rows == pop_size, (
        f"a pop_size={pop_size} run must keep {pop_size} individuals on axis 0 of "
        f"state.pop, got {rows} from node_keys {shapes['node_keys']}"
    )

    fitness = np.asarray(evaluate_population(pipeline, state, state.pop))
    assert fitness.shape == (pop_size,), (
        f"re-scoring the {pop_size}-individual population must return a "
        f"({pop_size},) fitness vector, got {fitness.shape}"
    )
    best = float(np.max(fitness))
    assert np.isfinite(best), (
        f"the pop_size={pop_size} run left no evaluable individual: the best "
        f"re-scored fitness is {best!r} (XOR maps a NaN prediction to -inf)"
    )


def test_a_population_size_change_runs_and_yields_the_new_length(
    tn, tiny_pipeline, run_pipeline, evaluate_population
):
    """Invariant: ``pop_size`` 8 and 16 each evolve at their own width, and 8-wide still works afterwards.

    Oracle: the configured ``pop_size`` on each side. ``state.pop`` carries the
    population on axis 0 (batch-first is the package-wide contract), and
    ``./conftest.evaluate_population`` turns that state's population into a
    ``(pop_size,)`` fitness vector, so "the new length" is observable without
    reading any pipeline internals.

    Assertion: the 8-wide and the 16-wide run each stop at ``generation == 1``,
    report ``pipeline.pop_size`` as configured, keep ``pop_size`` individuals on
    axis 0, and re-score to a finite ``(pop_size,)`` vector; a third run of a
    *freshly built* 8-wide pipeline — executed after the 16-wide one — again
    yields 8 individuals and 8 fitness values. A cache keyed by anything other
    than the shape signature (or by insertion order) produces a 16-long array, a
    reused 8-wide result, or an error in that third run.

    Notes:
        Each run gets its own ``Pipeline`` object: ``pop_size`` is a constructor
        argument of ``NEAT``, so a shape-signature change is a configuration
        change, and the subject here is the per-shape executable, not a mutation
        of one pipeline.
        The third run is what makes the test two-directional: without it, a
        pipeline that simply never reuses anything could not be distinguished
        from a correct one.

    Budget: three pipelines, one generation each at ``pop_size`` 8/16 — a few
    seconds on the numpy reference interpreter.

    Owner: ``pipeline.Pipeline.auto_run`` (per-shape executable cache),
    ``algorithm.neat.neat.NEAT.setup``.
    """
    widths = (("8-wide", 8), ("16-wide", 16))
    for label, pop_size in widths:
        pipeline = tiny_pipeline(pop_size=pop_size, **ONE_GENERATION)
        state, _ = run_pipeline(pipeline)
        _assert_one_generation_run(tn, evaluate_population, pipeline, state, pop_size)

    # The witness that the cache is keyed by shape rather than by run order: the
    # narrow signature is exercised again *after* the wide one went through.
    fresh = tiny_pipeline(pop_size=8, **ONE_GENERATION)
    state, _ = run_pipeline(fresh)
    _assert_one_generation_run(tn, evaluate_population, fresh, state, 8)


def test_a_larger_genome_shape_runs(tn, tiny_pipeline, run_pipeline):
    """Invariant: widening the genome (``max_nodes`` 8 → 16, ``max_conns`` 16 → 32, ``max_in_degree`` 4 → 8) is a legal rebuild.

    Oracle: the configured genome shape. ``max_nodes``/``max_conns`` are the static
    row counts of the gene payload and ``max_in_degree`` is the width ``K`` of the
    schedule's incoming-edge window, so a rebuilt pipeline must produce payloads
    with the new rows and a schedule without ``edge_overflow``.

    Assertion: both configurations run one generation; the wide run's
    ``state.pop.node_keys`` is ``(pop_size, 16)`` and ``conn_keys`` is
    ``(pop_size, 32, 2)``, its attributes keep 4 node columns and 1 connection
    column, the narrow run keeps ``(pop_size, 8)``/``(pop_size, 16, 2)``; and
    ``Transformed.edge_overflow`` from ``./conftest.transform_of`` is all zeros for
    every individual of *both* runs, so no connection was silently dropped into a
    window that is too small.

    Notes:
        ``max_conns`` and ``max_in_degree`` are widened together with ``max_nodes``
        deliberately: a 16-row node payload of a denser initial network needs the
        static connection rows and the incoming-edge window to grow with it, and
        widening only ``max_nodes`` would rebuild the signature without moving the
        bounds the rebuild exists for.
        The zero-overflow check is the documented diagnostic of the tensorization:
        an overflow silently truncates a genome's incoming edges, which is exactly
        the "falls back instead of rebuilding" failure the static-shape contract
        forbids.

    Budget: two pipelines, one generation each at the default ``pop_size=16``.

    Owner: ``genome.base.BaseGenome.__init__`` (static shape validation),
    ``pipeline.Pipeline.auto_run`` (per-shape executable),
    ``common.tensorization`` (the ``K``-wide window and ``edge_overflow``).
    """
    runs = {}
    for label, genome_shape in (("narrow", NARROW_GENOME), ("wide", WIDE_GENOME)):
        pipeline = tiny_pipeline(**genome_shape, **ONE_GENERATION)
        state, _ = run_pipeline(pipeline)
        generation = _generation(tn, state)
        assert generation == 1, (
            f"the {label} genome run must consume its one-generation budget, "
            f"got generation {generation}"
        )
        runs[label] = (pipeline, state)

    pop_size = int(runs["wide"][0].pop_size)
    for label, expected_nodes, expected_conns in (
        ("narrow", NARROW_GENOME["max_nodes"], NARROW_GENOME["max_conns"]),
        ("wide", WIDE_GENOME["max_nodes"], WIDE_GENOME["max_conns"]),
    ):
        pipeline, state = runs[label]
        shapes = _payload_shapes(state)
        assert shapes["node_keys"] == (pop_size, expected_nodes), (
            f"the {label} run must keep {expected_nodes} static node rows, got "
            f"{shapes['node_keys']}"
        )
        assert shapes["conn_keys"] == (pop_size, expected_conns, 2), (
            f"the {label} run must keep {expected_conns} static connection rows "
            f"with two key columns, got {shapes['conn_keys']}"
        )
        assert shapes["node_attrs"][-1] == 4, (
            f"the {label} run must keep the 4 node attribute columns, got "
            f"{shapes['node_attrs']}"
        )
        assert shapes["conn_attrs"][-1] == 1, (
            f"the {label} run must keep the 1 connection attribute column, got "
            f"{shapes['conn_attrs']}"
        )

        transformed = transform_of(pipeline, state, state.pop)
        overflow = getattr(transformed, "edge_overflow", None)
        assert overflow is not None, (
            "Transformed must expose edge_overflow, the documented per-individual "
            f"count of connections that did not fit; got attributes "
            f"{getattr(transformed, '_fields', None)}"
        )
        counts = np.asarray(overflow)
        assert counts.shape == (pop_size,), (
            f"edge_overflow must be a ({pop_size},) int vector, got {counts.shape}"
        )
        assert not np.any(counts), (
            f"the {label} genome run dropped "
            f"{int(np.max(counts))} connection(s) per individual into a too-small "
            f"edge window: edge_overflow is {counts.tolist()} for a signature whose "
            "max_in_degree covers the network"
        )


def test_a_second_generation_of_the_same_shape_does_not_compile_again(
    tiny_pipeline, compiles
):
    """Invariant: a second generation of the same shape signature adds no compile, a different width does.

    Oracle: ``./conftest.record_compiles``, which counts calls into
    ``evoxc.build/build_cached/lower/compile`` while the block runs. The count is
    the observation point precisely because it names no cache attribute: "the
    second generation of one shape does not compile again" is exactly what a
    reuse contract predicts, whatever ``pipeline.py`` stores its executables in.

    Assertion: after ``setup()`` (outside the block) the first ``step_once``
    records ``n1 > 0`` compile call(s) — proof that the instrumentation reaches the
    compile site — and a second ``step_once`` on the state it returned records the
    same count ``n2 == n1``. A second pipeline built with a different ``pop_size``
    then reaches its first generation and pushes the count past ``n1``
    (``n3 > n1``): a new shape signature is a new executable.

    Notes:
        The skip is the one documented observability gap and is not a placeholder:
        if the first step compiles nothing observable, the per-shape executable
        cannot be seen from outside at all, and the skip text names the entry
        points the compile site has to go through (``evoxc.build`` and friends) for
        this contract to become checkable.
        ``setup()`` runs before the block so that an initialization kernel cannot
        be mistaken for the generation kernel; building the second pipeline's
        object also happens before the block, while its ``setup()`` and first step
        happen inside it, so "a new width costs compiles" is measured on the whole
        of what a new signature requires rather than on a guessed subset.

    Budget: two pipelines at ``pop_size`` 8 and 16, two and one generation
    respectively; compiles are counted, never timed.

    Owner: ``pipeline.Pipeline.step`` (single compile site),
    ``pipeline.Pipeline.auto_run`` (per-shape executable cache).
    """
    first = tiny_pipeline(pop_size=8, **ONE_GENERATION)
    state = first.setup()
    wider = tiny_pipeline(pop_size=16, **ONE_GENERATION)

    with compiles() as calls:
        state, fitness = step_once(first, state)
        n1 = len(calls)
        if n1 == 0:
            pytest.skip(
                "no evoxc.build/build_cached/lower/compile call was observed during "
                "the first step, so the per-shape executable cannot be observed from "
                "outside; pipeline.py's compile site needs to go through one of those "
                "entry points"
            )
        state, fitness = step_once(first, state)
        n2 = len(calls)
        wider_state = wider.setup()
        wider_state, wider_fitness = step_once(wider, wider_state)
        n3 = len(calls)

    assert fitness.shape == (8,), (
        f"the second pop_size=8 generation must report 8 fitness values, got "
        f"{fitness.shape}"
    )
    assert wider_fitness.shape == (16,), (
        f"the pop_size=16 generation must report 16 fitness values, got "
        f"{wider_fitness.shape}"
    )
    assert n2 == n1, (
        "the second generation of the same shape signature compiled again: the "
        f"compile count went {n1} -> {n2} across two steps at pop_size=8, so the "
        "per-shape executable is rebuilt instead of reused"
    )
    assert n3 > n1, (
        "a pipeline built for a different pop_size reached its first generation "
        f"without compiling anything new ({n1} -> {n3}); a new shape signature "
        "must get its own executable rather than reuse the 8-wide one"
    )


def test_a_same_shape_rerun_leaves_the_pipeline_consistent(
    tn, tiny_pipeline, run_pipeline, evaluate_population
):
    """Invariant: one extra generation on an already-run pipeline keeps the population scorable at its width.

    Oracle: the generation counter and the host re-scoring of the resulting
    population. ``state.generation`` must reach 2, and
    ``./conftest.evaluate_population`` of the ``state.pop`` that second generation
    produced must return a finite ``(8,)`` vector — the same population read back
    through the documented kernel triple, independent of whatever executable the
    pipeline itself used.

    Assertion: after ``auto_run`` with a one-generation budget stops at generation
    1, one more ``pipeline.step`` returns a ``(8,)`` fitness vector and leaves
    ``state.generation == 2``; re-scoring that state's population yields ``(8,)``
    with a finite maximum.

    Notes:
        This replaces a broad ``pytest.raises(Exception)`` on an off-shape
        population: that assertion could only restate ``evoxc``'s own exception
        class for a payload the pipeline never accepts from outside, while what the
        static-shape contract actually protects is that a reused or rebuilt
        executable leaves a population that still evaluates at its configured
        width. The population is re-scored on the host, so a corrupted payload
        (wrong rows, misaligned keys, a schedule built for the previous shape)
        surfaces as a shape or value failure instead of a silently wrong run.
        The pipeline keeps ``generation_limit=1``; ``pipeline.step`` advances one
        generation regardless of that limit, and ``auto_run`` is not called again.

    Budget: one pipeline at ``pop_size=8``, two generations in total.

    Owner: ``pipeline.Pipeline.step`` (reuses or rebuilds its executable),
    ``pipeline.Pipeline.auto_run`` (per-shape executable cache).
    """
    pipeline = tiny_pipeline(pop_size=8, **ONE_GENERATION)
    state, _ = run_pipeline(pipeline)
    assert _generation(tn, state) == 1, (
        "the one-generation budget must leave the pipeline at generation 1, got "
        f"{_generation(tn, state)}"
    )

    state, fitness = step_once(pipeline, state)
    assert fitness.shape == (8,), (
        f"the extra generation must report 8 fitness values, got {fitness.shape}"
    )
    generation = _generation(tn, state)
    assert generation == 2, (
        f"one more pipeline.step must advance the rerun to generation 2, got "
        f"{generation}"
    )

    rerun_fitness = np.asarray(evaluate_population(pipeline, state, state.pop))
    assert rerun_fitness.shape == (8,), (
        "the population left by the rerun must re-score to an (8,) fitness vector, "
        f"got {rerun_fitness.shape}"
    )
    best = float(np.max(rerun_fitness))
    assert np.isfinite(best), (
        "the population left by the rerun has no evaluable individual: the best "
        f"re-scored fitness is {best!r}, so a rebuilt or reused executable "
        "corrupted the payload"
    )
