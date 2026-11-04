"""The tiled-evaluate chunk budget is numerically neutral (marker ``slow``).

What this module proves
-----------------------
``host_payload._prepared_query`` regroups a tiled batch wider than
``EVAL_CHUNK_ROWS`` into row-contiguous groups of whole individuals. That
regrouping is a *performance* device, so the only acceptable behaviour is that it
changes nothing a user can observe: the same seed, configuration and shapes must
produce the same per-generation fitness and the same population with the
production budget (tens of thousands of rows, i.e. no chunking for this tiny
case) and with a budget small enough to force several groups.

The oracle
----------
The production-budget run itself. Two fresh pipelines are built from the same
configuration and seed; the second one is stepped while
``tensorneat.host_payload.EVAL_CHUNK_ROWS`` is monkeypatched to
:data:`CHUNK_ROWS` (read at trace time), and
``tensorneat.host_payload._slice_batch`` is wrapped to record that the chunked
path actually ran. The comparison is tolerance-free — ``fitness`` and all four
``state.pop`` arrays must agree exactly (``NaN`` padding included) — because
every forward op is row-local.

Budget (measured, never asserted)
---------------------------------
``pop_size=8``, a five-sample ``CustomFuncFit`` and two generations: ``P * S =
40`` tiled rows, four groups at ``CHUNK_ROWS=10``. No test reads a clock.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import step_once

pytestmark = [pytest.mark.slow]

#: The population the pipeline is built with.
POP_SIZE = 8

#: Samples in the ``CustomFuncFit`` dataset; the tiled batch is ``P * S = 40``.
NUM_SAMPLES = 5

#: The monkeypatched budget: ``10 // 5 = 2`` individuals per group, four groups.
CHUNK_ROWS = 10

#: Generations both runs consume; two are enough to cover initialisation *and*
#: a ``tell``-driven generation.
GENERATIONS = 2

#: The four ``state.pop`` field names, in payload order.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")


def _pipeline(tn):
    """Build a small ``CustomFuncFit`` pipeline with an unreachable target.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture).

    Returns:
        A fresh ``Pipeline`` whose two generations always consume the whole
        budget (fitness is ``-mse <= 0``, so ``fitness_target=1.0`` is never
        reached).
    """
    neat = tn.algorithm.NEAT(
        genome=tn.genome.DefaultGenome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=8,
            max_conns=16,
            max_in_degree=4,
            max_depth=4,
        ),
        pop_size=POP_SIZE,
        species_size=2,
    )
    problem = tn.problem.CustomFuncFit(
        lambda point: float(point[0] + point[1]),
        low_bounds=[-1.0, -1.0],
        upper_bounds=[1.0, 1.0],
        method="sample",
        num_samples=NUM_SAMPLES,
        seed=7,
        error_method="mse",
    )
    return tn.Pipeline(
        algorithm=neat,
        problem=problem,
        seed=42,
        fitness_target=1.0,
        generation_limit=GENERATIONS,
        backend="numpy",
        is_detailed=False,
    )


def _payload_arrays(runner, state):
    """Unwrap the four ``state.pop`` arrays into host numpy, in payload order.

    Args:
        runner: The ``support.runner`` module (``to_numpy`` unwraps evoxc
            tensors; ``np.asarray`` on a tensor would compare objects).
        state: A ``State`` whose ``pop`` is the algorithm's gene payload.

    Returns:
        A ``{field name: ndarray}`` mapping, so an assertion can name the field
        that differs.

    Raises:
        AssertionError: If the payload does not carry exactly the four
            documented fields.
    """
    pop = runner.to_numpy(state.pop)
    names = tuple(getattr(pop, "_fields", POP_FIELDS))
    assert names == POP_FIELDS, (
        f"state.pop must carry the four GenomeTensors fields {POP_FIELDS}, got {names}"
    )
    return dict(zip(names, (np.asarray(field) for field in pop)))


def _run(tn, runner):
    """Run the configured pipeline for :data:`GENERATIONS` generations.

    Args:
        tn: The imported ``tensorneat`` package.
        runner: The ``support.runner`` module.

    Returns:
        ``(fitness, populations)``: the per-generation ``(POP_SIZE,)`` fitness
        arrays in generation order, and the four-array ``state.pop`` snapshot
        each step returned.
    """
    pipeline = _pipeline(tn)
    state = pipeline.setup()
    fitness, populations = [], []
    for _ in range(GENERATIONS):
        state, values = step_once(pipeline, state)
        assert values.shape == (POP_SIZE,), (
            f"step must report one fitness per individual, got {values.shape}"
        )
        fitness.append(values)
        populations.append(_payload_arrays(runner, state))
    return fitness, populations


def _assert_same_fitness(default, chunked):
    """Assert two per-generation fitness sequences agree exactly.

    Args:
        default: The production-budget run's ``fitness`` list.
        chunked: The small-budget run's ``fitness`` list.

    Raises:
        AssertionError: If the generation counts or any value differ.
    """
    assert len(default) == len(chunked) == GENERATIONS
    for generation, (left, right) in enumerate(zip(default, chunked), start=1):
        assert np.array_equal(left, right, equal_nan=True), (
            f"generation {generation}: the chunked budget changed the fitness: "
            f"{left!r} vs {right!r}"
        )


def _assert_same_populations(default, chunked):
    """Assert two per-generation ``state.pop`` snapshots agree exactly.

    Args:
        default: The production-budget run's ``populations`` list.
        chunked: The small-budget run's ``populations`` list.

    Raises:
        AssertionError: If a field's shape, dtype or value differs. ``NaN`` is
            the attribute padding sentinel of an unused gene row, so it compares
            equal to itself; no live value is tolerated to differ.
    """
    for generation, (left, right) in enumerate(zip(default, chunked), start=1):
        for name in POP_FIELDS:
            first, second = left[name], right[name]
            assert first.shape == second.shape and first.dtype == second.dtype, (
                f"generation {generation}: state.pop field {name} changed shape or "
                f"dtype: {first.shape}/{first.dtype} vs {second.shape}/{second.dtype}"
            )
            floating = np.issubdtype(first.dtype, np.floating)
            assert np.array_equal(first, second, equal_nan=floating), (
                f"generation {generation}: the chunked budget changed state.pop "
                f"field {name}"
            )


def test_chunk_budget_is_numerically_neutral(tn, runner, monkeypatch):
    """Invariant: forcing several chunk groups changes neither fitness nor population.

    Oracle: the production-budget run of the same seed and configuration. The
    chunked run is built while ``EVAL_CHUNK_ROWS`` is ``10`` — below the tiled
    width of ``8 * 5 = 40`` rows, i.e. four groups of two individuals — so the
    grouping really happens; a wrapped ``_slice_batch`` records that it did.

    Assertion: both runs consume both generations; every per-generation fitness
    vector and all four final-population arrays are exactly equal
    (``equal_nan=True``); the wrapped slicer saw at least one group. Exact
    equality is the claim: the regrouped pass is bit-identical, not merely
    close.

    Notes:
        ``EVAL_CHUNK_ROWS`` is read at trace time, so the monkeypatch wraps the
        second pipeline's first ``step`` and nothing else. The first run is
        completed before the patch is applied, so it cannot be affected by it.
    """
    import tensorneat.host_payload as host_payload

    default_fitness, default_populations = _run(tn, runner)

    slices = []
    original_slice = host_payload._slice_batch

    def recording_slice(tree, start, stop, pop_size):
        """Record one group slice, then delegate to the real slicer."""
        slices.append((start, stop, pop_size))
        return original_slice(tree, start, stop, pop_size)

    with monkeypatch.context() as patch:
        patch.setattr(host_payload, "EVAL_CHUNK_ROWS", CHUNK_ROWS)
        patch.setattr(host_payload, "_slice_batch", recording_slice)
        chunked_fitness, chunked_populations = _run(tn, runner)

    assert slices, (
        "the small budget did not chunk the tiled batch: the test would pass "
        "without exercising the regrouped path"
    )
    assert {pop_size for _, _, pop_size in slices} == {POP_SIZE}, (
        f"every slice must be taken at the population axis {POP_SIZE}, got {slices}"
    )

    _assert_same_fitness(default_fitness, chunked_fitness)
    _assert_same_populations(default_populations, chunked_populations)
