"""The tiled-evaluate chunk guard and its numerical neutrality for a ``HyperNEAT`` run (marker ``slow``).

What this module proves
-----------------------
``host_payload._prepared_query`` regroups a tiled batch wider than
``EVAL_CHUNK_ROWS`` into row-contiguous groups of whole individuals, and
``host_payload._slice_batch`` is the guard that makes the regrouping safe: inside
a declared population-major tree every non-scalar leaf must carry the population
on axis 0, and a leaf whose leading dimension differs raises
``common.TensorNEATError`` instead of being silently shared or mis-sliced.

Two tests pin the two halves of that contract:

* **The guard itself.**  A ``namedtuple`` tree of plain numpy leaves -- an
  ``(8, 3)`` table and a rank-0 scalar -- is sliced along axis 0 to the
  individuals ``[2, 5)`` while the scalar passes through untouched; the same tree
  with a ``(7, 2)`` leaf is refused with a message naming the population axis and
  both the offending and the expected leading dimension.
* **The neutrality of the forced path.**  Two fresh ``HyperNEAT`` +
  ``CustomFuncFit`` pipelines are built from the same configuration and seed; the
  second is stepped while ``EVAL_CHUNK_ROWS`` is tiny enough to force one
  individual per group, and ``_slice_batch`` is wrapped to record that the
  regrouped path actually ran.  Per-generation fitness and all four final
  ``state.pop`` arrays must agree exactly (``NaN`` padding included) -- this is a
  performance device, so it may change nothing observable.

The oracle
----------
The production-budget run of the same seed and configuration.  The comparison is
tolerance-free because every forward op of a ``HyperNEAT`` substrate pass is
row-local; the two runs differ only in how the rows are grouped.

Relation to ``./test_chunked_evaluate.py``
------------------------------------------
The plain ``NEAT``/``CustomFuncFit`` version of the chunk-equality claim already
lives in ``test_chunked_evaluate.py``.  This module covers the wrapper path: the
``HyperNEAT`` schedule is a nested ``SubstrateSchedule`` and its payload is a
CPPN, so the same regrouping has to slice both trees of a different shape.

Budget (measured, never asserted)
---------------------------------
``pop_size=4``, a six-sample ``CustomFuncFit`` and two generations: ``P * S = 24``
tiled rows, so ``max(1, 5 // 6) = 1`` individual per group and four groups per
generation.  Two whole runs at that size; no test reads a clock.

Why nothing here is a module-scope ``tensorneat`` import
--------------------------------------------------------
``tn``, ``runner`` and ``hyperneat_pipeline`` arrive as fixtures;
``tensorneat.host_payload`` is imported inside the tests that patch or call it.
"""

from __future__ import annotations

import numpy as np
import pytest

from integration.conftest import step_once

pytestmark = [pytest.mark.slow]

#: The population the shared HyperNEAT fixture is built with.
POP_SIZE = 4

#: Samples in the fixture's ``CustomFuncFit`` dataset, so the tiled batch is 24 rows.
NUM_SAMPLES = 6

#: The monkeypatched budget: below ``NUM_SAMPLES``, i.e. one individual per group.
CHUNK_ROWS = 5

#: Generations both runs consume.
GENERATIONS = 2

#: The four ``state.pop`` field names, in payload order.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")


def _payload_arrays(runner, state):
    """Unwrap the four ``state.pop`` arrays into host numpy, in payload order.

    Args:
        runner: The ``support.runner`` module (``to_numpy`` unwraps evoxc tensors;
            ``np.asarray`` on a tensor would compare objects).
        state: A ``State`` whose ``pop`` is the CPPN gene payload.

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


def _run(tn, runner, hyperneat_pipeline):
    """Run one fresh HyperNEAT pipeline for :data:`GENERATIONS` generations.

    Args:
        tn: The imported ``tensorneat`` package (the ``tn`` fixture).
        runner: The ``support.runner`` module.
        hyperneat_pipeline: The shared HyperNEAT pipeline factory.

    Returns:
        ``(state, fitness, populations)``: the final ``State``, the per-generation
        ``(POP_SIZE,)`` fitness arrays in generation order, and the four-array
        ``state.pop`` snapshot each step returned.
    """
    pipeline = hyperneat_pipeline(generation_limit=GENERATIONS)
    state = pipeline.setup()
    fitness, populations = [], []
    for _ in range(GENERATIONS):
        state, values = step_once(pipeline, state)
        assert values.shape == (POP_SIZE,), (
            f"step must report one fitness per individual, got {values.shape}"
        )
        fitness.append(values)
        populations.append(_payload_arrays(runner, state))
    return state, fitness, populations


def _assert_same_fitness(default, chunked):
    """Assert two per-generation fitness sequences agree exactly.

    Args:
        default: The production-budget run's fitness list.
        chunked: The forced-chunk run's fitness list.

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
        default: The production-budget run's population snapshots.
        chunked: The forced-chunk run's population snapshots.

    Raises:
        AssertionError: If a field's shape, dtype or value differs.  ``NaN`` is the
            attribute padding sentinel of an unused gene row, so it compares equal
            to itself; no live value is tolerated to differ.
    """
    assert len(default) == len(chunked) == GENERATIONS
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


def test_slice_batch_slices_the_population_axis_and_rejects_other_leading_dims(tn):
    """Invariant: ``_slice_batch`` slices every population-major leaf and refuses a leaf that is not.

    Oracle: plain numpy indexing on the same leaves -- ``table[2:5]`` for the
    population axis and the identical scalar object for the rank-0 leaf.

    Assertion: a ``namedtuple`` tree of an ``(8, 3)`` table and a rank-0
    ``np.array(3)`` sliced ``[2, 5)`` at ``pop_size=8`` returns the same container
    type with ``table[2:5]`` and the untouched rank-0 scalar; the same tree with a
    ``(7, 2)`` leaf raises ``tn.common.TensorNEATError`` whose message matches
    ``"population axis"`` and names both ``7`` (the offending leading dim) and
    ``8`` (the expected ``pop_size``).

    Notes:
        The guard is deliberately shape-only-free in spirit: a leaf whose axis
        cannot be established must never be silently shared between groups, which
        is why the refusal is a ``TensorNEATError`` from the package rather than a
        bare shape error from numpy.

    Owner: ``host_payload._slice_batch`` (the chunk guard).
    """
    from collections import namedtuple

    from tensorneat.host_payload import _slice_batch

    Tree = namedtuple("Tree", "table scalar")
    table = np.arange(8 * 3, dtype=np.int32).reshape(8, 3)
    scalar = np.array(3, dtype=np.int32)
    tree = Tree(table, scalar)

    sliced = _slice_batch(tree, 2, 5, 8)
    assert isinstance(sliced, Tree), (
        f"_slice_batch must preserve the tree container, got {type(sliced).__name__}"
    )
    np.testing.assert_array_equal(sliced.table, table[2:5])
    assert sliced.scalar.shape == (), (
        f"a rank-0 leaf must pass through untouched, got shape {sliced.scalar.shape}"
    )
    np.testing.assert_array_equal(sliced.scalar, scalar)

    broken = Tree(np.zeros((7, 2), dtype=np.float32), np.array(3, dtype=np.int32))
    with pytest.raises(tn.common.TensorNEATError, match="population axis") as excinfo:
        _slice_batch(broken, 0, 3, 8)
    message = str(excinfo.value)
    assert "7" in message and "8" in message, (
        "the guard's message must name both the offending leading dim (7) and the "
        f"expected pop_size (8), got {message!r}"
    )


def test_forced_chunk_budget_is_bit_identical_for_hyperneat(tn, runner, hyperneat_pipeline, monkeypatch):
    """Invariant: forcing several chunk groups changes neither fitness nor the CPPN population.

    Oracle: the production-budget run of the same seed and configuration.  The
    chunked run is stepped while ``EVAL_CHUNK_ROWS`` is :data:`CHUNK_ROWS` (below
    the six-sample tiling factor), so ``_prepared_query`` groups one individual at
    a time; a wrapped ``_slice_batch`` records that the regrouped path ran.

    Assertion: the wrapper recorded group slices, each narrower than the
    population (so more than one group really ran) and all taken at the population
    axis :data:`POP_SIZE`; both runs consume both generations; every per-generation
    fitness vector and all four final ``state.pop`` arrays are exactly equal
    (ints exact, floats ``equal_nan=True``); the final fitness arrays agree
    exactly and both runs end at generation :data:`GENERATIONS`.

    Notes:
        ``EVAL_CHUNK_ROWS`` is read at trace time, so the patch covers the second
        run's steps and nothing else: the default run is completed before the
        patch is applied.  ``evaluate_population`` is deliberately not used here --
        the claim is that two *whole runs* are identical, so the comparison is
        made on what ``Pipeline.step`` itself returned.

    Owner: ``host_payload._prepared_query`` (the chunk regrouping) and
    ``host_payload._slice_batch`` (what it slices with).
    """
    import tensorneat.host_payload as host_payload

    default_state, default_fitness, default_populations = _run(
        tn, runner, hyperneat_pipeline
    )

    slices = []
    original_slice = host_payload._slice_batch

    def recording_slice(tree, start, stop, pop_size):
        """Record one group slice, then delegate to the real slicer."""
        slices.append((start, stop, pop_size))
        return original_slice(tree, start, stop, pop_size)

    with monkeypatch.context() as patch:
        patch.setattr(host_payload, "EVAL_CHUNK_ROWS", CHUNK_ROWS)
        patch.setattr(host_payload, "_slice_batch", recording_slice)
        chunked_state, chunked_fitness, chunked_populations = _run(
            tn, runner, hyperneat_pipeline
        )

    assert slices, (
        "the small budget did not chunk the tiled HyperNEAT batch: the test would "
        "pass without exercising the regrouped path"
    )
    assert {pop_size for _, _, pop_size in slices} == {POP_SIZE}, (
        f"every slice must be taken at the population axis {POP_SIZE}, got {slices}"
    )
    assert all(stop - start < POP_SIZE for start, stop, _ in slices), (
        "every recorded group must be narrower than the population, otherwise more "
        f"than one individual was evaluated per group and the chunked path did not "
        f"run: {slices}"
    )

    _assert_same_fitness(default_fitness, chunked_fitness)
    _assert_same_populations(default_populations, chunked_populations)

    assert np.array_equal(default_fitness[-1], chunked_fitness[-1], equal_nan=True), (
        "the final fitness array of the forced-chunk run differs from the default "
        f"budget: {default_fitness[-1]!r} vs {chunked_fitness[-1]!r}"
    )
    default_generation = int(tn.common.tools.to_numpy(default_state.generation))
    chunked_generation = int(tn.common.tools.to_numpy(chunked_state.generation))
    assert default_generation == chunked_generation == GENERATIONS, (
        "both runs must consume the configured generations, got "
        f"{default_generation} and {chunked_generation}"
    )
