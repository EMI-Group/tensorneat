"""The tiled-evaluate chunk budget is a device-aware static in the executable-cache key (marker ``slow``).

What this module proves
-----------------------
``tensorneat.host_payload.eval_chunk_rows`` decides at *trace time* how many
unrolled group passes a tiled evaluate emits, and
``Pipeline._shape_signature`` appends its value as the last signature element
(``src/tensorneat/pipeline.py``). Two consequences follow, and each has a test
here:

* **The arithmetic is a pure function of ``(device kind, max_nodes)``.** The CPU
  branch (``device=None`` and ``"cpu"``) returns :data:`EVAL_CHUNK_ROWS`
  regardless of ``max_nodes``; a non-CPU kind returns
  ``DEVICE_EVAL_CHUNK_BYTES // (4 * max(1, max_nodes))`` clamped *up* to the CPU
  floor, so a device leg is never more chunked than the CPU one and a zero
  ``max_nodes`` cannot divide by zero. The expected integers are hard-coded in
  the table, not re-derived through the same expression.
* **The budget really reaches the cache key.** After ``setup()`` the last
  signature element is ``eval_chunk_rows(_device_kind(), genome.max_nodes)``;
  ``device=None`` (``_device_kind() is None``) and ``device="cpu"``
  (``_device_kind() == "cpu"``) both yield the same last element, so the default
  path and an explicit CPU placement stay byte-identical.
* **Changing the budget mid-run is a rebuild, and the rebuilt graph is neutral.**
  A same-seed run patched only on its second generation shows the signature tail
  ``12500, 24, 12500, 12500`` and ``builds`` ``1, 2, 2, 2``: the new budget is a
  new cache entry, and restoring it re-hits the *original* executable -- no
  third build. The patched generation is bit-identical to an unpatched reference
  run (fitness and all four ``state.pop`` arrays), and a wrapped
  ``_slice_batch`` records the group slices of the patched step only (three
  groups of six individuals, two trees each at ``pop_size=16``), proving the
  rebuilt graph really unrolled the chunked path.

The oracle
----------
For the arithmetic, the documented formula with hand-computed integers
(``DEVICE_EVAL_CHUNK_BYTES = 1 << 30``; ``4 * 50 = 200`` bytes per row at
``max_nodes=50``, and ``1073741824 // 200 == 5368709``).
For the cache key, :func:`tensorneat.host_payload.eval_chunk_rows` on the
pipeline's *own* ``_device_kind()`` and the genome's *own* ``max_nodes`` -- the
wiring, not a copied literal -- and then a full unpatched reference run of the
same seed for the neutrality claim.

Budgets (measured, never asserted)
----------------------------------
``pop_size=16`` and four generations on XOR with ``fitness_target=1.0``
(unreachable: XOR fitness is ``-mse <= 0``), so the whole budget is consumed.
XOR tiles four samples, so the tiled width is ``16 * 4 = 64`` rows and a budget
of ``24`` becomes ``24 // 4 = 6`` individuals per group, three groups. Every run
compiles once (the patched run twice); the module is a few seconds on the numpy
reference interpreter and never reads a clock.

A note on reachability
----------------------
``EVAL_CHUNK_ROWS`` and :func:`eval_chunk_rows` both read the module global at
call time, so a monkeypatched CPU budget drives the chunked path with no device
attached. The device branch's own window arithmetic is therefore table-tested
here, not exercised on hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import TINY_CONFIG

pytestmark = [pytest.mark.slow]

#: The four gene-payload field names, in ``GenomeTensors`` order.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

#: Generations the mid-run cache test consumes: enough to reach a patched
#: generation, restore the budget and observe the cache re-hit afterwards.
GENERATIONS = 4

#: ``(device kind, max_nodes, expected rows)``, with the arithmetic done by hand.
#:
#: ``DEVICE_EVAL_CHUNK_BYTES == 1 << 30 == 1073741824`` and the CPU floor is
#: ``12500``; the device rows are ``1073741824 // (4 * max(1, max_nodes))``.
CHUNK_TABLE = (
    (None, 8, 12_500),  # no device: the CPU budget
    ("cpu", 8, 12_500),  # the CPU branch
    ("cpu", 0, 12_500),  # the CPU branch ignores max_nodes
    ("cuda", 50, 5_368_709),  # 1073741824 // 200
    ("cuda", 8, 33_554_432),  # 1073741824 // 32
    ("rocm", 1, 268_435_456),  # 1073741824 // 4
    ("cuda", 0, 268_435_456),  # the max(1, max_nodes) guard
    ("cuda", 100_000, 12_500),  # 2684 byte-budget rows clamped up to the CPU floor
)

#: The budget forced on one generation; ``24 // 4 = 6`` individuals per group.
SMALL_CHUNK_ROWS = 24


def _run(tn, runner, pipeline):
    """Step a pipeline :data:`GENERATIONS` times, recording what each step reported.

    Args:
        tn: The imported ``tensorneat`` package (used for the generation counter).
        runner: The ``support.runner`` module (``to_numpy`` unwraps the payload).
        pipeline: A configured, not-yet-run ``Pipeline``.

    Returns:
        ``(final_state, signatures, builds, fitness, populations)``: the last
        ``State``, the last shape-signature element after every step, the
        ``builds`` count after every step, the per-generation ``(P,)`` fitness
        arrays and the per-generation four-array ``state.pop`` snapshots.
    """
    state = pipeline.setup()
    signatures, builds, fitness, populations = [], [], [], []
    for _ in range(GENERATIONS):
        state, _, values = pipeline.step(state)
        signatures.append(pipeline.shape_signature[-1])
        builds.append(pipeline.builds)
        fitness.append(values)
        populations.append(_payload_arrays(runner, state.pop))
    assert int(tn.common.tools.to_numpy(state.generation)) == GENERATIONS
    return state, signatures, builds, fitness, populations


def _payload_arrays(runner, payload):
    """Unwrap one payload into host numpy arrays, keyed by field name.

    Args:
        runner: The ``support.runner`` module (``to_numpy`` unwraps the
            ``evoxc.Tensor`` leaves; ``np.asarray`` on a tensor leaves an object
            array, so every comparison below goes through this helper).
        payload: A ``GenomeTensors`` (or any four-element payload sequence), the
            population on axis 0.

    Returns:
        A ``{field name: ndarray}`` mapping in documented order.

    Raises:
        AssertionError: If the payload's field names are not :data:`POP_FIELDS`.
    """
    arrays = runner.to_numpy(payload)
    names = tuple(getattr(arrays, "_fields", POP_FIELDS))
    assert names == POP_FIELDS, (
        f"a gene payload must carry the four GenomeTensors fields {POP_FIELDS}, "
        f"got {names}"
    )
    return dict(zip(names, (np.asarray(array) for array in arrays)))


def _assert_same_populations(expected, actual, label):
    """Assert two ``state.pop`` snapshots agree exactly, field for field.

    Args:
        expected: The reference run's ``{field: ndarray}`` snapshot.
        actual: The run under test's snapshot.
        label: A short description quoted in a failure message.

    Raises:
        AssertionError: If a field's shape, dtype or value differs. ``NaN`` is
            the attribute padding sentinel of an unused gene row, so it compares
            equal to itself; no live value is tolerated to differ.
    """
    for name in POP_FIELDS:
        first, second = expected[name], actual[name]
        assert first.shape == second.shape and first.dtype == second.dtype, (
            f"{label}: state.pop field {name} changed shape or dtype: "
            f"{first.shape}/{first.dtype} vs {second.shape}/{second.dtype}"
        )
        floating = np.issubdtype(first.dtype, np.floating)
        assert np.array_equal(first, second, equal_nan=floating), (
            f"{label}: state.pop field {name} differs from the reference run"
        )


@pytest.mark.parametrize("device_kind, max_nodes, expected", CHUNK_TABLE)
def test_eval_chunk_rows_matches_the_hand_computed_table(device_kind, max_nodes, expected):
    """``eval_chunk_rows`` is the documented pure function of its two arguments.

    The expected rows are hard-coded above (never re-derived through the same
    expression), every result is a Python ``int``, and no result is below the CPU
    budget: the device leg may only *raise* the chunk size, never make a graph
    more chunked than the CPU one.
    """
    import tensorneat.host_payload as host_payload

    rows = host_payload.eval_chunk_rows(device_kind, max_nodes)

    assert type(rows) is int, (
        f"eval_chunk_rows({device_kind!r}, {max_nodes}) must return an int, got "
        f"{type(rows).__name__}"
    )
    assert rows == expected, (
        f"eval_chunk_rows({device_kind!r}, {max_nodes}) is {rows}, expected "
        f"{expected} (DEVICE_EVAL_CHUNK_BYTES={host_payload.DEVICE_EVAL_CHUNK_BYTES}, "
        f"EVAL_CHUNK_ROWS={host_payload.EVAL_CHUNK_ROWS})"
    )
    assert rows >= host_payload.EVAL_CHUNK_ROWS, (
        f"eval_chunk_rows({device_kind!r}, {max_nodes}) is {rows}, below the CPU "
        f"budget {host_payload.EVAL_CHUNK_ROWS}; a device leg must never be more "
        "chunked than the CPU one"
    )


def test_eval_chunk_rows_reads_the_cpu_constant_at_call_time(monkeypatch):
    """The CPU value *and* the device floor come from the one module global.

    Patching ``EVAL_CHUNK_ROWS`` to ``50000`` moves the CPU result and the device
    floor together, while the device formula above the floor
    (``cuda``/``50``) is unchanged -- so the constant is not duplicated as a
    literal anywhere in the function.
    """
    import tensorneat.host_payload as host_payload

    monkeypatch.setattr(host_payload, "EVAL_CHUNK_ROWS", 50_000)

    assert host_payload.eval_chunk_rows(None, 8) == 50_000, (
        "the CPU branch must read EVAL_CHUNK_ROWS at call time"
    )
    assert host_payload.eval_chunk_rows("cpu", 8) == 50_000, (
        "the explicit CPU branch must read EVAL_CHUNK_ROWS at call time"
    )
    assert host_payload.eval_chunk_rows("cuda", 100_000) == 50_000, (
        "the device floor must be the patched EVAL_CHUNK_ROWS"
    )
    assert host_payload.eval_chunk_rows("cuda", 50) == 5_368_709, (
        "the device window arithmetic (1073741824 // 200) must be unaffected by "
        "the CPU floor"
    )


def test_the_chunk_budget_is_the_last_shape_signature_element(tn, tiny_pipeline):
    """The signature's last element is ``eval_chunk_rows(_device_kind(), max_nodes)``.

    Oracle: the pipeline's own ``_device_kind()`` (``None`` for the default
    ``device=None``) and the genome's own ``max_nodes`` -- the wiring itself,
    asserted against the constant the configuration declares.
    """
    import tensorneat.host_payload as host_payload

    pipeline = tiny_pipeline()
    state = pipeline.setup()

    genome = pipeline._genome()
    max_nodes = genome.max_nodes
    assert max_nodes == TINY_CONFIG["max_nodes"], (
        f"the pipeline's genome must own max_nodes={TINY_CONFIG['max_nodes']}, got "
        f"{max_nodes}"
    )
    assert pipeline._device_kind() is None, (
        "device=None must report no device kind (the CPU interpreter), got "
        f"{pipeline._device_kind()!r}"
    )
    assert pipeline._shape_signature(state)[-1] == host_payload.eval_chunk_rows(
        pipeline._device_kind(), max_nodes
    ) == 12_500, (
        "the last shape-signature element must be the device-aware chunk budget "
        f"eval_chunk_rows(None, {max_nodes}) == 12500, got "
        f"{pipeline._shape_signature(state)[-1]!r}"
    )


def test_an_explicit_cpu_device_keeps_the_same_last_signature_element(tn, tiny_pipeline):
    """``device="cpu"`` is the same budget as ``device=None``.

    Oracle: the default pipeline's signature. An explicit CPU placement resolves
    to ``_device_kind() == "cpu"`` but the same last element, so the default path
    and the explicit CPU placement share one executable signature.
    """
    default = tiny_pipeline()
    placed = tiny_pipeline(device="cpu")
    default_state = default.setup()
    placed_state = placed.setup()

    assert default._device_kind() is None
    assert placed._device_kind() == "cpu", (
        'device="cpu" must report the "cpu" device kind, got '
        f"{placed._device_kind()!r}"
    )
    assert placed._shape_signature(placed_state)[-1] == default._shape_signature(
        default_state
    )[-1] == 12_500, (
        'the last signature element must be identical for device=None and '
        f'device="cpu" (12500), got {default._shape_signature(default_state)[-1]!r} '
        f"and {placed._shape_signature(placed_state)[-1]!r}"
    )


def test_a_mid_run_budget_change_rebuilds_once_and_is_numerically_neutral(
    tn, runner, tiny_pipeline, monkeypatch
):
    """The budget is in the cache key, and a rebuilt chunked graph is bit-identical.

    A same-seed reference run of :data:`GENERATIONS` generations is stepped
    unpatched; a second same-seed pipeline patches ``EVAL_CHUNK_ROWS`` to
    :data:`SMALL_CHUNK_ROWS` for its second ``step`` only (read at trace time)
    and wraps ``_slice_batch`` to record the group slices. The signature tail must
    read ``12500, 24, 12500, 12500`` and ``builds`` ``1, 2, 2, 2``: the restored
    budget re-hits the original executable, so there is no third build. Every
    generation's fitness and the final four ``state.pop`` arrays must match the
    reference run exactly, and the recorded slices must be the patched step's
    three groups of ``24 // 4 = 6`` individuals, two trees each.
    """
    import tensorneat.host_payload as host_payload

    reference = tiny_pipeline(fitness_target=1.0, generation_limit=GENERATIONS)
    _, reference_signatures, reference_builds, reference_fitness, reference_pops = _run(
        tn, runner, reference
    )
    assert reference_signatures == [12_500] * GENERATIONS, (
        "the unpatched reference run must keep one budget in its signature, got "
        f"{reference_signatures}"
    )
    assert reference_builds == [1] * GENERATIONS, (
        f"the reference run must build once, got {reference_builds}"
    )

    slices = []
    original_slice = host_payload._slice_batch

    def recording_slice(tree, start, stop, pop_size):
        """Record one group slice, then delegate to the real slicer."""
        slices.append((start, stop, pop_size))
        return original_slice(tree, start, stop, pop_size)

    pipeline = tiny_pipeline(fitness_target=1.0, generation_limit=GENERATIONS)
    state = pipeline.setup()
    signatures, builds, fitness, populations = [], [], [], []
    for generation in range(GENERATIONS):
        if generation == 1:
            with monkeypatch.context() as patch:
                patch.setattr(host_payload, "EVAL_CHUNK_ROWS", SMALL_CHUNK_ROWS)
                patch.setattr(host_payload, "_slice_batch", recording_slice)
                state, _, values = pipeline.step(state)
        else:
            state, _, values = pipeline.step(state)
        signatures.append(pipeline.shape_signature[-1])
        builds.append(pipeline.builds)
        fitness.append(values)
        populations.append(_payload_arrays(runner, state.pop))

    assert signatures == [12_500, SMALL_CHUNK_ROWS, 12_500, 12_500], (
        "the chunk budget must be the last shape-signature element, so patching it "
        f"for one generation changes the key only there; got {signatures}"
    )
    assert builds == [1, 2, 2, 2], (
        "the patched budget must be a second cache entry and the restored budget "
        f"must re-hit the original executable (no third build); got {builds}"
    )
    assert slices == [
        (0, 6, 16),
        (0, 6, 16),
        (6, 12, 16),
        (6, 12, 16),
        (12, 16, 16),
        (12, 16, 16),
    ], (
        "the rebuilt graph must really take the chunked path: at pop 16 with four "
        "XOR samples a 24-row budget is 6 individuals per group, three groups, and "
        f"two trees (schedule + payload) sliced each; got {slices}"
    )

    for generation, (expected, actual) in enumerate(
        zip(reference_fitness, fitness), start=1
    ):
        assert np.array_equal(expected, actual, equal_nan=True), (
            f"generation {generation}: the patched budget changed the fitness: "
            f"{expected!r} vs {actual!r}"
        )
    for generation, (expected, actual) in enumerate(
        zip(reference_pops, populations), start=1
    ):
        _assert_same_populations(expected, actual, f"generation {generation} state.pop")
