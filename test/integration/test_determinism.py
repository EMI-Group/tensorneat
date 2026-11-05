"""Reproducibility of a whole run: ``(seed, config, shapes)`` determines every number (marker ``slow``).

What this module proves
-----------------------
``State`` is the only store the ``Pipeline`` writes, and every random draw flows through ``evoxc.random``'s stateless keys instead of the global ``numpy.random`` state.
Together those two rules give one testable claim: two runs of the same ``Pipeline`` configuration with the same ``seed`` are bit-identical, a different seed is a different run, and nothing happening to the global numpy generator can change either fact.
Each test builds fresh ``Pipeline`` objects and calls ``setup()`` itself, so no comparison can pass on shared mutable state.
Each run is forced to a fixed length (``fitness_target=1.0`` is unreachable because XOR fitness is ``-mse <= 0``), so two sequences can never differ in *length* for a reason unrelated to the seed.

The oracles
-----------
* The trajectory: the per-generation best fitness :func:`integration.conftest.step_loop` returns, one value per generation, read from the documented ``pipeline.step``.
* The final population: the four ``state.pop`` fields (``node_keys``, ``node_attrs``, ``conn_keys``, ``conn_attrs``), unwrapped with :func:`support.runner.to_numpy` and compared field by field with exact array equality — keys are ``int32`` and attributes ``float32``, so "identical run" is tolerance-free.
  ``NaN`` is the attribute padding sentinel of an unused gene row, so it compares equal to itself (``equal_nan=True``) while every live value still has to match bit for bit.
* The RNG key: ``state.randkey`` right after ``setup()`` and after the run; it advances per generation, the advance depends only on the seed, and two same-seed runs land on the same key.
* The global generator itself: a reference draw around a full run, which catches both a pipeline that *reads* the global stream and one that reseeds or consumes it.

Budget (measured, not asserted)
-------------------------------
At most three pipelines per test at ``pop_size=16``: eight generations for the trajectory/payload comparisons and two for the key test, so the module stays a few seconds on the numpy reference interpreter.
No test reads a clock (``test/CONTEXT.md``: a flaky determinism test is a bug in the test).

Owners
------
``pipeline.Pipeline.setup`` (creates ``randkey`` from ``evoxc.random.key(seed)``), ``algorithm.neat.neat.NEAT.tell`` (splits the key per generation and stores it back under ``state.randkey``), ``genome.base.BaseGenome.initialize`` (draws the initial population from that key).
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import numpy as np
import pytest

from integration.conftest import step_loop

pytestmark = [pytest.mark.slow]

#: Generations a fixed-length run always consumes.
FIXED_GENERATIONS = 8

#: The shorter budget of the randkey test, which builds three pipelines.
KEY_GENERATIONS = 2

#: The ``state.pop`` field names, used when a payload container carries no ``_fields``.
POP_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")


def _fixed_length(generations: int) -> dict[str, float]:
    """Knobs of a run that never stops early, so two runs are comparable position by position.

    Args:
        generations: How many generations the run must consume.

    Returns:
        ``generation_limit`` set to ``generations`` together with
        ``fitness_target=1.0``, which XOR can never reach (its fitness is
        ``-mse <= 0``); the run therefore always uses the whole budget.
    """
    return {"generation_limit": generations, "fitness_target": 1.0}


# --------------------------------------------------------------------------- #
# drivers and payload helpers over the documented ``Pipeline`` surface
# --------------------------------------------------------------------------- #


def _run(tiny_pipeline: Any, seed: int, generations: int) -> tuple[Any, list[float]]:
    """Build a fresh pipeline at ``seed``, run ``generations`` generations, return what it produced.

    Args:
        tiny_pipeline: The ``./conftest.py`` factory (``make(**overrides) -> Pipeline``).
        seed: The pipeline seed under test.
        generations: How many generations to consume, passed as ``generation_limit``
            together with an unreachable ``fitness_target`` so the budget is always used.

    Returns:
        ``(final_state, history)``: the state the last ``step`` returned and the
        per-generation best fitness in generation order.

    Notes:
        A new ``Pipeline`` is constructed per call: a cached executable or a
        module-level key carried between the two runs of a pair would make the
        comparison meaningless. The run goes through ``pipeline.step``
        (:func:`integration.conftest.step_loop`), never through ``auto_run``, so the
        trajectory needs no undocumented history attribute.
    """
    pipeline = tiny_pipeline(seed=seed, **_fixed_length(generations))
    state = pipeline.setup()
    return step_loop(pipeline, state, generations=generations)


def _payload_fields(runner: Any, state: Any) -> dict[str, np.ndarray]:
    """Map the four ``state.pop`` field names to host numpy arrays.

    Args:
        runner: The ``support.runner`` module (``to_numpy`` unwraps ``evoxc.Tensor``
            leaves; ``np.asarray`` on a raw ``evoxc`` tensor would compare objects).
        state: A ``State`` whose ``pop`` is a ``GenomeTensors`` payload.

    Returns:
        A ``{field_name: ndarray}`` mapping in documented order. A container without
        ``_fields`` is named positionally (``pop[0]`` …) instead of guessed, so the
        assertion message still says which array differed.

    Raises:
        AssertionError: If the payload is not the documented
        ``GenomeTensors`` contract: the field names must be exactly
        :data:`POP_FIELDS` in that order, ``node_keys``/``conn_keys`` must be
        ``int32`` and ``node_attrs``/``conn_attrs`` ``float32``, the ranks must be
        ``2/3/3/3``, ``conn_keys`` must carry a trailing ``(src, dst)`` pair, the
        attribute arrays must share the key arrays' first two axes, and all four
        must have the same population length on axis 0. Comparing a differently
        named, typed or shaped payload field by field would be a contract change,
        not a reproducibility failure.
    """
    pop = runner.to_numpy(state.pop)
    arrays = tuple(pop)
    declared = getattr(pop, "_fields", None)
    names = (
        tuple(declared)
        if declared
        else tuple(f"pop[{index}]" for index in range(len(arrays)))
    )
    assert names == POP_FIELDS, (
        f"state.pop must carry the {len(POP_FIELDS)} GenomeTensors fields "
        f"{POP_FIELDS} in documented order, got {names}"
    )
    fields = dict(zip(names, arrays))
    _assert_payload_contract(fields)
    return fields


def _assert_field_contract(
    name: str, array: np.ndarray, dtype: np.dtype, rank: int
) -> None:
    """Assert one ``state.pop`` field has the documented dtype and rank.

    Args:
        name: The field name, quoted in a failure message.
        array: The host numpy array of that field.
        dtype: The documented dtype of the field.
        rank: The documented number of axes of the field.

    Raises:
        AssertionError: If the dtype or rank differs, with a message naming the
            offending field.
    """
    assert array.dtype == np.dtype(dtype), (
        f"state.pop field {name!r} must be {np.dtype(dtype)}, got {array.dtype}"
    )
    assert array.ndim == rank, (
        f"state.pop field {name!r} must be rank {rank}, got shape {array.shape}"
    )


def _assert_payload_contract(fields: dict[str, np.ndarray]) -> None:
    """Assert a ``state.pop`` mapping satisfies the documented ``GenomeTensors`` geometry.

    Args:
        fields: The ``{field_name: ndarray}`` mapping of :func:`_payload_fields`,
            keyed by :data:`POP_FIELDS`.

    Raises:
        AssertionError: If a dtype, rank, connection-key width, attribute prefix or
            population length disagrees with the documented batch-first payload
            (``node_keys (P, N) int32``, ``node_attrs (P, N, A) float32``,
            ``conn_keys (P, C, 2) int32``, ``conn_attrs (P, C, W) float32``); the
            message names the offending field.
    """
    node_keys = fields["node_keys"]
    node_attrs = fields["node_attrs"]
    conn_keys = fields["conn_keys"]
    conn_attrs = fields["conn_attrs"]

    _assert_field_contract("node_keys", node_keys, np.int32, 2)
    _assert_field_contract("node_attrs", node_attrs, np.float32, 3)
    _assert_field_contract("conn_keys", conn_keys, np.int32, 3)
    _assert_field_contract("conn_attrs", conn_attrs, np.float32, 3)

    assert conn_keys.shape[-1] == 2, (
        "state.pop field 'conn_keys' must carry the (src, dst) key pair on its "
        f"last axis, got shape {conn_keys.shape}"
    )
    assert node_attrs.shape[:2] == node_keys.shape, (
        "state.pop field 'node_attrs' must have the node_keys prefix "
        f"{node_keys.shape}, got {node_attrs.shape}"
    )
    assert conn_attrs.shape[:2] == conn_keys.shape[:2], (
        "state.pop field 'conn_attrs' must have the conn_keys prefix "
        f"{conn_keys.shape[:2]}, got {conn_attrs.shape}"
    )

    lengths = {name: array.shape[0] for name, array in fields.items()}
    assert len(set(lengths.values())) == 1, (
        "every state.pop field must share the population length on axis 0, got "
        f"{lengths}"
    )


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare two payload arrays exactly, treating ``NaN`` padding as a value of its own.

    Args:
        left: A host array of a ``state.pop`` field.
        right: The array of the same field of the other state.

    Returns:
        ``True`` when shape, dtype and every element agree. A floating field is
        compared with ``equal_nan=True`` because ``NaN`` is the documented padding
        sentinel of an unused gene row (``support/payloads.py`` and
        ``genome/base.py``): two padded rows are the *same* padding, while every
        live value still has to match bit for bit — no tolerance is involved.

    Notes:
        ``np.array_equal`` on its own would report two identical NaN-padded
        attribute fields as different (``NaN != NaN``), which would make the module
        unable to see a real difference for the noise.
    """
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    floating = np.issubdtype(left.dtype, np.floating)
    return bool(np.array_equal(left, right, equal_nan=floating))


def _differing_fields(runner: Any, left: Any, right: Any) -> list[str]:
    """Names of the ``state.pop`` fields that are not exactly equal between two states.

    Args:
        runner: The ``support.runner`` module.
        left: The first final ``State``.
        right: The second final ``State``.

    Returns:
        The field names whose arrays fail :func:`_arrays_equal`, in documented order;
        empty means the two populations are identical bit for bit.
    """
    left_fields = _payload_fields(runner, left)
    right_fields = _payload_fields(runner, right)
    return [
        name
        for name, value in left_fields.items()
        if not _arrays_equal(value, right_fields[name])
    ]


def _assert_same_run(runner: Any, state_a: Any, state_b: Any, what: str) -> None:
    """Assert two final states hold bit-identical populations and the same generation.

    Args:
        runner: The ``support.runner`` module.
        state_a: The final ``State`` of the first run.
        state_b: The final ``State`` of the second run.
        what: A short description of the pair (e.g. ``"two seed=0 runs"``), quoted in
            the failure message together with the differing field names.

    Raises:
        AssertionError: If any ``state.pop`` field differs under
            :func:`_arrays_equal` or the two runs stopped at different generations.
    """
    differing = _differing_fields(runner, state_a, state_b)
    assert not differing, f"{what} ended with different state.pop field(s) {differing}"
    generation_a = _generation(runner, state_a)
    generation_b = _generation(runner, state_b)
    assert generation_a == generation_b, (
        f"{what} stopped at different generations: {generation_a} != {generation_b}"
    )


def _generation(runner: Any, state: Any) -> int:
    """Read ``state.generation`` as a Python int.

    Args:
        runner: The ``support.runner`` module.
        state: Any ``State`` carrying a 0-d ``generation`` tensor.

    Returns:
        The generation counter as an ``int``.
    """
    return int(np.asarray(runner.to_numpy(state.generation)))


def _randkey(runner: Any, state: Any) -> np.ndarray:
    """Read ``state.randkey`` as a host array, for ``np.array_equal`` comparisons.

    Args:
        runner: The ``support.runner`` module (``to_numpy`` unwraps the 0-d int64
            ``evoxc`` tensor; wrapping the tensor itself would compare objects).
        state: Any ``State`` carrying a ``randkey``.

    Returns:
        The key as a 0-d ``int64`` ndarray.
    """
    return np.asarray(runner.to_numpy(state.randkey))


@contextlib.contextmanager
def _global_stream_restored() -> Iterator[None]:
    """Restore the global ``numpy.random`` stream after the block.

    Yields:
        ``None``; the block runs with (and may freely disturb) the process-global
        generator.

    Notes:
        The tests that deliberately seed or consume the global stream put it back
        afterwards, so neither a replay nor another module's run depends on the
        order this file was collected in.
    """
    saved = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(saved)


# --------------------------------------------------------------------------- #
# the tests
# --------------------------------------------------------------------------- #


def test_the_same_seed_reproduces_the_whole_run(tiny_pipeline, runner):
    """Invariant: two fresh pipelines at ``seed=0`` give the same trajectory and the same final population.

    Oracle: the first run itself — the per-generation best-fitness sequence from
    :func:`integration.conftest.step_loop` and the four ``state.pop`` arrays. No
    re-implementation is involved: the claim is equality between two runs of the
    same documented configuration, which is exactly what "``(seed, config,
    shapes)`` reproduces a run" means.

    Assertion: both histories have :data:`FIXED_GENERATIONS` entries and compare
    ``==`` (Python floats, so bit-equal); every ``state.pop`` field is exactly
    equal under :func:`_arrays_equal` (``NaN`` padding included); both runs stopped
    at generation :data:`FIXED_GENERATIONS`. A single differing attribute bit fails
    — a tolerance here would let an RNG leak through.

    Budget: two pipelines, ``pop_size=16``, eight generations each.

    Owner: ``pipeline.Pipeline.setup`` (``evoxc.random.key(seed)``),
    ``algorithm.neat.neat.NEAT.tell`` (the per-generation key split).
    """
    state_a, history_a = _run(tiny_pipeline, seed=0, generations=FIXED_GENERATIONS)
    state_b, history_b = _run(tiny_pipeline, seed=0, generations=FIXED_GENERATIONS)

    assert len(history_a) == len(history_b) == FIXED_GENERATIONS, (
        "both same-seed runs must consume the whole fixed budget, got "
        f"{len(history_a)} and {len(history_b)} generations"
    )
    differing_positions = [
        position
        for position, (first, second) in enumerate(zip(history_a, history_b))
        if first != second
    ]
    assert not differing_positions, (
        "two seed=0 runs diverged in the per-generation best fitness at "
        f"generation(s) {differing_positions}: {history_a} vs {history_b}"
    )
    _assert_same_run(runner, state_a, state_b, "two seed=0 runs")
    assert _generation(runner, state_a) == FIXED_GENERATIONS, (
        f"eight generations must leave state.generation at {FIXED_GENERATIONS}, "
        f"got {_generation(runner, state_a)}"
    )


def test_a_different_seed_is_a_different_run(tiny_pipeline, runner):
    """Invariant: ``seed=0`` and ``seed=1`` do not produce the same trajectory or the same population.

    Oracle: the same two runnables as
    :func:`test_the_same_seed_reproduces_the_whole_run`; the assertion is the
    complement, which is what catches a seed that is accepted and then ignored (a
    hard-coded key, a constant initialisation, a cached executable whose population
    was never re-drawn).

    Assertion: the two histories have the same fixed length but differ in at least
    one generation — the failure message lists the *differing* positions, so "the
    last best fitness happens to coincide" never fails the test — and at least one
    ``state.pop`` field differs under :func:`_arrays_equal`.

    Notes:
        Deliberately not "every generation differs": two NEAT runs may agree early
        (the same initial topology family) and diverge later, so per-generation
        difference is not part of the claim. What must differ is the run.

    Budget: two pipelines, ``pop_size=16``, eight generations each.

    Owner: ``pipeline.Pipeline.setup`` (``evoxc.random.key(seed)``),
    ``genome.base.BaseGenome.initialize``.
    """
    state_0, history_0 = _run(tiny_pipeline, seed=0, generations=FIXED_GENERATIONS)
    state_1, history_1 = _run(tiny_pipeline, seed=1, generations=FIXED_GENERATIONS)

    assert len(history_0) == len(history_1) == FIXED_GENERATIONS, (
        "both fixed-length runs must consume the whole budget before they are "
        f"compared, got {len(history_0)} and {len(history_1)} generations"
    )
    differing_positions = [
        position
        for position, (first, second) in enumerate(zip(history_0, history_1))
        if first != second
    ]
    assert differing_positions, (
        "seed=0 and seed=1 produced the same per-generation best fitness "
        f"{history_0}; the seed does not reach evoxc.random"
    )
    differing_fields = _differing_fields(runner, state_0, state_1)
    assert differing_fields, (
        "seed=0 and seed=1 trajectories differ at generation(s) "
        f"{differing_positions}, but every state.pop field is identical "
        f"{tuple(_payload_fields(runner, state_0))}; the initial population does "
        "not depend on the seeding key"
    )


def test_a_run_ignores_the_global_numpy_random_state(tiny_pipeline, runner):
    """Invariant: disturbing the global ``numpy.random`` state between two runs changes nothing.

    Oracle: the two same-seed runs of
    :func:`test_the_same_seed_reproduces_the_whole_run`, with the global generator
    deliberately reseeded (``np.random.seed(999)``) and drawn from between them, so
    a pipeline that still reads the global stream produces a different trajectory
    and fails here.

    Assertion: both histories have :data:`FIXED_GENERATIONS` entries and compare
    ``==``, and every ``state.pop`` field is exactly equal under
    :func:`_arrays_equal` — the disturbance must be invisible.

    Notes:
        The disturbance sits *between* the runs, so it exercises the second run's
        ``setup()`` (where the pre-rewrite library reseeded globally) as well as its
        mutation draws. The global stream is restored afterwards
        (:func:`_global_stream_restored`), so this module never leaves the process
        generator in a state another test or a replay depends on.

    Budget: two pipelines, ``pop_size=16``, eight generations each.

    Owner: ``pipeline.Pipeline.setup`` — the seeding goes through
    ``evoxc.random.key`` and never through ``numpy.random``.
    """
    with _global_stream_restored():
        state_a, history_a = _run(
            tiny_pipeline, seed=0, generations=FIXED_GENERATIONS
        )

        np.random.seed(999)
        np.random.rand(4)  # the drawn values are irrelevant: any disturbance will do

        state_b, history_b = _run(
            tiny_pipeline, seed=0, generations=FIXED_GENERATIONS
        )

    assert len(history_a) == len(history_b) == FIXED_GENERATIONS, (
        "both same-seed runs must consume the whole fixed budget, got "
        f"{len(history_a)} and {len(history_b)} generations"
    )
    differing_positions = [
        position
        for position, (first, second) in enumerate(zip(history_a, history_b))
        if first != second
    ]
    assert not differing_positions, (
        "the global numpy.random state leaked into the run: two seed=0 runs around "
        "`np.random.seed(999)` differ at generation(s) "
        f"{differing_positions}: {history_a} vs {history_b}"
    )
    _assert_same_run(runner, state_a, state_b, "two seed=0 runs around a global reseed")


def test_a_run_leaves_the_global_numpy_generator_untouched(tiny_pipeline, runner):
    """Invariant: a run neither reseeds nor consumes the global ``numpy.random`` stream.

    Oracle: a reference draw — ``np.random.seed(1234)`` followed by two draws —
    compared with the two numbers drawn from a generator seeded identically but run
    *around* a full ``setup()`` + two-generation run. If the pipeline reseeded the
    stream or consumed draws from it, the second number shifts.

    Assertion: the pre-run draw equals the reference's first number and the post-run
    draw equals the reference's second, compared with ``==`` on ``np.float64``,
    which is bit equality for the same computation.

    Notes:
        This is the leak-side complement of
        :func:`test_a_run_ignores_the_global_numpy_random_state`: that test proves a
        run does not *read* the global stream, this one proves it does not *write*
        it either. The generator is seeded explicitly, so the test does not depend
        on whatever the previous module left behind; it is restored afterwards
        (:func:`_global_stream_restored`) for the same reason.

    Budget: one pipeline, ``pop_size=16``, two generations.

    Owner: ``pipeline.Pipeline.setup`` (stateless key creation through
    ``evoxc.random.key``).
    """
    with _global_stream_restored():
        np.random.seed(1234)
        reference_first = np.random.rand()
        reference_second = np.random.rand()

        np.random.seed(1234)
        before_run = np.random.rand()

        _, history = _run(tiny_pipeline, seed=0, generations=KEY_GENERATIONS)

        after_run = np.random.rand()

    assert len(history) == KEY_GENERATIONS, (
        "the run around the reference draw must be a real run, got "
        f"{len(history)} generation(s)"
    )
    assert before_run == reference_first, (
        "the pipeline consumed or reseeded the global numpy.random stream at "
        f"setup(): the first draw after seeding became {before_run!r} instead of "
        f"{reference_first!r}"
    )
    assert after_run == reference_second, (
        "the pipeline consumed or reseeded the global numpy.random stream while "
        f"stepping: the second draw became {after_run!r} instead of "
        f"{reference_second!r}"
    )


def test_the_state_randkey_advances_deterministically(tiny_pipeline, runner):
    """Invariant: ``state.randkey`` advances during a run, and the advance is a function of the seed alone.

    Oracle: the keys themselves — ``state.randkey`` observed right after ``setup()``
    and after the run of two same-seed pipelines plus one ``seed=1`` pipeline,
    unwrapped with ``support.runner.to_numpy`` (an ``evoxc`` key is not an
    ``ndarray``; ``np.asarray`` on the tensor would compare objects, not values).

    Assertion: for both ``seed=0`` pipelines the post-run key differs from the key
    ``setup()`` left behind (the run consumed and advanced it); the two post-run
    keys are ``np.array_equal``; the ``seed=1`` post-run key differs from the
    ``seed=0`` one; and the ``seed=1`` *setup* key already differs from the
    ``seed=0`` setup key — so the key is a function of ``(seed, generation count)``
    and not a constant.

    Notes:
        ``randkey`` is the only RNG state the pipeline is documented to carry
        (``setup`` creates it with ``evoxc.random.key(seed)``; ``NEAT.tell`` splits it
        per generation and writes the advanced key back), so its trajectory is the
        cheapest complete witness that the RNG stream is reproducible without
        comparing populations. ``setup()`` itself already consumes one split, which
        is why the post-setup key is compared with ``evoxc.random.key(seed)`` never
        and with the other seed always.

    Budget: three pipelines, ``pop_size=16``, :data:`KEY_GENERATIONS` generations
    each — the short budget is what keeps the three-way key comparison cheap.

    Owner: ``pipeline.Pipeline.setup``, ``algorithm.neat.neat.NEAT.tell``.
    """
    runs: dict[str, tuple[np.ndarray, np.ndarray, list[float]]] = {}
    for label, seed in (("a", 0), ("b", 0), ("c", 1)):
        pipeline = tiny_pipeline(seed=seed, **_fixed_length(KEY_GENERATIONS))
        state = pipeline.setup()
        # The post-setup key is only observable on the *same* pipeline before it
        # steps: a third pipeline per seed would be a different unit of work.
        setup_key = _randkey(runner, state)
        state, history = step_loop(pipeline, state, generations=KEY_GENERATIONS)
        runs[label] = (setup_key, _randkey(runner, state), history)

    for label, (_, _, history) in runs.items():
        assert len(history) == KEY_GENERATIONS, (
            f"seed {label!r} must consume the whole {KEY_GENERATIONS}-generation "
            f"budget before its key is compared, got {len(history)} generation(s)"
        )

    setup_a, key_a, _ = runs["a"]
    setup_b, key_b, _ = runs["b"]
    setup_c, key_c, _ = runs["c"]

    assert not np.array_equal(setup_a, key_a), (
        "the seed=0 run did not advance state.randkey: it is still "
        f"{setup_a!r} after {KEY_GENERATIONS} generation(s)"
    )
    assert not np.array_equal(setup_b, key_b), (
        "the second seed=0 run did not advance state.randkey: it is still "
        f"{setup_b!r} after {KEY_GENERATIONS} generation(s)"
    )
    assert np.array_equal(key_a, key_b), (
        "two seed=0 runs ended on different state.randkey values: "
        f"{key_a!r} vs {key_b!r}; the per-generation split is not reproducible"
    )
    assert not np.array_equal(key_a, key_c), (
        "seed=0 and seed=1 runs ended on the same state.randkey "
        f"{key_a!r}; the key does not depend on the seed"
    )
    assert not np.array_equal(setup_a, setup_c), (
        "seed=0 and seed=1 produced the same post-setup state.randkey "
        f"{setup_a!r}; `evoxc.random.key(seed)` is not reached"
    )
