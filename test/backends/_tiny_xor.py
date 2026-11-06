"""The tiny XOR configuration and the assertions both compiled-backend legs share.

``test_iree_pipeline.py`` (the llvm-cpu leg) and ``test_rocm_pipeline.py`` (the
AMD GPU leg) drive the *same* tiny XOR pipeline and prove the *same* two
properties: the run is populated, and the same seed on ``backend="numpy"`` is the
same run.  Only the device, the compile options and the measured float tolerance
differ, so the configuration, the driver and the shared assertions live here
instead of being copied -- a copy would drift the moment one leg's tolerance were
re-measured.

This is not a test module: the leading underscore keeps it out of pytest's
``test_*.py`` collection, and it contains no test body.  It imports no optional
dependency, so ``test_iree_pipeline.py`` can import it only *after* its
``importorskip`` guard and ``test/backends/conftest.py`` stays cheap.

What the pieces are:

* :func:`tiny_xor_pipeline` -- the one ``Pipeline`` both legs are built from
  (``pop_size=8``, ``max_nodes=8``, ``max_conns=16``, ``seed=42``, two
  generations) plus the ``backend``/``device``/``compile_options`` a leg needs;
* :func:`drive` -- ``setup()`` plus :data:`GENERATIONS` ``step()`` calls, keeping
  one ``(payload, fitnesses, generation)`` tuple per generation for the
  comparison;
* :func:`run_configuration` -- the seven values two runs must agree on *before*
  any number is compared;
* :func:`assert_populated` -- the generation-shaped assertion set, labelled with
  the leg it runs for;
* :func:`assert_same_run` -- the cross-backend comparison: integer key payloads
  bit-identical, float payloads and fitness vectors within the caller's
  tolerance, on *every* traced generation (generation 1 as well as the final
  one, because generation 1 is the host-seeded population while every later one
  also exercises the backend's mutation/crossover path).

The tolerances stay in the test modules: each of them documents its own
measurement, and :func:`assert_same_run` takes them as arguments instead of
baking a number in here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT, to_numpy

#: Population of the shared configuration: small enough that a one-off compile
#: stays a few seconds (measured: ~5.9 s on llvm-cpu, ~6.3 s on ROCm), large
#: enough that a "populated" result means something.
POP_SIZE = 8
#: Generations each backend runs (``setup()`` plus this many ``step()`` calls).
GENERATIONS = 2
#: Static genome geometry: the smallest window that holds a two-input XOR net
#: (2 inputs + 1 bias + 1 output = 4 nodes, 3 connections).
MAX_NODES = 8
MAX_CONNS = 16
#: The one seed every leg of the shared configuration is driven with.
SEED = 42
#: The ``compile_options`` key used by the no-silent-fallback test; it is valid
#: for no evoxc backend stage, so a compiled backend must reject it by name.
UNKNOWN_OPTION = "not_a_real_compile_option"

#: The four payload arrays of ``Pipeline.step``'s ``previous_pop``, in order, with
#: their per-individual rank (keys carry one trailing axis, attrs one more).
PAYLOAD_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")


def tiny_xor_pipeline(
    backend: str,
    *,
    device: object | None = None,
    compile_options: dict | None = None,
) -> Pipeline:
    """Build the shared tiny XOR pipeline on ``backend``, placed on ``device``.

    The configuration mirrors :func:`test.integration.conftest.tiny_pipeline`:
    a sigmoid read-out so ``-1e-3`` is reachable, ``species_size=4`` with the
    default ``min_species_size`` (2) inside ``pop_size=8``, and ``is_detailed``
    off so stdout stays readable.

    Args:
        backend: Forwarded verbatim to ``Pipeline`` (``"numpy"`` or a compiled
            ``evoxc`` backend such as ``"iree"``).
        device: Forwarded to ``Pipeline``'s ``device``. ``None`` keeps every
            tensor on the host; ``"rocm"`` makes the pipeline place the runtime
            state on the IREE HIP device (an unreachable device raises from
            ``setup()`` -- nothing falls back to CPU).
        compile_options: Forwarded verbatim to ``Pipeline``, which expands it
            into ``evoxc.build``'s ``**options``.

    Returns:
        Pipeline: The configured, not yet set-up pipeline.
    """
    return Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=POP_SIZE,
            species_size=4,
            survival_threshold=0.01,
            genome=genome.DefaultGenome(
                num_inputs=2,
                num_outputs=1,
                max_nodes=MAX_NODES,
                max_conns=MAX_CONNS,
                output_transform=ACT.sigmoid,
            ),
        ),
        problem=problem.XOR(),
        seed=SEED,
        generation_limit=GENERATIONS,
        fitness_target=-1e-3,
        backend=backend,
        device=device,
        compile_options=compile_options,
        is_detailed=False,
    )


def drive(pipeline: Pipeline) -> SimpleNamespace:
    """``setup()`` plus :data:`GENERATIONS` ``step()`` calls on ``pipeline``.

    The pipeline is passed in (rather than a backend name) so a leg can place it
    on a device and give it compile options; the driver itself stays backend
    agnostic.

    Args:
        pipeline: A configured, not yet set-up pipeline.

    Returns:
        SimpleNamespace: ``pipeline``, ``initial`` (the state out of ``setup``),
        ``state`` (the state after the last step) and ``trace`` -- one
        ``(payload, fitnesses, generation)`` tuple per generation, where
        ``payload`` is the four host arrays the generation evaluated and
        ``fitnesses`` its ``(P,)`` host fitness.
    """
    initial = pipeline.setup()
    state = initial
    trace = []
    for _ in range(GENERATIONS):
        state, payload, fitnesses = pipeline.step(state)
        payload = tuple(np.array(field) for field in payload)
        trace.append((payload, np.array(fitnesses), int(to_numpy(state.generation))))
    return SimpleNamespace(
        pipeline=pipeline, initial=initial, state=state, trace=trace
    )


def run_configuration(run: SimpleNamespace) -> tuple:
    """The seven configuration values a cross-backend comparison requires.

    Two runs are only comparable when all seven agree, so a leg pins its own
    tuple before comparing any payload value: the comparison can then never be
    satisfied by two different runs.

    Args:
        run: A :func:`drive` result.

    Returns:
        tuple: ``(backend, seed, pop_size, num_inputs, num_outputs, max_nodes,
        max_conns)``.
    """
    return (
        run.pipeline.backend,
        run.pipeline.seed,
        run.pipeline.algorithm.pop_size,
        run.pipeline.algorithm.genome.num_inputs,
        run.pipeline.algorithm.genome.num_outputs,
        run.pipeline.algorithm.genome.max_nodes,
        run.pipeline.algorithm.genome.max_conns,
    )


def assert_populated(run: SimpleNamespace, *, label: str) -> None:
    """Assert ``run`` is a populated, finite, single-build result.

    Oracle: the configuration itself -- ``pop_size`` individuals each carrying a
    finite fitness out of ``problem.XOR`` and a four-array payload of the static
    shape the genome declares.

    Args:
        run: A :func:`drive` result.
        label: How the leg is named in every failure message (e.g. ``"the iree
            backend"``, ``"the ROCm leg"``).
    """
    assert len(run.trace) == GENERATIONS, (
        f"setup() + {GENERATIONS} step() calls on {label} must yield "
        f"{GENERATIONS} generations, got {len(run.trace)}"
    )

    for index, (payload, fitnesses, generation) in enumerate(run.trace):
        assert generation == index + 1, (
            f"the generation counter must advance by one per step on {label}, "
            f"got {generation} at step {index}"
        )
        assert len(payload) == 4, (
            "step's payload must be the four gene arrays "
            f"(node_keys/node_attrs/conn_keys/conn_attrs) ({label}), got {len(payload)}"
        )
        node_keys, node_attrs, conn_keys, conn_attrs = payload
        assert node_keys.shape == (POP_SIZE, MAX_NODES), (
            f"node_keys ({label}): expected {(POP_SIZE, MAX_NODES)}, got "
            f"{node_keys.shape}"
        )
        # ``conn_keys`` carries one column per key component (``(in_key, out_key)``
        # for ``DefaultGenome``), so only the leading two axes are static here.
        assert conn_keys.shape[:2] == (POP_SIZE, MAX_CONNS), (
            f"conn_keys ({label}): expected leading axes {(POP_SIZE, MAX_CONNS)}, "
            f"got {conn_keys.shape}"
        )
        assert node_attrs.shape[:2] == (POP_SIZE, MAX_NODES), (
            f"node_attrs ({label}): expected leading axes {(POP_SIZE, MAX_NODES)}, "
            f"got {node_attrs.shape}"
        )
        assert conn_attrs.shape[:2] == (POP_SIZE, MAX_CONNS), (
            f"conn_attrs ({label}): expected leading axes {(POP_SIZE, MAX_CONNS)}, "
            f"got {conn_attrs.shape}"
        )
        assert node_keys.dtype == np.int32 and conn_keys.dtype == np.int32, (
            f"key payloads must stay int32 ({label}), got "
            f"{node_keys.dtype}/{conn_keys.dtype}"
        )
        assert node_attrs.dtype == np.float32 and conn_attrs.dtype == np.float32, (
            f"attr payloads must stay float32 ({label}), got "
            f"{node_attrs.dtype}/{conn_attrs.dtype}"
        )

        assert fitnesses.shape == (POP_SIZE,), (
            f"fitness shape ({label}): expected {(POP_SIZE,)}, got {fitnesses.shape}"
        )
        assert np.all(np.isfinite(fitnesses)), (
            f"every individual must carry a finite fitness on {label}, got "
            f"{fitnesses!r} -- a non-finite value means the graph produced NaN, "
            "not that the problem scored it so"
        )
        assert len(np.unique(fitnesses)) > 1, (
            f"a populated population must not collapse to a single fitness "
            f"({label}), got {fitnesses!r}"
        )

    assert int(to_numpy(run.state.generation)) == GENERATIONS, (
        f"the final generation counter ({label}) must be {GENERATIONS}, got "
        f"{int(to_numpy(run.state.generation))}"
    )
    assert not np.array_equal(
        to_numpy(run.state.randkey), to_numpy(run.initial.randkey)
    ), f"step must advance the randkey (one key split per generation) on {label}"
    assert run.pipeline.builds == 1, (
        "the executable must be built once and then reused -- a second build for "
        f"the same shape signature is a cache bug ({label}), got "
        f"{run.pipeline.builds}"
    )
    assert run.pipeline.build_seconds > 0.0, (
        f"the build must be timed into pipeline.build_seconds ({label}), got "
        f"{run.pipeline.build_seconds}"
    )


def assert_same_run(
    reference: SimpleNamespace,
    other: SimpleNamespace,
    *,
    reference_label: str,
    other_label: str,
    rtol: float,
    atol: float,
) -> None:
    """Assert ``other`` is the same run as ``reference``, up to float reduction order.

    Oracle: ``reference``, the numpy reference interpreter on the identical
    configuration and seed.

    Assertions, per traced generation: the generation counters are equal; the
    int32 key payloads are *bit-identical* (key generation is integer-only --
    randint, comparison, select -- so no backend may differ there); the float32
    attribute payloads and the fitness vectors agree to ``rtol``/``atol``, never
    to ``==``; every float payload carries at least one finite, non-zero entry on
    *both* sides, so the agreement can never be established by comparing padding
    against padding; and both fitness vectors hold more than one distinct value.

    Args:
        reference: The :func:`drive` result to compare against.
        other: The :func:`drive` result under test.
        reference_label: How ``reference`` is named in failure messages.
        other_label: How ``other`` is named in failure messages.
        rtol: The caller's measured relative tolerance.
        atol: The caller's measured absolute tolerance.
    """
    assert len(reference.trace) == len(other.trace) == GENERATIONS, (
        f"{reference_label} and {other_label} must both run {GENERATIONS} "
        f"generations, got {len(reference.trace)} and {len(other.trace)}"
    )

    for index, (reference_gen, other_gen) in enumerate(zip(reference.trace, other.trace)):
        (reference_payload, reference_fitness, reference_counter) = reference_gen
        (other_payload, other_fitness, other_counter) = other_gen

        assert other_counter == reference_counter, (
            f"generation counter diverged at step {index}: "
            f"{other_label} {other_counter} vs {reference_label} {reference_counter}"
        )

        for name, expected, actual in zip(
            PAYLOAD_FIELDS, reference_payload, other_payload
        ):
            assert expected.dtype == actual.dtype and expected.shape == actual.shape, (
                f"{name}: {other_label} {actual.dtype}{actual.shape} vs "
                f"{reference_label} {expected.dtype}{expected.shape}"
            )
            if expected.dtype.kind in "iu":
                np.testing.assert_array_equal(
                    expected,
                    actual,
                    err_msg=(
                        f"{name} at step {index} must be bit-identical between "
                        f"{reference_label} and {other_label}: the key payload is "
                        "produced by integer ops only (randint/comparison/select), "
                        "so any difference is a real semantic divergence, not float "
                        "reassociation"
                    ),
                )
            else:
                for side, array in ((reference_label, expected), (other_label, actual)):
                    finite = array[np.isfinite(array)]
                    assert finite.size > 0 and np.count_nonzero(finite) > 0, (
                        f"{name} at step {index} carries no finite, non-zero entry "
                        f"on the {side} side: comparing an all-padding payload "
                        "would make the allclose below vacuous"
                    )
                np.testing.assert_allclose(
                    expected,
                    actual,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,  # NaN marks an unconnected node in both backends
                    err_msg=(
                        f"{name} at step {index}: {reference_label} and "
                        f"{other_label} must agree on the float32 payload "
                        f"(rtol={rtol}, atol={atol}); the tolerance covers the "
                        "float reduction reassociation of the compiled leg only"
                    ),
                )

        np.testing.assert_allclose(
            reference_fitness,
            other_fitness,
            rtol=rtol,
            atol=atol,
            err_msg=(
                f"fitness at step {index} diverged: {other_label} "
                f"{other_fitness!r} vs {reference_label} {reference_fitness!r}"
            ),
        )
        assert (
            len(np.unique(reference_fitness)) > 1
            and len(np.unique(other_fitness)) > 1
        ), (
            "both sides must evaluate a population with more than one distinct "
            f"fitness, otherwise the agreement above is trivial: {other_label} "
            f"{other_fitness!r} vs {reference_label} {reference_fitness!r}"
        )
