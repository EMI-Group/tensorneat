"""The ``backend="iree"`` tests: a small XOR pipeline really runs on the compiled backend.

What this module proves, on one tiny XOR configuration (``pop_size=8``,
``max_nodes=8``, ``max_conns=16``, ``seed=42``) that compiles in a few seconds:

1. ``Pipeline(..., backend="iree")`` executes ``setup()`` plus two real
   generations and returns a *populated* payload: ``pop_size`` individuals, a
   finite fitness per individual, an advanced ``randkey``/``generation`` counter
   and exactly one ``evoxc.build`` (the second step reuses the cached
   executable).
2. The same seed on ``backend="numpy"`` produces the *same run*: the int32 key
   payloads and the generation counters are bit-identical, while every float32
   value agrees to :data:`FLOAT_RTOL` / :data:`FLOAT_ATOL` -- never to ``==``,
   because iree reassociates float reductions (see the comment on those
   constants).
3. A construct the backend cannot take raises the backend's own error instead of
   silently falling back: an unknown ``compile_options`` key is *rejected by the
   iree backend* (``evoxc.core.errors.BackendError`` naming the option) while the
   numpy reference interpreter -- which documents no option validation -- accepts
   the very same dictionary. That contrast is what proves the pipeline forwards
   options verbatim and never retries elsewhere.

Cost: the compile dominates. On this box ``setup()`` is ~1 s and the first
``step()`` (which pays the evoxc build/compile/load) ~8 s for this
configuration, after which a generation is milliseconds; the whole module stays
well under two minutes. The three tests share one iree and one numpy run through
module-scoped fixtures, so the iree compile happens once.

Run it:

    uv sync --extra iree                  # or: uv pip install "evoxc[iree]"
    pytest -q test/backends               # or: pytest -q -m requires_iree

``pytest.importorskip("iree")`` below is a structural optional-dependency skip,
not a disabled test: the import exists only when the extra is installed, and the
same file then runs its assertions for real.  The two ways of reaching this
module have deliberately different consequences when the extra is absent, and
``conftest.py`` enforces the difference: a whole-tree/default run (or merely
naming this directory) keeps the one structural skip, while ``-m requires_iree``
-- a run that explicitly asks for the compiled-backend leg -- fails with the
install command instead of reporting a green "1 skipped".
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip(
    "iree",
    reason=(
        "structural optional-dependency skip: the compiled-backend suite needs "
        "the optional iree extra, which is not installed; install it with "
        '`uv sync --extra iree` (equivalently `uv pip install "evoxc[iree]"`: '
        "iree-base-compiler + iree-base-runtime). A default or whole-tree run "
        "skips this module on purpose; an explicit `pytest -m requires_iree` run "
        "fails in ./conftest.py instead of skipping silently."
    ),
)

from evoxc.core import BackendError  # noqa: E402  (after the skip guard)

from tensorneat import Pipeline, algorithm, genome, problem  # noqa: E402
from tensorneat.common import ACT, to_numpy  # noqa: E402

pytestmark = pytest.mark.requires_iree

#: Population of the shared configuration: small enough that the one-off iree
#: compile stays ~8 s, large enough that a "populated" result means something.
POP_SIZE = 8
#: Generations each backend runs (``setup()`` + this many ``step()`` calls).
GENERATIONS = 2
#: Static genome geometry: the smallest window that holds a two-input XOR net
#: (2 inputs + 1 bias + 1 output = 4 nodes, 3 connections).
MAX_NODES = 8
MAX_CONNS = 16
#: The one seed both backends are driven with.
SEED = 42
#: The ``compile_options`` key used by the no-silent-fallback test; it is valid
#: for no evoxc backend stage, so iree must reject it by name.
UNKNOWN_OPTION = "not_a_real_compile_option"

#: Relative tolerance for float32 values across backends. iree's llvm-cpu target
#: reassociates floating-point reductions (a sum over a matrix column is a
#: different tree than numpy's pairwise loop), and each reassociation can move a
#: float32 result by one ulp (~1.19e-7 at unit magnitude) per reordering, so the
#: error grows with the reduction depth but stays within a small multiple of eps.
#: Measured on pop 8/16 over several seeds: max |iree - numpy| <= 6e-8, so 1e-5
#: leaves ~two orders of magnitude of headroom for a box- or build-dependent
#: reordering while still failing on any real divergence (a wrong operand, a
#: dropped edge or a changed activation moves values by >> 1e-2).
FLOAT_RTOL = 1e-5
#: Absolute floor for the same comparison: ``node_attrs``/``conn_attrs`` can sit
#: near zero, where a pure relative bound is meaningless. 1e-6 is ~8x the largest
#: difference measured here and still ~1 % of the smallest meaningful weight.
FLOAT_ATOL = 1e-6

#: The four payload arrays of ``Pipeline.step``'s ``previous_pop``, in order, with
#: their per-individual rank (keys carry one trailing axis, attrs one more).
PAYLOAD_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")


def _pipeline(backend: str, compile_options: dict | None = None) -> Pipeline:
    """Build the shared tiny XOR pipeline on ``backend``.

    The configuration mirrors :func:`test.integration.conftest.tiny_pipeline`:
    a sigmoid read-out so ``-1e-3`` is reachable, ``species_size=4`` with the
    default ``min_species_size`` (2) inside ``pop_size=8``, and ``is_detailed``
    off so stdout stays readable.
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
        compile_options=compile_options,
        is_detailed=False,
    )


def _drive(backend: str) -> SimpleNamespace:
    """``setup()`` plus :data:`GENERATIONS` ``step()`` calls on ``backend``.

    Returns:
        SimpleNamespace: ``pipeline``, ``initial`` (state out of ``setup``),
        ``state`` (state after the last step) and ``trace`` -- one
        ``(payload, fitnesses, generation)`` tuple per generation, where
        ``payload`` is the four host arrays the generation evaluated and
        ``fitnesses`` its ``(P,)`` host fitness.
    """
    pipeline = _pipeline(backend)
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


@pytest.fixture(scope="module")
def iree_run() -> SimpleNamespace:
    """One iree run of the shared configuration (the compile happens here once)."""
    return _drive("iree")


@pytest.fixture(scope="module")
def numpy_run() -> SimpleNamespace:
    """The numpy run of the *same* configuration and seed, to compare against."""
    return _drive("numpy")


def test_iree_step_produces_a_finite_population(iree_run: SimpleNamespace) -> None:
    """Two real iree generations yield a populated, finite, advanced state.

    Oracle: the configuration itself -- ``pop_size`` individuals each carrying a
    finite fitness out of ``problem.XOR`` and a four-array payload of the static
    shape the genome declares.
    """
    assert len(iree_run.trace) == GENERATIONS, (
        f"setup() + {GENERATIONS} step() calls must yield {GENERATIONS} "
        f"generations, got {len(iree_run.trace)}"
    )

    for index, (payload, fitnesses, generation) in enumerate(iree_run.trace):
        assert generation == index + 1, (
            "the generation counter must advance by one per step, got "
            f"{generation} at step {index}"
        )
        assert len(payload) == 4, (
            "step's payload must be the four gene arrays "
            f"(node_keys/node_attrs/conn_keys/conn_attrs), got {len(payload)}"
        )
        node_keys, node_attrs, conn_keys, conn_attrs = payload
        assert node_keys.shape == (POP_SIZE, MAX_NODES), node_keys.shape
        # ``conn_keys`` carries one column per key component (``(in_key, out_key)``
        # for ``DefaultGenome``), so only the leading two axes are static here.
        assert conn_keys.shape[:2] == (POP_SIZE, MAX_CONNS), conn_keys.shape
        assert node_attrs.shape[:2] == (POP_SIZE, MAX_NODES), node_attrs.shape
        assert conn_attrs.shape[:2] == (POP_SIZE, MAX_CONNS), conn_attrs.shape
        assert node_keys.dtype == np.int32 and conn_keys.dtype == np.int32, (
            f"key payloads must stay int32, got {node_keys.dtype}/{conn_keys.dtype}"
        )
        assert node_attrs.dtype == np.float32 and conn_attrs.dtype == np.float32, (
            f"attr payloads must stay float32, got {node_attrs.dtype}/{conn_attrs.dtype}"
        )

        assert fitnesses.shape == (POP_SIZE,), fitnesses.shape
        assert np.all(np.isfinite(fitnesses)), (
            "every individual must carry a finite fitness on the iree backend, got "
            f"{fitnesses!r} -- a non-finite value means the compiled graph "
            "produced NaN, not that the problem scored it so"
        )
        assert len(np.unique(fitnesses)) > 1, (
            "a populated population must not collapse to a single fitness, got "
            f"{fitnesses!r}"
        )

    assert int(to_numpy(iree_run.state.generation)) == GENERATIONS
    assert not np.array_equal(
        to_numpy(iree_run.state.randkey), to_numpy(iree_run.initial.randkey)
    ), "step must advance the randkey (one key split per generation)"
    assert iree_run.pipeline.builds == 1, (
        "the executable must be built once and then reused -- a second build for "
        f"the same shape signature is a cache bug, got {iree_run.pipeline.builds}"
    )
    assert iree_run.pipeline.build_seconds > 0.0, (
        "the iree build must be timed into pipeline.build_seconds, got "
        f"{iree_run.pipeline.build_seconds}"
    )


def test_iree_matches_numpy_on_the_same_seed(
    iree_run: SimpleNamespace, numpy_run: SimpleNamespace
) -> None:
    """The same seed on iree and numpy is the same run, up to float reduction order.

    Oracle: the numpy reference interpreter on the identical configuration.

    Assertions: generation counters and the int32 key payloads are exactly
    equal (key generation is integer-only: randint, comparison, select, so no
    backend may differ there); the float32 attribute payloads and the fitness
    vectors agree to :data:`FLOAT_RTOL` / :data:`FLOAT_ATOL`, never to ``==``.
    Both fixtures are also pinned to the *same* configuration and seed first, and
    every float comparison must see a finite, non-zero entry, so agreement can
    never be established by comparing padding against padding.
    """
    assert len(iree_run.trace) == len(numpy_run.trace) == GENERATIONS

    # A comparison is only meaningful between the same run configuration, so pin
    # the values both fixtures were built from before looking at a number.
    iree_config = (
        iree_run.pipeline.backend,
        iree_run.pipeline.seed,
        iree_run.pipeline.algorithm.pop_size,
        iree_run.pipeline.algorithm.genome.num_inputs,
        iree_run.pipeline.algorithm.genome.num_outputs,
        iree_run.pipeline.algorithm.genome.max_nodes,
        iree_run.pipeline.algorithm.genome.max_conns,
    )
    numpy_config = (
        numpy_run.pipeline.backend,
        numpy_run.pipeline.seed,
        numpy_run.pipeline.algorithm.pop_size,
        numpy_run.pipeline.algorithm.genome.num_inputs,
        numpy_run.pipeline.algorithm.genome.num_outputs,
        numpy_run.pipeline.algorithm.genome.max_nodes,
        numpy_run.pipeline.algorithm.genome.max_conns,
    )
    assert iree_config == ("iree", SEED, POP_SIZE, 2, 1, MAX_NODES, MAX_CONNS), (
        "the iree fixture must be the shared tiny XOR configuration, got "
        f"{iree_config}"
    )
    assert numpy_config == ("numpy", SEED, POP_SIZE, 2, 1, MAX_NODES, MAX_CONNS), (
        "the numpy reference run must be the *same* configuration and seed, got "
        f"{numpy_config}"
    )

    for index, (iree_gen, numpy_gen) in enumerate(zip(iree_run.trace, numpy_run.trace)):
        (iree_payload, iree_fitness, iree_counter) = iree_gen
        (numpy_payload, numpy_fitness, numpy_counter) = numpy_gen

        assert iree_counter == numpy_counter, (
            f"generation counter diverged at step {index}: "
            f"iree {iree_counter} vs numpy {numpy_counter}"
        )

        for name, left, right in zip(PAYLOAD_FIELDS, iree_payload, numpy_payload):
            assert left.dtype == right.dtype and left.shape == right.shape, (
                f"{name}: iree {left.dtype}{left.shape} vs numpy "
                f"{right.dtype}{right.shape}"
            )
            if left.dtype.kind in "iu":
                np.testing.assert_array_equal(
                    left,
                    right,
                    err_msg=(
                        f"{name} at step {index} must be bit-identical across "
                        "backends: the key payload is produced by integer ops only "
                        "(randint/comparison/select), so any difference is a real "
                        "semantic divergence, not float reassociation"
                    ),
                )
            else:
                finite = left[np.isfinite(left)]
                assert finite.size > 0 and np.count_nonzero(finite) > 0, (
                    f"{name} at step {index} carries no finite, non-zero entry on "
                    "the iree side: comparing an all-padding payload would make "
                    "the allclose below vacuous"
                )
                np.testing.assert_allclose(
                    left,
                    right,
                    rtol=FLOAT_RTOL,
                    atol=FLOAT_ATOL,
                    equal_nan=True,  # NaN marks an unconnected node in both backends
                    err_msg=(
                        f"{name} at step {index}: iree and numpy must agree on the "
                        f"float32 payload (rtol={FLOAT_RTOL}, atol={FLOAT_ATOL}); "
                        "the tolerance covers iree's reduction reassociation only"
                    ),
                )

        np.testing.assert_allclose(
            iree_fitness,
            numpy_fitness,
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
            err_msg=(
                f"fitness at step {index} diverged: iree {iree_fitness!r} vs "
                f"numpy {numpy_fitness!r}"
            ),
        )
        assert len(np.unique(iree_fitness)) > 1 and len(np.unique(numpy_fitness)) > 1, (
            "both backends must evaluate a population with more than one distinct "
            f"fitness, otherwise the agreement above is trivial: iree "
            f"{iree_fitness!r} vs numpy {numpy_fitness!r}"
        )


def test_iree_rejects_an_unknown_compile_option_instead_of_falling_back() -> None:
    """An unsupported construct raises the backend's own error -- no fallback.

    Oracle: the evoxc option contract. ``Pipeline(compile_options=...)`` is
    expanded verbatim into ``evoxc.build``'s ``**options`` and every backend
    validates the keys against its own ``KNOWN_OPTIONS``; the numpy reference
    interpreter deliberately declares no validation.

    Assertions: with the *same* bogus option dictionary, ``backend="iree"``
    raises :class:`evoxc.core.BackendError` naming the option while
    ``backend="numpy"`` completes a generation. If the pipeline silently fell
    back from iree to numpy (or dropped the unknown key) the first half could
    not raise.

    Notes:
        The exception surfaces in ``setup()``, not in ``step()``: ``NEAT.setup``
        builds its own (genetic-operator) graph with the state's build options, so
        the first ``evoxc.build`` of a run happens there. Both calls are wrapped in
        one ``pytest.raises`` so the test stays about the error, not about which
        of the two graphs happens to be traced first.
    """
    iree_pipeline = _pipeline("iree", compile_options={UNKNOWN_OPTION: 1})
    with pytest.raises(BackendError) as excinfo:
        iree_state = iree_pipeline.setup()
        iree_pipeline.step(iree_state)
    message = str(excinfo.value)
    assert UNKNOWN_OPTION in message, (
        "the error must name the offending option so a user can fix it, got: "
        f"{message}"
    )
    assert "iree" in message, f"the error must come from the iree backend, got: {message}"

    numpy_pipeline = _pipeline("numpy", compile_options={UNKNOWN_OPTION: 1})
    numpy_state = numpy_pipeline.setup()
    _, _, fitnesses = numpy_pipeline.step(numpy_state)
    assert fitnesses.shape == (POP_SIZE,) and np.all(np.isfinite(fitnesses)), (
        "the numpy reference interpreter declares no option validation (a "
        "documented ignore), so the same dictionary must run there -- this is "
        "what proves the rejection above is iree's, not the pipeline's"
    )
