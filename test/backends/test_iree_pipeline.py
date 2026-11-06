"""The ``backend="iree"`` tests: a small XOR pipeline really runs on the compiled backend.

What this module proves, on one tiny XOR configuration (``pop_size=8``,
``max_nodes=8``, ``max_conns=16``, ``seed=42``, defined once in
``_tiny_xor.py`` and shared with the ROCm leg) that compiles in a few seconds:

1. ``Pipeline(..., backend="iree")`` executes ``setup()`` plus two real
   generations and returns a *populated* payload: ``pop_size`` individuals, a
   finite fitness per individual, an advanced ``randkey``/``generation`` counter
   and exactly one ``evoxc.build`` (the second step reuses the cached
   executable).
2. The same seed on ``backend="numpy"`` produces the *same run*: the int32 key
   payloads and the generation counters are bit-identical, while every float32
   value agrees to :data:`FLOAT_RTOL` / :data:`FLOAT_ATOL` -- never to ``==``,
   because the compiled leg reassociates float reductions (see the comment on
   those constants).
3. A construct the backend cannot take raises the backend's own error instead of
   silently falling back: an unknown ``compile_options`` key is *rejected by the
   iree backend* (``evoxc.core.errors.BackendError`` naming the option) while the
   numpy reference interpreter -- which documents no option validation -- accepts
   the very same dictionary. That contrast is what proves the pipeline forwards
   options verbatim and never retries elsewhere.

Cost on IREE 3.13.0 (llvm-cpu, this box): ``setup()`` is ~0.9 s, the first
``step()`` (which pays the ``evoxc`` build/compile/load) ~5.9 s, matching
``pipeline.build_seconds`` ~5.9 s, and every later generation ~2 ms. The three
tests share one iree and one numpy run through module-scoped fixtures, so the
compile happens once and the module costs ~7 s.

Run it:

    uv sync --extra iree       # IREE 3.13.0 (iree-base-compiler + iree-base-runtime)
    pytest -q test/backends    # or: pytest -q -m requires_iree

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

import numpy as np
import pytest

pytest.importorskip(
    "iree",
    reason=(
        "structural optional-dependency skip: the compiled-backend suite needs "
        "the optional iree extra, which is not installed; install it with "
        "`uv sync --extra iree` (IREE 3.13.0; without uv the equivalent is `pip "
        'install "iree-base-compiler>=3.13,<3.14" '
        '"iree-base-runtime>=3.13,<3.14"`). A default or whole-tree run skips '
        "this module on purpose; an explicit `pytest -m requires_iree` run fails "
        "in ./conftest.py instead of skipping silently."
    ),
)

from evoxc.core import BackendError  # noqa: E402  (after the skip guard)

from backends._tiny_xor import (  # noqa: E402  (after the skip guard)
    MAX_CONNS,
    MAX_NODES,
    POP_SIZE,
    SEED,
    UNKNOWN_OPTION,
    assert_populated,
    assert_same_run,
    drive,
    run_configuration,
    tiny_xor_pipeline,
)

pytestmark = pytest.mark.requires_iree

#: Relative tolerance for float32 values across backends. iree's llvm-cpu target
#: reassociates floating-point reductions (a sum over a matrix column is a
#: different tree than numpy's pairwise loop), and each reassociation can move a
#: float32 result by one ulp (~1.19e-7 at unit magnitude) per reordering, so the
#: error grows with the reduction depth but stays within a small multiple of eps.
#: Measured on this box (IREE 3.13.0, seeds 42/43/44/7/123/2024, both generations,
#: every float payload field plus the fitness vector): max |iree - numpy| =
#: 2.3841858e-07, exactly 2 ulp. 1e-5 is ~42x that, which leaves room for a
#: box- or build-dependent reordering while still failing on any real divergence
#: (a wrong operand, a dropped edge or a changed activation moves values by
#: >> 1e-2).
FLOAT_RTOL = 1e-5
#: Absolute floor for the same comparison: ``node_attrs``/``conn_attrs`` can sit
#: near zero, where a pure relative bound is meaningless. 1e-6 is 4.2x the
#: largest difference measured here (2.4e-7) and still ~1 % of the smallest
#: meaningful weight.
FLOAT_ATOL = 1e-6


@pytest.fixture(scope="module")
def iree_run():
    """One iree run of the shared configuration (the compile happens here once)."""
    return drive(tiny_xor_pipeline("iree"))


@pytest.fixture(scope="module")
def numpy_run():
    """The numpy run of the *same* configuration and seed, to compare against."""
    return drive(tiny_xor_pipeline("numpy"))


def test_iree_step_produces_a_finite_population(iree_run) -> None:
    """Two real iree generations yield a populated, finite, advanced state.

    Oracle: the configuration itself -- ``pop_size`` individuals each carrying a
    finite fitness out of ``problem.XOR`` and a four-array payload of the static
    shape the genome declares.
    """
    assert_populated(iree_run, label="the iree backend")


def test_iree_matches_numpy_on_the_same_seed(iree_run, numpy_run) -> None:
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
    # A comparison is only meaningful between the same run configuration, so pin
    # the values both fixtures were built from before looking at a number.
    assert run_configuration(iree_run) == (
        "iree",
        SEED,
        POP_SIZE,
        2,
        1,
        MAX_NODES,
        MAX_CONNS,
    ), (
        "the iree fixture must be the shared tiny XOR configuration, got "
        f"{run_configuration(iree_run)}"
    )
    assert run_configuration(numpy_run) == (
        "numpy",
        SEED,
        POP_SIZE,
        2,
        1,
        MAX_NODES,
        MAX_CONNS,
    ), (
        "the numpy reference run must be the *same* configuration and seed, got "
        f"{run_configuration(numpy_run)}"
    )

    assert_same_run(
        numpy_run,
        iree_run,
        reference_label="numpy",
        other_label="the iree backend",
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
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
    iree_pipeline = tiny_xor_pipeline("iree", compile_options={UNKNOWN_OPTION: 1})
    with pytest.raises(BackendError) as excinfo:
        iree_state = iree_pipeline.setup()
        iree_pipeline.step(iree_state)
    message = str(excinfo.value)
    assert UNKNOWN_OPTION in message, (
        "the error must name the offending option so a user can fix it, got: "
        f"{message}"
    )
    assert "iree" in message, f"the error must come from the iree backend, got: {message}"

    numpy_pipeline = tiny_xor_pipeline("numpy", compile_options={UNKNOWN_OPTION: 1})
    numpy_state = numpy_pipeline.setup()
    _, _, fitnesses = numpy_pipeline.step(numpy_state)
    assert fitnesses.shape == (POP_SIZE,) and np.all(np.isfinite(fitnesses)), (
        "the numpy reference interpreter declares no option validation (a "
        "documented ignore), so the same dictionary must run there -- this is "
        "what proves the rejection above is iree's, not the pipeline's"
    )
