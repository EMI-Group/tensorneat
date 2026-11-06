"""The ROCm/HIP leg: the same tiny XOR pipeline really runs on an AMD GPU.

What this module proves, on the *same* tiny XOR configuration the llvm-cpu leg
uses (``pop_size=8``, ``max_nodes=8``, ``max_conns=16``, ``seed=42``, two
generations; all of it defined once in ``_tiny_xor.py``):

1. ``Pipeline(..., backend="iree", device="rocm",
   compile_options={"target_backends": ["rocm"]})`` executes ``setup()`` plus two
   real generations and returns a *populated* payload (the shared
   ``assert_populated`` set), **and the generation really ran on the GPU**: every
   ``evoxc.Tensor`` leaf of the runtime state reports ``device.kind == "rocm"``
   and ``state.device == "rocm"``. A silent CPU fallback would leave those leaves
   on ``cpu`` while the fitness vector still looked fine, which is exactly what
   this assertion rules out.
2. The same seed on ``backend="numpy"`` is the *same run*: int32 key payloads and
   generation counters bit-identical, float32 payloads and fitness within
   :data:`FLOAT_RTOL` / :data:`FLOAT_ATOL`, on both generations.

Measured on this box (AMD Instinct MI300X VF, IREE 3.13.0): ``setup()`` ~1.15 s,
``pipeline.builds == 1``, ``pipeline.build_seconds`` ~6.3 s, the first ``step()``
~6.3 s (it pays the compile) and the second ``step()`` ~0.004 s.
The worst |rocm - numpy| over seeds 42/43/44/7/123/2024 x both generations x every
float payload field and the fitness vector is **2.3841858e-07** -- exactly 2 ulp of
float32 (eps = 1.1920929e-07) -- while the int32 key payloads are bit-identical
and the NaN patterns agree on every seed. :data:`FLOAT_ATOL` = 1e-6 is 4.2x that
measured worst difference (:data:`FLOAT_RTOL` = 1e-5 is ~42x), which is enough
headroom for another box's reduction order and still far below any real
divergence (a wrong operand, a dropped edge or a changed activation moves values
by >> 1e-2).

Run it:

    uv sync --extra iree       # IREE 3.13.0, the first release whose HIP runtime
                               # executes loop-carried while/scan graphs
    pytest -q test/backends -m requires_rocm

Skip conditions (a module-level probe, applied as ``pytest.mark.skipif``): the
leg cannot run when ``/dev/kfd`` is absent, when ``iree.runtime.get_driver("hip")``
raises, or when the HIP driver reports zero devices; the reason names which of the
three is missing. The probe catches only around the driver query -- an error from
a test body itself always propagates. ``pytest.importorskip("iree")`` below is the
structural optional-dependency skip for the IREE wheels themselves.

``./conftest.py`` makes a requested-but-unavailable leg loud: an explicit
``-m requires_rocm`` (or ``-m requires_iree``) run in a checkout without ``iree``
exits non-zero with the install command instead of reporting a green "skipped".
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "iree",
    reason=(
        "structural optional-dependency skip: the ROCm leg needs the optional "
        "iree extra (the ROCm/HIP runtime is part of the IREE wheels), which is "
        "not installed; install it with `uv sync --extra iree` (IREE 3.13.0; "
        'without uv the equivalent is `pip install "iree-base-compiler>=3.13,<3.14" '
        '"iree-base-runtime>=3.13,<3.14"`). A default or whole-tree run skips '
        "this module on purpose; an explicit `pytest -m requires_iree` or `pytest "
        "-m requires_rocm` run fails in ./conftest.py instead of skipping "
        "silently."
    ),
)

import evoxc  # noqa: E402  (after the skip guard)
import iree.runtime  # noqa: E402  (after the skip guard)

from backends._tiny_xor import (  # noqa: E402  (after the skip guard)
    MAX_CONNS,
    MAX_NODES,
    POP_SIZE,
    SEED,
    assert_populated,
    assert_same_run,
    drive,
    run_configuration,
    tiny_xor_pipeline,
)

#: The device ``Pipeline`` places the runtime state on, and the compile options
#: that make ``evoxc`` emit a ROCm-targeted artifact. Both are forwarded verbatim
#: (``device`` to the pipeline, the options into ``evoxc.build``'s ``**options``);
#: nothing here falls back to CPU or to another target.
ROCM_DEVICE = "rocm"
ROCM_COMPILE_OPTIONS = {"target_backends": ["rocm"]}

#: The character device the AMD compute kernel driver exposes. Its absence means
#: the box has no ROCm kernel interface at all, independent of the IREE wheels.
KFD_DEVICE = Path("/dev/kfd")

#: Relative tolerance for float32 values between the ROCm leg and numpy. The HIP
#: target reassociates floating-point reductions (a column sum is a different tree
#: than numpy's pairwise loop) exactly as llvm-cpu does, so the error is a small
#: multiple of the float32 ulp: measured worst |rocm - numpy| = 2.3841858e-07
#: (2 ulp) over six seeds, both generations, every float payload field and the
#: fitness vector, so 1e-5 leaves ~42x headroom.
FLOAT_RTOL = 1e-5
#: Absolute floor for the same comparison, where ``node_attrs``/``conn_attrs`` sit
#: near zero and a relative bound says nothing: 1e-6 is 4.2x the measured worst
#: difference (2.4e-07) and still ~1 % of the smallest meaningful weight.
FLOAT_ATOL = 1e-6


def _rocm_unavailable_reason() -> str | None:
    """Why the ROCm leg cannot run here, or ``None`` when it can.

    Three independent preconditions, checked in the order that produces the most
    specific message: the kernel driver's character device, a queryable IREE HIP
    driver, and at least one HIP device behind it.

    Returns:
        str | None: ``None`` when the leg is runnable, otherwise a one-sentence
        reason naming exactly what is missing.

    Notes:
        The ``try`` covers the driver probe only, so an error raised by a test
        body is never mistaken for an environment problem. Both ``get_driver`` and
        ``query_available_devices`` are inside it because either failing means the
        HIP driver could not be queried; the message names the exception type and
        text so the missing piece is diagnosable from the skip line alone.
    """
    if not KFD_DEVICE.exists():
        return (
            f"the ROCm leg needs {KFD_DEVICE}, the AMD compute kernel driver's "
            "character device, and this box does not have it"
        )
    try:
        driver = iree.runtime.get_driver("hip")
        devices = driver.query_available_devices()
    except Exception as exc:  # the driver probe only -- never a test body
        return (
            "the IREE HIP driver could not be queried "
            f"({type(exc).__name__}: {exc}): the ROCm leg needs a working "
            "`iree.runtime.get_driver(\"hip\")` from the installed iree runtime"
        )
    if not devices:
        return (
            "the IREE HIP driver is present but reports zero devices, so there is "
            "no AMD GPU to run the generation on"
        )
    return None


#: ``None`` on a box that can run the leg; otherwise the skip reason. Computed
#: once at import, before any fixture is built.
ROCM_UNAVAILABLE = _rocm_unavailable_reason()

pytestmark = [
    pytest.mark.requires_iree,
    pytest.mark.requires_rocm,
    pytest.mark.skipif(ROCM_UNAVAILABLE is not None, reason=ROCM_UNAVAILABLE or ""),
]


@pytest.fixture(scope="module")
def rocm_run():
    """One ROCm run of the shared configuration (the compile happens here once)."""
    return drive(
        tiny_xor_pipeline(
            "iree",
            device=ROCM_DEVICE,
            compile_options=ROCM_COMPILE_OPTIONS,
        )
    )


@pytest.fixture(scope="module")
def numpy_run():
    """The numpy run of the *same* configuration and seed, to compare against."""
    return drive(tiny_xor_pipeline("numpy"))


def _tensor_device_kinds(state) -> tuple[set[str], int]:
    """The ``device.kind`` of every ``evoxc.Tensor`` leaf in ``state``, and how many.

    Walks every registered state entry with :func:`evoxc.tree_map`, so the walk
    follows the same pytree structure the pipeline hands to ``evoxc.run`` instead
    of a hand-listed field set. Non-tensor entries (the host configuration: the
    backend name, the compile-option dict, ...) are visited but contribute no
    kind, which is why the count is returned as well: a set can only be asserted
    ``== {"rocm"}`` if at least one tensor leaf was actually seen.

    Args:
        state: The runtime ``State`` of a finished run.

    Returns:
        tuple[set[str], int]: The distinct device kinds and the tensor-leaf count.
    """
    kinds: set[str] = set()
    count = 0

    def visit(leaf):
        nonlocal count
        if isinstance(leaf, evoxc.Tensor):
            count += 1
            kinds.add(leaf.device.kind)
        return leaf

    evoxc.tree_map(
        visit, {name: getattr(state, name) for name in state.registered_keys()}
    )
    return kinds, count


def test_rocm_step_produces_a_populated_gpu_run(rocm_run) -> None:
    """Two real ROCm generations are populated, finite and GPU-resident.

    Oracle: the configuration itself (through the shared ``assert_populated``:
    ``pop_size`` individuals, a finite fitness each, a static-shape four-array
    payload, an advanced ``randkey``/``generation``, exactly one build) plus the
    placement of the runtime state.

    Assertions: the shared populated set; then every ``evoxc.Tensor`` leaf of the
    final state reports ``device.kind == "rocm"`` (so a fallback to the host is
    impossible -- it would leave them on ``cpu``) and ``state.device`` is the
    string the pipeline was constructed with.
    """
    assert_populated(rocm_run, label="the ROCm leg")

    kinds, tensor_leaves = _tensor_device_kinds(rocm_run.state)
    assert tensor_leaves > 0, (
        "the ROCm run's state holds no evoxc.Tensor leaf at all, so the device "
        "assertion below would be vacuous"
    )
    assert kinds == {ROCM_DEVICE}, (
        "every runtime tensor of the ROCm run must live on the ROCm device: a "
        f"generation that fell back to the host would report 'cpu' here. Saw "
        f"{sorted(kinds)} over {tensor_leaves} tensor leaves; the state entries "
        f"are {sorted(rocm_run.state.registered_keys())}"
    )
    assert rocm_run.state.device == ROCM_DEVICE, (
        "the state must record the device the pipeline was constructed with, got "
        f"{rocm_run.state.device!r}"
    )

    assert run_configuration(rocm_run) == (
        "iree",
        SEED,
        POP_SIZE,
        2,
        1,
        MAX_NODES,
        MAX_CONNS,
    ), (
        "the ROCm fixture must be the shared tiny XOR configuration on the iree "
        f"backend, got {run_configuration(rocm_run)}"
    )


def test_rocm_matches_numpy_on_the_same_seed(rocm_run, numpy_run) -> None:
    """The same seed on ROCm and numpy is the same run, up to float reduction order.

    Oracle: the numpy reference interpreter on the identical configuration and
    seed.

    Assertions: both fixtures are pinned to the shared configuration first (the
    ROCm one on ``"iree"``, the reference on ``"numpy"``), then the shared
    ``assert_same_run`` compares *every* traced generation -- bit-identical
    integer key payloads and generation counters, float32 payloads and fitness
    within :data:`FLOAT_RTOL` / :data:`FLOAT_ATOL` with matching NaN patterns.
    """
    assert run_configuration(rocm_run) == (
        "iree",
        SEED,
        POP_SIZE,
        2,
        1,
        MAX_NODES,
        MAX_CONNS,
    ), (
        "the ROCm fixture must be the shared tiny XOR configuration on the iree "
        f"backend, got {run_configuration(rocm_run)}"
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
        rocm_run,
        reference_label="numpy",
        other_label="the ROCm leg",
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
    )
