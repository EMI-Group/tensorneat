"""Unit tests for the Pipeline's device-placement layer (``device=`` end to end).

What this module proves
-----------------------
``Pipeline(..., device=...)`` runs its setup graphs on the CPU and then places the
runtime state on the requested device once, keeping it device-resident across
generations.  No GPU is needed here: the placement contract is exercised against
the ``cpu`` device (whose transfer is the identity) plus a *registered fake kind*,
which is exactly the seam evoxc exposes for host-only tests
(``evoxc.core.register_device_transfer_provider``).

The behaviours pinned, one test each:

* :meth:`Pipeline._placement_device` resolves ``None``, a kind string and an
  ``evoxc.Device`` the way ``evoxc.build`` does, so a caller never has to know
  which spelling the compile site accepts;
* :meth:`Pipeline._host_build_options` strips exactly the *device-targeting*
  option keys (``device``, ``target_backends``) and preserves every other one —
  an artifact compiled for an accelerator cannot run on the CPU device the
  setup graphs run on;
* ``setup()`` hands the setup phase ``device=None`` and the stripped options, and
  restores the caller's real values on the returned state (spied by a subclass
  that records and then aborts, so nothing is compiled with GPU options);
* :meth:`Pipeline._place_state` is the identity for ``device=None`` and for an
  already-correct ``cpu`` device (the same tensor objects come back — no copy),
  while a cross-kind move uploads every runtime leaf through host memory
  (evoxc refuses a payload-to-device transfer), keeps a rank-0 ``randkey``
  rank-0, and leaves every non-tensor leaf alone;
* :meth:`Pipeline._host_view` is the CPU side-channel: ``device=None`` plus
  cpu-kind leaves for a device-resident state, and the very same state object
  when no device was requested;
* :func:`tensorneat.common.tools.to_numpy` is the single explicit D2H step: a
  non-cpu tensor is transferred to ``Device("cpu", 0)`` *before* ``.numpy()``,
  and a CPU tensor is read without any transfer;
* ``device="cpu"`` and ``device=None`` runs are bit-identical over several
  generations and keep one compiled executable per shape.

Oracles: values, object identity and the recorded transfer calls — never the
implementation's own docstrings.  ``numpy``-backend only, no optional extras.
"""

from __future__ import annotations

import numpy as np
import pytest

import evoxc

from tensorneat import Pipeline, algorithm, common, genome, problem
from tensorneat.common.tools import to_numpy
from tensorneat.pipeline import _DEVICE_TARGETING_OPTIONS, _RUNTIME_KEYS

#: The static payload shapes of the tiny pipeline (mirrors the integration suite).
MAX_NODES = 8
MAX_CONNS = 16
POP_SIZE = 16


def _pipeline(*, device=None, compile_options=None, backend="numpy"):
    """A configured (never run) XOR pipeline on the numpy backend."""
    gene = genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        max_in_degree=4,
        max_depth=8,
        output_transform=common.ACT.sigmoid,
    )
    algo = algorithm.NEAT(gene, pop_size=POP_SIZE, species_size=4, survival_threshold=0.01)
    return Pipeline(
        algo,
        problem.XOR(error_method="mse"),
        seed=0,
        fitness_target=-0.01,
        generation_limit=60,
        backend=backend,
        device=device,
        compile_options=compile_options,
    )


def _runtime_leaves(state):
    """``(name, leaf)`` for every tensor leaf of the runtime keys, in state order."""
    out = []
    for name in _RUNTIME_KEYS:
        if name not in state.registered_keys():
            continue
        for index, leaf in enumerate(evoxc.tree_leaves(getattr(state, name))):
            out.append((f"{name}[{index}]", leaf))
    return out


# --------------------------------------------------------------------------- #
# device resolution and setup-phase options
# --------------------------------------------------------------------------- #
def test_placement_device_resolves_like_evoxc_build():
    assert _pipeline(device=None)._placement_device() is None

    cpu = _pipeline(device="cpu")._placement_device()
    assert cpu == evoxc.Device("cpu", 0) and cpu.kind == "cpu"

    cuda = _pipeline(device="cuda")._placement_device()
    assert cuda == evoxc.Device("cuda", 0) and cuda.kind == "cuda"

    explicit = evoxc.Device("rocm", 1)
    assert _pipeline(device=explicit)._placement_device() is explicit


def test_host_build_options_strip_only_device_targeting_keys():
    options = {
        "target_backends": ["rocm"],
        "device": evoxc.Device("rocm", 0),
        "iree_compile_args": ["--iree-hal-target-device=hip"],
        "opt_level": 2,
    }
    host = _pipeline(device="rocm", compile_options=options)._host_build_options()

    assert set(host) == set(options) - set(_DEVICE_TARGETING_OPTIONS)
    assert host["iree_compile_args"] == options["iree_compile_args"]
    assert host["opt_level"] == 2

    assert _pipeline(compile_options=None)._host_build_options() is None
    assert _pipeline(compile_options={})._host_build_options() == {}


class _SetupAborted(Exception):
    """Sentinel raised by the spy so no graph is built with GPU options."""


class _SetupSpy(algorithm.NEAT):
    """Record the state the setup graph would see, then abort before building."""

    seen = None

    def setup(self, state=None):
        type(self).seen = (state.device, dict(state.compile_options or {}))
        raise _SetupAborted


def test_setup_registers_cpu_device_and_stripped_options_then_restores_them():
    gene = genome.DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=MAX_NODES,
                                max_conns=MAX_CONNS)
    spy = _SetupSpy(gene, pop_size=POP_SIZE, species_size=4)
    options = {"target_backends": ["rocm"], "iree_compile_args": ["--flag"]}
    pipeline = Pipeline(
        spy,
        problem.XOR(),
        seed=0,
        backend="numpy",
        device="rocm",
        compile_options=options,
    )

    with pytest.raises(_SetupAborted):
        pipeline.setup()

    device, seen_options = _SetupSpy.seen
    assert device is None
    assert "target_backends" not in seen_options
    assert seen_options["iree_compile_args"] == ["--flag"]


def test_setup_restores_the_requested_configuration_on_the_state():
    pipeline = _pipeline(device="cpu", compile_options={"opt_level": 1})
    state = pipeline.setup()

    assert state.device == "cpu"
    assert state.compile_options == {"opt_level": 1}
    assert state.backend == "numpy"


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #
def test_place_state_without_a_device_returns_the_same_state_and_tensors():
    pipeline = _pipeline(device=None)
    state = pipeline.setup()

    placed = pipeline._place_state(state)

    assert placed is state
    for (_, before), (_, after) in zip(_runtime_leaves(state), _runtime_leaves(placed)):
        assert before is after


def test_place_state_on_the_cpu_device_copies_nothing():
    pipeline = _pipeline(device="cpu")
    state = pipeline.setup()

    placed = pipeline._place_state(state)

    before = dict(_runtime_leaves(state))
    after = dict(_runtime_leaves(placed))
    assert set(before) == set(after)
    for name in before:
        assert before[name] is after[name], f"{name} was copied"
        assert after[name].device == evoxc.Device("cpu", 0)


class _Mark:
    """What the fake provider returns: the uploaded host array plus its device."""

    def __init__(self, array, device):
        self.array = array
        self.device = device


def _register_fake_provider():
    """Register a ``fakedev`` transfer provider and return the uploads it saw."""
    uploads = []

    def provider(tensor, device):
        assert isinstance(tensor.data, np.ndarray), "the provider gets a host tensor"
        assert tensor.device == evoxc.Device("cpu", 0)
        array = tensor.numpy()
        assert array.flags["C_CONTIGUOUS"], "an upload must be contiguous"
        uploads.append((array.shape, array.copy()))
        return _Mark(array.copy(), device)

    evoxc.core.register_device_transfer_provider("fakedev", provider)
    return uploads


def test_place_state_crosses_devices_through_host_memory():
    uploads = _register_fake_provider()
    pipeline = _pipeline(device="fakedev")
    state = pipeline.setup()

    # setup() placed every runtime leaf exactly once (4 payload arrays, the
    # randkey, the generation counter and the species leaves).
    placed = pipeline._place_state(state)  # a second placement adds nothing
    expected = len(_runtime_leaves(placed))
    assert len(uploads) == expected
    assert expected >= 4 + 2 + 1

    leaves = dict(_runtime_leaves(placed))
    for name, leaf in leaves.items():
        assert isinstance(leaf, _Mark), f"{name} was not placed"
        assert leaf.device == evoxc.Device("fakedev", 0)

    # The rank of every leaf survives the host hop (np.ascontiguousarray would
    # promote a 0-d array to (1,) and break evoxc.random.split).
    reference = _pipeline(device=None).setup()
    wanted = {name: to_numpy(leaf) for name, leaf in _runtime_leaves(reference)}
    for name, leaf in leaves.items():
        assert leaf.array.shape == wanted[name].shape, f"{name} lost its rank"
        np.testing.assert_array_equal(leaf.array, wanted[name])

    ranks = {name: leaf.array.ndim for name, leaf in leaves.items()}
    assert ranks["randkey[0]"] == 0
    assert ranks["generation[0]"] == 0
    assert ranks["pop[0]"] == 2


def test_place_state_keeps_the_configuration_leaves_on_the_host():
    _register_fake_provider()
    pipeline = _pipeline(device="fakedev")
    state = pipeline.setup()

    # Only the runtime keys are walked: the configuration stays exactly as the
    # pipeline declared it, and no marker leaked into a non-tensor entry.
    assert state.device == "fakedev"
    assert state.backend == "numpy"
    assert state.compile_options is pipeline.compile_options
    if "pop_size" in state.registered_keys():
        assert state.pop_size == POP_SIZE


# --------------------------------------------------------------------------- #
# host view and the explicit device-to-host step
# --------------------------------------------------------------------------- #
def test_host_view_is_the_same_state_without_a_device():
    pipeline = _pipeline(device=None)
    state = pipeline.setup()

    assert pipeline._host_view(state) is state


def test_host_view_moves_a_device_state_to_the_cpu_side_channel():
    pipeline = _pipeline(device="cpu")
    state = pipeline.setup()

    view = pipeline._host_view(state)

    assert view is not state
    assert view.device is None
    for name, leaf in _runtime_leaves(view):
        assert leaf.device == evoxc.Device("cpu", 0), name
    # show/checkpoint read the view through to_numpy without a transfer.
    for (name, leaf), (_, source) in zip(_runtime_leaves(view), _runtime_leaves(state)):
        np.testing.assert_array_equal(to_numpy(leaf), to_numpy(source))


class _DeviceTensor:
    """A tensor stand-in whose payload lives on a non-cpu device."""

    def __init__(self, array, kind="cuda"):
        self._array = np.asarray(array)
        self.device = evoxc.Device(kind, 0)
        self.transfers = []

    def to(self, device):
        self.transfers.append(device)
        return evoxc.from_numpy(self._array)

    def numpy(self):  # pragma: no cover - the transfer must happen first
        raise AssertionError("to_numpy called .numpy() on device memory")


def test_to_numpy_transfers_to_host_before_materializing():
    payload = _DeviceTensor([1.0, 2.0, 3.0])

    out = to_numpy(payload)

    assert payload.transfers == [evoxc.Device("cpu", 0)]
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_to_numpy_does_not_transfer_a_cpu_tensor():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = evoxc.from_numpy(array)

    out = to_numpy(tensor)

    np.testing.assert_array_equal(out, array)
    assert out is array or np.shares_memory(out, array)


# --------------------------------------------------------------------------- #
# end-to-end equivalence of the cpu device and the default path
# --------------------------------------------------------------------------- #
def test_cpu_device_and_default_path_are_bit_identical():
    default = _pipeline(device=None)
    placed = _pipeline(device="cpu")

    default_state = default.setup()
    placed_state = placed.setup()

    for _ in range(3):
        default_state, _, default_fitness = default.step(default_state)
        placed_state, _, placed_fitness = placed.step(placed_state)
        np.testing.assert_array_equal(placed_fitness, default_fitness)

    np.testing.assert_array_equal(
        to_numpy(placed_state.generation), to_numpy(default_state.generation)
    )
    for (name, left), (_, right) in zip(
        _runtime_leaves(placed_state), _runtime_leaves(default_state)
    ):
        np.testing.assert_array_equal(to_numpy(left), to_numpy(right), err_msg=name)

    # One compiled executable per shape on both paths: a repeated step never
    # rebuilds, and the cpu device must not add a second build.
    assert default.builds == 1
    assert placed.builds == 1
