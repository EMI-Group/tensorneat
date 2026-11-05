"""Shared helpers for the ROCm spike scripts.

Nothing here touches ``tensorneat`` internals beyond the documented host API
(``Pipeline.setup``/``step``, ``_static_state``/``_host_view``) and ``evoxc``'s
public device API, because the spikes must not modify committed source.

Device discipline: a failed HIP device is poisoned for the rest of the process and
the interpreter often hangs at teardown, so every driver runs ONE graph per
subprocess (``timeout 900``) and ends with ``os._exit`` after flushing.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

import evoxc

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT

DEV = evoxc.Device("rocm", 0)
CPU = evoxc.Device("cpu", 0)
RUNTIME_KEYS = ("pop", "generation", "species", "randkey")

#: ``benchmarks/cases.py``'s xor_p500 / xor3d_p500 / custom_pagie_p500 geometry.
XOR_GENOME = dict(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100,
                  output_transform=ACT.sigmoid)


def report(name: str, fn) -> int:
    """Run one probe, print ``PROBE <name> <OK|FAIL> <detail>``, return the code."""
    print(f"PROBE {name} START", flush=True)
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
        first = traceback.format_exc().strip().splitlines()[-1]
        print(f"PROBE {name} FAIL {type(exc).__name__}: {exc} :: {first}", flush=True)
        return 1
    print(f"PROBE {name} OK {detail}", flush=True)
    return 0


def finish(code: int) -> None:
    """Flush both streams and leave the process without running teardown."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def kind(leaf) -> str:
    """The device kind of one leaf (``host`` for a plain ndarray)."""
    device = getattr(leaf, "device", None)
    return "host" if device is None else str(device.kind)


def device_of(state) -> str:
    """Device kinds of the four runtime tensors."""
    pop = state.pop
    species = evoxc.tree_leaves(state.species)
    return (
        f"backend={state.backend} state.device={getattr(state, 'device', None)!r} "
        f"pop[0]={kind(pop[0])} pop[3]={kind(pop[3])} randkey={kind(state.randkey)} "
        f"generation={kind(state.generation)} species[{len(species)}]={kind(species[0])}"
    )


def make_pipeline(
    pop_size: int = 500,
    *,
    num_inputs: int = 2,
    num_outputs: int = 1,
    max_nodes: int = 50,
    max_conns: int = 100,
    species_size: int = 20,
    survival_threshold: float = 0.01,
    problem_obj=None,
    seed: int = 42,
    backend: str = "iree",
    device=None,
    compile_options=None,
) -> Pipeline:
    """A Pipeline at one ``benchmarks/cases.py`` geometry."""
    return Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=pop_size,
            species_size=species_size,
            survival_threshold=survival_threshold,
            genome=genome.DefaultGenome(
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                max_nodes=max_nodes,
                max_conns=max_conns,
                output_transform=ACT.sigmoid,
            ),
        ),
        problem=problem.XOR() if problem_obj is None else problem_obj,
        seed=seed,
        backend=backend,
        device=device,
        compile_options=compile_options,
    )


def hop_to_rocm(state, dev=DEV):
    """Place the four runtime tensors on rocm through an explicit host hop.

    ``Pipeline._place_state`` calls ``Tensor.to(rocm)`` directly on the tensors a
    CPU setup graph returned. Those are IREE *payload*-backed ``cpu:0`` tensors, so
    ``evoxc`` refuses the cross-device copy ("cross-device copies are not
    implemented in v1"). The same tensors reach the GPU through an explicit host
    copy first, which is what this helper does -- and what ``_place_state`` would
    have to do for an ``iree`` setup.
    """
    def hop(leaf):
        """Payload-backed ``cpu:0`` tensor -> fresh host array -> target device."""
        if not isinstance(leaf, evoxc.Tensor) or leaf.device == dev:
            return leaf
        shape = tuple(int(size) for size in leaf.shape)
        # ``.numpy()`` on a cpu:0 IREE payload works; ``ascontiguousarray`` would
        # promote a rank-0 array to shape (1,), so the declared shape is restored.
        host = np.ascontiguousarray(np.array(leaf.numpy())).reshape(shape)
        return evoxc.core.Tensor(host).to(dev)

    placed = {
        name: evoxc.tree_map(hop, getattr(state, name))
        for name in RUNTIME_KEYS
        if name in state.registered_keys()
    }
    return state.update(device=dev, **placed)


def rocm_state(p: Pipeline):
    """A rocm-resident state for ``p``: CPU setup, then the explicit hop."""
    state = p.setup()
    return hop_to_rocm(state)


def place(array, dev=DEV):
    """Upload a host array as a tensor on ``dev``."""
    return evoxc.core.Tensor(np.ascontiguousarray(array)).to(dev)


def host(value) -> np.ndarray:
    """Explicit device-to-host copy of one tensor (or a passthrough ndarray)."""
    if isinstance(value, evoxc.Tensor):
        return np.asarray(value.to(CPU).numpy())
    return np.asarray(value)


def build(fn, specs, backend: str = "iree", device=DEV, target_backends=("rocm",)):
    """One ``evoxc.build`` with the spike's backend/device/target spelling."""
    options = {"backend": backend, "device": device}
    if target_backends:
        options["target_backends"] = list(target_backends)
    return evoxc.build(fn, *specs, **options)


def spec_of(leaf):
    """``TensorSpec`` of an existing tensor."""
    return evoxc.TensorSpec(tuple(int(s) for s in leaf.shape), leaf.dtype)
