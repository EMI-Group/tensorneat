"""Task A: validate the Pipeline's ``device=`` placement on the real ROCm path.

One case per process (a failed HIP device is poisoned, and the interpreter can hang
at teardown), each driven from ``rocm_run_all.sh`` with ``timeout 900``:

    timeout 900 uv run --no-sync python src/tensorneat/_rocm_spikes/rocm_a_pipeline.py --case a1

Cases:
  a1        the objective's literal spelling -- ``backend="iree"``, ``device="rocm"``,
            ``compile_options={"target_backends": ["rocm"]}``. Records that ``setup``
            itself is where it stops: the setup graphs run with ``device=None`` but
            carry the rocm compile target.
  a1_infer  same pipeline with **no** ``compile_options``: evoxc infers the target
            from each build's device, so setup runs on CPU and the generation graph
            compiles for rocm. Prints the device of every runtime tensor, the setup
            and build seconds, then runs one generation (expected to abort in the HAL).
  a2numpy   ``backend="numpy"`` XOR pop 500 for 2 generations with ``device="cpu"``
            and with ``device=None``: fitness + payload must be bit-identical.
  a2iree    ``backend="iree"`` + ``target_backends=["llvm-cpu"]``, same comparison.
  a3        ``device="rocm"`` setup then ``_host_view(state)``: the reporting view
            must carry CPU-device tensors, ``.numpy()`` must not raise, and a real
            reporting graph (``_schedule_of``: the transform stage) must build and
            run through that view.

Every run prints one ``PROBE <name> <OK|FAIL> <detail>`` line per probe.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

import evoxc

from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT

POP = 500
SPECIES = 20
SURVIVAL = 0.01
SEED = 42


def make_pipeline(backend: str, device, compile_options):
    """The exact xor_p500 geometry ``benchmarks/cases.py`` declares."""
    return Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=POP,
            species_size=SPECIES,
            survival_threshold=SURVIVAL,
            genome=genome.DefaultGenome(
                num_inputs=2,
                num_outputs=1,
                max_nodes=50,
                max_conns=100,
                output_transform=ACT.sigmoid,
            ),
        ),
        problem=problem.XOR(),
        seed=SEED,
        backend=backend,
        device=device,
        compile_options=compile_options,
    )


def report(name: str, fn) -> int:
    print(f"PROBE {name} START", flush=True)
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
        first = traceback.format_exc().strip().splitlines()[-1]
        print(f"PROBE {name} FAIL {type(exc).__name__}: {exc} :: {first}", flush=True)
        return 1
    print(f"PROBE {name} OK {detail}", flush=True)
    return 0


def _kind(leaf) -> str:
    """The device kind of one leaf, as ``Tensor.device`` spells it."""
    device = getattr(leaf, "device", None)
    return "host" if device is None else str(device.kind)


def _device_of(state) -> str:
    """Device kinds of the four runtime tensors (``_RUNTIME_KEYS``)."""
    pop = state.pop
    species = evoxc.tree_leaves(state.species)
    return (
        f"backend={state.backend} state.device={getattr(state, 'device', None)!r} "
        f"pop[0]={_kind(pop[0])} pop[3]={_kind(pop[3])} "
        f"randkey={_kind(state.randkey)} generation={_kind(state.generation)} "
        f"species[{len(species)} leaves]={_kind(species[0])}"
    )


# --------------------------------------------------------------------- A1


def a1() -> str:
    """The objective's literal spelling: setup must succeed, step must abort."""
    p = make_pipeline("iree", "rocm", {"target_backends": ["rocm"]})
    start = time.perf_counter()
    try:
        p.setup()
    except Exception as exc:  # noqa: BLE001 - the finding, reported not raised
        elapsed = time.perf_counter() - start
        print(
            f"PROBE a1_setup FAILED_FAST {type(exc).__name__}: {str(exc)[:400]} "
            f"after {elapsed:.3f}s -- the setup graphs register device=None but "
            f"forward compile_options={{'target_backends': ['rocm']}} verbatim",
            flush=True,
        )
        return (
            "setup itself refuses: rocm-compiled setup artifact cannot run on the "
            "device=None (cpu) setup device; see a1_infer for the placement path"
        )
    raise AssertionError("expected setup to fail with target_backends=['rocm']")


def _placement_then_generation(compile_options) -> str:
    """Setup, report placement, time the build, then run one generation."""
    p = make_pipeline("iree", "rocm", compile_options)
    start = time.perf_counter()
    state = p.setup()
    setup_seconds = time.perf_counter() - start
    print(
        f"PROBE a1_setup OK setup={setup_seconds:.3f}s {_device_of(state)} "
        f"pop_shapes={tuple(tuple(int(s) for s in f.shape) for f in state.pop)} "
        f"species_leaves={len(evoxc.tree_leaves(state.species))}",
        flush=True,
    )

    start = time.perf_counter()
    p._executable_for(state)
    elapsed = time.perf_counter() - start
    print(
        f"PROBE a1_build OK build={elapsed:.3f}s (pipeline build_seconds="
        f"{p.build_seconds:.3f}s) signature={p.shape_signature}",
        flush=True,
    )

    print("PROBE a1_step START (one full generation on rocm)", flush=True)
    start = time.perf_counter()
    new_state, _previous_pop, fitnesses = p.step(state)
    elapsed = time.perf_counter() - start
    print(
        f"PROBE a1_step OK step={elapsed:.3f}s fitness[:4]={fitnesses[:4].tolist()} "
        f"next_pop_device={_kind(new_state.pop.node_keys)}",
        flush=True,
    )
    return "unreachable on this box"


def a1_infer() -> str:
    """No compile_options: evoxc infers the target per build from the device."""
    return _placement_then_generation(None)


RUNTIME_KEYS = ("pop", "generation", "species", "randkey")


def hop_to_rocm(state, dev=None):
    """Place the four runtime tensors on rocm through an explicit host hop.

    ``Pipeline._place_state`` calls ``Tensor.to(rocm)`` directly on the tensors a
    CPU setup graph returned. Those are IREE *payload*-backed cpu:0 tensors, so
    ``evoxc`` refuses the cross-device copy (see the ``a1_infer`` result). The
    same tensors reach the GPU through an explicit host copy first, which is what
    this helper does -- and what ``_place_state`` would have to do.
    """
    dev = evoxc.Device("rocm", 0) if dev is None else dev

    def hop(leaf):
        """Payload-backed cpu:0 tensor -> fresh host array -> target device.

        ``leaf.to(cpu)`` is the identity on a cpu:0 tensor (``Tensor.to`` returns
        ``self`` when the device matches), so the host hop has to be an explicit
        ``.numpy()`` copy; only that ndarray-backed tensor can be uploaded.
        """
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


def a1_hop() -> str:
    """``_place_state``'s missing host hop, applied by hand, then one generation.

    ``Pipeline._place_state`` calls ``Tensor.to(rocm)`` directly on the tensors a
    CPU setup graph returned. Those are IREE *payload*-backed cpu:0 tensors, so
    ``evoxc`` refuses the cross-device copy (see the ``a1_infer`` result). The
    same tensors reach the GPU through the documented two-hop form
    ``leaf.to(cpu).to(rocm)``, which is what this case does before timing the
    build and running the generation.
    """
    p = make_pipeline("iree", None, None)
    start = time.perf_counter()
    state = p.setup()
    setup_seconds = time.perf_counter() - start
    print(
        f"PROBE a1_hop_setup OK setup={setup_seconds:.3f}s (CPU local-task) "
        f"pop[0]={_kind(state.pop[0])} randkey={_kind(state.randkey)}",
        flush=True,
    )

    state = hop_to_rocm(state)
    print(f"PROBE a1_hop_place OK {_device_of(state)}", flush=True)

    start = time.perf_counter()
    p._executable_for(state)
    elapsed = time.perf_counter() - start
    print(
        f"PROBE a1_hop_build OK build={elapsed:.3f}s signature={p.shape_signature}",
        flush=True,
    )

    print("PROBE a1_hop_step START (one full generation on rocm)", flush=True)
    start = time.perf_counter()
    new_state, _previous_pop, fitnesses = p.step(state)
    elapsed = time.perf_counter() - start
    print(
        f"PROBE a1_hop_step OK step={elapsed:.3f}s fitness[:4]={fitnesses[:4].tolist()} "
        f"next_pop_device={_kind(new_state.pop.node_keys)}",
        flush=True,
    )
    return "unreachable on this box"


def a1_numpy() -> str:
    """``backend="numpy"`` with ``device="rocm"``: where does it stop?"""
    p = make_pipeline("numpy", "rocm", None)
    state = p.setup()
    print(f"PROBE a1_numpy_setup OK {_device_of(state)}", flush=True)
    p._executable_for(state)
    return "unreachable (numpy refuses a non-cpu device at build)"


# --------------------------------------------------------------------- A2


def _two_generations(backend: str, device, compile_options):
    """Run the same pipeline twice (``device`` as given) and return its history."""
    p = make_pipeline(backend, device, compile_options)
    state = p.setup()
    fitness_rows = []
    pop_rows = []
    for _ in range(2):
        state, previous_pop, fitnesses = p.step(state)
        fitness_rows.append(np.asarray(fitnesses))
        pop_rows.append(tuple(np.asarray(f).copy() for f in previous_pop))
    return fitness_rows, pop_rows


def _bit_identical(left, right) -> bool:
    """``np.array_equal`` with ``equal_nan`` over two payload lists."""
    return len(left) == len(right) and all(
        np.array_equal(a, b, equal_nan=True) for a, b in zip(left, right)
    )


def a2(backend: str, compile_options) -> str:
    """``device="cpu"`` must be bit-identical to ``device=None``."""
    explicit_fit, explicit_pop = _two_generations(backend, "cpu", compile_options)
    implicit_fit, implicit_pop = _two_generations(backend, None, compile_options)
    fit_ok = all(
        np.array_equal(a, b, equal_nan=True) for a, b in zip(explicit_fit, implicit_fit)
    )
    pop_ok = all(_bit_identical(a, b) for a, b in zip(explicit_pop, implicit_pop))
    if not (fit_ok and pop_ok):
        raise AssertionError(
            f"device='cpu' differs from device=None: fitness_equal={fit_ok} "
            f"payload_equal={pop_ok}"
        )
    return (
        f"backend={backend} 2 generations bit-identical to device=None; "
        f"fitness[0][:4]={explicit_fit[0][:4].tolist()} "
        f"fitness[1][:4]={explicit_fit[1][:4].tolist()}"
    )


# --------------------------------------------------------------------- A3


def a3() -> str:
    """The host-reporting view of a device-resident state is CPU-resident.

    ``Pipeline.setup(device="rocm")`` cannot finish on this box (see ``a1_infer``),
    so the rocm-resident state is produced by the ``a1_hop`` route and the
    pipeline's placement device is set afterwards -- which is exactly the state
    ``_host_view`` sees after a successful setup.
    """
    p = make_pipeline("iree", None, None)
    state = p.setup()
    state = hop_to_rocm(state)
    p.device = "rocm"
    print(f"PROBE a3_state OK {_device_of(state)}", flush=True)

    view = p._host_view(state)
    kinds = {
        "pop[0]": _kind(view.pop[0]),
        "pop[3]": _kind(view.pop[3]),
        "randkey": _kind(view.randkey),
        "generation": _kind(view.generation),
        "species[0]": _kind(evoxc.tree_leaves(view.species)[0]),
    }
    bad = {name: kind for name, kind in kinds.items() if kind != "cpu"}
    if bad:
        raise AssertionError(f"host view is not CPU-resident: {bad}")
    if getattr(view, "device", "missing") is not None:
        raise AssertionError(f"host view must register device=None, got {view.device!r}")
    zeros = np.asarray(view.pop.node_keys.numpy())
    randkey = np.asarray(view.randkey.numpy())
    print(
        f"PROBE a3_host_view OK kinds={kinds} device=None "
        f"node_keys{zeros.shape} randkey{randkey.shape}",
        flush=True,
    )

    # The explicit D2H step the pipeline uses (``common.tools.to_numpy``).
    from tensorneat.common.tools import to_numpy

    keys = to_numpy(state.pop.node_keys)
    same = np.array_equal(keys, np.asarray(view.pop.node_keys.numpy()))
    print(
        f"PROBE a3_to_numpy OK to_numpy(rocm node_keys)->host{keys.shape} "
        f"identical_to_host_view={same}",
        flush=True,
    )

    # A real reporting graph through the view: the transform stage (schedule
    # derivation) on the host, from a rocm-resident population.
    payload = type(state.pop)(*(to_numpy(f) for f in state.pop))
    start = time.perf_counter()
    schedule = p._schedule_of(view, payload)
    elapsed = time.perf_counter() - start
    print(
        f"PROBE a3_schedule_of OK {elapsed:.3f}s "
        f"in_conn{tuple(int(s) for s in schedule.in_conn.shape)} "
        f"output_slot{tuple(int(s) for s in schedule.output_slot.shape)} "
        f"device={_kind(schedule.in_conn)}",
        flush=True,
    )
    return "host view CPU-resident, to_numpy D2H works, reporting transform runs"


CASES = {
    "a1": a1,
    "a1_infer": a1_infer,
    "a1_hop": a1_hop,
    "a1_numpy": a1_numpy,
    "a2numpy": lambda: a2("numpy", None),
    "a2iree": lambda: a2("iree", {"target_backends": ["llvm-cpu"]}),
    "a3": a3,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    args = parser.parse_args()

    import iree.runtime as rt

    devices = rt.get_driver("hip").query_available_devices()
    print(f"PROBE {args.case} devices={devices}", flush=True)

    code = report(args.case, CASES[args.case])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
