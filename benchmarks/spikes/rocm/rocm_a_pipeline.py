"""Task A: validate the Pipeline's ``device=`` placement on the real ROCm path.

One case per process (a failed HIP device is poisoned, and the interpreter can hang
at teardown), each driven from ``rocm_run_all.sh`` with ``timeout 900``:

    timeout 900 uv run --no-sync python benchmarks/spikes/rocm/rocm_a_pipeline.py --case a1

Cases:
  a1        the objective's literal spelling -- ``backend="iree"``, ``device="rocm"``,
            ``compile_options={"target_backends": ["rocm"]}``. Records where it stops:
            ``setup`` itself, because the setup graphs run with ``device=None`` but
            carry the rocm compile target.
  a1_infer  same pipeline with **no** ``compile_options`` (evoxc infers the target
            from each build's device): records the second stop, ``_place_state``'s
            cross-device copy of the setup graph's CPU payload tensors.
  a1_hop    the same placement applied by hand through an explicit host hop (what
            ``_place_state`` would need), then the timed build and one generation
            (expected to abort in the HAL).
  a1_numpy  ``backend="numpy"`` with ``device="rocm"``: placement succeeds, the
            first build refuses the device.
  a2numpy   ``backend="numpy"`` XOR pop 500 for 2 generations with ``device="cpu"``
            and with ``device=None``: fitness + payload must be bit-identical.
  a2iree    ``backend="iree"`` + ``target_backends=["llvm-cpu"]``, same comparison.
  a3        a rocm-resident state (:func:`rocm_state`) then ``_host_view(state)``:
            the reporting view must carry CPU-device tensors, ``.numpy()`` must not
            raise, and a real reporting graph (``_schedule_of``) must build and run
            through that view.

Every run prints one ``PROBE <name> <OK|FAIL> <detail>`` line per probe.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import evoxc

from rocm_support import (
    device_of,
    finish,
    hop_to_rocm,
    kind,
    make_pipeline,
    report,
    rocm_state,
)


# --------------------------------------------------------------------- A1


def a1() -> str:
    """The objective's literal spelling: setup fails on its own compile target."""
    p = make_pipeline(500, device="rocm", compile_options={"target_backends": ["rocm"]})
    start = time.perf_counter()
    try:
        p.setup()
    except Exception as exc:  # noqa: BLE001 - the finding, reported not raised
        elapsed = time.perf_counter() - start
        print(
            f"PROBE a1_setup FAILED {type(exc).__name__}: {str(exc)[:300]} after "
            f"{elapsed:.3f}s -- setup graphs register device=None but forward "
            f"compile_options={{'target_backends': ['rocm']}} verbatim",
            flush=True,
        )
        return "setup refuses: rocm-compiled setup artifact cannot run on cpu:0"
    raise AssertionError("expected setup to fail with target_backends=['rocm']")


def a1_infer() -> str:
    """No compile_options: setup gets further, then placement itself refuses."""
    p = make_pipeline(500, device="rocm")
    start = time.perf_counter()
    try:
        state = p.setup()
    except Exception as exc:  # noqa: BLE001 - the finding, reported not raised
        elapsed = time.perf_counter() - start
        print(
            f"PROBE a1_infer_setup FAILED {type(exc).__name__}: {str(exc)[:300]} "
            f"after {elapsed:.3f}s -- the CPU setup graphs returned IREE cpu:0 "
            f"payload tensors and _place_state copies them in one hop",
            flush=True,
        )
        return "setup builds+runs on cpu:0, then _place_state raises DeviceError"
    raise AssertionError("expected _place_state to refuse the cross-device copy")


def a1_hop() -> str:
    """Hand-applied host hop, then the timed generation build and one generation."""
    p = make_pipeline(500)
    start = time.perf_counter()
    state = p.setup()
    setup_seconds = time.perf_counter() - start
    print(
        f"PROBE a1_hop_setup OK setup={setup_seconds:.3f}s (CPU local-task) "
        f"pop[0]={kind(state.pop[0])} randkey={kind(state.randkey)}",
        flush=True,
    )

    state = hop_to_rocm(state)
    print(f"PROBE a1_hop_place OK {device_of(state)}", flush=True)

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
        f"next_pop_device={kind(new_state.pop.node_keys)}",
        flush=True,
    )
    return "unreachable on this box"


def a1_numpy() -> str:
    """``backend="numpy"`` with ``device="rocm"``: where does it stop?"""
    p = make_pipeline(500, backend="numpy", device="rocm")
    state = p.setup()
    print(f"PROBE a1_numpy_setup OK {device_of(state)}", flush=True)
    p._executable_for(state)
    return "unreachable (numpy refuses a non-cpu device at build)"


# --------------------------------------------------------------------- A2


def _two_generations(backend: str, device, compile_options):
    """Run the same pipeline twice (``device`` as given) and return its history."""
    p = make_pipeline(
        500, backend=backend, device=device, compile_options=compile_options
    )
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
    """The host-reporting view of a device-resident state is CPU-resident."""
    p = make_pipeline(500)
    state = rocm_state(p)
    p.device = "rocm"
    print(f"PROBE a3_state OK {device_of(state)}", flush=True)

    view = p._host_view(state)
    kinds = {
        "pop[0]": kind(view.pop[0]),
        "pop[3]": kind(view.pop[3]),
        "randkey": kind(view.randkey),
        "generation": kind(view.generation),
        "species[0]": kind(evoxc.tree_leaves(view.species)[0]),
    }
    bad = {name: value for name, value in kinds.items() if value != "cpu"}
    if bad:
        raise AssertionError(f"host view is not CPU-resident: {bad}")
    if getattr(view, "device", "missing") is not None:
        raise AssertionError(f"host view must register device=None, got {view.device!r}")
    keys = np.asarray(view.pop.node_keys.numpy())
    randkey = np.asarray(view.randkey.numpy())
    print(
        f"PROBE a3_host_view OK kinds={kinds} device=None "
        f"node_keys{keys.shape} randkey{randkey.shape}",
        flush=True,
    )

    # The explicit D2H step the pipeline uses (``common.tools.to_numpy``).
    from tensorneat.common.tools import to_numpy

    direct = to_numpy(state.pop.node_keys)
    same = np.array_equal(direct, keys)
    print(
        f"PROBE a3_to_numpy OK to_numpy(rocm node_keys)->host{direct.shape} "
        f"identical_to_host_view={same}",
        flush=True,
    )

    # A real reporting graph through the view: the transform stage (schedule
    # derivation), from a rocm-resident population, on the CPU side-channel.
    payload = type(state.pop)(*(to_numpy(field) for field in state.pop))
    start = time.perf_counter()
    schedule = p._schedule_of(view, payload)
    elapsed = time.perf_counter() - start
    print(
        f"PROBE a3_schedule_of OK {elapsed:.3f}s "
        f"in_conn{tuple(int(s) for s in schedule.in_conn.shape)} "
        f"output_slot{tuple(int(s) for s in schedule.output_slot.shape)} "
        f"device={kind(schedule.in_conn)}",
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

    print(f"PROBE {args.case} devices={rt.get_driver('hip').query_available_devices()}",
          flush=True)
    finish(report(args.case, CASES[args.case]))


if __name__ == "__main__":
    main()
