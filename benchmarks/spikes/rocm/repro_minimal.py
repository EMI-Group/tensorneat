"""Minimal evoxc-level reproducer of the ROCm/HIP runtime abort in the TensorNEAT generation graph.

The TensorNEAT generation graph on ``backend="iree", target_backends=["rocm"]`` dies at
``evoxc.run`` with

    RuntimeError: Error invoking function: c/runtime/src/iree/hal/drivers/hip/
    event_semaphore.c:786: ABORTED; the semaphore was aborted; while invoking
    native function hal.fence.await

The whole graph is not needed to reproduce it. Two `evoxc.while_loop` graphs that differ
**only in the dtype of the loop counter's comparison** separate the failure from the
construct:

    variant "abort"   int32 counter, condition `less(counter, 5)` with a Python int
                      literal (widened to int64 in the condition region)  -> ABORTS
    variant "ok"      the identical loop with int32/int32 or int64/int64   -> RUNS

and an `evoxc.scan` -- even one whose body does no work at all (the constructs
TensorNEAT uses for speciation, and the desugaring target of every `scan`) -- fails at a
different, earlier point:

    variant "scan"    trivial scan, body = (carry, x)                      -> FAILS
                      (two signatures observed for the same graph:
                      stream_command_buffer.c:452 INVALID_ARGUMENT
                      hipErrorInvalidValue ... hal.device.queue.alloca, and
                      event_semaphore.c:786 ABORTED ... hal.fence.await)

Usage (one graph per process -- a failed HIP device poisons the process and the
interpreter can hang at teardown, so every variant ends with ``os._exit``):

    export PATH=$HOME/.local/bin:$PATH
    uv sync --all-extras && uv pip install -e /root/evoxc --no-deps
    for v in abort ok scan; do
        timeout 900 uv run --no-sync python benchmarks/spikes/rocm/repro_minimal.py --variant $v
    done

Expected on AMD MI300X VF / ROCm 7.1 / iree-base-runtime 3.11.0:
``abort`` -> RuntimeError hal.fence.await ABORTED, ``ok``/``ok_i64`` -> OK, ``scan``
-> RuntimeError at the alloca or the fence (both signatures are the same graph).

Measured with the local evoxc checkout; the failure is in iree's HIP HAL, so evoxc's own
version does not enter into it (only the graph shape does). Every HIP runtime flag tried
(`--hip_use_streams=true/false`, `--hip_allow_inline_execution=true`,
`--hip_async_allocations=false`, via ``EVOXC_IREE_RUNTIME_ARGS``) leaves all three
unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

import evoxc
import evoxc.numpy as enp

DEVICE = evoxc.Device("rocm", 0)
HOST = evoxc.Device("cpu", 0)
F32 = evoxc.float32
I32 = evoxc.int32
I64 = evoxc.int64


def _place(array):
    """Host data -> device memory (evoxc never transfers implicitly)."""
    return evoxc.core.Tensor(np.ascontiguousarray(array)).to(DEVICE)


def _run(fn, specs, *inputs):
    executable = evoxc.build(
        fn, *specs, backend="iree", device=DEVICE, target_backends=["rocm"]
    )
    out = evoxc.run(executable, *(_place(value) for value in inputs))
    return evoxc.tree_map(lambda leaf: leaf.to(HOST).numpy(), out)


def loop(counter_dtype, bound_kind):
    """A 5-iteration while_loop carrying a ``(64,)`` float array.

    Mirrors the genome's forward level loop (``genome/default.py``: an int counter
    carried beside the values, condition on the counter, body multiplying the values).
    ``bound_kind`` selects the comparison's right-hand operand: a Python int literal
    (promoted, the failing form), or an int32/int64 graph constant (the clean forms).
    Every constant is created INSIDE the traced function -- outside a trace, tensor
    creation is an error.
    """
    def fn(x):
        if bound_kind == "literal":
            bound = 5
        elif bound_kind == "int32":
            bound = enp.full((), 5, dtype=I32)
        else:
            bound = enp.full((), 5, dtype=I64)
        init = (enp.zeros((), counter_dtype), x)

        def cond(carry):
            counter, _values = carry
            return evoxc.less(counter, bound)

        def body(carry):
            counter, values = carry
            one = enp.full((), 1, dtype=counter_dtype)
            return evoxc.add(counter, one), evoxc.multiply(values, 1.01)

        _counter, values = evoxc.while_loop(cond, body, init)
        return values

    return fn


def variant_abort():
    """int32 counter compared against a Python int literal -> widened cond."""
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    return _run(loop(I32, "literal"), (evoxc.TensorSpec((64,), F32),), x)


def variant_ok():
    """The same loop, condition operands both int32: no promotion in the cond region."""
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    fn = loop(I32, "int32")
    return _run(fn, (evoxc.TensorSpec((64,), F32),), x)


def variant_ok_i64():
    """The same loop with an int64 counter: also clean (the shape evoxc documents)."""
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    return _run(loop(I64, "int64"), (evoxc.TensorSpec((64,), F32),), x)


def variant_scan():
    """A trivial ``evoxc.scan``: no arithmetic, carry passed through, output = x."""
    xs = np.arange(20, dtype=np.float32) + 1.0

    def fn(xs):
        init = enp.zeros((), F32)

        def step(carry, x):
            return carry, x

        carry, ys = evoxc.scan(step, init, xs)
        return carry, ys

    return _run(fn, (evoxc.TensorSpec((20,), F32),), xs)


VARIANTS = {
    "abort": variant_abort,
    "ok": variant_ok,
    "ok_i64": variant_ok_i64,
    "scan": variant_scan,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    args = parser.parse_args()

    print(f"REPRO {args.variant} START", flush=True)
    code = 0
    try:
        out = VARIANTS[args.variant]()
    except Exception as exc:  # noqa: BLE001 - the reproducer reports, never raises
        frame = traceback.format_exc().strip().splitlines()[-1]
        print(f"REPRO {args.variant} FAIL {type(exc).__name__}: {exc}", flush=True)
        print(f"REPRO {args.variant} FRAME {frame}", flush=True)
        code = 1
    else:
        leaves = [np.asarray(leaf) for leaf in evoxc.tree_leaves(out)]
        print(
            f"REPRO {args.variant} OK "
            + " ".join(f"leaf{index}={leaf.ravel()[:2].tolist()}" for index, leaf in enumerate(leaves)),
            flush=True,
        )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
