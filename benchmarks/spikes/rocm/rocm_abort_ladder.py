"""Minimal evoxc construct ladder for the ROCm/HIP ``hal.fence.await`` abort.

One probe per process (a failed HIP device is poisoned for the rest of the
process and the interpreter often hangs at teardown):

    timeout 600 uv run --no-sync python benchmarks/spikes/rocm/rocm_abort_ladder.py --probe p1
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

import evoxc
import evoxc.numpy as enp

DEV = evoxc.Device("rocm", 0)
CPU = evoxc.Device("cpu", 0)
SPEC = evoxc.TensorSpec


def _build(fn, *specs):
    return evoxc.build(
        fn, *specs, backend="iree", device=DEV, target_backends=["rocm"]
    )


def place(arr):
    return evoxc.core.Tensor(np.ascontiguousarray(arr)).to(DEV)


def host(value):
    if isinstance(value, evoxc.Tensor):
        return np.asarray(value.to(CPU).numpy())
    return np.asarray(value)


def _report(name, fn):
    print(f"PROBE {name} START", flush=True)
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
        first = traceback.format_exc().strip().splitlines()[-1]
        print(f"PROBE {name} FAIL {type(exc).__name__}: {exc} :: {first}", flush=True)
        return 1
    print(f"PROBE {name} OK {detail}", flush=True)
    return 0


def p0():
    """Control: elementwise + reduce on (512, 512)."""
    x = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)

    def fn(x):
        y = evoxc.multiply(x, x)
        return evoxc.sum(y, axes=(1,))

    exe = _build(fn, SPEC((512, 512), evoxc.float32))
    out = host(evoxc.run(exe, place(x)))
    return f"sum[:3]={out[:3].tolist()}"


def p1():
    """scan whose carry is seeded from a graph CONSTANT (enp.zeros)."""
    length = 20
    xs = np.arange(length, dtype=np.float32) + 1.0

    def fn(xs):
        init = enp.zeros((), evoxc.float32)

        def f(carry, x):
            new = evoxc.add(carry, x)
            return new, new

        carry, ys = evoxc.scan(f, init, xs)
        return carry, ys

    exe = _build(fn, SPEC((length,), evoxc.float32))
    carry, ys = evoxc.run(exe, place(xs))
    return f"carry={host(carry).item()} ys[-1]={host(ys)[-1].item()}"


def p2():
    """Same scan with an INPUT-derived zero carry."""
    length = 20
    xs = np.arange(length, dtype=np.float32) + 1.0

    def fn(xs):
        init = evoxc.multiply(evoxc.sum(xs, axis=0), enp.zeros((), evoxc.float32))

        def f(carry, x):
            new = evoxc.add(carry, x)
            return new, new

        carry, ys = evoxc.scan(f, init, xs)
        return carry, ys

    exe = _build(fn, SPEC((length,), evoxc.float32))
    carry, _ys = evoxc.run(exe, place(xs))
    return f"carry={host(carry).item()} (want {xs.sum()})"


def p3():
    """1-D constant-seeded carry (the speciate reserve scan shape)."""
    species_size = 20
    rows = np.linspace(-1.0, 1.0, species_size * 4, dtype=np.float32).reshape(
        species_size, 4
    )

    def fn(rows):
        init = enp.zeros((species_size,), evoxc.float32)

        def f(carry, row):
            carry = evoxc.add(carry, evoxc.sum(row, axis=0))
            return carry, carry

        carry, _ys = evoxc.scan(f, init, rows)
        return carry

    exe = _build(fn, SPEC((species_size, 4), evoxc.float32))
    out = host(evoxc.run(exe, place(rows)))
    return f"carry[:3]={out[:3].tolist()}"


def p4():
    """while_loop: i64 counter condition, input-derived carries (forward level loop)."""
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    def fn(x):
        init = (enp.zeros((), evoxc.int32), x)

        def cond(c):
            i, _v = c
            return evoxc.less(i, 5)

        def body(c):
            i, v = c
            return evoxc.add(i, enp.full((), 1, dtype=evoxc.int32)), evoxc.multiply(v, 1.01)

        _i, v = evoxc.while_loop(cond, body, init)
        return v

    exe = _build(fn, SPEC((64,), evoxc.float32))
    out = host(evoxc.run(exe, place(x)))
    return f"v[0]={out[0].item()}"


def p5():
    """while_loop whose condition reads a carried FLOAT (cycle check shape)."""
    x = np.linspace(0.1, 0.9, 64, dtype=np.float32)

    def fn(x):
        init = (enp.zeros((), evoxc.int32), evoxc.sum(x, axis=0), x)

        def cond(c):
            i, acc, _v = c
            return evoxc.logical_and(
                evoxc.less(i, 5),
                evoxc.greater(evoxc.subtract(acc, acc), -1.0),
            )

        def body(c):
            i, acc, v = c
            acc = evoxc.add(acc, evoxc.sum(v, axis=0))
            return evoxc.add(i, enp.full((), 1, dtype=evoxc.int32)), acc, evoxc.multiply(v, 1.01)

        _i, acc, v = evoxc.while_loop(cond, body, init)
        return acc, v

    exe = _build(fn, SPEC((64,), evoxc.float32))
    acc, v = evoxc.run(exe, place(x))
    return f"acc={host(acc).item()} v[0]={host(v)[0].item()}"


def p6():
    """scan with a WIDE carried tree + gather/select in the body (speciate scan)."""
    n_pop = 32
    n_slots = 8
    xs = np.arange(n_pop, dtype=np.float32) + 1.0
    idx = (np.arange(n_pop) % n_slots).astype(np.int32)

    def fn(xs, idx):
        zeros = enp.zeros((n_slots,), evoxc.float32)
        init = (
            enp.zeros((), evoxc.int32),
            zeros,
            enp.zeros((n_pop,), evoxc.float32),
            enp.zeros((n_pop,), evoxc.float32),
            zeros,
            zeros,
            zeros,
        )

        def f(carry, x):
            c0, sums, base, dist, slotv, a, b = carry
            step = evoxc.gather(idx, c0, axis=0)
            sel = evoxc.equal(step, enp.zeros((), evoxc.int32))
            upd = evoxc.add(evoxc.add(evoxc.multiply(base, 1.001), evoxc.subtract(x, x)), enp.zeros((n_pop,), evoxc.float32))
            base = evoxc.select(sel.reshape((1,)), upd, base)
            dist = evoxc.add(dist, evoxc.multiply(base, 0.5))
            sums = evoxc.add(sums, enp.zeros((n_slots,), evoxc.float32))
            return (evoxc.add(c0, enp.full((), 1, dtype=evoxc.int32)), sums, base, dist, slotv, a, b), evoxc.sum(
                dist, axis=0
            )

        carry, ys = evoxc.scan(f, init, (xs, idx))
        return carry[3], ys

    exe = _build(fn, SPEC((n_pop,), evoxc.float32), SPEC((n_pop,), evoxc.int32))
    dist, ys = evoxc.run(exe, place(xs), place(idx))
    return f"dist[0]={host(dist)[0].item()} ys[-1]={host(ys)[-1].item()}"


def p7():
    """sort / argsort / argmax / runtime-indexed gather (no control flow)."""
    n = 501
    fit = np.linspace(1.0, -1.0, n, dtype=np.float32)
    rows = np.arange(n * 8, dtype=np.float32).reshape(n, 8)

    def fn(fit, rows):
        order = evoxc.argsort(fit)
        best = evoxc.argmax(fit)
        picked = evoxc.gather(rows, order, axis=0)
        return order[:10], best, evoxc.sum(picked, axes=(1,))[:10]

    exe = _build(fn, SPEC((n,), evoxc.float32), SPEC((n, 8), evoxc.float32))
    order, best, sums = evoxc.run(exe, place(fit), place(rows))
    return f"order[:3]={host(order)[:3].tolist()} best={host(best).item()} sum0={host(sums)[0].item()}"


def p8():
    """gather with data-dependent indices + scatter (memory-safety suspect)."""
    n_pop = 500
    n_slots = 50
    rows = np.arange(n_pop, dtype=np.int32)
    keys = (np.arange(n_pop * 3, dtype=np.int32).reshape(n_pop, 3) % n_slots)

    def fn(rows, keys):
        gathered = evoxc.gather(keys, rows, axis=0)
        upd = evoxc.add(gathered, 1)
        base = enp.zeros((n_slots,), evoxc.int32)
        return evoxc.scatter(base, keys, upd, axis=0)

    exe = _build(fn, SPEC((n_pop,), evoxc.int32), SPEC((n_pop, 3), evoxc.int32))
    out = host(evoxc.run(exe, place(rows), place(keys)))
    return f"scatter[0]={int(out[0])}"



def p9():
    """while_loop with an i64 counter (the construct family evoxc documents as working)."""
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    def fn(x):
        init = (enp.zeros((), evoxc.int64), x)

        def cond(c):
            i, _v = c
            return evoxc.less(i, enp.full((), 5, dtype=evoxc.int64))

        def body(c):
            i, v = c
            return evoxc.add(i, enp.full((), 1, dtype=evoxc.int64)), evoxc.multiply(v, 1.01)

        _i, v = evoxc.while_loop(cond, body, init)
        return v

    exe = _build(fn, SPEC((64,), evoxc.float32))
    out = host(evoxc.run(exe, place(x)))
    return f"v[0]={out[0].item()}"


def p10():
    """while_loop with ONE 0-d f32 carry (no tensor payload at all)."""
    def fn(x):
        init = evoxc.sum(x, axis=0)

        def cond(c):
            return evoxc.less(c, 5.0)

        def body(c):
            return evoxc.add(c, 1.0)

        out = evoxc.while_loop(cond, body, init)
        return out

    exe = _build(fn, SPEC((16,), evoxc.float32))
    out = host(evoxc.run(exe, place(np.ones(16, np.float32))))
    return f"out={out.item()} (want 21)"


def p11():
    """p4 with an explicit int32 bound: no dtype mixing anywhere in the loop."""
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    def fn(x):
        init = (enp.zeros((), evoxc.int32), x)

        def cond(c):
            i, _v = c
            return evoxc.less(i, enp.full((), 5, dtype=evoxc.int32))

        def body(c):
            i, v = c
            return evoxc.add(i, enp.full((), 1, dtype=evoxc.int32)), evoxc.multiply(v, 1.01)

        _i, v = evoxc.while_loop(cond, body, init)
        return v

    exe = _build(fn, SPEC((64,), evoxc.float32))
    out = host(evoxc.run(exe, place(x)))
    return f"v[0]={out[0].item()}"


def p12():
    """scan with a TRIVIAL body: no arithmetic, carry passed through, y = x."""
    length = 20
    xs = np.arange(length, dtype=np.float32) + 1.0

    def fn(xs):
        init = enp.zeros((), evoxc.float32)

        def f(carry, x):
            return carry, x

        carry, ys = evoxc.scan(f, init, xs)
        return carry, ys

    exe = _build(fn, SPEC((length,), evoxc.float32))
    carry, ys = evoxc.run(exe, place(xs))
    return f"carry={host(carry).item()} ys[-1]={host(ys)[-1].item()}"


def p13():
    """p4 with an int64 bound on an int32 counter: the dtype-mixing hypothesis."""
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    def fn(x):
        init = (enp.zeros((), evoxc.int32), x)

        def cond(c):
            i, _v = c
            return evoxc.less(i, enp.full((), 5, dtype=evoxc.int64))

        def body(c):
            i, v = c
            return evoxc.add(i, enp.full((), 1, dtype=evoxc.int32)), evoxc.multiply(v, 1.01)

        _i, v = evoxc.while_loop(cond, body, init)
        return v

    exe = _build(fn, SPEC((64,), evoxc.float32))
    out = host(evoxc.run(exe, place(x)))
    return f"v[0]={out[0].item()}"


def p14():
    """while_loop with a 0-d gather inside the body (the scan stack read)."""
    length = 20
    xs = np.arange(length, dtype=np.float32) + 1.0

    def fn(xs):
        init = (enp.zeros((), evoxc.int32), enp.zeros((), evoxc.float32))

        def cond(c):
            i, _acc = c
            return evoxc.less(i, enp.full((), 5, dtype=evoxc.int32))

        def body(c):
            i, acc = c
            step = evoxc.gather(xs, i, axis=0)
            return evoxc.add(i, enp.full((), 1, dtype=evoxc.int32)), evoxc.add(acc, step)

        i, acc = evoxc.while_loop(cond, body, init)
        return i, acc

    exe = _build(fn, SPEC((length,), evoxc.float32))
    i, acc = evoxc.run(exe, place(xs))
    return f"i={host(i).item()} acc={host(acc).item()}"


def p15():
    """while_loop with a scatter into a carried stack (the scan stack write)."""
    length = 8
    xs = np.arange(length, dtype=np.float32) + 1.0

    def fn(xs):
        stack0 = enp.zeros((length,), evoxc.float32)
        init = (enp.zeros((), evoxc.int32), stack0)

        def cond(c):
            i, _s = c
            return evoxc.less(i, enp.full((), 5, dtype=evoxc.int32))

        def body(c):
            i, stack = c
            step = evoxc.gather(xs, i, axis=0)
            row = evoxc.reshape(step, (1,))
            stack = evoxc.scatter(stack, i, row, axis=0)
            return evoxc.add(i, enp.full((), 1, dtype=evoxc.int32)), stack

        _i, stack = evoxc.while_loop(cond, body, init)
        return stack

    exe = _build(fn, SPEC((length,), evoxc.float32))
    out = host(evoxc.run(exe, place(xs)))
    return f"stack={out.tolist()}"


PROBES = {"p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
          "p6": p6, "p7": p7, "p8": p8, "p9": p9, "p10": p10, "p11": p11, "p12": p12, "p13": p13, "p14": p14, "p15": p15}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, choices=sorted(PROBES))
    args = parser.parse_args()

    import iree.runtime as rt

    devices = rt.get_driver("hip").query_available_devices()
    print(f"PROBE {args.probe} devices={devices}", flush=True)

    code = _report(args.probe, PROBES[args.probe])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
