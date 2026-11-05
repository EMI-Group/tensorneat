"""Task C: per-kernel cost of the shipped stages at each case's real geometry.

No full generation runs on ROCm today (``rocm_b_stages.py``: every stage that
contains a real ``while_loop``/``scan`` region aborts or segfaults), so this script
measures the **un-looped** building blocks at the exact shapes the generation graph
uses and leaves the extrapolation to ``ROCM_FINDINGS.md``.

    timeout 900 uv run --no-sync python src/tensorneat/_rocm_spikes/rocm_c_kernels.py \
        --kernel level_step --case xor_p500 --backend rocm --repeats 10

Kernels
  level_step        one level step of the genome window reduce, mirroring
                    ``genome/step.py::_node_step``'s op mix (flat gather, ``xv == xv``
                    liveness, weight multiply, K-fold reduce, bias add, ``any`` mask,
                    select) at ``(rows, max_nodes, max_in_degree)``.
  level_step_x3     three chained level steps in one graph = a depth-3 forward's
                    level work without the ``while_loop`` around it.
  distance_row      the real speciation distance: ``genome.prepare_distance`` once,
                    then one population-shaped ``genome.measure_distance`` row.
  forward_unrolled  ``queries`` unrolled ``forward_prepared`` calls on a
                    CPU-derived schedule (the RL per-query shape). Needs a working
                    level ``while_loop``, so it is a numpy-only anchor.

Cases are the ``benchmarks/cases.py`` rows; ``rows = pop * samples`` is the tiled
batch width and a width above ``EVAL_CHUNK_ROWS`` (= 12 500) is measured one chunk
at a time, exactly like ``host_payload._prepared_query``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import evoxc
import evoxc.numpy as enp

from tensorneat import problem as problem_mod
from tensorneat.genome.utils import GenomeTensors

from rocm_support import (
    DEV,
    build,
    finish,
    host,
    make_pipeline,
    place,
    report,
    rocm_state,
    spec_of,
)

#: ``host_payload.EVAL_CHUNK_ROWS``: the width a tiled evaluate is cut into.
EVAL_CHUNK_ROWS = 12_500
#: Whole-generation executed-op counts from ``benchmarks/PERF_NUMPY.md`` pass 2.
GENERATION_OPS = {
    "xor_p500": 8_263,
    "custom_pagie_p500": 11_490,
    "cartpole_p100": 31_604,
}

CASES = {
    # name: (pop, samples per individual, num_inputs, num_outputs, max_nodes, K)
    "xor_p100": dict(pop=100, samples=4, num_inputs=2, num_outputs=1, chunk=True),
    "xor_p500": dict(pop=500, samples=4, num_inputs=2, num_outputs=1, chunk=True),
    "xor3d_p500": dict(pop=500, samples=8, num_inputs=3, num_outputs=1, chunk=True),
    "custom_pagie_p500": dict(pop=500, samples=500, num_inputs=2, num_outputs=1,
                              chunk=True),
    "cartpole_p100": dict(pop=100, samples=1, num_inputs=4, num_outputs=2, chunk=True),
}
MAX_NODES = 50
MAX_IN_DEGREE = 4
BACKEND = "rocm"
REPEATS = 10


def _shape(tensor) -> str:
    return str(tuple(int(size) for size in tensor.shape))


def _median_seconds(fn, repeats: int = REPEATS, warmup: int = 3):
    """``(value, median_seconds, min_seconds)`` over ``warmup + repeats`` runs."""
    value = None
    for _ in range(warmup):
        value = fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return value, samples[len(samples) // 2], samples[0]


def _build(kernel, specs):
    if BACKEND == "numpy":
        return build(kernel, specs, backend="numpy", device=None, target_backends=None)
    return build(kernel, specs)


def _arg(array):
    """One graph argument on the selected backend's device."""
    if isinstance(array, (evoxc.Tensor, evoxc.SymbolicTensor)):
        return array
    if BACKEND == "numpy":
        return evoxc.core.Tensor(np.ascontiguousarray(array))
    return place(array)


def _problem(name: str):
    """The problem object of one case (the genome's input width must agree)."""
    if name.startswith("xor3d"):
        return problem_mod.XOR3d()
    if name.startswith("custom_pagie"):
        return problem_mod.CustomFuncFit(
            method="sample",
            num_samples=500,
            low_bounds=[-1.0, -1.0],
            upper_bounds=[1.0, 1.0],
            func=lambda point: 1.0 / (1.0 + point[0] ** -4) + 1.0 / (1.0 + point[1] ** -4),
        )
    if name.startswith("cartpole"):
        return problem_mod.CartPole(max_step=200, repeat_times=1)
    return problem_mod.XOR()


def _geometry(name: str):
    """``(rows, chunks, chunk_rows)`` of one case's tiled evaluate."""
    case = CASES[name]
    rows = case["pop"] * case["samples"]
    chunk_rows = min(rows, EVAL_CHUNK_ROWS)
    chunks = rows // chunk_rows
    return rows, chunks, chunk_rows


# ------------------------------------------------------------------ kernels


def _level_step(values, flat_index, present, weights, bias, mask, rows):
    """One level step, mirroring ``genome/step.py::_node_step``'s op mix.

    ``xv = gather(values, flat_index)``; ``live = present & (xv == xv)``;
    ``msg = xv * weight``; ``agg = sum(msg)`` over the K window; ``z = bias + agg``;
    ``values = select(mask & any(live), z, values)``.
    """
    xv = evoxc.gather(evoxc.reshape(values, (rows * MAX_NODES,)), flat_index, axis=0)
    live = evoxc.logical_and(present, evoxc.equal(xv, xv))
    messages = evoxc.multiply(xv, weights)
    agg = evoxc.sum(messages, axes=(2,))
    z = evoxc.add(bias, agg)
    any_slot = evoxc.greater(evoxc.reduce_max(evoxc.cast(live, evoxc.int32), axes=(2,)), 0)
    update = evoxc.logical_and(mask, any_slot)
    return evoxc.select(update, z, values)


def _level_inputs(rows: int):
    """The ``(rows, N, K)`` operands of one level step, with live/dense slots."""
    rng = np.random.default_rng(7)
    values = rng.standard_normal((rows, MAX_NODES)).astype(np.float32)
    flat_index = rng.integers(0, rows * MAX_NODES, (rows, MAX_NODES, MAX_IN_DEGREE))
    present = rng.random((rows, MAX_NODES, MAX_IN_DEGREE)) > 0.5
    weights = rng.standard_normal((rows, MAX_NODES, MAX_IN_DEGREE)).astype(np.float32)
    bias = rng.standard_normal((rows, MAX_NODES)).astype(np.float32)
    mask = rng.random((rows, MAX_NODES)) > 0.3
    return (
        values,
        flat_index.astype(np.int32),
        present,
        weights,
        bias,
        mask,
    )


def _specs(rows: int):
    return [
        evoxc.TensorSpec((rows, MAX_NODES), evoxc.float32),
        evoxc.TensorSpec((rows, MAX_NODES, MAX_IN_DEGREE), evoxc.int32),
        evoxc.TensorSpec((rows, MAX_NODES, MAX_IN_DEGREE), evoxc.bool_),
        evoxc.TensorSpec((rows, MAX_NODES, MAX_IN_DEGREE), evoxc.float32),
        evoxc.TensorSpec((rows, MAX_NODES), evoxc.float32),
        evoxc.TensorSpec((rows, MAX_NODES), evoxc.bool_),
    ]


def level_step(name: str) -> str:
    """One level step at the case's real chunk width."""
    rows, chunks, chunk_rows = _geometry(name)
    arrays = _level_inputs(chunk_rows)

    def kernel(values, flat_index, present, weights, bias, mask):
        return _level_step(values, flat_index, present, weights, bias, mask, chunk_rows)

    exe, build_seconds = _timed_build(kernel, _specs(chunk_rows))
    args = [_arg(a) for a in arrays]
    _value, median, best = _median_seconds(lambda: evoxc.run(exe, *args))
    elements = chunk_rows * MAX_NODES * MAX_IN_DEGREE
    gb = elements * 4 / 1e9
    return (
        f"level_step rows={chunk_rows} chunks={chunks} elements={elements} "
        f"build={build_seconds:.3f}s median={median * 1e3:.3f}ms best={best * 1e3:.3f}ms "
        f"per_step_GBps={gb / median:.2f} whole_batch={median * chunks * 1e3:.3f}ms"
    )


def level_step_x3(name: str) -> str:
    """Three chained level steps: a depth-3 forward's level work, no ``while_loop``."""
    rows, chunks, chunk_rows = _geometry(name)
    arrays = _level_inputs(chunk_rows)

    def kernel(values, flat_index, present, weights, bias, mask):
        for _ in range(3):
            values = _level_step(
                values, flat_index, present, weights, bias, mask, chunk_rows
            )
        return values

    exe, build_seconds = _timed_build(kernel, _specs(chunk_rows))
    args = [_arg(a) for a in arrays]
    _value, median, best = _median_seconds(lambda: evoxc.run(exe, *args))
    return (
        f"level_step_x3 rows={chunk_rows} chunks={chunks} build={build_seconds:.3f}s "
        f"median={median * 1e3:.3f}ms best={best * 1e3:.3f}ms "
        f"per_level={median * 1e3 / 3:.3f}ms whole_batch={median * chunks * 1e3:.3f}ms"
    )


def _timed_build(kernel, specs):
    """``(executable, seconds)`` of one build."""
    start = time.perf_counter()
    exe = _build(kernel, specs)
    return exe, time.perf_counter() - start


def distance_row(name: str) -> str:
    """The real speciation distance: one digest, one population-shaped row."""
    case = CASES[name]
    p = make_pipeline(
        case["pop"],
        num_inputs=case["num_inputs"],
        num_outputs=case["num_outputs"],
        problem_obj=_problem(name),
    )
    state = p.setup() if BACKEND == "numpy" else rocm_state(p)
    cfg = p.algorithm._static_state(state)
    specs = [spec_of(field) for field in state.pop]

    def kernel(*arrays):
        payload = GenomeTensors(*arrays)
        digest = p.algorithm.genome.prepare_distance(cfg, payload)
        row = p.algorithm.genome.measure_distance(cfg, payload, digest)
        return [row, evoxc.sum(row, axes=(0,))]

    exe, build_seconds = _timed_build(kernel, specs)
    args = [_arg(field) for field in state.pop]
    out, median, best = _median_seconds(lambda: evoxc.run(exe, *args))
    return (
        f"distance_row pop={case['pop']} build={build_seconds:.3f}s "
        f"median={median * 1e3:.3f}ms best={best * 1e3:.3f}ms "
        f"digest_plus_one_row (speciate needs species_size*2 = 40 rows) "
        f"rowsum={float(host(out[1])):.3f}"
    )


def distance_digest(name: str) -> str:
    """``genome.prepare_distance`` alone: the once-per-generation digest."""
    case = CASES[name]
    p = make_pipeline(
        case["pop"],
        num_inputs=case["num_inputs"],
        num_outputs=case["num_outputs"],
        problem_obj=_problem(name),
    )
    state = p.setup() if BACKEND == "numpy" else rocm_state(p)
    cfg = p.algorithm._static_state(state)
    specs = [spec_of(field) for field in state.pop]

    def kernel(*arrays):
        payload = GenomeTensors(*arrays)
        digest = p.algorithm.genome.prepare_distance(cfg, payload)
        return [leaf for leaf in evoxc.tree_leaves(digest) if _is_float(leaf)]

    exe, build_seconds = _timed_build(kernel, specs)
    args = [_arg(field) for field in state.pop]
    _out, median, best = _median_seconds(lambda: evoxc.run(exe, *args))
    return (
        f"distance_digest pop={case['pop']} build={build_seconds:.3f}s "
        f"median={median * 1e3:.3f}ms best={best * 1e3:.3f}ms"
    )


def _is_float(leaf) -> bool:
    """True for a float leaf (the digest's integer key arrays are not returned)."""
    return str(leaf.dtype).startswith("float")


def forward_unrolled(name: str, queries: int = 20) -> str:
    """``queries`` unrolled ``forward_prepared`` calls on a CPU-derived schedule."""
    case = CASES[name]
    p = make_pipeline(
        case["pop"],
        num_inputs=case["num_inputs"],
        num_outputs=case["num_outputs"],
        problem_obj=_problem(name),
    )
    state = p.setup()
    p.device = None
    cfg = p.algorithm._static_state(state)
    payload = GenomeTensors(*(host(field) for field in state.pop))
    schedule = p._schedule_of(state, payload)
    leaves = evoxc.tree_leaves(schedule)
    structure = evoxc.tree_structure(schedule)
    specs = [spec_of(leaf) for leaf in leaves]
    specs.extend(spec_of(field) for field in state.pop)
    specs.append(evoxc.TensorSpec((case["pop"], case["num_inputs"]), evoxc.float32))
    inputs = np.zeros((case["pop"], case["num_inputs"]), dtype=np.float32)

    def kernel(*arrays):
        sched = evoxc.tree_unflatten(list(arrays[: len(leaves)]), structure)
        body = GenomeTensors(*arrays[len(leaves) : -1])
        prepared = p.algorithm.prepare_forward(cfg, sched, body, 1)
        total = None
        for _ in range(queries):
            out = p.algorithm.forward_prepared(cfg, sched, prepared, arrays[-1])
            total = out if total is None else evoxc.add(total, out)
        return total

    exe, build_seconds = _timed_build(kernel, specs)
    args = [_arg(root) for root in leaves] + [_arg(field) for field in state.pop]
    args.append(_arg(inputs))
    _out, median, best = _median_seconds(lambda: evoxc.run(exe, *args), repeats=3)
    return (
        f"forward_unrolled pop={case['pop']} queries={queries} "
        f"build={build_seconds:.3f}s median={median * 1e3:.3f}ms "
        f"best={best * 1e3:.3f}ms per_query={median * 1e3 / queries:.4f}ms "
        f"scaled_200={median * 1e3 / queries * 200:.2f}ms"
    )


KERNELS = {
    "level_step": level_step,
    "level_step_x3": level_step_x3,
    "distance_row": distance_row,
    "distance_digest": distance_digest,
    "forward_unrolled": lambda name: forward_unrolled(name),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, choices=sorted(KERNELS))
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--backend", default="rocm", choices=("rocm", "numpy"))
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    global BACKEND, REPEATS
    BACKEND, REPEATS = args.backend, args.repeats
    if BACKEND == "rocm":
        import iree.runtime as rt

        print(
            f"PROBE {args.kernel} devices="
            f"{rt.get_driver('hip').query_available_devices()}",
            flush=True,
        )
    print(
        f"PROBE {args.kernel} case={args.case} backend={BACKEND} repeats={REPEATS}",
        flush=True,
    )
    name = f"{args.kernel}_{args.case}"
    finish(report(name, lambda: KERNELS[args.kernel](args.case)))


if __name__ == "__main__":
    main()
