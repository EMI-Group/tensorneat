#!/usr/bin/env python3
"""Probe the evoxc ``iree`` backend with the graphs the benchmark harness builds.

`BACKENDS.md` is the write-up; this file is the runnable evidence behind it.
Every mode prints one line (``OK ...`` or ``FAIL ...``) and exits 0/1, so
``--repeat N`` can drive N *fresh* interpreter processes and report the
pass/fail pattern of a non-deterministic failure.

Modes
-----
``sanity``      a 64x64 ``matmul`` graph: proves the iree/llvm-cpu stack works
                in this environment before blaming TensorNEAT for anything.
``scatter``     the minimal reproducer for the v1 stablehlo exporter gate: a
                rank-3 axis-0 row scatter (``species.py:295``'s construct).
``node-levels`` TensorNEAT's own ``common.tensorization.node_levels`` kernel --
                the 6-carry ``while_loop`` with ``select`` in the body that sits
                in every NEAT forward graph -- built standalone.
``while``       a synthetic ``while_loop`` with a chosen number of loop-carried
                tensors and a ``select``/plain body: isolates the construct the
                iree compiler crash correlates with, with no TensorNEAT code.
``pipeline``    the real benchmark case through ``Pipeline.setup``/``step`` on
                ``backend="iree"``, optionally with probe-only trace-time
                bypasses of the evoxc gates that would otherwise fire first.

Only stdlib is imported at module scope: ``evoxc``/``numpy``/``tensorneat`` are
imported inside the modes, so ``--help`` and the numpy-free harness path
(``python benchmarks/run_benchmark.py --list``) never depend on them.

Usage
-----
    python benchmarks/spikes/iree_probe.py --mode sanity
    python benchmarks/spikes/iree_probe.py --mode scatter --repeat 5
    python benchmarks/spikes/iree_probe.py --mode node-levels --max-nodes 8 --pop 4 --window 3 --repeat 5
    python benchmarks/spikes/iree_probe.py --mode pipeline --case xor_p100
    python benchmarks/spikes/iree_probe.py --mode pipeline --case xor_p100 --bypass scatter
    python benchmarks/spikes/iree_probe.py --mode pipeline --case xor_p100 --bypass scatter,uint32
    python benchmarks/spikes/iree_probe.py --mode while --carries 4 --body select --repeat 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

BENCH_DIR = Path(__file__).resolve().parents[1]
"""``benchmarks/``: ``cases.py`` lives here."""
REPO_ROOT = BENCH_DIR.parent
SRC = REPO_ROOT / "src"

BYPASSES = ("scatter", "uint32")
"""Probe-only trace-time bypasses for the ``pipeline`` mode.

``scatter`` rewrites a rank-3 axis-0 row scatter into the exporter-supported
rank-2 form on the flattened trailing dims (identical semantics);
``uint32`` drops iree's declared-dtype rejection of ``uint32`` so the next gate
becomes visible. Nothing here is a library change: both patches are installed on
the imported module objects by this process only.
"""


# --- shared helpers ---------------------------------------------------------------------


def _evoxc():
    """Import ``evoxc`` or exit with the install hint the mode needs."""
    try:
        import evoxc
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            f"{exc}: these probes need evoxc plus its iree extra "
            "(`uv sync --extra iree`, or `pip install "
            "\"evoxc[iree] @ git+https://github.com/EMI-Group/evoxc@v0.1.1\"`)"
        ) from exc
    return evoxc


def _first_line(exc: BaseException) -> str:
    """The exception's last non-empty traceback line, truncated for one-line output."""
    lines = [line.strip() for line in str(exc).strip().splitlines() if line.strip()]
    return lines[0][:240] if lines else type(exc).__name__


def _report(ok: bool, detail: str) -> int:
    """Print the single result line and return the process exit code."""
    print(f"{'OK' if ok else 'FAIL'} {detail}", flush=True)
    return 0 if ok else 1


# --- modes ------------------------------------------------------------------------------


def run_sanity() -> int:
    """Compile and run a 64x64 ``matmul`` graph on iree/llvm-cpu."""
    evoxc = _evoxc()
    import numpy as np

    @evoxc.defn
    def matmul(x, y):
        return evoxc.matmul(x, y)

    x = np.random.default_rng(0).random((64, 64), dtype=np.float32)
    y = np.random.default_rng(1).random((64, 64), dtype=np.float32)
    specs = [evoxc.TensorSpec((64, 64), evoxc.float32)] * 2
    try:
        start = time.perf_counter()
        exe = evoxc.build(matmul, *specs, backend="iree")
        compile_s = time.perf_counter() - start
        got = np.asarray(evoxc.run(exe, x, y).numpy())
    except Exception as exc:  # noqa: BLE001 - the probe reports, never swallows
        return _report(False, f"matmul/iree: {type(exc).__name__}: {_first_line(exc)}")
    err = float(np.max(np.abs(got - x @ y)))
    return _report(
        err < 1e-4, f"matmul/iree: compile_s={compile_s:.3f} max_abs_err={err:.2e}"
    )


def run_scatter() -> int:
    """Minimal reproducer: rank-3 axis-0 row scatter (``species.py:295``)."""
    evoxc = _evoxc()
    import numpy as np

    @evoxc.defn
    def seed_centre(dead, centre):
        # (species_size, max_nodes, width) <- (max_nodes, width) into row 0.
        return evoxc.scatter(dead, 0, centre)

    specs = [
        evoxc.TensorSpec((20, 50, 4), evoxc.float32),
        evoxc.TensorSpec((50, 4), evoxc.float32),
    ]
    dead = np.full((20, 50, 4), np.nan, np.float32)
    centre = np.zeros((50, 4), np.float32)
    ref = np.asarray(evoxc.run(evoxc.build(seed_centre, *specs, backend="numpy"), dead, centre).numpy())
    if not bool(np.all(ref[0] == 0)):
        return _report(False, "rank-3 scatter: numpy reference did not write row 0")
    try:
        exe = evoxc.build(seed_centre, *specs, backend="iree")
        evoxc.run(exe, dead, centre)
    except Exception as exc:  # noqa: BLE001 - the probe reports, never swallows
        return _report(
            False,
            f"rank-3 scatter/iree: {type(exc).__name__}: {_first_line(exc)}",
        )
    return _report(True, "rank-3 scatter/iree: built and ran")


def run_node_levels(max_nodes: int, pop: int, window: int) -> int:
    """Build TensorNEAT's ``node_levels`` kernel (a 6-carry while+select) on iree."""
    evoxc = _evoxc()
    import numpy as np

    sys.path.insert(0, str(SRC))
    from tensorneat.common.tensorization import node_levels

    n, p, k = max_nodes, pop, window
    i_inf = np.int32(2**30)

    @evoxc.defn
    def levels(in_conn, in_src, input_slot):
        return node_levels(in_conn, in_src, input_slot, n)

    in_conn = np.full((p, n, k), i_inf, dtype=np.int32)
    in_src = np.zeros((p, n, k), dtype=np.int32)
    for row in range(p):
        for slot in range(n):
            if slot < 2 or slot >= n - 1:  # two inputs, one output, hidden unwired
                continue
            sources = [s for s in range(slot) if s < 2 or in_conn[row, s, 0] != i_inf]
            for col, src in enumerate(sources[:k]):
                in_conn[row, slot, col] = 0
                in_src[row, slot, col] = src
    input_slot = np.tile(np.arange(2, dtype=np.int32), (p, 1))

    specs = [
        evoxc.TensorSpec((p, n, k), evoxc.int32),
        evoxc.TensorSpec((p, n, k), evoxc.int32),
        evoxc.TensorSpec((p, 2), evoxc.int32),
    ]
    try:
        start = time.perf_counter()
        exe = evoxc.build(levels, *specs, backend="iree")
        compile_s = time.perf_counter() - start
        got = np.asarray(evoxc.run(exe, in_conn, in_src, input_slot).numpy())
    except Exception as exc:  # noqa: BLE001 - the probe reports, never swallows
        return _report(
            False,
            f"node_levels/iree N={n} P={p} K={k}: "
            f"{type(exc).__name__}: {_first_line(exc)}",
        )
    ref = np.asarray(
        evoxc.run(
            evoxc.build(levels, *specs, backend="numpy"), in_conn, in_src, input_slot
        ).numpy()
    )
    return _report(
        bool(np.array_equal(got, ref)),
        f"node_levels/iree N={n} P={p} K={k}: compile_s={compile_s:.3f} "
        f"equal_numpy={bool(np.array_equal(got, ref))}",
    )


def run_while(carries: int, body: str) -> int:
    """Synthetic ``while_loop`` with a chosen number of loop-carried tensors.

    ``carries`` counts every carried tensor, the scalar counter included; the
    body is either ``select``-based or plain elementwise. This isolates the
    construct the iree compiler crash correlates with (carry count), separately
    from any TensorNEAT code.
    """
    evoxc = _evoxc()
    import numpy as np

    steps = 6
    data = carries - 1

    @evoxc.defn
    def looped(x, counter):
        def cond(c):
            return c[0] < steps

        def body_fn(c):
            nxt = [c[0] + 1.0]
            for k in range(1, data + 1):
                v = c[k]
                if body == "select":
                    nxt.append(
                        evoxc.select(
                            evoxc.less(v, 0.2),
                            evoxc.add(v, 0.01),
                            evoxc.multiply(v, 0.99),
                        )
                    )
                else:
                    nxt.append(evoxc.multiply(v, 0.9))
            return tuple(nxt)

        init = [counter] + [evoxc.add(x, float(k)) for k in range(data)]
        return evoxc.while_loop(cond, body_fn, tuple(init))

    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    counter = np.asarray(0.0, dtype=np.float32)
    specs = [evoxc.TensorSpec((256,), evoxc.float32), evoxc.TensorSpec((), evoxc.float32)]
    try:
        start = time.perf_counter()
        exe = evoxc.build(looped, *specs, backend="iree")
        compile_s = time.perf_counter() - start
        got = np.asarray(evoxc.run(exe, x, counter)[1].numpy())
    except Exception as exc:  # noqa: BLE001 - the probe reports, never swallows
        return _report(
            False,
            f"while/iree carried={carries} body={body}: "
            f"{type(exc).__name__}: {_first_line(exc)}",
        )
    ref = np.asarray(
        evoxc.run(evoxc.build(looped, *specs, backend="numpy"), x, counter)[1].numpy()
    )
    return _report(
        bool(np.allclose(got, ref, atol=1e-6)),
        f"while/iree carried={carries} body={body}: compile_s={compile_s:.3f} "
        f"equal_numpy={bool(np.allclose(got, ref, atol=1e-6))}",
    )


def _install_bypass(evoxc, names: list[str]) -> None:
    """Install the probe-only trace-time bypasses named in ``names``."""
    import numpy as np

    if "scatter" in names:
        original = evoxc.scatter

        def flattening_scatter(x, indices, updates, axis=0):
            """Rank-3 axis-0 row scatter -> rank-2 scatter on flattened rows."""
            if hasattr(x, "shape") and len(x.shape) == 3 and axis == 0:
                tail = int(np.prod(x.shape[1:]))
                flat = evoxc.reshape(x, (x.shape[0], tail))
                upd = evoxc.reshape(updates, (int(np.prod(updates.shape)),))
                return evoxc.reshape(original(flat, indices, upd, axis=0), x.shape)
            return original(x, indices, updates, axis)

        evoxc.scatter = flattening_scatter

    if "uint32" in names:
        from evoxc.backends.compiler import CompilerBackend

        original_check = CompilerBackend._check_value_dtype

        def lenient_check(self, dtype, where):
            """Ignore iree's declared-dtype rejection of ``uint32`` only."""
            if np.dtype(dtype) == np.dtype("uint32"):
                return
            return original_check(self, dtype, where)

        CompilerBackend._check_value_dtype = lenient_check


def run_pipeline(case_name: str, bypass: list[str], generations: int) -> int:
    """Build/run one real benchmark case on ``backend="iree"`` like the harness does."""
    evoxc = _evoxc()
    import numpy as np

    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(BENCH_DIR))
    if bypass:
        _install_bypass(evoxc, bypass)

    import tensorneat as tn

    from cases import get_cases

    case = {c.name: c for c in get_cases([case_name])}[case_name]
    algorithm, problem = case.build(tn, "new")
    pipeline = tn.Pipeline(
        algorithm,
        problem,
        seed=case.seed,
        fitness_target=case.fitness_target,
        generation_limit=case.generation_limit,
        backend="iree",
        device=None,
        compile_options=None,
        is_save=False,
        is_detailed=False,
    )
    label = f"pipeline {case_name}/iree" + (f" bypass={'+'.join(bypass)}" if bypass else "")
    try:
        start = time.perf_counter()
        state = pipeline.setup()
        setup_s = time.perf_counter() - start
        start = time.perf_counter()
        state, _, fitness = pipeline.step(state)
        first_s = time.perf_counter() - start
        best = float(np.max(fitness))
        for _ in range(generations - 1):
            state, _, fitness = pipeline.step(state)
    except Exception as exc:  # noqa: BLE001 - the probe reports, never swallows
        return _report(False, f"{label}: {type(exc).__name__}: {_first_line(exc)}")
    return _report(
        True,
        f"{label}: setup_s={setup_s:.3f} first_step_s={first_s:.3f} "
        f"best={best:.4f} final_best={float(np.max(fitness)):.4f}",
    )


# --- repeat driver ----------------------------------------------------------------------


def _without_repeat_flag(argv: list[str]) -> list[str]:
    """Drop ``--repeat N`` (or ``--repeat=N``) so the child runs exactly one probe."""
    forwarded: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--repeat":
            skip_next = True
            continue
        if token.startswith("--repeat="):
            continue
        forwarded.append(token)
    return forwarded


def _repeat(argv: list[str], repeats: int) -> int:
    """Run this script ``repeats`` times in fresh processes and summarise the pattern."""
    ok = 0
    codes: list[int] = []
    for index in range(1, repeats + 1):
        proc = subprocess.run(
            [sys.executable, *argv, "--repeat", "1"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": ""},
        )
        line = (proc.stdout.strip().splitlines() or [""])[-1]
        codes.append(proc.returncode)
        print(f"[{index}/{repeats}] exit={proc.returncode} {line}", flush=True)
        if proc.returncode == 0:
            ok += 1
    print(f"summary: ok={ok}/{repeats} codes={codes}", flush=True)
    return 0 if ok == repeats else 1


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the probe."""
    parser = argparse.ArgumentParser(
        prog="benchmarks/spikes/iree_probe.py",
        description="Probe the evoxc iree backend with the benchmark's own graphs.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("sanity", "scatter", "node-levels", "while", "pipeline"),
        help="which graph to build on backend='iree'",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="run the same probe in N fresh processes and report the pass/fail pattern",
    )
    parser.add_argument(
        "--case", default="xor_p100", metavar="NAME", help="pipeline mode: case from cases.py"
    )
    parser.add_argument(
        "--bypass",
        default="",
        metavar="LIST",
        help=f"pipeline mode: comma-separated probe-only bypasses from {BYPASSES}",
    )
    parser.add_argument(
        "--generations", type=int, default=2, metavar="N", help="pipeline mode: timed steps"
    )
    parser.add_argument("--max-nodes", type=int, default=8, metavar="N", help="node-levels: N")
    parser.add_argument(
        "--carries",
        type=int,
        default=4,
        metavar="N",
        help="while mode: total loop-carried tensors, counter included (>= 2)",
    )
    parser.add_argument(
        "--body",
        default="select",
        choices=("select", "plain"),
        help="while mode: select-based or plain elementwise loop body",
    )
    parser.add_argument("--pop", type=int, default=4, metavar="N", help="node-levels: P")
    parser.add_argument("--window", type=int, default=3, metavar="N", help="node-levels: K")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one probe mode (or the fresh-process repeat driver)."""
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.repeat > 1:
        forwarded = _without_repeat_flag(raw)
        return _repeat([str(Path(__file__).resolve()), *forwarded], args.repeat)

    if args.mode == "sanity":
        return run_sanity()
    if args.mode == "scatter":
        return run_scatter()
    if args.mode == "node-levels":
        return run_node_levels(args.max_nodes, args.pop, args.window)
    if args.mode == "while":
        if args.carries < 2:
            raise SystemExit("--carries must be >= 2 (the counter is always carried)")
        return run_while(args.carries, args.body)

    unknown = [name for name in args.bypass.split(",") if name and name not in BYPASSES]
    if unknown:
        raise SystemExit(f"--bypass: unknown {unknown}; choose from {list(BYPASSES)}")
    bypass = [name for name in args.bypass.split(",") if name]
    return run_pipeline(args.case, bypass, max(1, args.generations))


if __name__ == "__main__":
    sys.exit(main())
