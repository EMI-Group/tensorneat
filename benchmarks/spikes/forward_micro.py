#!/usr/bin/env python3
"""Micro-benchmark of the two feed-forward formulations behind the rewrite.

This measures *formulations*, not libraries:

``legacy``
    The frozen pre-rewrite formulation: one ``fori_loop`` iteration per padded
    node slot, each iteration vmap'ing over every padded connection slot
    (``max_nodes`` steps x ``max_conns`` slots per individual).  Timed through
    ``jax.jit(jax.vmap(...))`` around ``algorithm.transform`` +
    ``algorithm.forward`` of ``benchmarks/legacy_jax``, plus the forward alone
    (transform hoisted out) so the two cost centres are separated.

``unrolled``
    A self-contained numpy prototype of the depth-parallel formulation: the
    graph is cut into ``depth`` levels and the level loop is Python-unrolled, in
    two forms of the same idea.  ``window`` gathers a ``(P, max_nodes, window)``
    index table per level and reduces over the window (``depth x P x max_nodes x
    window`` element-ops); ``segment`` gathers only the level's real edges and
    reduces them with one ``np.add.reduceat`` segment sum (``P x real_edges``
    element-ops).  Both are compared against the legacy ``max_nodes x P x
    max_conns`` of the per-node formulation.

Only ``numpy`` (always) and ``jax`` (``--mode legacy``/``all``) are imported.
Nothing from ``src/tensorneat`` is imported: the point is to measure the target
formulation before/independently of its implementation.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = REPO_ROOT / "benchmarks" / "legacy_jax"

NUM_INPUTS = 2
NUM_OUTPUTS = 1
ACTIVATION = np.tanh


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def render_table(header: list[str], rows: list[list[str]], indent: str = "  ") -> str:
    """Render an aligned plain-text table."""
    widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    lines = [indent + "  ".join(str(c).ljust(w) for c, w in zip(header, widths)).rstrip()]
    lines.append(indent + "  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(indent + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def environment_report() -> list[list[str]]:
    """Everything needed to know *where* a number was measured.

    The jax version is read from the installed metadata, never by importing jax, so
    ``--mode unrolled`` stays a numpy-only process.
    """
    rows = [
        ["python", sys.version.split()[0]],
        ["executable", sys.executable],
        ["platform", platform.platform()],
        ["machine", f"{platform.machine()} / {platform.processor() or 'unknown'}"],
        ["cpu_count", str(os.cpu_count())],
        ["numpy", np.__version__],
    ]
    try:
        from importlib.metadata import version

        rows.append(["jax (installed)", f"{version('jax')} (only imported by --mode legacy)"])
    except Exception:  # noqa: BLE001 - metadata is best effort, the run itself probes the import
        rows.append(["jax (installed)", "not installed (legacy modes unavailable)"])
    threads = {
        name: os.environ.get(name, "unset")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "XLA_FLAGS",
        )
    }
    rows.append(["threads", " ".join(f"{k}={v}" for k, v in threads.items())])
    return rows


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


# --------------------------------------------------------------------------- #
# synthetic graph (shared by both formulations)
# --------------------------------------------------------------------------- #
def level_assignment(max_nodes: int, depth: int) -> np.ndarray:
    """Assign every node slot a topological level.

    The slot layout is the legacy genome's: inputs first (slots ``0..num_inputs-1``),
    then the output node(s), then hidden nodes appended in creation order - so the
    very same population is legal for both formulations.  Hidden nodes are spread
    round-robin over the inner levels ``1..depth-2``.
    """
    if depth < 3:
        raise SystemExit(f"--depth must be >= 3, got {depth}")
    levels = np.zeros(max_nodes, dtype=np.int64)
    out_start = NUM_INPUTS
    hidden_start = NUM_INPUTS + NUM_OUTPUTS
    levels[out_start:hidden_start] = depth - 1
    inner = depth - 2
    for j in range(max_nodes - hidden_start):
        levels[hidden_start + j] = 1 + (j % inner)
    return levels


def window_for(args) -> int:
    """Window width of the depth-parallel formulation (max fan-in per node).

    ``ceil(max_conns / real_nodes)`` is the fan-in a graph needs just to *carry*
    ``max_conns`` edges; ``--window`` overrides it.
    """
    if getattr(args, "window", 0):
        return int(args.window)
    real_nodes = max(args.max_nodes - NUM_INPUTS, 1)
    return max(2, -(-args.max_conns // real_nodes))


def build_population(
    pop: int, max_nodes: int, max_conns: int, depth: int, window: int, seed: int
):
    """Synthetic, acyclic population: ``pop`` individuals with up to ``max_conns`` real edges.

    Every individual gets its own random edge set (same node/level skeleton, so the
    level tables stay dense in the node dimension).  Fan-in per node is capped at
    ``window`` and edges always point from an earlier level to a later one, so no
    individual can be cyclic.  Returns ``(edges, levels)`` where ``edges[i]`` is a
    ``(src, dst, weight)`` triple of arrays for individual ``i``.
    """
    rng = np.random.default_rng(seed)
    levels = level_assignment(max_nodes, depth)
    n = max_nodes
    # allowed source/target pairs: strictly increasing level
    allowed = levels[:, None] < levels[None, :]
    capacity = np.minimum(allowed.sum(axis=0), window)
    capacity[:NUM_INPUTS] = 0

    # fan-in allocation: hand out the 1st edge to every target, then the 2nd, ...
    # until --max-conns is reached -> every non-input node is fed, none is stranded.
    fan_in = np.zeros(n, dtype=np.int64)
    need = max_conns
    for slot in range(window):
        if need <= 0:
            break
        eligible = np.flatnonzero(capacity > slot)
        take = eligible[:need]
        fan_in[take] += 1
        need -= take.size
    if need > 0:
        print(
            f"  note: the level skeleton (max_nodes={max_nodes}, depth={depth}, "
            f"window={window}) can only carry {max_conns - need} of {max_conns} edges",
            file=sys.stderr,
        )

    targets = np.flatnonzero(fan_in > 0)
    edges = []
    for _ in range(pop):
        priority = np.where(allowed, rng.random((n, n)), -1.0)
        srcs: list[int] = []
        dsts: list[int] = []
        for t in targets:
            candidates = np.flatnonzero(allowed[:, t])
            picked = candidates[np.argsort(-priority[candidates, t])[: fan_in[t]]]
            srcs.extend(picked.tolist())
            dsts.extend([int(t)] * picked.size)
        src_arr = np.asarray(srcs, dtype=np.int32)
        dst_arr = np.asarray(dsts, dtype=np.int32)
        weight = rng.uniform(-1.0, 1.0, src_arr.size).astype(np.float32)
        edges.append((src_arr, dst_arr, weight))
    return edges, levels


def real_edge_count(edges) -> float:
    return float(np.mean([e[0].size for e in edges]))


# --------------------------------------------------------------------------- #
# numpy prototype of the depth-parallel formulation
# --------------------------------------------------------------------------- #
class UnrolledTables:
    """Level-grouped gather/weight tables of the depth-parallel formulation.

    ``src_idx[l][r, k]`` is the flat value slot feeding window slot ``k`` of node
    slot ``r`` (flat over the population), ``weight[l][r, k]`` its connection
    weight, ``dst_rows[l]`` the flat node slots computed at level ``l``.  Rows that
    are not computed at level ``l`` gather themselves with weight 0.
    """

    def __init__(self, pop, max_nodes, depth, window, src_idx, weight, dst_rows, bias, out_rows):
        self.pop = pop
        self.max_nodes = max_nodes
        self.pf = max_nodes
        self.depth = depth
        self.window = window
        self.src_idx = src_idx
        self.weight = weight
        self.dst_rows = dst_rows
        self.bias = bias
        self.out_rows = out_rows

    @property
    def input_rows(self) -> np.ndarray:
        return (np.arange(self.pop)[:, None] * self.pf + np.arange(NUM_INPUTS)[None, :]).ravel()

    @property
    def elements_per_forward(self) -> int:
        return self.depth * self.pop * self.pf * self.window


def empty_tables(pop: int, max_nodes: int, depth: int, window: int, levels: np.ndarray, seed: int):
    """Allocate the level tables."""
    pf = max_nodes
    self_idx = np.arange(pop * pf, dtype=np.int32)[:, None]
    src_idx = np.broadcast_to(self_idx, (depth, pop * pf, window)).copy()
    weight = np.zeros((depth, pop * pf, window), dtype=np.float32)
    dst_rows = []
    for level in range(depth):
        nodes = np.flatnonzero(levels == level)
        rows = (np.arange(pop)[:, None] * pf + nodes[None, :]).ravel().astype(np.int64)
        if level == 0:  # inputs are injected, not computed
            rows = np.empty(0, dtype=np.int64)
        dst_rows.append(rows)
    bias = np.random.default_rng(seed + 1).uniform(-0.5, 0.5, pop * pf).astype(np.float32)
    out_rows = (np.arange(pop) * pf + NUM_INPUTS).astype(np.int64)
    return src_idx, weight, dst_rows, bias, out_rows


def fill_edge_loop(src_idx, weight, edges, levels, pop: int, max_nodes: int):
    """Fill the tables with the obvious per-edge Python loop."""
    pf = max_nodes
    for i, (srcs, dsts, w) in enumerate(edges):
        base = i * pf
        used: dict[int, int] = {}
        for e in range(srcs.size):
            target = int(dsts[e])
            row = base + target
            slot = used.get(row, 0)
            used[row] = slot + 1
            level = int(levels[target])
            src_idx[level][row, slot] = base + int(srcs[e])
            weight[level][row, slot] = w[e]


def fill_vectorised(src_idx, weight, edges, levels, pop: int, max_nodes: int, depth: int, window: int):
    """Fill the tables by sorting all edges once per level (lower-bound build)."""
    pf = max_nodes
    counts = np.array([e[0].size for e in edges], dtype=np.int64)
    owner = np.repeat(np.arange(pop), counts)
    src_all = np.concatenate([e[0] for e in edges]).astype(np.int64)
    dst_all = np.concatenate([e[1] for e in edges]).astype(np.int64)
    w_all = np.concatenate([e[2] for e in edges])
    src_flat = (owner * pf + src_all).astype(np.int32)
    dst_flat = owner * pf + dst_all
    edge_level = levels[dst_all]
    for level in range(1, depth):
        sel = edge_level == level
        if not sel.any():
            continue
        fs, fd, fw = src_flat[sel], dst_flat[sel], w_all[sel]
        order = np.argsort(fd, kind="stable")
        fs, fd, fw = fs[order], fd[order], fw[order]
        _, starts, group_sizes = np.unique(fd, return_index=True, return_counts=True)
        if group_sizes.max() > window:
            raise SystemExit(f"fan-in {group_sizes.max()} exceeds window {window}")
        slots = np.arange(fd.size) - np.repeat(starts, group_sizes)
        src_idx[level][fd, slots] = fs
        weight[level][fd, slots] = fw


def build_unrolled_tables(
    edges, levels, pop: int, max_nodes: int, depth: int, window: int, seed: int, vectorised: bool
) -> UnrolledTables:
    src_idx, weight, dst_rows, bias, out_rows = empty_tables(
        pop, max_nodes, depth, window, levels, seed
    )
    if vectorised:
        fill_vectorised(src_idx, weight, edges, levels, pop, max_nodes, depth, window)
    else:
        fill_edge_loop(src_idx, weight, edges, levels, pop, max_nodes)
    return UnrolledTables(pop, max_nodes, depth, window, src_idx, weight, dst_rows, bias, out_rows)


def make_unrolled_forward(tables: UnrolledTables):
    """Python-unrolled level loop: gather -> window reduce -> bias + activation."""
    src_idx = tables.src_idx
    weight = tables.weight
    dst_rows = tables.dst_rows
    bias = tables.bias
    inputs = tables.input_rows
    outputs = tables.out_rows
    depth = tables.depth
    size = tables.pop * tables.pf

    def forward(input_vec: np.ndarray) -> np.ndarray:
        values = np.zeros(size, dtype=np.float32)
        values[inputs] = input_vec
        for level in range(depth):
            window_gather = np.take(values, src_idx[level])  # (P*max_nodes, window)
            reduced = (window_gather * weight[level]).sum(axis=1)  # window reduce
            rows = dst_rows[level]
            values[rows] = ACTIVATION(reduced[rows] + bias[rows])
        return values[outputs]

    return forward


class SegmentTables:
    """Level-grouped *edge-list* tables of the depth-parallel formulation.

    Per level the real edges (not padded slots) are gathered and reduced with one
    segment sum by destination, so the executed op count is ``P x real_edges`` for
    the whole forward - it scales with the connections that exist and with depth,
    never with ``max_nodes**2``.  ``unfed[l]`` lists the level-``l`` node slots that
    receive no edge at all; they still need the bias + activation.
    """

    def __init__(self, pop, max_nodes, depth, src, weight, starts, dst_flat, unfed, bias, out_rows):
        self.pop = pop
        self.max_nodes = max_nodes
        self.pf = max_nodes
        self.depth = depth
        self.src = src
        self.weight = weight
        self.starts = starts
        self.dst_flat = dst_flat
        self.unfed = unfed
        self.bias = bias
        self.out_rows = out_rows
        self.real_edges = sum(int(s.size) for s in src)

    @property
    def input_rows(self) -> np.ndarray:
        return (np.arange(self.pop)[:, None] * self.pf + np.arange(NUM_INPUTS)[None, :]).ravel()

    @property
    def elements_per_forward(self) -> int:
        return self.real_edges


def build_segment_tables(edges, levels, pop: int, max_nodes: int, depth: int, seed: int) -> SegmentTables:
    """Sort every level's real edges by destination; that sort *is* the segment table."""
    pf = max_nodes
    counts = np.array([e[0].size for e in edges], dtype=np.int64)
    owner = np.repeat(np.arange(pop), counts)
    src_all = np.concatenate([e[0] for e in edges]).astype(np.int64)
    dst_all = np.concatenate([e[1] for e in edges]).astype(np.int64)
    src_flat = (owner * pf + src_all).astype(np.int32)
    dst_flat = owner * pf + dst_all
    w_all = np.concatenate([e[2] for e in edges]).astype(np.float32)
    edge_level = levels[dst_all]

    src, weight, starts, dst, unfed = [], [], [], [], []
    for level in range(depth):
        level_nodes = np.flatnonzero(levels == level)
        level_rows = (np.arange(pop)[:, None] * pf + level_nodes[None, :]).ravel().astype(np.int64)
        if level == 0:  # inputs are injected, not computed
            level_rows = np.empty(0, dtype=np.int64)
        sel = edge_level == level
        if not sel.any():
            src.append(np.empty(0, dtype=np.int32))
            weight.append(np.empty(0, dtype=np.float32))
            starts.append(np.empty(0, dtype=np.int64))
            dst.append(np.empty(0, dtype=np.int64))
            unfed.append(level_rows)
            continue
        order = np.argsort(dst_flat[sel], kind="stable")
        seg_dst = dst_flat[sel][order]
        src.append(src_flat[sel][order])
        weight.append(w_all[sel][order])
        unique_dst, first = np.unique(seg_dst, return_index=True)
        starts.append(first.astype(np.int64))
        dst.append(unique_dst)
        unfed.append(np.setdiff1d(level_rows, unique_dst, assume_unique=True))
    bias = np.random.default_rng(seed + 1).uniform(-0.5, 0.5, pop * pf).astype(np.float32)
    out_rows = (np.arange(pop) * pf + NUM_INPUTS).astype(np.int64)
    return SegmentTables(pop, max_nodes, depth, src, weight, starts, dst, unfed, bias, out_rows)


def make_segment_forward(tables: SegmentTables):
    """Python-unrolled level loop: gather real edges -> segment sum -> bias + activation."""
    src = tables.src
    weight = tables.weight
    starts = tables.starts
    dst = tables.dst_flat
    unfed = tables.unfed
    bias = tables.bias
    inputs = tables.input_rows
    outputs = tables.out_rows
    depth = tables.depth
    size = tables.pop * tables.pf

    def forward(input_vec: np.ndarray) -> np.ndarray:
        values = np.zeros(size, dtype=np.float32)
        values[inputs] = input_vec
        for level in range(depth):
            rows = dst[level]
            if rows.size:
                contrib = np.take(values, src[level]) * weight[level]  # one entry per real edge
                reduced = np.add.reduceat(contrib, starts[level])  # segment sum by destination
                values[rows] = ACTIVATION(reduced + bias[rows])
            lonely = unfed[level]
            if lonely.size:  # node slots with no incoming edge still get bias + activation
                values[lonely] = ACTIVATION(bias[lonely])
        return values[outputs]

    return forward


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def timed_ms(fn, iters: int) -> tuple[float, float, float]:
    """Run ``fn`` ``iters`` times; return (median, min, max) in milliseconds."""
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples), max(samples)


# --------------------------------------------------------------------------- #
# legacy formulation
# --------------------------------------------------------------------------- #
def legacy_imports():
    """Import the frozen implementation; exit non-zero with a clear message if unusable."""
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "--mode legacy/all needs jax (the `bench` extra) and it is not importable: "
            f"{exc}\nInstall it, or run --mode unrolled for the numpy-only prototype."
        )
    try:
        from tensorneat_legacy.algorithm import NEAT
        from tensorneat_legacy.common import State
        from tensorneat_legacy.genome import DefaultGenome
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            f"--mode legacy/all could not import benchmarks/legacy_jax ({LEGACY_DIR}): {exc}"
        )
    return jax, jnp, DefaultGenome, NEAT, State


def legacy_population(state, jnp, genome, edges, pop: int, max_nodes: int, max_conns: int, seed: int):
    """Pack the shared synthetic graph into legacy genome arrays.

    ``nodes[p, :, 0]`` holds node keys (slot index, unique), the remaining columns
    the node gene attributes; ``conns[p, :, :3]`` holds ``(in_key, out_key, weight)``
    with NaN padding - exactly the layout ``DefaultGenome.transform/forward`` expects.
    """
    import jax

    node_len = genome.node_gene.length
    conn_len = genome.conn_gene.length
    attrs = jax.vmap(genome.node_gene.new_random_attrs, in_axes=(None, 0))(
        state, jax.random.split(jax.random.PRNGKey(seed), max_nodes)
    )
    nodes = np.full((pop, max_nodes, node_len), np.nan, dtype=np.float32)
    nodes[:, :, 0] = np.arange(max_nodes)
    nodes[:, :, 1:] = np.asarray(attrs)[None, :, :]
    conns = np.full((pop, max_conns, conn_len), np.nan, dtype=np.float32)
    for i, (srcs, dsts, w) in enumerate(edges):
        conns[i, : srcs.size, 0] = srcs
        conns[i, : srcs.size, 1] = dsts
        conns[i, : srcs.size, 2] = w
    return jnp.asarray(nodes), jnp.asarray(conns)


def run_legacy(args) -> dict:
    jax, jnp, DefaultGenome, NEAT, State = legacy_imports()

    genome = DefaultGenome(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        max_nodes=args.max_nodes,
        max_conns=args.max_conns,
    )
    algorithm = NEAT(genome, pop_size=args.pop, species_size=20)
    state = State().register(randkey=jax.random.PRNGKey(args.seed))
    start = time.perf_counter()
    state = algorithm.setup(state)
    setup_s = time.perf_counter() - start

    edges, _levels = build_population(
        args.pop, args.max_nodes, args.max_conns, args.depth, window_for(args), args.seed
    )
    nodes, conns = legacy_population(
        state, jnp, genome, edges, args.pop, args.max_nodes, args.max_conns, args.seed
    )
    inputs = jnp.asarray(
        np.random.default_rng(args.seed + 2).uniform(-1.0, 1.0, (args.pop, NUM_INPUTS)),
        dtype=jnp.float32,
    )

    def transform(state, nodes, conns):
        return jax.vmap(algorithm.transform, in_axes=(None, 0))(state, (nodes, conns))

    def forward(state, transformed, inputs):
        return jax.vmap(algorithm.forward, in_axes=(None, 0, 0))(state, transformed, inputs)

    def step(state, nodes, conns, inputs):
        # transform + forward + a dummy fitness: the pipeline's per-generation shape
        return jnp.sum(forward(state, transform(state, nodes, conns), inputs))

    def forward_only(state, transformed, inputs):
        return jnp.sum(forward(state, transformed, inputs))

    def compile_once() -> float:
        # clear the compilation cache first: the median is a real compile, not a cache hit
        jax.clear_caches()
        start = time.perf_counter()
        jax.jit(step).lower(state, nodes, conns, inputs).compile()
        return (time.perf_counter() - start) * 1e3

    compile_ms = statistics.median([compile_once() for _ in range(args.build_iters)])
    compiled_step = jax.jit(step).lower(state, nodes, conns, inputs).compile()
    # hoisted transform: the per-node forward measured without the topological sort
    transformed = jax.tree_util.tree_map(jnp.asarray, jax.device_get(transform(state, nodes, conns)))
    compiled_forward = jax.jit(forward_only).lower(state, transformed, inputs).compile()

    jax.block_until_ready(compiled_step(state, nodes, conns, inputs))
    jax.block_until_ready(compiled_forward(state, transformed, inputs))

    outputs = np.asarray(jax.device_get(forward(state, transformed, inputs)))
    valid = int(np.isfinite(outputs).sum())

    step_median, step_min, step_max = timed_ms(
        lambda: jax.block_until_ready(compiled_step(state, nodes, conns, inputs)), args.iters
    )
    fwd_median, fwd_min, fwd_max = timed_ms(
        lambda: jax.block_until_ready(compiled_forward(state, transformed, inputs)), args.iters
    )

    return {
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "setup_s": setup_s,
        "compile_ms": compile_ms,
        "real_edges": real_edge_count(edges),
        "valid_outputs": valid,
        "transform_forward_ms": step_median,
        "transform_forward_min_ms": step_min,
        "transform_forward_max_ms": step_max,
        "forward_only_ms": fwd_median,
        "forward_only_min_ms": fwd_min,
        "forward_only_max_ms": fwd_max,
        "ms_per_individual": fwd_median / args.pop,
        "ms_per_individual_step": step_median / args.pop,
    }


# --------------------------------------------------------------------------- #
# unrolled formulation
# --------------------------------------------------------------------------- #
def run_unrolled(args) -> dict:
    window = window_for(args)
    start = time.perf_counter()
    edges, levels = build_population(
        args.pop, args.max_nodes, args.max_conns, args.depth, window, args.seed
    )
    population_ms = (time.perf_counter() - start) * 1e3
    inputs = (
        np.random.default_rng(args.seed + 3).uniform(-1.0, 1.0, (args.pop * NUM_INPUTS,)).astype(np.float32)
    )

    def build(vectorised: bool):
        return build_unrolled_tables(
            edges, levels, args.pop, args.max_nodes, args.depth, window, args.seed, vectorised
        )

    loop_ms, _, _ = timed_ms(lambda: build(vectorised=False), args.build_iters)
    vec_ms, _, _ = timed_ms(lambda: build(vectorised=True), args.build_iters)
    tables = build(vectorised=False)
    if not (
        np.array_equal(tables.src_idx, build(vectorised=True).src_idx)
        and np.array_equal(tables.weight, build(vectorised=True).weight)
    ):
        raise SystemExit("internal error: the two window table builders disagree")

    forward = make_unrolled_forward(tables)
    outputs = forward(inputs)
    if not np.isfinite(outputs).all():
        raise SystemExit("internal error: the window prototype produced non-finite outputs")
    forward_ms, forward_min, forward_max = timed_ms(lambda: forward(inputs), args.iters)

    segment_build_ms, _, _ = timed_ms(
        lambda: build_segment_tables(edges, levels, args.pop, args.max_nodes, args.depth, args.seed),
        args.build_iters,
    )
    segment_tables = build_segment_tables(edges, levels, args.pop, args.max_nodes, args.depth, args.seed)
    segment_forward = make_segment_forward(segment_tables)
    # both formulations compute the same function; only the summation order differs
    max_dev = float(np.abs(segment_forward(inputs) - outputs).max())
    if max_dev > 1e-5:
        raise SystemExit(
            f"internal error: the two prototypes disagree on the outputs (max deviation {max_dev})"
        )
    seg_ms, seg_min, seg_max = timed_ms(lambda: segment_forward(inputs), args.iters)

    return {
        "window": window,
        "depth": args.depth,
        "population_ms": population_ms,
        "real_edges": real_edge_count(edges),
        "build_edge_loop_ms": loop_ms,
        "build_vectorised_ms": vec_ms,
        "build_segment_ms": segment_build_ms,
        "forward_ms": forward_ms,
        "forward_min_ms": forward_min,
        "forward_max_ms": forward_max,
        "ms_per_individual": forward_ms / args.pop,
        "elements_per_forward": tables.elements_per_forward,
        "segment_forward_ms": seg_ms,
        "segment_forward_min_ms": seg_min,
        "segment_forward_max_ms": seg_max,
        "segment_ms_per_individual": seg_ms / args.pop,
        "segment_elements_per_forward": segment_tables.elements_per_forward,
    }


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #
def print_legacy(results: dict, args) -> None:
    legacy = results["legacy"]
    print("\nlegacy per-node formulation (jax.jit + vmap, benchmarks/legacy_jax)")
    print(
        render_table(
            ["metric", "value"],
            [
                ["jax", f"{legacy['jax_version']} ({legacy['jax_backend']} backend)"],
                ["NEAT.setup (one-off)", f"{legacy['setup_s'] * 1e3:.0f} ms"],
                ["compile transform+forward (one-off)", f"{legacy['compile_ms']:.0f} ms"],
                ["real edges per individual", f"{legacy['real_edges']:.1f}"],
                ["finite outputs", f"{legacy['valid_outputs']}/{args.pop}"],
                [
                    "transform+forward, whole population",
                    f"{legacy['transform_forward_ms']:.2f} ms"
                    f"  (min {legacy['transform_forward_min_ms']:.2f}"
                    f" / max {legacy['transform_forward_max_ms']:.2f})",
                ],
                [
                    "forward only, whole population",
                    f"{legacy['forward_only_ms']:.2f} ms"
                    f"  (min {legacy['forward_only_min_ms']:.2f}"
                    f" / max {legacy['forward_only_max_ms']:.2f})",
                ],
                ["forward only, per individual", f"{legacy['ms_per_individual'] * 1e3:.2f} us"],
                ["transform+forward, per individual", f"{legacy['ms_per_individual_step'] * 1e3:.2f} us"],
                [
                    "padded element-ops / forward (est.)",
                    f"{args.pop * args.max_nodes * args.max_conns / 1e6:.2f} M"
                    f"   (P x max_nodes x max_conns)",
                ],
            ],
        )
    )


def unrolled_sweep(args, depths: list[int], results: dict) -> list[dict]:
    sweep = []
    for depth in depths:
        sweep_args = argparse.Namespace(**vars(args))
        sweep_args.depth = depth
        info = run_unrolled(sweep_args)
        results["unrolled" if depth == args.depth else f"unrolled_d{depth}"] = info
        sweep.append(info)
    return sweep


def print_unrolled(sweep: list[dict], args) -> None:
    print("\ndepth-parallel formulation (numpy prototype, level loop Python-unrolled)")
    rows = [
        [
            str(info["depth"]),
            str(info["window"]),
            f"{info['build_edge_loop_ms']:.1f}",
            f"{info['build_segment_ms']:.1f}",
            f"{info['forward_ms']:.2f}",
            f"{info['segment_forward_ms']:.2f}",
            f"{info['segment_ms_per_individual'] * 1e3:.2f}",
            f"{info['real_edges']:.0f}",
            f"{info['elements_per_forward'] / 1e6:.2f} M",
            f"{info['segment_elements_per_forward'] / 1e6:.2f} M",
        ]
        for info in sweep
    ]
    print(
        render_table(
            [
                "depth",
                "K (window)",
                "build ms window (edge loop)",
                "build ms segment (sort)",
                "forward ms window gather",
                "forward ms edge segments",
                "edge-seg per individual us",
                "real edges",
                "elem-ops/forward window (est.)",
                "elem-ops/forward segment (est.)",
            ],
            rows,
        )
    )
    for info in sweep:
        print(
            f"  depth {info['depth']}: forward min/max "
            f"window {info['forward_min_ms']:.2f}/{info['forward_max_ms']:.2f} ms, "
            f"segments {info['segment_forward_min_ms']:.2f}/{info['segment_forward_max_ms']:.2f} ms"
            f"  (window tables built vectorised: {info['build_vectorised_ms']:.1f} ms)"
        )


def print_ratio(sweep: list[dict], results: dict, args) -> None:
    legacy = results["legacy"]
    print("\nratio: per-node work is (depth x window) or (real edges x depth) instead of max_conns")
    rows = []
    for info in sweep:
        rows.append(
            [
                str(info["depth"]),
                f"{legacy['forward_only_ms']:.2f}",
                f"{info['forward_ms']:.2f}",
                f"{info['segment_forward_ms']:.2f}",
                f"{legacy['forward_only_ms'] / info['forward_ms']:.2f}x",
                f"{legacy['forward_only_ms'] / info['segment_forward_ms']:.2f}x",
                f"{legacy['transform_forward_ms'] / info['segment_forward_ms']:.2f}x",
                f"{args.max_conns / (info['depth'] * info['window']):.2f}x",
                f"{args.max_nodes * args.max_conns / info['real_edges']:.2f}x",
            ]
        )
    print(
        render_table(
            [
                "depth",
                "legacy forward ms",
                "window fwd ms",
                "segment fwd ms",
                "speed-up vs legacy (window)",
                "speed-up vs legacy (segment)",
                "speed-up vs legacy transform+forward (segment)",
                "elem-op ratio window (est.)",
                "elem-op ratio segment (est.)",
            ],
            rows,
        )
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Micro-benchmark: legacy per-node vs depth-parallel forward formulation."
    )
    parser.add_argument("--mode", choices=["legacy", "unrolled", "all"], default="all")
    parser.add_argument("--pop", type=int, default=1000, help="population size")
    parser.add_argument("--max-nodes", type=int, default=50, help="padded node slots per individual")
    parser.add_argument("--max-conns", type=int, default=150, help="real connections per individual")
    parser.add_argument("--depth", type=int, default=8, help="levels in the depth-parallel formulation")
    parser.add_argument("--levels", default="4,8,16", help="comma separated depths for the sweep")
    parser.add_argument("--window", type=int, default=0, help="window width (0 = auto = max fan-in)")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations, median reported")
    parser.add_argument("--build-iters", type=int, default=5, help="timed build/compile iterations")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    depths = [int(item) for item in args.levels.split(",") if item.strip()]

    print("conditions")
    print(
        render_table(
            ["name", "value"],
            [
                ["mode", args.mode],
                ["pop (P)", str(args.pop)],
                ["max_nodes", str(args.max_nodes)],
                ["max_conns (real edges/individual)", str(args.max_conns)],
                ["depth (unrolled levels)", str(args.depth)],
                ["depth sweep (--levels)", args.levels],
                ["window K", f"{args.window or 'auto'} -> {window_for(args)}"],
                ["timed iterations", str(args.iters)],
                ["build/compile iterations", str(args.build_iters)],
                ["seed", str(args.seed)],
                ["num_inputs / num_outputs", f"{NUM_INPUTS} / {NUM_OUTPUTS}"],
                ["activation (unrolled prototype)", "tanh"],
                ["dummy fitness (legacy)", "jnp.sum(outputs); the problem evaluate loop is excluded"],
                [
                    "population",
                    "synthetic random DAG, level-ordered, per-individual edges, node keys = slot index",
                ],
            ],
        )
    )

    print("\nenvironment")
    print(render_table(["name", "value"], environment_report()))
    print(f"\ncommand\n  {command_line()}")

    results: dict[str, dict] = {}
    sweep: list[dict] = []

    if args.mode in ("legacy", "all"):
        results["legacy"] = run_legacy(args)
        print_legacy(results, args)

    if args.mode in ("unrolled", "all"):
        sweep = unrolled_sweep(args, depths, results)
        print_unrolled(sweep, args)

    if args.mode == "all":
        print_ratio(sweep, results, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
