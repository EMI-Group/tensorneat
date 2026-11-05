"""Task B: stage decomposition of one generation graph on ROCm -- which stage dies first.

Every case builds exactly ONE stage graph (or the smallest graph that still
contains the stage under test) at the ``xor_p500`` geometry and runs it once on
``Device("rocm", 0)``. The prerequisite stages of a later case were proven working
by the earlier ones, so the first case that aborts names the stage:

    init -> transform -> forward -> update_species -> next generation -> speciate

One case per process (a failed HIP device is poisoned):

    timeout 900 uv run --no-sync python src/tensorneat/_rocm_spikes/rocm_b_stages.py --case b_forward

The stage bodies call the real kernels (``NEAT.setup`` itself, ``algorithm.transform``,
``algorithm.forward``, ``SpeciesController.update_species``, ``NEAT.tell``); nothing in
``tensorneat`` is modified and no kernel is re-implemented.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import evoxc
import evoxc.numpy as enp

from tensorneat.common import State
from tensorneat.genome.utils import GenomeTensors

from rocm_support import (
    DEV,
    build,
    device_of,
    finish,
    host,
    make_pipeline,
    place,
    report,
    rocm_state,
    spec_of,
)

POP = 500
SEED = 42
#: ``"rocm"`` (default) or ``"numpy"``; set from ``--backend`` before any case runs.
#: A numpy run is the same-session A/B baseline for the stages that work on rocm.
BACKEND = "rocm"
#: Timed repetitions of one stage run (0 = a single untimed run, the B verdict run).
REPEATS = 0
#: XOR's dataset size: the problem tiles 4 samples per individual (``P*S`` rows).
XOR_SAMPLES = 4


def _shape(tensor) -> str:
    """``(500, 50)`` for a tensor, as a printable string."""
    return str(tuple(int(size) for size in tensor.shape))


def _timed(fn):
    """``(value, seconds)`` of one call."""
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def _build(kernel, specs):
    """``evoxc.build`` for the selected backend (numpy builds on the host)."""
    if BACKEND == "numpy":
        return build(kernel, specs, backend="numpy", device=None, target_backends=None)
    return build(kernel, specs)


def _measure(exe, args, repeats: int | None = None):
    """``(value, median_seconds)`` over ``3 warmup + repeats`` runs (one if repeats=0)."""
    repeats = REPEATS if repeats is None else repeats
    if repeats <= 0:
        return _timed(lambda: evoxc.run(exe, *args))
    value = None
    for _ in range(3):
        value = evoxc.run(exe, *args)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = evoxc.run(exe, *args)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return value, samples[len(samples) // 2]


def _pipeline():
    return make_pipeline(POP, seed=SEED)


def _context():
    """(pipeline, device-resident state, tensor-free config) for the stage graphs."""
    p = _pipeline()
    state = p.setup() if BACKEND == "numpy" else rocm_state(p)
    return p, state, p.algorithm._static_state(state)


def _arg(array):
    """One graph argument on the selected backend's device."""
    if isinstance(array, (evoxc.Tensor, evoxc.SymbolicTensor)):
        return array
    return place(array) if BACKEND == "rocm" else evoxc.core.Tensor(
        np.ascontiguousarray(array)
    )


def _inner_state(cfg, **runtime):
    """Rebuild a full ``State`` inside a trace: the static config plus the tensors.

    ``_static_state``/``_static_config`` drop every tensor (that is why they are safe
    to bake), so the algebra's ``state.pop``/``state.species``/``state.generation``/
    ``state.randkey`` entries are re-registered here from the graph's own arguments.
    """
    entries = {name: getattr(cfg, name) for name in cfg.registered_keys()}
    entries.update(runtime)
    return State(**entries)


def _fitness(pop_size: int = POP) -> np.ndarray:
    """A (P,) float32 fitness vector: distinct, finite, all negative."""
    rng = np.random.default_rng(SEED)
    return (-rng.random(pop_size).astype(np.float32) * 0.5 - 0.1).astype(np.float32)


# ------------------------------------------------------------------ stages


def b_init() -> str:
    """The genome-initialisation graph, run through the real ``NEAT.setup``.

    ``NEAT.setup`` builds ``_init_kernel`` (split key -> ``genome.initialize`` ->
    ``_seed_species``) with ``_build_options(state)``; the only change here is that
    the incoming ``randkey`` is already rocm-resident, so the graph runs on the GPU
    instead of the CPU setup path ``Pipeline.setup`` uses.
    """
    p = _pipeline()
    key = evoxc.random.key(SEED)
    if BACKEND == "numpy":
        state = State().register(randkey=key)
        state = state.register(backend="numpy", device=None, compile_options=None)
    else:
        state = State().register(randkey=key.to(DEV))
        state = state.register(backend="iree", device=DEV, compile_options=None)
    state, elapsed = _timed(lambda: p.algorithm.setup(state))
    return (
        f"init graph ran on rocm in {elapsed:.3f}s pop{_shape(state.pop.node_keys)} "
        f"species_leaves={len(evoxc.tree_leaves(state.species))} "
        f"pop_device={device_of(state)}"
    )


def b_transform() -> str:
    """Schedule derivation (``algorithm.transform`` -> ``derive_schedule``)."""
    p, state, cfg = _context()
    specs = [spec_of(field) for field in state.pop]

    def kernel(*arrays):
        return p.algorithm.transform(cfg, GenomeTensors(*arrays))

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    out, run_seconds = _measure(exe, list(state.pop))
    overflow = int(host(evoxc.max(out.edge_overflow)))
    return (
        f"transform build={build_seconds:.3f}s run={run_seconds * 1e3:.3f}ms "
        f"in_conn{_shape(out.in_conn)} node_level{_shape(out.node_level)} "
        f"overflow_max={overflow} total={build_seconds + run_seconds:.3f}s"
    )


def b_forward() -> str:
    """``transform`` + the feed-forward level ``while_loop`` (``algorithm.forward``)."""
    p, state, cfg = _context()
    specs = [spec_of(field) for field in state.pop]
    specs.append(evoxc.TensorSpec((POP * XOR_SAMPLES, 2), evoxc.float32))
    inputs = np.zeros((POP * XOR_SAMPLES, 2), dtype=np.float32)

    def kernel(*arrays):
        payload = GenomeTensors(*arrays[:-1])
        transformed = p.algorithm.transform(cfg, payload)
        return p.algorithm.forward(cfg, transformed, arrays[-1], payload)

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    out, run_seconds = _measure(exe, [*state.pop, _arg(inputs)])
    values = host(out)
    return (
        f"forward build={build_seconds:.3f}s run={run_seconds * 1e3:.3f}ms "
        f"outputs{values.shape} finite={int(np.isfinite(values).sum())}"
    )


def b_update_species() -> str:
    """``SpeciesController.update_species`` (ranking, elites, parent pairs)."""
    p, state, cfg = _context()
    species = state.species
    structure = evoxc.tree_structure(species)
    leaves = evoxc.tree_leaves(species)
    specs = [spec_of(field) for field in state.pop]
    specs.extend(spec_of(leaf) for leaf in leaves)
    specs.append(spec_of(state.randkey))
    specs.append(evoxc.TensorSpec((POP,), evoxc.float32))

    def kernel(*arrays):
        payload = GenomeTensors(*arrays[:4])
        table = evoxc.tree_unflatten(list(arrays[4 : 4 + len(leaves)]), structure)
        inner = _inner_state(
            cfg, pop=payload, species=table, generation=enp.full((), 0.0, evoxc.float32),
            randkey=arrays[-2],
        )
        new_state, _winner, _loser, _elite = p.algorithm.species.update_species(
            inner, arrays[-1]
        )
        return [
            *evoxc.tree_leaves(new_state.species),
            evoxc.sum(new_state.species.member_count, axes=(0,)),
        ]

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    out, run_seconds = _measure(
        exe, [*state.pop, *leaves, state.randkey, _arg(_fitness())]
    )
    return (
        f"update_species build={build_seconds:.3f}s run={run_seconds * 1e3:.3f}ms "
        f"species_leaves={len(out) - 1} member_count_total={float(host(out[-1])):.1f}"
    )


def b_nextgen() -> str:
    """``update_species`` + ``_create_next_generation`` = ``tell`` minus speciate."""
    p, state, cfg = _context()
    structure = evoxc.tree_structure(state.species)
    leaves = evoxc.tree_leaves(state.species)
    specs = [spec_of(field) for field in state.pop]
    specs.extend(spec_of(leaf) for leaf in leaves)
    specs.append(spec_of(state.randkey))
    specs.append(evoxc.TensorSpec((POP,), evoxc.float32))

    def kernel(*arrays):
        payload = GenomeTensors(*arrays[:4])
        table = evoxc.tree_unflatten(list(arrays[4 : 4 + len(leaves)]), structure)
        inner = _inner_state(
            cfg, pop=payload, species=table, generation=enp.full((), 0.0, evoxc.float32),
            randkey=arrays[-2],
        )
        inner, winner, loser, elite = p.algorithm.species.update_species(
            inner, arrays[-1]
        )
        inner = p.algorithm._create_next_generation(inner, winner, loser, elite)
        return [*inner.pop, inner.randkey]

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    out, run_seconds = _measure(
        exe, [*state.pop, *leaves, state.randkey, _arg(_fitness())]
    )
    return (
        f"next_generation build={build_seconds:.3f}s run={run_seconds * 1e3:.3f}ms "
        f"node_keys{_shape(out[0])} conn_keys{_shape(out[2])} key{_shape(out[-1])}"
    )


def b_speciate() -> str:
    """``prepare_distance`` + ``SpeciesController.speciate`` (its fixed-length scans)."""
    p, state, cfg = _context()
    structure = evoxc.tree_structure(state.species)
    leaves = evoxc.tree_leaves(state.species)
    specs = [spec_of(field) for field in state.pop]
    specs.extend(spec_of(leaf) for leaf in leaves)

    def kernel(*arrays):
        payload = GenomeTensors(*arrays[:4])
        table = evoxc.tree_unflatten(list(arrays[4:]), structure)
        inner = _inner_state(
            cfg, pop=payload, species=table, generation=enp.full((), 0.0, evoxc.float32)
        )
        digest = p.algorithm.genome.prepare_distance(inner, payload)
        distance_fn = p.algorithm._distance_rows(digest, payload)
        out = p.algorithm.species.speciate(inner, distance_fn, payload)
        return [
            *evoxc.tree_leaves(out.species),
            evoxc.sum(out.species.idx2species, axes=(0,)),
        ]

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    print(f"PROBE b_speciate BUILT {build_seconds:.3f}s", flush=True)
    out, run_seconds = _measure(exe, [*state.pop, *leaves])
    return (
        f"speciate build={build_seconds:.3f}s run={run_seconds * 1e3:.3f}ms "
        f"species_leaves={len(out) - 1} idx2species_total={float(host(out[-1])):.1f}"
    )


def b_tell() -> str:
    """The whole ``tell`` stage (everything ``step`` does after ``evaluate``)."""
    p, state, cfg = _context()
    structure = evoxc.tree_structure(state.species)
    leaves = evoxc.tree_leaves(state.species)
    specs = [spec_of(field) for field in state.pop]
    specs.extend(spec_of(leaf) for leaf in leaves)
    specs.append(spec_of(state.randkey))
    specs.append(evoxc.TensorSpec((POP,), evoxc.float32))

    def kernel(*arrays):
        payload = GenomeTensors(*arrays[:4])
        table = evoxc.tree_unflatten(list(arrays[4 : 4 + len(leaves)]), structure)
        inner = _inner_state(
            cfg, pop=payload, species=table, generation=enp.full((), 0.0, evoxc.float32),
            randkey=arrays[-2],
        )
        nxt = p.algorithm.tell(inner, arrays[-1])
        return [nxt.randkey, nxt.generation, *nxt.pop, *evoxc.tree_leaves(nxt.species)]

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    print(f"PROBE b_tell BUILT {build_seconds:.3f}s", flush=True)
    out, run_seconds = _measure(
        exe, [*state.pop, *leaves, state.randkey, _arg(_fitness())]
    )
    return (
        f"tell build={build_seconds:.3f}s run={run_seconds * 1e3:.3f}ms "
        f"outputs={len(out)} generation={float(host(out[1])):.1f}"
    )


def b_sched_tables() -> str:
    """``derive_schedule`` WITHOUT ``node_levels``: the table/sort/cumsum-free steps.

    Replicates the table half of ``common/tensorization.derive_schedule`` (its
    helper calls only, no arithmetic changed) so the abort of ``b_transform`` can be
    attributed: ``key_slot_table`` -> ``_slot_of`` -> ``edge_order`` (one stable
    sort) -> ``incoming_table`` (window scatter) -> the ``topo_key`` arithmetic.
    """
    from tensorneat.common import tensorization as tz
    from tensorneat.genome.utils import constant_int32

    p, state, cfg = _context()
    genome = p.algorithm.genome
    pop_size = POP
    max_nodes, max_conns = genome.max_nodes, genome.max_conns
    max_in_degree, key_span = genome.max_in_degree, genome.key_span
    specs = [spec_of(field) for field in state.pop]

    def kernel(node_keys, _node_attrs, conn_keys, _conn_attrs):
        """The table half of ``derive_schedule``: the same helper calls, no levels."""
        table = tz.key_slot_table(node_keys, key_span)
        src = tz._slot_of(
            table, genome.conn_gene.key_column(conn_keys, "input_index"), key_span
        )
        dst = tz._slot_of(
            table, genome.conn_gene.key_column(conn_keys, "output_index"), key_span
        )
        edge_valid = tz.enp.logical_and(
            tz.enp.not_equal(src, tz._iinf_like(src)),
            tz.enp.not_equal(dst, tz._iinf_like(dst)),
        )
        input_slot = tz._slot_of(
            table, tz._row_broadcast(constant_int32(genome.input_idx), pop_size), key_span
        )
        output_slot = tz._slot_of(
            table, tz._row_broadcast(constant_int32(genome.output_idx), pop_size), key_span
        )
        rank, cc = tz.edge_order(dst, edge_valid, max_nodes, max_conns)
        in_conn, in_src, overflow = tz.incoming_table(
            rank, cc, dst, src, edge_valid, max_nodes, max_in_degree
        )
        return [
            in_conn,
            in_src,
            evoxc.cast(input_slot, evoxc.int32),
            evoxc.cast(output_slot, evoxc.int32),
            evoxc.reduce_max(evoxc.cast(overflow, evoxc.int32)),
            evoxc.sum(evoxc.cast(edge_valid, evoxc.int32), axes=(0, 1)),
        ]

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    out, run_seconds = _measure(exe, list(state.pop))
    return (
        f"sched_tables (no node_levels) build={build_seconds:.3f}s "
        f"run={run_seconds * 1e3:.3f}ms in_conn{_shape(out[0])} "
        f"edges_ok={int(host(out[-1]))} overflow={int(host(out[4]))}"
    )


def b_sched_levels() -> str:
    """``node_levels`` alone: the max-plus ``while_loop`` (+ its cycle ``cond``)."""
    from tensorneat.common import tensorization as tz
    from tensorneat.genome.utils import constant_int32

    p, state, cfg = _context()
    genome = p.algorithm.genome
    key_span, max_nodes = genome.key_span, genome.max_nodes
    specs = [spec_of(state.pop.node_keys), spec_of(state.pop.conn_keys)]

    def kernel(node_keys, conn_keys):
        """The table half (already proven by ``b_sched_tables``) then ``node_levels``."""
        table = tz.key_slot_table(node_keys, key_span)
        src = tz._slot_of(
            table, genome.conn_gene.key_column(conn_keys, "input_index"), key_span
        )
        dst = tz._slot_of(
            table, genome.conn_gene.key_column(conn_keys, "output_index"), key_span
        )
        edge_valid = tz.enp.logical_and(
            tz.enp.not_equal(src, tz._iinf_like(src)),
            tz.enp.not_equal(dst, tz._iinf_like(dst)),
        )
        input_slot = tz._slot_of(
            table, tz._row_broadcast(constant_int32(genome.input_idx), POP), key_span
        )
        rank, cc = tz.edge_order(dst, edge_valid, max_nodes, genome.max_conns)
        in_conn, in_src, _overflow = tz.incoming_table(
            rank, cc, dst, src, edge_valid, max_nodes, genome.max_in_degree
        )
        level = tz.node_levels(
            in_conn, in_src, input_slot, max_nodes, False, (src, dst, edge_valid)
        )
        return [level, evoxc.reduce_max(evoxc.cast(level, evoxc.int32))]

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    out, run_seconds = _measure(exe, [state.pop.node_keys, state.pop.conn_keys])
    return (
        f"sched_levels build={build_seconds:.3f}s run={run_seconds * 1e3:.3f}ms "
        f"level{_shape(out[0])} max={int(host(out[1]))}"
    )


def b_sched_levels_nocond() -> str:
    """``node_levels(..., acyclic=True)``: the same relaxation WITHOUT the cycle ``cond``.

    Not the shipped schedule (a feed-forward genome passes ``acyclic=True`` only when
    the caller asserts a cycle-free population; ``DefaultGenome.transform`` passes
    ``acyclic=False``), but the same max-plus ``while_loop`` at the same geometry, so
    it separates "the loop crashes" from "the ``cond`` wrapping a loop crashes".
    """
    from tensorneat.common import tensorization as tz
    from tensorneat.genome.utils import constant_int32

    p, state, cfg = _context()
    genome = p.algorithm.genome
    key_span, max_nodes = genome.key_span, genome.max_nodes
    specs = [spec_of(state.pop.node_keys), spec_of(state.pop.conn_keys)]

    def kernel(node_keys, conn_keys):
        table = tz.key_slot_table(node_keys, key_span)
        src = tz._slot_of(
            table, genome.conn_gene.key_column(conn_keys, "input_index"), key_span
        )
        dst = tz._slot_of(
            table, genome.conn_gene.key_column(conn_keys, "output_index"), key_span
        )
        edge_valid = tz.enp.logical_and(
            tz.enp.not_equal(src, tz._iinf_like(src)),
            tz.enp.not_equal(dst, tz._iinf_like(dst)),
        )
        input_slot = tz._slot_of(
            table, tz._row_broadcast(constant_int32(genome.input_idx), POP), key_span
        )
        rank, cc = tz.edge_order(dst, edge_valid, max_nodes, genome.max_conns)
        in_conn, in_src, _overflow = tz.incoming_table(
            rank, cc, dst, src, edge_valid, max_nodes, genome.max_in_degree
        )
        level = tz.node_levels(in_conn, in_src, input_slot, max_nodes, True)
        return [evoxc.cast(level, evoxc.int32), evoxc.reduce_max(evoxc.cast(level, evoxc.int32))]

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    out, run_seconds = _measure(exe, [state.pop.node_keys, state.pop.conn_keys])
    return (
        f"sched_levels acyclic=True (no cond) build={build_seconds:.3f}s "
        f"run={run_seconds * 1e3:.3f}ms level{_shape(out[0])} max={int(host(out[1]))}"
    )


def b_forward_schedule() -> str:
    """``algorithm.forward`` alone: the schedule is fed in as graph inputs.

    The schedule is derived on the host (``Pipeline._schedule_of``, the reporting
    path), so this isolates the per-pass prelude plus the level ``while_loop`` from
    the schedule derivation that ``b_transform`` proves fatal.
    """
    from tensorneat.common import Transformed

    p, state, cfg = _context()
    # ``_host_view`` is a no-op unless the pipeline knows its device, and a view
    # that is still rocm-resident would build the reporting graph for rocm.
    p.device = "rocm"
    host_state = p._host_view(state)
    payload_np = type(state.pop)(*(host(field) for field in state.pop))
    schedule_np = p._schedule_of(host_state, payload_np)
    leaves = evoxc.tree_leaves(schedule_np)
    structure = evoxc.tree_structure(schedule_np)
    specs = [spec_of(leaf) for leaf in leaves]
    specs.extend(spec_of(field) for field in state.pop)
    specs.append(evoxc.TensorSpec((POP * XOR_SAMPLES, 2), evoxc.float32))
    inputs = np.zeros((POP * XOR_SAMPLES, 2), dtype=np.float32)
    assert isinstance(schedule_np, Transformed)

    def kernel(*arrays):
        schedule = evoxc.tree_unflatten(list(arrays[: len(leaves)]), structure)
        payload = GenomeTensors(*arrays[len(leaves) : -1])
        return p.algorithm.forward(cfg, schedule, arrays[-1], payload)

    exe, build_seconds = _timed(lambda: _build(kernel, specs))
    devices = [_arg(host(root)) for root in leaves]
    args = [*devices, *state.pop, _arg(inputs)]
    out, run_seconds = _measure(exe, args)
    values = host(out)
    return (
        f"forward(given schedule) build={build_seconds:.3f}s run={run_seconds * 1e3:.3f}ms "
        f"outputs{values.shape} finite={int(np.isfinite(values).sum())}"
    )


CASES = {
    "b_init": b_init,
    "b_transform": b_transform,
    "b_sched_tables": b_sched_tables,
    "b_sched_levels": b_sched_levels,
    "b_sched_levels_nocond": b_sched_levels_nocond,
    "b_forward": b_forward,
    "b_forward_schedule": b_forward_schedule,
    "b_update_species": b_update_species,
    "b_nextgen": b_nextgen,
    "b_speciate": b_speciate,
    "b_tell": b_tell,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--backend", default="rocm", choices=("rocm", "numpy"))
    parser.add_argument("--repeats", type=int, default=0,
                        help="timed repetitions (0 = one untimed verdict run)")
    args = parser.parse_args()
    global BACKEND, REPEATS
    BACKEND, REPEATS = args.backend, args.repeats

    if BACKEND == "rocm":
        import iree.runtime as rt

        print(
            f"PROBE {args.case} devices="
            f"{rt.get_driver('hip').query_available_devices()}",
            flush=True,
        )
    print(f"PROBE {args.case} backend={BACKEND} repeats={REPEATS}", flush=True)
    finish(report(args.case, CASES[args.case]))


if __name__ == "__main__":
    main()
