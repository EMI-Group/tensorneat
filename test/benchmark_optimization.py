"""
Benchmark for Issue #38 (unflatten_conns) and Issue #40 (crossover) optimizations.

Tests:
  1. Correctness — ensures optimized code produces *identical* results
  2. Performance — measures execution time at multiple scales
  3. Memory     — reports theoretical intermediate sizes and actual peak memory

Usage:
  # Before optimization: save current outputs as baseline
  python test/benchmark_optimization.py --save-baseline

  # After optimization: verify correctness + compare performance
  python test/benchmark_optimization.py --check

  # Quick performance run (no baseline I/O)
  python test/benchmark_optimization.py
"""

import argparse
import gc
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import jax
import jax.numpy as jnp
from jax import vmap

from tensorneat.common import State
from tensorneat.genome import DefaultGenome
from tensorneat.genome.gene.node.default import DefaultNode
from tensorneat.genome.gene.node.origin import OriginNode
from tensorneat.genome.gene.conn.default import DefaultConn
from tensorneat.genome.gene.conn.origin import OriginConn
from tensorneat.genome.operations.crossover.default import DefaultCrossover
from tensorneat.genome.operations.mutation.default import DefaultMutation
from tensorneat.genome.operations.distance.default import DefaultDistance
from tensorneat.genome.utils import unflatten_conns

BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_baseline")
SEED = 42

# ================================================================
# Scale configurations
# ================================================================

CORRECTNESS_SCALES = [
    {"label": "tiny",   "pop_size": 10,  "max_nodes": 8,   "max_conns": 10},
    {"label": "small",  "pop_size": 50,  "max_nodes": 15,  "max_conns": 30},
    {"label": "medium", "pop_size": 100, "max_nodes": 30,  "max_conns": 80},
]

PERF_SCALES = [
    {"label": "small",  "pop_size": 100,  "max_nodes": 20,  "max_conns": 50},
    {"label": "medium", "pop_size": 300,  "max_nodes": 50,  "max_conns": 150},
    {"label": "large",  "pop_size": 500,  "max_nodes": 80,  "max_conns": 300},
]

# ================================================================
# Helpers
# ================================================================

def get_device_info():
    dev = jax.local_devices()[0]
    return {"platform": dev.platform, "device": str(dev)}


def get_peak_memory_mb():
    dev = jax.local_devices()[0]
    if dev.platform == "gpu":
        try:
            return dev.memory_stats()["peak_bytes_in_use"] / 1e6
        except Exception:
            pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def clear_caches():
    jax.clear_caches()
    gc.collect()


def arrays_equal(a, b):
    return np.array_equal(a, b, equal_nan=True)


def fmt_time(s):
    if s < 0.001:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    return f"{s:.2f} s"


def fmt_bytes(b):
    if b < 1024**2:
        return f"{b / 1024:.1f} KB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def theoretical_intermediate_bytes(pop_size, max_conns, max_nodes):
    """Theoretical peak intermediate memory for the *current* (unoptimized) vmap impl."""
    return {
        "crossover_batch": pop_size * max_conns * max_conns * 1,
        "unflatten_batch": pop_size * max_conns * max_nodes * 1,
    }

# ================================================================
# Genome & population factories
# ================================================================

def make_genome(max_nodes, max_conns, use_origin=False):
    if use_origin:
        node_gene = OriginNode(response_init_std=1)
        conn_gene = OriginConn()
    else:
        node_gene = DefaultNode()
        conn_gene = DefaultConn()

    return DefaultGenome(
        node_gene=node_gene,
        conn_gene=conn_gene,
        mutation=DefaultMutation(conn_add=0.8, node_add=0.3,
                                 conn_delete=0, node_delete=0),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        num_inputs=3,
        num_outputs=1,
        max_nodes=max_nodes,
        max_conns=max_conns,
    )


def make_population(genome, pop_size, seed=SEED):
    state = genome.setup()
    key = jax.random.PRNGKey(seed)

    k1, k2 = jax.random.split(key)
    init_keys = jax.random.split(k1, pop_size)
    pop_nodes, pop_conns = vmap(genome.initialize, in_axes=(None, 0))(state, init_keys)

    has_marker = "historical_marker" in genome.conn_gene.fixed_attrs

    max_node_key = jnp.nanmax(pop_nodes[:, :, 0])
    max_marker = jnp.float32(0)
    if has_marker:
        markers = pop_conns[:, :, 2]
        max_marker = jnp.nanmax(jnp.where(jnp.isnan(markers), -1, markers))

    for i in range(3):
        k2, k3 = jax.random.split(k2)
        mut_keys = jax.random.split(k3, pop_size)
        new_node_keys = (
            jnp.arange(pop_size, dtype=jnp.float32) + max_node_key + 1 + i * pop_size
        )
        if has_marker:
            new_conn_keys = (
                jnp.arange(pop_size * 3, dtype=jnp.float32).reshape(pop_size, 3)
                + max_marker + 1 + i * pop_size * 3
            )
        else:
            new_conn_keys = jnp.zeros((pop_size, 3), dtype=jnp.float32)

        pop_nodes, pop_conns = vmap(
            genome.execute_mutation, in_axes=(None, 0, 0, 0, 0, 0)
        )(state, mut_keys, pop_nodes, pop_conns, new_node_keys, new_conn_keys)

        max_node_key = jnp.nanmax(pop_nodes[:, :, 0])
        if has_marker:
            markers = pop_conns[:, :, 2]
            max_marker = jnp.nanmax(jnp.where(jnp.isnan(markers), -1, markers))

    return state, pop_nodes, pop_conns

# ================================================================
# Benchmark functions — return numpy arrays + timing
# ================================================================

def bench_unflatten(scale, use_origin=False):
    pop_size, max_nodes, max_conns = scale["pop_size"], scale["max_nodes"], scale["max_conns"]
    genome = make_genome(max_nodes, max_conns, use_origin)
    state, pop_nodes, pop_conns = make_population(genome, pop_size)

    # --- individual ---
    nodes0, conns0 = pop_nodes[0], pop_conns[0]
    jit_fn = jax.jit(unflatten_conns)
    res = jit_fn(nodes0, conns0)
    res.block_until_ready()

    t0 = time.perf_counter()
    res = jit_fn(nodes0, conns0)
    res.block_until_ready()
    t_ind = time.perf_counter() - t0

    # --- batch (vmapped over population) ---
    jit_batch = jax.jit(vmap(unflatten_conns))
    res_batch = jit_batch(pop_nodes, pop_conns)
    res_batch.block_until_ready()

    t0 = time.perf_counter()
    res_batch = jit_batch(pop_nodes, pop_conns)
    res_batch.block_until_ready()
    t_batch = time.perf_counter() - t0

    return {
        "result_individual": np.asarray(jax.device_get(res)),
        "result_batch": np.asarray(jax.device_get(res_batch)),
        "time_individual": t_ind,
        "time_batch": t_batch,
    }


def bench_crossover_individual(scale, use_origin=False):
    max_nodes, max_conns = scale["max_nodes"], scale["max_conns"]
    genome = make_genome(max_nodes, max_conns, use_origin)
    state, pop_nodes, pop_conns = make_population(genome, pop_size=10)

    nodes1, conns1 = pop_nodes[0], pop_conns[0]
    nodes2, conns2 = pop_nodes[1], pop_conns[1]
    crossover_key = jax.random.PRNGKey(SEED + 100)

    jit_fn = jax.jit(genome.execute_crossover)
    cn, cc = jit_fn(state, crossover_key, nodes1, conns1, nodes2, conns2)
    cn.block_until_ready()

    t0 = time.perf_counter()
    cn, cc = jit_fn(state, crossover_key, nodes1, conns1, nodes2, conns2)
    cn.block_until_ready()
    t_elapsed = time.perf_counter() - t0

    return {
        "child_nodes": np.asarray(jax.device_get(cn)),
        "child_conns": np.asarray(jax.device_get(cc)),
        "time": t_elapsed,
    }


def bench_crossover_batch(scale, use_origin=False):
    pop_size, max_nodes, max_conns = scale["pop_size"], scale["max_nodes"], scale["max_conns"]
    genome = make_genome(max_nodes, max_conns, use_origin)
    state, pop_nodes, pop_conns = make_population(genome, pop_size)

    key = jax.random.PRNGKey(SEED + 200)
    k1, k2, k3 = jax.random.split(key, 3)
    winner = jax.random.randint(k1, (pop_size,), 0, pop_size)
    loser = jax.random.randint(k2, (pop_size,), 0, pop_size)
    crossover_keys = jax.random.split(k3, pop_size)

    wpn, wpc = pop_nodes[winner], pop_conns[winner]
    lpn, lpc = pop_nodes[loser], pop_conns[loser]

    batch_fn = jax.jit(
        vmap(genome.execute_crossover, in_axes=(None, 0, 0, 0, 0, 0))
    )
    nn, nc = batch_fn(state, crossover_keys, wpn, wpc, lpn, lpc)
    nn.block_until_ready()

    t0 = time.perf_counter()
    nn, nc = batch_fn(state, crossover_keys, wpn, wpc, lpn, lpc)
    nn.block_until_ready()
    t_elapsed = time.perf_counter() - t0

    return {
        "new_nodes": np.asarray(jax.device_get(nn)),
        "new_conns": np.asarray(jax.device_get(nc)),
        "time": t_elapsed,
    }

# ================================================================
# Baseline I/O
# ================================================================

def save_baseline(name, data):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}
    np.savez(os.path.join(BASELINE_DIR, f"{name}.npz"), **arrays)
    with open(os.path.join(BASELINE_DIR, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_baseline(name):
    npz_path = os.path.join(BASELINE_DIR, f"{name}.npz")
    if not os.path.exists(npz_path):
        return None
    data = dict(np.load(npz_path, allow_pickle=True))
    meta_path = os.path.join(BASELINE_DIR, f"{name}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data.update(json.load(f))
    return data


def compare_results(name, current, baseline):
    failures = []
    for key in baseline:
        if not isinstance(baseline[key], np.ndarray):
            continue
        if key not in current:
            failures.append(f"  {key}: missing in current results")
            continue
        if not arrays_equal(current[key], baseline[key]):
            diff = ~(
                (current[key] == baseline[key])
                | (np.isnan(current[key]) & np.isnan(baseline[key]))
            )
            n_diff = int(np.sum(diff))
            failures.append(f"  {key}: {n_diff}/{diff.size} elements differ")
    if failures:
        return False, f"FAIL {name}:\n" + "\n".join(failures)
    return True, f"PASS {name}"

# ================================================================
# Run one test and handle save / check / report
# ================================================================

def run_test(name, bench_fn, args, baseline_times=None):
    """Run a benchmark, optionally save or check. Returns (passed, time_sec)."""
    result = bench_fn()
    time_keys = [k for k in result if k.startswith("time")]
    primary_time_key = "time_batch" if "time_batch" in result else (time_keys[0] if time_keys else None)
    t = result.get(primary_time_key, 0)

    if args.save_baseline:
        save_baseline(name, result)
        baseline_t = None
        status = "saved"
    elif args.check:
        baseline = load_baseline(name)
        if baseline is None:
            print(f"  {name} ... NO BASELINE FOUND")
            return False, t
        ok, msg = compare_results(name, result, baseline)
        baseline_t = baseline.get(primary_time_key)
        status = msg.split("\n")[0]
        if not ok:
            print(f"  {name} ... {status}")
            for line in msg.split("\n")[1:]:
                print(f"    {line}")
            return False, t
    else:
        baseline_t = None
        status = "ok"

    time_str = fmt_time(t)
    if baseline_t is not None and baseline_t > 0:
        speedup = baseline_t / t if t > 0 else float("inf")
        time_str += f"  (baseline {fmt_time(baseline_t)}, {speedup:.2f}x)"

    print(f"  {name} ... {status}  [{time_str}]")
    return True, t

# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark for crossover / unflatten_conns optimization"
    )
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current outputs as baseline (run BEFORE optimization)")
    parser.add_argument("--check", action="store_true",
                        help="Compare against saved baseline (run AFTER optimization)")
    args = parser.parse_args()

    info = get_device_info()
    print(f"Device: {info['device']} ({info['platform']})")
    print()

    all_passed = True

    # ----------------------------------------------------------
    # Part 1: Correctness tests
    # ----------------------------------------------------------
    print("=" * 64)
    print("CORRECTNESS TESTS")
    print("=" * 64)

    for gene_label, use_origin in [("default", False), ("origin", True)]:
        for scale in CORRECTNESS_SCALES:
            sl = scale["label"]
            tag = f"{gene_label}_{sl}"
            print(f"\n[{tag}]  pop={scale['pop_size']}  nodes={scale['max_nodes']}  conns={scale['max_conns']}")

            ok, _ = run_test(
                f"unflatten_{tag}",
                lambda s=scale, u=use_origin: bench_unflatten(s, u),
                args,
            )
            all_passed &= ok

            ok, _ = run_test(
                f"crossover_ind_{tag}",
                lambda s=scale, u=use_origin: bench_crossover_individual(s, u),
                args,
            )
            all_passed &= ok

            ok, _ = run_test(
                f"crossover_batch_{tag}",
                lambda s=scale, u=use_origin: bench_crossover_batch(s, u),
                args,
            )
            all_passed &= ok

    # ----------------------------------------------------------
    # Part 2: Performance & memory analysis
    # ----------------------------------------------------------
    print()
    print("=" * 64)
    print("PERFORMANCE & MEMORY ANALYSIS")
    print("=" * 64)

    for scale in PERF_SCALES:
        p, n, c = scale["pop_size"], scale["max_nodes"], scale["max_conns"]
        theory = theoretical_intermediate_bytes(p, c, n)
        print(f"\n[{scale['label']}]  pop={p}  nodes={n}  conns={c}")
        print(f"  Theoretical intermediate (current vmap impl):")
        print(f"    batch crossover:  {fmt_bytes(theory['crossover_batch'])}")
        print(f"    batch unflatten:  {fmt_bytes(theory['unflatten_batch'])}")

        try:
            genome = make_genome(n, c)
            state, pop_nodes, pop_conns = make_population(genome, p)

            # -- batch crossover --
            key = jax.random.PRNGKey(SEED + 300)
            k1, k2, k3 = jax.random.split(key, 3)
            winner = jax.random.randint(k1, (p,), 0, p)
            loser = jax.random.randint(k2, (p,), 0, p)
            ckeys = jax.random.split(k3, p)
            wpn, wpc = pop_nodes[winner], pop_conns[winner]
            lpn, lpc = pop_nodes[loser], pop_conns[loser]

            batch_cross = jax.jit(
                vmap(genome.execute_crossover, in_axes=(None, 0, 0, 0, 0, 0))
            )
            nn, nc = batch_cross(state, ckeys, wpn, wpc, lpn, lpc)
            nn.block_until_ready()

            mem_before = get_peak_memory_mb()
            t0 = time.perf_counter()
            nn, nc = batch_cross(state, ckeys, wpn, wpc, lpn, lpc)
            nn.block_until_ready()
            t_cross = time.perf_counter() - t0
            mem_after = get_peak_memory_mb()

            print(f"  Batch crossover:  {fmt_time(t_cross)}  (peak mem ≈ {fmt_bytes(mem_after * 1e6)})")

            # -- batch unflatten --
            batch_unfl = jax.jit(vmap(unflatten_conns))
            _ = batch_unfl(pop_nodes, pop_conns)
            _.block_until_ready()

            t0 = time.perf_counter()
            _ = batch_unfl(pop_nodes, pop_conns)
            _.block_until_ready()
            t_unfl = time.perf_counter() - t0

            print(f"  Batch unflatten:  {fmt_time(t_unfl)}")
        except Exception as e:
            print(f"  ERROR: {e}")

        clear_caches()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print()
    print("=" * 64)
    if args.save_baseline:
        print(f"Baseline saved to {BASELINE_DIR}/")
    elif args.check:
        if all_passed:
            print("ALL CORRECTNESS CHECKS PASSED")
        else:
            print("SOME CHECKS FAILED")
            sys.exit(1)
    else:
        print("Done. Use --save-baseline / --check for before/after comparison.")


if __name__ == "__main__":
    main()
