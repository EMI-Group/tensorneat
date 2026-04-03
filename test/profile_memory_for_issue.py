"""
Profile GPU peak memory for NEAT pipeline.step() under various configurations.

Each configuration runs in an isolated subprocess so that GPU memory statistics
are independent.  Output is a Markdown table ready to paste into a GitHub issue.

Usage:
    conda run -n jax_env python test/profile_memory_for_issue.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

WORKER_TEMPLATE = r'''
import os, sys, gc, time, json
sys.path.insert(0, "{src_dir}")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax, jax.numpy as jnp

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat.neat import NEAT
from tensorneat.genome import DefaultGenome
from tensorneat.common import ACT
from tensorneat.problem import XOR3d

pop, N, C = {pop}, {N}, {C}
pbs = {pbs}

algo = NEAT(
    pop_size=pop, species_size=10, survival_threshold=0.01,
    genome=DefaultGenome(
        num_inputs=3, num_outputs=1,
        max_nodes=N, max_conns=C,
        output_transform=ACT.sigmoid,
    ),
    pop_batch_size=pbs,
)
pipeline = Pipeline(algo, XOR3d(), generation_limit=5, fitness_target=-1e-6, seed=42)
state = pipeline.setup()

t0 = time.time()
compiled_step = jax.jit(pipeline.step).lower(state).compile()
compile_sec = time.time() - t0

t0 = time.time()
for _ in range(3):
    state, _, fitnesses = compiled_step(state)
    fitnesses.block_until_ready()
run_sec = (time.time() - t0) / 3

jax.effects_barrier()
stats = jax.local_devices()[0].memory_stats()
peak = stats["peak_bytes_in_use"]
gpu_name = str(jax.local_devices()[0])

result = dict(
    peak_bytes=peak,
    compile_sec=compile_sec,
    step_sec=run_sec,
    gpu=gpu_name,
)

with open("{out_file}", "w") as f:
    json.dump(result, f)
'''


def fmt_bytes(b):
    if isinstance(b, str):
        return b
    if b < 1024 ** 2:
        return f"{b / 1024:.0f} KB"
    if b < 1024 ** 3:
        return f"{b / 1024 ** 2:.0f} MB"
    return f"{b / 1024 ** 3:.2f} GB"


def run_config(pop, N, C, pbs, timeout=600):
    """Run one config in an isolated subprocess, return result dict or error string."""
    out_file = tempfile.mktemp(suffix=".json")
    script_file = tempfile.mktemp(suffix=".py")

    code = WORKER_TEMPLATE.format(
        src_dir=SRC_DIR,
        pop=pop, N=N, C=C,
        pbs="None" if pbs is None else str(pbs),
        out_file=out_file,
    )
    with open(script_file, "w") as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["conda", "run", "-n", "jax_env", "python", script_file],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    finally:
        if os.path.exists(script_file):
            os.remove(script_file)

    if os.path.exists(out_file):
        with open(out_file) as f:
            data = json.load(f)
        os.remove(out_file)
        return data

    stderr = result.stderr or ""
    if "RESOURCE_EXHAUSTED" in stderr or "Out of memory" in stderr:
        return "OOM"
    return f"ERR(rc={result.returncode})"


def main():
    sys.stdout.reconfigure(line_buffering=True)

    configs = [
        # (pop_size, max_nodes, max_conns)
        (1000, 100, 500),
        (2000, 200, 1000),
        (2500, 250, 1500),
        (5000, 200, 1500),
        (5000, 300, 2000),
        (10000, 200, 1500),
    ]

    # For each config, test: no batching, pop/10, pop/5
    def batch_variants(pop):
        return [
            ("None", None),
            (f"{pop // 10}", pop // 10),
            (f"{pop // 5}", pop // 5),
        ]

    print("Running memory profiling (each config in isolated subprocess)...")
    print("This will take several minutes.\n")

    results = {}
    total = len(configs) * 3
    done = 0

    for pop, N, C in configs:
        for label, pbs in batch_variants(pop):
            done += 1
            tag = f"pop={pop}, N={N}, C={C}, pbs={label}"
            print(f"  [{done}/{total}] {tag} ...", end=" ", flush=True)
            t0 = time.time()
            res = run_config(pop, N, C, pbs)
            elapsed = time.time() - t0
            if isinstance(res, dict):
                print(f"{fmt_bytes(res['peak_bytes'])} ({elapsed:.0f}s)")
            else:
                print(f"{res} ({elapsed:.0f}s)")
            results[(pop, N, C, label)] = res

    # --- Print Markdown table ---
    print("\n")
    print("=" * 80)
    print("MARKDOWN TABLE (paste into issue)")
    print("=" * 80)
    print()

    gpu_name = "unknown"
    for v in results.values():
        if isinstance(v, dict) and "gpu" in v:
            gpu_name = v["gpu"]
            break

    print(f"**GPU**: `{gpu_name}`\n")

    header = "| Config (pop, N, C) | `pop_batch_size=None` | `pop_batch_size=pop/10` | `pop_batch_size=pop/5` | Reduction (pop/10) |"
    sep = "|---|---|---|---|---|"
    print(header)
    print(sep)

    for pop, N, C in configs:
        label_none = "None"
        label_10 = f"{pop // 10}"
        label_5 = f"{pop // 5}"

        r_none = results.get((pop, N, C, label_none))
        r_10 = results.get((pop, N, C, label_10))
        r_5 = results.get((pop, N, C, label_5))

        def cell(r):
            if isinstance(r, dict):
                return f"{fmt_bytes(r['peak_bytes'])}"
            return str(r)

        reduction = ""
        if isinstance(r_none, dict) and isinstance(r_10, dict):
            pct = (1 - r_10["peak_bytes"] / r_none["peak_bytes"]) * 100
            reduction = f"**-{pct:.0f}%**"

        row = f"| {pop}, {N}, {C} | {cell(r_none)} | {cell(r_10)} | {cell(r_5)} | {reduction} |"
        print(row)

    # --- Compile time + step time table ---
    print()
    print("### Compile time & step execution time\n")
    header2 = "| Config (pop, N, C) | `pop_batch_size` | Compile (s) | Step (ms) | Peak Memory |"
    sep2 = "|---|---|---|---|---|"
    print(header2)
    print(sep2)

    for pop, N, C in configs:
        for label, _ in batch_variants(pop):
            r = results.get((pop, N, C, label))
            if isinstance(r, dict):
                row = (f"| {pop}, {N}, {C} | {label} "
                       f"| {r['compile_sec']:.1f} "
                       f"| {r['step_sec'] * 1000:.1f} "
                       f"| {fmt_bytes(r['peak_bytes'])} |")
            else:
                row = f"| {pop}, {N}, {C} | {label} | - | - | {r} |"
            print(row)

    print()


if __name__ == "__main__":
    main()
