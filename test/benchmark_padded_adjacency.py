"""
Benchmark dense and bounded feed-forward graph operations on one revision.

The dense path is selected with ``max_in_degree=None``. The bounded path uses
the supplied cap. Both variants receive the same initialized population. The
benchmark covers transformation, forward evaluation, and structural mutation,
whose cycle check otherwise reconstructs the dense adjacency.

Usage:
    python test/benchmark_padded_adjacency.py
    python test/benchmark_padded_adjacency.py \
        --population 256 --max-nodes 512 --max-conns 2048 --max-in-degree 64
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import jax
import jax.numpy as jnp

from tensorneat.genome import DefaultGenome


def format_bytes(value):
    if value < 1024 ** 2:
        return f"{value / 1024:.1f} KiB"
    return f"{value / 1024 ** 2:.1f} MiB"


def tree_bytes(tree):
    return sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(tree)
        if hasattr(leaf, "size") and hasattr(leaf, "dtype")
    )


def block_until_ready(tree):
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def time_compiled(compiled, args, warmup, repeats):
    for _ in range(warmup):
        block_until_ready(compiled(*args))

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        block_until_ready(compiled(*args))
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def make_genome(max_nodes, max_conns, max_in_degree):
    return DefaultGenome(
        num_inputs=3,
        num_outputs=1,
        max_nodes=max_nodes,
        max_conns=max_conns,
        max_in_degree=max_in_degree,
    )


def measure(args, max_in_degree):
    genome = make_genome(args.max_nodes, args.max_conns, max_in_degree)
    state = genome.setup()
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.population)
    pop_nodes, pop_conns = jax.vmap(genome.initialize, in_axes=(None, 0))(
        state, keys
    )
    inputs = jnp.array([0.25, -0.5, 0.75])
    mutation_keys = jax.random.split(
        jax.random.PRNGKey(args.seed + 1), args.population
    )
    new_node_keys = (
        jnp.arange(args.population, dtype=jnp.float32) + args.max_nodes
    )
    new_conn_markers = (
        jnp.arange(args.population * 3, dtype=jnp.float32)
        .reshape(args.population, 3)
        + args.max_conns
    )

    batch_transform = jax.jit(
        jax.vmap(genome.transform, in_axes=(None, 0, 0))
    )
    batch_mutation = jax.jit(
        jax.vmap(
            genome.execute_mutation,
            in_axes=(None, 0, 0, 0, 0, 0),
        )
    )

    def transform_and_forward(nodes, conns):
        transformed = genome.transform(state, nodes, conns)
        return genome.forward(state, transformed, inputs)

    batch_full = jax.jit(jax.vmap(transform_and_forward))
    compiled_transform = batch_transform.lower(
        state, pop_nodes, pop_conns
    ).compile()
    compiled_mutation = batch_mutation.lower(
        state,
        mutation_keys,
        pop_nodes,
        pop_conns,
        new_node_keys,
        new_conn_markers,
    ).compile()
    compiled_full = batch_full.lower(pop_nodes, pop_conns).compile()

    transformed = compiled_transform(state, pop_nodes, pop_conns)
    block_until_ready(transformed)
    mutated = compiled_mutation(
        state,
        mutation_keys,
        pop_nodes,
        pop_conns,
        new_node_keys,
        new_conn_markers,
    )
    block_until_ready(mutated)
    outputs = compiled_full(pop_nodes, pop_conns)
    block_until_ready(outputs)

    transform_memory = compiled_transform.memory_analysis()
    mutation_memory = compiled_mutation.memory_analysis()
    return {
        "variant": "dense" if max_in_degree is None else f"K={max_in_degree}",
        "transform_output_bytes": tree_bytes(transformed),
        "transform_peak_bytes": transform_memory.peak_memory_in_bytes,
        "mutation_peak_bytes": mutation_memory.peak_memory_in_bytes,
        "transform_ms": time_compiled(
            compiled_transform,
            (state, pop_nodes, pop_conns),
            args.warmup,
            args.repeats,
        ),
        "mutation_ms": time_compiled(
            compiled_mutation,
            (
                state,
                mutation_keys,
                pop_nodes,
                pop_conns,
                new_node_keys,
                new_conn_markers,
            ),
            args.warmup,
            args.repeats,
        ),
        "full_ms": time_compiled(
            compiled_full,
            (pop_nodes, pop_conns),
            args.warmup,
            args.repeats,
        ),
    }


def ratio(baseline, candidate, field):
    return baseline[field] / candidate[field]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--max-conns", type=int, default=2048)
    parser.add_argument("--max-in-degree", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_in_degree > args.max_nodes:
        parser.error("--max-in-degree cannot exceed --max-nodes")

    print(f"Device: {jax.devices()[0]}")
    print(
        f"Population={args.population}, N={args.max_nodes}, "
        f"C={args.max_conns}, K={args.max_in_degree}"
    )
    dense = measure(args, None)
    bounded = measure(args, args.max_in_degree)

    print()
    print(
        "| variant | transform output | transform peak | mutation peak "
        "| transform (ms) | mutation (ms) | transform + forward (ms) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for result in (dense, bounded):
        print(
            f"| {result['variant']} "
            f"| {format_bytes(result['transform_output_bytes'])} "
            f"| {format_bytes(result['transform_peak_bytes'])} "
            f"| {format_bytes(result['mutation_peak_bytes'])} "
            f"| {result['transform_ms']:.2f} "
            f"| {result['mutation_ms']:.2f} "
            f"| {result['full_ms']:.2f} |"
        )

    print()
    print("Dense / bounded ratios (>1 favors bounded):")
    for field in (
        "transform_output_bytes",
        "transform_peak_bytes",
        "mutation_peak_bytes",
        "transform_ms",
        "mutation_ms",
        "full_ms",
    ):
        print(f"  {field}: {ratio(dense, bounded, field):.2f}x")


if __name__ == "__main__":
    main()
