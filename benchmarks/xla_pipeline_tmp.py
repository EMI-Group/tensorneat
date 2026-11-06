"""Scratch: TensorNEAT XOR through backend="xla" on the ROCm device."""
import sys
import time
import numpy as np
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common import ACT


def run(backend, device, plugin, gens=3):
    opts = None
    if plugin:
        opts = {"plugin_path": plugin}
    pipeline = Pipeline(
        algorithm=algorithm.NEAT(
            pop_size=50,
            species_size=5,
            survival_threshold=0.01,
            genome=genome.DefaultGenome(
                num_inputs=2,
                num_outputs=1,
                max_nodes=12,
                output_transform=ACT.sigmoid,
            ),
        ),
        problem=problem.XOR(),
        seed=42,
        generation_limit=gens,
        fitness_target=-1e-6,
        backend=backend,
        device=device,
        compile_options=opts,
        is_detailed=False,
    )
    t0 = time.perf_counter()
    state = pipeline.setup()
    t1 = time.perf_counter()
    times = []
    best = None
    for _ in range(gens):
        t = time.perf_counter()
        state, best, _ = pipeline.step(state)
        times.append((time.perf_counter() - t) * 1e3)
    print(f"[{backend}/{device}] setup {t1 - t0:.2f}s build {pipeline.build_seconds:.2f}s "
          f"step_ms {['%.1f' % t for t in times]} best {pipeline.best_fitness!r} "
          f"best {None if best is None else type(best).__name__}")
    return pipeline.best_fitness, times


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "numpy"
    device = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None
    plugin = sys.argv[3] if len(sys.argv) > 3 else None
    run(backend, device, plugin)
