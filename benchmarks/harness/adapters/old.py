"""Adapter for the frozen pre-rewrite JAX implementation.

This module is imported only on the old side, *after* the benchmark worker has
put ``benchmarks/legacy_jax`` on ``sys.path`` (see ``../CONTEXT.md``). The
module-scope ``jax`` / ``tensorneat_legacy`` imports are therefore safe and are
the only place in the harness allowed to load JAX.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import jax
import numpy as np
import tensorneat_legacy as tn

__all__ = ["OldAdapter"]


class OldAdapter:
    """Measure the frozen JAX baseline behind ``jax.jit(...).lower().compile()``."""

    impl = "old"
    display = "tensorneat-legacy (JAX)"
    backend_label = "jax"

    def build(self, case: Any, backend: str, seed: int) -> Any:
        """Build the legacy pipeline for ``case``.

        ``backend`` is ignored: the old side is always JAX (the requested evoxc
        backend only exists on the new side).
        """
        algorithm, problem = case.build(tn, "old")
        return tn.pipeline.Pipeline(
            algorithm,
            problem,
            seed=seed,
            fitness_target=case.fitness_target,
            generation_limit=case.generation_limit,
            is_save=False,
        )

    def setup(self, pipeline: Any) -> Any:
        """Return the initial legacy ``State``."""
        return pipeline.setup()

    def prepare(
        self, pipeline: Any, state: Any
    ) -> tuple[Any, float, int, Callable[[Any], tuple[Any, np.ndarray]]]:
        """Compile ``pipeline.step`` once and report the compile cost.

        The legacy compile does not advance the state, so zero generations are
        consumed before timing.

        Args:
            pipeline: The pipeline returned by :meth:`build`.
            state: The state returned by :meth:`setup`.

        Returns:
            ``(state, compile_seconds, 0, step_fn)``.
        """
        start = time.perf_counter()
        compiled = jax.jit(pipeline.step).lower(state).compile()
        compile_seconds = time.perf_counter() - start
        pop_size = pipeline.pop_size

        def step_fn(state: Any) -> tuple[Any, np.ndarray]:
            """Run one compiled generation, fitness synced to the host."""
            out = compiled(state)
            if not isinstance(out, (tuple, list)) or len(out) < 3:
                raise RuntimeError(
                    "old pipeline.step must return (state, previous_pop, "
                    f"fitnesses); got {type(out).__name__} of length "
                    f"{len(out) if hasattr(out, '__len__') else 'n/a'}"
                )
            # Sync inside the timed region: the fitnesses must be host-side
            # before the timer stops, exactly as a user calling `step` sees them.
            fitness = jax.device_get(out[2])
            return out[0], _as_fitness(fitness, pop_size)

        return state, compile_seconds, 0, step_fn


def _as_fitness(fitness: Any, pop_size: int) -> np.ndarray:
    """Normalise a legacy fitness array to contiguous ``float64`` shape ``(P,)``.

    Args:
        fitness: Whatever the legacy step produced (a JAX or numpy array).
        pop_size: Expected population size.

    Returns:
        A contiguous ``float64`` array of shape ``(pop_size,)``.

    Raises:
        RuntimeError: If the array does not hold exactly ``pop_size`` values.
    """
    array = np.asarray(fitness)
    if array.size != pop_size:
        raise RuntimeError(
            "old fitness array has "
            f"{array.size} value(s) (shape {array.shape}) but pop_size is "
            f"{pop_size}"
        )
    return np.ascontiguousarray(array.reshape(-1), dtype=np.float64)
