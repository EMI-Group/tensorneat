"""Adapter for the evoxc rewrite (``src/tensorneat``).

The worker prepends ``src`` for this side, but this module also inserts the
tree under test itself (``REPO_ROOT / "src"``) at ``sys.path[0]`` so an
installed ``tensorneat`` can never shadow it. The import-time release gate below
fails loudly if ``import tensorneat`` drags ``jax`` or ``torch`` in (root
``CONTEXT.md``: the rewrite has no JAX and no PyTorch).
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import Any, Callable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
"""Repository root: ``adapters`` -> ``harness`` -> ``benchmarks`` -> repo root."""

_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402  (must follow the sys.path fix above)
import tensorneat as tn  # noqa: E402

_LEAKED = [name for name in ("jax", "torch") if name in sys.modules]
if _LEAKED:
    raise RuntimeError(
        "`import tensorneat` pulled in "
        f"{_LEAKED}; the evoxc rewrite must not import jax or torch (root "
        "CONTEXT.md constraints, enforced by test/release_gate_no_jax.py). "
        "A legacy JAX implementation of src/tensorneat is not the code under "
        "test here."
    )

__all__ = ["NewAdapter"]


class NewAdapter:
    """Measure the evoxc rewrite; its first ``step`` call performs the build."""

    impl = "new"
    backend_label = "numpy"

    def build(self, case: Any, backend: str, seed: int) -> Any:
        """Build the evoxc pipeline for ``case`` on ``backend``."""
        self.backend_label = backend
        algorithm, problem = case.build(tn, "new")
        return tn.Pipeline(
            algorithm,
            problem,
            seed=seed,
            fitness_target=case.fitness_target,
            generation_limit=case.generation_limit,
            backend=backend,
            device=None,
            compile_options=None,
            is_save=False,
            is_detailed=False,
        )

    def setup(self, pipeline: Any) -> Any:
        """Return the initial ``State``."""
        return pipeline.setup()

    def prepare(
        self, pipeline: Any, state: Any
    ) -> tuple[Any, float, int, Callable[[Any], tuple[Any, np.ndarray]]]:
        """Time the first ``step``, which performs the one-off evoxc build.

        That call also advances the search by one generation, hence the reported
        ``generations_consumed = 1`` (the worker subtracts it from the warmup so
        both sides time the same generation index).

        Args:
            pipeline: The pipeline returned by :meth:`build`.
            state: The state returned by :meth:`setup`.

        Returns:
            ``(new_state, compile_seconds, 1, step_fn)``.
        """
        pop_size = pipeline.pop_size

        start = time.perf_counter()
        result = pipeline.step(state)
        compile_seconds = time.perf_counter() - start
        new_state, _ = _split_step_result(result, pop_size, "prepare")

        def step_fn(state: Any) -> tuple[Any, np.ndarray]:
            """Run one generation, fitness materialised on the host."""
            return _split_step_result(pipeline.step(state), pop_size, "step")

        return new_state, compile_seconds, 1, step_fn


def _split_step_result(
    result: Any, pop_size: int, phase: str
) -> tuple[Any, np.ndarray]:
    """Split a pipeline step result into ``(state, fitness)``.

    The cross-node contract with the Pipeline author is that fitness is the
    *last* element of the returned tuple, so both ``(state, pop, fitness)`` and
    ``(state, fitness)`` are accepted; the state is always the first.

    Args:
        result: Whatever ``pipeline.step`` returned.
        pop_size: Expected population size.
        phase: ``"prepare"`` or ``"step"``, used in the error message.

    Returns:
        ``(state, contiguous float64 ndarray of shape (pop_size,))``.

    Raises:
        RuntimeError: If ``result`` is not a non-empty tuple/list, or its last
            element does not hold exactly ``pop_size`` values.
    """
    if not isinstance(result, (tuple, list)) or not result:
        raise RuntimeError(
            f"new {phase}: pipeline.step must return a non-empty tuple/list "
            f"whose last element is the fitness array; got "
            f"{type(result).__name__}"
        )
    fitness = result[-1]
    array = np.asarray(fitness)
    if array.size != pop_size:
        raise RuntimeError(
            f"new {phase}: the last element of pipeline.step's result must be "
            f"the host-side fitness array of length {pop_size}; got "
            f"{type(fitness).__name__} of shape {array.shape} ({array.size} "
            "value(s))"
        )
    return result[0], np.ascontiguousarray(array.reshape(-1), dtype=np.float64)
