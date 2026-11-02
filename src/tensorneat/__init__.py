"""TensorNEAT: NEAT / HyperNEAT on the EvoX Compiler (``evoxc``).

The documented entry point is ``from tensorneat import Pipeline, algorithm,
genome, problem, common``: a :class:`~tensorneat.pipeline.Pipeline` drives one
algorithm over one problem, and every numerical operation underneath is a
batch-first ``evoxc`` kernel (population on axis 0, static shapes) executed on a
switchable backend.

No ``jax`` and no ``torch`` is imported by this package -- the compute layer is
``evoxc``, and ``test/release_gate_no_jax.py`` enforces it.
"""

__all__ = ["Pipeline", "algorithm", "common", "genome", "problem"]

from . import algorithm, common, genome, problem
from .pipeline import Pipeline
