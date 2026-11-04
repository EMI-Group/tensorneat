"""TensorNEAT: NEAT / HyperNEAT on the EvoX Compiler (``evoxc``).

The documented entry point is ``from tensorneat import Pipeline, algorithm,
genome, problem, common``: a :class:`~tensorneat.pipeline.Pipeline` drives one
algorithm over one problem, and every numerical operation underneath is a
batch-first ``evoxc`` kernel (population on axis 0, static shapes) executed on a
switchable backend.

No ``jax`` and no ``torch`` is imported by this package -- the compute layer is
``evoxc``, and ``test/release_gate_no_jax.py`` enforces it.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["Pipeline", "algorithm", "common", "genome", "problem", "__version__"]

try:
    __version__ = version("tensorneat")
except PackageNotFoundError:  # a source checkout that was never installed
    __version__ = "0.0.0"

from . import algorithm, common, genome, problem
from .pipeline import Pipeline
