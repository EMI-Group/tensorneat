"""NEAT on the EvoX Compiler: the population cycle and its species bookkeeping.

``NEAT`` (:mod:`tensorneat.algorithm.neat.neat`) is the algorithm driven by the
pipeline; ``SpeciesController`` and ``SpeciesState``
(:mod:`tensorneat.algorithm.neat.species`) own speciation, stagnation, spawn
allocation and parent pairing.

Nothing here imports ``jax`` or ``torch``: kernels are ``evoxc`` graphs that the
pipeline builds once per shape signature.
"""

from .neat import NEAT
from .species import SpeciesController, SpeciesState

__all__ = ["NEAT", "SpeciesController", "SpeciesState"]
