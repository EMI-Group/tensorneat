"""Evolutionary algorithm layer of TensorNEAT, expressed as evoxc kernels.

Re-exports the batch-first algorithm contract and the two concrete algorithms.
"""

from .base import BaseAlgorithm
from .neat import NEAT
from .hyperneat import HyperNEAT

__all__ = ["BaseAlgorithm", "NEAT", "HyperNEAT"]
