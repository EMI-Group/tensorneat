"""Evolutionary algorithm layer of TensorNEAT, expressed as evoxc kernels.

Re-exports the batch-first algorithm contract, the two concrete algorithms and the
feedforward HyperNEAT variant.
"""

from .base import BaseAlgorithm
from .neat import NEAT
from .hyperneat import HyperNEAT, HyperNEATFeedForward

__all__ = ["BaseAlgorithm", "NEAT", "HyperNEAT", "HyperNEATFeedForward"]
