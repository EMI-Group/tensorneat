"""HyperNEAT: CPPN evolution (``NEAT``) plus a fixed substrate phenotype decoder.

Re-exports the two algorithm variants, the two substrate gene kernels, the substrate family, and
``SubstrateSchedule`` -- the container that carries the substrate genome's execution schedule
together with the dense gene payload it executes (the pair a single ``transformed`` argument has to
hold).  ``HyperNEAT`` drives a recurrent substrate (``RecurrentGenome``), ``HyperNEATFeedForward`` a
feedforward one (``DefaultGenome``); both share one CPPN-to-weight decode.
"""

from .hyperneat import HyperNEAT, HyperNEATConn, HyperNEATNode, SubstrateSchedule
from .hyperneat_feedforward import HyperNEATFeedForward
from .substrate import BaseSubstrate, DefaultSubstrate, FullSubstrate, MLPSubstrate

__all__ = [
    "HyperNEAT",
    "HyperNEATFeedForward",
    "HyperNEATNode",
    "HyperNEATConn",
    "SubstrateSchedule",
    "BaseSubstrate",
    "DefaultSubstrate",
    "FullSubstrate",
    "MLPSubstrate",
]
