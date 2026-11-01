"""HyperNEAT: CPPN evolution (``NEAT``) plus a fixed substrate phenotype decoder.

Re-exports the two algorithm variants, the two substrate gene kernels, and the substrate family.
``HyperNEAT`` drives a recurrent substrate (``RecurrentGenome``), ``HyperNEATFeedForward`` a
feedforward one (``DefaultGenome``); both share one CPPN-to-weight decode.
"""

from .hyperneat import HyperNEAT, HyperNEATConn, HyperNEATNode
from .hyperneat_feedforward import HyperNEATFeedForward
from .substrate import BaseSubstrate, DefaultSubstrate, FullSubstrate, MLPSubstrate

__all__ = [
    "HyperNEAT",
    "HyperNEATFeedForward",
    "HyperNEATNode",
    "HyperNEATConn",
    "BaseSubstrate",
    "DefaultSubstrate",
    "FullSubstrate",
    "MLPSubstrate",
]
