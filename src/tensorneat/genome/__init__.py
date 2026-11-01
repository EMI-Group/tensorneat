"""Genome subtree: gene payloads, genetic operators and the forward/grad algebra.

Public surface of this package: ``GenomeTensors`` (the four-array payload), the genome classes
(``BaseGenome``, ``DefaultGenome``, ``RecurrentGenome``), the gene classes and the operator
classes.  Re-exports are explicit — no wildcard import — so the public names stay greppable.
Nothing here imports ``jax``/``torch``: the compute layer is evoxc.
"""

from .base import BaseGenome
from .default import DefaultGenome
from .gene import (
    BaseConn,
    BaseGene,
    BaseNode,
    BiasNode,
    DefaultConn,
    DefaultNode,
    GeneAttr,
    OriginConn,
    OriginNode,
)
from .operations import (
    BaseCrossover,
    BaseDistance,
    BaseMutation,
    DefaultCrossover,
    DefaultDistance,
    DefaultMutation,
    RecurrentMutation,
)
from .recurrent import RecurrentGenome
from .utils import GenomeTensors

__all__ = [
    # payload
    "GenomeTensors",
    # genomes
    "BaseGenome",
    "DefaultGenome",
    "RecurrentGenome",
    # genes
    "BaseGene",
    "GeneAttr",
    "BaseNode",
    "DefaultNode",
    "BiasNode",
    "OriginNode",
    "BaseConn",
    "DefaultConn",
    "OriginConn",
    # operators
    "BaseMutation",
    "DefaultMutation",
    "RecurrentMutation",
    "BaseCrossover",
    "DefaultCrossover",
    "BaseDistance",
    "DefaultDistance",
]
