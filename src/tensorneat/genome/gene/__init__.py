"""Gene subtrees: node and connection genes over the shared attr-table machinery.

Public surface of this package: ``BaseGene``/``GeneAttr`` plus the concrete node
families (``BaseNode``, ``DefaultNode``, ``BiasNode``, ``OriginNode``) and
connection families (``BaseConn``, ``DefaultConn``, ``OriginConn``).
Re-exports are explicit -- no ``import *`` -- so the public names stay greppable.
"""

from .base import BaseGene, GeneAttr
from .conn import BaseConn, DefaultConn, OriginConn
from .node import BaseNode, BiasNode, DefaultNode, OriginNode

__all__ = [
    "BaseGene",
    "GeneAttr",
    "BaseNode",
    "DefaultNode",
    "BiasNode",
    "OriginNode",
    "BaseConn",
    "DefaultConn",
    "OriginConn",
]
