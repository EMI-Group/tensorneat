"""Node gene families: key columns, aggregation, combine and activation selection.

``BaseNode`` fixes the ``(index,)`` key column and the forward contract; the families
differ only in their attribute tables (``DefaultNode`` has ``response``, ``BiasNode``
does not, ``OriginNode`` only changes crossover).
"""

from .base import BaseNode
from .bias import BiasNode
from .default import DefaultNode
from .origin import OriginNode

__all__ = [
    "BaseNode",
    "DefaultNode",
    "BiasNode",
    "OriginNode",
]
