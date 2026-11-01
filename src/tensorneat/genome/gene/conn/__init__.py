"""Connection gene families: key columns, weight attributes and edge messages.

``BaseConn`` fixes the ``(input_index, output_index)`` key columns and the forward
contract; ``DefaultConn`` adds the single ``weight`` attribute column and
``OriginConn`` adds the ``historical_marker`` key column with whole-gene crossover.
"""

from .base import BaseConn
from .default import DefaultConn
from .origin import OriginConn

__all__ = [
    "BaseConn",
    "DefaultConn",
    "OriginConn",
]
