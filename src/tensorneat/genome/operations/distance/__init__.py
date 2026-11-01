"""Distance operators: compatibility distance between two genomes per parent pair."""

from .base import BaseDistance
from .default import DefaultDistance

__all__ = [
    "BaseDistance",
    "DefaultDistance",
]
