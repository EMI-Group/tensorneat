"""Batched genetic operators: mutation, crossover and compatibility distance.

Every operator is a trace-time kernel over the split gene payload (``node_keys``,
``node_attrs``, ``conn_keys``, ``conn_attrs``) with the population on axis 0 and shapes static
at build time.  Operators never allocate RNG keys or node/connection keys — those are drawn or
allocated by the caller and passed in — and structural invariants (``max_nodes``,
``max_conns``, ``max_in_degree``) are enforced here rather than silently repaired.
"""

from .crossover import BaseCrossover, DefaultCrossover
from .distance import BaseDistance, DefaultDistance
from .mutation import BaseMutation, DefaultMutation, RecurrentMutation

__all__ = [
    "BaseCrossover",
    "BaseDistance",
    "BaseMutation",
    "DefaultCrossover",
    "DefaultDistance",
    "DefaultMutation",
    "RecurrentMutation",
]
