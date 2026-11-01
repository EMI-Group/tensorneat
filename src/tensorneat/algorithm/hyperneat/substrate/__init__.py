"""HyperNEAT substrates: host-side coordinate geometry for fixed phenotypes.

A substrate maps neuron coordinates onto the fixed topology the CPPN is queried
over -- one CPPN query per phenotype connection.  Everything here is plain NumPy
built once at construction time, never inside an ``evoxc`` trace, and never part
of a ``State``.
"""

from .base import (
    BaseSubstrate,
    as_coor_array,
    cartesian_product,
    check_same_coordinate_dim,
    layer_coors,
    node_key_blocks,
)
from .default import DefaultSubstrate
from .full import FullSubstrate
from .mlp import MLPSubstrate

__all__ = [
    "BaseSubstrate",
    "DefaultSubstrate",
    "FullSubstrate",
    "MLPSubstrate",
    "as_coor_array",
    "cartesian_product",
    "check_same_coordinate_dim",
    "layer_coors",
    "node_key_blocks",
]
