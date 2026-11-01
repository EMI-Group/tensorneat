"""Feedforward MLP substrate: layered grid, adjacent layers fully connected.

:class:`MLPSubstrate` lays ``layers`` neuron columns out as a 2-D grid -- layer
``0`` (inputs) on the ``y_min`` row, layer ``-1`` (outputs) on the ``y_max`` row,
hidden layers evenly spaced in between, every layer spread evenly along ``x``
inside ``[x_min, x_max]``.  Only adjacent layers are connected, so the phenotype
is a feedforward network with no self- or backward edges.

Connection row order (part of the contract, it decides the row order of
``query_coors``/``conn_index``): layer pairs in ascending order, then source
neuron ascending, then target neuron ascending -- i.e. ``row = (layer, source,
target)`` in row-major nesting.  Node keys keep the TensorNEAT order input
block, output block, hidden blocks.
"""

from collections.abc import Sequence

import numpy as np

from tensorneat.common.errors import TensorNEATError

from .base import layer_coors, node_key_blocks
from .default import DefaultSubstrate

__all__ = ["MLPSubstrate", "analysis_substrate"]


class MLPSubstrate(DefaultSubstrate):
    """Fully connected feedforward substrate over a layered grid layout.

    Attributes:
        connection_type (str): ``"feedforward"`` -- only adjacent layers are
            connected, so the phenotype is executed by the feedforward genome.
        layers (tuple[int, ...]): Per-layer neuron counts, ``layers[0]`` inputs
            (bias neuron included) and ``layers[-1]`` outputs.
        coor_range (tuple[float, float, float, float]): ``(x_min, x_max, y_min,
            y_max)`` bounding box the grid is laid out in.

    Notes:
        * ``len(layers) == 2`` is the no-hidden-layer case and is supported: the
          inputs connect straight to the outputs.
        * Node keys: inputs ``0 .. layers[0]-1``, outputs ``layers[0] ..
          layers[0]+layers[-1]-1``, hidden neurons afterwards layer by layer in
          ascending layer order.
        * Coordinates are always 2-D, hence ``query_coors.shape[1] == 4``, and the
          grid is materialised once at construction time, outside any trace.
    """

    connection_type = "feedforward"

    def __init__(
        self,
        layers: Sequence[int],
        coor_range: tuple[float, float, float, float] = (-1, 1, -1, 1),
    ):
        """Build the layered grid geometry from a layer-size list.

        Args:
            layers: Per-layer neuron counts, at least two layers, every count
                ``>= 1``; ``layers[0]`` counts the bias neuron.
            coor_range: ``(x_min, x_max, y_min, y_max)``; each pair must be
                strictly increasing, since a degenerate axis makes every neuron
                of a layer share a coordinate and the CPPN queries degenerate.

        Raises:
            TensorNEATError: If fewer than two layers are given, a layer size is
                not a positive integer, ``coor_range`` does not hold four
                numbers, or an axis is not strictly increasing.
        """
        raise NotImplementedError

    @property
    def num_inputs(self) -> int:
        """Number of input neurons, bias neuron included.

        Returns:
            int: ``layers[0]``; HyperNEAT reports ``num_inputs - 1`` upward and
            appends the trailing bias ``1.0`` itself.
        """
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:
        """Number of output neurons.

        Returns:
            int: ``layers[-1]``.
        """
        raise NotImplementedError


def analysis_substrate(
    layers: Sequence[int],
    coor_range: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a layered layout into an adjacent-layer-connected topology.

    Pure host-side geometry with no object state: it validates the layout, spreads
    every layer along ``x`` at its own ``y`` row, then emits the three arrays a
    substrate is built from.

    Args:
        layers: Per-layer neuron counts, at least two layers, every count
            ``>= 1``.
        coor_range: ``(x_min, x_max, y_min, y_max)`` layout box with strictly
            increasing axis pairs.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ``(query_coors,
        nodes, conn_index)`` where

        * ``query_coors`` is ``(sum_i layers[i] * layers[i+1], 4)`` ``float32``
          of ``[coor_of_source, coor_of_target]`` rows,
        * ``nodes`` is ``(sum(layers),)`` ``int32`` in the order input block,
          output block, hidden blocks,
        * ``conn_index`` is ``(sum_i layers[i] * layers[i+1], 2)`` ``int32`` of
          ``(in_key, out_key)`` rows.

        Rows are ordered by layer pair, then source neuron, then target neuron,
        so ``query_coors[q]`` is always the coordinate pair of ``conn_index[q]``.

    Raises:
        TensorNEATError: If ``len(layers) < 2``, a layer size is not a positive
            integer, or an axis of ``coor_range`` is not strictly increasing.
    """
    raise NotImplementedError
