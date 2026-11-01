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

from .base import _as_float, cartesian_product, layer_coors, node_key_blocks
from .default import DefaultSubstrate

__all__ = ["MLPSubstrate", "analysis_substrate"]


def _validated_layers(layers: Sequence[int]) -> tuple[int, ...]:
    """Validate the per-layer neuron counts of a layered layout.

    Args:
        layers: Candidate layer sizes, ``layers[0]`` inputs and ``layers[-1]``
            outputs, every count ``>= 1``.

    Returns:
        tuple[int, ...]: The counts as plain Python ints.

    Raises:
        TensorNEATError: If fewer than two layers are given, or a count is a
            ``bool``, not an integer, or ``< 1``.
    """
    try:
        sizes = tuple(layers)
    except TypeError as err:
        raise TensorNEATError(
            f"layers must be a sequence of layer sizes, got {layers!r}"
        ) from err
    if len(sizes) < 2:
        raise TensorNEATError(
            f"layers must hold at least two layers (inputs and outputs), "
            f"got {list(sizes)!r}"
        )
    for layer_idx, size in enumerate(sizes):
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise TensorNEATError(
                f"layers[{layer_idx}] must be an int, got {size!r}"
            )
        if size < 1:
            raise TensorNEATError(
                f"layers[{layer_idx}] must be >= 1, got {size!r}"
            )
    return tuple(int(size) for size in sizes)


def _validated_coor_range(
    coor_range: Sequence[float],
) -> tuple[float, float, float, float]:
    """Validate the layout box the grid is laid out in.

    Args:
        coor_range: ``(x_min, x_max, y_min, y_max)``.

    Returns:
        tuple[float, float, float, float]: The box as plain Python floats.

    Raises:
        TensorNEATError: If the tuple does not hold four real numbers, or an axis
            pair is not strictly increasing.
    """
    try:
        values = tuple(coor_range)
    except TypeError as err:
        raise TensorNEATError(
            f"coor_range must be (x_min, x_max, y_min, y_max), got {coor_range!r}"
        ) from err
    if len(values) != 4:
        raise TensorNEATError(
            f"coor_range must hold four numbers (x_min, x_max, y_min, y_max), "
            f"got {values!r}"
        )
    x_min = _as_float(values[0], "coor_range[0] (x_min)")
    x_max = _as_float(values[1], "coor_range[1] (x_max)")
    y_min = _as_float(values[2], "coor_range[2] (y_min)")
    y_max = _as_float(values[3], "coor_range[3] (y_max)")
    if x_min >= x_max:
        raise TensorNEATError(
            f"coor_range x axis must be strictly increasing, got x_min={x_min} "
            f"and x_max={x_max}"
        )
    if y_min >= y_max:
        raise TensorNEATError(
            f"coor_range y axis must be strictly increasing, got y_min={y_min} "
            f"and y_max={y_max}"
        )
    return x_min, x_max, y_min, y_max


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
        sizes = _validated_layers(layers)
        box = _validated_coor_range(coor_range)
        self.layers = sizes
        self.coor_range = box
        query_coors, nodes, conn_index = analysis_substrate(sizes, box)
        super().__init__(
            sizes[0],
            sizes[-1],
            query_coors,
            nodes,
            conn_index,
        )

    @property
    def num_inputs(self) -> int:
        """Number of input neurons, bias neuron included.

        Returns:
            int: ``layers[0]``; HyperNEAT reports ``num_inputs - 1`` upward and
            appends the trailing bias ``1.0`` itself.
        """
        return self.layers[0]

    @property
    def num_outputs(self) -> int:
        """Number of output neurons.

        Returns:
            int: ``layers[-1]``.
        """
        return self.layers[-1]


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
    sizes = _validated_layers(layers)
    x_min, x_max, y_min, y_max = _validated_coor_range(coor_range)

    layer_cnt = len(sizes)
    input_keys, output_keys, hidden_keys = node_key_blocks(
        sizes[0], sizes[-1], sizes[1:-1]
    )

    layer_keys: list[np.ndarray] = []
    layer_rows: list[np.ndarray] = []
    hidden_offset = 0
    for layer_idx, size in enumerate(sizes):
        if layer_idx == 0:
            keys = input_keys
        elif layer_idx == layer_cnt - 1:
            keys = output_keys
        else:
            keys = hidden_keys[hidden_offset : hidden_offset + size]
            hidden_offset += size
        y_coor = y_min + (y_max - y_min) * layer_idx / (layer_cnt - 1)
        layer_keys.append(keys)
        layer_rows.append(layer_coors(size, x_min, x_max, y_coor))

    coors_blocks: list[np.ndarray] = []
    conn_blocks: list[np.ndarray] = []
    for layer_idx in range(layer_cnt - 1):
        block_coors, block_conns = cartesian_product(
            layer_keys[layer_idx],
            layer_keys[layer_idx + 1],
            layer_rows[layer_idx],
            layer_rows[layer_idx + 1],
        )
        coors_blocks.append(block_coors)
        conn_blocks.append(block_conns)

    query_coors = np.ascontiguousarray(
        np.concatenate(coors_blocks, axis=0), dtype=np.float32
    )
    conn_index = np.ascontiguousarray(
        np.concatenate(conn_blocks, axis=0), dtype=np.int32
    )
    nodes = np.ascontiguousarray(
        np.concatenate([input_keys, output_keys, hidden_keys]), dtype=np.int32
    )
    return query_coors, nodes, conn_index
