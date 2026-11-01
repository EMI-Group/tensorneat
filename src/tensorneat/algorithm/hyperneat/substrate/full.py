"""Recurrent full-mesh substrate: input -> hidden -> hidden -> output.

:class:`FullSubstrate` reproduces the legacy TensorNEAT ``FullSubstrate``
geometry exactly, minus the weight column: the hidden layer is the only path
between inputs and outputs, hidden neurons are fully interconnected including
self-loops and both directions (which is what makes the phenotype recurrent),
and every connection's weight comes from the CPPN at the endpoints' coordinates.

The block order is fixed and is part of the contract -- it decides the row order
of ``query_coors``/``conn_index``:

1. input -> hidden, ``num_inputs * num_hidden`` rows,
2. hidden -> hidden, ``num_hidden * num_hidden`` rows,
3. hidden -> output, ``num_hidden * num_outputs`` rows.

Within one block the source index varies slowest (see
:func:`tensorneat.algorithm.hyperneat.substrate.base.cartesian_product`).
"""

import numpy as np
from numpy.typing import ArrayLike

from .base import (
    as_coor_array,
    cartesian_product,
    check_same_coordinate_dim,
    node_key_blocks,
)
from .default import DefaultSubstrate

__all__ = ["FullSubstrate", "analysis_substrate"]

_DEFAULT_INPUT_COORS = ((-1, -1), (0, -1), (1, -1))
_DEFAULT_HIDDEN_COORS = ((-1, 0), (0, 0), (1, 0))
_DEFAULT_OUTPUT_COORS = ((0, 1),)


class FullSubstrate(DefaultSubstrate):
    """Fully connected recurrent substrate over an input/hidden/output layout.

    Attributes:
        connection_type (str): ``"recurrent"`` -- hidden neurons are fully
            interconnected, self-loops included, so the phenotype must be run by
            the recurrent executor.
        input_coors (numpy.ndarray): ``(num_inputs, cd)`` ``float32`` input
            coordinates; the last one is the bias neuron.
        hidden_coors (numpy.ndarray): ``(num_hidden, cd)`` ``float32`` hidden
            coordinates, may be ordered freely.
        output_coors (numpy.ndarray): ``(num_outputs, cd)`` ``float32`` output
            coordinates.

    Notes:
        * ``num_inputs``/``num_outputs``/``num_hidden`` come from the three
          coordinate arrays, so the geometry is fully described by them.
        * Node keys: inputs ``0 .. si-1``, outputs ``si .. si+so-1``, hidden
          ``si+so .. si+so+sh-1`` -- note the hidden block sits *after* the
          output block even though it is the conceptual middle layer.
        * Shapes are materialised once here, at construction time, outside any
          trace; HyperNEAT then freezes the arrays as graph constants.
    """

    connection_type = "recurrent"

    def __init__(
        self,
        input_coors: ArrayLike = _DEFAULT_INPUT_COORS,
        hidden_coors: ArrayLike = _DEFAULT_HIDDEN_COORS,
        output_coors: ArrayLike = _DEFAULT_OUTPUT_COORS,
    ):
        """Build the geometry from three coordinate sets.

        Args:
            input_coors: ``(num_inputs, cd)`` coordinates of the input neurons,
                bias neuron last.  Defaults to a 2-D row of three neurons, i.e.
                two real inputs plus the bias.
            hidden_coors: ``(num_hidden, cd)`` coordinates of the hidden neurons.
                Must be non-empty: the hidden layer is the only path from inputs
                to outputs.
            output_coors: ``(num_outputs, cd)`` coordinates of the output
                neurons.

        Raises:
            TensorNEATError: If any coordinate set is not 2-D, ``cd < 1``, the
                sets disagree on ``cd``, or ``hidden_coors`` is empty.
        """
        self.input_coors = as_coor_array(input_coors, "input_coors")
        self.hidden_coors = as_coor_array(hidden_coors, "hidden_coors")
        self.output_coors = as_coor_array(output_coors, "output_coors")
        query_coors, nodes, conn_index = analysis_substrate(
            self.input_coors, self.hidden_coors, self.output_coors
        )
        super().__init__(
            int(self.input_coors.shape[0]),
            int(self.output_coors.shape[0]),
            query_coors,
            nodes,
            conn_index,
        )

    @property
    def num_inputs(self) -> int:
        """Number of input neurons, bias neuron included.

        Returns:
            int: ``input_coors.shape[0]``.
        """
        return int(self.input_coors.shape[0])

    @property
    def num_outputs(self) -> int:
        """Number of output neurons.

        Returns:
            int: ``output_coors.shape[0]``.
        """
        return int(self.output_coors.shape[0])


def analysis_substrate(
    input_coors: ArrayLike,
    hidden_coors: ArrayLike,
    output_coors: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn three coordinate sets into a full-mesh topology.

    Pure host-side geometry with no object state: it validates and normalises the
    coordinate sets, then emits the three arrays a substrate is built from.

    Args:
        input_coors: ``(si, cd)`` coordinates of the input neurons.
        hidden_coors: ``(sh, cd)`` coordinates of the hidden neurons, ``sh >= 1``.
        output_coors: ``(so, cd)`` coordinates of the output neurons.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ``(query_coors,
        nodes, conn_index)`` where

        * ``query_coors`` is ``(si*sh + sh*sh + sh*so, 2 * cd)`` ``float32``,
        * ``nodes`` is ``(si + so + sh,)`` ``int32`` in the order input block,
          output block, hidden block,
        * ``conn_index`` is ``(si*sh + sh*sh + sh*so, 2)`` ``int32`` of
          ``(in_key, out_key)`` rows.

        Rows appear block by block -- input->hidden, hidden->hidden,
        hidden->output -- and inside a block the source varies slowest, so
        ``query_coors[q]`` is always the coordinate pair of ``conn_index[q]``.

    Raises:
        TensorNEATError: If a coordinate set is not 2-D, the sets disagree on
            ``cd``, ``hidden_coors`` is empty, or ``input_coors`` /
            ``output_coors`` is empty.
    """
    inputs = as_coor_array(input_coors, "input_coors")
    hidden = as_coor_array(hidden_coors, "hidden_coors")
    outputs = as_coor_array(output_coors, "output_coors")
    check_same_coordinate_dim(
        ("input_coors", inputs),
        ("hidden_coors", hidden),
        ("output_coors", outputs),
    )

    input_keys, output_keys, hidden_keys = node_key_blocks(
        int(inputs.shape[0]), int(outputs.shape[0]), (int(hidden.shape[0]),)
    )

    input_hidden_coors, input_hidden_conns = cartesian_product(
        input_keys, hidden_keys, inputs, hidden
    )
    hidden_hidden_coors, hidden_hidden_conns = cartesian_product(
        hidden_keys, hidden_keys, hidden, hidden
    )
    hidden_output_coors, hidden_output_conns = cartesian_product(
        hidden_keys, output_keys, hidden, outputs
    )

    query_coors = np.ascontiguousarray(
        np.concatenate(
            [input_hidden_coors, hidden_hidden_coors, hidden_output_coors], axis=0
        ),
        dtype=np.float32,
    )
    conn_index = np.ascontiguousarray(
        np.concatenate(
            [input_hidden_conns, hidden_hidden_conns, hidden_output_conns], axis=0
        ),
        dtype=np.int32,
    )
    nodes = np.ascontiguousarray(
        np.concatenate([input_keys, output_keys, hidden_keys]), dtype=np.int32
    )
    return query_coors, nodes, conn_index
