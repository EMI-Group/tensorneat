"""The plain container substrate: pre-computed topology, no geometry maths.

:class:`DefaultSubstrate` is what users subclass for custom topologies and what
:class:`~tensorneat.algorithm.hyperneat.substrate.full.FullSubstrate` /
:class:`~tensorneat.algorithm.hyperneat.substrate.mlp.MLPSubstrate` inherit
their storage and validation from.  It owns no coordinate layout logic: the
caller supplies ``query_coors``, ``nodes`` and ``conn_index`` and this class
only validates and exposes them.
"""

import numpy as np

from tensorneat.common.errors import TensorNEATError

from .base import BaseSubstrate

__all__ = ["DefaultSubstrate"]


class DefaultSubstrate(BaseSubstrate):
    """Store a caller-supplied phenotype topology and validate its invariants.

    Attributes:
        connection_type (str): ``"recurrent"`` -- the generic container requires
            the recurrent executor even though the actual topology may well be
            a DAG; the attribute describes the executor contract, not the edge
            set.
        _query_coors (numpy.ndarray): ``(conns_cnt, 2 * cd)`` ``float32`` CPPN
            query coordinates, stored read-only in intent.
        _nodes (numpy.ndarray): ``(nodes_cnt,)`` ``int32`` node keys.
        _conn_index (numpy.ndarray): ``(conns_cnt, 2)`` ``int32`` endpoint pairs.
        _num_inputs (int): Input neuron count including the bias neuron.
        _num_outputs (int): Output neuron count.

    Notes:
        * Public access goes through the inherited read-only properties
          (``query_coors``/``nodes``/``conn_index``/``num_inputs``/...), so the
          backing fields are private on purpose: a property without a setter on
          the base class would reject ``self.nodes = ...`` assignments anyway.
        * The arrays are copied and dtype-normalised once in ``__init__``;
          HyperNEAT later freezes them as graph constants, so nothing may mutate
          them afterwards.
    """

    connection_type = "recurrent"

    _query_coors: np.ndarray
    _nodes: np.ndarray
    _conn_index: np.ndarray
    _num_inputs: int
    _num_outputs: int

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        query_coors: np.ndarray,
        nodes: np.ndarray,
        conn_index: np.ndarray,
    ):
        """Normalise and validate a fully materialised substrate topology.

        Args:
            num_inputs: Number of input neurons, **including** the trailing bias
                neuron; must be ``>= 1``.
            num_outputs: Number of output neurons; must be ``>= 1``.
            query_coors: ``(conns_cnt, 2 * cd)`` coordinate array where row ``q``
                is ``[coor_of_source, coor_of_target]`` of connection ``q``.
                Converted to ``float32``; ``conns_cnt`` must be strictly positive
                and ``shape[1]`` even.
            nodes: Node keys ordered input block, output block, hidden block.
                Accepted as ``(nodes_cnt,)`` or as a legacy ``(nodes_cnt, 1)``
                column and flattened to ``(nodes_cnt,)`` ``int32``.
            conn_index: ``(conns_cnt, 2)`` ``int32`` array of ``(in_key,
                out_key)`` rows aligned row-for-row with ``query_coors``.

        Raises:
            TensorNEATError: If ``num_inputs``/``num_outputs`` are not positive
                integers; if ``query_coors`` is not 2-D with an even, non-zero
                trailing width; if ``nodes`` is not 1-D (or an ``(n, 1)``
                column) of positive length; if ``conn_index`` is not ``(m, 2)``;
                if the row counts of ``query_coors`` and ``conn_index``
                disagree; if a key is duplicated; if ``nodes`` is not a
                permutation of ``0 .. nodes_cnt-1``; if a connection endpoint is
                not a node key; if the connection rows are not unique; or if the
                first ``num_inputs + num_outputs`` keys are not exactly the input
                and output blocks.
        """
        raise NotImplementedError

    @property
    def query_coors(self) -> np.ndarray:
        """CPPN query coordinates aligned with :attr:`conn_index`.

        Returns:
            numpy.ndarray: The stored ``(conns_cnt, 2 * cd)`` ``float32`` array.
        """
        raise NotImplementedError

    @property
    def nodes(self) -> np.ndarray:
        """Phenotype node keys in TensorNEAT order.

        Returns:
            numpy.ndarray: The stored ``(nodes_cnt,)`` ``int32`` array, a
            permutation of ``0 .. nodes_cnt-1``.
        """
        raise NotImplementedError

    @property
    def conn_index(self) -> np.ndarray:
        """Phenotype connection endpoints aligned with :attr:`query_coors`.

        Returns:
            numpy.ndarray: The stored ``(conns_cnt, 2)`` ``int32`` array of
            ``(in_key, out_key)`` rows.
        """
        raise NotImplementedError

    @property
    def num_inputs(self) -> int:
        """Number of input neurons, bias neuron included.

        Returns:
            int: The constructor's ``num_inputs``.
        """
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:
        """Number of output neurons.

        Returns:
            int: The constructor's ``num_outputs``.
        """
        raise NotImplementedError

    @property
    def nodes_cnt(self) -> int:
        """Number of neurons in the phenotype.

        Returns:
            int: ``nodes.shape[0]``.
        """
        raise NotImplementedError

    @property
    def conns_cnt(self) -> int:
        """Number of connections in the phenotype.

        Returns:
            int: ``conn_index.shape[0]``.
        """
        raise NotImplementedError

    def validate(self) -> None:
        """Re-check the storage invariants documented on ``__init__``.

        Public so a user subclass that rewrites topology fields can re-run the
        same checks instead of duplicating them.

        Raises:
            TensorNEATError: On any violated invariant listed in ``__init__``.
        """
        raise NotImplementedError
