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

from .base import BaseSubstrate, _positive_int, _to_int32

__all__ = ["DefaultSubstrate"]


def _as_query_coors(query_coors: np.ndarray) -> np.ndarray:
    """Coerce one CPPN query-coordinate array to its stored ``float32`` form.

    Args:
        query_coors: ``(conns_cnt, 2 * cd)`` coordinate pairs.

    Returns:
        numpy.ndarray: A fresh ``float32`` copy of ``query_coors``.

    Raises:
        TensorNEATError: If the array is not 2-D, has no rows, or has a zero or
            odd trailing width (a query row must hold two whole coordinates).
    """
    try:
        array = np.array(query_coors, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as err:
        raise TensorNEATError(
            f"query_coors must be a rectangular (conns_cnt, 2 * cd) array, "
            f"got {query_coors!r}"
        ) from err
    if array.ndim != 2:
        raise TensorNEATError(
            f"query_coors must be a 2-D (conns_cnt, 2 * cd) array, "
            f"got shape {array.shape}"
        )
    if array.shape[0] < 1:
        raise TensorNEATError("query_coors must hold at least one connection row")
    if array.shape[1] < 1 or array.shape[1] % 2 != 0:
        raise TensorNEATError(
            f"query_coors must have an even, non-zero trailing width "
            f"(two coordinate halves), got {array.shape[1]}"
        )
    return array


def _as_nodes(nodes: np.ndarray) -> np.ndarray:
    """Coerce node keys to the stored ``(nodes_cnt,)`` ``int32`` form.

    Args:
        nodes: ``(nodes_cnt,)`` or legacy ``(nodes_cnt, 1)`` node keys.

    Returns:
        numpy.ndarray: A fresh 1-D ``int32`` copy of ``nodes``.

    Raises:
        TensorNEATError: If ``nodes`` is not 1-D (or an ``(n, 1)`` column), is
            empty, or holds non-integral keys.
    """
    array = np.asarray(nodes)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise TensorNEATError(
            f"nodes must be a 1-D (nodes_cnt,) array or an (nodes_cnt, 1) "
            f"column, got shape {np.asarray(nodes).shape}"
        )
    if array.shape[0] < 1:
        raise TensorNEATError("nodes must hold at least one node key")
    return _to_int32(array, "nodes")


def _as_conn_index(conn_index: np.ndarray) -> np.ndarray:
    """Coerce connection endpoints to the stored ``(conns_cnt, 2)`` ``int32`` form.

    Args:
        conn_index: ``(conns_cnt, 2)`` ``(in_key, out_key)`` rows.

    Returns:
        numpy.ndarray: A fresh ``int32`` copy of ``conn_index``.

    Raises:
        TensorNEATError: If the array is not ``(m, 2)`` with ``m >= 1``, or holds
            non-integral endpoint keys.
    """
    array = np.asarray(conn_index)
    if array.ndim != 2 or array.shape[1] != 2:
        raise TensorNEATError(
            f"conn_index must be an (conns_cnt, 2) array of (in_key, out_key) "
            f"rows, got shape {array.shape}"
        )
    if array.shape[0] < 1:
        raise TensorNEATError("conn_index must hold at least one connection row")
    return _to_int32(array, "conn_index")


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
        self._num_inputs = _positive_int(num_inputs, "num_inputs")
        self._num_outputs = _positive_int(num_outputs, "num_outputs")
        self._query_coors = _as_query_coors(query_coors)
        self._nodes = _as_nodes(nodes)
        self._conn_index = _as_conn_index(conn_index)
        self.validate()

    @property
    def query_coors(self) -> np.ndarray:
        """CPPN query coordinates aligned with :attr:`conn_index`.

        Returns:
            numpy.ndarray: The stored ``(conns_cnt, 2 * cd)`` ``float32`` array.
        """
        return self._query_coors

    @property
    def nodes(self) -> np.ndarray:
        """Phenotype node keys in TensorNEAT order.

        Returns:
            numpy.ndarray: The stored ``(nodes_cnt,)`` ``int32`` array, a
            permutation of ``0 .. nodes_cnt-1``.
        """
        return self._nodes

    @property
    def conn_index(self) -> np.ndarray:
        """Phenotype connection endpoints aligned with :attr:`query_coors`.

        Returns:
            numpy.ndarray: The stored ``(conns_cnt, 2)`` ``int32`` array of
            ``(in_key, out_key)`` rows.
        """
        return self._conn_index

    @property
    def num_inputs(self) -> int:
        """Number of input neurons, bias neuron included.

        Returns:
            int: The constructor's ``num_inputs``.
        """
        return self._num_inputs

    @property
    def num_outputs(self) -> int:
        """Number of output neurons.

        Returns:
            int: The constructor's ``num_outputs``.
        """
        return self._num_outputs

    @property
    def nodes_cnt(self) -> int:
        """Number of neurons in the phenotype.

        Returns:
            int: ``nodes.shape[0]``.
        """
        return int(self._nodes.shape[0])

    @property
    def conns_cnt(self) -> int:
        """Number of connections in the phenotype.

        Returns:
            int: ``conn_index.shape[0]``.
        """
        return int(self._conn_index.shape[0])

    def validate(self) -> None:
        """Re-check the storage invariants documented on ``__init__``.

        Public so a user subclass that rewrites topology fields can re-run the
        same checks instead of duplicating them.

        Raises:
            TensorNEATError: On any violated invariant listed in ``__init__``.
        """
        num_inputs = _positive_int(self._num_inputs, "num_inputs")
        num_outputs = _positive_int(self._num_outputs, "num_outputs")

        query_coors = np.asarray(self._query_coors)
        if query_coors.ndim != 2:
            raise TensorNEATError(
                f"query_coors must be a 2-D (conns_cnt, 2 * cd) array, "
                f"got shape {query_coors.shape}"
            )
        if query_coors.shape[0] < 1:
            raise TensorNEATError(
                "query_coors must hold at least one connection row"
            )
        if query_coors.shape[1] < 1 or query_coors.shape[1] % 2 != 0:
            raise TensorNEATError(
                f"query_coors must have an even, non-zero trailing width "
                f"(two coordinate halves), got {query_coors.shape[1]}"
            )

        nodes = np.asarray(self._nodes)
        if nodes.ndim != 1:
            raise TensorNEATError(
                f"nodes must be a 1-D (nodes_cnt,) array, got shape {nodes.shape}"
            )
        if nodes.shape[0] < 1:
            raise TensorNEATError("nodes must hold at least one node key")

        conn_index = np.asarray(self._conn_index)
        if conn_index.ndim != 2 or conn_index.shape[1] != 2:
            raise TensorNEATError(
                f"conn_index must be an (conns_cnt, 2) array of "
                f"(in_key, out_key) rows, got shape {conn_index.shape}"
            )
        if conn_index.shape[0] < 1:
            raise TensorNEATError("conn_index must hold at least one connection row")
        if query_coors.shape[0] != conn_index.shape[0]:
            raise TensorNEATError(
                f"query_coors has {query_coors.shape[0]} rows but conn_index has "
                f"{conn_index.shape[0]}; query row q must describe connection q"
            )

        nodes_cnt = int(nodes.shape[0])
        if num_inputs + num_outputs > nodes_cnt:
            raise TensorNEATError(
                f"nodes_cnt={nodes_cnt} cannot hold {num_inputs} input and "
                f"{num_outputs} output neurons"
            )
        unique_cnt = int(np.unique(nodes).shape[0])
        if unique_cnt != nodes_cnt:
            raise TensorNEATError(
                f"nodes must hold unique keys, got {nodes_cnt - unique_cnt} "
                f"duplicate(s) in {nodes.tolist()}"
            )
        if not np.array_equal(np.sort(nodes), np.arange(nodes_cnt, dtype=nodes.dtype)):
            raise TensorNEATError(
                f"nodes must be a permutation of 0 .. {nodes_cnt - 1}, "
                f"got {nodes.tolist()}"
            )
        input_block = np.arange(num_inputs, dtype=nodes.dtype)
        if not np.array_equal(nodes[:num_inputs], input_block):
            raise TensorNEATError(
                f"the first num_inputs={num_inputs} node keys must be the input "
                f"block 0 .. {num_inputs - 1} (bias neuron last), "
                f"got {nodes[:num_inputs].tolist()}"
            )
        output_block = np.arange(
            num_inputs, num_inputs + num_outputs, dtype=nodes.dtype
        )
        output_slice = nodes[num_inputs : num_inputs + num_outputs]
        if not np.array_equal(output_slice, output_block):
            raise TensorNEATError(
                f"the node keys num_inputs .. num_inputs+num_outputs-1 must be the "
                f"output block {num_inputs} .. {num_inputs + num_outputs - 1}, "
                f"got {output_slice.tolist()}"
            )

        known_keys = set(nodes.tolist())
        unknown = [
            (int(in_key), int(out_key))
            for in_key, out_key in conn_index.tolist()
            if in_key not in known_keys or out_key not in known_keys
        ]
        if unknown:
            raise TensorNEATError(
                f"conn_index endpoints must be node keys, got unknown "
                f"{sorted(set(unknown))[:8]}"
            )
        unique_rows = int(np.unique(conn_index, axis=0).shape[0])
        if unique_rows != conn_index.shape[0]:
            raise TensorNEATError(
                f"conn_index rows must be unique, got "
                f"{conn_index.shape[0] - unique_rows} duplicate(s)"
            )
