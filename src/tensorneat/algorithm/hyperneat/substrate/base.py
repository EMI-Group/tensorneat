"""Substrate protocol and the shared host-side geometry helpers.

A substrate is HyperNEAT geometry, not computation: it turns a neuron layout
(one coordinate per neuron, in ``cd`` dimensions) into the fixed topology the
CPPN is queried over.  Everything here is plain host-side Python/NumPy that runs
at construction time, outside any ``evoxc`` trace: no ``State``, no kernel, no
``jax``/``torch``/``evoxc`` import.  ``tensorneat.algorithm.hyperneat``
materialises these arrays as **graph constants** -- ``query_coors`` is the input
batch fed through the CPPN, ``nodes`` / ``conn_index`` are packed into gene
payload rows by the genome's payload builder.

Node keys are the TensorNEAT convention: input neurons occupy the contiguous
block ``0 .. num_inputs-1`` (the last input is the bias neuron), output neurons
the block ``num_inputs .. num_inputs+num_outputs-1``, and hidden neurons follow.
A substrate's ``nodes`` array is therefore a permutation of
``0 .. nodes_cnt-1``, which is what lets the genome executor address inputs and
outputs by index.

Connection row order is part of the contract: for every ``q``,
``query_coors[q]`` is ``[coor_of_source, coor_of_target]`` of the connection
``conn_index[q] = (in_key, out_key)``, so the CPPN output on row ``q`` is the
weight of that connection and nothing else.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from tensorneat.common.errors import TensorNEATError

__all__ = [
    "BaseSubstrate",
    "as_coor_array",
    "check_same_coordinate_dim",
    "node_key_blocks",
    "cartesian_product",
    "layer_coors",
]


class BaseSubstrate:
    """Host-side geometry contract every substrate implements.

    A substrate is a passive, stateless description of a phenotype topology.
    It is built once at construction time, holds only plain NumPy arrays, and is
    never part of a ``State``: the HyperNEAT algorithm reads it while tracing to
    emit graph constants.

    Attributes:
        connection_type (str): ``"feedforward"`` or ``"recurrent"``, the
            executor requirement of the phenotype.  ``HyperNEAT`` (recurrent
            genome) accepts ``"recurrent"``, ``HyperNEATFeedForward`` requires
            ``"feedforward"``; the attribute describes the executor, not the
            literal edge set, and is ``None`` on the abstract base.

    Notes:
        * Subclasses must validate in ``__init__`` and raise
          :class:`tensorneat.common.errors.TensorNEATError` on bad
          configuration -- never a bare ``assert``.
        * All array properties must be cheap: they return already materialised
          arrays, never recompute geometry per access.
        * Every array property is read-only in intent; callers must not mutate
          the returned arrays, because HyperNEAT may have frozen them as graph
          constants already.
    """

    connection_type: str = None

    @property
    def num_inputs(self) -> int:
        """Number of substrate input neurons, **including** the bias neuron.

        Returns:
            int: Count of input coordinates.  ``HyperNEAT`` appends a trailing
            ``1.0`` to its input vector and reports ``num_inputs - 1`` upward, so
            this value counts the bias coordinate too.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:
        """Number of substrate output neurons.

        Returns:
            int: Count of output coordinates, which is also the width of the
            phenotype's output vector.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    @property
    def nodes_cnt(self) -> int:
        """Number of neurons in the phenotype.

        Returns:
            int: ``nodes.shape[0]``; also the ``max_nodes`` of the hyper genome,
            so the phenotype node array is dense and carries no NaN padding.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    @property
    def conns_cnt(self) -> int:
        """Number of connections in the phenotype.

        Returns:
            int: ``conn_index.shape[0]``; also the ``max_conns`` of the hyper
            genome, so the phenotype connection array is dense and no row is
            padding.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    @property
    def query_coors(self) -> np.ndarray:
        """CPPN query coordinates, one row per phenotype connection.

        Returns:
            numpy.ndarray: ``(conns_cnt, 2 * cd)`` ``float32`` array; row ``q``
            is ``[coor_of_source, coor_of_target]`` for connection
            ``conn_index[q]``.  ``conns_cnt == query_coors.shape[0]`` and
            ``query_coors.shape[1] == 2 * cd`` is the contract HyperNEAT asserts
            against the CPPN's ``num_inputs``.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    @property
    def nodes(self) -> np.ndarray:
        """Phenotype node keys, in the one-dimensional TensorNEAT order.

        Returns:
            numpy.ndarray: ``(nodes_cnt,)`` ``int32`` array forming a
            permutation of ``0 .. nodes_cnt-1`` with input keys first
            (``0 .. num_inputs-1``), output keys next
            (``num_inputs .. num_inputs+num_outputs-1``) and hidden keys last.
            Because the array is dense in key space, index ``k`` of the genome's
            node axis is node key ``k``.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    @property
    def conn_index(self) -> np.ndarray:
        """Phenotype connection endpoints, aligned with :attr:`query_coors`.

        Returns:
            numpy.ndarray: ``(conns_cnt, 2)`` ``int32`` array whose row ``q`` is
            ``(in_key, out_key)`` for the connection whose CPPN query lives on
            row ``q`` of :attr:`query_coors`.  There is deliberately no weight
            column: weights come from the CPPN and are packed by the genome's
            gene payload builder, so this layer stays blind to gene layout.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    def validate(self) -> None:
        """Re-check the topology invariants of this substrate.

        Constructors call this after storing their arrays, and user subclasses
        may call it again after replacing topology fields.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a one-line host-side description of this substrate.

        Returns:
            str: Class name plus ``connection_type``, ``num_inputs``,
            ``num_outputs``, ``nodes_cnt`` and ``conns_cnt`` -- never the arrays
            themselves, which would flood a log.

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError


def as_coor_array(coors: Any, name: str = "coors") -> np.ndarray:
    """Coerce one coordinate set to a validated ``float32`` array.

    Args:
        coors: Sequence of coordinate tuples/arrays, or an already-shaped array,
            conceptually ``(n, cd)``.
        name: Field name used in the raised error message.

    Returns:
        numpy.ndarray: ``(n, cd)`` ``float32`` array with ``n >= 1`` and
        ``cd >= 1``; a 1-D input is rejected rather than silently promoted, so a
        single-neuron layout must be written ``((x, y),)``.

    Raises:
        TensorNEATError: If ``coors`` is not 2-dimensional, empty, or contains a
            zero coordinate dimension.
    """
    raise NotImplementedError


def check_same_coordinate_dim(*named_coors: tuple[str, np.ndarray]) -> int:
    """Verify that several coordinate sets live in the same space.

    Args:
        *named_coors: ``(name, array)`` pairs, each an ``(n, cd)`` array already
            normalised by :func:`as_coor_array`.

    Returns:
        int: The shared coordinate dimension ``cd >= 1``; it is what
        ``query_coors.shape[1] == 2 * cd`` is derived from.

    Raises:
        TensorNEATError: If the pairs disagree on ``cd``.
    """
    raise NotImplementedError


def node_key_blocks(
    num_inputs: int,
    num_outputs: int,
    hidden_sizes: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay out the contiguous node-key blocks of a phenotype.

    Implements the TensorNEAT key order: inputs first, outputs second, hidden
    neurons last, so a substrate's ``nodes`` array is a dense permutation of
    ``0 .. nodes_cnt-1`` regardless of the conceptual layer order.

    Args:
        num_inputs: Number of input keys (bias neuron included).
        num_outputs: Number of output keys.
        hidden_sizes: Per-hidden-layer neuron counts, in layer order; may be
            empty, and is treated as a single block of ``sum(hidden_sizes)``
            hidden keys concatenated layer by layer.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: ``(input_keys,
        output_keys, hidden_keys)`` as ``int32`` arrays of shape
        ``(num_inputs,)``, ``(num_outputs,)`` and ``(sum(hidden_sizes),)``.
        Hidden key ``hidden_keys[sum(hidden_sizes[:i]) + j]`` belongs to neuron
        ``j`` of hidden layer ``i``.

    Raises:
        TensorNEATError: If ``num_inputs`` or ``num_outputs`` is not positive, or
            a hidden size is negative.
    """
    raise NotImplementedError


def cartesian_product(
    keys1: np.ndarray,
    keys2: np.ndarray,
    coors1: np.ndarray,
    coors2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate a fully connected bipartite block plus its query coordinates.

    Row order is the outer product with the **source varying slowest**: the
    ``keys1``/``coors1`` axis is repeated ``len(keys2)`` times, so row
    ``i * len(keys2) + j`` pairs source ``keys1[i]`` with target ``keys2[j]``.
    Both concrete substrates depend on this order, and so does the alignment of
    :attr:`BaseSubstrate.query_coors` with :attr:`BaseSubstrate.conn_index`.

    Args:
        keys1: ``(n1,)`` ``int32`` source node keys.
        keys2: ``(n2,)`` ``int32`` target node keys.
        coors1: ``(n1, cd)`` ``float32`` source coordinates.
        coors2: ``(n2, cd)`` ``float32`` target coordinates.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(coors, key_pairs)`` where
        ``coors`` is ``(n1 * n2, 2 * cd)`` ``float32`` -- ``[coors1[i],
        coors2[j]]`` per row -- and ``key_pairs`` is ``(n1 * n2, 2)`` ``int32``
        -- ``(keys1[i], keys2[j])`` per row.  An empty ``keys1`` or ``keys2``
        yields zero rows with the correct trailing width.
    """
    raise NotImplementedError


def layer_coors(
    neuron_cnt: int,
    x_min: float,
    x_max: float,
    y_coor: float,
) -> np.ndarray:
    """Spread one neuron layer evenly along ``x`` at a fixed ``y``.

    Args:
        neuron_cnt: Number of neurons in the layer, ``>= 1``.
        x_min: Left edge of the layer.
        x_max: Right edge of the layer.
        y_coor: The ``y`` coordinate shared by every neuron of the layer.

    Returns:
        numpy.ndarray: ``(neuron_cnt, 2)`` ``float32`` array of ``(x, y)`` pairs
        spanning ``x_min .. x_max`` inclusively; a single neuron is centred at
        ``((x_min + x_max) / 2, y_coor)`` so a 1-neuron layer does not collapse
        onto the left edge.
    """
    raise NotImplementedError
