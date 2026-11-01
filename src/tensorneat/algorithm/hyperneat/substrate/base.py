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

        Notes:
            The five values are read through the abstract properties, so this
            method is usable on concrete substrates only; the bare
            :class:`BaseSubstrate` has no counts to report.
        """
        return (
            f"{type(self).__name__}("
            f"connection_type={self.connection_type!r}, "
            f"num_inputs={self.num_inputs}, "
            f"num_outputs={self.num_outputs}, "
            f"nodes_cnt={self.nodes_cnt}, "
            f"conns_cnt={self.conns_cnt})"
        )


def _positive_int(value: Any, name: str) -> int:
    """Return ``value`` as a plain ``int``, requiring ``int`` and ``>= 1``.

    ``bool`` is rejected because ``True`` would silently pass as ``1``.

    Args:
        value: The candidate configuration value.
        name: Field name used in the raised error message.

    Raises:
        TensorNEATError: If ``value`` is a ``bool``, not an integer, or ``< 1``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TensorNEATError(f"{name} must be an int, got {value!r}")
    if value < 1:
        raise TensorNEATError(f"{name} must be >= 1, got {value!r}")
    return int(value)


def _as_float(value: Any, name: str) -> float:
    """Return ``value`` as a plain finite ``float``.

    ``bool`` is rejected because ``True`` would silently pass as ``1.0``.

    Args:
        value: The candidate configuration value.
        name: Field name used in the raised error message.

    Raises:
        TensorNEATError: If ``value`` is a ``bool``, not a real number, or not
            finite.
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TensorNEATError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise TensorNEATError(f"{name} must be finite, got {value!r}")
    return number


def _to_int32(values: Any, name: str) -> np.ndarray:
    """Return a fresh ``int32`` copy of key data, refusing lossy casts.

    A floating-point key array is accepted only when every entry is finite and
    integral; anything a cast would truncate or corrupt is an error.

    Args:
        values: Array-like of node keys / endpoint keys.
        name: Field name used in the raised error message.

    Raises:
        TensorNEATError: If ``values`` is not numeric, or is floating point with
            a non-integral or non-finite entry.
    """
    array = np.asarray(values)
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise TensorNEATError(f"{name} must hold integer keys, got {values!r}")
    elif array.dtype.kind not in "iub":
        raise TensorNEATError(f"{name} must be numeric, got dtype {array.dtype}")
    return np.array(array, dtype=np.int32, copy=True)


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
        TensorNEATError: If ``coors`` is not 2-dimensional, empty, contains a
            zero coordinate dimension, or holds a non-finite coordinate.
    """
    try:
        array = np.array(coors, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as err:
        raise TensorNEATError(
            f"{name} must be a rectangular (n, cd) sequence of coordinates, "
            f"got {coors!r}"
        ) from err
    if array.ndim != 2:
        raise TensorNEATError(
            f"{name} must be a 2-D (n, cd) array, got shape {array.shape}; "
            f"a single-neuron layout is written ((x, y),)"
        )
    if array.shape[0] < 1:
        raise TensorNEATError(f"{name} must hold at least one coordinate")
    if array.shape[1] < 1:
        raise TensorNEATError(f"{name} must have at least one coordinate dimension")
    if not np.all(np.isfinite(array)):
        raise TensorNEATError(f"{name} must hold finite coordinates, got {coors!r}")
    return array


def check_same_coordinate_dim(*named_coors: tuple[str, np.ndarray]) -> int:
    """Verify that several coordinate sets live in the same space.

    Args:
        *named_coors: ``(name, array)`` pairs, each an ``(n, cd)`` array already
            normalised by :func:`as_coor_array`.

    Returns:
        int: The shared coordinate dimension ``cd >= 1``; it is what
        ``query_coors.shape[1] == 2 * cd`` is derived from.

    Raises:
        TensorNEATError: If no pair is given, a pair is not a 2-D array with a
            positive coordinate dimension, or the pairs disagree on ``cd``.
    """
    if not named_coors:
        raise TensorNEATError(
            "check_same_coordinate_dim needs at least one (name, array) pair"
        )
    shared_cd = None
    first_name = ""
    for name, coors in named_coors:
        array = np.asarray(coors)
        if array.ndim != 2 or array.shape[1] < 1:
            raise TensorNEATError(
                f"{name} must be a 2-D (n, cd) array with cd >= 1, "
                f"got shape {array.shape}"
            )
        if shared_cd is None:
            shared_cd, first_name = int(array.shape[1]), name
        elif array.shape[1] != shared_cd:
            raise TensorNEATError(
                f"{name} has {array.shape[1]} coordinate dimensions while "
                f"{first_name} has {shared_cd}; all coordinate sets must live in "
                f"the same coordinate space"
            )
    return shared_cd


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
        TensorNEATError: If ``num_inputs`` or ``num_outputs`` is not a positive
            integer, or a hidden size is not an integer or is negative.
    """
    inputs_cnt = _positive_int(num_inputs, "num_inputs")
    outputs_cnt = _positive_int(num_outputs, "num_outputs")

    hidden_total = 0
    for layer_idx, size in enumerate(hidden_sizes):
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise TensorNEATError(
                f"hidden_sizes[{layer_idx}] must be an int, got {size!r}"
            )
        if size < 0:
            raise TensorNEATError(
                f"hidden_sizes[{layer_idx}] must be >= 0, got {size!r}"
            )
        hidden_total += int(size)

    input_keys = np.arange(inputs_cnt, dtype=np.int32)
    output_keys = np.arange(inputs_cnt, inputs_cnt + outputs_cnt, dtype=np.int32)
    hidden_start = inputs_cnt + outputs_cnt
    hidden_keys = np.arange(hidden_start, hidden_start + hidden_total, dtype=np.int32)
    return input_keys, output_keys, hidden_keys


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

    Raises:
        TensorNEATError: If the key arrays are not 1-D, a coordinate array is not
            2-D, the coordinate sets disagree on ``cd``, or a coordinate array's
            length disagrees with its key array's.
    """
    source_keys = np.asarray(keys1)
    target_keys = np.asarray(keys2)
    for keys, name in ((source_keys, "keys1"), (target_keys, "keys2")):
        if keys.ndim != 1:
            raise TensorNEATError(
                f"{name} must be a 1-D key array, got shape {keys.shape}"
            )
    source_coors = np.asarray(coors1, dtype=np.float32)
    target_coors = np.asarray(coors2, dtype=np.float32)
    for coors, keys, name in (
        (source_coors, source_keys, "coors1"),
        (target_coors, target_keys, "coors2"),
    ):
        if coors.ndim != 2:
            raise TensorNEATError(
                f"{name} must be a 2-D (n, cd) array, got shape {coors.shape}"
            )
        if coors.shape[0] != keys.shape[0]:
            raise TensorNEATError(
                f"{name} has {coors.shape[0]} rows but its key array has "
                f"{keys.shape[0]} entries"
            )
    if source_coors.shape[1] != target_coors.shape[1]:
        raise TensorNEATError(
            f"coors1 has {source_coors.shape[1]} coordinate dimensions while "
            f"coors2 has {target_coors.shape[1]}; both ends of a connection must "
            f"live in the same coordinate space"
        )

    cd = source_coors.shape[1]
    n1, n2 = source_keys.shape[0], target_keys.shape[0]
    if n1 == 0 or n2 == 0:
        return (
            np.zeros((0, 2 * cd), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
        )

    coors = np.concatenate(
        [
            np.repeat(source_coors, n2, axis=0),
            np.tile(target_coors, (n1, 1)),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    key_pairs = np.stack(
        [np.repeat(source_keys, n2), np.tile(target_keys, n1)], axis=1
    ).astype(np.int32, copy=False)
    return coors, key_pairs


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

    Raises:
        TensorNEATError: If ``neuron_cnt`` is not a positive integer, or an edge
            or ``y`` value is not a finite real number.
    """
    count = _positive_int(neuron_cnt, "neuron_cnt")
    left = _as_float(x_min, "x_min")
    right = _as_float(x_max, "x_max")
    row = _as_float(y_coor, "y_coor")
    if count == 1:
        xs = np.full(1, (left + right) / 2.0, dtype=np.float32)
    else:
        xs = np.linspace(left, right, count, dtype=np.float32)
    ys = np.full(count, row, dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32, copy=False)
