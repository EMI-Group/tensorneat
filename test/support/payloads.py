"""Hand-built gene payloads for the test suites: plain numpy in, plain numpy out.

A unit test must control the topology instead of drawing one, so this module
builds the four payload arrays of :class:`~tensorneat.genome.utils.GenomeTensors`
by hand, in the documented order and with the documented padding sentinels:

=============  =========================  ======================================
field          shape / dtype              absent-row sentinel
=============  =========================  ======================================
``node_keys``  ``(P, N) int32``           ``I_INF``
``node_attrs`` ``(P, N, NA) float32``     ``NaN``
``conn_keys``  ``(P, E, CK) int32``       ``I_INF``
``conn_attrs`` ``(P, E, CA) float32``     ``NaN``
=============  =========================  ======================================

Nothing here touches ``evoxc``: a payload is host data, and the only module that
may call ``evoxc.build``/``evoxc.run`` is :mod:`support.runner`.

## Attribute-column layout this module assumes

Verified against the gene classes (``attr_specs`` order **is** the column order,
``BaseGene.attr_width == len(attr_specs)``, ``BaseGene.key_width ==
len(fixed_attrs)``):

* ``node_attrs`` — ``tensorneat/genome/gene/node/default.py::DefaultNode.attr_specs``
  declares four columns in this order:

  ==== ============= =========================================================
  col  name          value
  ==== ============= =========================================================
  0    ``bias``      float, the additive term of ``bias + response * agg``
  1    ``response``  float, the multiplicative term of ``bias + response * agg``
  2    ``aggregation`` int-valued float: **position** in the gene's
       ``aggregation_options`` list (not the global ``AGG`` registry index)
  3    ``activation``  int-valued float: position in ``activation_options``;
       ``-1`` is not an option index, it means identity
  ==== ============= =========================================================

* ``conn_keys`` — ``tensorneat/genome/gene/conn/base.py::BaseConn.fixed_attrs``
  is ``("input_index", "output_index")``, i.e. two columns holding the source
  and destination **node keys**; only ``OriginConn`` appends a third
  ``historical_marker`` column (whose documented value is the connection row
  index).
* ``conn_attrs`` — ``tensorneat/genome/gene/conn/default.py::DefaultConn.attr_specs``
  declares a single column ``weight``; the edge message is
  ``sanitized(weight) * sanitized(source value)``.

Consequently the module defaults are ``NA = 4``, ``CK = 2``, ``CA = 1`` — the
widths a default ``DefaultGenome``/``RecurrentGenome`` actually allocates. All
three are knobs, so a test can build a narrower (legacy-style, ``NA = 3``) or
wider (``OriginConn``, ``CK = 3``) payload; only the *column order* is fixed.

## Conventions every builder here follows

* A genome's node key equals its row index: present nodes are rows
  ``0 .. num_inputs-1`` (inputs) and ``max_nodes-num_outputs .. max_nodes-1``
  (outputs), matching the row/key invariant of ``BaseGenome``; hidden rows sit
  in between, so ``get_input_idx()``/``get_output_idx()`` double as row
  indices and as keys.
* Unused rows keep the padding sentinels, so ``isnan(node_attrs[..., 0])`` is
  exactly "row absent".
* The neutral rows :func:`set_node` / :func:`set_conn` write when ``attrs`` is
  omitted are the documented no-ops: node ``(bias=0, response=1, ids=0)`` (the
  aggregation passes through, option ``0`` is the gene's configured default)
  and connection ``(weight=1)`` (the message reproduces its source).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from tensorneat.common.tools import I_INF

__all__ = [
    "I_INF",
    "Payload",
    "new_payload",
    "set_node",
    "set_conn",
    "default_genome",
    "recurrent_genome",
    "genome_state",
    "chain_payload",
    "skip_payload",
    "isolated_payload",
    "ring_payload",
    "DEFAULT_POP_SIZE",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_CONNS",
    "DEFAULT_NUM_INPUTS",
    "DEFAULT_NUM_OUTPUTS",
    "DEFAULT_NODE_ATTR_WIDTH",
    "DEFAULT_CONN_KEY_WIDTH",
    "DEFAULT_CONN_ATTR_WIDTH",
]

#: Default ``node_gene.attr_width`` (``DefaultNode``: bias, response, agg id, act id).
DEFAULT_NODE_ATTR_WIDTH = 4
#: Default ``conn_gene.key_width`` (``BaseConn``: input key, output key).
DEFAULT_CONN_KEY_WIDTH = 2
#: Default ``conn_gene.attr_width`` (``DefaultConn``: weight).
DEFAULT_CONN_ATTR_WIDTH = 1

#: Population size of a hand-built payload: one individual unless a test wants a batch.
DEFAULT_POP_SIZE = 1
#: Node capacity used by the canonical payloads and the genome factories.
DEFAULT_MAX_NODES = 8
#: Connection capacity used by the canonical payloads and the genome factories.
DEFAULT_MAX_CONNS = 16
#: Genome input width used by the canonical payloads and the genome factories.
DEFAULT_NUM_INPUTS = 2
#: Genome output width used by the canonical payloads and the genome factories.
DEFAULT_NUM_OUTPUTS = 1


# --------------------------------------------------------------------------- helpers
def _as_int(values: Any, name: str) -> int:
    """Return ``values`` as a Python ``int``, rejecting bools and non-integers."""
    if isinstance(values, bool) or not isinstance(values, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(values).__name__}")
    return int(values)


def _positive(values: Any, name: str) -> int:
    """Return ``values`` as a positive Python ``int`` (widths, capacities, sizes)."""
    number = _as_int(values, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _int32_array(values: Any, name: str) -> np.ndarray:
    """View ``values`` as an ``int32`` array, rejecting float/bool input."""
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integer values, got dtype {array.dtype}")
    return array.astype(np.int32, copy=False)


def _float32_array(values: Any, name: str) -> np.ndarray:
    """View ``values`` as a ``float32`` array, rejecting integer input."""
    array = np.asarray(values)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must hold float values, got dtype {array.dtype}")
    return array.astype(np.float32, copy=False)


def _node_attr_row(attr_width: int) -> np.ndarray:
    """The neutral ``DefaultNode`` attribute row: ``bias=0``, ``response=1``, ids ``0``."""
    row = np.zeros(attr_width, dtype=np.float32)
    if attr_width >= 2:
        row[1] = 1.0
    return row


def _conn_attr_row(attr_width: int, weight: float = 1.0) -> np.ndarray:
    """The neutral ``DefaultConn`` attribute row: ``weight=1`` (message = source value)."""
    row = np.zeros(attr_width, dtype=np.float32)
    row[0] = float(weight)
    return row


# --------------------------------------------------------------------------- payload
class Payload:
    """The four payload arrays of one batch of genomes, in ``GenomeTensors`` order.

    Batch-first like everything else in the package (population on axis 0), with
    the static shapes ``P = pop_size``, ``N = max_nodes``, ``E = max_conns`` and
    the per-gene widths ``NA``/``CK``/``CA``. Absent rows carry the padding
    sentinels (``I_INF`` in ``*_keys``, ``NaN`` in ``*_attrs``); the two sentinel
    families are never mixed.

    This is deliberately *not* ``GenomeTensors``: that one holds traced
    ``evoxc.SymbolicTensor`` leaves, this one holds host ``numpy`` arrays, so a
    test can print, slice and tamper with it. ``.args()`` is the bridge — it
    returns the tuple :func:`support.runner.run_defn` expects.

    Args:
        node_keys: ``(P, N) int32`` node key per row, ``I_INF`` for an absent row.
        node_attrs: ``(P, N, NA) float32`` node attribute rows, ``NaN`` for an
            absent row.
        conn_keys: ``(P, E, CK) int32`` ``(in_key, out_key[, historical_marker])``
            per connection row, ``I_INF`` for an absent row.
        conn_attrs: ``(P, E, CA) float32`` connection attribute rows, ``NaN`` for
            an absent row.

    Raises:
        TypeError: If a key tensor is not integer-valued or an attribute tensor
            is not float-valued (a payload never silently casts a float key
            into an integer one).
        ValueError: If the shapes disagree on ``P`` or have the wrong rank.
    """

    field_names: tuple[str, ...] = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

    __slots__ = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")

    def __init__(
        self,
        node_keys: Any,
        node_attrs: Any,
        conn_keys: Any,
        conn_attrs: Any,
    ):
        self.node_keys = _int32_array(node_keys, "node_keys")
        self.node_attrs = _float32_array(node_attrs, "node_attrs")
        self.conn_keys = _int32_array(conn_keys, "conn_keys")
        self.conn_attrs = _float32_array(conn_attrs, "conn_attrs")
        if self.node_keys.ndim != 2:
            raise ValueError(f"node_keys must be (P, N), got {self.node_keys.shape}")
        pop_size, max_nodes = self.node_keys.shape
        if self.node_attrs.ndim != 3 or self.node_attrs.shape[:2] != (pop_size, max_nodes):
            raise ValueError(
                "node_attrs must be (P, N, NA) matching node_keys "
                f"{self.node_keys.shape}, got {self.node_attrs.shape}"
            )
        if self.node_attrs.shape[2] < 1:
            raise ValueError("node_attrs needs at least one attribute column")
        if self.conn_keys.ndim != 3:
            raise ValueError(f"conn_keys must be (P, E, CK), got {self.conn_keys.shape}")
        if self.conn_attrs.ndim != 3:
            raise ValueError(f"conn_attrs must be (P, E, CA), got {self.conn_attrs.shape}")
        if self.conn_keys.shape[0] != pop_size or self.conn_attrs.shape[0] != pop_size:
            raise ValueError(
                "every payload tensor must share the population axis "
                f"{pop_size}, got conn_keys {self.conn_keys.shape} and "
                f"conn_attrs {self.conn_attrs.shape}"
            )
        if self.conn_keys.shape[1] != self.conn_attrs.shape[1]:
            raise ValueError(
                "conn_keys and conn_attrs must share the connection axis, got "
                f"{self.conn_keys.shape} and {self.conn_attrs.shape}"
            )
        if self.conn_keys.shape[2] < 2:
            raise ValueError(
                "conn_keys needs at least the (input_index, output_index) columns, "
                f"got {self.conn_keys.shape[2]}"
            )
        if self.conn_attrs.shape[2] < 1:
            raise ValueError("conn_attrs needs at least one attribute column")

    # ------------------------------------------------------------------ geometry
    @property
    def pop_size(self) -> int:
        """``P``, the population size (axis 0 of every tensor)."""
        return int(self.node_keys.shape[0])

    @property
    def max_nodes(self) -> int:
        """``N``, the padded node-row count."""
        return int(self.node_keys.shape[1])

    @property
    def max_conns(self) -> int:
        """``E``, the padded connection-row count."""
        return int(self.conn_keys.shape[1])

    @property
    def node_attr_width(self) -> int:
        """``NA``, the node attribute column count (``4`` for ``DefaultNode``)."""
        return int(self.node_attrs.shape[2])

    @property
    def conn_key_width(self) -> int:
        """``CK``, the connection key column count (``2`` for ``BaseConn``)."""
        return int(self.conn_keys.shape[2])

    @property
    def conn_attr_width(self) -> int:
        """``CA``, the connection attribute column count (``1`` for ``DefaultConn``)."""
        return int(self.conn_attrs.shape[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(pop_size, max_nodes, max_conns)`` — the static shape signature."""
        return (self.pop_size, self.max_nodes, self.max_conns)

    # ------------------------------------------------------------------ bridging
    def args(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The four arrays in ``GenomeTensors`` order, ready for ``run_defn``."""
        return (self.node_keys, self.node_attrs, self.conn_keys, self.conn_attrs)

    def of(self, p: int) -> "Payload":
        """One individual's view, as a ``P == 1`` batch.

        Kept batched on purpose: every consumer in ``src/tensorneat`` (``forward``,
        ``network_dict``, ``repr``) takes batch-first arrays, and a ``P == 1``
        batch is what ``network_dict`` documents as "one individual".

        Args:
            p: Individual index in ``[0, pop_size)``.

        Returns:
            A new ``Payload`` holding row ``p`` of each tensor, with shapes
            ``(1, N)``, ``(1, N, NA)``, ``(1, E, CK)``, ``(1, E, CA)``.

        Raises:
            IndexError: If ``p`` is outside ``[0, pop_size)``.
        """
        index = _as_int(p, "p")
        if not 0 <= index < self.pop_size:
            raise IndexError(f"individual {index} outside [0, {self.pop_size})")
        return Payload(
            self.node_keys[index : index + 1],
            self.node_attrs[index : index + 1],
            self.conn_keys[index : index + 1],
            self.conn_attrs[index : index + 1],
        )

    def __getitem__(self, item: Any) -> Any:
        """``payload[p]`` is :meth:`of`; ``payload["node_keys"]`` is a field array."""
        if isinstance(item, str):
            if item not in self.field_names:
                raise KeyError(
                    f"unknown payload field {item!r}; expected one of {self.field_names}"
                )
            return getattr(self, item)
        return self.of(item)

    def __repr__(self) -> str:
        return (
            f"Payload(pop_size={self.pop_size}, max_nodes={self.max_nodes}, "
            f"max_conns={self.max_conns}, node_attr_width={self.node_attr_width}, "
            f"conn_key_width={self.conn_key_width}, "
            f"conn_attr_width={self.conn_attr_width})"
        )


def new_payload(
    pop_size: int = DEFAULT_POP_SIZE,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_conns: int = DEFAULT_MAX_CONNS,
    *,
    node_attr_width: int = DEFAULT_NODE_ATTR_WIDTH,
    conn_key_width: int = DEFAULT_CONN_KEY_WIDTH,
    conn_attr_width: int = DEFAULT_CONN_ATTR_WIDTH,
) -> Payload:
    """Allocate an empty payload: every row absent, at the documented sentinels.

    Args:
        pop_size: ``P``, the population size.
        max_nodes: ``N``, the padded node-row count.
        max_conns: ``E``, the padded connection-row count.
        node_attr_width: ``NA``, node attribute columns (``4`` = ``DefaultNode``).
        conn_key_width: ``CK``, connection key columns (``2`` = ``BaseConn``,
            ``3`` = ``OriginConn`` with its ``historical_marker``).
        conn_attr_width: ``CA``, connection attribute columns (``1`` = ``DefaultConn``).

    Returns:
        A ``Payload`` whose ``node_keys``/``conn_keys`` are ``I_INF`` and whose
        ``node_attrs``/``conn_attrs`` are ``NaN``.

    Raises:
        TypeError: If a size or width is not an ``int``.
        ValueError: If a size or width is not positive, or ``conn_key_width < 2``.
    """
    pop_size = _positive(pop_size, "pop_size")
    max_nodes = _positive(max_nodes, "max_nodes")
    max_conns = _positive(max_conns, "max_conns")
    node_attr_width = _positive(node_attr_width, "node_attr_width")
    conn_key_width = _positive(conn_key_width, "conn_key_width")
    conn_attr_width = _positive(conn_attr_width, "conn_attr_width")
    if conn_key_width < 2:
        raise ValueError(
            "conn_key_width must be at least 2 (input_index, output_index), "
            f"got {conn_key_width}"
        )
    return Payload(
        np.full((pop_size, max_nodes), I_INF, dtype=np.int32),
        np.full((pop_size, max_nodes, node_attr_width), np.nan, dtype=np.float32),
        np.full((pop_size, max_conns, conn_key_width), I_INF, dtype=np.int32),
        np.full((pop_size, max_conns, conn_attr_width), np.nan, dtype=np.float32),
    )


def set_node(
    payload: Payload,
    p: int,
    row: int,
    key: int,
    attrs: Sequence[float] | np.ndarray | None = None,
) -> Payload:
    """Write one node gene row in place.

    ``attrs is None`` writes the neutral ``DefaultNode`` row
    ``(bias=0, response=1, aggregation=0, activation=0)`` (see
    :func:`_node_attr_row`). Writing ``key == I_INF`` clears the row instead:
    the key columns get ``I_INF`` and the attribute row ``NaN``, so the
    "absent row" invariant of ``GenomeTensors`` is never broken.

    Args:
        payload: The payload to edit; mutated in place.
        p: Individual index in ``[0, pop_size)``.
        row: Node row in ``[0, max_nodes)``; rows ``0..num_inputs-1`` and
            ``max_nodes-num_outputs..max_nodes-1`` are the input/output rows of the
            row/key invariant.
        key: ``int32`` node key written to ``node_keys[p, row]``.
        attrs: ``NA`` attribute values (``bias, response, aggregation id, activation
            id, ...``), or ``None`` for the neutral row.

    Returns:
        The same ``payload``, so a builder can chain calls.

    Raises:
        IndexError: If ``p`` or ``row`` is out of range.
        ValueError: If ``attrs`` does not have exactly ``node_attr_width`` entries.
    """
    index = _as_int(p, "p")
    node_row = _as_int(row, "row")
    if not 0 <= index < payload.pop_size:
        raise IndexError(f"individual {index} outside [0, {payload.pop_size})")
    if not 0 <= node_row < payload.max_nodes:
        raise IndexError(f"node row {node_row} outside [0, {payload.max_nodes})")
    node_key = _as_int(key, "key")
    if node_key == I_INF:
        payload.node_keys[index, node_row] = I_INF
        payload.node_attrs[index, node_row, :] = np.nan
        return payload
    if attrs is None:
        attribute_row = _node_attr_row(payload.node_attr_width)
    else:
        attribute_row = _float32_array(attrs, "attrs").reshape(-1)
        if attribute_row.shape != (payload.node_attr_width,):
            raise ValueError(
                f"attrs must have {payload.node_attr_width} entries "
                f"(the payload's node_attr_width), got {attribute_row.shape}"
            )
    payload.node_keys[index, node_row] = node_key
    payload.node_attrs[index, node_row, :] = attribute_row
    return payload


def set_conn(
    payload: Payload,
    p: int,
    row: int,
    in_key: int,
    out_key: int,
    attrs: Sequence[float] | np.ndarray | None = None,
) -> Payload:
    """Write one connection gene row in place.

    ``attrs is None`` writes the neutral ``DefaultConn`` row ``(weight=1)``, whose
    message reproduces its source value (see :func:`_conn_attr_row`). Key columns
    beyond ``(input_index, output_index)`` are written as the connection **row
    index**, which is the documented ``historical_marker`` value
    (``BaseGenome.initialize``). Writing ``in_key == I_INF`` or ``out_key ==
    I_INF`` clears the row instead (all key columns ``I_INF``, all attribute
    columns ``NaN``).

    Args:
        payload: The payload to edit; mutated in place.
        p: Individual index in ``[0, pop_size)``.
        row: Connection row in ``[0, max_conns)``.
        in_key: Source node key (``conn_keys[..., 0]``).
        out_key: Destination node key (``conn_keys[..., 1]``).
        attrs: ``CA`` attribute values (``weight, ...``), or ``None`` for ``(weight=1)``.

    Returns:
        The same ``payload``, so a builder can chain calls.

    Raises:
        IndexError: If ``p`` or ``row`` is out of range.
        ValueError: If ``attrs`` does not have exactly ``conn_attr_width`` entries.
    """
    index = _as_int(p, "p")
    conn_row = _as_int(row, "row")
    if not 0 <= index < payload.pop_size:
        raise IndexError(f"individual {index} outside [0, {payload.pop_size})")
    if not 0 <= conn_row < payload.max_conns:
        raise IndexError(f"connection row {conn_row} outside [0, {payload.max_conns})")
    source = _as_int(in_key, "in_key")
    target = _as_int(out_key, "out_key")
    if source == I_INF or target == I_INF:
        payload.conn_keys[index, conn_row, :] = I_INF
        payload.conn_attrs[index, conn_row, :] = np.nan
        return payload
    if attrs is None:
        attribute_row = _conn_attr_row(payload.conn_attr_width)
    else:
        attribute_row = _float32_array(attrs, "attrs").reshape(-1)
        if attribute_row.shape != (payload.conn_attr_width,):
            raise ValueError(
                f"attrs must have {payload.conn_attr_width} entries "
                f"(the payload's conn_attr_width), got {attribute_row.shape}"
            )
    payload.conn_keys[index, conn_row, 0] = source
    payload.conn_keys[index, conn_row, 1] = target
    payload.conn_keys[index, conn_row, 2:] = conn_row
    payload.conn_attrs[index, conn_row, :] = attribute_row
    return payload


# --------------------------------------------------------------------------- factories
def default_genome(
    num_inputs: int = DEFAULT_NUM_INPUTS,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_conns: int = DEFAULT_MAX_CONNS,
    **kwargs: Any,
):
    """A ``DefaultGenome`` with the default gene/operator stack and small shapes.

    Only constructs the genome (and thereby its dense initial network) — it never
    calls ``initialize``/``forward``, so the factory stays cheap. A constructor
    that raises ``NotImplementedError`` is *not* worked around here: the error
    belongs to the source.

    Args:
        num_inputs: Input node count; also the network input width.
        num_outputs: Output node count; also the network output width.
        max_nodes: Static node capacity ``N``.
        max_conns: Static connection capacity ``E``.
        **kwargs: Forwarded to ``DefaultGenome`` (``max_in_degree``,
            ``init_hidden_layers``, ``input_transform``, ``output_transform``, ...).

    Returns:
        A ``DefaultGenome`` instance whose ``network_type`` is ``"feedforward"``.
    """
    from tensorneat.genome import (
        DefaultConn,
        DefaultCrossover,
        DefaultDistance,
        DefaultGenome,
        DefaultMutation,
        DefaultNode,
    )

    return DefaultGenome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=max_nodes,
        max_conns=max_conns,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        **kwargs,
    )


def recurrent_genome(
    num_inputs: int = DEFAULT_NUM_INPUTS,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_conns: int = DEFAULT_MAX_CONNS,
    **kwargs: Any,
):
    """A ``RecurrentGenome`` with the default gene/operator stack and small shapes.

    Same contract as :func:`default_genome`, for the tick-unrolled genome whose
    ``network_type`` is ``"recurrent"`` (it never applies ``input_transform``).

    Args:
        num_inputs: Input node count; also the network input width.
        num_outputs: Output node count; also the network output width.
        max_nodes: Static node capacity ``N``.
        max_conns: Static connection capacity ``E``.
        **kwargs: Forwarded to ``RecurrentGenome`` (``activate_time``,
            ``max_in_degree``, ``init_hidden_layers``, ``output_transform``, ...).

    Returns:
        A ``RecurrentGenome`` instance.
    """
    from tensorneat.genome import (
        DefaultConn,
        DefaultCrossover,
        DefaultDistance,
        DefaultMutation,
        DefaultNode,
        RecurrentGenome,
    )

    return RecurrentGenome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=max_nodes,
        max_conns=max_conns,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        **kwargs,
    )


def genome_state(genome: Any):
    """The ``State`` a genome registers its configuration on.

    Args:
        genome: A constructed genome (see :func:`default_genome`).

    Returns:
        The immutable state returned by ``genome.setup(State())`` — the value
        every kernel of that genome expects as its ``state`` argument.
    """
    from tensorneat.common import State

    return genome.setup(State())


# --------------------------------------------------------------------------- topologies
def _geometry(
    max_nodes: int,
    num_inputs: int,
    num_outputs: int,
    hidden: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split the node rows of the row/key invariant into ``(inputs, hidden, outputs)``.

    Args:
        max_nodes: ``N``, the padded node-row count.
        num_inputs: Input rows ``0 .. num_inputs-1``.
        num_outputs: Output rows ``max_nodes-num_outputs .. max_nodes-1``.
        hidden: How many hidden rows the topology needs.

    Returns:
        ``(input_rows, hidden_rows, output_rows)`` as Python lists of row indices.

    Raises:
        ValueError: If ``hidden`` is negative, or if the three regions do not fit
            into ``max_nodes`` rows (they would overlap).
    """
    if hidden < 0:
        raise ValueError(
            f"topology needs at least {num_inputs + num_outputs} node rows "
            f"({num_inputs} inputs + {num_outputs} outputs) but max_nodes is "
            f"{max_nodes}"
        )
    needed = num_inputs + hidden + num_outputs
    if needed > max_nodes:
        raise ValueError(
            f"topology needs {needed} node rows ({num_inputs} inputs + {hidden} "
            f"hidden + {num_outputs} outputs) but max_nodes is {max_nodes}"
        )
    input_rows = list(range(num_inputs))
    hidden_rows = list(range(num_inputs, num_inputs + hidden))
    output_rows = list(range(max_nodes - num_outputs, max_nodes))
    return input_rows, hidden_rows, output_rows


def _fill(
    payload: Payload,
    rows: Sequence[int],
    edges: Sequence[tuple[int, int]],
    weight: float,
) -> Payload:
    """Write identical node rows and edges into every individual of a fresh payload.

    Args:
        payload: The payload to fill (from :func:`new_payload`).
        rows: Node rows to mark present; the node key of each is its row index.
        edges: ``(source_row, destination_row)`` pairs, written in order, each with
            the neutral connection attribute row and weight ``weight``.
        weight: ``DefaultConn`` weight of every edge.

    Returns:
        The same ``payload``.

    Raises:
        ValueError: If the topology needs more connection rows than ``max_conns``.
    """
    if len(edges) > payload.max_conns:
        raise ValueError(
            f"topology needs {len(edges)} connection rows but max_conns is "
            f"{payload.max_conns}"
        )
    for p in range(payload.pop_size):
        for row in rows:
            set_node(payload, p, row, row)
        for conn_row, (source, target) in enumerate(edges):
            set_conn(
                payload,
                p,
                conn_row,
                source,
                target,
                _conn_attr_row(payload.conn_attr_width, weight),
            )
    return payload


def _new(
    pop_size: int,
    max_nodes: int,
    max_conns: int,
    node_attr_width: int,
    conn_key_width: int,
    conn_attr_width: int,
) -> Payload:
    """``new_payload`` with the canonical builders' keyword widths."""
    return new_payload(
        pop_size,
        max_nodes,
        max_conns,
        node_attr_width=node_attr_width,
        conn_key_width=conn_key_width,
        conn_attr_width=conn_attr_width,
    )


def chain_payload(
    pop_size: int = DEFAULT_POP_SIZE,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_conns: int = DEFAULT_MAX_CONNS,
    *,
    num_inputs: int = DEFAULT_NUM_INPUTS,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    node_attr_width: int = DEFAULT_NODE_ATTR_WIDTH,
    conn_key_width: int = DEFAULT_CONN_KEY_WIDTH,
    conn_attr_width: int = DEFAULT_CONN_ATTR_WIDTH,
    weight: float = 1.0,
) -> Payload:
    """Dense chain: every input feeds the first hidden row, then one edge per row.

    With the default shapes (``max_nodes=8``, ``num_inputs=2``, ``num_outputs=1``)
    and node key == row index::

        row     0     1     2     3     4     5     6     7
        role   in0   in1   hid   hid   hid   hid   hid   out
        edges  0->2  1->2  2->3  3->4  4->5  5->6  6->7
        level   0     0     1     2     3     4     5     6

    So every non-input row is reachable, the depth grows by one per row and the
    output row sits at level ``max_nodes - num_inputs - 1``. The last
    ``num_outputs`` rows are chained into each other, so a multi-output chain is
    a sequence as well.

    Args:
        pop_size: Number of identical individuals.
        max_nodes: Node capacity ``N``; every row is present.
        max_conns: Connection capacity ``E``; must hold ``num_inputs + N -
            num_inputs - 1`` edges.
        num_inputs: Input width; input rows are ``0 .. num_inputs-1``.
        num_outputs: Output width; output rows are the last ``num_outputs`` rows.
        node_attr_width: Node attribute columns (``4`` = ``DefaultNode``).
        conn_key_width: Connection key columns (``2`` = ``BaseConn``).
        conn_attr_width: Connection attribute columns (``1`` = ``DefaultConn``).
        weight: Weight of every edge (its message is ``weight * source value``).

    Returns:
        The filled ``Payload``.

    Raises:
        ValueError: If the topology does not fit ``max_nodes``/``max_conns``.
    """
    input_rows, hidden_rows, output_rows = _geometry(
        max_nodes, num_inputs, num_outputs, max_nodes - num_inputs - num_outputs
    )
    spine = hidden_rows + output_rows
    edges = [(source, spine[0]) for source in input_rows]
    edges += list(zip(spine, spine[1:]))
    payload = _new(pop_size, max_nodes, max_conns, node_attr_width, conn_key_width, conn_attr_width)
    return _fill(payload, input_rows + hidden_rows + output_rows, edges, weight)


def skip_payload(
    pop_size: int = DEFAULT_POP_SIZE,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_conns: int = DEFAULT_MAX_CONNS,
    *,
    num_inputs: int = DEFAULT_NUM_INPUTS,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    node_attr_width: int = DEFAULT_NODE_ATTR_WIDTH,
    conn_key_width: int = DEFAULT_CONN_KEY_WIDTH,
    conn_attr_width: int = DEFAULT_CONN_ATTR_WIDTH,
    weight: float = 1.0,
) -> Payload:
    """A node bypassed by its consumer: the second hidden row reads input ``0`` directly.

    With the default shapes and node key == row index::

        row     0     1     2     3     4   ...   7
        role   in0   in1  skip  hid   --   ...  out
        edges  0->2  1->2  0->3  3->7
        level   0     0     1    1     --     2

    Node ``2`` receives the full input signal (in-degree ``num_inputs``) but its
    natural consumer (row ``3``) bypasses it and reads input ``0`` instead, so
    row ``2`` never reaches an output: mutating or NaN-poisoning row ``2`` must
    not change the network output, and row ``3`` sits at level ``1`` — not at
    ``2`` — which is exactly what a level-synchronous schedule has to get right.

    Args:
        pop_size: Number of identical individuals.
        max_nodes: Node capacity ``N``; only the rows above are present.
        max_conns: Connection capacity ``E``; must hold ``num_inputs + 2`` edges.
        num_inputs: Input width; the bypassed node reads input ``0``.
        num_outputs: Output width; every edge of the second hidden row feeds each
            output row.
        node_attr_width: Node attribute columns (``4`` = ``DefaultNode``).
        conn_key_width: Connection key columns (``2`` = ``BaseConn``).
        conn_attr_width: Connection attribute columns (``1`` = ``DefaultConn``).
        weight: Weight of every edge.

    Returns:
        The filled ``Payload``.

    Raises:
        ValueError: If the topology does not fit ``max_nodes``/``max_conns``
            (it needs at least ``num_inputs + 2 + num_outputs`` node rows).
    """
    input_rows, hidden_rows, output_rows = _geometry(max_nodes, num_inputs, num_outputs, 2)
    bypassed, consumer = hidden_rows[0], hidden_rows[1]
    edges = [(source, bypassed) for source in input_rows]
    edges.append((input_rows[0], consumer))
    edges += [(consumer, out_row) for out_row in output_rows]
    payload = _new(pop_size, max_nodes, max_conns, node_attr_width, conn_key_width, conn_attr_width)
    rows = input_rows + [bypassed, consumer] + output_rows
    return _fill(payload, rows, edges, weight)


def isolated_payload(
    pop_size: int = DEFAULT_POP_SIZE,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_conns: int = DEFAULT_MAX_CONNS,
    *,
    num_inputs: int = DEFAULT_NUM_INPUTS,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    node_attr_width: int = DEFAULT_NODE_ATTR_WIDTH,
    conn_key_width: int = DEFAULT_CONN_KEY_WIDTH,
    conn_attr_width: int = DEFAULT_CONN_ATTR_WIDTH,
    weight: float = 1.0,
) -> Payload:
    """A present node with no edges at all, next to a dense input-to-output layer.

    With the default shapes and node key == row index::

        row     0     1     2     3   ...   7
        role   in0   in1   lone   --   ...  out
        edges  0->7  1->7
        level   0     0    I_INF    --      1

    Row ``num_inputs`` is present (its key is not ``I_INF``, its attribute row is
    finite) but has neither an incoming nor an outgoing edge. Its value therefore
    stays the ``NaN`` seed and its level is ``I_INF`` — the "unconnected node
    value is NaN" semantics the parity suite pins down. Its descendants (none
    here) would inherit the ``NaN``.

    Args:
        pop_size: Number of identical individuals.
        max_nodes: Node capacity ``N``; only the rows above are present.
        max_conns: Connection capacity ``E``; must hold ``num_inputs * num_outputs``
            edges.
        num_inputs: Input width.
        num_outputs: Output width; every input row feeds every output row.
        node_attr_width: Node attribute columns (``4`` = ``DefaultNode``).
        conn_key_width: Connection key columns (``2`` = ``BaseConn``).
        conn_attr_width: Connection attribute columns (``1`` = ``DefaultConn``).
        weight: Weight of every edge.

    Returns:
        The filled ``Payload``.

    Raises:
        ValueError: If the topology does not fit ``max_nodes``/``max_conns``.
    """
    input_rows, hidden_rows, output_rows = _geometry(max_nodes, num_inputs, num_outputs, 1)
    edges = [(source, out_row) for source in input_rows for out_row in output_rows]
    payload = _new(pop_size, max_nodes, max_conns, node_attr_width, conn_key_width, conn_attr_width)
    rows = input_rows + hidden_rows + output_rows
    return _fill(payload, rows, edges, weight)


def ring_payload(
    pop_size: int = DEFAULT_POP_SIZE,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_conns: int = DEFAULT_MAX_CONNS,
    *,
    num_inputs: int = DEFAULT_NUM_INPUTS,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    node_attr_width: int = DEFAULT_NODE_ATTR_WIDTH,
    conn_key_width: int = DEFAULT_CONN_KEY_WIDTH,
    conn_attr_width: int = DEFAULT_CONN_ATTR_WIDTH,
    weight: float = 1.0,
) -> Payload:
    """A recurrent ring: two hidden rows feeding each other, read out by the outputs.

    With the default shapes and node key == row index::

        row     0     1     2     3     4   ...   7
        role   in0   in1   h1    h2    --   ...  out
        edges  0->2  1->2  2->3  3->2  3->7
        level   0     0   I_INF I_INF   --   I_INF

    Rows ``2`` and ``3`` are on a cycle, so neither they nor anything downstream
    of them has a finite longest-path level: ``level`` is ``I_INF`` for all of
    them, which is what makes this payload the recurrent case (the tick loop of
    ``RecurrentGenome`` ignores ``level`` and re-injects the raw inputs every
    tick). It is *not* usable with a level-synchronous forward pass.

    Args:
        pop_size: Number of identical individuals.
        max_nodes: Node capacity ``N``; only the rows above are present.
        max_conns: Connection capacity ``E``; must hold ``num_inputs + 3`` edges.
        num_inputs: Input width; every input row feeds ``h1``.
        num_outputs: Output width; ``h2`` feeds every output row.
        node_attr_width: Node attribute columns (``4`` = ``DefaultNode``).
        conn_key_width: Connection key columns (``2`` = ``BaseConn``).
        conn_attr_width: Connection attribute columns (``1`` = ``DefaultConn``).
        weight: Weight of every edge.

    Returns:
        The filled ``Payload``.

    Raises:
        ValueError: If the topology does not fit ``max_nodes``/``max_conns``
            (it needs at least ``num_inputs + 2 + num_outputs`` node rows).
    """
    input_rows, hidden_rows, output_rows = _geometry(max_nodes, num_inputs, num_outputs, 2)
    first, second = hidden_rows[0], hidden_rows[1]
    edges = [(source, first) for source in input_rows]
    edges += [(first, second), (second, first)]
    edges += [(second, out_row) for out_row in output_rows]
    payload = _new(pop_size, max_nodes, max_conns, node_attr_width, conn_key_width, conn_attr_width)
    rows = input_rows + [first, second] + output_rows
    return _fill(payload, rows, edges, weight)
