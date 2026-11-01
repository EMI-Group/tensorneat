"""Two-way conversion between the legacy JAX gene payload and the evoxc payload.

Parity compares *kernels on identical inputs*, so the first job of this module is
to make one hand-built genome acceptable to both libraries.  It is plain Python:
no ``evoxc``, no ``jax`` arithmetic, no tracing — just numpy reshaping of host
arrays, so it can be unit-tested (``./test_convert.py``) long before either
library is finished.

## The two column layouts (read off the sources, not guessed)

The rewrite splits a gene row into **integer keys** and **float attributes**
(``src/tensorneat/genome/gene/base.py``: ``key_width == len(fixed_attrs)``,
``attr_width == len(attr_specs)``)::

    node_keys   (P, N)    int32    node key per row, ``I_INF`` = absent row
    node_attrs  (P, N,NA) float32  (bias, response, aggregation, activation, ...)
    conn_keys   (P, E,CK) int32    (input_index, output_index[, historical_marker])
    conn_attrs  (P, E,CA) float32  (weight, ...)

The frozen pre-rewrite library keeps one float row per gene
(``benchmarks/legacy_jax/tensorneat_legacy/genome/gene/base.py``:
``length == len(fixed_attrs) + len(custom_attrs)``) and marks an absent row with
``NaN`` everywhere::

    nodes  (N, 1 + NA)        float32  col 0 = node key, cols 1: = node attrs
    conns  (E, CK + CA)       float32  cols :CK = (in key, out key[, marker]), rest = attrs

``DefaultNode`` contributes one fixed column (``index``) and four custom ones
(``bias, response, aggregation, activation``); ``BaseConn``/``DefaultConn``
contribute two fixed columns (``input_index, output_index``) and one custom one
(``weight``).  ``OriginConn`` appends a third fixed column
(``historical_marker``), which is why :func:`from_legacy` takes a
``conn_key_width``: the legacy side stores no width metadata, so the caller has
to name the gene it converted from.

So the conversion is exactly:

* key column(s): ``int32`` key ``<->`` integer-valued ``float32``,
  ``I_INF`` ``<->`` ``NaN`` (the two "absent" sentinels of the two libraries);
* attribute columns: copied unchanged, ``float32`` on both sides, ``NaN``
  padding included — a ``NaN`` bias is legitimate data (it is the legacy NaN
  poisoning of an unconnected node), not a padding artefact to clean up.

## The ``max_in_degree`` rule (:func:`max_in_degree_for`)

The legacy forward aggregates over a full ``max_nodes``-wide column
(``u_conns[:, i]`` is one column of the dense ``(max_nodes, max_nodes)`` matrix)
and truncates **nothing**; the rewrite keeps a ``(P, max_nodes, K)`` incoming
window and reports whatever does not fit in ``Transformed.edge_overflow``.  A
parity genome must therefore be built with ``K = max_nodes``; the fixtures in
``./conftest.py`` assert ``edge_overflow == 0`` before any value is compared, so
a silently dropped edge fails the test instead of drifting the numbers.

## The legacy genome layout rule (:func:`legacy_init_hidden_layers`)

The legacy ``output_idx`` is derived from ``init_hidden_layers`` (the last dense
layer), while the rewrite's row/key invariant puts the output keys at rows
``max_nodes - num_outputs .. max_nodes - 1``.  A legacy genome built with the
default ``init_hidden_layers=()`` would read the outputs from keys
``num_inputs .. num_inputs + num_outputs - 1`` instead — a *different* network.
:func:`legacy_init_hidden_layers` returns the one hidden-layer width that makes
the two agree, and :func:`legacy_initial_edge_count` the dense edge count that
``max_conns`` must be able to hold (the legacy constructor rejects a smaller
capacity).

## Sentinels duplicated here on purpose

``I_INF`` and ``KEY_MAX`` are re-declared instead of imported: importing
``tensorneat`` at this module's import time would break collection while ``src/``
is a skeleton (``import tensorneat`` currently raises ``NotImplementedError``
from ``common/state.py``).  Both are computed from numpy's ``int32``/``key``
bounds exactly as ``tensorneat/common/tools.py`` does, and ``./test_convert.py``
asserts the equality against the real constants as soon as they are importable.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = [
    "I_INF",
    "KEY_MAX",
    "LEGACY_CONN_KEY_WIDTH",
    "LEGACY_NODE_KEY_WIDTH",
    "from_legacy",
    "legacy_initial_edge_count",
    "legacy_init_hidden_layers",
    "max_in_degree_for",
    "to_legacy",
]

#: ``int32`` max, the "no index / absent row" sentinel of the rewrite.
#: Equal to ``tensorneat.common.tools.I_INF`` by construction (see the module docstring).
I_INF: int = int(np.iinfo(np.int32).max)

#: Exclusive upper bound of gene keys: the first integer ``float32`` cannot separate
#: from its neighbour, i.e. ``tensorneat.common.tools.KEY_MAX``.
KEY_MAX: int = 1 << 24

#: Fixed key columns of the legacy node gene (``BaseNode.fixed_attrs == ["index"]``).
LEGACY_NODE_KEY_WIDTH: int = 1

#: Fixed key columns of the legacy connection gene
#: (``BaseConn.fixed_attrs == ("input_index", "output_index")``; ``OriginConn`` adds one).
LEGACY_CONN_KEY_WIDTH: int = 2


# --------------------------------------------------------------------------- rules
def max_in_degree_for(max_nodes: int) -> int:
    """The ``max_in_degree`` a parity genome must be built with (``>= max_nodes``).

    The legacy forward has no in-degree window at all: node ``i`` receives one
    message per **source node slot**, i.e. up to ``max_nodes`` messages, and the
    per-edge validity mask does the truncation.  The rewrite keeps exactly ``K``
    slots per node and counts the surplus in ``Transformed.edge_overflow``.
    Choosing ``K = max_nodes`` makes the rewrite's window at least as wide as the
    legacy column for every node of every genome of that payload, so
    ``edge_overflow`` is ``0`` by construction and the two kernels see the same
    incoming edges.

    Args:
        max_nodes: Static node capacity ``N`` of the parity payload.

    Returns:
        ``max_nodes`` — never the genome's default
        ``max(initial in-degree, 4)``, which is far too small for a hand-built
        topology and would silently drop edges.

    Raises:
        TypeError: If ``max_nodes`` is not an ``int``.
        ValueError: If ``max_nodes`` is not positive.
    """
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, (int, np.integer)):
        raise TypeError(f"max_nodes must be an int, got {type(max_nodes).__name__}")
    if int(max_nodes) <= 0:
        raise ValueError(f"max_nodes must be positive, got {max_nodes}")
    return int(max_nodes)


def legacy_init_hidden_layers(
    max_nodes: int, num_inputs: int, num_outputs: int
) -> tuple[int, ...]:
    """The ``init_hidden_layers`` that makes the legacy output keys match the rewrite.

    The rewrite pins the output nodes to rows/keys
    ``max_nodes - num_outputs .. max_nodes - 1``.  The legacy library instead lays
    its initial network out as ``num_inputs -> init_hidden_layers -> num_outputs``,
    so its output keys start at ``num_inputs + sum(init_hidden_layers)``.  One
    dense hidden layer filling the gap makes the two identical::

        init_hidden_layers = (max_nodes - num_inputs - num_outputs,)
        # legacy layers: [I] -> [I .. I+H-1] -> [I+H .. I+H+O-1] == [N-O .. N-1]

    A width of ``0`` is legal and is the ``max_nodes == num_inputs +
    num_outputs`` case, where the two layouts coincide even with the legacy
    default.

    Args:
        max_nodes: Static node capacity ``N``.
        num_inputs: Input node count ``I`` (keys/rows ``0 .. I-1``).
        num_outputs: Output node count ``O`` (keys/rows ``N-O .. N-1``).

    Returns:
        A one-element tuple holding the hidden-layer width ``N - I - O``.

    Raises:
        TypeError: If an argument is not an ``int``.
        ValueError: If a size is not positive or if ``I + O > N`` (the two
            invariants would overlap).
    """
    for name, value in (
        ("max_nodes", max_nodes),
        ("num_inputs", num_inputs),
        ("num_outputs", num_outputs),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    hidden = int(max_nodes) - int(num_inputs) - int(num_outputs)
    if hidden < 0:
        raise ValueError(
            f"max_nodes={max_nodes} cannot hold {num_inputs} inputs + "
            f"{num_outputs} outputs; the row/key invariants would overlap"
        )
    return (hidden,)


def legacy_initial_edge_count(
    max_nodes: int, num_inputs: int, num_outputs: int
) -> int:
    """Dense edge count of the legacy initial network (:func:`legacy_init_hidden_layers`).

    The legacy genome constructor raises when ``max_conns`` cannot hold its dense
    initial network, so a parity layout has to satisfy
    ``max_conns >= legacy_initial_edge_count(...)`` *before* the two genomes can
    be built.  With one hidden layer of width ``H`` this is
    ``I * H + H * O`` edges.

    Args:
        max_nodes: Static node capacity ``N``.
        num_inputs: Input node count ``I``.
        num_outputs: Output node count ``O``.

    Returns:
        The number of dense initial connections.

    Raises:
        TypeError: If an argument is not an ``int``.
        ValueError: If a size is not positive or ``I + O > N``.
    """
    (hidden,) = legacy_init_hidden_layers(max_nodes, num_inputs, num_outputs)
    return int(num_inputs) * hidden + hidden * int(num_outputs)


# --------------------------------------------------------------------------- helpers
def _float_matrix(values: Any, name: str) -> np.ndarray:
    """View ``values`` as a 2-D ``float32`` array (one legacy gene matrix)."""
    array = np.asarray(values)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must hold float values, got dtype {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (one individual), got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def _int_keys(values: Any, name: str) -> np.ndarray:
    """View ``values`` as an ``int32`` key array, rejecting float/bool input."""
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integer values, got dtype {array.dtype}")
    return array.astype(np.int32, copy=False)


def _keys_from_column(column: np.ndarray, name: str) -> np.ndarray:
    """Turn one legacy key column into ``int32`` keys, ``NaN`` -> ``I_INF``.

    Args:
        column: ``(M,)`` float column holding integer-valued keys or ``NaN``.
        name: Human-readable column name for the error messages.

    Returns:
        ``(M,) int32`` keys with every ``NaN`` entry replaced by ``I_INF``.

    Raises:
        ValueError: If a present entry is not integer-valued or outside
            ``[0, KEY_MAX)`` — the documented key range, which is also the range
            a ``float32`` column can carry without loss.
    """
    values = np.asarray(column)
    absent = np.isnan(values)
    present = values[~absent].astype(np.float64)
    if present.size:
        if not np.all(present == np.floor(present)):
            raise ValueError(
                f"{name} must hold integer-valued keys; got non-integral entries "
                f"{present[present != np.floor(present)][:4].tolist()}"
            )
        if np.any(present < 0) or np.any(present >= KEY_MAX):
            raise ValueError(
                f"{name} keys must lie in [0, {KEY_MAX}); got "
                f"[{present.min():.0f}, {present.max():.0f}]"
            )
    keys = np.full(values.shape, I_INF, dtype=np.int32)
    # ``astype`` on the selected entries only: casting the whole column would first
    # round ``NaN`` (and ``I_INF``, which no float32 can represent) and warn.
    keys[~absent] = present.astype(np.int32)
    return keys


def _default_payload_factory() -> Callable[..., Any]:
    """Import ``support.payloads.Payload`` lazily (it imports ``tensorneat``).

    ``support.payloads`` reads ``tensorneat.common.tools.I_INF`` at import time,
    and ``import tensorneat`` raises ``NotImplementedError`` while ``src/`` is a
    skeleton.  Keeping the import inside the default factory lets this module be
    imported and tested regardless; a caller that needs ``from_legacy`` before
    ``src/`` is finished passes its own ``payload_factory``.
    """
    from support.payloads import Payload

    return Payload


# --------------------------------------------------------------------------- conversion
def to_legacy(payload: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert one evoxc ``Payload`` individual into the legacy gene matrices.

    Column layout produced (see the module docstring)::

        nodes  (N, 1 + NA) float32   [node key | bias, response, agg, act, ...]
        conns  (E, CK + CA) float32  [in key, out key[, marker] | weight, ...]

    where ``N``/``E`` are the payload's ``max_nodes``/``max_conns`` and ``CK`` its
    ``conn_key_width`` — every key column is carried, so nothing is dropped
    silently for an ``OriginConn`` (``CK == 3``) payload.

    An absent row (``I_INF`` in its first key column) becomes a legacy row that is
    ``NaN`` in **every one of its key columns** — that is how the legacy code
    detects an absent gene (``isnan(nodes[:, 0])`` / ``isnan(conns[:, 0])``).
    Attribute columns are copied verbatim, ``NaN`` padding included: a ``NaN``
    bias is data (the legacy NaN poisoning), not something to sanitise here.

    Args:
        payload: A ``support.payloads.Payload`` (or anything exposing the four
            ``node_keys``/``node_attrs``/``conn_keys``/``conn_attrs`` arrays) with
            ``pop_size == 1``, because a legacy genome is one individual.

    Returns:
        ``(nodes, conns)`` as ``float32`` numpy arrays.

    Raises:
        TypeError: If a key tensor is not integer-valued or an attribute tensor is
            not float-valued.
        ValueError: If the population axis is not ``1``, the tensors have the wrong
            rank, ``conn_attrs`` has no column left for the keys, or a connection
            row is only partially absent.
    """
    node_keys = _int_keys(payload.node_keys, "node_keys")
    conn_keys = _int_keys(payload.conn_keys, "conn_keys")
    node_attrs = np.asarray(payload.node_attrs)
    conn_attrs = np.asarray(payload.conn_attrs)
    if node_attrs.dtype.kind != "f" or conn_attrs.dtype.kind != "f":
        raise TypeError(
            "node_attrs/conn_attrs must hold float values, got "
            f"{node_attrs.dtype}/{conn_attrs.dtype}"
        )
    if node_keys.ndim != 2 or conn_keys.ndim != 3:
        raise ValueError(
            f"expected node_keys (P, N) and conn_keys (P, E, CK), got "
            f"{node_keys.shape} and {conn_keys.shape}"
        )
    if node_keys.shape[0] != 1:
        raise ValueError(
            f"a legacy genome is one individual; got pop_size={node_keys.shape[0]} "
            "(slice the batch first, e.g. payload.of(p))"
        )

    max_nodes = node_keys.shape[1]
    max_conns, key_width = conn_keys.shape[1], conn_keys.shape[2]
    if node_attrs.shape[:2] != (1, max_nodes) or conn_attrs.shape[:2] != (1, max_conns):
        raise ValueError(
            "attribute tensors must match the key tensors; got "
            f"{node_attrs.shape} vs {node_keys.shape} and "
            f"{conn_attrs.shape} vs {conn_keys.shape}"
        )

    nodes = np.empty((max_nodes, 1 + node_attrs.shape[2]), dtype=np.float32)
    absent_node = node_keys[0, :] == I_INF
    nodes[:, 0] = np.where(absent_node, np.nan, node_keys[0, :].astype(np.float32))
    nodes[:, 1:] = node_attrs[0]

    keys = conn_keys[0]
    absent_conn = keys[:, 0] == I_INF
    if np.any(absent_conn & np.any(keys != I_INF, axis=1)):
        rows = np.flatnonzero(absent_conn & np.any(keys != I_INF, axis=1))
        raise ValueError(
            "connection rows must be entirely I_INF when absent; rows "
            f"{rows[:4].tolist()} carry a mix of I_INF and real keys"
        )
    conns = np.empty((max_conns, key_width + conn_attrs.shape[2]), dtype=np.float32)
    for column in range(key_width):
        conns[:, column] = np.where(
            absent_conn, np.nan, keys[:, column].astype(np.float32)
        )
    conns[:, key_width:] = conn_attrs[0]
    return nodes, conns


def from_legacy(
    nodes: Any,
    conns: Any,
    *,
    conn_key_width: int = LEGACY_CONN_KEY_WIDTH,
    payload_factory: Callable[..., Any] | None = None,
) -> Any:
    """Convert one legacy genome (``nodes``, ``conns``) into a ``Payload``.

    The inverse of :func:`to_legacy`: the node key column and the first
    ``conn_key_width`` connection columns are read back as ``int32`` keys
    (``NaN`` -> ``I_INF``), every remaining column is copied as a ``float32``
    attribute.

    Args:
        nodes: ``(N, 1 + NA)`` legacy node matrix (column 0 is the node key).
        conns: ``(E, CK + CA)`` legacy connection matrix.
        conn_key_width: How many leading columns of ``conns`` are keys — ``2`` for
            ``BaseConn``/``DefaultConn``, ``3`` for ``OriginConn`` (whose third key
            column is the ``historical_marker``).  The legacy matrix stores no
            width metadata, so the caller names the gene it converted.
        payload_factory: Optional callable
            ``(node_keys, node_attrs, conn_keys, conn_attrs) -> payload``; defaults
            to ``support.payloads.Payload``, imported lazily (see
            :func:`_default_payload_factory`).

    Returns:
        A **one-individual** payload (``pop_size == 1``), i.e. the four arrays
        ``(1, N)``/``(1, N, NA)``/``(1, E, CK)``/``(1, E, CA)``.

    Raises:
        TypeError: If ``nodes``/``conns`` are not float matrices or
            ``payload_factory`` is not callable.
        ValueError: If ``conn_key_width`` cannot hold the documented key columns,
            a matrix has no attribute column left, a key is not integer-valued or
            is outside ``[0, KEY_MAX)``, or a legacy row is only partially absent.
    """
    nodes = _float_matrix(nodes, "nodes")
    conns = _float_matrix(conns, "conns")
    if isinstance(conn_key_width, bool) or not isinstance(conn_key_width, (int, np.integer)):
        raise TypeError(
            f"conn_key_width must be an int, got {type(conn_key_width).__name__}"
        )
    key_width = int(conn_key_width)
    if key_width < LEGACY_CONN_KEY_WIDTH:
        raise ValueError(
            f"conn_key_width must be at least {LEGACY_CONN_KEY_WIDTH} "
            f"(input_index, output_index), got {key_width}"
        )
    if nodes.shape[1] < LEGACY_NODE_KEY_WIDTH + 1:
        raise ValueError(
            "nodes needs the key column plus at least one attribute column, got "
            f"{nodes.shape[1]} columns"
        )
    if conns.shape[1] < key_width + 1:
        raise ValueError(
            f"conns needs {key_width} key columns plus at least one attribute "
            f"column, got {conns.shape[1]}"
        )
    if payload_factory is None:
        payload_factory = _default_payload_factory()
    if not callable(payload_factory):
        raise TypeError(f"payload_factory must be callable, got {payload_factory!r}")

    node_keys = _keys_from_column(nodes[:, 0], "nodes[:, 0]")[None, :]
    node_attrs = np.ascontiguousarray(nodes[:, 1:], dtype=np.float32)[None, ...]

    key_columns = conns[:, :key_width]
    absent = np.isnan(key_columns[:, 0])
    partially_absent = absent & np.any(~np.isnan(key_columns), axis=1)
    if np.any(partially_absent):
        rows = np.flatnonzero(partially_absent)
        raise ValueError(
            "a legacy absent connection row (NaN in column 0) must be NaN in every "
            f"key column; rows {rows[:4].tolist()} are only partially absent"
        )
    conn_keys = np.empty(key_columns.shape, dtype=np.int32)
    for column in range(key_width):
        conn_keys[:, column] = _keys_from_column(
            key_columns[:, column], f"conns[:, {column}]"
        )
    conn_attrs = np.ascontiguousarray(conns[:, key_width:], dtype=np.float32)
    return payload_factory(
        node_keys,
        node_attrs,
        conn_keys[None, ...],
        conn_attrs[None, ...],
    )
