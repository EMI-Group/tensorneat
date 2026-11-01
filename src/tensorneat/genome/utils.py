"""Batched, in-graph genome-array editing plus host-only renumbering.

Every kernel here is batch-first (population on axis 0) with static shapes, so a whole
population is edited by one elementwise/``gather``/``scatter``/``reduce`` call: no Python
loop over individuals, no ``vmap``, no ``scan``/``while_loop``/``cond``.
Absent rows are ``I_INF`` (``int32`` max) in ``*_keys`` and NaN in ``*_attrs``; the two
sentinels never mix.  ``re_cound_idx`` is host-only: it converts with ``Tensor.numpy()``
and never feeds its result back into a graph.
"""

from typing import NamedTuple

import evoxc
import evoxc.numpy as enp  # noqa: F401 - tensor creation for the implementation wave
import numpy as np

from ..common import I_INF


class GenomeTensors(NamedTuple):
    """The four payload arrays of a batch of genomes, population on axis 0.

    ``P`` is the population size, ``N`` ``max_nodes``, ``E`` ``max_conns``; ``NA``/``CK``/
    ``CA`` are ``node_gene.attr_width``, ``conn_gene.key_width`` and
    ``conn_gene.attr_width``.  All four shapes are static at trace time, so a change forces
    a rebuild of the cached executable.

    Row/key invariant: ``initialize`` writes nodes in key order, so the input keys occupy
    rows ``0..num_inputs-1`` and the output keys occupy rows
    ``max_nodes-num_outputs..max_nodes-1``.  Those rows are never deleted or moved, which is
    what makes ``get_input_idx()``/``get_output_idx()`` usable as *row indices*.  Hidden
    node keys are arbitrary and may move rows, so graph identity is compared on keys, never
    on row positions.

    Attributes:
        node_keys: ``(P, N) int32`` node key per row, ``I_INF`` for an absent row.
        node_attrs: ``(P, N, NA) float32`` node attribute rows, NaN for an absent row.
        conn_keys: ``(P, E, CK) int32`` ``(in_key, out_key)`` (plus ``historical_marker``
            when the conn gene declares it), ``I_INF`` for an absent row.
        conn_attrs: ``(P, E, CA) float32`` connection attribute rows (e.g. ``weight``),
            NaN for an absent row.
    """

    node_keys: evoxc.SymbolicTensor
    node_attrs: evoxc.SymbolicTensor
    conn_keys: evoxc.SymbolicTensor
    conn_attrs: evoxc.SymbolicTensor

    @property
    def field_names(self) -> tuple[str, ...]:
        """Field names in payload order, for generic payload walks.

        Returns:
            ``("node_keys", "node_attrs", "conn_keys", "conn_attrs")``.
        """
        return self._fields


def valid_count(kind_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Count the existing rows of one gene family, per individual.

    Args:
        kind_keys: ``(P, M) int32`` key rows of a node *or* connection family (only the
            leading key column is inspected); ``I_INF`` marks an absent row.

    Returns:
        ``(P,) int32`` number of rows whose first key column is not ``I_INF``.

    Raises:
        NotImplementedError: Skeleton stub; the reduce lands in the implementation wave.
    """
    raise NotImplementedError(
        "reduce_sum(kind_keys[:, :1] != I_INF, axis=1) -> (P,) int32"
    )


def first_free_row(kind_keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Locate the row a new gene of one family would be written to.

    Args:
        kind_keys: ``(P, M) int32`` key rows of one gene family, ``I_INF`` padded.

    Returns:
        ``(P,) int32`` first row whose first key column is ``I_INF``, or ``I_INF`` when the
        family is full (``M`` existing rows).

    Raises:
        NotImplementedError: Skeleton stub; the ``argmax`` + validity select is pending.
    """
    raise NotImplementedError(
        "first I_INF row per individual via argmax(mask) + mask-check -> (P,) int32"
    )


def fetch_random_index(
    key: evoxc.SymbolicTensor, mask: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Draw one uniformly chosen True index per row of a mask.

    ``evoxc`` has no ``choice`` op, so the draw is a uniform float per row compared against
    the normalised ``cumsum`` of the mask; the first ``cumsum`` entry reaching the target is
    the selected index, which makes the choice uniform over the True entries of that row.

    Args:
        key: Rank-0 ``evoxc.random`` key; the single draw covers the whole batch, so keys
            are never split per individual.
        mask: ``(P, M) bool`` selection mask, one row per individual.

    Returns:
        ``(P,) int32`` chosen column index, ``I_INF`` where the row has no True entry.

    Raises:
        NotImplementedError: Skeleton stub; the cumsum/compare/argmax chain is pending.
    """
    raise NotImplementedError(
        "one evoxc.random.uniform(key, (P,)) draw against cumsum(mask) -> (P,) int32, I_INF "
        "when the row has no True entry"
    )


def add_node(
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    new_keys: evoxc.SymbolicTensor,
    new_attrs: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Write new node rows at caller-chosen free rows of every individual.

    The write is unconditional at ``rows``, so a caller must gate the result with ``select``
    when ``rows == I_INF`` for some individual (as ``..operations.mutation`` does); garbage
    is written at such a row instead of the write being skipped, because no scatter can be a
    no-op per row.

    Args:
        node_keys: ``(P, N) int32`` node key rows, ``I_INF`` padded.
        node_attrs: ``(P, N, NA) float32`` node attribute rows, NaN padded.
        rows: ``(P,) int32`` destination row per individual (``first_free_row``), ``I_INF``
            to mean "no write".
        new_keys: ``(P,) int32`` caller-allocated, population-unique node keys; never drawn
            from the RNG argument.
        new_attrs: ``(P, NA) float32`` attribute rows of the new nodes.

    Returns:
        ``(node_keys, node_attrs)`` with the new rows written at ``rows``.

    Raises:
        NotImplementedError: Skeleton stub; the two row scatters are pending.
    """
    raise NotImplementedError(
        "scatter new_keys into key column 0 and new_attrs into the attr rows at `rows`"
    )


def delete_node_by_pos(
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Delete node rows by row position, restoring the padding sentinels.

    Args:
        node_keys: ``(P, N) int32`` node key rows.
        node_attrs: ``(P, N, NA) float32`` node attribute rows.
        rows: ``(P,) int32`` row to clear per individual, ``I_INF`` to mean "no delete".

    Returns:
        ``(node_keys, node_attrs)`` with key row ``rows`` set to ``I_INF`` and attribute row
        ``rows`` set to NaN; input and output rows must never be passed here.

    Raises:
        NotImplementedError: Skeleton stub; the two sentinel scatters are pending.
    """
    raise NotImplementedError(
        "scatter I_INF into the key row and NaN into the attr row selected by `rows`"
    )


def add_conn(
    conn_keys: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    new_keys: evoxc.SymbolicTensor,
    new_attrs: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Write new connection rows at caller-chosen free rows of every individual.

    Same unconditional-write contract as :func:`add_node`: garbage is written when
    ``rows == I_INF`` and the caller gates the result with ``select``.

    Args:
        conn_keys: ``(P, E, CK) int32`` connection key rows, ``I_INF`` padded.
        conn_attrs: ``(P, E, CA) float32`` connection attribute rows, NaN padded.
        rows: ``(P,) int32`` destination row per individual, ``I_INF`` to mean "no write".
        new_keys: ``(P, CK) int32`` caller-allocated, population-unique connection keys.
        new_attrs: ``(P, CA) float32`` attribute rows of the new connections.

    Returns:
        ``(conn_keys, conn_attrs)`` with the new rows written at ``rows``.

    Raises:
        NotImplementedError: Skeleton stub; the two row scatters are pending.
    """
    raise NotImplementedError(
        "scatter new_keys into the key columns and new_attrs into the attr rows at `rows`"
    )


def delete_conn_by_pos(
    conn_keys: evoxc.SymbolicTensor,
    conn_attrs: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Delete connection rows by row position, restoring the padding sentinels.

    Args:
        conn_keys: ``(P, E, CK) int32`` connection key rows.
        conn_attrs: ``(P, E, CA) float32`` connection attribute rows.
        rows: ``(P,) int32`` row to clear per individual, ``I_INF`` to mean "no delete".

    Returns:
        ``(conn_keys, conn_attrs)`` with the ``CK`` key columns of row ``rows`` set to
        ``I_INF`` and its ``CA`` attribute columns set to NaN.

    Raises:
        NotImplementedError: Skeleton stub; the two sentinel scatters are pending.
    """
    raise NotImplementedError(
        "scatter I_INF into every key column and NaN into every attr column of row `rows`"
    )


def check_cycles(
    node_keys: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    src_keys: evoxc.SymbolicTensor,
    dst_keys: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Ask whether adding one edge per individual would close a cycle.

    The predicate is a *bounded* reachability relaxation over the existing connection set —
    a static Python ``for`` loop over at most ``max_nodes`` hops (never ``while_loop``),
    which is enough because a simple path in a graph of ``N`` nodes has at most ``N-1``
    edges.  A cycle is closed exactly when ``src`` is reachable from ``dst`` in the graph
    that already exists (the new edge ``src -> dst`` closes that path).

    Args:
        node_keys: ``(P, N) int32`` node key rows, ``I_INF`` padded.
        conn_keys: ``(P, E, CK) int32`` connection key rows — columns 0/1 are the source and
            destination node keys; absent rows are ``I_INF``.
        src_keys: ``(P,) int32`` new edge source node key, arbitrary (possibly absent) key.
        dst_keys: ``(P,) int32`` new edge destination node key.

    Returns:
        ``(P,) bool`` True where the new edge would close a cycle.  Absent or degenerate
        keys (``src == dst`` with an absent node, either endpoint ``I_INF`` or not a node of
        the genome) return False: the mutation is refused for another reason, not this one.

    Raises:
        NotImplementedError: Skeleton stub; the relaxation is pending.
    """
    raise NotImplementedError(
        "bounded reachability from dst_keys to src_keys over conn_keys "
        "(max_nodes static steps) -> (P,) bool"
    )


def re_cound_idx(
    node_keys: evoxc.SymbolicTensor,
    node_attrs: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    input_idx: list[int],
    output_idx: list[int],
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Renumber hidden node keys to be contiguous above the input/output keys (host-only).

    Host-only display helper: it converts its arguments with ``Tensor.numpy()``, applies the
    renumbering in Python and returns plain ``numpy`` arrays, which must never be fed back
    into a graph.  Input and output keys are left untouched (they are the external
    interface); hidden keys are replaced by consecutive integers starting at
    ``max(*input_idx, *output_idx) + 1``, in row order, and every connection key referring
    to a hidden node is rewritten accordingly.

    Args:
        node_keys: ``(N,) int32`` node key row of one genome, ``I_INF`` padded.
        node_attrs: ``(N, NA) float32`` node attribute row of the same genome, NaN padded.
        conn_keys: ``(E, CK) int32`` connection key rows of the same genome, ``I_INF``
            padded.
        input_idx: Keys (not rows) of the input nodes, kept verbatim.
        output_idx: Keys of the output nodes, kept verbatim.

    Returns:
        ``(node_keys, node_attrs, conn_keys)`` as ``numpy`` arrays: renumbered node keys,
        the unchanged node attributes (carried along so a payload can be rebuilt in one
        step) and renumbered connection keys.

    Raises:
        NotImplementedError: Skeleton stub; the Python relabelling is pending.
    """
    raise NotImplementedError(
        "numpy: map each hidden key to max(input_idx + output_idx) + 1 + row order and "
        "rewrite the connection keys"
    )
