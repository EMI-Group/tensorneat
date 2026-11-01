"""Tensorization: turn a ragged evolved topology into a static execution schedule.

Dependency DAG position: ``tensorneat.common`` bottom layer. Imports
:mod:`~tensorneat.common.tools` (flat-index helpers) and ``evoxc``; imports
nothing from ``genome``, ``algorithm``, ``problem`` or ``pipeline``, and no
``jax``/``torch``. This module replaces the pre-rewrite ``common/graph.py``:
the JAX-era topological sort over a dense ``(N, N)`` adjacency matrix is gone,
together with its ``max_nodes ** 2`` cost.

What a schedule is
------------------
NEAT topologies are ragged — different population members have different node
and connection counts — but evoxc kernels need static shapes. Tensorization
answers *how to execute* one whole population with fixed-shape arrays:

* every connection row is resolved to ``(src, dst)`` **node slots** (slot
  indices into that genome's own arrays, never innovation numbers),
* the incoming edges of each node are packed into a ``(P, N, K)`` table, so a
  forward pass is a level-ordered sweep whose work is proportional to the real
  connections and the network depth (never to ``max_nodes ** 2``),
* each node gets a ``level`` (longest-path depth), which is a valid level-sync
  topological order for feedforward execution.

Output contract: :class:`Transformed` holds **arrays only** — no Python
statics, no callables — so it is a plain pytree that can cross a graph
boundary and can be cached per shape signature. Any static fact a consumer
needs (``max_conns``, ``max_in_degree``, ``key_span``, ``pop_size``) is implied
by a shape or a dtype.

Cost: ``O(P * (S + C log C + N * K * D))`` with ``S = key_span``,
``C = max_conns``, ``K = max_in_degree`` and ``D`` = max depth — i.e.
edge-proportional apart from the padded ``N * K`` table and the one-off
key→slot table. ``derive_schedule`` runs once per generation (inside
``Pipeline.step``), so that cost is amortized over the whole generation.

Deliberate evoxc constraints (validated; see ``tools`` for the full list):
no ``cumsum`` (a StableHLO export deferral, hence the ``cc`` counting trick),
no scatter-with-reduce, no batched ``take_along_axis`` (hence flat indexing),
no ``where`` (``evoxc.select``), and out-of-range indices raise instead of
clamping — every index into a table must therefore be sanitized *and* masked.
"""

from typing import Any, NamedTuple

import evoxc

from .tools import I_INF


class Transformed(NamedTuple):
    """Static execution schedule of one population (all fields batch-first).

    Every field has population size ``P`` on axis 0; ``N = max_nodes``,
    ``C = max_conns``, ``K = max_in_degree``, ``I = len(input_keys)``,
    ``O = len(output_keys)`` are static at build time. Sentinels: ``I_INF``
    (``2147483647``, ``int32`` max) means "no index / unmatched /
    unavailable" in the integer fields; ``NaN`` never appears here (these are
    integer/bool arrays), it lives in the gene payload instead.

    Invariances consumers may rely on:

    * ``in_conn`` and ``in_src`` are ``I_INF`` in exactly the same positions,
      and a node's columns ``0..d-1`` are dense whenever its in-degree is
      ``d``; columns ``d..K-1`` are padding.
    * for every ``(p, c)`` with ``edge_valid[p, c]``, exactly one
      ``(p, i, k)`` satisfies ``in_conn[p, i, k] == c`` and
      ``in_src[p, i, k] == src[p, c]`` — except edges whose in-group rank
      reached ``K``, which are counted in ``edge_overflow``.
    * ``in_conn[p, i, 0] == I_INF`` means "node ``i`` has no incoming edge"
      (an input node, or an isolated node).
    * ``level[p, i]`` is ``I_INF`` iff node ``i`` is absent
      (``isnan(node_keys[p, i])``) or unreachable from the inputs / on a
      cycle; ``reduce_max(level)`` bounds the execution loop.
    * executing levels ``0, 1, ...`` in order is a valid topological order;
      nodes within one level are independent.
    * ``edge_overflow[p] > 0`` means the schedule is lossy for genome ``p``:
      the host must rebuild with a larger ``max_in_degree`` (or raise
      :class:`~tensorneat.common.errors.ShapeBoundError`). Kernels never
      silently drop edges.
    """

    #: (P, C) int32 — source node slot of each connection row; I_INF = absent row or unmatched key.
    src: Any
    #: (P, C) int32 — destination node slot of each connection row; I_INF likewise.
    dst: Any
    #: (P, C) bool — True iff the row is a real connection AND both endpoint keys matched a node.
    edge_valid: Any
    #: (P, N, K) int32 — conn row index of the k-th incoming edge of node i; I_INF padding.
    in_conn: Any
    #: (P, N, K) int32 — source slot of that same incoming edge; I_INF in the same positions as in_conn.
    in_src: Any
    #: (P, N) int32 — longest-path depth from the input nodes (inputs = 0); I_INF = absent node or on a cycle.
    level: Any
    #: (P, N) int32 — level * N + slot: a cheap total order for a topological sweep; I_INF where level is.
    topo_key: Any
    #: (P, I) int32 — node slot carrying input j, in the order of input_keys; I_INF if unmatched.
    input_slot: Any
    #: (P, O) int32 — node slot carrying output j, in the order of output_keys; I_INF if unmatched.
    output_slot: Any
    #: (P,) int32 — number of valid edges dropped because their in-group rank reached K.
    edge_overflow: Any


@evoxc.defn
def derive_schedule(
    node_keys,
    conn_in,
    conn_out,
    input_keys,
    output_keys,
    *,
    pop_size,
    max_nodes,
    max_conns,
    max_in_degree,
    key_span,
):
    """Derive the execution schedule of one whole population (one evoxc graph).

    Args:
        node_keys: ``(P, max_nodes)`` ``float32`` — the node **key column**
            (``nodes[:, :, 0]``); ``NaN`` marks an absent row.
        conn_in: ``(P, max_conns)`` ``float32`` — source endpoint key per
            connection row; ``NaN`` marks an absent row.
        conn_out: ``(P, max_conns)`` ``float32`` — destination endpoint key
            per connection row; ``NaN`` marks an absent row.
        input_keys: ``(I,)`` ``float32`` — keys of the input nodes, shared by
            every genome in the batch (no batch axis).
        output_keys: ``(O,)`` ``float32`` — keys of the output nodes, likewise
            shared.
        pop_size (int): ``P``; trace-time static (part of the shape signature).
        max_nodes (int): ``N``, the padded node count; trace-time static.
        max_conns (int): ``C``, the padded connection-row count; trace-time
            static.
        max_in_degree (int): ``K``, the static in-degree bound of the
            ``(P, N, K)`` incoming table; trace-time static.
        key_span (int): ``S``, the static key-magnitude bound of the dense
            key→slot table (``(P, S + 1)``); trace-time static.

    Returns:
        Transformed: the schedule described by :class:`Transformed`. Arrays
        only — no Python statics — so the pytree can cross a graph boundary
        and be cached per shape signature.

    Raises:
        tensorneat.common.errors.ConfigError: Host-side, before tracing, for a
            non-positive static, or keys that cannot fit ``key_span``.
        tensorneat.common.errors.GraphError: If a required input/output key is
            missing from ``node_keys`` for some genome.
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.

    Notes:
        Composition (each step is a separate, individually testable helper in
        this module):

        1. :func:`key_slot_table` — dense key→slot table ``(P, S+1) int32``,
           one flat scatter over the node keys, trailing dump column.
        2. :func:`slot_lookup` — ``src``/``dst`` from that table, with
           sanitized keys (``NaN -> dump``, clamp to ``[0, S-1]``); an
           unmatched endpoint is ``I_INF`` and makes the row invalid.
        3. :func:`edge_group_rank` — global rank of each edge in the
           ``(dst, conn_index)`` order.
        4. :func:`cum_in_degree` — ``cc[i]`` = number of edges with
           ``dst <= i``, via one stable ``argsort`` of the combined edge/node
           array and one flat scatter for the inverse permutation (this is the
           ``cumsum``-free counting trick).
        5. :func:`incoming_table` — ``in_conn``/``in_src``/``edge_overflow``
           from ``col = rank - start[dst]``.
        6. :func:`node_levels` — longest-path relaxation in a ``while_loop``
           capped at ``N`` iterations, inputs pinned to level 0.
        7. ``topo_key``, ``input_slot`` and ``output_slot`` are then free
           (``level * N + slot``, and two lookups of the shared key vectors).
    """
    raise NotImplementedError(
        "compose the six schedule steps into one evoxc graph and return a Transformed"
    )


def key_slot_table(node_keys, key_span):
    """Build the dense per-genome key→node-slot table.

    Args:
        node_keys: ``(P, N)`` ``float32`` node key column; ``NaN`` = absent
            row. Keys are integer-valued in ``[0, key_span)``.
        key_span (int): ``S``, the static key-magnitude bound; trace-time
            static.

    Returns:
        ``(P, S + 1)`` ``int32`` table: column ``table[p, k]`` holds the slot
        of the node of genome ``p`` whose key is ``k``, or ``I_INF`` when no
        node holds that key. Column ``S`` is the trailing **dump slot**: rows
        whose key is absent (``NaN``) are written there instead of index
        ``0``, which is what makes one unmasked ``flat_scatter`` safe (evoxc
        raises on out-of-range indices, and writing garbage into slot ``0``
        would corrupt a real node).

    Raises:
        tensorneat.common.errors.ConfigError: If ``key_span`` is not a
            positive ``int``, or a non-``NaN`` key lies outside
            ``[0, key_span)``.
        evoxc.BackendError: Propagated unchanged.

    Notes:
        Keys above ``2**24`` (``tools.KEY_MAX``) are not integer-exact in
        ``float32`` and therefore cannot be matched reliably — that is a
        configuration error, not a schedule error.
    """
    raise NotImplementedError(
        "scatter node slots into a (P, key_span + 1) int32 table with a dump column"
    )


def slot_lookup(table, keys, key_span):
    """Look up node slots for a batch of keys, yielding ``I_INF`` on a miss.

    Args:
        table: ``(P, S + 1)`` ``int32`` output of :func:`key_slot_table`.
        keys: ``(P, L)`` ``float32`` keys to resolve (connection endpoints, or
            the broadcast shared input/output key vectors); ``NaN`` allowed
            and always a miss.
        key_span (int): ``S``, matching the table; trace-time static.

    Returns:
        ``(P, L)`` ``int32`` slots in ``[0, N)``, with ``I_INF`` where the key
        is ``NaN`` or unknown.

    Raises:
        evoxc.ShapeError: Propagated unchanged if the table width is not
            ``key_span + 1``.

    Notes:
        Indices are made safe *before* the gather (``NaN -> 0``, then clamp to
        ``[0, S-1]``) because evoxc's numpy backend raises on out-of-range
        gathers; the ``NaN``/miss cases are then overwritten with ``I_INF`` by
        ``evoxc.select`` after the gather, so the clamped ``0`` never leaks
        into the result.
    """
    raise NotImplementedError(
        "sanitize keys, gather slots from the table and map misses to I_INF"
    )


def edge_group_rank(dst, edge_valid, max_conns):
    """Rank each connection row within its destination node's in-edge group.

    Args:
        dst: ``(P, C)`` ``int32`` destination slots; ``I_INF`` where the row is
            absent or unmatched.
        edge_valid: ``(P, C)`` ``bool`` from the lookup.
        max_conns (int): ``C``; trace-time static.

    Returns:
        ``(P, C)`` ``int32``: ``rank[p, c]`` = number of valid rows with the
        same ``dst`` and a smaller connection index, i.e. the position of row
        ``c`` in its destination's in-edge list. ``I_INF`` for invalid rows.

    Raises:
        evoxc.ShapeError: Propagated unchanged on a ``dst``/``edge_valid``
            mismatch.

    Notes:
        One ``argsort(..., stable=True)`` (``int64`` result cast down) over the
        composite key ``dst * C + conn_index``, then one ``flat_scatter`` of
        ``arange(C)`` to invert the permutation. ``stable=True`` is mandatory:
        ties in ``dst`` must keep connection-row order for the result to be
        deterministic. Invalid rows must be keyed *after* every valid row (and
        ``dst`` sanitized first — ``I_INF * C`` would overflow ``int32``).
    """
    raise NotImplementedError(
        "rank connection rows per destination with one stable argsort and its inverse"
    )


def cum_in_degree(dst, edge_valid, max_nodes):
    """Count, per node, how many valid edges point at it or at a lower slot.

    Args:
        dst: ``(P, C)`` ``int32`` destination slots (``I_INF`` for invalid
            rows); sanitized before use.
        edge_valid: ``(P, C)`` ``bool``.
        max_nodes (int): ``N``; trace-time static.

    Returns:
        ``(P, N)`` ``int32``: ``cc[p, i]`` = number of valid edges of genome
        ``p`` whose destination slot is ``<= i``. The start offset of node
        ``i``'s in-edge group is ``cc[i - 1]`` (``0`` for ``i == 0``), so the
        column of an edge is ``rank - start[dst]``.

    Raises:
        evoxc.ShapeError: Propagated unchanged.

    Notes:
        Deliberately ``cumsum``-free (``cumsum`` is a StableHLO export
        deferral in evoxc, numpy-interpreter only): build the combined array of
        ``C`` edge entries keyed ``2 * dst`` and ``N`` node entries keyed
        ``2 * slot + 1``, take one stable ``argsort``, invert it with one
        ``flat_scatter`` of ``arange(C + N)``, then read
        ``cc[i] = inverse[C + i] - i``. Edge keys precede their own node key
        (ties) so an edge is counted as "``<= i``".
    """
    raise NotImplementedError(
        "compute per-node cumulative in-degree with the argsort-inverse counting trick"
    )


def incoming_table(rank, cc, dst, src, edge_valid, max_nodes, max_in_degree):
    """Pack each node's incoming edges into the static ``(P, N, K)`` table.

    Args:
        rank: ``(P, C)`` ``int32`` in-group rank from
            :func:`edge_group_rank`.
        cc: ``(P, N)`` ``int32`` cumulative in-degree from
            :func:`cum_in_degree`.
        dst: ``(P, C)`` ``int32`` destination slots.
        src: ``(P, C)`` ``int32`` source slots.
        edge_valid: ``(P, C)`` ``bool``.
        max_nodes (int): ``N``; trace-time static.
        max_in_degree (int): ``K``; trace-time static.

    Returns:
        tuple: ``(in_conn, in_src, edge_overflow)`` where ``in_conn`` is
        ``(P, N, K)`` ``int32`` (connection row index of the k-th incoming
        edge of node ``i``), ``in_src`` is ``(P, N, K)`` ``int32`` (that edge's
        source slot, ``I_INF`` in exactly the same positions) and
        ``edge_overflow`` is ``(P,)`` ``int32`` (number of valid edges dropped
        because their column reached ``K``).

    Raises:
        evoxc.ShapeError: Propagated unchanged.

    Notes:
        ``start[i] = cc[i - 1]`` (``0`` for ``i == 0``) and
        ``col = rank - start[dst]``; the flat scatter target is
        ``dst * K + col`` into a ``(P, N * K + 1)`` buffer whose trailing cell
        is the dump slot, so edges with ``col >= K`` or with an invalid row go
        to the dump instead of throwing an out-of-range error. ``edge_overflow``
        is a ``reduce_sum`` of ``(col >= K) & edge_valid`` — data, never an
        exception (the host decides to rebuild or to raise
        :class:`~tensorneat.common.errors.ShapeBoundError`).
    """
    raise NotImplementedError(
        "scatter incoming edge ids and sources into the (P, N, K) table with overflow counting"
    )


def node_levels(in_conn, in_src, input_slot, max_nodes):
    """Longest-path depth of every node (the level-synchronous schedule).

    Args:
        in_conn: ``(P, N, K)`` ``int32`` incoming-edge table (``I_INF``
            padding); a padded column is never relaxed.
        in_src: ``(P, N, K)`` ``int32`` source slots of those edges.
        input_slot: ``(P, I)`` ``int32`` slots of the input nodes, pinned to
            level ``0``.
        max_nodes (int): ``N``; trace-time static, and the iteration cap of the
            relaxation loop.

    Returns:
        ``(P, N)`` ``int32`` levels: ``0`` for input nodes, ``max(pred) + 1``
        otherwise, ``I_INF`` for an absent node, a node unreachable from the
        inputs, or a node on a cycle (which never settles within ``N``
        iterations).

    Raises:
        evoxc.ShapeError / evoxc.TraceError: Propagated unchanged.

    Notes:
        One ``evoxc.while_loop`` relaxation of
        ``level[i] = max(level[in_src[i, k]] + 1)`` over the ``K`` axis, capped
        at ``N`` iterations; the loop carries only tensors of fixed
        shape/dtype and its condition is a scalar ``bool`` (a requirement of
        evoxc control flow), and no ``vmap`` may be applied over it — the batch
        axis is written explicitly, as everywhere in this package. Nothing is
        iterated per node in Python: the loop cost is ``O(N * K)`` per
        iteration, i.e. edge-proportional.
    """
    raise NotImplementedError(
        "relax node levels in a while_loop until the longest path settles"
    )


def validate_schedule(transformed):
    """Check :class:`Transformed` invariants on the host (test/debug helper).

    Args:
        transformed (Transformed): A schedule produced by
            :func:`derive_schedule`, whose arrays have already been brought
            back to host memory (numpy).

    Returns:
        None: Returns ``None`` when every invariant of :class:`Transformed`
        holds.

    Raises:
        tensorneat.common.errors.ConfigError: Describing the *first* violated
            invariant — a shape that does not match the documented signature,
            ``in_conn``/``in_src`` disagreeing on a padded position, a level of
            ``I_INF`` with a non-``I_INF`` ``topo_key``, a negative
            ``edge_overflow``, or an ``input_slot``/``output_slot`` that points
            outside ``[0, N)`` and is not ``I_INF``.

    Notes:
        Host-only and deliberately not a graph function: it is the oracle the
        unit tests use against a numpy reference implementation of the
        derivation. It never repairs a schedule and never runs inside a
        generation, so it must stay free of any evoxc tracing.
    """
    raise NotImplementedError(
        "assert the Transformed invariants on host arrays and raise ConfigError on the first violation"
    )
