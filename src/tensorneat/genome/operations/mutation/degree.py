"""Diagnostic in-degree audit of a post-write gene payload.

The mutation stage keeps its capacity invariants with cheap per-plan guards (an added
connection is refused when its destination is already saturated, a split is refused when
the caller-allocated key is already present).  Those guards are the hot path; this module
is their *audit*: a small, standalone kernel that measures the in-degree a payload
actually has, so a caller (a soak harness, a regression test, an interactive debugging
session) can assert the invariant on the payload the pipeline is about to evaluate.

It is **diagnostic only**:

- :meth:`~.base.BaseMutation.mutate_structure` never calls it, and it must stay out of the
  hot path — its ``(P, N, E)`` equality intermediate is much larger than any op the
  operators emit, so wiring it into a generation would cost real time for a quantity the
  guards already bound.
- The count is key-based, so **duplicate node keys make it a union**: two node rows sharing
  a key both report every edge that targets that key.  That is the honest reading of a
  payload (the execution schedule folds duplicate keys onto one slot) and it is exactly the
  symptom a failed allocation shows.

Every function is an ordinary trace-time kernel: call it from inside an ``evoxc`` trace
(e.g. through :func:`evoxc.build` on a bound method), never as eager NumPy.

Presence is derived the way the write path derives it: a connection row is present when its
**source key** is not ``I_INF`` (absent rows are ``I_INF`` in every key column), so a padded
row can never contribute an edge and a padded node row — key ``I_INF`` — always counts ``0``.
Only columns ``0`` (source) and ``1`` (destination) are read, so connection genes with and
without a ``historical_marker`` (``CK`` of 2 or 3) are both supported.
"""

import evoxc
import evoxc.numpy as enp

from ....common import I_INF
from .arrays import i32 as _int32

__all__ = ["in_degree", "max_in_degree", "violation"]


def in_degree(
    node_keys: evoxc.SymbolicTensor, conn_keys: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Count the present incoming connection rows of every node row.

    One broadcast ``(P, N, E)`` equality between each node row's key and each present
    connection row's destination key, folded by a reduce over the connection axis.  This is
    the *post-write* in-degree of the payload it is handed: pass the arrays
    ``execute_mutation`` returned (or the ones the pipeline is about to evaluate) to audit
    the invariant, and the arrays before the stage to check the stage's effect.

    Args:
        node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded; a duplicate key is counted
            once per row it appears on (the count is a union over rows sharing the key).
        conn_keys: ``(P, E, CK) int32`` connection keys ``(src, dst[, historical_marker])``,
            ``I_INF`` padded.

    Returns:
        ``(P, N) int32`` in-degree per node row; padded node rows and node keys no edge
        targets report ``0``.
    """
    pop_size, node_count = node_keys.shape
    edge_count = conn_keys.shape[1]
    present = evoxc.not_equal(conn_keys[:, :, 0], _int32(I_INF))
    matches = evoxc.logical_and(
        evoxc.equal(
            enp.reshape(node_keys, (pop_size, node_count, 1)),
            enp.reshape(conn_keys[:, :, 1], (pop_size, 1, edge_count)),
        ),
        enp.reshape(present, (pop_size, 1, edge_count)),
    )
    counts = evoxc.reduce_sum(evoxc.cast(matches, evoxc.int32), axes=2)
    return evoxc.cast(counts, evoxc.int32)


def max_in_degree(
    node_keys: evoxc.SymbolicTensor, conn_keys: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Report each individual's largest :func:`in_degree`.

    The genome compares this against ``genome.max_in_degree`` (its static window width ``K``)
    to answer "did any node receive more edges than the schedule can hold?" without a host
    pass over the payload.

    Args:
        node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
        conn_keys: ``(P, E, CK) int32`` connection keys ``(src, dst[, marker])``.

    Returns:
        ``(P,) int32`` maximum in-degree per individual (``0`` for an individual with no
        present connection row).
    """
    return evoxc.reduce_max(in_degree(node_keys, conn_keys), axes=1)


def violation(
    node_keys: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    limit: "int | evoxc.SymbolicTensor",
) -> evoxc.SymbolicTensor:
    """Count how many node rows of each individual exceed an in-degree ``limit``.

    ``limit`` is usually the genome's ``max_in_degree``: a non-zero result means the payload
    would make ``derive_schedule`` report ``edge_overflow`` and the pipeline raise, so an
    audit asserts it is all-zero.

    Args:
        node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
        conn_keys: ``(P, E, CK) int32`` connection keys ``(src, dst[, marker])``.
        limit: ``int`` (static ``K``) or a rank-0 ``int32`` symbolic tensor.  An ``int`` is
            wrapped in a typed rank-0 constant so the comparison cannot widen to ``int64``.

    Returns:
        ``(P,) int32`` number of node rows whose in-degree is strictly greater than
        ``limit``.
    """
    counts = in_degree(node_keys, conn_keys)
    threshold = limit if isinstance(limit, evoxc.SymbolicTensor) else _int32(int(limit))
    over = evoxc.cast(evoxc.greater(counts, threshold), evoxc.int32)
    return evoxc.cast(evoxc.reduce_sum(over, axes=1), evoxc.int32)
