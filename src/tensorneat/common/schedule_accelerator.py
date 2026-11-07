"""Accelerator formulations of the two writes the execution schedule makes.

This module is the second spelling of two steps of
:func:`~tensorneat.common.tensorization.derive_schedule`; both answer the identical question and
return byte-identical values, so the schedule a caller sees does not depend on which one a graph
was traced with.
What changes on a compiled backend is everything a ``scatter`` touches: StableHLO lowers
``scatter`` to **one workgroup of one thread** (grid ``1x1x1``, untiled, independent of the
destination width; see ``benchmarks/spikes/rocm/CONTEXT.md``), while a gather, an elementwise op
and a reduction spread over the whole grid.  The two writes left in the schedule are therefore
pure serialisation, and each has a **read-only** formulation:

1. **``cc`` (the ``cumsum``-free in-degree prefix) without the inverse permutation.**
   :func:`~tensorneat.common.tensorization.edge_order` inverts its ``argsort`` with a
   ``(P, C + N)`` flat scatter — 75 000 serialised update elements at ``xor_p500`` (pop 500,
   ``max_nodes`` 50, ``max_conns`` 100).  The value it
   produces is ``cc[i] = inv[C + i] - i``, i.e. *the number of valid edges whose destination slot
   is at most* ``i``, which is exactly a cumulative count over the ``(P, N, C)`` destination table
   ``dst <= i``; :func:`cumulative_indegree` reduces that table directly.  The ``argsort`` itself
   stays — ``order`` is what :func:`~tensorneat.common.tensorization.incoming_table` gathers its
   window out of, and no formulation needs the *inverse* permutation except this prefix.
2. **the key→slot table without the dense scatter.**  ``key_slot_table`` writes ``P * N``
   updates into ``P * (key_span + 1)`` cells with one flat scatter — 25 000 serialised update
   elements at the same geometry.  :func:`key_template` replaces it with the
   equality template the distance layer already uses
   (:func:`~tensorneat.genome.operations.distance.index_free.resolve` and
   :func:`~tensorneat.genome.operations.mutation.cycle_accelerator`): one broadcast ``equal`` of
   the query cells against the node cells plus a ``reduce_max``, all parallel.

Bit identity
------------
Both formulations are **exactly** equal to the host ones, not merely close, because they consume
the same sanitized integer cells the table would be addressed with
(:func:`~tensorneat.common.tensorization._key_indices`) rather than the raw keys:

* ``key_slot_table`` writes a visible node's *slot* at cell ``key``; duplicate cells resolve in
  flat traversal order, so the cell ends up holding the **highest** slot that names it.
  :func:`key_template` takes ``reduce_max`` of ``select(match, slot, -1)``, i.e. the same highest
  slot, so even a duplicate key — and even two ``float32`` keys that truncate to the same integer
  cell, which is what the table's ``cast(..., int32)`` address does — resolve identically.
* ``cc`` is a count in both routes: the host form reads it out of the inverse permutation, this
  form counts the same predicate, so ``edge_overflow`` (a function of ``cc``'s increments) and
  every window position agree too.

Selection
---------
The switch is the generic formulation static ``state.kernel_mode`` (:data:`KERNEL_MODE_KEY`):
``"host"`` is the shipped scatter forms, ``"accelerator"`` is this module's, a missing entry
means ``"host"``, and anything else raises.  It is a plain Python string on the trace-time
configuration ``State`` — read on the host, never a traced value — so a graph contains exactly one
of the two formulations and no backend ever sees both.
The numpy reference interpreter pays this module's element volume in full (an ``(P, L, N)``
comparison table where the host form writes ``P * N`` cells), which is why ``"host"`` stays the
default and a GPU-class device is what :meth:`tensorneat.Pipeline.setup` asks for the accelerator
route with.
"""

from typing import Any, Callable

import evoxc
import evoxc.numpy as enp

from .errors import ConfigError
from .tools import I_INF, _positive_static, _scalar_constant

#: State entry selecting the schedule's formulation.
#:
#: ``"host"`` is the shipped scatter forms (``key_slot_table`` + ``edge_order``'s inverse
#: permutation); ``"accelerator"`` is this module's scatter-free pair.  The entry is a plain
#: Python string on the trace-time configuration ``State`` — read on the host, never a traced
#: value — and a state that does not carry it gets ``"host"``.
KERNEL_MODE_KEY = "kernel_mode"

#: Accepted values of :data:`KERNEL_MODE_KEY`, in documentation order.
KERNEL_MODES = ("host", "accelerator")

#: Formulation a state without the entry gets — the kernel-side default, matching the shipped
#: behaviour for a state no host code registered the entry on.
_DEFAULT_KERNEL_MODE = "host"

#: Cell value an *invisible* query is rewritten to before the equality template compares it.
#: Outside ``[0, key_span)`` and distinct from :data:`_NO_NODE`, so a masked query can never
#: match a masked node row.
_NO_QUERY = -1

#: Cell value an *invisible* node row (``I_INF`` padding or an out-of-span key) is rewritten to
#: before the comparison.  Never equals :data:`_NO_QUERY`.
_NO_NODE = -2

__all__ = [
    "KERNEL_MODE_KEY",
    "KERNEL_MODES",
    "cumulative_indegree",
    "is_accelerator",
    "kernel_mode",
    "key_template",
    "resolve_mode",
]


def kernel_mode(state: Any) -> str:
    """Resolve :data:`KERNEL_MODE_KEY` from the trace-time ``State``.

    Args:
        state (Any): The configuration ``State`` the caller holds.  The entry is a host value,
            never a traced one, so this is a plain attribute read: it emits no graph op and
            behaves identically inside and outside a trace.

    Returns:
        str: ``"host"`` when the entry is absent, else the stored formulation.

    Raises:
        tensorneat.common.errors.ConfigError: If the entry holds anything but the two documented
            formulations, so a typo fails loudly instead of silently selecting the other route.
    """
    return resolve_mode(getattr(state, KERNEL_MODE_KEY, _DEFAULT_KERNEL_MODE))


def resolve_mode(mode: Any) -> str:
    """Validate a formulation name, as stored on a state or passed as a graph static.

    Args:
        mode (Any): A value of :data:`KERNEL_MODES`.

    Returns:
        str: ``mode`` unchanged.

    Raises:
        tensorneat.common.errors.ConfigError: If ``mode`` is not one of the documented
            formulations.
    """
    if mode not in KERNEL_MODES:
        raise ConfigError(
            f"{KERNEL_MODE_KEY} must be one of {KERNEL_MODES}, got {mode!r}"
        )
    return mode


def is_accelerator(mode: Any) -> bool:
    """Whether ``mode`` asks for this module's formulations.

    Args:
        mode (Any): A value of :data:`KERNEL_MODES` (a state's entry, already resolved).

    Returns:
        bool: ``True`` when ``mode`` resolves to ``"accelerator"``.

    Raises:
        tensorneat.common.errors.ConfigError: If ``mode`` is not one of the documented
            formulations.
    """
    return resolve_mode(mode) == "accelerator"


def key_template(
    node_cells: Any, node_visible: Any, rows: Any
) -> Callable[[Any, Any], Any]:
    """Build the scatter-free ``key -> node slot`` resolver of one population.

    Args:
        node_cells (evoxc.SymbolicTensor): ``(P, N) int32`` **sanitized** node key cells — the
            output of :func:`~tensorneat.common.tensorization._key_indices`, i.e. the integer the
            dense table would be addressed with (``0`` where the key is absent or out of span).
        node_visible (evoxc.SymbolicTensor): ``(P, N) bool`` "this node row carries a key inside
            ``[0, key_span)``" — ``_key_indices``'s second return value.
        rows (evoxc.SymbolicTensor): ``(P, N) int32`` the node slot of every row
            (:func:`~tensorneat.common.tensorization._node_slots`).

    Returns:
        A callable ``resolve(query_cells, query_visible)`` returning ``(P, L) int32`` slots in
        ``[0, N)`` with ``I_INF`` on a miss, with exactly
        :func:`~tensorneat.common.tensorization._slot_of`'s contract for the same
        ``(query_cells, query_visible)`` pair.

    Notes:
        The template is one broadcast ``equal`` of the ``L`` query cells against the ``N`` node
        cells of the same individual, then the **highest** matching slot (``reduce_max`` of
        ``select(match, row, -1)``).  The highest is the representative the dense table keeps: a
        flat ``scatter`` writes the ``N`` rows of an individual in increasing slot order, so the
        last write wins on a duplicated cell.  A row whose key is invisible is rewritten to
        :data:`_NO_NODE` and an invisible query to :data:`_NO_QUERY`, two distinct values below
        every real cell, so neither can match a real row and they cannot match each other.
        ``L`` is static, so the ``(P, L, N)`` table is a normal static-shape temporary: no
        ``scatter``, no permutation and no ``iota`` — every op below it is a compare, a select or
        a reduction over a materialised tensor.
    """
    pop_size, max_nodes = (int(size) for size in node_cells.shape)
    if tuple(node_visible.shape) != (pop_size, max_nodes):
        raise ConfigError(
            "key_template: node_visible must match node_cells, got "
            f"{tuple(node_visible.shape)} against {(pop_size, max_nodes)}"
        )
    if tuple(rows.shape) != (pop_size, max_nodes):
        raise ConfigError(
            "key_template: rows must match node_cells, got "
            f"{tuple(rows.shape)} against {(pop_size, max_nodes)}"
        )
    cells = evoxc.select(
        node_visible, node_cells, _scalar_constant(_NO_NODE, evoxc.int32)
    )
    slot_of_row = evoxc.select(
        node_visible, rows, _scalar_constant(_NO_NODE, evoxc.int32)
    )
    minus_one = _scalar_constant(-1, evoxc.int32)
    zero = _scalar_constant(0, evoxc.int32)
    iinf = _scalar_constant(I_INF, evoxc.int32)

    def resolve(query_cells: Any, query_visible: Any) -> Any:
        """Resolve ``(P, L)`` sanitized key cells to node slots; ``I_INF`` on a miss."""
        width = int(query_cells.shape[1])
        if tuple(query_visible.shape) != (pop_size, width):
            raise ConfigError(
                "key_template: query_visible must match query_cells, got "
                f"{tuple(query_visible.shape)} against {(pop_size, width)}"
            )
        queries = evoxc.select(
            query_visible, query_cells, _scalar_constant(_NO_QUERY, evoxc.int32)
        )
        shape = (pop_size, width, max_nodes)
        match = enp.equal(
            enp.broadcast_to(evoxc.reshape(queries, (pop_size, width, 1)), shape),
            enp.broadcast_to(evoxc.reshape(cells, (pop_size, 1, max_nodes)), shape),
        )
        candidates = evoxc.select(
            match,
            enp.broadcast_to(evoxc.reshape(slot_of_row, (pop_size, 1, max_nodes)), shape),
            minus_one,
        )
        best = evoxc.reduce_max(candidates, axes=(2,))
        return evoxc.select(enp.greater_equal(best, zero), best, iinf)

    return resolve


def cumulative_indegree(dst: Any, edge_valid: Any, max_nodes: Any) -> Any:
    """``cc[i]`` = the number of valid edges whose destination slot is at most ``i``.

    Args:
        dst: ``(P, C) int32`` destination slots of the connection rows; ``I_INF`` where the row is
            absent or unmatched.  edge_valid: ``(P, C) bool`` = a real row whose endpoint keys both
            matched.  max_nodes (int): ``N``, the destination slot count.

    Returns:
        ``(P, N) int32`` — exactly :func:`~tensorneat.common.tensorization.edge_order`'s ``cc``,
        so ``degree = cc - start`` and ``start[i] = cc[i - 1]`` are unchanged downstream.

    Raises:
        tensorneat.common.errors.ConfigError: At trace time, if ``max_nodes`` is not a positive
            static or ``dst``/``edge_valid`` disagree on their connection axis.

    Notes:
        The host form reads this count out of an inverted permutation:
        ``cc[i] = inv[C + i] - i``, because exactly ``cc[i]`` valid edges and ``i`` node entries
        sort below node entry ``C + i``.  Counting the predicate directly is the same number
        without the permutation: one ``(P, N, C)`` comparison ``dst <= i``, an ``and`` with
        ``edge_valid``, and a reduction over the connection axis.  An ``I_INF`` destination fails
        ``<= i`` for every real slot and is masked by ``edge_valid`` as well, so a padded row
        contributes nothing; ``cc`` is bounded by ``C``, so no overflow is possible in ``int32``.
    """
    max_nodes = _positive_static(max_nodes, "max_nodes")
    pop_size, max_conns = (int(size) for size in dst.shape)
    if tuple(edge_valid.shape) != (pop_size, max_conns):
        raise ConfigError(
            "cumulative_indegree: edge_valid must match dst, got "
            f"{tuple(edge_valid.shape)} against {(pop_size, max_conns)}"
        )
    shape = (pop_size, max_conns, max_nodes)
    slots = enp.reshape(
        enp.arange(0, max_nodes, dtype=evoxc.int32), (1, 1, max_nodes)
    )
    reached = enp.logical_and(
        enp.broadcast_to(evoxc.reshape(edge_valid, (pop_size, max_conns, 1)), shape),
        enp.less_equal(
            enp.broadcast_to(evoxc.reshape(dst, (pop_size, max_conns, 1)), shape),
            enp.broadcast_to(slots, shape),
        ),
    )
    counts = evoxc.reduce_sum(evoxc.cast(reached, evoxc.int32), axes=(1,))
    return evoxc.cast(counts, evoxc.int32)
