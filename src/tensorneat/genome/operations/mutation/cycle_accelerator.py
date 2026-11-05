"""Accelerator formulation of the cycle predicate: a **scatter-free** hop.

This module is the second spelling of :func:`~.cycle.closes_cycle`.  Both answer the same
question for one candidate edge per individual — *would adding ``src -> dst`` close a cycle?*,
i.e. *is ``src`` already reachable from ``dst``?* — with the same key resolution and the same
boolean answer, so the caller's decision is byte-identical whichever route
:data:`KERNEL_MODE_KEY` selects.

What changes is the **hop**.  The host formulation stores the reached set in a flat
``(P, N + 1)`` frontier and writes each hop with one ``scatter`` of ``P * E`` updates.  That
scatter is the single most expensive kernel of the predicate on a compiled accelerator:
StableHLO lowers ``scatter`` to one workgroup of one thread, so it costs about 17-21 ns per
*update* element — ``P * max_conns`` = 50 000 update cells = **1.05 ms per hop** at pop 500,
three dispatches per generation on the ``xor_p500`` case, i.e. **3.18 ms** of the generation
and the largest single kernel of the predicate (measured on the box's MI300X, ROCm, IREE
3.13).  ``gather``, elementwise ops and reductions are properly parallel there, so the
accelerator route pays one extra elementwise/reduction pass instead of a serialised write:

1. carry ``reach`` as a plain ``(P, N) bool`` vector — no dump column, because nothing is
   index-written any more;
2. seed it with the destination one-hot ``equal(arange(N), dst_row)`` masked by ``dst_hit``,
   again **not** a scatter (a scatter as a ``while_loop`` init also SIGSEGVs
   ``iree-compile``; see :mod:`~.cycle`);
3. hoist the two loop-invariant operands once per call — the destination one-hot
   ``dst_onehot (P, E, N)`` and the flat source index ``src_flat = p * N + edge_src`` (the
   resolution already clamps every row into ``[0, N)``, so the index is always addressable
   and the liveness mask multiplies afterwards);
4. hop body: ``live = edge_valid & gather(reach, src_flat)``, then
   ``new = reduce_max(live[:, :, None] & dst_onehot, axes=1)`` — the edge axis collapses into
   the node axis, so a reached edge marks its destination through a broadcast reduction
   instead of a write; ``grown = reach | new`` and ``growing`` is the one-op growth test.

The cost is an ``O(P * E * N)`` boolean temporary per hop (2.5 M elements at pop 500 /
``max_conns`` 100 / ``max_nodes`` 50) and four more traced ops per hop than the scatter form,
which the numpy reference interpreter pays in full — that is why ``"host"`` stays the default
and this route is selected by :data:`KERNEL_MODE_KEY` on a device where the serialised
scatter dominates.  The switch is a **static host branch**: both spellings are plain traced
helpers (never ``@evoxc.defn``), so a graph is exactly one of them and no backend ever sees
both.

**The loop's growth flag must be consumed outside the loop** (``iree-compile`` release gate).
This hop carries ``(reach, growing)`` and only ``reach`` is read after the loop, so the
``growing`` result is dead.  ``iree-compile`` 3.13 runs an ``scf.while`` canonicalization that
drops such an unused result together with the corresponding body block argument but *keeps*
the yield operand and the operand, which makes the op illegal
(``region branch point has 2 operands, but region successor needs 4 inputs``) — the failure
surfaces much later, in ``iree-stream-conversion``, and only for the **assembled** generation
graph (the same kernel compiles and runs standalone, and the ``"host"`` spelling escapes it
only because its destination-passing scatter aliases the carried value, so its body reduces to
``scf.yield %arg`` and the rewrite stays consistent).  The answer therefore ends with
``logical_not(growing)``: a value-identity term (the loop only exits once its condition — the
carried flag itself — is ``False``, so the final flag is always ``False``) that keeps the
result live.  It also *reads* as what it is, a convergence guard: an unconverged reached set
would answer ``False``, i.e. refuse the mutation instead of admitting a cycle.
Do not delete it as dead code, and do not "simplify" the pair into one carried tensor without
re-running ``iree-compile`` on a traced generation graph in this mode.

Correctness is the host formulation's contract, unchanged: the same dense ``key -> row``
resolution (:func:`~.cycle._key_table` / :func:`~.cycle._rows_of`), so the out-of-span-key
precondition, the duplicate-key invariance and the "a degenerate candidate answers ``False``"
rule all hold for the same reasons :mod:`~.cycle` documents.
"""

import numpy as np
import evoxc
import evoxc.numpy as enp

from ....common import TensorNEATError
from .cycle import _key_table, _rows_of, _static_constant

#: State entry selecting the cycle predicate's formulation.
#:
#: ``"host"`` is the shipped scatter hop (:func:`~.cycle.closes_cycle`); ``"accelerator"`` is
#: this module's scatter-free hop.  The entry is a plain Python string on the trace-time
#: configuration ``State`` — read on the host, never a traced value — and a state that does
#: not carry it gets ``"host"``.
KERNEL_MODE_KEY = "kernel_mode"

#: Accepted values of :data:`KERNEL_MODE_KEY`, in documentation order.
KERNEL_MODES = ("host", "accelerator")

#: Formulation a state without the entry gets — the kernel-side default, matching the shipped
#: behaviour for a state no host code registered the entry on.
_DEFAULT_KERNEL_MODE = "host"


def kernel_mode(state) -> str:
    """Resolve :data:`KERNEL_MODE_KEY` from the trace-time ``State``.

    Args:
        state (Any): The configuration ``State`` the mutation kernel received.  The entry is
            a host value, never a traced one, so this is a plain attribute read: it emits no
            graph op and behaves identically inside and outside a trace.

    Returns:
        str: ``"host"`` when the entry is absent, else the stored formulation.

    Raises:
        TensorNEATError: If the entry holds anything but the two documented formulations, so
            a typo fails loudly instead of silently selecting the other route.
    """
    mode = getattr(state, KERNEL_MODE_KEY, _DEFAULT_KERNEL_MODE)
    if mode not in KERNEL_MODES:
        raise TensorNEATError(
            f"state.{KERNEL_MODE_KEY} must be one of {KERNEL_MODES}, got {mode!r}"
        )
    return mode


def selected(state) -> bool:
    """Whether ``state`` asks for this module's formulation.

    Args:
        state (Any): The trace-time configuration ``State``.

    Returns:
        bool: ``True`` when :func:`kernel_mode` resolves to ``"accelerator"``.

    Raises:
        TensorNEATError: If the state's entry is not one of the documented formulations.
    """
    return kernel_mode(state) == "accelerator"


def closes_cycle_accelerator(
    node_keys: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    src_keys: evoxc.SymbolicTensor,
    dst_keys: evoxc.SymbolicTensor,
    key_span: int,
) -> evoxc.SymbolicTensor:
    """Ask, scatter-free, whether adding one edge per individual would close a cycle.

    Args:
        node_keys: ``(P, N) int32`` node key rows, ``I_INF`` padded.
        conn_keys: ``(P, E, CK) int32`` connection key rows — columns ``0``/``1`` are the
            source and destination node keys; absent rows are ``I_INF``.
        src_keys: ``(P,) int32`` candidate edge source node key.
        dst_keys: ``(P,) int32`` candidate edge destination node key.
        key_span (int): ``S``, ``genome.key_span`` — the static addressable key window of the
            dense key table, the same one the execution schedule uses.  It cannot be derived
            from any operand shape, so the caller passes it.

    Returns:
        ``(P,) bool`` True where the candidate edge would close a cycle.  A degenerate
        candidate (either endpoint absent from the genome, ``I_INF``, or outside
        ``[0, key_span)``) answers ``False``: the mutation is refused for another reason,
        never this one.

    Raises:
        ConfigError: when ``key_span`` does not describe the table the shared resolution
            builds (a host-side configuration error, never a traced branch).
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.

    Notes:
        The key resolution is literally :mod:`~.cycle`'s — one dense table build and one flat
        ``gather`` for all ``2 * E + 2`` endpoints — so the representative function, and with
        it both correctness arguments of that module, are shared rather than re-derived.
        Everything the hop needs that does not depend on ``reach`` (``dst_onehot``,
        ``src_flat``, ``valid_flat``) is traced once, outside the loop region.
        The ``scatter`` that the host formulation pays per hop is the whole point of this
        route: it is replaced by one broadcast ``logical_and`` plus one reduction over the
        edge axis, both of which a compiled backend spreads over its workgroups.
        Both loop results are consumed (the second through a ``logical_not`` that is the
        identity on every backend), because an unused ``while`` result makes
        ``iree-compile`` produce an illegal ``scf.while`` for the assembled generation graph —
        the module note holds the compiler evidence and the exact rewrite at fault.
    """
    pop_size, node_count = node_keys.shape
    edge_count = conn_keys.shape[1]
    table = _key_table(node_keys, key_span)
    queries = enp.concatenate(
        [
            conn_keys[:, :, 0],
            conn_keys[:, :, 1],
            enp.reshape(src_keys, (pop_size, 1)),
            enp.reshape(dst_keys, (pop_size, 1)),
        ],
        axis=1,
    )
    rows, hits = _rows_of(table, queries, key_span, node_count)
    edge_src = evoxc.slice(rows, (0, 0), (pop_size, edge_count))
    edge_dst = evoxc.slice(rows, (0, edge_count), (pop_size, edge_count))
    src_hit = evoxc.slice(hits, (0, 2 * edge_count), (pop_size, 1))
    dst_hit = evoxc.slice(hits, (0, 2 * edge_count + 1), (pop_size, 1))
    src_row = evoxc.slice(rows, (0, 2 * edge_count), (pop_size, 1))
    dst_row = evoxc.slice(rows, (0, 2 * edge_count + 1), (pop_size, 1))
    edge_valid = evoxc.logical_and(
        evoxc.slice(hits, (0, 0), (pop_size, edge_count)),
        evoxc.slice(hits, (0, edge_count), (pop_size, edge_count)),
    )
    # Row offsets of the ``(P, N)`` reach vector: pure static data, one constant, so a row
    # index plus these offsets is the flat index of the reach gather.
    offsets = _static_constant(
        np.arange(0, pop_size * node_count, node_count, dtype=np.int32).reshape(
            pop_size, 1
        )
    )
    positions = pop_size * edge_count
    # ``edge_src`` is clamped into ``[0, N)`` by the resolution, so this flat index always
    # addresses a real cell; an edge with an absent endpoint is masked afterwards
    # (``valid_flat``) instead of being redirected to a dump column.
    src_flat = enp.reshape(evoxc.add(edge_src, offsets), (positions,))
    valid_flat = enp.reshape(edge_valid, (positions,))
    columns = enp.arange(0, node_count, dtype=evoxc.int32)
    # Loop-invariant: the destination one-hot over the node axis, ``(P, E, N)``.
    dst_onehot = evoxc.equal(
        enp.reshape(edge_dst, (pop_size, edge_count, 1)), columns
    )
    # Seed the reached set at the destination.  ``dst_row`` lies in ``[0, N)`` and the
    # ``dst_hit`` term clears an absent destination, so the seed needs no dump column and
    # - like the host formulation - is never a ``scatter``.
    # The carried tensor is ``(P, N + 1)``: columns ``0 .. N-1`` are the reached set and column
    # ``N`` is the loop's growth flag, broadcast over the population rows.  The flag lives
    # *inside* the single carried value on purpose - see the module note.
    seed_flags = enp.broadcast_to(
        enp.full((1, 1), True, dtype=evoxc.bool_), (pop_size, 1)
    )
    reach = enp.concatenate(
        [
            enp.reshape(
                evoxc.logical_and(evoxc.equal(columns, dst_row), dst_hit),
                (pop_size, node_count),
            ),
            seed_flags,
        ],
        axis=1,
    )
    flag_length = (pop_size, 1)

    def cond(carried):
        """Continue while the reached set gained a node (the carried flag's column)."""
        return evoxc.reduce_max(evoxc.slice(carried, (0, node_count), flag_length))

    def body(carried):
        """One hop: every live edge whose source is reached marks its destination."""
        current = evoxc.slice(carried, (0, 0), (pop_size, node_count))
        reached = evoxc.gather(
            enp.reshape(current, (pop_size * node_count,)), src_flat, axis=0
        )
        live = evoxc.logical_and(valid_flat, reached)
        marked = evoxc.reduce_max(
            evoxc.logical_and(enp.reshape(live, (pop_size, edge_count, 1)), dst_onehot),
            axes=1,
        )
        grown = evoxc.logical_or(current, marked)
        grew = enp.reshape(
            evoxc.reduce_max(evoxc.not_equal(grown, current), axes=1), (pop_size, 1)
        )
        return enp.concatenate([grown, grew], axis=1)

    carried = evoxc.while_loop(cond, body, reach)
    reach = evoxc.slice(carried, (0, 0), (pop_size, node_count))
    # Read the reached set at the candidate source's row.
    src_slot = enp.reshape(evoxc.add(src_row, offsets), (pop_size,))
    reached = enp.reshape(
        evoxc.gather(enp.reshape(reach, (pop_size * node_count,)), src_slot, axis=0),
        (pop_size, 1),
    )
    # The carried flag is always ``False`` here - the loop's condition IS that flag, so it only
    # exits on a hop that grew nothing - which makes this term the identity on every backend.
    # Read it as a convergence guard: a reached set that had not stabilised answers ``False``,
    # i.e. refuses the mutation rather than admitting a cycle.
    settled = evoxc.slice(carried, (0, node_count), flag_length)
    return enp.reshape(
        evoxc.logical_and(
            evoxc.logical_and(evoxc.logical_and(reached, src_hit), dst_hit),
            evoxc.logical_not(settled),
        ),
        (pop_size,),
    )


__all__ = [
    "KERNEL_MODE_KEY",
    "KERNEL_MODES",
    "closes_cycle_accelerator",
    "kernel_mode",
    "selected",
]
