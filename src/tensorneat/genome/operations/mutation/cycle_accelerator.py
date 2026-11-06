"""Accelerator formulation of the cycle predicate: a **scatter-free** key resolution *and* hop.

This module is the second spelling of :func:`~.cycle.closes_cycle`.  Both answer the same
question for one candidate edge per individual — *would adding ``src -> dst`` close a cycle?*,
i.e. *is ``src`` already reachable from ``dst``?* — with the same boolean answer, so the
caller's decision is byte-identical whichever route :data:`KERNEL_MODE_KEY` selects.

What changes on a compiled backend is everything a ``scatter`` touches, because StableHLO
lowers ``scatter`` to **one workgroup of one thread**:

1. **Key resolution: an equality template, no scatter.**  The dense ``key -> row`` table
   (:func:`~.cycle._key_table`) is a flat ``scatter`` of ``P * N`` updates into
   ``P * (key_span + 1)`` cells, i.e. 25 000 serialised writes into 75 500 cells at pop 500,
   and it measured **0.515-0.578 ms of the predicate's 0.769 ms per call** there (90 % at pop
   1000).  The template — the one
   :func:`tensorneat.genome.operations.distance.index_free.resolve` already uses — is one
   broadcast ``equal`` of the ``Q = 2 * E + 2`` query keys against the node row axis, then the
   *lowest* matching row (``reduce_min`` of ``select(match, arange(N), N)``): a ``(P, Q, N)``
   compare plus a reduction, all parallel.  Same representative-function contract (an
   injective map key -> row; :mod:`~.cycle` proves the answer is invariant over the choice),
   same visibility law, different, deterministic representative.
2. **Hop: a broadcast reduction, no scatter, with a static trip count.**  The reached set is
   carried as a ``(P, N)`` slice of one ``(P, N + 1) int32`` tensor whose last column is the hop
   counter, seeded with the destination one-hot ``equal(arange(N), dst_row)``; each hop is
   ``live = edge_valid & gather(reach, src_flat)`` then
   ``new = reduce_max(live[:, :, None] & dst_onehot, axes=1)``, so a reached edge marks its
   destination through a reduction instead of a write.  The loop is an ``evoxc.while_loop``
   whose condition is ``hops < node_count`` — a **static** trip count, never a data-dependent
   one: the relaxation is monotone and each hop only adds the successors of an already-reached
   node, so the breadth-first closure from a single source over ``node_count`` nodes saturates
   after at most ``node_count - 1`` hops and the ``node_count``-th hop grows nothing.  The
   reached set after the full budget is therefore bit-identical to the one the early-exiting
   data-dependent loop reached, for every topology.
   ``evoxc.scan`` is not an option here: it requires at least one stacked tensor output (a body
   that only reduces is refused), and that stack would add a per-hop ``scatter`` plus a
   ``(length, ...)`` accumulator — more dispatches and element volume than the counted region,
   for the same static trip count.

Measured on the box's MI300X VF (gfx942, ROCm 7.1, IREE 3.13) with the predicate isolated at
the ``xor`` geometry (``max_nodes`` 50, ``max_conns`` 100, ``key_span`` 150), warmed, device ms
per call from ``rocprofv3 --kernel-trace``: **0.769 -> 0.173 ms at pop 500** and
**1.425 -> 0.282 ms at pop 1000**, i.e. -77 % / -80 %, with the graph's last ``scatter`` gone
(scatters 1 -> 0).  Those numbers are the **key resolution's** (dense ``scatter`` table against
the equality template) on the data-dependent hop loop that shipped with it, so they are not this
spelling's per-call cost: the counted loop spends ``node_count`` hops where that loop exited at
the payload's breadth, which only a device measurement can price.  Splitting the resolution into
one ``(P, E, N)`` pass per key family halves the transient but adds ten dispatches and measured
0.206 / 0.345 ms — so the single pass is the shipped form.  The numpy reference interpreter pays
the ``(P, Q, N)`` element volume in full (it is ~6-8x the host call there), which is exactly why
``"host"`` stays the default and this route is selected by :data:`KERNEL_MODE_KEY` on a device
where the serialised scatters dominate.

The switch is a **static host branch**: both spellings are plain traced helpers (never
``@evoxc.defn``), so a graph is exactly one of them and no backend ever sees both.

**The loop carries one value, and that value is the one the answer reads** (``iree-compile``
release gate).  A ``while_loop`` result nothing reads is dropped by ``iree-compile`` 3.13's
``scf.while`` canonicalization together with the corresponding body block argument, while the
yield operand and the loop operand stay, which makes the op illegal
(``region branch point has 2 operands, but region successor needs 4 inputs``); the failure
surfaces much later, in ``iree-stream-conversion``, and only for the **assembled** generation
graph — the same kernel compiles and runs standalone.  A separate ``(reach, growing)`` pair
therefore fails on the assembled graph even when ``growing`` is consumed by a value-identity
``logical_not``.  So the loop here has exactly one carried tensor — the ``(P, N + 1) int32``
reached set with the hop counter in column ``N`` — whose result is what the answer reads; the
counter rides inside it instead of beside it, and no ``while_loop`` with an unread result belongs
in this predicate.

Correctness is the host formulation's contract, unchanged: the out-of-span-key precondition,
the duplicate-key invariance and the "a degenerate candidate answers ``False``" rule all hold
for the same reasons :mod:`~.cycle` documents, and the window law is shared with it
(:func:`~.cycle._in_span`) rather than re-spelled.
"""

import numpy as np
import evoxc
import evoxc.numpy as enp

from ....common import TensorNEATError
from .cycle import _in_span, _static_constant

#: State entry selecting the cycle predicate's formulation.
#:
#: ``"host"`` is the shipped scatter hop (:func:`~.cycle.closes_cycle`); ``"accelerator"`` is
#: this module's scatter-free formulation.  The entry is a plain Python string on the
#: trace-time configuration ``State`` — read on the host, never a traced value — and a state
#: that does not carry it gets ``"host"``.
KERNEL_MODE_KEY = "kernel_mode"

#: Accepted values of :data:`KERNEL_MODE_KEY`, in documentation order.
KERNEL_MODES = ("host", "accelerator")

#: Formulation a state without the entry gets — the kernel-side default, matching the shipped
#: behaviour for a state no host code registered the entry on.
_DEFAULT_KERNEL_MODE = "host"

#: Key value an *invisible* query is rewritten to before the equality template compares it.
#: Outside ``[0, key_span)`` and distinct from :data:`_NO_NODE`, so a masked query can never
#: match a masked node row.
_NO_QUERY = -1

#: Key value an *invisible* node row (``I_INF`` padding or an out-of-span key) is rewritten
#: to before the comparison.  Never equals :data:`_NO_QUERY`.
_NO_NODE = -2


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


def _key_resolver(node_keys: evoxc.SymbolicTensor, key_span: int):
    """Build the scatter-free ``key -> node row`` resolver of one population.

    Args:
        node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node key rows (``I_INF`` padded).
        key_span (int): ``S``, the genome's static ``key_span`` — the addressable window.

    Returns:
        A callable ``resolve(keys) -> (rows, hit)`` with exactly
        :func:`~.cycle._rows_of`'s contract: ``rows`` is ``(P, Q) int32`` in ``[0, N)`` (safe
        to index with) and ``hit`` is ``(P, Q) bool``.

    Notes:
        The equality template is
        :func:`tensorneat.genome.operations.distance.index_free.resolve`'s: for every query
        key, one broadcast ``equal`` against the whole node row axis, then the *lowest*
        matching row (``reduce_min`` of ``select(match, arange(N), N)``), so a query block
        costs one ``(P, Q, N)`` compare, one ``(P, Q, N) int32`` select and one reduction, and
        **no ``scatter`` at all**.
        The representative function differs from the dense table's (lowest row here,
        ``scatter`` write order there) and the answer does **not**: both are injective on keys
        (:mod:`~.cycle` proves the invariance), every edge endpoint and both candidate
        endpoints go through the same map, so the traversed row graph is the key-level graph
        under that map and reachability is one predicate for every choice.
        Visibility is the dense table's, spelled as *value* masks instead of scatter targets:
        a padded (``I_INF``) or out-of-span node row is replaced by :data:`_NO_NODE` once per
        call, and an out-of-span query by :data:`_NO_QUERY`, so the two ``(P, Q, N)`` boolean
        masks a literal spelling would need are folded into the comparison itself (the masks
        stay ``(P, N)`` / ``(P, Q)`` selects).  A miss therefore lands on ``N`` — one past the
        last row — which :func:`~.cycle._rows_of`'s ``hit = got != N`` test and
        ``minimum(got, N - 1)`` clamp read back identically.
        Every rank-0 constant and the ``(N,)`` row index are built **once** here, and the two
        visibility masks reuse :func:`~.cycle._in_span`, so the window law has one spelling in
        the package.
        The bounds and the mask literals are static per trace and the node axis is static, so
        the whole resolution is a fixed-shape pass; no ``iota``/``argmax`` is emitted (both
        are ``iree-compile`` hazards, see
        :func:`tensorneat.genome.operations.distance.index_free.resolve`).
    """
    pop_size, node_count = (int(size) for size in node_keys.shape)
    span = int(key_span)
    zero = _static_constant(np.asarray(0))
    span_constant = _static_constant(np.asarray(span))
    no_query = _static_constant(np.asarray(_NO_QUERY))
    no_node = _static_constant(np.asarray(_NO_NODE))
    no_match = _static_constant(np.asarray(node_count))
    last_row = _static_constant(np.asarray(node_count - 1))
    columns = enp.arange(0, node_count, dtype=evoxc.int32)
    visible_rows = enp.reshape(
        evoxc.select(_in_span(node_keys, zero, span_constant), node_keys, no_node),
        (pop_size, 1, node_count),
    )

    def resolve(keys: evoxc.SymbolicTensor) -> tuple:
        """Resolve one ``(P, Q) int32`` key block to ``(rows, hit)``, both ``(P, Q)``."""
        rows, count = (int(size) for size in keys.shape)
        masked = evoxc.select(_in_span(keys, zero, span_constant), keys, no_query)
        table = evoxc.equal(enp.reshape(masked, (rows, count, 1)), visible_rows)
        lowest = evoxc.reduce_min(evoxc.select(table, columns, no_match), axes=2)
        return evoxc.minimum(lowest, last_row), evoxc.less(lowest, no_match)

    return resolve


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
        key_span (int): ``S``, ``genome.key_span`` — the static addressable key window, the
            same one the execution schedule uses.  It cannot be derived from any operand
            shape, so the caller passes it.

    Returns:
        ``(P,) bool`` True where the candidate edge would close a cycle.  A degenerate
        candidate (either endpoint absent from the genome, ``I_INF``, or outside
        ``[0, key_span)``) answers ``False``: the mutation is refused for another reason,
        never this one.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.

    Notes:
        The key resolution is :func:`_key_resolver` — one broadcast equality for all
        ``2 * E + 2`` endpoints, no ``scatter`` — so the representative function, and with it
        both correctness arguments of :mod:`~.cycle`, are shared rather than re-derived.
        Everything the hop needs that does not depend on ``reach`` (``dst_onehot``,
        ``src_flat``, ``valid_flat``) is traced once, outside the loop region.
        The loop carries a single ``(P, N + 1) int32`` tensor — the reached set plus its hop
        counter in column ``N`` — and runs exactly ``node_count`` hops, a static trip count, so
        the reached set it hands the read-back is the saturated breadth-first closure whichever
        topology it walked (see the module note for the saturation argument and for why the
        counter is packed into that one value).  A ``while`` result nothing reads makes
        ``iree-compile`` produce an illegal ``scf.while`` for the assembled generation graph, so
        the loop has one result and the answer reads it.
    """
    pop_size, node_count = node_keys.shape
    edge_count = conn_keys.shape[1]
    queries = enp.concatenate(
        [
            conn_keys[:, :, 0],
            conn_keys[:, :, 1],
            enp.reshape(src_keys, (pop_size, 1)),
            enp.reshape(dst_keys, (pop_size, 1)),
        ],
        axis=1,
    )
    rows, hits = _key_resolver(node_keys, key_span)(queries)
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
    # The carried value is one ``(P, N + 1) int32`` tensor: columns ``0 .. N-1`` are the reached
    # set as ``0``/``1`` and column ``N`` is the hop counter, identical on every population row
    # (one value only - see the module note on the ``while`` result the compiler drops), so the
    # flat index of a row is its column plus these per-individual row offsets.
    width = node_count + 1
    offsets = _static_constant(
        np.arange(0, pop_size * width, width, dtype=np.int32).reshape(pop_size, 1)
    )
    positions = pop_size * edge_count
    # ``edge_src`` is clamped into ``[0, N)`` by the resolution, so this flat index always
    # addresses a real cell; an edge with an absent endpoint is masked afterwards
    # (``valid_flat``) instead of being redirected to a dump column.
    src_flat = enp.reshape(evoxc.add(edge_src, offsets), (positions,))
    # ``0``/``1`` masks, so one hop of the ``int32`` reached set is one ``multiply`` and the
    # ``maximum`` that applies a hop is its logical OR.
    valid_flat = evoxc.cast(enp.reshape(edge_valid, (positions,)), evoxc.int32)
    columns = enp.arange(0, node_count, dtype=evoxc.int32)
    # Loop-invariant: the destination one-hot over the node axis, ``(P, E, N)``.
    dst_onehot = evoxc.cast(
        evoxc.equal(enp.reshape(edge_dst, (pop_size, edge_count, 1)), columns),
        evoxc.int32,
    )
    zero = _static_constant(np.asarray(0))
    one = _static_constant(np.asarray(1))
    # The hop budget is the node count.  The relaxation is monotone and a hop only ever adds
    # the successors of an already-reached node, so a breadth-first closure from one source over
    # ``node_count`` nodes saturates after at most ``node_count - 1`` hops and the ``node_count``
    # -th hop grows nothing: a fixed-length loop ends on the saturated reached set for every
    # topology, which is why the trip count is a static constant and not a growth flag.
    hop_budget = _static_constant(np.asarray(node_count))

    # Seed the reached set at the destination, hop counter ``0``.  ``dst_row`` lies in
    # ``[0, N)`` and the ``dst_hit`` term clears an absent destination, so the seed needs no
    # dump column and - like the host formulation - is never a ``scatter``.
    carried = enp.concatenate(
        [
            evoxc.cast(
                evoxc.logical_and(evoxc.equal(columns, dst_row), dst_hit), evoxc.int32
            ),
            enp.zeros((pop_size, 1), dtype=evoxc.int32),
        ],
        axis=1,
    )

    def cond(state):
        """Run the whole static budget, whatever the payload's depth is."""
        # Column ``N`` is uniform across the population rows, so one row answers for all of
        # them; the reshape makes the predicate the rank-0 ``bool`` the region op requires.
        hops = evoxc.reshape(evoxc.slice(state, (0, node_count), (1, 1)), ())
        return evoxc.less(hops, hop_budget)

    def body(state):
        """One hop: every live edge whose source is reached marks its destination."""
        current = evoxc.slice(state, (0, 0), (pop_size, node_count))
        reached = evoxc.gather(
            enp.reshape(state, (pop_size * width,)), src_flat, axis=0
        )
        live = evoxc.multiply(valid_flat, reached)
        marked = evoxc.reduce_max(
            evoxc.multiply(enp.reshape(live, (pop_size, edge_count, 1)), dst_onehot),
            axes=1,
        )
        hops = evoxc.add(evoxc.slice(state, (0, node_count), (pop_size, 1)), one)
        return enp.concatenate([evoxc.maximum(current, marked), hops], axis=1)

    carried = evoxc.while_loop(cond, body, carried)
    # Read the reached set at the candidate source's row, in the same flat ``(P, width)``
    # address space the hop uses - the counter column is simply never addressed.
    src_slot = enp.reshape(evoxc.add(src_row, offsets), (pop_size,))
    reached = enp.reshape(
        evoxc.gather(enp.reshape(carried, (pop_size * width,)), src_slot, axis=0),
        (pop_size, 1),
    )
    # No convergence guard: the loop spends its full ``node_count`` hops, so the carried reached
    # set is the saturated closure and the data-dependent spelling's "the last hop grew nothing"
    # term (``logical_not(settled)``, the identity by that saturation argument) is gone with the
    # flag that carried it.
    return enp.reshape(
        evoxc.logical_and(
            evoxc.logical_and(evoxc.not_equal(reached, zero), src_hit), dst_hit
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
