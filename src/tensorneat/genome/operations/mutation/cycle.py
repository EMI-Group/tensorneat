"""Cycle detection for one candidate edge per individual, in a single loop region.

The predicate answers "would adding ``src -> dst`` close a cycle?", i.e. "is ``src`` already
reachable from ``dst``?".  It is **one** ``evoxc.while_loop`` monotone frontier relaxation,
not an unrolled ``max_nodes - 1`` hop relaxation: the reference interpreter pays per traced
op, so 49 hand-unrolled hops cost far more than a 7-op loop body that runs once per *real*
hop of the shallow DAGs a NEAT run produces.

Key resolution is a **dense key -> node-row table**, the same shape the execution schedule
builds (:func:`tensorneat.common.tensorization.key_slot_table`): a ``(P, key_span + 1) int32``
table whose column ``k`` holds the node *row* owning key ``k``, built with **one** flat
rank-1 ``scatter`` of ``P * N`` updates and read with **one** flat rank-1 ``gather`` per
lookup.  Both endpoints of every existing edge plus both endpoints of the candidate edge are
resolved in a single lookup pass, so the whole predicate pays one table build and one gather.
Anything wider than rank 1 is expensive on a compiled backend: StableHLO lowers ``scatter``
to one workgroup of one thread, so a multi-element scatter is fully serialized, and a wide
element type doubles the traffic of a sort.  The dense table keeps every index block flat and
its element type ``int32``: on numpy the graph is **81 traced ops** and **3.2 / 3.5 / 3.7 ms**
per call at pop 500 on evolved payloads of generations 1 / 5 / 10, against **127 ops** and
**9.0 / 9.3 / 9.8 ms** for the sort-based route it replaced in the same interleaved process
A/B (per-round ratios 0.35-0.41), which also moves ``Pipeline.step`` on ``xor_p500`` from
164.4 to 161.1 ms per generation (median ratio 0.967 over six interleaved process pairs) and
lowers peak RSS by 15 MiB.

Static precondition (a host-side law, never a traced branch): every node key and every
connection endpoint that must be visible lies in ``[0, key_span)``.  That is exactly the
window the schedule's own dense key table addresses, so a key outside it is invisible to the
phenotype as well: the predicate agrees by treating it as a miss (an edge with an out-of-span
endpoint is invalid, a candidate endpoint there answers ``False``, i.e. the mutation is
refused for another reason).  ``algorithm/neat/_keys.py::node_keys_for_split`` together with
``first_free_keys`` keeps every structurally allocated key inside the window, and a census on
measured payloads reports 0 out-of-span node keys, so the window is only reachable from an
out-of-band caller.

Duplicate node keys: two rows may carry the same key, and the table is then a function
key -> one *representative* row whose identity is ``scatter`` write order and therefore
undefined.  The answer is invariant, which is why the predicate may rely on it.  Write
``phi`` for that map: every edge endpoint and both candidate endpoints are resolved through
the same ``phi``, so the traversed graph is the key-level graph with each key identified with
its representative, and ``phi`` is *injective* on keys — a row only ever writes its own key's
cell, so ``phi(a) == phi(b)`` means one row carries both keys, i.e. ``a == b``.  A path in the
row graph therefore lifts edge by edge to a key path (consecutive edges agree because the
row-to-key inverse is a function), so "``src`` reachable from ``dst``" is one predicate for
every representative choice.  The dump cell (column ``key_span``) is order-independent by
construction: every out-of-span row writes the sentinel there, so it can never read back as a
live row.

The loop is a monotone set-growth relaxation on a **flat** frontier with one trailing dump
column (``scatter`` is set-only, so a refused edge writes the dump instead of clearing a real
node).  The frontier only ever gains bits and the node count bounds it, so the set stabilises
within ``max_nodes + 1`` hops: the "still growing" flag alone terminates the loop, with no
iteration counter to carry (and no hop cap to fall back on).

A hop as one batched ``dot`` against a dense ``(P, N, N) float32`` adjacency — which turns the
three serialized hop scatters into three wide matrix-vector dispatches — is decision-identical
(it needs ``frontier | A @ frontier``, not ``A @ frontier`` alone, or ``dst`` itself leaves the
set after the first hop) but **rejected on measurement**: +24 traced ops and 30-33 % slower per
call on numpy (median ratio 1.30-1.31 over six interleaved rounds on each of the three evolved
payloads), while adding a quadratic ``(P, N*N + 1)`` float32 buffer (5 MB at pop 500 /
``max_nodes`` 50, 160 MB at pop 1000 / ``max_nodes`` 200) that no static ceiling in this file
bounds.  The scatter hop stays.

The shipped formulation is selected by ``state.kernel_mode`` (:mod:`~.cycle_accelerator`): the
default ``"host"`` is the scatter hop above, and ``"accelerator"`` is the scatter-free spelling
that resolves keys with an equality template (no dense table build) and carries a ``(P, N)``
reached set marked with a broadcast reduction and one reduce over the edge axis, because a
compiled backend lowers ``scatter`` to one workgroup of one thread (about 17-21 ns per *update*
element, i.e. 1.05 ms per hop for the 50 000 update cells of the ``xor_p500`` case at pop 500)
while a reduction spreads over its workgroups.  Both spellings share this module's window law
(:func:`_in_span`) and answer the same boolean for the same payload, and the switch is a static
read of a plain string on the configuration ``State`` — never a traced value — so a graph
contains exactly one of them.

IREE constraint (release gate): the frontier is seeded with a one-hot ``equal`` over the flat
``(P, width)`` index, **never** with a boolean ``scatter``.  A ``scatter`` whose result is the
``while_loop`` init makes ``iree-compile`` (IREE 3.11, llvm-cpu) SIGSEGV on the generation
graph — the same scatter inside the loop body is fine, so only the always-taken seed form is
restricted.  Verify with ``pytest test/backends`` (the ``evoxc[iree]`` extra); the one-hot seed
is the same values and 2 traced ops cheaper than the boolean ``scatter`` it replaced.
"""

import numpy as np
import evoxc
import evoxc.numpy as enp

from ....common import ConfigError
from ...window import fill


def _static_constant(values: np.ndarray) -> evoxc.SymbolicTensor:
    """Embed a host-side ``int32`` array as a single graph constant.

    Args:
        values (np.ndarray): An ``int32`` array of the shape the kernel needs.

    Returns:
        evoxc.SymbolicTensor: The same values inside the graph, one traced op.
    """
    return evoxc.constant(evoxc.from_numpy(np.asarray(values, dtype=np.int32)))


__all__ = ["closes_cycle"]


def _table_offsets(pop_size: int, span: int) -> evoxc.SymbolicTensor:
    """Per-individual flat offset of a ``(P, span + 1)`` table, as one constant.

    Args:
        pop_size (int): ``P``, the number of individuals.
        span (int): ``S``, the genome's static ``key_span``.

    Returns:
        evoxc.SymbolicTensor: ``(P, 1) int32`` holding ``p * (span + 1)`` per row, so a
        column index plus these offsets is a flat index into the table.  The whole block is
        ``4 * P`` bytes — ``(P, 1)`` static data broadcast across the row axis rather than a
        materialised ``(P, N)`` / ``(P, Q)`` index constant.
    """
    return _static_constant(
        np.arange(0, pop_size * (span + 1), span + 1).reshape(pop_size, 1)
    )


def _in_span(
    keys: evoxc.SymbolicTensor,
    zero: evoxc.SymbolicTensor,
    span_constant: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """``0 <= key < key_span``, the window every visible key must lie in.

    Args:
        keys: ``(P, L) int32`` keys (node keys, connection endpoints, or candidate
            endpoints).
        zero (evoxc.SymbolicTensor): The rank-0 ``int32`` literal ``0``, shared per trace.
        span_constant (evoxc.SymbolicTensor): The rank-0 ``int32`` literal ``key_span``,
            shared per trace.

    Returns:
        ``(P, L) bool``: "this key is addressable".

    Notes:
        The range test is the schedule's own ``0 <= key < key_span`` in the key's **signed**
        dtype (:func:`tensorneat.common.tensorization._key_indices`), the two-comparison form
        that keeps every operand an ``int32``/``bool``; the one-comparison
        ``cast(key, uint32) < span`` alternative is not portable, because no graph in this
        package may carry an unsigned dtype.
        The two bound constants are operands rather than arguments so a caller tracing
        several key blocks (:mod:`~.cycle_accelerator`) pays each ``constant`` op once; this
        is the single spelling of the window law, shared by the dense table and the
        equality-template resolution so the two formulations can never disagree about which
        keys are visible.
    """
    return evoxc.logical_and(
        evoxc.greater_equal(keys, zero), evoxc.less(keys, span_constant)
    )


def _table_cell(keys: evoxc.SymbolicTensor, span: int) -> tuple:
    """Map keys to dense-table columns, sending every out-of-range key to the dump cell.

    Args:
        keys: ``(P, L) int32`` keys (node keys, connection endpoints, or candidate
            endpoints).
        span (int): ``S``, the genome's static ``key_span``.

    Returns:
        ``(cell, in_range)``: ``(P, L) int32`` columns in ``[0, span]`` — ``span`` is the
        dump cell — and the ``(P, L) bool`` mask "this key is addressable".

    Notes:
        The range test is :func:`_in_span`, the schedule's own window law.
    """
    span_constant = _static_constant(np.asarray(span))
    in_range = _in_span(keys, _static_constant(np.asarray(0)), span_constant)
    return evoxc.select(in_range, keys, span_constant), in_range


def _key_table(
    node_keys: evoxc.SymbolicTensor,
    key_span: int,
) -> evoxc.SymbolicTensor:
    """Build the dense key -> node-row table of one population, with one flat scatter.

    Args:
        node_keys (evoxc.SymbolicTensor): ``(P, N) int32`` node key rows (``I_INF`` padded).
        key_span (int): ``S``, the genome's static ``key_span`` — the addressable key window.

    Returns:
        evoxc.SymbolicTensor: ``(P, S + 1) int32``; ``table[p, k]`` is the row of the node
        whose key is ``k``, and ``N`` (the **sentinel**, never a live row index) where no row
        holds ``k``.  Column ``S`` is the dump cell every out-of-range key resolves to.

    Notes:
        One ``scatter`` of ``P * N`` updates, flat: a row whose key is in range writes its
        own row index at that key's cell, and a row whose key is outside ``[0, S)`` writes
        the sentinel into the dump cell — so the dump cell is ``N`` however many rows land
        there.  The seed is a **one-element** constant plus one ``broadcast_to``
        (:func:`tensorneat.genome.window.fill`); a ``(P, S + 1)`` graph constant would be
        300 kB at pop 500 and 600 kB at pop 1000.
    """
    pop_size, node_count = node_keys.shape
    span = int(key_span)
    sentinel = _static_constant(np.asarray(node_count))
    cell, in_range = _table_cell(node_keys, span)
    # The row index each row writes: its own row when its key is addressable, the sentinel
    # otherwise, so the dump cell never holds a live row.
    values = evoxc.select(
        in_range,
        _static_constant(np.arange(0, node_count).reshape(1, node_count)),
        sentinel,
    )
    flat_index = enp.reshape(
        evoxc.add(cell, _table_offsets(pop_size, span)), (pop_size * node_count,)
    )
    table = evoxc.scatter(
        fill((pop_size * (span + 1),), node_count, dtype=evoxc.int32),
        flat_index,
        enp.reshape(values, (pop_size * node_count,)),
        axis=0,
    )
    return enp.reshape(table, (pop_size, span + 1))


def _rows_of(
    table: evoxc.SymbolicTensor,
    keys: evoxc.SymbolicTensor,
    key_span: int,
    node_count: int,
) -> tuple:
    """Resolve a block of keys to node rows through the dense table, with one flat gather.

    Args:
        table (evoxc.SymbolicTensor): ``(P, S + 1) int32`` output of :func:`_key_table`.
        keys (evoxc.SymbolicTensor): ``(P, Q) int32`` keys to resolve — connection endpoints
            or candidate endpoints.
        key_span (int): ``S``, matching the table.
        node_count (int): ``N``, the sentinel the table stores for a miss.

    Returns:
        ``(rows, hit)``, both ``(P, Q)``: the row owning each key (always in
        ``[0, node_count)``, so it is safe to index with) and the "this key is a live node"
        mask.  Every consumer must still gate on ``hit``: a missed key resolves to the last
        row, which is a real, unrelated node.

    Notes:
        The gather is flat because a per-row gather index is the only spelling ``evoxc``
        offers (``axis=0`` with a ``(P, Q)`` index splices the index shape *after* the axis,
        giving ``(P, P, Q)``), so the table is flattened once and addressed by
        ``cell + p * (S + 1)``.  ``hit`` is one ``not_equal`` against the sentinel and
        ``rows`` one ``minimum`` clamp; no ``select`` is needed because the clamp is exact
        (the sentinel is ``node_count``, the largest addressable row is ``node_count - 1``).
    """
    pop_size, width = table.shape
    span = int(key_span)
    if int(width) != span + 1:
        raise ConfigError(
            f"key table width {int(width)} does not match key_span + 1 ({span + 1})"
        )
    cell, _in_range = _table_cell(keys, span)
    flat_index = enp.reshape(
        evoxc.add(cell, _table_offsets(pop_size, span)),
        (int(pop_size) * int(keys.shape[1]),),
    )
    gathered = evoxc.gather(
        enp.reshape(table, (pop_size * (span + 1),)), flat_index, axis=0
    )
    got = enp.reshape(gathered, tuple(keys.shape))
    return (
        evoxc.minimum(got, _static_constant(np.asarray(node_count - 1))),
        enp.not_equal(got, _static_constant(np.asarray(node_count))),
    )


def _closes_cycle_host(
    node_keys: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    src_keys: evoxc.SymbolicTensor,
    dst_keys: evoxc.SymbolicTensor,
    key_span: int,
) -> evoxc.SymbolicTensor:
    """The shipped scatter hop — :func:`closes_cycle`'s default formulation.

    Args:
        node_keys: ``(P, N) int32`` node key rows, ``I_INF`` padded.
        conn_keys: ``(P, E, CK) int32`` connection key rows.
        src_keys: ``(P,) int32`` candidate edge source node key.
        dst_keys: ``(P,) int32`` candidate edge destination node key.
        key_span (int): ``S``, the static addressable key window.

    Returns:
        ``(P,) bool``, the answer :func:`closes_cycle` documents.

    Raises:
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.
        ConfigError: when ``key_span`` does not describe the table it builds.

    Notes:
        Split out of :func:`closes_cycle` unchanged, so the formulation switch is one host
        branch and the shipped graph keeps its exact op order and constants.
    """
    pop_size, node_count = node_keys.shape
    edge_count = conn_keys.shape[1]
    width = node_count + 1
    table = _key_table(node_keys, key_span)
    # Both endpoints of every existing edge plus the candidate edge, in one key-match pass.
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
    # Flat index blocks, built once: every hop then indexes the frontier directly.  The row
    # offsets and the dump column are pure static data, so they are graph constants (one op
    # each) rather than an ``arange`` plus a broadcast.
    offsets = _static_constant(
        np.arange(0, pop_size * width, width, dtype=np.int32).reshape(pop_size, 1)
    )
    edge_positions = pop_size * edge_count
    src_flat = enp.reshape(evoxc.add(edge_src, offsets), (edge_positions,))
    dst_flat = enp.reshape(evoxc.add(edge_dst, offsets), (edge_positions,))
    # The dump column's flat index is the per-individual row offset plus ``node_count``,
    # identical for every edge of that individual: ``(P, 1)`` static data broadcast across
    # the edge axis instead of a materialised ``(P * E,)`` constant (``P * E`` entries --
    # 40 k, 200 k and 400 k at pop 100/500/1000).
    dump_flat = enp.reshape(
        enp.broadcast_to(
            evoxc.add(offsets, _static_constant(np.asarray(node_count))),
            (pop_size, edge_count),
        ),
        (edge_positions,),
    )
    # An edge with an absent endpoint resolves to a row the hit test rejected; redirecting it
    # to the dump column *once* here folds the ``valid`` term into the hop's gather (the dump
    # column is never written, so it always reads ``False``) and saves one ``O(P*E)`` boolean
    # op on every hop.  ``select`` is loop-invariant, the ``logical_and`` it replaces was not.
    src_flat = evoxc.select(
        enp.reshape(edge_valid, (edge_positions,)), src_flat, dump_flat
    )
    # Seed the frontier at the destination as a one-hot row over the flat ``(P, width)``
    # frontier.  ``width`` carries the dump column, and ``dst_row`` lies in
    # ``[0, node_count)``, so that column starts ``False``.  The seed is **not** a
    # ``scatter``: a boolean scatter whose result is the ``while_loop`` init makes
    # ``iree-compile`` (IREE 3.11, llvm-cpu) SIGSEGV on the generation graph, while this
    # ``equal``/``logical_and`` pair (4 ops, same values) compiles — see the module note.
    frontier = enp.reshape(
        evoxc.logical_and(
            evoxc.equal(enp.arange(0, width, dtype=evoxc.int32), dst_row), dst_hit
        ),
        (pop_size * width,),
    )
    def cond(carried):
        """Continue while the frontier gained a node."""
        _frontier, growing = carried
        return growing

    def body(carried):
        """One hop: every edge whose source is reached marks its destination."""
        frontier, _growing = carried
        live = evoxc.gather(frontier, src_flat, axis=0)
        grown = evoxc.scatter(
            frontier, evoxc.select(live, dst_flat, dump_flat), live, axis=0
        )
        return grown, evoxc.reduce_max(evoxc.not_equal(grown, frontier), axes=0)

    frontier, _ = evoxc.while_loop(
        cond,
        body,
        (frontier, enp.full((), True, dtype=evoxc.bool_)),
    )
    # Read the frontier at the candidate source's slot: flat index = row offset + column.
    src_slot = enp.reshape(evoxc.add(src_row, offsets), (pop_size,))
    reached = enp.reshape(
        evoxc.gather(frontier, src_slot, axis=0), (pop_size, 1)
    )
    return enp.reshape(
        evoxc.logical_and(evoxc.logical_and(reached, src_hit), dst_hit), (pop_size,)
    )


def closes_cycle(
    state,
    node_keys: evoxc.SymbolicTensor,
    conn_keys: evoxc.SymbolicTensor,
    src_keys: evoxc.SymbolicTensor,
    dst_keys: evoxc.SymbolicTensor,
    key_span: int,
) -> evoxc.SymbolicTensor:
    """Ask whether adding one edge per individual would close a cycle.

    A cycle is closed exactly when ``src`` is reachable from ``dst`` in the graph that
    already exists, so the predicate seeds a frontier at ``dst`` and grows it along the
    existing edges until it stops growing; a self-loop on an existing node answers ``True``.

    Args:
        state (Any): The trace-time configuration ``State``; its
            :data:`~.cycle_accelerator.KERNEL_MODE_KEY` entry selects the formulation.
        node_keys: ``(P, N) int32`` node key rows, ``I_INF`` padded.
        conn_keys: ``(P, E, CK) int32`` connection key rows — columns ``0``/``1`` are the
            source and destination node keys; absent rows are ``I_INF``.
        src_keys: ``(P,) int32`` candidate edge source node key.
        dst_keys: ``(P,) int32`` candidate edge destination node key.
        key_span (int): ``S``, ``genome.key_span`` — the static addressable key window of the
            dense table, the same one the execution schedule uses.  It cannot be derived from
            any operand shape, so the caller passes it.

    Returns:
        ``(P,) bool`` True where the candidate edge would close a cycle.  A degenerate
        candidate (either endpoint absent from the genome, ``I_INF``, or outside
        ``[0, key_span)``) answers ``False``: the mutation is refused for another reason,
        never this one.

    Raises:
        TensorNEATError: If the state's kernel-mode entry is not one of the documented
            formulations.
        evoxc.ShapeError / evoxc.BackendError: Propagated unchanged.
        ConfigError: when ``key_span`` does not describe the table it builds (a host-side
            configuration error, never a traced branch).

    Notes:
        Both formulations answer the same boolean — the visibility law (:func:`_in_span`) and
        the "every endpoint goes through the same injective key -> row map" argument are
        shared — so the caller's decision never depends on the switch: the default
        ``"host"`` is :func:`_closes_cycle_host`, whose frontier is a flat ``(P, N + 1)``
        vector written with one ``scatter`` per hop and whose keys come from the dense table,
        and ``"accelerator"`` is
        :func:`~.cycle_accelerator.closes_cycle_accelerator`, which resolves keys with an
        equality template (no ``scatter``) and carries a ``(P, N)`` reached set marked with a
        broadcast reduction instead of a write, because a compiled backend serialises
        ``scatter`` (see that module's note and its measured device numbers).
        The accelerator route runs a **static** trip count of ``node_count`` hops (a breadth-first
        closure from one source over ``node_count`` nodes saturates inside that budget), so its
        loop condition is a hop counter and not a growth flag; its one carried value is the
        reached set with that counter in its last column, and the answer reads that value, because
        an unused ``while`` result makes ``iree-compile`` emit an illegal ``scf.while`` for the
        assembled generation graph (same module note).
        The branch is a static host read — the entry is a plain string on the configuration
        ``State`` — so a graph contains exactly one of the two spellings and no traced value
        ever drives it.  The batch axis is written explicitly — no ``vmap`` over the region.
    """
    from . import cycle_accelerator

    if cycle_accelerator.selected(state):
        return cycle_accelerator.closes_cycle_accelerator(
            node_keys, conn_keys, src_keys, dst_keys, key_span
        )
    return _closes_cycle_host(node_keys, conn_keys, src_keys, dst_keys, key_span)
