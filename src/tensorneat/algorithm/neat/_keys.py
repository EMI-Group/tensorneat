"""Bounded structural node-key allocation for :class:`NEAT`.

The mutation kernel splices a fresh node row into an individual's gene payload
with the key the algorithm hands it.  The *schedule* that turns that payload
into the fixed-shape execution graph addresses node keys through a **dense key
table of width** ``genome.key_span``
(:func:`tensorneat.common.tensorization.key_slot_table`), and a key outside
``[0, key_span)`` is silently unmatched there: the node row exists in the
payload but never appears in the phenotype.
``Transformed.edge_overflow`` reports only an in-degree overflow, so an
out-of-span key trips none of the pipeline's guards.

:func:`node_keys_for_split` is the allocator :class:`NEAT` uses: a **wrapped
per-individual spread** over the free-key window ``[max_nodes, key_span)`` -- the
only part of the table no payload row can occupy -- with
:func:`first_free_keys` as the collision fallback.
:func:`validate_key_space` rejects a genome whose table is too narrow to
guarantee that a free key always exists.

All three live here rather than in ``_kernels.py`` because that module is at its
line budget and this is a self-contained allocation concern:
``validate_key_space`` is host-side (constructor validation) while
``first_free_keys`` and ``node_keys_for_split`` are plain graph helpers that run
inside the active trace.
"""

from __future__ import annotations

import operator
from typing import Any

import evoxc
import evoxc.numpy as enp
import numpy as np

from tensorneat.common.errors import TensorNEATError
from tensorneat.common.tools import batch_offsets

from ._kernels import static_i32

__all__ = ["first_free_keys", "node_keys_for_split", "validate_key_space"]


def validate_key_space(genome: Any) -> None:
    """Reject a genome whose key table cannot hold one more structural node.

    Host-side and static: called from :meth:`NEAT.__init__`, before anything is
    traced.  An individual's payload holds at most ``max_nodes`` live node rows,
    so ``key_span > max_nodes`` proves that a key in ``[0, key_span)`` is always
    free and :func:`first_free_keys` can never run out.  At the default bound
    (``max_nodes + max_conns``) the slack is exactly ``max_conns`` keys.

    Args:
        genome: The genome :class:`NEAT` will evolve.  Only its host-side
            ``key_span`` and ``max_nodes`` statics are read; nothing is traced.

    Returns:
        None.

    Raises:
        TensorNEATError: If ``key_span`` or ``max_nodes`` is not an integer, or
            if ``key_span <= max_nodes`` -- the schedule would then be allowed
            to drop a structurally allocated key without any guard noticing.
    """
    try:
        key_span = operator.index(genome.key_span)
        max_nodes = operator.index(genome.max_nodes)
    except (AttributeError, TypeError):
        raise TensorNEATError(
            f"NEAT: genome {type(genome).__name__} must expose integer "
            "'key_span' and 'max_nodes' statics for the structural node-key "
            f"allocator, got key_span={getattr(genome, 'key_span', None)!r} and "
            f"max_nodes={getattr(genome, 'max_nodes', None)!r}"
        ) from None
    if key_span <= max_nodes:
        raise TensorNEATError(
            "structural node-key allocation needs at least one free key inside "
            f"[0, key_span): key_span={key_span}, max_nodes={max_nodes}; an "
            "individual holds at most max_nodes live node keys, so key_span must "
            "exceed max_nodes or the schedule's dense key table silently drops a "
            "structurally allocated node. Raise "
            f"{type(genome).__name__}(key_span={max_nodes + 1}) (or lower "
            "max_nodes)"
        )


def first_free_keys(node_keys: Any, key_span: int) -> Any:
    """Lowest node key in ``[0, key_span)`` that each individual does not hold.

    Runs inside the active trace.  One occupancy table per individual is built
    as "every key free except the ones this row holds", and the allocated key is
    the first cell still free.  This is the **fallback primitive** of
    :func:`node_keys_for_split`: it is free by construction, so it is only ever
    consulted for the rows whose spread candidate the payload already holds.

    Args:
        node_keys: ``(P, max_nodes) int32`` node keys, ``I_INF`` padded.  Only
            the row's own keys are read: a key held by *another* individual is
            irrelevant (see the note on cross-individual sharing below).
        key_span (int): Static key bound ``S`` of the schedule's dense key
            table; ``[0, key_span)`` is the addressable range.

    Returns:
        ``(P,) int32``: for every row, the lowest key in ``[0, key_span)`` that
        the row does not hold, hence a key that is free for that individual and
        always visible to the schedule.

    Raises:
        TensorNEATError: If ``node_keys`` is not 2-D ``(P, max_nodes)`` or
            ``key_span`` is not positive -- both are static wiring errors.

    Note:
        **Availability proof.**  A row holds at most ``max_nodes`` keys and
        ``key_span > max_nodes`` is enforced statically by
        :func:`validate_key_space`, so at least ``key_span - max_nodes`` cells
        are always free (``max_conns`` at the default bound).  The ``argmin``
        below therefore always returns an index inside ``[0, key_span)``.

        **Dump-cell guard.**  The occupancy row is one cell wider than the key
        range and its last cell is permanently ``1``.  Absent (``I_INF``) and
        out-of-range keys are redirected there by one ``select`` *before* the
        scatter, so (a) an illegal payload can never produce an out-of-span
        index -- ``argmin`` would have to walk past the dump cell -- and (b) no
        index outside ``[0, key_span)`` is ever formed, which is what lets the
        scatter run without the escape redirect.  Multiple keys landing on the
        dump cell write the same ``1``, so duplicate indices are harmless on
        both the numpy interpreter (last write wins) and StableHLO (the result
        value is identical whichever write is kept).

        **Cross-individual sharing is legal and expected.**  Keys are
        per-individual namespaces: two individuals with the same occupied set
        receive the same new key, and the compatibility distance treats equal
        keys as homologues -- which is exactly the intended meaning (both
        individuals invented "the same" structural node).  The mutation kernel
        only ever inspects an individual's own rows, so nothing requires a key
        to be unique across the population.

        **Op cost.**  Twenty-two static-shape ops per generation, measured on
        the traced graph: six ``constant`` (one-element blocks, the dump index,
        the range bounds and the flat row offsets), three ``broadcast``, one
        ``concatenate``, the range test (``greater_equal``, ``less``,
        ``logical_and``), one ``select``, ``add`` + four ``reshape`` for the
        flat index and the buffer round-trips, one rank-1 ``scatter``, one
        ``argmin`` and one ``cast``.  A ``(P, key_span)`` graph constant would
        be 600 kB at ``pop_size=1000`` (and would warn at trace time), which is
        why both blocks are a broadcast of a one-element constant instead.
        The scatter is spelled rank-1 (``flat = row * width + col``) because the
        StableHLO v1 exporter only supports rank-1 ``axis=0`` and rank-2
        row/column writes, and a ``(P, max_nodes)`` index block is neither.
    """
    shape = tuple(int(size) for size in node_keys.shape)
    if len(shape) != 2:
        raise TensorNEATError(
            "first_free_keys needs a 2-D node_keys of shape (pop_size, "
            f"max_nodes), got shape {shape}"
        )
    span = operator.index(key_span)
    if span <= 0:
        raise TensorNEATError(
            f"first_free_keys needs a positive key_span, got {key_span!r}"
        )
    pop_size, row_count = shape
    if node_keys.dtype != evoxc.int32:
        # The genome documents ``(P, N) int32``; cast only when it does not
        # already hold that dtype, so the default path pays no ``cast`` op.
        node_keys = evoxc.cast(node_keys, evoxc.int32)

    width = span + 1
    # Occupancy table: ``(P, span)`` zeros plus the trailing dump cell ``1``.
    zero = enp.full((1, 1), 0, dtype=evoxc.int32)
    one = enp.full((1, 1), 1, dtype=evoxc.int32)
    occupancy = enp.concatenate(
        [
            enp.broadcast_to(zero, (pop_size, span)),
            enp.broadcast_to(one, (pop_size, 1)),
        ],
        axis=1,
    )

    # In-span test; ``I_INF`` padding and any negative key fail it and are
    # redirected to the dump cell before an index is formed.
    dump = enp.full((), span, dtype=evoxc.int32)
    in_range = evoxc.logical_and(
        evoxc.greater_equal(node_keys, enp.full((), 0, dtype=evoxc.int32)),
        evoxc.less(node_keys, dump),
    )
    slot = evoxc.select(in_range, node_keys, dump)

    # Rank-1 scatter on the flattened table: cell ``p * width + slot[p, n]``.
    flat_index = evoxc.reshape(
        evoxc.add(slot, batch_offsets(pop_size, width)), (pop_size * row_count,)
    )
    written = evoxc.scatter(
        evoxc.reshape(occupancy, (pop_size * width,)),
        flat_index,
        enp.broadcast_to(enp.full((1,), 1, dtype=evoxc.int32), (pop_size * row_count,)),
        axis=0,
    )

    lowest = evoxc.argmin(evoxc.reshape(written, (pop_size, width)), axis=1)
    return evoxc.cast(lowest, evoxc.int32)


def node_keys_for_split(
    node_keys: Any, key_span: int, max_nodes: int, base: Any
) -> Any:
    """Wrapped per-individual spread node key, with a lowest-free fallback.

    Runs inside the active trace.  Per individual ``p`` the candidate is

    ``max_nodes + (base[p] + (1 + p)) mod (key_span - max_nodes)``

    i.e. the pre-rewrite ``max_node_key + 1 + slot`` formula wrapped into the
    *free-key window* ``[max_nodes, key_span)`` -- the part of the dense key
    table that no payload row can occupy, since a row holds at most
    ``max_nodes`` keys.  The spread keeps the character of the unbounded
    formula: a per-individual monotone base plus the population-slot offset, so
    different individuals (and different generations) receive different keys
    instead of all landing on the lowest free one.  That matters for search
    quality, not for correctness: the compatibility distance aligns node genes
    *by key* and charges non-homologous keys to the disjoint term, so a compact
    lowest-free policy inflates node homology across the population and
    collapses speciation (measured on XOR3d, pop 500: final best ``-0.25`` with
    a peak of 3 species, against ``-8.76e-06``, the ``-0.05`` target met at
    generation 73, with a peak of 19-20 species for the spread).

    The window is only ``key_span - max_nodes`` keys wide -- ``max_conns`` at
    the default span -- so with ``pop_size`` above that width the wrapped
    candidates collide with keys the individual already holds.  The candidate is
    therefore tested exactly like the genome's ``plan_add_node`` alias guard
    (``reduce_max(equal(node_keys, candidate))``), and the rows that hold it
    fall back to :func:`first_free_keys`, which is free by construction.  An
    allocation is thus never a key the payload already owns and never outside
    ``[0, key_span)``.

    Args:
        node_keys: ``(P, max_nodes) int32`` node keys of the payload that will
            receive the key (the crossover child), ``I_INF`` padded.
        key_span (int): Static width ``S`` of the schedule's dense key table.
        max_nodes (int): Static node-row count, i.e. the number of keys an
            individual can hold, hence the start of the free-key window.
        base (Any): ``(P,) int32`` per-individual spread base, normally
            ``genome.max_node_key(state, node_keys)``.  Taken as an argument so
            this module stays a plain graph helper with no ``State``/genome
            wiring; ``NEAT._new_gene_keys`` supplies it.

    Returns:
        ``(P,) int32``: for every row, ``max_nodes + (base + 1 + p) mod
        (key_span - max_nodes)`` when that key is free for the row, else the
        row's lowest free key in ``[0, key_span)``.

    Raises:
        TensorNEATError: If ``node_keys`` is not 2-D ``(P, max_nodes)``,
            ``base`` is not ``(P,)``, ``key_span`` is not positive, or
            ``key_span <= max_nodes`` (no free-key window) -- all static wiring
            errors.

    Note:
        **No overflow concern.**  ``base`` is the individual's largest present
        key and every key ``genome.initialize`` or this allocator writes is below
        ``key_span``, so ``base + 1 + p < key_span + pop_size``; the sum stays far
        inside the int32 range even at ``pop_size`` in the tens of thousands, and
        ``remainder`` then lands it in ``[0, key_span - max_nodes)``.  A negative
        ``base`` would break that (``remainder`` keeps the dividend's sign), but
        no legal payload carries one: keys start at ``0`` and nothing here emits
        a negative key.

        **Cross-individual sharing.**  The window is ``key_span - max_nodes``
        keys wide, so with ``pop_size`` above it two individuals can be handed
        the same key; that is the pre-rewrite behaviour and the intended meaning
        of "both invented the same structural node" -- the distance reads the
        shared key as a homologue.

        **Op cost.**  Thirteen traced ops plus the fallback's twenty-two: four
        ``constant`` (the ``(1 + p) mod W`` offset vector, ``W``, ``max_nodes``
        and ``0``), ``add``, ``remainder``, ``add``, ``reshape`` (the candidate
        as a column), ``equal``, ``cast``, ``reduce_max``, ``not_equal`` and
        ``select``.  The fallback runs unconditionally rather than behind a
        ``cond``: its result is discarded on every collision-free row, and an
        ``if`` region costs more than those rows save.
    """
    shape = tuple(int(size) for size in node_keys.shape)
    if len(shape) != 2:
        raise TensorNEATError(
            "node_keys_for_split needs a 2-D node_keys of shape (pop_size, "
            f"max_nodes), got shape {shape}"
        )
    span = operator.index(key_span)
    nodes = operator.index(max_nodes)
    window = span - nodes
    if window <= 0:
        raise TensorNEATError(
            "node_keys_for_split needs a non-empty free-key window: "
            f"key_span={span}, max_nodes={nodes}; see validate_key_space"
        )
    pop_size = shape[0]
    base_shape = tuple(int(size) for size in base.shape)
    if base_shape != (pop_size,):
        raise TensorNEATError(
            f"node_keys_for_split needs base of shape ({pop_size},), got "
            f"{base_shape}"
        )
    if node_keys.dtype != evoxc.int32:
        node_keys = evoxc.cast(node_keys, evoxc.int32)
    if base.dtype != evoxc.int32:
        base = evoxc.cast(base, evoxc.int32)

    # ``(1 + p) mod W`` is static, so it is baked as one constant op; adding it
    # to the base is the same value as ``(base + 1 + p) mod W``.
    offsets = static_i32((np.arange(pop_size, dtype=np.int64) + 1) % window)
    wrapped = evoxc.remainder(
        evoxc.add(base, offsets), enp.full((), window, dtype=evoxc.int32)
    )
    candidate = evoxc.add(wrapped, enp.full((), nodes, dtype=evoxc.int32))

    column = evoxc.reshape(candidate, (pop_size, 1))
    held = evoxc.reduce_max(
        evoxc.cast(evoxc.equal(node_keys, column), evoxc.int32), axes=1
    )
    fallback = first_free_keys(node_keys, span)
    return evoxc.select(
        evoxc.not_equal(held, enp.full((), 0, dtype=evoxc.int32)),
        fallback,
        candidate,
    )
