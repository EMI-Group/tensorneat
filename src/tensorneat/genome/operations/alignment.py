"""Exact-key homologue alignment, shared by crossover and compatibility distance.

Both operators ask the same question about two batches of parents — *which gene of parent 1
has a homologue in parent 2?* — so the answer is computed here once, as a **rank merge**:

1. each gene row is reduced to one sortable integer *identity* (:func:`row_identity`) —
   for a gene with one or two ``int32`` key columns the columns are packed side by side,
   which is injective, not merely a hash.  The carrier is ``int64`` (the only 64-bit dtype
   every backend declares) and the packed value has its sign bit flipped
   (:func:`_order_ready`) so a *signed* sort orders it exactly like the unsigned value the
   packing describes;
2. the two parents' identities are concatenated and ``argsort``-ed **once**, so inside a
   group of equal identities every parent-1 row sorts before every parent-2 row;
3. each gene's rank in that sorted order is read back by **inverting the permutation with
   one ``sort``** (:func:`_successor_ranks`) instead of a second ``argsort``: the positions
   are packed into the low bits of the permutation's own values, so one ``int32`` sort of
   that encoded block *is* the inverse permutation — cheaper than the ``scatter`` write it
   replaced, and it emits no scatter at all;
4. a parent-1 gene has a homologue exactly when **the element right after it in sorted
   order is a parent-2 row carrying the same identity**: one row gather of the permutation
   followed by one row gather of the merged identities.

Homology is exact equality across the whole key-column tuple (``in_key``/``out_key`` plus
``historical_marker`` when the gene declares one), so the table is independent of the gene
family and of the row a gene happens to occupy — parent 2's rows may be in any order.
That is ``O(M log M)`` per individual in one batch-wide kernel over the population axis —
no Python loop over individuals, no ``vmap``, no ``scan``/``while_loop``/``cond``.

A gene with more than two key columns (``OriginConn``'s ``historical_marker``) cannot be
packed into one 64-bit identity, so :func:`row_identity` packs the **first two** columns —
the pair that identifies a gene, see the precondition below — and
:func:`find_homologous` **confirms** every candidate against the remaining columns, which
keeps homology equal to whole-tuple key-row equality.

The one precondition of the rank merge is the payload's own uniqueness invariant: *the first
two key columns are unique within one parent*.  It holds for both gene families the NEAT
payload ships — a node gene has a single key column (the node key, unique per genome) and a
connection gene's first two columns are its ``(in_key, out_key)`` pair, which
``execute_mutation`` refuses to duplicate and ``initialize`` builds once — and it is what
makes "the successor in sorted order" the *one* candidate partner of a gene.  A payload
that repeats an ``(in_key, out_key)`` pair would leave the pair's rows indistinguishable,
and homology there is undefined rather than merely expensive.

Two contract details drive the implementation, both about cost rather than maths:

* ``evoxc.gather`` follows numpy ``take`` semantics, so a per-individual row read is a flat
  1-D gather with ``row * row_len + column`` indices (that is what `gather_attr_rows`
  builds).  :func:`~tensorneat.common.tools.flat_gather` pays for escape checks and a fresh
  ``arange`` on every call; every index here is built in range by construction — a clamped
  rank, or a row already redirected by ``select`` — so the plain op is enough and the
  ``arange`` block is hoisted per row length (:func:`~tensorneat.common.tools.batch_offsets`).
* the reference interpreter charges per element, not merely per op: the rank merge is
  **two sorts** (the merge ``argsort`` of the ``(P, 2M)`` identities and the encoded
  ``int32`` rank sort), two row gathers and the masks around them, every one of them a
  single pass over a ``(P, M)``/``(P, 2M)`` block rather than one pass per attribute
  column.  The merge emits **no scatter**: the rank inversion is a value sort, which is
  both cheaper on the reference interpreter and one portability rule fewer.

Nothing here is differentiable (neither operator enters the ``grad`` graph); the sort and the
``int32`` index arithmetic are integer ops only.

:func:`key_rows` sits beside the merge because it is the same contract's front door: the
alignment is written against ``(P, M, K)`` key columns, so a caller with a two-dimensional
node-key table normalises it with that one helper instead of re-deriving the rank rule.
"""

import evoxc
import evoxc.numpy as enp

from ...common import I_INF, ConfigError, batch_offsets

#: Bits one key column occupies in the packed row identity.  A key is a non-negative
#: ``int32`` — the payload contract says keys live in ``[0, key_span)`` and ``I_INF =
#: 2**31 - 1`` is the pad sentinel — so 31 bits hold every value a column can carry.
_COLUMN_BITS = 31

#: Low bits of the packed identity reserved for the parent tag.
_TAG_BITS = 2

#: Tag bit of a parent-2 row.  Parent 1's tag is ``0``, so its rows need no extra op.
_TAG_PARENT_2 = 1

#: Number of key columns packed into the identity.  ``2 * 31`` bits plus the tag bits fill
#: an ``int64`` exactly, so the identity is injective over the whole legal key domain; the
#: columns of a wider gene are confirmed by :func:`_confirm_rows` instead.
_PACKED_KEY_WIDTH = 2

#: Largest value the ``int64`` identity carrier holds.  A flipped identity above it has to
#: be read back as the negative two's-complement value, exactly as the traced
#: ``bitwise_xor`` wraps it.
_INT64_MAX = (1 << 63) - 1


def _column(x: evoxc.SymbolicTensor, index: int) -> evoxc.SymbolicTensor:
    """Read one static column of the last axis of a traced tensor.

    Args:
        x (evoxc.SymbolicTensor): Tensor whose last axis is the column axis.
        index (int): Static (non-negative) column index.

    Returns:
        evoxc.SymbolicTensor: ``x.shape[:-1]``-shaped view of column ``index``.

    Notes:
        Written as a kept-axis slice plus ``reshape`` like
        :func:`~tensorneat.genome.gene.base.column`, because ``evoxc`` rejects
        ``Ellipsis``/``None`` static indexing and an axis-dropping integer index is not a
        legal traced read.
    """
    rank = len(x.shape)
    slicer = (slice(None),) * (rank - 1) + (slice(index, index + 1),)
    return evoxc.reshape(x[slicer], tuple(x.shape[:-1]))


def _scalar(value: int, dtype) -> evoxc.SymbolicTensor:
    """Rank-0 constant of ``dtype`` (a bare literal would widen an ``int32`` tensor).

    Args:
        value (int): Static integer baked into the trace.
        dtype: ``evoxc`` dtype of the result.

    Returns:
        evoxc.SymbolicTensor: Rank-0 tensor inside the active trace.
    """
    return enp.full((), value, dtype=dtype)


def _order_ready(value: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Flip the sign bit so a *signed* sort orders a packed identity like the unsigned value.

    Args:
        value (evoxc.SymbolicTensor): ``int64`` packed row identity, tag bits clear.

    Returns:
        evoxc.SymbolicTensor: The same identity with bit 63 inverted.

    Notes:
        The pack (:func:`row_identity`) is two 31-bit columns plus two tag bits, which fills
        64 bits exactly: the top column's bits do reach the sign bit, so a *signed*
        ``<``/``sort``/``argsort`` would order the identities differently from the unsigned
        value the packing was designed for (and the ``I_INF``-column padding identity would
        sort as a small number instead of the largest one).  An ``xor`` with the single sign
        bit is a bijection, so *equality* is bit-identical — every equality site
        (:func:`find_homologous`'s successor test, the padding test, homology itself) is
        unaffected — while it maps unsigned order onto signed order.  It also commutes with
        setting the low tag bit (the tag is bit 0, the flip is bit 63), so the parent-2
        tag can be applied before or after it.

        Same-width two's-complement arithmetic is bit-identical for ``add``/``or``/``shift``
        / ``cast``, so the ``int64`` carrier computes exactly the same 64 bits an unsigned
        carrier would; only *ordering* needs this flip.  ``-(1 << 63)`` is spelled
        negatively because a *value* of ``1 << 63`` does not fit ``int64``; the two have the
        same bit pattern.
    """
    return evoxc.bitwise_xor(value, _scalar(-(1 << 63), evoxc.int64))


def key_rows(keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Normalise a gene key table to the ``(P, M, K)`` key-column block.

    This is the key-table rank normaliser shared by crossover and compatibility distance:
    the alignment is written against three-dimensional key columns, while the payload keeps
    a node key in a single ``int32`` column, so ``node_keys`` is ``(P, N)`` while
    ``conn_keys`` is already ``(P, E, CK)``.  The implicit key axis is added here rather
    than special-cased inside the kernels.

    Args:
        keys (evoxc.SymbolicTensor): ``(P, M)`` or ``(P, M, K)`` ``int32`` key table,
            ``I_INF`` padded.

    Returns:
        evoxc.SymbolicTensor: ``(P, M, K)`` view of ``keys`` — the very same tensor when it
        already carries a key axis, otherwise a ``(P, M, 1)`` view of the single key column.

    Raises:
        ConfigError: If the table is neither 2-D nor 3-D (both ranks are static at trace
            time, so this is a host-side validation of a build-time property).  It is a
            :class:`~tensorneat.common.errors.TensorNEATError` subclass.
    """
    rank = len(keys.shape)
    if rank == 3:
        return keys
    if rank != 2:
        raise ConfigError(
            "gene key tables must be (P, M) or (P, M, K), "
            f"got shape {tuple(keys.shape)}"
        )
    pop_size, count = keys.shape
    return evoxc.reshape(keys, (pop_size, count, 1))


def row_identity(keys: evoxc.SymbolicTensor) -> evoxc.SymbolicTensor:
    """Reduce one key row to the sortable integer that stands for its key tuple.

    Args:
        keys (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of one gene
            family, ``I_INF`` padded.

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` ``int64`` row identities, low :data:`_TAG_BITS`
        bits clear (they hold the parent tag, see :func:`find_homologous`) and sign-flipped
        by :func:`_order_ready` so signed order equals unsigned order.

    Notes:
        Packing is ``column 0 << 31 | column 1``, shifted up by :data:`_TAG_BITS`: a key is
        a non-negative ``int32`` and ``I_INF = 2**31 - 1`` is the largest value one column
        can carry, so the pack is **injective over the whole legal key domain** — pad
        sentinel included — and two distinct key prefixes can never share an identity.  The
        carrier is ``int64``, the only 64-bit dtype every backend declares, and same-width
        two's-complement ``or``/``shift``/``cast`` produce the identical 64-bit pattern an
        unsigned carrier would; only the *ordering* of the result differs, which is why the
        value is sign-flipped before it is returned (see :func:`_order_ready`).

        Columns past :data:`_PACKED_KEY_WIDTH` do not fit one 64-bit integer, so they are
        left out of the sort key: they make the identity a *prefix* of the key tuple, and
        :func:`find_homologous` confirms every candidate against them with
        :func:`_confirm_rows`.  That is exact as long as the packed prefix is unique per
        parent, which the payload guarantees (a node gene has one key column; a connection
        gene's first two columns are its unique ``(in_key, out_key)`` pair).
    """
    pop_size, count = keys.shape[0], keys.shape[1]
    width = keys.shape[2]
    fold = evoxc.int64
    if width == 1:
        identity = evoxc.cast(evoxc.reshape(keys, (pop_size, count)), fold)
    else:
        identity = evoxc.cast(_column(keys, 0), fold)
        shift = _scalar(_COLUMN_BITS, fold)
        for index in range(1, min(width, _PACKED_KEY_WIDTH)):
            identity = evoxc.bitwise_or(
                evoxc.bitwise_left_shift(identity, shift),
                evoxc.cast(_column(keys, index), fold),
            )
    return _order_ready(evoxc.bitwise_left_shift(identity, _scalar(_TAG_BITS, fold)))


def _pad_identity(width: int) -> int:
    """The row identity of a padded row, i.e. of ``width`` ``I_INF`` key columns.

    Args:
        width (int): Static number of key columns of the gene family.

    Returns:
        int: The identity :func:`row_identity` produces for an absent row, computed on the
        host with the same arithmetic — so one ``not_equal`` separates parent 1's padding
        from a real gene instead of a per-column comparison.

    Notes:
        This is the host mirror of :func:`row_identity`, **sign flip included**, because the
        two are compared elementwise.  It has to stay in step with the traced pack (same
        stride, same reserved tag bits, same ``_order_ready`` flip), or padding rows would
        stop being recognised; both read :data:`_COLUMN_BITS`/:data:`_TAG_BITS` and apply
        the same bit, so only a change of *formula* can make them diverge.

        The pad identity is the largest value the pack can produce (``2**64 - 4`` for a
        two-column key, the ``I_INF`` column pair), which read as a signed ``int64`` is
        ``-4`` — a small number, so an unflipped signed sort would order padding rows
        *below* every real gene, the opposite of the unsigned reading.  Flipping the same
        sign bit :func:`row_identity` flips restores that order (``2**63 - 4`` for two
        columns) and keeps padding above every real identity of its gene family.

        For a one-column key the flip pushes the value past the carrier's ceiling
        (:data:`_INT64_MAX`), so the result is wrapped back into ``int64`` range the way the
        traced XOR wraps it; ordering is unaffected because every one-column identity of the
        pack is wrapped by the same bit.
    """
    identity = I_INF
    for _ in range(1, min(width, _PACKED_KEY_WIDTH)):
        identity = (identity << _COLUMN_BITS) | I_INF
    flipped = (identity << _TAG_BITS) ^ (1 << 63)
    # The traced ``bitwise_xor`` wraps in two's complement, so this host mirror must wrap
    # too: a one-column pad identity lands above ``int64``'s ceiling and comes back
    # negative, and an unwrapped value would not even fit ``enp.full(..., int64)``.
    return flipped - (1 << 64) if flipped > _INT64_MAX else flipped


def _gather_rows(
    source: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    offsets: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Gather one row of a ``(P, R)`` source per individual: ``out[p, i] = source[p, rows[p, i]]``.

    Args:
        source (evoxc.SymbolicTensor): ``(P, R)`` batch-first source.
        rows (evoxc.SymbolicTensor): ``(P, M)`` ``int32`` row indices, every one already in
            ``[0, R)`` — this helper does not clamp, so a caller that can produce an
            escaping index must redirect it first.
        offsets (evoxc.SymbolicTensor): ``(P, 1)`` result of
            :func:`~tensorneat.common.tools.batch_offsets` for ``R``, hoisted by the caller
            so several gathers over the same source share one ``arange``.

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` gathered rows.
    """
    pop_size, row_len = source.shape
    flat = evoxc.reshape(source, (pop_size * row_len,))
    return evoxc.gather(flat, evoxc.add(rows, offsets), axis=0)


def _successor_ranks(
    order: evoxc.SymbolicTensor,
    pop_size: int,
    count: int,
) -> evoxc.SymbolicTensor:
    """Invert the sort permutation and advance one position — one ``sort``, no ``scatter``.

    Args:
        order (evoxc.SymbolicTensor): ``(P, 2M)`` ``int32`` ``argsort`` permutation, i.e.
            position ``s`` of ``order`` holds the merged row that sorts at ``s``.
        pop_size (int): Static ``P``.
        count (int): Static ``M`` (genes per parent, so the block is ``2M`` wide).

    Returns:
        evoxc.SymbolicTensor: ``(P, M) ``int32`` where entry ``i`` is the sorted position of
        the successor of merged row ``i`` — the rank of row ``i`` plus one, clamped to the
        last position.  Only the ``M`` parent-1 rows are returned; the parent-2 half is not
        read by the merge.

    Notes:
        The inverse of a permutation can be *sorted* out instead of *written*: encoding
        ``order[s] * 2**K + min(s + 1, 2M - 1)`` (``K`` = bit length of ``2M - 1``, so the
        two fields never overlap) makes the encoded value unique per row, and sorting it
        ascending puts ``row t`` at position ``t`` — so the low ``K`` bits of entry ``t`` are
        exactly ``rank(t) + 1`` clamped.  The successor position, the ``+1`` and the clamp
        are one static table (:func:`_successor_positions`) that rides in the low field, so
        no ``(P, 2M)`` ``add`` or ``minimum`` follows the inversion.

        Measured against the equivalent ``scatter`` inversion (a rank-1 ``axis=0`` write of
        the same positions): the ``int32`` ``sort`` of the encoded block is 0.05 ms (M = 50)
        and 0.12 ms (M = 100) at ``P`` = 500 against 0.42 ms for the two scatters, and it
        drops the ``(P, 2M)`` blank/update blocks (five ops) with them — 5 ops here against
        twelve.  The encoding is exact for every permutation: ``order`` is a bijection onto
        ``[0, 2M)``, so no two encoded values tie and the sort's stability is irrelevant.

        The clamp is safe: only the gene with the largest identity can sit at the last
        sorted position, and its "successor" is then itself, whose tag is parent 1's while
        the test looks for parent 2's tag — so the clamp cannot invent a homologue.
    """
    cols = 2 * count
    field_bits = (cols - 1).bit_length()
    encoded = evoxc.bitwise_or(
        evoxc.bitwise_left_shift(order, _scalar(field_bits, evoxc.int32)),
        enp.broadcast_to(_successor_positions(cols), (pop_size, cols)),
    )
    ranks = evoxc.sort(encoded, axis=1)
    return evoxc.bitwise_and(
        evoxc.slice(ranks, (0, 0), (pop_size, count)),
        _scalar((1 << field_bits) - 1, evoxc.int32),
    )


def _successor_positions(cols: int) -> evoxc.SymbolicTensor:
    """Static ``(1, 2M)`` block of ``min(position + 1, 2M - 1)`` values for the rank sort.

    Args:
        cols (int): Static block width ``2M``.

    Returns:
        evoxc.SymbolicTensor: ``(1, 2M)`` ``int32`` block, broadcast by the caller.

    Notes:
        This is a *rank*-sized table, not a population-sized one, so it stays a few hundred
        elements however large the batch is: the ``(P, 2M)`` broadcast happens in the
        caller and the clamp costs one op on this table instead of one on the full block.
    """
    return evoxc.minimum(
        enp.arange(1, cols + 1, dtype=evoxc.int32), _scalar(cols - 1, evoxc.int32)
    )


def _flat_row_index(
    rows: evoxc.SymbolicTensor, width: int, pop_size: int, count: int
) -> evoxc.SymbolicTensor:
    """Flat index block reading one ``width``-wide row per gene of a ``(P, M * width)`` payload.

    Args:
        rows (evoxc.SymbolicTensor): ``(P, M)`` ``int32`` row indices, every one already in
            ``[0, M)`` — the read is unmasked, so a caller with absent positions must have
            redirected them first.
        width (int): Static number of columns per row.
        pop_size (int): Static ``P``.
        count (int): Static ``M``.

    Returns:
        evoxc.SymbolicTensor: ``(P, M, width)`` ``int32`` flat indices
        ``row * width + column + p * (M * width)``.

    Notes:
        The per-individual stride is folded into the row term *before* the column term is
        broadcast, so exactly one of the three index ops works on the full ``(P, M, width)``
        block; the other two run on the ``(P, M, 1)`` row block.  Element volume, not op
        count, is what a wide index block costs on the reference interpreter.
    """
    offsets = batch_offsets(pop_size, count * width)
    base = evoxc.add(
        evoxc.multiply(enp.reshape(rows, (pop_size, count)), _scalar(width, evoxc.int32)),
        offsets,
    )
    return evoxc.add(
        enp.reshape(base, (pop_size, count, 1)),
        enp.arange(0, width, dtype=evoxc.int32),
    )


def gather_attr_rows(
    attrs: evoxc.SymbolicTensor, rows: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Pick one attribute row per individual out of parent 2's payload.

    Args:
        attrs (evoxc.SymbolicTensor): ``(P, M, A)`` attribute block of parent 2, NaN padded.
        rows (evoxc.SymbolicTensor): ``(P, M)`` ``int32`` rows into ``attrs``, every one
            already in ``[0, M)`` — an absent homologue must have been redirected by the
            caller's ``select`` (row ``0`` is the usual target) because the read itself is
            unmasked.

    Returns:
        evoxc.SymbolicTensor: ``(P, M, A)`` rows of ``attrs``, one row per gene of parent 1.

    Notes:
        A single flat gather per attribute **row** — not one gather per attribute column —
        so the cost is one op for the whole block.  The index is
        ``row * A + column + p * (M * A)``, the per-individual stride coming from
        :func:`~tensorneat.common.tools.batch_offsets`.

        A one-column family (``DefaultConn``) needs no column term at all: the read
        degenerates to one row gather of the collapsed ``(P, M)`` view, which drops the
        ``(P, M, 1)`` index block (its ``multiply`` and ``arange`` broadcast) in favour of
        the plain ``row + p * M`` index.  The attribute axis is put back with one
        ``reshape`` so the caller's shape contract is unchanged.
    """
    pop_size, count, width = attrs.shape
    if width == 1:
        collapsed = evoxc.reshape(attrs, (pop_size, count))
        gathered = _gather_rows(
            collapsed, rows, batch_offsets(pop_size, count)
        )
        return evoxc.reshape(gathered, (pop_size, count, 1))
    flat = evoxc.reshape(attrs, (pop_size * count * width,))
    return evoxc.gather(flat, _flat_row_index(rows, width, pop_size, count), axis=0)


def _confirm_rows(
    keys1: evoxc.SymbolicTensor,
    keys2: evoxc.SymbolicTensor,
    rows: evoxc.SymbolicTensor,
    candidates: evoxc.SymbolicTensor,
) -> evoxc.SymbolicTensor:
    """Check proposed homologues against the full key rows, column for column.

    Args:
        keys1 (evoxc.SymbolicTensor): ``(P, M, K)`` key columns of parent 1.
        keys2 (evoxc.SymbolicTensor): ``(P, M, K)`` key columns of parent 2.
        rows (evoxc.SymbolicTensor): ``(P, M)`` proposed row in parent 2 per gene of parent
            1; entries that are not candidates are redirected to row ``0`` here.
        candidates (evoxc.SymbolicTensor): ``(P, M)`` bool, the rows worth confirming.

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` bool, True where parent 2's row holds exactly
        parent 1's key tuple.

    Notes:
        Only needed when the gene has more key columns than :data:`_PACKED_KEY_WIDTH` packed
        into :func:`row_identity`; with at most two key columns the identity is injective
        and this pass would always agree.  The packed columns are skipped — they match by
        construction — so a three-column gene pays one gather plus one ``equal`` per spare
        column, folded with ``logical_and`` rather than an innermost-axis reduce (a reduce
        over the column axis costs 10-20 ordinary ops on the reference interpreter).
    """
    pop_size, count, width = keys2.shape
    safe = evoxc.select(candidates, rows, _scalar(0, evoxc.int32))
    flat = evoxc.reshape(keys2, (pop_size * count * width,))
    echoed = evoxc.gather(flat, _flat_row_index(safe, width, pop_size, count), axis=0)
    identical = None
    for index in range(_PACKED_KEY_WIDTH, width):
        column_equal = evoxc.equal(_column(echoed, index), _column(keys1, index))
        identical = (
            column_equal
            if identical is None
            else evoxc.logical_and(identical, column_equal)
        )
    return identical


def find_homologous(
    keys1: evoxc.SymbolicTensor, keys2: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Locate, for every gene of parent 1, its homologous row in parent 2.

    Exact match on the whole key-column tuple (``in_key``/``out_key`` plus
    ``historical_marker`` when the gene declares one), so the answer is independent of the
    row a gene happens to occupy — parent 2's rows may be in any order.

    Args:
        keys1 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 1,
            ``I_INF`` padded.
        keys2 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 2,
            ``I_INF`` padded. Both parents share ``M`` and ``K``: the payload shapes are
            static at build time (the caller validates that, see
            :meth:`~tensorneat.genome.operations.crossover.base.BaseCrossover.__call__`).

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` ``int32`` row index into ``keys2`` per gene of
        parent 1, with ``I_INF`` where parent 2 has no homologue and for parent 1's padded
        rows.

    Raises:
        evoxc.ShapeError: Propagated unchanged if the two parents' shapes differ.

    Notes:
        One ``argsort``, one encoded ``sort`` and two row gathers carry the whole merge —
        the sorted identity sequence is *gathered* through the permutation instead of being
        sorted a second time, and the rank of each gene is the encoded ``sort`` of the
        permutation.  "Immediately
        after me" is the right successor to look at because the **parent tag lives in the
        low bits of the sort key** (:func:`row_identity` leaves them clear and parent 2's
        rows set them), so inside a group of equal identities parent 1's row is strictly
        smaller than parent 2's and the ordering does not depend on the sort being stable —
        which is what lets the sort run unstable, 4x faster on the reference interpreter.
        The ``int64`` identities are already sign-flipped by :func:`row_identity`, so this
        ``argsort`` orders them exactly as the packed unsigned values would; without that
        flip the top key column's bits would reorder the group here.

        The successor's identity is therefore tested against *own identity with parent 2's
        tag set*: an equal answer means the successor is a parent-2 row holding exactly this
        key tuple, and it needs no separate "is it a parent-2 row" comparison.  Two padded
        rows also get different tags, so the tag alone keeps them apart; the explicit
        padding-identity test is what keeps a padded parent-1 row from reporting a row at
        all.

        The successor index is clamped to the last position.  Only the gene with the largest
        identity can ever be clamped, and then the successor is the gene itself, whose tag
        differs from the tag being tested for — so the clamp cannot invent a homologue.

        A gene family with more key columns than :data:`_PACKED_KEY_WIDTH` gets one extra
        pass (:func:`_confirm_rows`) because its identity covers only the packed prefix.
    """
    rows, present = _homologous(keys1, keys2)
    return _masked_rows(rows, present)


def homologous_pair(
    keys1: evoxc.SymbolicTensor, keys2: evoxc.SymbolicTensor
) -> tuple:
    """The two halves of :func:`find_homologous`'s answer, without the ``I_INF`` select.

    A caller that needs both *where* the homologue is and *whether* there is one — as
    crossover does, since it gathers parent 2's attributes through the row and then masks
    the recombined result with the same flag — would otherwise pay for the ``I_INF``
    marking here plus a ``not_equal`` and a redirect ``select`` at the call site.  The row
    index is therefore returned already redirected to ``0``, which is the row the caller's
    gather must read for an absent homologue (the value is discarded by the mask).

    Args:
        keys1 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 1,
            ``I_INF`` padded.
        keys2 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 2,
            ``I_INF`` padded.

    Returns:
        tuple: ``(rows (P, M) int32, present (P, M) bool)`` — the row of parent 2 per gene
        of parent 1 with ``0`` where there is no homologue (or the gene is padding), and
        the mask saying which genes have one.

    Raises:
        evoxc.ShapeError: Propagated unchanged if the two parents' shapes differ.

    Notes:
        This is :func:`find_homologous` with ``present`` still available: the merge, the
        successor test and the padding test are shared verbatim, so the two entry points
        cannot disagree about homology.  ``present`` is computed before the redirect, so
        the mask is exactly "parent 2 holds this key tuple".
    """
    return _homologous(keys1, keys2)


def _homologous(
    keys1: evoxc.SymbolicTensor, keys2: evoxc.SymbolicTensor
) -> tuple:
    """The rank merge: raw successor rows plus the "has a homologue" mask.

    Args:
        keys1 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 1,
            ``I_INF`` padded.
        keys2 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 2,
            ``I_INF`` padded.

    Returns:
        tuple: ``(rows (P, M) int32, present (P, M) bool)`` — the row of parent 2 per gene
        of parent 1, redirected to ``0`` where there is no homologue so an unmasked gather
        cannot escape, and the mask that says which genes have one.

    Raises:
        evoxc.ShapeError: Propagated unchanged if the two parents' shapes differ.

    Notes:
        The single implementation behind :func:`find_homologous` and
        :func:`homologous_pair`, which differ only in how they spell "no homologue"
        (``I_INF`` versus a mask), so the two entry points cannot disagree about homology.
        The successor is read as *one* gather of ``order`` (the successor's row) plus one
        gather of ``merged`` at that row — the intermediate ``(P, 2M)`` sorted identity
        sequence of the two-sort formulation is never materialised, and the rank
        inversion is an encoded ``int32`` ``sort`` (:func:`_successor_ranks`) rather than a
        second ``argsort`` or a ``scatter``.
    """
    pop_size, count = keys1.shape[0], keys1.shape[1]
    width = keys1.shape[2]
    tag = _scalar(_TAG_PARENT_2, evoxc.int64)
    identity1 = row_identity(keys1)
    identity2 = evoxc.bitwise_or(row_identity(keys2), tag)
    merged = evoxc.concatenate((identity1, identity2), axis=1)

    offsets = batch_offsets(pop_size, 2 * count)
    order = evoxc.cast(evoxc.argsort(merged, axis=1), evoxc.int32)
    succ_rows = _gather_rows(order, _successor_ranks(order, pop_size, count), offsets)
    matched = evoxc.equal(
        _gather_rows(merged, succ_rows, offsets),
        evoxc.bitwise_or(identity1, tag),
    )
    rows = evoxc.subtract(succ_rows, _scalar(count, evoxc.int32))
    if width > _PACKED_KEY_WIDTH:
        matched = evoxc.logical_and(matched, _confirm_rows(keys1, keys2, rows, matched))
    present = evoxc.logical_and(
        matched,
        evoxc.not_equal(identity1, _scalar(_pad_identity(width), evoxc.int64)),
    )
    return _redirect_rows(rows, present), present


def _redirect_rows(
    rows: evoxc.SymbolicTensor, present: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Point every absent homologue at row ``0`` so an unmasked gather cannot escape.

    Args:
        rows (evoxc.SymbolicTensor): ``(P, M)`` ``int32`` row index, ``-1`` or negative
            where there is no homologue (the successor test's raw answer).
        present (evoxc.SymbolicTensor): ``(P, M)`` bool mask of genes with a homologue.

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` ``int32`` rows, all in ``[0, M)``.
    """
    return evoxc.select(present, rows, _scalar(0, evoxc.int32))


def _masked_rows(
    rows: evoxc.SymbolicTensor, present: evoxc.SymbolicTensor
) -> evoxc.SymbolicTensor:
    """Mark absent homologues with ``I_INF`` — the ``find_homologous`` return form.

    Args:
        rows (evoxc.SymbolicTensor): ``(P, M)`` ``int32`` row index into parent 2.
        present (evoxc.SymbolicTensor): ``(P, M)`` bool mask of genes with a homologue.

    Returns:
        evoxc.SymbolicTensor: ``(P, M)`` ``int32`` rows, ``I_INF`` where absent.
    """
    return evoxc.select(present, rows, _scalar(I_INF, evoxc.int32))


__all__ = [
    "find_homologous",
    "gather_attr_rows",
    "homologous_pair",
    "key_rows",
    "row_identity",
]

