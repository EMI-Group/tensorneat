"""Index-free homologue resolution: an equality table instead of the two-sort rank merge.

This module is the second spelling of :func:`~.alignment.homologous_pair`.  Both answer the
same question for every gene of parent 1 — *which row of parent 2 carries the same key?* —
with the same ``(rows, present)`` record, so the crossover driver is byte-identical whichever
route :data:`KERNEL_MODE_KEY` selects on a contract-conforming payload.

What changes on a compiled backend is the price of ordering.  The merge turns both parents'
packed identities into one ``(P, 2M)`` ``int64`` block, ``argsort``-s it and inverts the
permutation with a second, encoded ``int32`` ``sort``; a compiled backend lowers each of
those to a bitonic sort, one workgroup per row, so the work is nothing like the
``O(M log M)`` the formulation describes.  Measured on the box's MI300X (gfx942, IREE 3.11,
``--iree-hal-target-backends=rocm``) at the ``xor_p500`` geometry (``max_nodes`` 50,
``max_conns`` 100), the conn
family's ``(500, 200)`` ``int32`` sort alone is one dispatch at **2.03 ms** and the node
family's ``(500, 100)`` one **0.28 ms** — the second-largest device cost of a generation
(see ``## Measured cost`` in this node's ``CONTEXT.md``).

This formulation drops both sorts and asks the question directly, in the shape
:func:`tensorneat.genome.operations.distance.index_free.resolve` already uses:

* ``tagged1 = identity1 | _TAG_PARENT_2`` — the merge's hand-built "own identity with
  parent 2's tag set", one elementwise op on ``(P, M)``;
* ``table[p, i, j] = tagged1[p, i] == identity2[p, j]`` — one broadcast ``equal``, a
  ``(P, M, M)`` boolean table, **no permutation and no sort**;
* ``col = reduce_min(select(table, arange(M), M), axes=2)`` — the *leftmost* matching row of
  parent 2, because a non-matching column carries ``M`` (one past the last index) and can
  never win the minimum; the all-``False`` row lands on ``M``, which ``col < M`` reads back
  as "no homologue", so one reduction answers the row and the flag together;
* ``present = found & (identity1 != pad)`` and ``rows = select(present, col, 0)`` — the
  merge's own padding term and redirect, so the two routes' records are equal element by
  element.

Exactness
---------
Under the payload's documented precondition — *the first two key columns are unique within
one parent* (:mod:`~.alignment`) — the identity is injective per parent, so parent 2 has at
most one row per identity and the "leftmost matching row" **is** the merge's successor row.
The two routes therefore produce bit-identical ``(rows, present)``: there is no tie to break
and no tie-breaking rule to reproduce.  The proof is one line per term:

* a real parent-1 gene matches exactly when parent 2 holds its identity, which is exactly
  the merge's ``merged[successor] == identity1 | tag`` test (parent 1's tag is ``0``, so
  inside an identity group every parent-1 row sorts before every parent-2 row and the
  *successor* of the group's single parent-1 row is the group's single parent-2 row);
* padded rows all share :func:`~.alignment._pad_identity`, so they *do* match each other in
  the table — and are removed by exactly the same padding term the merge applies, which is
  why ``present`` is ``False`` for every padded row on both routes;
* a key wider than :data:`~.alignment._PACKED_KEY_WIDTH` (``OriginConn``'s
  ``historical_marker``) is confirmed with the merge's own
  :func:`~.alignment._confirm_rows`, on the candidate row and against the pre-padding mask,
  so homology stays full-tuple key equality.

Duplicate key rows
------------------
A payload that repeats the packed key prefix inside one parent — the documented contract
violation — is the one place the routes can differ, and only there: the merge reads the
*unstable* ``argsort`` order inside the identity group (so its answer is not reproducible
even between two runs of the host route), while this route deterministically picks the
leftmost matching row and marks every duplicate that has one.  That divergence is measured
and documented rather than repaired, exactly as :mod:`.distance.index_free` documents its
own duplicate-row case, and the host formulation stays the shipped contract.

Portability
-----------
``arange`` + ``select`` + ``reduce_min`` is the spelling proven to compile and run on
``llvm-cpu`` and ``rocm`` in :mod:`.distance.index_free`.  ``argmax`` is deliberately never
used: it materialises a full ``(P, M, M)`` ``int64`` iota whose dispatch has no operands, so
IREE's scheduling leaves it outside every ``stream.async.execute`` region and
``iree-compile`` fails the assembled generation graph.  No unsigned dtype, no ``scatter``, no
deferred op (``cumsum``/``erf``/…), no ``iota``.

The switch is a **static host branch**: both spellings are plain traced helpers (never
``@evoxc.defn``), so a graph is exactly one of them and no backend ever sees both.  The
numpy reference interpreter takes the merge, because the ``(P, M, M)`` table is element
volume it pays in full and the merge's sorts are cheap there; ``"host"`` is therefore the
default and only :data:`KERNEL_MODE_KEY` selects this route.
"""

import evoxc
import evoxc.numpy as enp

from ...common import TensorNEATError
from .alignment import (
    _PACKED_KEY_WIDTH,
    _TAG_PARENT_2,
    _confirm_rows,
    _pad_identity,
    _scalar,
    row_identity,
)

#: State entry selecting the homologue-resolution formulation.
#:
#: ``"host"`` is the shipped two-sort rank merge (:func:`~.alignment.homologous_pair`);
#: ``"accelerator"`` is this module's equality table.  The entry is a plain Python string on
#: the trace-time configuration ``State`` — read on the host, never a traced value — and a
#: state that does not carry it gets ``"host"``.
KERNEL_MODE_KEY = "kernel_mode"

#: Accepted values of :data:`KERNEL_MODE_KEY`, in documentation order.
KERNEL_MODES = ("host", "accelerator")

#: Formulation a state without the entry gets — the kernel-side default, matching the
#: shipped behaviour for a state no host code registered the entry on.
_DEFAULT_KERNEL_MODE = "host"


def kernel_mode(state) -> str:
    """Resolve :data:`KERNEL_MODE_KEY` from the trace-time ``State``.

    Args:
        state (Any): The configuration ``State`` the alignment kernel received, or ``None``
            for a direct kernel call.  The entry is a host value, never a traced one, so this
            is a plain attribute read: it emits no graph op and behaves identically inside
            and outside a trace.

    Returns:
        str: ``"host"`` when ``state`` is ``None`` or the entry is absent, else the stored
        formulation.

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
        state (Any): The trace-time configuration ``State``, or ``None``.

    Returns:
        bool: ``True`` when :func:`kernel_mode` resolves to ``"accelerator"``.

    Raises:
        TensorNEATError: If the state's entry is not one of the documented formulations.
    """
    return kernel_mode(state) == "accelerator"


def homologous_pair_accelerator(
    keys1: evoxc.SymbolicTensor, keys2: evoxc.SymbolicTensor
) -> tuple:
    """The equality-table homologue resolution: ``(rows, present)`` with no sort.

    Args:
        keys1 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 1,
            ``I_INF`` padded.
        keys2 (evoxc.SymbolicTensor): ``(P, M, K)`` ``int32`` key columns of parent 2,
            ``I_INF`` padded.

    Returns:
        tuple: ``(rows (P, M) int32, present (P, M) bool)`` — exactly
        :func:`~.alignment.homologous_pair`'s record: the row of parent 2 per gene of parent
        1, redirected to ``0`` where there is no homologue, and the mask that says which
        genes have one.

    Raises:
        evoxc.ShapeError: Propagated unchanged if the two parents' shapes differ.

    Notes:
        The table is the whole cost: ``P * M * M`` booleans compared in one broadcast pass,
        then one ``int32`` ``select`` of the same shape feeding a single innermost-axis
        ``reduce_min``.  Every rank-0 constant and the ``(M,)`` column index are built once
        per call, so a call traces a fixed, population-independent op count.
        The two identifiers are the merge's own (:func:`~.alignment.row_identity` plus the
        parent tag), so this module re-derives no packing rule and cannot disagree with the
        host route about what a homologue is.
        ``identity2`` is compared **tagged** against ``identity1 | tag``, which is literally
        the merge's successor test; subtracting the tag from ``identity2`` first would be the
        same comparison spelled with one op on the other side of the table, so it is not
        done.
    """
    pop_size, count = int(keys1.shape[0]), int(keys1.shape[1])
    width = int(keys1.shape[2])
    tag = _scalar(_TAG_PARENT_2, evoxc.int64)
    identity1 = row_identity(keys1)
    identity2 = evoxc.bitwise_or(row_identity(keys2), tag)
    # ``(P, M, 1)`` against ``(P, 1, M)``: ``evoxc.equal`` broadcasts a static pair, so the
    # table costs two view reshapes and one compare, never a materialised operand.
    table = evoxc.equal(
        evoxc.reshape(evoxc.bitwise_or(identity1, tag), (pop_size, count, 1)),
        evoxc.reshape(identity2, (pop_size, 1, count)),
    )
    # "Leftmost matching parent-2 row", folded in one reduction: every matching column keeps
    # its own index, every other column carries ``count`` -- one past the last index, so it
    # can never win a minimum -- and the row's minimum is therefore the row the merge's
    # successor test finds.  An all-False row lands on ``count``, which ``col < count`` reads
    # back as "no homologue", so the same reduction answers both questions.
    columns = enp.arange(0, count, dtype=evoxc.int32)
    no_match = _scalar(count, evoxc.int32)
    leftmost = evoxc.reduce_min(evoxc.select(table, columns, no_match), axes=2)
    found = evoxc.less(leftmost, no_match)
    if width > _PACKED_KEY_WIDTH:
        # A key width the ``int64`` identity cannot hold is confirmed row-for-row against the
        # candidate row, against the pre-padding mask, exactly as the merge does.
        found = evoxc.logical_and(
            found, _confirm_rows(keys1, keys2, leftmost, found)
        )
    present = evoxc.logical_and(
        found,
        evoxc.not_equal(identity1, _scalar(_pad_identity(width), evoxc.int64)),
    )
    return evoxc.select(present, leftmost, _scalar(0, evoxc.int32)), present


__all__ = [
    "KERNEL_MODE_KEY",
    "KERNEL_MODES",
    "homologous_pair_accelerator",
    "kernel_mode",
    "selected",
]
