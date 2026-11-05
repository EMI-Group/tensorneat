"""Index-free homologue resolution: an equality table instead of the sort merge.

This module is the second spelling of :func:`~.digest.measure_alignment`.  Both answer the
same question for every gene of a centre genome — *which digest row carries the same key?* —
and both return a :class:`~.digest.AlignedGenes`, so the distance tail
(:meth:`~tensorneat.genome.operations.distance.base.BaseDistance.distance_from_alignment`,
:meth:`~.default.DefaultDistance.distance_from_alignment` and its single scalar tail) is
literally the same code on both routes.  That is what makes the two routes **bit-identical**
rather than merely close: they differ only in how the homologue table is produced.

The merge sorts the ``2 * M`` population-plus-centre composites and reads each centre gene's
*successor*, which is ``O(M log M)`` but spends two ``argsort`` calls per family per call —
on a compiled backend those expand into ``(P, M, M)`` count-mode blocks anyway (see
``## Compiled-backend dispatch census`` in this node's ``CONTEXT.md``).

This formulation asks the question directly and without an index permutation: it compares
every centre composite against every population composite of the same individual with one
broadcast equality, so the answer is the ``(P, M, M)`` boolean table
``table[p, i, j] = (centre[p, i] == digest[p, j])``.  Two reductions read it:

* ``homologous = reduce_max(table, axes=2)`` — "some population gene carries this key" (a
  boolean *any*, so the order of the reduction is irrelevant);
* ``col = argmax(table, axis=2)`` — the **first** matching population column, which is the
  very row the merge's successor test finds: the digest half is sorted ascending and stably,
  so equal composites appear in original row order.

The nested table is what the formulation buys and what it costs: no ``argsort``, no
permutation, no merge-wide ``int64`` block, and every op below the table is row-local — at the
price of an ``O(M^2)`` temporary that the numpy reference interpreter is memory-bound on (see
``## Formulation seam`` in this node's ``CONTEXT.md``).  It is therefore selected by
``state.distance_formulation`` and defaults to the merge: a GPU-class device asks for it
(``host_payload._formulation_statics``), a CPU keeps the merge.

Side tags
---------
The merge gives the centre tag ``0`` and the population tag ``1`` and looks for
``composite + 1``.  This route gives the **centre the population tag** instead, so a
homologue is a plain equality and no ``+1`` neighbour test is needed — the comparison *is*
the homology test.  The two sides cannot alias:

* a real population composite is odd and at most ``2**63 - 2**32 - 3``;
* a real centre composite is the same odd value of the digest's own fold, so equality is
  exact key equality (the fold is injective on the folded columns);
* both padding sentinels are **even** (``_PAD_CENTRE`` = ``2**63 - 2``, ``_PAD_POPULATION`` =
  ``2**63 - 4``), while a real composite of either side is odd and tops out at
  ``2**63 - 2**32 - 3``, so an absent centre row equals no population composite and **no
  presence mask is needed on the population side** — the digest's ``2M``-wide ``valid`` block
  is never read here.

That is why this module adds nothing to :func:`~.digest.prepare_family`: the digest it
consumes is byte-for-byte the one the merge consumes, and a generation's ``prepare`` op and
dispatch counts are unchanged by the switch.

Duplicate key rows
------------------
The payload contract requires unique key rows inside one genome.  When it is violated the
routes can disagree, and only there: with two identical centre genes and one digest row for
that key, the merge's successor test marks **only the last** of the two (the first one's
successor is its twin, not the population composite) while this route marks both and points
both at the same column.  That divergence is measured and documented, not repaired: the
merge is the shipped contract and ``measure_alignment`` is deliberately left alone — see
``## Test Strategy`` in this node's ``CONTEXT.md``.
"""

from typing import TYPE_CHECKING

import evoxc

from ....common import ConfigError, TensorNEATError
from . import digest as _digest
from .consts import ConstantPool
from .digest import AlignedGenes, _FOLDED_KEYS, _TAG_POPULATION

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .layout import MeasureStatics

#: State entry selecting the homologue-resolution formulation of ``measure``.
#:
#: ``"merge"`` is the shipped sort-based successor test
#: (:func:`~.digest.measure_alignment`); ``"index_free"`` is this module's equality table.
#: The entry is a plain Python string on the trace-time configuration ``State`` — read on the
#: host, never a traced value — and a state that does not carry it gets ``"merge"``.
DISTANCE_FORMULATION_KEY = "distance_formulation"

#: Accepted values of :data:`DISTANCE_FORMULATION_KEY`, in documentation order.
DISTANCE_FORMULATION_MODES = ("merge", "index_free")

#: Formulation a state without the entry gets — the kernel-side default, matching
#: ``host_payload``'s fallback for a state no ``Pipeline.setup`` produced.
_DEFAULT_FORMULATION = "merge"


def formulation_mode(state) -> str:
    """Resolve :data:`DISTANCE_FORMULATION_KEY` from the trace-time ``State``.

    Args:
        state (Any): The configuration ``State`` the distance kernel received.  The entry is
            a host value, never a traced one, so this is a plain attribute read: it emits no
            graph op and behaves identically inside and outside a trace.

    Returns:
        str: ``"merge"`` when the entry is absent, else the stored formulation.

    Raises:
        TensorNEATError: If the entry holds anything but the two documented formulations, so
            a typo fails loudly instead of silently selecting the other route.
    """
    mode = getattr(state, DISTANCE_FORMULATION_KEY, _DEFAULT_FORMULATION)
    if mode not in DISTANCE_FORMULATION_MODES:
        raise TensorNEATError(
            f"state.{DISTANCE_FORMULATION_KEY} must be one of "
            f"{DISTANCE_FORMULATION_MODES}, got {mode!r}"
        )
    return mode


def selected(state) -> bool:
    """Whether ``state`` asks for this module's formulation.

    Args:
        state (Any): The trace-time configuration ``State``.

    Returns:
        bool: ``True`` when :func:`formulation_mode` resolves to ``"index_free"``.

    Raises:
        TensorNEATError: If the state's entry is not one of the documented formulations.

    Notes:
        The dense single-centre route (:mod:`.dense`) is scatter-based and is a measured
        regression on the accelerator target this formulation exists for, so
        :meth:`~.base.BaseDistance.measure` skips it whenever this predicate holds: the
        formulation switch must never take that branch.
    """
    return formulation_mode(state) == "index_free"


def alignment(
    state,
    gene,
    keys: evoxc.SymbolicTensor,
    digest_keys: evoxc.SymbolicTensor,
    digest_key_cols: evoxc.SymbolicTensor,
    *,
    statics: "MeasureStatics",
    consts: ConstantPool | None = None,
) -> AlignedGenes:
    """Resolve one centre gene family by the formulation ``state`` selects.

    Args:
        state (Any): The trace-time configuration ``State``; its
            :data:`DISTANCE_FORMULATION_KEY` entry picks the route.
        gene: Node or connection gene declaring the key width.
        keys (evoxc.SymbolicTensor): ``(P, M)`` or ``(P, M, K) int32`` centre key array.
        digest_keys (evoxc.SymbolicTensor): ``(P, M) int64`` population composites.
        digest_key_cols (evoxc.SymbolicTensor): ``(P, M, K) int32`` population key columns.
        statics (MeasureStatics): The family's per-generation scaffolding from the digest.
        consts (ConstantPool | None): Pool for the rank-0 constants of this trace.

    Returns:
        AlignedGenes: the digest row, the homology mask, the centre's gene count and its
        homologue count, in the record both routes return.

    Raises:
        TensorNEATError: If the state's formulation entry is not one of the documented
            spellings.
        ConfigError: Propagated from the selected route.

    Notes:
        The one seam :meth:`~.base.BaseDistance._measure_family` calls, so the choice of
        formulation is a single host branch and no call site duplicates the two spellings.
        Both branches are plain traced helpers (never ``@evoxc.defn``) evaluated inside the
        caller's graph.
    """
    if selected(state):
        return resolve(
            gene,
            keys,
            digest_keys,
            digest_key_cols,
            statics=statics,
            consts=consts,
        )
    return _digest.measure_alignment(
        gene, keys, digest_keys, digest_key_cols, statics=statics, consts=consts
    )


def resolve(
    gene,
    keys: evoxc.SymbolicTensor,
    digest_keys: evoxc.SymbolicTensor,
    digest_key_cols: evoxc.SymbolicTensor,
    *,
    statics: "MeasureStatics",
    consts: ConstantPool | None = None,
) -> AlignedGenes:
    """Locate every centre gene's homologue row with one broadcast equality.

    Args:
        gene: Node or connection gene declaring the key width.
        keys (evoxc.SymbolicTensor): ``(P, M)`` or ``(P, M, K) int32`` centre key array,
            ``I_INF`` padded.
        digest_keys (evoxc.SymbolicTensor): ``(P, M) int64`` population composites, as
            produced by :func:`~.digest.prepare_family` — ascending, padding last.
        digest_key_cols (evoxc.SymbolicTensor): ``(P, M, K) int32`` population key columns in
            the same order — the exact-match oracle for a key wider than ``_FOLDED_KEYS``.
        statics (MeasureStatics): The family's per-generation scaffolding: every rank-0
            scalar and both shapes come from it, so this call traces **no** ``constant`` op
            of its own (the marker-column path draws the same two pooled constants the merge
            draws).
        consts (ConstantPool | None): Pool to draw rank-0 constants from; ``None`` starts a
            private one.

    Returns:
        AlignedGenes: ``rows`` is the first matching population column (``0`` where there is
        none, like the merge's redirect), ``homologous`` the equality mask, ``count`` the
        centre's real genes and ``matched`` its homologue count — the same shapes and dtypes
        the merge returns, so the shared tail reads neither route differently.

    Raises:
        ConfigError: If the payload key width disagrees with the gene, or if the centre and
            digest key tables carry a different gene-axis length.

    Notes:
        The centre fold is *exactly* :func:`~.digest.measure_alignment`'s, with the side tag
        flipped from ``_TAG_CENTRE`` to ``_TAG_POPULATION``: same branch on the key width,
        same folded columns, same padding sentinel ``_PAD_CENTRE`` for an absent row (which
        no odd population composite can equal).  That is the whole bit-identity argument for
        a payload that respects the unique-key contract — the fold is injective, so tag-plus-
        equality is full key equality, and :func:`~.digest._key_row_matches` confirms the
        columns the ``int64`` composite cannot hold, on the same candidate row.
        ``argmax`` returns ``0`` for an all-``False`` row and every column it can name is in
        range, so no clamp and no ``I_INF`` redirect are needed; ``rows`` is redirected to
        ``statics.zero_i32`` by the ``select`` regardless, which is the value the tail masks
        away anyway.
        Everything below the table is row-local, so a :class:`~.grouped.GroupedDigest`
        (whose statics sit at ``group * P`` rows) and a centre broadcast to the digest's
        batch are measured unchanged — neither is special-cased here.
    """
    pool = consts if consts is not None else ConstantPool()
    width = _digest._key_width("measure", gene, keys)
    pop_size, count = int(keys.shape[0]), int(keys.shape[1])
    if tuple(int(size) for size in digest_keys.shape) != (pop_size, count):
        raise ConfigError(
            f"measure: the centre's key table is {(pop_size, count)} but the digest's is "
            f"{tuple(digest_keys.shape)}; pass one digest row per centre row"
        )
    valid, tagged = _centre_composites(keys, statics)
    # ``(P, M, 1)`` against ``(P, 1, M)``: ``evoxc.equal`` broadcasts a static pair, so the
    # table costs the two view reshapes and one compare, never a materialised operand.
    table = evoxc.equal(
        evoxc.reshape(tagged, (pop_size, count, 1)),
        evoxc.reshape(digest_keys, (pop_size, 1, count)),
    )
    # A boolean *any* over the population axis: the reduction order is irrelevant, so no
    # float summation-order question arises.  ``argmax`` of the same table is the first
    # matching column — the row the merge's successor test finds (see the module docstring).
    homologous = evoxc.reduce_max(table, axes=2)
    col = evoxc.cast(evoxc.argmax(table, axis=2), evoxc.int32)
    if width > _FOLDED_KEYS:
        # A key width the ``int64`` composite cannot hold (``OriginConn``'s
        # ``historical_marker``) is confirmed row-for-row against the candidate digest row,
        # exactly as the merge does, so homology is full-tuple key equality.
        homologous = evoxc.logical_and(
            homologous,
            _digest._key_row_matches(
                keys,
                digest_key_cols,
                col,
                homologous,
                width,
                statics,
                pool,
            ),
        )
    rows = evoxc.select(homologous, col, statics.zero_i32)
    return AlignedGenes(
        rows,
        homologous,
        evoxc.reduce_sum(valid, axes=1),
        evoxc.reduce_sum(homologous, axes=1),
    )


def _centre_composites(
    keys: evoxc.SymbolicTensor, statics: "MeasureStatics"
) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
    """Fold the centre key table into population-tagged composites.

    Args:
        keys (evoxc.SymbolicTensor): ``(P, M)`` or ``(P, M, K) int32`` centre key array.
        statics (MeasureStatics): The family's per-generation scaffolding, read for the fold
            radix, the side-tag scale, the two sentinels and the ``(2, 1)`` fold weights.

    Returns:
        ``(valid (P, M) bool, tagged (P, M) int64)`` — the real-gene mask and the composite
        the equality table compares.

    Notes:
        The branch is :func:`~.digest.measure_alignment`'s, unchanged: a key row with at
        least ``_FOLDED_KEYS`` columns folds with one ``dot`` (:func:`~.digest._pack_centre`)
        and reaches the padding sentinel by construction, everything else folds with
        :func:`~.digest._pack` and is tagged by :func:`~.digest._tagged`.  Only the tag value
        differs — ``_TAG_POPULATION`` here, ``_TAG_CENTRE`` there — which is what turns the
        merge's ``composite + 1`` neighbour test into the table's plain equality.
        Keeping the branch (rather than asserting one shape) is what keeps the fold
        bit-identical to the merge's for *every* payload the merge accepts, a rank-3,
        one-column key table included.
    """
    if len(keys.shape) == 3 and int(keys.shape[2]) >= _FOLDED_KEYS:
        # ``_pack_centre`` already lands on ``_PAD_CENTRE`` for an absent row, so only the tag
        # ``add`` and the ``select`` remain: two traced ops.
        composite, valid = _digest._pack_centre(keys, statics)
        return valid, evoxc.select(
            valid,
            evoxc.add(composite, statics.one_i64),
            statics.pad_centre_i64,
        )
    # ``scale`` below is the fused side-tag multiply of a rank-2 key table; ``_tagged`` then
    # adds the tag itself (``scale=None``), so the two spellings cost the same three ops.
    pack, valid = _digest._pack(
        keys, statics.iinf_i32, statics.key_base_i64, scale=statics.tag_scale_i64
    )
    return valid, _digest._tagged(
        pack,
        valid,
        _TAG_POPULATION,
        statics.pad_centre_i64,
        None,
        statics.one_i64,
    )


__all__ = [
    "DISTANCE_FORMULATION_KEY",
    "DISTANCE_FORMULATION_MODES",
    "alignment",
    "formulation_mode",
    "resolve",
    "selected",
]
