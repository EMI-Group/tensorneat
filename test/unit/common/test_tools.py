"""Unit tests for ``tensorneat.common.tools`` — the flat-index kernel helpers.

What this module proves
-----------------------
Every helper in ``tools.py`` is *graph code*: it runs inside an ``@evoxc.defn``
trace, receives batch-first tensors and may only use ``evoxc`` ops. Each test
therefore builds a one-kernel graph around one helper through
:func:`support.runner.run_defn` (the only sanctioned build/run bridge) and
compares the result with an oracle written *here*:

* a plain-numpy expression (row-wise indexing, ``argsort``/``argsort``), or a
  Python write loop for the scatter helpers,
* for the two mutators and :func:`~tensorneat.common.tools.hash_rows`, a
  *property* — a law that no implementation satisfies by accident (zero rate is
  the identity, only legal options are emitted, equal rows hash equal) because
  the exact draw stream is an ``evoxc.random`` implementation detail.

Three post-JAX contracts drive most of the assertions:

* ``evoxc``'s numpy backend **raises** on an out-of-range gather/scatter index
  where JAX clamped, so the masked helpers must sanitize *and* mask, while the
  unmasked ones must keep raising (``test_flat_gather_raises_...``).
* ``evoxc.scatter`` is set-only, so duplicate flat indices resolve
  last-write-wins, and a dropped write must land in a trailing *dump* column —
  never in index ``0``, which is a real element
  (``test_masked_flat_scatter_routes_dropped_writes_to_the_dump_column``).
* Sentinels are typed: ``I_INF`` (``int32`` max) in integer arrays, ``NaN`` in
  float arrays, never mixed (``test_i_inf_and_key_max_are_the_documented_sentinels``).

Everything runs on ``backend="numpy"`` through the shared runner; no optional
dependency, no network, no GPU. ``tensorneat.common`` is reached through the
``common`` fixture (see ``./conftest.py``) because ``import tensorneat`` still
raises from the ``src/`` design skeletons.
"""

from __future__ import annotations

import numpy as np
import pytest

import evoxc

from support.runner import run_defn

# --------------------------------------------------------------------------- #
# oracles: plain numpy, written from the docstrings of tools.py, never from
# the kernel they check (see ../../support/CONTEXT.md)
# --------------------------------------------------------------------------- #


def _np_flat_gather(x, idx):
    """Oracle for ``flat_gather``: index row ``p`` with row ``p`` of ``idx``.

    ``(P, R)`` source, ``(P, *shape)`` indices; the batch axis is explicit, so
    this is exactly the ``take_along_axis`` semantics the flattening emulates.
    """
    rows = np.arange(x.shape[0]).reshape((-1,) + (1,) * (idx.ndim - 1))
    return x[rows, idx]


def _np_flat_scatter(x, idx, updates):
    """Oracle for ``flat_scatter``: a Python write loop, last write wins.

    Restricted to the 2-D index blocks the kernels use; duplicate indices are
    written in flat traversal order, which is what makes the last write stick.
    """
    out = np.array(x, copy=True)
    for p in range(x.shape[0]):
        for k in range(idx.shape[1]):
            out[p, idx[p, k]] = updates[p, k]
    return out


def _np_rank_descending(x):
    """Oracle for ``rank_elements``: rank ``0`` is the largest, ties keep order.

    The rank of an element is its position in the sorted order, i.e. the
    inverse permutation of a *stable* descending ``argsort``.
    """
    return np.argsort(np.argsort(-x, axis=-1, kind="stable"), axis=-1)


def _np_fetch_first(mask, default):
    """Oracle for ``fetch_first``/``fetch_random``: first ``True`` column, else ``default``."""
    found = mask.any(axis=-1)
    return np.where(found, mask.argmax(axis=-1), np.int32(default)).astype(np.int32)


def _np_argmin_with_mask(arr, mask, default):
    """Oracle for ``argmin_with_mask``: minimal selectable column, ``default`` if none."""
    scores = np.where(mask, arr, np.inf)
    return np.where(mask.any(axis=-1), scores.argmin(axis=-1), np.int32(default)).astype(
        np.int32
    )


def _seeded(helper, seed):
    """Wrap a key-taking helper so the closure carries the key as a host constant.

    ``support.runner`` passes tensors and statics only: an ``evoxc.Tensor`` key
    has no ``TensorSpec`` (``spec_like`` refuses it) and a rank-0 numpy key
    arrives with rank 1. A stateless key is a compile-time constant anyway, so
    rebuilding it from the integer seed inside the trace is both simpler and
    closer to how the package uses keys.
    """

    def kernel(*arrays):
        return helper(evoxc.random.key(seed), *arrays)

    return kernel


def _run(common, name, *arrays):
    """Execute one ``tools`` kernel as a single-kernel graph on the numpy backend.

    The module under test comes from the ``common`` fixture and ``name`` is the
    kernel's own name, so a test body reads ``_run(common, "flat_gather", x, idx)``
    and compares the result with the oracle written above. Everything goes through
    :func:`support.runner.run_defn`, the only sanctioned build/run bridge.
    """
    return run_defn(getattr(common.tools, name), *arrays)


# --------------------------------------------------------------------------- #
# sentinels
# --------------------------------------------------------------------------- #


def test_i_inf_and_key_max_are_the_documented_sentinels(common):
    """Invariant: ``I_INF == 2**31 - 1`` (``int32`` max) and ``KEY_MAX == 2**24``.

    Oracle: the ``numpy`` integer limits themselves — ``np.iinfo(np.int32).max``
    for the "no index" sentinel, and the first integer ``float32`` can no longer
    distinguish from its neighbour (``np.float32(KEY_MAX) == np.float32(KEY_MAX + 1)``
    while ``np.float32(KEY_MAX - 1) != np.float32(KEY_MAX)``) for the key bound,
    which is what lets the gene keys be compared by exact float equality.
    """
    pytest.skip(
        "not implemented: I_INF/KEY_MAX equality with the int32/float32 limits "
        "(owner: common/tools.py module constants)"
    )


# --------------------------------------------------------------------------- #
# batch_offsets
# --------------------------------------------------------------------------- #


def test_batch_offsets_matches_the_row_major_oracle(common):
    """Invariant: ``batch_offsets(P, R)`` is ``[[0], [R], [2R], ...]``, shape ``(P, 1)``.

    Oracle: ``np.arange(P)[:, None] * R``; the documented ``(P, 1) int32`` shape
    is asserted too, because broadcasting it against a ``(P, K)`` column block is
    the only thing the helper exists for.
    """
    pytest.skip(
        "not implemented: batch_offsets row offsets, shape (P, 1) and int32 dtype "
        "vs np.arange(P)[:, None] * R (owner: common/tools.batch_offsets)"
    )


@pytest.mark.parametrize("pop_size, row_len", [(0, 4), (4, 0), (-1, 4), (4, -1)])
def test_batch_offsets_rejects_a_non_positive_static(common, pop_size, row_len):
    """Invariant: a non-positive ``pop_size``/``row_len`` is a ``ConfigError``.

    Oracle: the documented ``Raises`` clause — validation happens on the host
    before tracing, so the error must be a ``TensorNEATError`` and not an
    ``IndexError`` from ``arange`` or a traced ``evoxc`` failure.
    """
    pytest.skip(
        "not implemented: ConfigError for pop_size/row_len <= 0 "
        "(owner: common/tools.batch_offsets host validation)"
    )


@pytest.mark.parametrize("pop_size, row_len", [(2**16, 2**15), (2**31, 1)])
def test_batch_offsets_rejects_an_int32_overflowing_flat_range(common, pop_size, row_len):
    """Invariant: ``pop_size * row_len >= I_INF`` raises ``ConfigError``.

    Oracle: the documented bound ``P * R < I_INF``; a flat index must stay inside
    the ``int32`` range the helpers use, and silently wrapping would index a
    different row's data instead of failing.
    """
    pytest.skip(
        "not implemented: ConfigError when pop_size * row_len reaches I_INF "
        "(owner: common/tools.batch_offsets host validation)"
    )


# --------------------------------------------------------------------------- #
# flat_gather / masked_flat_gather
# --------------------------------------------------------------------------- #


def test_flat_gather_matches_row_wise_numpy_indexing(common):
    """Invariant: ``out[p, k] == x[p, idx[p, k]]`` for a ``(P, K)`` index block.

    Oracle: :func:`_np_flat_gather` (row-wise numpy indexing). A ``(P, K)`` block
    is the shape the schedule's ``in_conn``/``in_src`` gathers use.
    """
    pytest.skip(
        "not implemented: flat_gather values vs row-wise numpy indexing on a (P, K) block "
        "(owner: common/tools.flat_gather)"
    )


def test_flat_gather_accepts_a_multi_dimensional_index_block(common):
    """Invariant: a ``(P, I, J)`` index block gathers to ``(P, I, J)``.

    Oracle: :func:`_np_flat_gather` with a broadcast batch axis. The docstring
    promises ``(P, *shape)`` for any static ``*shape``; the ``(P, N, K)``
    incoming-edge tables are exactly such a block.
    """
    pytest.skip(
        "not implemented: flat_gather with a (P, I, J) index block, shape preserved "
        "(owner: common/tools.flat_gather)"
    )


def test_flat_gather_raises_when_an_index_escapes_the_row(common):
    """Invariant: an out-of-range index raises ``evoxc.ShapeError``; it never clamps.

    Oracle: the documented "propagated unchanged" contract, checked against
    ``evoxc.ShapeError`` (what the numpy backend really raises for an
    out-of-range ``gather``). ``I_INF`` is the index that matters here: a caller
    that forgot to mask must fail loudly instead of reading row ``p``'s column 0.
    """
    pytest.skip(
        "not implemented: evoxc.ShapeError for idx == I_INF / idx >= R, never a clamp "
        "(owner: common/tools.flat_gather error contract)"
    )


def test_masked_flat_gather_fills_masked_positions(common):
    """Invariant: ``mask == False`` positions yield ``fill``, masked-in ones the gathered value.

    Oracle: ``np.where(mask, _np_flat_gather(x, idx), fill)`` — the default
    ``fill=NaN`` included, since "absent / not computed" is the forward pass's
    padding value.
    """
    pytest.skip(
        "not implemented: masked_flat_gather fill/mask semantics vs np.where(mask, gather, fill) "
        "(owner: common/tools.masked_flat_gather)"
    )


def test_masked_flat_gather_treats_i_inf_as_absent_without_an_explicit_mask(common):
    """Invariant: ``idx == I_INF`` yields ``fill`` even when ``mask`` is ``None``.

    Oracle: the docstring's two-part predicate — fill where the position is
    masked out **or** where ``idx == I_INF``. The ``I_INF`` half must be enforced
    by the helper itself, otherwise the ``I_INF``-padded schedule tables would
    raise at the backend instead of producing padding.
    """
    pytest.skip(
        "not implemented: I_INF index -> fill with mask=None and no backend error "
        "(owner: common/tools.masked_flat_gather)"
    )


@pytest.mark.parametrize("fill", [0.0, 1.0, float("-inf"), float("inf")])
def test_masked_flat_gather_writes_the_aggregation_neutral_element(common, fill):
    """Invariant: ``fill`` is used verbatim and cast to ``x``'s dtype.

    Oracle: ``np.full(shape, fill, dtype=x.dtype)`` at the masked positions. The
    four parameter values are the neutral elements of ``sum``/``product``/
    ``max``/``min`` in ``common/functions/agg.py``, which is why an unmasked
    aggregation can consume the result: the padding must be the *value* the
    reduction ignores, not just a different bit pattern.
    """
    pytest.skip(
        "not implemented: fill is written verbatim and cast to x's dtype for each neutral element "
        "(owner: common/tools.masked_flat_gather)"
    )


# --------------------------------------------------------------------------- #
# flat_scatter / masked_flat_scatter
# --------------------------------------------------------------------------- #


def test_flat_scatter_matches_a_python_write_loop(common):
    """Invariant: ``out[p, idx[p, k]] == updates[p, k]``, every other cell unchanged.

    Oracle: :func:`_np_flat_scatter`, a plain Python write loop over the same
    ``(P, K)`` block.
    """
    pytest.skip(
        "not implemented: flat_scatter values vs a Python write loop "
        "(owner: common/tools.flat_scatter)"
    )


def test_flat_scatter_resolves_duplicate_indices_last_write_wins(common):
    """Invariant: repeated flat indices resolve to the last update in flat traversal order.

    Oracle: the documented "set-only" note — ``evoxc.scatter`` has no add/max
    mode, so the duplicate must overwrite rather than accumulate; the Python loop
    oracle writes in the same order and therefore predicts the same winner.
    """
    pytest.skip(
        "not implemented: duplicate flat indices keep the last update, no accumulation "
        "(owner: common/tools.flat_scatter)"
    )


def test_masked_flat_scatter_routes_dropped_writes_to_the_dump_column(common):
    """Invariant: masked writes land in column ``R`` and leave columns ``0..R-1`` alone.

    Oracle: ``_np_flat_scatter`` restricted to the masked-in writes, compared
    against ``out[:, :R]``; the dump column is the *only* place a dropped write
    may appear. Routing it to index ``0`` would corrupt a real element, which is
    the pre-rewrite bug this helper exists to prevent.
    """
    pytest.skip(
        "not implemented: masked writes only reach the trailing dump column "
        "(owner: common/tools.masked_flat_scatter)"
    )


def test_masked_flat_scatter_accepts_i_inf_index_when_the_position_is_masked(common):
    """Invariant: a masked position may hold any index, ``I_INF`` included, without raising.

    Oracle: the documented contract — masked indices are replaced by the dump
    index ``R`` *before* the backend sees them, so an ``I_INF``-padded index
    block is safe exactly where the schedule pads it.
    """
    pytest.skip(
        "not implemented: I_INF index on a masked position is routed to the dump slot, no raise "
        "(owner: common/tools.masked_flat_scatter)"
    )


# --------------------------------------------------------------------------- #
# fetch_first / fetch_random / rank_elements / argmin_with_mask
# --------------------------------------------------------------------------- #


def test_fetch_first_matches_the_python_oracle(common):
    """Invariant: the column of the first ``True`` entry of each mask row.

    Oracle: :func:`_np_fetch_first` (``argmax`` of the boolean row). ``argmax`` is
    what the kernel documentation prescribes, so ties and all-``False`` rows are
    the interesting cases.
    """
    pytest.skip(
        "not implemented: fetch_first first-True column vs argmax of the boolean row "
        "(owner: common/tools.fetch_first)"
    )


@pytest.mark.parametrize("default", [None, 0, 7])
def test_fetch_first_returns_the_documented_default_for_an_empty_mask(common, default):
    """Invariant: a row with no ``True`` entry yields ``default`` (``I_INF`` when unset).

    Oracle: :func:`_np_fetch_first` with the same default. The parameter set covers
    the sentinel and two real columns (``0`` and ``7``), because a caller must be
    able to tell "no candidate" from "candidate 0".
    """
    pytest.skip(
        "not implemented: empty mask row -> default (I_INF by default) "
        "(owner: common/tools.fetch_first)"
    )


def test_fetch_random_only_returns_true_positions(common):
    """Invariant: every returned index is a ``True`` column of its own mask row.

    Oracle: re-indexing the mask with the result and asserting it is ``True`` for
    every seeded key; this is the property the ``argmax``-of-uniforms recipe must
    preserve. Keys come from the ``seed`` fixture and are rebuilt inside the
    closure (:func:`_seeded`), so the draw is a function of the key alone.
    """
    pytest.skip(
        "not implemented: fetch_random result is always a True column of its row "
        "(owner: common/tools.fetch_random)"
    )


def test_fetch_random_is_uniform_over_the_row(common):
    """Invariant: over many keys, each ``True`` column of a row is drawn with equal frequency.

    Oracle: the documented sampling argument (the maximiser of i.i.d. draws
    restricted to a set is uniform over that set), checked as an empirical
    frequency bound on a 4-column row over a few hundred keys — loose enough to
    be deterministic, tight enough to catch a biased recipe (e.g. always ``argmax``
    of the mask, or a cumsum over the wrong axis).
    """
    pytest.skip(
        "not implemented: fetch_random empirical uniformity over the True columns "
        "(owner: common/tools.fetch_random)"
    )


def test_fetch_random_returns_the_default_for_an_empty_mask(common):
    """Invariant: a row with no ``True`` entry yields ``default``, never column 0.

    Oracle: :func:`_np_fetch_first` with ``default``; the ``I_INF`` sentinel is
    what lets the caller detect "no candidate" (the legacy helper returned a real
    slot instead, which is the bug this port fixes).
    """
    pytest.skip(
        "not implemented: fetch_random empty mask row -> default "
        "(owner: common/tools.fetch_random)"
    )


def test_rank_elements_matches_the_stable_descending_oracle(common):
    """Invariant: rank ``0`` is the **largest** element (the legacy orientation).

    Oracle: :func:`_np_rank_descending`, i.e. the inverse permutation of a stable
    descending ``argsort``; the default ``reverse=False`` must reproduce it
    exactly, since a flipped orientation silently reorders every parent choice.
    """
    pytest.skip(
        "not implemented: rank_elements default (large-to-small) vs the stable argsort inverse "
        "(owner: common/tools.rank_elements)"
    )


def test_rank_elements_reverse_ranks_small_to_large(common):
    """Invariant: ``reverse=True`` returns the small-to-large ranks, i.e. the complement.

    Oracle: :func:`_np_rank_descending` of ``-x`` (equivalently ``L - 1 - rank``
    for a tie-free row): the two orientations must be permutations of each other
    with mirrored positions.
    """
    pytest.skip(
        "not implemented: rank_elements(reverse=True) small-to-large orientation "
        "(owner: common/tools.rank_elements)"
    )


def test_rank_elements_is_stable_on_ties(common):
    """Invariant: equal values keep their column order, so the rank is deterministic.

    Oracle: :func:`_np_rank_descending` on a deliberately tie-heavy row; stability
    is mandated because the schedule's in-group ranks must not depend on the
    backend's ``argsort`` tie-breaking (a rebuild must not reorder edges).
    """
    pytest.skip(
        "not implemented: rank_elements stable tie-break vs the stable descending oracle "
        "(owner: common/tools.rank_elements)"
    )


def test_argmin_with_mask_matches_the_masked_numpy_oracle(common):
    """Invariant: the index of the minimal ``True`` entry per row.

    Oracle: :func:`_np_argmin_with_mask` (masked-out entries replaced by ``+inf``).
    A row whose masked-in values are all larger than a masked-out one is included,
    so a missing mask shows up as the wrong column instead of a passing test.
    """
    pytest.skip(
        "not implemented: argmin_with_mask values vs np.argmin of the +inf-padded row "
        "(owner: common/tools.argmin_with_mask)"
    )


def test_argmin_with_mask_returns_i_inf_for_an_empty_mask(common):
    """Invariant: a row without a selectable entry yields ``I_INF``, not slot ``0``.

    Oracle: the documented sentinel contract; ``0`` is a real individual, so the
    "empty" answer must be distinguishable — the divergence from the legacy
    helper, which returned ``0``.
    """
    pytest.skip(
        "not implemented: empty mask row -> I_INF from argmin_with_mask "
        "(owner: common/tools.argmin_with_mask)"
    )


# --------------------------------------------------------------------------- #
# mutate_float / mutate_int
# --------------------------------------------------------------------------- #


def test_mutate_float_zero_rates_are_the_identity(common, seed):
    """Invariant: ``mutate_rate == 0`` and ``replace_rate == 0`` returns ``val`` unchanged.

    Oracle: bitwise equality with the input (``np.array_equal`` plus ``dtype``
    and ``shape``), so neither a spurious perturbation nor a dtype promotion can
    pass. ``val`` deliberately contains ``NaN`` slots: the caller filters them,
    and a "keep" branch must keep them.
    """
    pytest.skip(
        "not implemented: mutate_float keeps every value bit-exactly at zero rates "
        "(owner: common/tools.mutate_float)"
    )


def test_mutate_float_perturbation_and_replacement_match_their_documented_laws(common, seed):
    """Invariant: perturbation adds ``normal(0, 1) * mutate_power``; replacement redraws.

    Oracle: the documented two-band rule, checked statistically over a few
    thousand elements (sample standard deviation of ``out - val`` near
    ``mutate_power`` for ``mutate_rate=1``; sample mean/std near
    ``init_mean, init_std`` for ``replace_rate=1``). The exact draw stream is an
    ``evoxc.random`` detail, the *law* is the contract; "bounded" here means the
    result keeps ``val``'s shape and dtype and stays finite for finite input and
    finite parameters.
    """
    pytest.skip(
        "not implemented: mutate_float perturbation/replacement statistics "
        "match the documented laws "
        "(owner: common/tools.mutate_float)"
    )


def test_mutate_float_does_not_write_through_its_input(common, seed):
    """Invariant: the input array is untouched; the result is a fresh tensor.

    Oracle: a copy of ``val`` taken before the call, compared afterwards — graph
    helpers must be pure, because the same gene payload is read by several kernels
    of one generation.
    """
    pytest.skip(
        "not implemented: mutate_float leaves its val argument unmodified "
        "(owner: common/tools.mutate_float)"
    )


@pytest.mark.parametrize(
    "mutate_rate, replace_rate",
    [(-0.1, 0.0), (1.1, 0.0), (0.0, -0.1), (0.0, 1.1), (0.6, 0.5), (0.0, 1.0)],
)
def test_mutate_float_rejects_impossible_rates(common, seed, mutate_rate, replace_rate):
    """Invariant: rates outside ``[0, 1)`` or summing above ``1`` raise ``ConfigError``.

    Oracle: the documented host-side validation (``mutate_rate + replace_rate <= 1``
    with ``replace_rate < 1``). Validated before tracing, so the error is a
    ``TensorNEATError`` and not a silently dead branch inside the graph.
    """
    pytest.skip(
        "not implemented: ConfigError for out-of-range / non-partitioning mutation rates "
        "(owner: common/tools.mutate_float host validation)"
    )


@pytest.mark.parametrize("rate", [0.0, 0.25, 1.0])
def test_mutate_int_emits_only_legal_options(common, seed, rate):
    """Invariant: an int slot is either kept or replaced by an entry of ``options``.

    Oracle: ``np.isin(out, np.append(options, val))`` plus, for ``rate == 0``,
    bit-wise identity and, for ``rate == 1``, membership in ``options`` only. Also
    covers "no rate is a no-op": the replace-only rule has no perturb band.
    """
    pytest.skip(
        "not implemented: mutate_int output is in options, zero rate is the identity "
        "(owner: common/tools.mutate_int)"
    )


def test_mutate_int_rejects_an_empty_option_list(common, seed):
    """Invariant: ``options`` may not be empty — that is a ``ConfigError``.

    Oracle: the documented "an empty option list is a configuration bug"; the
    kernel would otherwise draw ``randint(0, 0)`` and produce garbage.
    """
    pytest.skip(
        "not implemented: ConfigError for an empty mutate_int option list "
        "(owner: common/tools.mutate_int host validation)"
    )


def test_mutation_is_a_function_of_the_key_alone(common, seed):
    """Invariant: the same key reproduces the same mutation; different keys differ.

    Oracle: two runs of the same closure with the same seed are bit-identical
    (the stateless-key contract of ``tensorneat.common``), and a run with a
    different seed is not — a globally seeded RNG would break the first, a
    constant draw the second.
    """
    pytest.skip(
        "not implemented: mutate_float/mutate_int determinism per key and difference across keys "
        "(owner: common/tools.mutate_float, common/tools.mutate_int)"
    )


# --------------------------------------------------------------------------- #
# hash_rows (host only)
# --------------------------------------------------------------------------- #


def test_hash_rows_matches_the_row_equality_oracle(common):
    """Invariant: equal rows hash equal, different rows hash differently in practice.

    Oracle: a numpy row-equality matrix written here (``(a[:, None] == b).all(-1)``),
    compared against hash equality; ``hash_rows`` is host-only bit-viewing +
    ``0x9E3779B9`` folding, so its *values* are asserted only for dtype
    (``uint32``), shape and independence from row order — the equality relation is
    the actual contract, and it is checked on a payload with two identical rows.
    """
    pytest.skip(
        "not implemented: hash_rows preserves row equality, returns uint32 and ignores row order "
        "(owner: common/tools.hash_rows)"
    )


def test_hash_rows_rejects_a_one_dimensional_array(common):
    """Invariant: a 1-D input raises ``ValueError``.

    Oracle: the documented ``Raises`` clause; a 1-D array silently hashed as one
    row would return a scalar-like result and hide the caller's mistake.
    """
    pytest.skip(
        "not implemented: hash_rows ValueError for a non-2-D input "
        "(owner: common/tools.hash_rows)"
    )
