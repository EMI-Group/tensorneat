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
``common`` fixture (see ``./conftest.py``), not a module-scope import, so a
package that fails to import names a skip instead of a collection error.
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


def _run_seeded(common, name, seed, *arrays):
    """Execute a key-taking ``tools`` kernel with its key rebuilt from ``seed``.

    Same bridge as :func:`_run`, but the key is the first graph argument and is
    supplied by :func:`_seeded`: ``_run_seeded(common, "mutate_float", seed, val,
    ...)`` calls ``mutate_float(evoxc.random.key(seed), val, ...)`` inside the
    trace, so the draw is a function of ``seed`` alone.
    """
    return run_defn(_seeded(getattr(common.tools, name), seed), *arrays)


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
    tools = common.tools
    assert tools.I_INF == int(np.iinfo(np.int32).max) == 2**31 - 1
    assert isinstance(tools.I_INF, int)
    # I_INF is the largest value the int32 arrays it lives in can hold
    assert np.int32(tools.I_INF) == tools.I_INF
    assert int(np.asarray([tools.I_INF], dtype=np.int32)[0]) == tools.I_INF
    assert tools.KEY_MAX == 1 << 24 == 16_777_216
    assert isinstance(tools.KEY_MAX, int)
    # every integer below KEY_MAX is exact in float32 (the parity spelling)
    assert float(np.float32(tools.KEY_MAX - 1)) == tools.KEY_MAX - 1
    assert np.float32(tools.KEY_MAX - 1) != np.float32(tools.KEY_MAX)
    # ... and KEY_MAX is the first integer float32 can no longer distinguish
    assert np.float32(tools.KEY_MAX) == np.float32(tools.KEY_MAX + 1)


# --------------------------------------------------------------------------- #
# batch_offsets
# --------------------------------------------------------------------------- #


def test_batch_offsets_matches_the_row_major_oracle(common):
    """Invariant: ``batch_offsets(P, R)`` is ``[[0], [R], [2R], ...]``, shape ``(P, 1)``.

    Oracle: ``np.arange(P)[:, None] * R``; the documented ``(P, 1) int32`` shape
    is asserted too, because broadcasting it against a ``(P, K)`` column block is
    the only thing the helper exists for.
    """
    pop_size, row_len = 5, 7
    out = _run(common, "batch_offsets", pop_size, row_len)
    expected = np.arange(pop_size, dtype=np.int32)[:, None] * np.int32(row_len)
    np.testing.assert_array_equal(out, expected)
    assert out.shape == (pop_size, 1)
    assert out.dtype == np.int32
    # the point of the table: (P, K) columns + offsets are flat 1-D indices
    columns = np.arange(3, dtype=np.int32)[None, :]
    np.testing.assert_array_equal(
        out + columns,
        [[0, 1, 2], [7, 8, 9], [14, 15, 16], [21, 22, 23], [28, 29, 30]],
    )


@pytest.mark.parametrize("pop_size, row_len", [(0, 4), (4, 0), (-1, 4), (4, -1)])
def test_batch_offsets_rejects_a_non_positive_static(common, pop_size, row_len):
    """Invariant: a non-positive ``pop_size``/``row_len`` is a ``ConfigError``.

    Oracle: the documented ``Raises`` clause — validation happens on the host
    before tracing, so the error must be a ``TensorNEATError`` and not an
    ``IndexError`` from ``arange`` or a traced ``evoxc`` failure.
    """
    assert issubclass(common.errors.ConfigError, common.errors.TensorNEATError)
    with pytest.raises(common.errors.ConfigError):
        _run(common, "batch_offsets", pop_size, row_len)


@pytest.mark.parametrize("pop_size, row_len", [(2**16, 2**15), (2**31, 1)])
def test_batch_offsets_rejects_an_int32_overflowing_flat_range(common, pop_size, row_len):
    """Invariant: ``pop_size * row_len >= I_INF`` raises ``ConfigError``.

    Oracle: the documented bound ``P * R < I_INF``; a flat index must stay inside
    the ``int32`` range the helpers use, and silently wrapping would index a
    different row's data instead of failing.
    """
    assert pop_size * row_len >= common.tools.I_INF  # the documented bound is crossed
    with pytest.raises(common.errors.ConfigError):
        _run(common, "batch_offsets", pop_size, row_len)


# --------------------------------------------------------------------------- #
# flat_gather / masked_flat_gather
# --------------------------------------------------------------------------- #


def test_flat_gather_matches_row_wise_numpy_indexing(common):
    """Invariant: ``out[p, k] == x[p, idx[p, k]]`` for a ``(P, K)`` index block.

    Oracle: :func:`_np_flat_gather` (row-wise numpy indexing). A ``(P, K)`` block
    is the shape the schedule's ``in_conn``/``in_src`` gathers use.
    """
    x = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    idx = np.array([[0, 5, 3], [4, 4, 0], [1, 2, 3], [5, 0, 2]], dtype=np.int32)
    out = _run(common, "flat_gather", x, idx)
    np.testing.assert_array_equal(out, _np_flat_gather(x, idx))
    # closed form: row p holds p*6 + c, so the result names the *source row* too
    np.testing.assert_array_equal(
        out, np.arange(x.shape[0], dtype=np.float32)[:, None] * x.shape[1] + idx
    )
    assert out.shape == idx.shape
    assert out.dtype == x.dtype


def test_flat_gather_accepts_a_multi_dimensional_index_block(common):
    """Invariant: a ``(P, I, J)`` index block gathers to ``(P, I, J)``.

    Oracle: :func:`_np_flat_gather` with a broadcast batch axis. The docstring
    promises ``(P, *shape)`` for any static ``*shape``; the ``(P, N, K)``
    incoming-edge tables are exactly such a block.
    """
    x = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    idx = np.array(
        [[[0, 1], [2, 3]], [[4, 3], [2, 1]], [[0, 0], [4, 4]]], dtype=np.int32
    )
    out = _run(common, "flat_gather", x, idx)
    assert out.shape == (3, 2, 2)
    assert out.dtype == x.dtype
    np.testing.assert_array_equal(out, _np_flat_gather(x, idx))
    # closed form: x[p, c] == p*5 + c, still per-row (no cross-row flattening)
    np.testing.assert_array_equal(
        out, np.arange(3, dtype=np.float32)[:, None, None] * 5 + idx
    )


def test_flat_gather_raises_when_an_index_escapes_the_row(common):
    """Invariant: an out-of-range index raises ``evoxc.ShapeError``; it never clamps.

    Oracle: the documented "propagated unchanged" contract, checked against
    ``evoxc.ShapeError`` (what the numpy backend really raises for an
    out-of-range ``gather``). ``I_INF`` is the index that matters here: a caller
    that forgot to mask must fail loudly instead of reading row ``p``'s column 0.
    """
    x = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    for escaped in (common.tools.I_INF, 3, -1, 2**31 - 2):
        idx = np.zeros((4, 2), dtype=np.int32)
        idx[1, 0] = escaped
        with pytest.raises(evoxc.ShapeError):
            _run(common, "flat_gather", x, idx)
    # the same block with an in-range cell is accepted, so the raise is about
    # the escaping index and not about the shape of the block
    accepted = np.zeros((4, 2), dtype=np.int32)
    accepted[1, 0] = 2
    np.testing.assert_array_equal(
        _run(common, "flat_gather", x, accepted), _np_flat_gather(x, accepted)
    )


def test_masked_flat_gather_fills_masked_positions(common):
    """Invariant: ``mask == False`` positions yield ``fill``, masked-in ones the gathered value.

    Oracle: ``np.where(mask, _np_flat_gather(x, idx), fill)`` — the default
    ``fill=NaN`` included, since "absent / not computed" is the forward pass's
    padding value.
    """
    x = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    idx = np.array([[0, 1], [3, 2], [1, 0]], dtype=np.int32)
    mask = np.array([[True, False], [False, False], [True, True]])
    out = _run(common, "masked_flat_gather", x, idx, mask)
    assert out.shape == idx.shape
    assert out.dtype == x.dtype
    np.testing.assert_allclose(
        out, np.where(mask, _np_flat_gather(x, idx), np.float32("nan")), equal_nan=True
    )
    # the masked cells are the default fill, NaN, i.e. "absent / not computed"
    assert np.isnan(out[~mask]).all()
    # the selected cells still read their own row and column
    np.testing.assert_array_equal(
        out[mask], (np.arange(3, dtype=np.float32)[:, None] * 4 + idx)[mask]
    )


def test_masked_flat_gather_treats_i_inf_as_absent_without_an_explicit_mask(common):
    """Invariant: ``idx == I_INF`` yields ``fill`` even when ``mask`` is ``None``.

    Oracle: the docstring's two-part predicate — fill where the position is
    masked out **or** where ``idx == I_INF``. The ``I_INF`` half must be enforced
    by the helper itself, otherwise the ``I_INF``-padded schedule tables would
    raise at the backend instead of producing padding.
    """
    x = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    idx = np.array(
        [[common.tools.I_INF, 1], [2, 3], [common.tools.I_INF, common.tools.I_INF]],
        dtype=np.int32,
    )
    out = _run(common, "masked_flat_gather", x, idx)  # mask=None
    present = idx != common.tools.I_INF
    np.testing.assert_allclose(
        out,
        np.where(
            present,
            np.arange(3, dtype=np.float32)[:, None] * 4 + np.where(present, idx, 0),
            np.float32("nan"),
        ),
        equal_nan=True,
    )
    assert np.isnan(out[~present]).all()
    # a *selected* out-of-range index is still a caller bug and propagates
    escaping = np.array([[0, 4], [1, 1], [1, 1]], dtype=np.int32)  # 4 == R
    with pytest.raises(evoxc.ShapeError):
        _run(common, "masked_flat_gather", x, escaping)
    # ... while masking that same position out turns it into a fill
    keep = np.array([[True, False], [True, True], [True, True]])
    filled = _run(common, "masked_flat_gather", x, escaping, keep)
    np.testing.assert_allclose(
        filled,
        np.where(keep, _np_flat_gather(x, np.where(keep, escaping, 0)), np.float32("nan")),
        equal_nan=True,
    )
    assert np.isnan(filled[0, 1])


@pytest.mark.parametrize("fill", [0.0, 1.0, float("-inf"), float("inf")])
def test_masked_flat_gather_writes_the_aggregation_neutral_element(common, fill):
    """Invariant: ``fill`` is used verbatim and cast to ``x``'s dtype.

    Oracle: ``np.full(shape, fill, dtype=x.dtype)`` at the masked positions. The
    four parameter values are the neutral elements of ``sum``/``product``/
    ``max``/``min`` in ``common/functions/agg.py``, which is why an unmasked
    aggregation can consume the result: the padding must be the *value* the
    reduction ignores, not just a different bit pattern.
    """
    x = np.linspace(-2.0, 2.0, 12, dtype=np.float32).reshape(3, 4)
    idx = np.array([[0, 3], [2, 1], [3, 0]], dtype=np.int32)
    mask = np.array([[True, False], [False, False], [True, False]])
    out = _run(common, "masked_flat_gather", x, idx, mask, fill)
    assert out.dtype == x.dtype
    assert out.shape == mask.shape
    # the padding is exactly the value the reduction ignores, cast to x's dtype
    np.testing.assert_array_equal(
        out,
        np.where(mask, _np_flat_gather(x, idx), np.full(mask.shape, fill, x.dtype)),
    )
    np.testing.assert_array_equal(
        out[~mask], np.full((int((~mask).sum()),), fill, dtype=x.dtype)
    )


# --------------------------------------------------------------------------- #
# flat_scatter / masked_flat_scatter
# --------------------------------------------------------------------------- #


def test_flat_scatter_matches_a_python_write_loop(common):
    """Invariant: ``out[p, idx[p, k]] == updates[p, k]``, every other cell unchanged.

    Oracle: :func:`_np_flat_scatter`, a plain Python write loop over the same
    ``(P, K)`` block.
    """
    x = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    idx = np.array([[0, 4], [2, 2], [3, 1]], dtype=np.int32)
    updates = np.array([[100.0, 400.0], [200.0, 210.0], [310.0, -1.0]], dtype=np.float32)
    out = _run(common, "flat_scatter", x, idx, updates)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    np.testing.assert_array_equal(out, _np_flat_scatter(x, idx, updates))
    written = np.zeros(x.shape, dtype=bool)
    expected = x.copy()
    for row in range(idx.shape[0]):
        written[row, idx[row]] = True
        for column in range(idx.shape[1]):
            expected[row, idx[row, column]] = updates[row, column]
    np.testing.assert_array_equal(out[written], expected[written])
    np.testing.assert_array_equal(out[~written], x[~written])


def test_flat_scatter_resolves_duplicate_indices_last_write_wins(common):
    """Invariant: repeated flat indices resolve to the last update in flat traversal order.

    Oracle: the documented "set-only" note — ``evoxc.scatter`` has no add/max
    mode, so the duplicate must overwrite rather than accumulate; the Python loop
    oracle writes in the same order and therefore predicts the same winner.
    """
    x = np.zeros((2, 4), dtype=np.float32)
    idx = np.array([[1, 1, 3], [2, 2, 2]], dtype=np.int32)
    updates = np.array([[5.0, 7.0, 9.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    out = _run(common, "flat_scatter", x, idx, updates)
    np.testing.assert_array_equal(out, _np_flat_scatter(x, idx, updates))
    # set-only: the last write per row wins and nothing accumulates
    np.testing.assert_array_equal(
        out, [[0.0, 7.0, 0.0, 9.0], [0.0, 0.0, 3.0, 0.0]]
    )
    assert out[0, 1] == 7.0
    assert out[1, 2] == 3.0


def test_masked_flat_scatter_routes_dropped_writes_to_the_dump_column(common):
    """Invariant: masked writes land in column ``R`` and leave columns ``0..R-1`` alone.

    Oracle: ``_np_flat_scatter`` restricted to the masked-in writes, compared
    against ``out[:, :R]``; the dump column is the *only* place a dropped write
    may appear. Routing it to index ``0`` would corrupt a real element, which is
    the pre-rewrite bug this helper exists to prevent.
    """
    payload = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)  # R == 4
    x = np.concatenate([payload, np.zeros((3, 1), dtype=np.float32)], axis=1)
    idx = np.array([[0, 3], [2, 1], [3, 1]], dtype=np.int32)
    updates = np.array([[100.0, 900.0], [200.0, 210.0], [880.0, 310.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, True], [False, True]])
    out = _run(common, "masked_flat_scatter", x, idx, updates, mask)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    expected = payload.copy()
    for row in range(idx.shape[0]):
        for column in range(idx.shape[1]):
            if mask[row, column]:
                expected[row, idx[row, column]] = updates[row, column]
    np.testing.assert_array_equal(out[:, :4], expected)
    # the two dropped writes show up in the dump column only, never in 0..R-1
    np.testing.assert_array_equal(out[[0, 2], 4], updates[[0, 2], [1, 0]])
    assert 900.0 not in out[:, :4] and 880.0 not in out[:, :4]
    # row 1 dropped nothing, so its dump column keeps its incoming content
    assert out[1, 4] == x[1, 4]


def test_masked_flat_scatter_accepts_i_inf_index_when_the_position_is_masked(common):
    """Invariant: a masked position may hold any index, ``I_INF`` included, without raising.

    Oracle: the documented contract — masked indices are replaced by the dump
    index ``R`` *before* the backend sees them, so an ``I_INF``-padded index
    block is safe exactly where the schedule pads it.
    """
    payload = np.zeros((2, 3), dtype=np.float32)  # R == 3, dump column 3
    x = np.concatenate([payload, np.full((2, 1), -1.0, dtype=np.float32)], axis=1)
    idx = np.array([[0, common.tools.I_INF], [2**31 - 2, 1]], dtype=np.int32)
    updates = np.array([[5.0, 99.0], [88.0, 6.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, True]])
    out = _run(common, "masked_flat_scatter", x, idx, updates, mask)
    np.testing.assert_array_equal(out[:, :3], [[5.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    # both impossible indices were redirected to the dump column instead
    np.testing.assert_array_equal(out[:, 3], [99.0, 88.0])


# --------------------------------------------------------------------------- #
# fetch_first / fetch_random / rank_elements / argmin_with_mask
# --------------------------------------------------------------------------- #


def test_fetch_first_matches_the_python_oracle(common):
    """Invariant: the column of the first ``True`` entry of each mask row.

    Oracle: :func:`_np_fetch_first` (``argmax`` of the boolean row). ``argmax`` is
    what the kernel documentation prescribes, so ties and all-``False`` rows are
    the interesting cases.
    """
    mask = np.array(
        [
            [False, True, True, False],
            [False, False, False, False],
            [True, False, True, True],
        ],
        dtype=bool,
    )
    out = _run(common, "fetch_first", mask)
    assert out.shape == (mask.shape[0],)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, _np_fetch_first(mask, common.tools.I_INF))
    # the *first* True wins even where later columns are True as well
    assert out[0] == 1
    assert out[2] == 0
    # an all-False row is the sentinel, not column 0
    assert out[1] == common.tools.I_INF


@pytest.mark.parametrize("default", [None, 0, 7])
def test_fetch_first_returns_the_documented_default_for_an_empty_mask(common, default):
    """Invariant: a row with no ``True`` entry yields ``default`` (``I_INF`` when unset).

    Oracle: :func:`_np_fetch_first` with the same default. The parameter set covers
    the sentinel and two real columns (``0`` and ``7``), because a caller must be
    able to tell "no candidate" from "candidate 0".
    """
    mask = np.array(
        [
            [False, False, False],
            [True, False, False],
            [False, False, False],
        ],
        dtype=bool,
    )
    sentinel = common.tools.I_INF if default is None else default
    out = _run(common, "fetch_first", mask, default)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, _np_fetch_first(mask, sentinel))
    assert out[0] == sentinel
    assert out[2] == sentinel
    # a real hit still reports its own column: the default is not blanket-applied
    assert out[1] == 0
    if default is not None:
        assert sentinel in out  # "no candidate" and "candidate 0" stay distinguishable


def test_fetch_random_only_returns_true_positions(common, seed):
    """Invariant: every returned index is a ``True`` column of its own mask row.

    Oracle: re-indexing the mask with the result and asserting it is ``True`` for
    every seeded key; this is the property the ``argmax``-of-uniforms recipe must
    preserve. Keys come from the ``seed`` fixture and are rebuilt inside the
    closure (:func:`_seeded`), so the draw is a function of the key alone.
    """
    mask = np.array(
        [
            [True, True, True, False],
            [False, True, False, True],
            [False, False, False, False],
            [True, False, True, True],
            [False, True, True, True],
        ],
        dtype=bool,
    )
    hit = mask.any(axis=-1)
    picked = {}
    for offset in range(16):
        out = _run_seeded(common, "fetch_random", seed + offset, mask)
        assert out.shape == (mask.shape[0],)
        assert out.dtype == np.int32
        # "no candidate" is reported as the sentinel, and only there
        np.testing.assert_array_equal(out != common.tools.I_INF, hit)
        rows = np.flatnonzero(hit)
        assert mask[rows, out[rows]].all()  # every pick is a True column of its row
        for row in rows.tolist():
            picked.setdefault(row, set()).add(int(out[row]))
    # the pick follows the key: a constant recipe ("always the first True
    # column") would map row 0's three candidates onto a single value
    assert len(picked[0]) > 1


def test_fetch_random_is_uniform_over_the_row(common, seed):
    """Invariant: over many keys, each ``True`` column of a row is drawn with equal frequency.

    Oracle: the documented sampling argument (the maximiser of i.i.d. draws
    restricted to a set is uniform over that set), checked as an empirical
    frequency bound on a 4-column row over a few hundred keys — loose enough to
    be deterministic, tight enough to catch a biased recipe (e.g. always ``argmax``
    of the mask, or a cumsum over the wrong axis).
    """
    pattern = np.array([True, False, True, True])
    rows = 512
    mask = np.tile(pattern, (rows, 1))
    counts = np.zeros(pattern.size, dtype=np.int64)
    for offset in range(2):
        out = _run_seeded(common, "fetch_random", seed + offset, mask)
        counts += np.bincount(out, minlength=pattern.size)
    total = int(counts.sum())
    assert total == rows * 2
    # a False column is never drawn
    assert int(counts[~pattern].sum()) == 0
    # uniform over the three True columns: total/3 each, within ~4.5 sigma of a
    # binomial draw (sigma computed from the same total, so the band is a law)
    expected = total / float(pattern.sum())
    sigma = float(np.sqrt(total * (1.0 / 3.0) * (2.0 / 3.0)))
    assert np.all(np.abs(counts[pattern] - expected) <= 4.5 * sigma)
    assert np.all(counts[pattern] > 0)


def test_fetch_random_returns_the_default_for_an_empty_mask(common, seed):
    """Invariant: a row with no ``True`` entry yields ``default``, never column 0.

    Oracle: :func:`_np_fetch_first` with ``default``; the ``I_INF`` sentinel is
    what lets the caller detect "no candidate" (the legacy helper returned a real
    slot instead, which is the bug this port fixes).
    """
    mask = np.array(
        [
            [False, False, False, False],
            [True, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    out = _run_seeded(common, "fetch_random", seed, mask)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, _np_fetch_first(mask, common.tools.I_INF))
    assert out[0] == common.tools.I_INF
    assert out[2] == common.tools.I_INF
    assert out[1] == 0  # the only candidate
    # an explicit default replaces the sentinel, and only on the empty rows
    custom = _run_seeded(common, "fetch_random", seed, mask, 3)
    np.testing.assert_array_equal(custom, _np_fetch_first(mask, 3))


def test_rank_elements_matches_the_stable_descending_oracle(common):
    """Invariant: rank ``0`` is the **largest** element (the legacy orientation).

    Oracle: :func:`_np_rank_descending`, i.e. the inverse permutation of a stable
    descending ``argsort``; the default ``reverse=False`` must reproduce it
    exactly, since a flipped orientation silently reorders every parent choice.
    """
    x = np.array([[3.5, -1.0, 2.0, 0.5], [9.0, 8.0, 7.0, 6.0]], dtype=np.float32)
    out = _run(common, "rank_elements", x)
    assert out.shape == x.shape
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, _np_rank_descending(x))
    # hand-enumerated: rank 0 is the largest element of each row
    np.testing.assert_array_equal(out, [[0, 3, 1, 2], [0, 1, 2, 3]])
    for row in out:
        np.testing.assert_array_equal(
            np.sort(row), np.arange(x.shape[1], dtype=np.int32)
        )


def test_rank_elements_reverse_ranks_small_to_large(common):
    """Invariant: ``reverse=True`` returns the small-to-large ranks, i.e. the complement.

    Oracle: :func:`_np_rank_descending` of ``-x`` (equivalently ``L - 1 - rank``
    for a tie-free row): the two orientations must be permutations of each other
    with mirrored positions.
    """
    x = np.array([[3.5, -1.0, 2.0, 0.5], [9.0, 8.0, 7.0, 6.0]], dtype=np.float32)
    descending = _run(common, "rank_elements", x)
    ascending = _run(common, "rank_elements", x, True)
    np.testing.assert_array_equal(ascending, _np_rank_descending(-x))
    # hand-enumerated small-to-large ranks, mirroring the descending case
    np.testing.assert_array_equal(ascending, [[3, 0, 2, 1], [3, 2, 1, 0]])
    # tie-free rows: the two orientations are mirror images of each other
    np.testing.assert_array_equal(ascending, x.shape[1] - 1 - descending)


def test_rank_elements_is_stable_on_ties(common):
    """Invariant: equal values keep their column order, so the rank is deterministic.

    Oracle: :func:`_np_rank_descending` on a deliberately tie-heavy row; stability
    is mandated because the schedule's in-group ranks must not depend on the
    backend's ``argsort`` tie-breaking (a rebuild must not reorder edges).
    """
    x = np.array(
        [[2.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], [5.0, 3.0, 5.0, 3.0]],
        dtype=np.float32,
    )
    out = _run(common, "rank_elements", x)
    np.testing.assert_array_equal(out, _np_rank_descending(x))
    # hand-enumerated: equal values keep their column order, so the earlier
    # column gets the smaller rank (a non-stable sort would swap 0/2 and 1/3)
    np.testing.assert_array_equal(out, [[1, 3, 2, 0], [0, 1, 2, 3], [0, 2, 1, 3]])


def test_argmin_with_mask_matches_the_masked_numpy_oracle(common):
    """Invariant: the index of the minimal ``True`` entry per row.

    Oracle: :func:`_np_argmin_with_mask` (masked-out entries replaced by ``+inf``).
    A row whose masked-in values are all larger than a masked-out one is included,
    so a missing mask shows up as the wrong column instead of a passing test.
    """
    arr = np.array(
        [[5.0, 1.0, 0.5, 3.0], [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    mask = np.array(
        [[True, False, True, False], [False, False, False, False], [False, True, True, False]]
    )
    out = _run(common, "argmin_with_mask", arr, mask)
    assert out.shape == (arr.shape[0],)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, _np_argmin_with_mask(arr, mask, common.tools.I_INF))
    # row 0: the selectable minimum is 0.5 at column 2
    assert out[0] == 2
    # row 2: column 3 (1.0) is smaller but masked out, so column 2 (2.0) wins —
    # an ignored mask would answer 3 here
    assert out[2] == 2


def test_argmin_with_mask_returns_i_inf_for_an_empty_mask(common):
    """Invariant: a row without a selectable entry yields ``I_INF``, not slot ``0``.

    Oracle: the documented sentinel contract; ``0`` is a real individual, so the
    "empty" answer must be distinguishable — the divergence from the legacy
    helper, which returned ``0``.
    """
    arr = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, False]])
    out = _run(common, "argmin_with_mask", arr, mask)
    # the legacy helper answered slot 0 here; 0 is a real individual, so the
    # "nothing selectable" answer must be distinguishable from it
    assert out[1] == common.tools.I_INF
    assert out[0] == 0
    np.testing.assert_array_equal(out, _np_argmin_with_mask(arr, mask, common.tools.I_INF))


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
    val = np.array([0.0, 1.0, np.nan, -2.5, 3.25], dtype=np.float32)
    before = val.copy()
    out = _run_seeded(common, "mutate_float", seed, val, 0.0, 1.0, 0.5, 0.0, 0.0)
    assert out.shape == val.shape
    assert out.dtype == val.dtype
    np.testing.assert_array_equal(out, before)
    # bit-exact, not merely "close": no perturbation is allowed to leak through
    assert out.tobytes() == before.tobytes()
    assert np.isnan(out[2])  # a NaN slot is kept, not filtered here


def test_mutate_float_perturbation_and_replacement_match_their_documented_laws(common, seed):
    """Invariant: perturbation adds ``normal(0, 1) * mutate_power``; replacement redraws.

    Oracle: the documented two-band rule, checked statistically over a few
    thousand elements (sample standard deviation of ``out - val`` near
    ``mutate_power`` for ``mutate_rate=1``; sample mean/std near
    ``init_mean, init_std`` among the replaced slots for ``replace_rate=0.5``,
    which is the largest rate the contract allows — a replacement band needs a
    non-empty keep band). The exact draw stream is an
    ``evoxc.random`` detail, the *law* is the contract; "bounded" here means the
    result keeps ``val``'s shape and dtype and stays finite for finite input and
    finite parameters.
    """
    zeros = np.zeros(4000, dtype=np.float32)

    # perturb band only: out - val ~ N(0, mutate_power)
    perturbed = _run_seeded(common, "mutate_float", seed, zeros, 0.0, 1.0, 0.5, 1.0, 0.0)
    assert perturbed.shape == zeros.shape
    assert perturbed.dtype == zeros.dtype
    assert np.isfinite(perturbed).all()  # mutate_rate == 1 is "always perturb"
    delta = perturbed - zeros
    assert abs(float(delta.mean())) <= 0.03  # ~5 sigma of 0.5/sqrt(4000) = 0.008
    assert abs(float(delta.std()) - 0.5) <= 0.03  # a dropped mutate_power would read 1.0

    # replace band only: replaced slots are redrawn from N(init_mean, init_std),
    # the rest keep the zero they started from, so the selection is exact
    replaced = _run_seeded(common, "mutate_float", seed, zeros, 2.0, 3.0, 1.0, 0.0, 0.5)
    assert replaced.shape == zeros.shape
    assert replaced.dtype == zeros.dtype
    drawn = replaced[replaced != 0.0]
    fraction = drawn.size / replaced.size
    assert abs(fraction - 0.5) <= 0.05  # ~3 sigma of sqrt(0.25/4000) = 0.008
    assert abs(float(drawn.mean()) - 2.0) <= 0.3  # 3/sqrt(2000) = 0.067
    assert abs(float(drawn.std()) - 3.0) <= 0.3  # a dropped init_std would read 0 or 1


def test_mutate_float_does_not_write_through_its_input(common, seed):
    """Invariant: the input array is untouched; the result is a fresh tensor.

    Oracle: a copy of ``val`` taken before the call, compared afterwards — graph
    helpers must be pure, because the same gene payload is read by several kernels
    of one generation.
    """
    val = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    before = val.copy()
    out = _run_seeded(common, "mutate_float", seed, val, 0.0, 1.0, 10.0, 1.0, 0.0)
    np.testing.assert_array_equal(val, before)
    assert val.tobytes() == before.tobytes()
    assert out is not val
    # the perturbation really happened (power 10 over 4 slots)
    assert not np.array_equal(out, before)
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "mutate_rate, replace_rate",
    [(-0.1, 0.0), (1.1, 0.0), (0.0, -0.1), (0.0, 1.1), (0.6, 0.5), (0.0, 1.0)],
)
def test_mutate_float_rejects_impossible_rates(common, seed, mutate_rate, replace_rate):
    """Invariant: rates outside ``[0, 1]``/``[0, 1)`` or summing above ``1`` raise ``ConfigError``.

    Oracle: the documented host-side validation (``mutate_rate + replace_rate <= 1``
    with ``replace_rate < 1``). Validated before tracing, so the error is a
    ``TensorNEATError`` and not a silently dead branch inside the graph.
    """
    with pytest.raises(common.errors.ConfigError):
        _run_seeded(
            common, "mutate_float", seed, np.zeros(3, dtype=np.float32), 0.0, 1.0, 0.1,
            mutate_rate, replace_rate,
        )


@pytest.mark.parametrize("rate", [0.0, 0.25, 1.0])
def test_mutate_int_emits_only_legal_options(common, seed, rate):
    """Invariant: an int slot is either kept or replaced by an entry of ``options``.

    Oracle: ``np.isin(out, np.union1d(options, val))`` plus, for ``rate == 0``,
    bit-wise identity and, for ``rate == 1``, membership in ``options`` only. Also
    covers "no rate is a no-op": the replace-only rule has no perturb band.
    """
    options = np.array([0, 2, 3], dtype=np.int32)
    val = np.array([0, 1, 2, 3] * 100, dtype=np.int32)  # 1 is not an option
    out = _run_seeded(common, "mutate_int", seed, val, options, rate)
    assert out.shape == val.shape
    assert out.dtype == val.dtype
    # keep-or-replace: every slot is either its old value or a legal option
    assert np.isin(out, np.union1d(options, val)).all()
    if rate == 0.0:
        np.testing.assert_array_equal(out, val)  # the no-op rate
    elif rate == 1.0:
        assert np.isin(out, options).all()  # replace-only: no keep band left
        assert not np.array_equal(out, val)
    else:
        # both bands fire on 400 slots; a dead branch would lose one of them
        assert (out == val).any()
        assert (out != val).any()


def test_mutate_int_rejects_an_empty_option_list(common, seed):
    """Invariant: ``options`` may not be empty — that is a ``ConfigError``.

    Oracle: the documented "an empty option list is a configuration bug"; the
    kernel would otherwise draw ``randint(0, 0)`` and produce garbage.
    """
    val = np.array([0, 1, 2, 3], dtype=np.int32)
    assert issubclass(common.errors.ConfigError, common.errors.TensorNEATError)
    with pytest.raises(common.errors.ConfigError):
        _run_seeded(common, "mutate_int", seed, val, np.zeros((0,), dtype=np.int32), 0.5)


def test_mutation_is_a_function_of_the_key_alone(common, seed):
    """Invariant: the same key reproduces the same mutation; different keys differ.

    Oracle: two runs of the same closure with the same seed are bit-identical
    (the stateless-key contract of ``tensorneat.common``), and a run with a
    different seed is not — a globally seeded RNG would break the first, a
    constant draw the second.
    """
    val = np.zeros(64, dtype=np.float32)
    params = (0.0, 1.0, 0.5, 1.0, 0.0)
    first = _run_seeded(common, "mutate_float", seed, val, *params)
    again = _run_seeded(common, "mutate_float", seed, val, *params)
    other = _run_seeded(common, "mutate_float", seed + 1, val, *params)
    np.testing.assert_array_equal(first, again)
    assert first.tobytes() == again.tobytes()
    assert not np.array_equal(first, other)

    options = np.arange(5, dtype=np.int32)
    slots = np.zeros(64, dtype=np.int32)
    int_first = _run_seeded(common, "mutate_int", seed, slots, options, 0.5)
    int_again = _run_seeded(common, "mutate_int", seed, slots, options, 0.5)
    int_other = _run_seeded(common, "mutate_int", seed + 1, slots, options, 0.5)
    np.testing.assert_array_equal(int_first, int_again)
    assert not np.array_equal(int_first, int_other)


# --------------------------------------------------------------------------- #
# hash_rows (host only)
# --------------------------------------------------------------------------- #


def test_hash_rows_matches_the_row_equality_oracle(common):
    """Invariant: equal rows hash equal, different rows hash differently in practice.

    Oracle: a numpy row-equality matrix written here (``(a[:, None] == b).all(-1)``),
    compared against hash equality; ``hash_rows`` is host-only bit-viewing +
    ``0x9E3779B9`` folding, so its *values* are asserted only for dtype
    (``uint32``), shape and independence from row order — the equality relation is
    the actual contract, and it is checked on a payload with repeated rows.
    """
    rows = np.array(
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [1, 2, 3], [-7, 0, 12]], dtype=np.int32
    )
    hashes = common.tools.hash_rows(rows)
    assert hashes.shape == (rows.shape[0],)
    assert hashes.dtype == np.uint32
    equal_rows = (rows[:, None, :] == rows[None, :, :]).all(axis=-1)
    equal_hashes = hashes[:, None] == hashes[None, :]
    # the contract: equal rows hash equal, unequal rows do not collide here
    assert equal_hashes[equal_rows].all()
    assert not equal_hashes[~equal_rows].any()
    # row order is irrelevant, only the row content matters
    order = np.array([4, 2, 0, 3, 1])
    np.testing.assert_array_equal(common.tools.hash_rows(rows[order]), hashes[order])
    # column order *does* matter: a permuted row is a different row
    assert common.tools.hash_rows(np.array([[1, 2, 3]], dtype=np.int32))[0] != (
        common.tools.hash_rows(np.array([[3, 2, 1]], dtype=np.int32))[0]
    )
    # float attribute rows (the parity spelling, NaN padding included)
    attrs = np.array([[1.0, np.nan], [1.0, np.nan], [2.5, 0.0]], dtype=np.float32)
    attr_hashes = common.tools.hash_rows(attrs)
    assert attr_hashes.dtype == np.uint32
    assert attr_hashes[0] == attr_hashes[1]
    assert attr_hashes[0] != attr_hashes[2]


def test_hash_rows_rejects_a_one_dimensional_array(common):
    """Invariant: a 1-D input raises ``ValueError``.

    Oracle: the documented ``Raises`` clause; a 1-D array silently hashed as one
    row would return a scalar-like result and hide the caller's mistake.
    """
    with pytest.raises(ValueError):
        common.tools.hash_rows(np.array([1, 2, 3], dtype=np.int32))
    with pytest.raises(ValueError):
        common.tools.hash_rows(np.zeros((2, 2, 2), dtype=np.int32))
    # a 2-D array is what the helper promises to accept, and it does
    assert common.tools.hash_rows(np.zeros((2, 2), dtype=np.int32)).shape == (2,)
