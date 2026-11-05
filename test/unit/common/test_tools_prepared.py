"""The *prepared* half of the ``tools`` flat-index suites — ``trusted=True``.

This module is the second of a two-module pair covering
``tensorneat.common.tools``: :mod:`unit.common.test_tools` keeps the one-shot
helpers (``flat_gather``/``masked_flat_gather``/``flat_scatter``/
``masked_flat_scatter``) and the rest of the module, while this half owns the
pair a loop-invariant caller actually uses: :func:`~tensorneat.common.tools.
prepare_flat_index` builds the ``(P, n)`` flat-index block once and
:func:`~tensorneat.common.tools.flat_gather_prepared` /
:func:`~tensorneat.common.tools.flat_scatter_prepared` spend it, plus the
``trusted`` flag that decides whether the out-of-range redirect is emitted.
Both were previously reached only *through* the one-shot wrappers, so
``trusted=True`` — the switch that trades the "backend raises" net for op count
— was passed nowhere in the suite. Splitting the pair keeps each file under the
repo's ~1000-line baseline without dropping an assertion.

What this module proves
-----------------------
* ``trusted=True`` and ``trusted=False`` return the **same** block for an
  in-range index block, for a 2-D and a 3-D ``idx``, and for a ``(P, R)`` shape
  pair instead of a payload — the redirect is dead code there, so dropping it
  changes nothing.
* They **differ exactly** where an index escapes ``[0, R)``: untrusted rewrites
  the cell to ``P * R`` (one past the flattened payload, so the backend raises),
  trusted leaves the raw column, and ``p * R + col`` then names a **different
  row's** cell for ``R <= col < 2R`` and wraps to the last element for a
  negative column. Both halves are asserted as arithmetic, and the silent
  misread is executed, not just described.
* A prepared block built with ``trusted=True`` gathers and scatters exactly what
  :func:`~tensorneat.common.tools.flat_gather` /
  :func:`~tensorneat.common.tools.flat_scatter` produce, against a plain-numpy
  oracle, including ``last write wins`` for duplicate columns.
* ``present``/``fallback`` are applied **before** the index is formed, so a
  masked-out sentinel never reaches the backend under either spelling, while an
  *unmasked* sentinel under ``trusted=True`` is a caller bug whose raw
  arithmetic is asserted (`I_INF` propagates unchanged; ``R + 1`` lands inside
  the payload and reads a neighbour).
* :func:`~tensorneat.common.tools.batch_offsets` is the ``[[0], [R], ...]`` half
  of the same contract, including its ``ConfigError`` message.
* The prepared helpers propagate the backend's ``evoxc.ShapeError`` for an
  escaping block — they never clamp and never re-wrap.

Oracles
-------
Plain numpy, written from the helpers' docstrings and never from their bodies:

* :func:`_np_prepare_flat_index` — ``row * R + col`` with the two safety rules
  of the docstring applied in the documented order (mask/fill, then redirect),
  reshaped to ``(P, n)``;
* ``.test_tools._np_flat_gather`` / ``.test_tools._np_flat_scatter`` — the
  row-wise indexing expression and the Python write loop the one-shot wrappers
  are already checked against (imported, so the pair shares one oracle);
* literals for ``batch_offsets``, and the one-shot wrapper itself as the second
  witness for every prepared call.

Gotchas this module encodes
---------------------------
* The redirect writes ``P * R``, which is *one past* the flattened payload —
  ``assert block[0, 0] == P * R`` only makes sense against a gather that then
  raises, so the escaping cases pair the arithmetic with the raise.
* ``trusted=True`` does not suppress the ``I_INF`` handling of
  ``masked_flat_gather``: that helper computes ``present`` first, so a masked
  sentinel is replaced by ``fill`` even with the redirect gone. Only an
  unmasked sentinel escapes it.
* ``masked_flat_scatter`` routes dropped writes to the trailing dump column
  ``R`` of a ``(P, R + 1)`` destination — that redirect is not the one
  ``trusted`` removes, so it must survive ``trusted=True``.
* ``evoxc`` rejects a static ``NaN`` at run time ("graph was specialized on
  nan"), so the default ``fill`` is exercised through a closure that bakes it
  into the trace (:func:`_run_bound`); every other static is positional.
* ``I_INF`` never enters a raw index sum outside row ``0``: ``p * R + I_INF``
  would overflow ``int32``, and a wrapped value would make the assertion a
  statement about overflow rather than about the redirect. Where a sentinel sits
  in a later row it is masked out first, so it never reaches the arithmetic.

Backend is ``"numpy"`` through :func:`support.runner.run_defn`; no optional
dependency, no ``jax``/``torch`` import, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

import evoxc

from support.runner import run_defn

# The two one-shot oracles are shared with the ``flat_gather``/``flat_scatter``
# suites rather than copied: one row-wise indexing expression and one write loop
# is what both halves of the pair are measured against.
from .test_tools import _np_flat_gather, _np_flat_scatter

#: ``(P, R)`` of every payload here: small enough to write the flat array out.
POP_SIZE, ROW_LEN = 4, 6


# --------------------------------------------------------------------------- #
# oracles: plain numpy, from the docstrings of tools.py, never from a kernel
# --------------------------------------------------------------------------- #
def _np_prepare_flat_index(
    pop_size, row_len, idx, present=None, fallback=0, trusted=False
):
    """Oracle for ``prepare_flat_index``: ``row * R + col`` plus the two safety rules.

    The documented order is asserted, not assumed: the ``present``/``fallback``
    replacement happens *before* the index is formed, and the out-of-range
    redirect (only when ``trusted`` is false) then rewrites every remaining cell
    outside ``[0, R)`` to ``P * R``. The result is reshaped to ``(P, n)`` with
    ``n`` the product of ``idx.shape[1:]``.
    """
    columns = np.asarray(idx)
    if present is not None:
        columns = np.where(present, columns, np.int32(fallback))
    if not trusted:
        escaped = (columns < 0) | (columns >= row_len)
        columns = np.where(escaped, np.int32(pop_size * row_len), columns)
    flat = np.arange(pop_size)[:, None] * row_len + columns.reshape(pop_size, -1)
    return flat.astype(np.int32)


def _np_masked_scatter(x, idx, updates, mask):
    """Oracle for ``masked_flat_scatter``: a write loop whose dropped writes go to ``R``.

    ``x`` is ``(P, R + 1)``: column ``R`` is the trailing dump slot, so a masked
    position is routed there instead of into a real element or the backend.
    """
    out = np.array(x, copy=True)
    dump_column = x.shape[1] - 1
    for p in range(x.shape[0]):
        for k in range(idx.shape[1]):
            out[p, idx[p, k] if mask[p, k] else dump_column] = updates[p, k]
    return out


def _payload(pop_size=POP_SIZE, row_len=ROW_LEN):
    """A ``(P, R)`` ``float32`` payload whose every cell equals its flat index."""
    return np.arange(pop_size * row_len, dtype=np.float32).reshape(pop_size, row_len)


def _in_range_block(tail, pop_size=POP_SIZE, row_len=ROW_LEN):
    """A ``(P,) + tail`` column block with every entry inside ``[0, R)``."""
    size = pop_size * int(np.prod(tail))
    columns = (np.arange(size) * 2 + 1) % row_len
    return columns.reshape((pop_size,) + tail).astype(np.int32)


# --------------------------------------------------------------------------- #
# the bridge: one kernel, one graph (the house spelling of test_tools)
# --------------------------------------------------------------------------- #
def _run(common, name, *args):
    """Execute one ``tools`` kernel on the numpy backend with positional args.

    ``support.runner.run_defn`` is the only sanctioned build/run bridge, and
    ``name`` is the kernel's own name so a test body reads
    ``_run(common, "flat_gather_prepared", x, block, (P, K))``.
    """
    return run_defn(getattr(common.tools, name), *args)


def _run_bound(common, name, *arrays, **statics):
    """Execute a kernel with its static keywords bound inside the traced closure.

    ``run_defn`` already forwards positional statics, but a static ``float``
    ``NaN`` never compares equal to itself and ``evoxc`` rejects the run with a
    ``TraceError`` ("graph was specialized on nan"), so the *default* ``fill`` of
    ``masked_flat_gather`` can only be reached by baking it into the trace.
    """
    kernel = getattr(common.tools, name)

    def bound(*values):
        return kernel(*values, **statics)

    return run_defn(bound, *arrays)


# --------------------------------------------------------------------------- #
# prepare_flat_index: the redirect is dead for an in-range block
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tail", [(3,), (2, 2)])
def test_prepare_flat_index_trusted_agrees_with_untrusted_in_range(common, tail):
    """Invariant: for an in-range block, ``trusted`` changes nothing at all.

    Oracle: :func:`_np_prepare_flat_index` with the redirect dropped (``trusted=
    True``) and with it kept (``False``), i.e. ``row * R + col`` reshaped to
    ``(P, n)``. ``idx`` may be any static ``(P, *shape)``, so a ``(P, 3)`` block
    and a ``(P, 2, 2)`` block are both covered.
    """
    x = _payload()
    idx = _in_range_block(tail)
    assert ((idx >= 0) & (idx < ROW_LEN)).all()  # the premise: nothing escapes

    trusted = _run(common, "prepare_flat_index", x, idx, None, 0, True)
    untrusted = _run(common, "prepare_flat_index", x, idx, None, 0, False)

    expected = _np_prepare_flat_index(POP_SIZE, ROW_LEN, idx, trusted=True)
    assert expected.shape == (POP_SIZE, int(np.prod(tail)))
    np.testing.assert_array_equal(trusted, expected)
    np.testing.assert_array_equal(untrusted, expected)
    # the flat index of (p, ...) is p * R + col, spelled out per cell
    np.testing.assert_array_equal(
        trusted,
        np.arange(POP_SIZE, dtype=np.int32)[:, None] * ROW_LEN + idx.reshape(POP_SIZE, -1),
    )
    assert trusted.shape == (POP_SIZE, int(np.prod(tail)))
    assert trusted.dtype == np.int32
    assert untrusted.dtype == np.int32


def test_prepare_flat_index_accepts_a_shape_pair_instead_of_a_payload(common):
    """Invariant: ``(P, R)`` as a shape pair builds the block of the same payload.

    Oracle: the block built from the ``(P, R)`` **tensor** (already compared with
    ``row * R + col`` above). The docstring's "only the shape is needed" clause
    is what lets a loop-invariant index be hoisted out of a ``while_loop`` that
    carries a differently-shaped tensor of the same ``(P, R)``.
    """
    idx = _in_range_block((3,))
    from_tensor = _run(common, "prepare_flat_index", _payload(), idx, None, 0, True)
    from_shape = _run(common, "prepare_flat_index", (POP_SIZE, ROW_LEN), idx, None, 0, True)

    np.testing.assert_array_equal(from_shape, from_tensor)
    np.testing.assert_array_equal(
        from_shape, _np_prepare_flat_index(POP_SIZE, ROW_LEN, idx, trusted=True)
    )
    assert from_shape.shape == (POP_SIZE, 3)


# --------------------------------------------------------------------------- #
# prepare_flat_index: the two spellings diverge exactly at the row bound
# --------------------------------------------------------------------------- #
def test_prepare_flat_index_trusted_passes_an_escaping_column_through_raw(common):
    """Invariant: ``trusted`` keeps the raw column; untrusted rewrites it to ``P * R``.

    Oracle: ``row * R + col`` (trusted) and its documented redirect value
    ``P * R`` (untrusted). The escaping column ``R + 1`` is chosen because the
    raw arithmetic stays *inside* the flattened payload — it names the
    neighbouring row's cell — which is exactly the silent misread the redirect
    exists to prevent.
    """
    x = _payload()
    idx = _in_range_block((3,))
    idx[0, 0] = ROW_LEN + 1  # 0 <= R < col < 2R: the neighbouring-row case
    escaped = (idx < 0) | (idx >= ROW_LEN)
    assert escaped[0, 0] and escaped.sum() == 1

    trusted = _run(common, "prepare_flat_index", x, idx, None, 0, True)
    untrusted = _run(common, "prepare_flat_index", x, idx, None, 0, False)

    raw = _np_prepare_flat_index(POP_SIZE, ROW_LEN, idx, trusted=True)
    np.testing.assert_array_equal(trusted, raw)
    np.testing.assert_array_equal(untrusted, _np_prepare_flat_index(POP_SIZE, ROW_LEN, idx))

    # the divergence is exactly the escaping cell, and its raw value is legal
    np.testing.assert_array_equal(trusted[~escaped], untrusted[~escaped])
    assert int(trusted[0, 0]) == ROW_LEN + 1 == int(raw[0, 0])
    assert int(untrusted[0, 0]) == POP_SIZE * ROW_LEN
    assert int(trusted[0, 0]) < POP_SIZE * ROW_LEN  # inside the payload, i.e. readable
    # ... and it names a *different* row's column, one row down from its own
    assert int(trusted[0, 0]) == 1 * ROW_LEN + 1


def test_prepare_flat_index_trusted_passes_a_negative_column_through_raw(common):
    """Invariant: a negative column is redirected when untrusted, passed raw when trusted.

    Oracle: ``row * R + col`` — for ``col == -1`` and ``row == 0`` that is ``-1``,
    which numpy reads as the last element of the flattened payload (the docstring
    says negative indices *wrap*). ``I_INF`` only lives in row ``0`` so the raw
    arithmetic stays representable in ``int32``.
    """
    x = _payload()
    idx = _in_range_block((3,))
    idx[0, 0] = -1
    idx[0, 1] = common.tools.I_INF
    assert int(idx[1:, :].min()) >= 0  # no other row can overflow the raw sum

    trusted = _run(common, "prepare_flat_index", x, idx, None, 0, True)
    untrusted = _run(common, "prepare_flat_index", x, idx, None, 0, False)

    np.testing.assert_array_equal(
        trusted, _np_prepare_flat_index(POP_SIZE, ROW_LEN, idx, trusted=True)
    )
    np.testing.assert_array_equal(untrusted, _np_prepare_flat_index(POP_SIZE, ROW_LEN, idx))
    assert int(trusted[0, 0]) == -1
    assert int(trusted[0, 1]) == common.tools.I_INF
    assert int(untrusted[0, 0]) == POP_SIZE * ROW_LEN
    assert int(untrusted[0, 1]) == POP_SIZE * ROW_LEN

    # spent on a payload the raw value wraps: ``-1`` at row 0 reads the *last*
    # element, and ``p * R - 1`` at a later row reads the previous row's last
    # cell — neither is the caller's own row, and neither raises.
    wrap = np.array([[-1, 1], [2, 3], [4, 5], [-1, 0]], dtype=np.int32)
    wrap_block = _run(common, "prepare_flat_index", x, wrap, None, 0, True)
    assert int(wrap_block[0, 0]) == -1
    assert int(wrap_block[3, 0]) == 3 * ROW_LEN - 1 == 17
    silent = _run(common, "flat_gather", x, wrap, True)
    assert float(silent[0, 0]) == float(x.reshape(-1)[-1]) == float(x[POP_SIZE - 1, ROW_LEN - 1])
    assert float(silent[3, 0]) == float(x[2, ROW_LEN - 1])
    with pytest.raises(evoxc.ShapeError):
        _run(common, "flat_gather", x, wrap, False)


# --------------------------------------------------------------------------- #
# the prepared pair spends a trusted block exactly like the one-shot wrapper
# --------------------------------------------------------------------------- #
def test_flat_gather_prepared_with_a_trusted_block_matches_the_one_shot(common):
    """Invariant: a ``trusted=True`` block gathers what ``flat_gather`` gathers.

    Oracle: ``.test_tools._np_flat_gather`` (row-wise numpy indexing) and the
    one-shot wrapper itself, for both ``trusted`` spellings — the reason to hoist
    the block is op count, never a different result.
    """
    x = _payload()
    idx = _in_range_block((3,))
    block = _run(common, "prepare_flat_index", x, idx, None, 0, True)

    out = _run(common, "flat_gather_prepared", x, block, (POP_SIZE, 3))
    expected = _np_flat_gather(x, idx)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(out, _run(common, "flat_gather", x, idx))
    np.testing.assert_array_equal(out, _run(common, "flat_gather", x, idx, True))
    assert out.shape == idx.shape
    assert out.dtype == x.dtype


def test_flat_scatter_prepared_with_a_trusted_block_matches_the_one_shot(common):
    """Invariant: a ``trusted=True`` block scatters what ``flat_scatter`` scatters.

    Oracle: ``.test_tools._np_flat_scatter`` (Python write loop, last write wins)
    and the one-shot wrapper. Duplicate columns are included on purpose: the
    docstring's "resolve in flat traversal order" must not depend on whether the
    block was hoisted.
    """
    x = _payload()
    idx = _in_range_block((3,))
    idx[1, 0] = idx[1, 1]  # a duplicate column: the later write must survive
    updates = (np.arange(POP_SIZE * 3).reshape(POP_SIZE, 3) + 100).astype(np.float32)
    block = _run(common, "prepare_flat_index", x, idx, None, 0, True)

    out = _run(common, "flat_scatter_prepared", x, block, updates)
    expected = _np_flat_scatter(x, idx, updates)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(out, _run(common, "flat_scatter", x, idx, updates))
    np.testing.assert_array_equal(out, _run(common, "flat_scatter", x, idx, updates, True))
    assert out.shape == x.shape
    assert out.dtype == x.dtype


# --------------------------------------------------------------------------- #
# present / fallback: the mask runs before the index, the redirect after it
# --------------------------------------------------------------------------- #
def test_masked_position_lands_on_fallback_under_both_spellings(common):
    """Invariant: a masked-out sentinel becomes ``fallback`` before the index is formed.

    Oracle: :func:`_np_prepare_flat_index` with ``present``/``fallback`` (the
    documented order: replace, then redirect), so the masked cell is
    ``p * R + fallback`` and — being in range — the two spellings must agree
    cell for cell. ``fallback == 0`` is the gather spelling the docstring names.
    """
    x = _payload()
    idx = _in_range_block((3,))
    idx[0, 0] = common.tools.I_INF
    idx[0, 2] = common.tools.I_INF
    present = idx != common.tools.I_INF
    assert present.sum() == idx.size - 2
    # `present` crosses the graph boundary as a tensor, so it is an argument
    trusted = _run(common, "prepare_flat_index", x, idx, present, 0, True)
    untrusted = _run(common, "prepare_flat_index", x, idx, present, 0, False)

    expected = _np_prepare_flat_index(POP_SIZE, ROW_LEN, idx, present, 0, trusted=True)
    np.testing.assert_array_equal(trusted, expected)
    np.testing.assert_array_equal(untrusted, expected)
    # the masked cells hold the fallback column, not the sentinel
    assert int(trusted[0, 0]) == 0 * ROW_LEN + 0
    assert int(trusted[0, 2]) == 0 * ROW_LEN + 0
    assert (trusted[present] != common.tools.I_INF).all()
    assert (untrusted[~present] == 0).all()


def test_unmasked_sentinel_under_trusted_keeps_the_raw_arithmetic(common):
    """Invariant: with the redirect gone, an *unmasked* sentinel is passed through raw.

    Oracle: ``row * R + col``. Untrusted answers ``P * R`` for the same cell; with
    ``trusted=True`` the block carries ``I_INF`` itself, i.e. a caller bug whose
    only remaining net is the backend. ``I_INF`` sits in row ``0`` so the raw
    ``0 * R + I_INF`` stays representable in ``int32``.
    """
    x = _payload()
    idx = _in_range_block((2,))
    idx[0, 0] = common.tools.I_INF

    trusted = _run(common, "prepare_flat_index", x, idx, None, 0, True)
    untrusted = _run(common, "prepare_flat_index", x, idx, None, 0, False)

    np.testing.assert_array_equal(
        trusted, _np_prepare_flat_index(POP_SIZE, ROW_LEN, idx, trusted=True)
    )
    assert int(trusted[0, 0]) == common.tools.I_INF  # raw: 0 * R + I_INF
    assert int(untrusted[0, 0]) == POP_SIZE * ROW_LEN
    assert int(trusted[0, 0]) != int(untrusted[0, 0])
    # observed behaviour: the sentinel is far outside the payload, so the gather
    # raises rather than reading anything
    with pytest.raises(evoxc.ShapeError):
        _run(common, "flat_gather", x, idx, True)


def test_trusted_block_silently_reads_the_neighbouring_row(common):
    """Invariant: ``R <= col < 2R`` without the redirect reads a *different* row.

    Oracle: the flattened payload itself. ``1 * R + (R + 1) == 13`` is a legal
    index into an ``(3, 6)`` payload (``13 < 18``), so the gather succeeds and
    returns ``x[2, 1]`` — row ``2``'s value for row ``1``'s query. The untrusted
    spelling rewrites the same cell to ``P * R`` and the backend raises, which is
    the whole reason the redirect exists.
    """
    pop_size, row_len = 3, 6
    x = _payload(pop_size, row_len)
    idx = np.array([[0, 1], [row_len + 1, 2], [3, 4]], dtype=np.int32)

    block = _run(common, "prepare_flat_index", x, idx, None, 0, True)
    assert int(block[1, 0]) == 1 * row_len + (row_len + 1) == 13
    assert 13 < pop_size * row_len  # inside the flattened payload: readable

    out = _run(common, "flat_gather", x, idx, True)
    assert float(out[1, 0]) == float(x.reshape(-1)[13]) == float(x[2, 1])
    # every in-range cell still reads its own row (the oracle is fed the
    # sanitized block, because the escaping column is not a legal column)
    in_range = idx < row_len
    np.testing.assert_array_equal(
        out[in_range],
        _np_flat_gather(x, np.where(in_range, idx, 0))[in_range],
    )
    # the redirect turns the same block into a loud backend error
    with pytest.raises(evoxc.ShapeError):
        _run(common, "flat_gather", x, idx, False)


def test_masked_flat_gather_trusted_fills_the_masked_positions(common):
    """Invariant: ``trusted=True`` does not weaken ``masked_flat_gather``'s fill.

    Oracle: ``np.where(present, _np_flat_gather(x, idx-with-the-sentinel-replaced
    -by-0), fill)`` for an explicit ``fill`` and for the default ``NaN``;
    ``present`` is computed by the helper itself, so a masked sentinel never
    reaches an index under either spelling.
    """
    x = _payload()
    idx = _in_range_block((2,))
    idx[0, 0] = common.tools.I_INF
    idx[2, 1] = common.tools.I_INF
    present = idx != common.tools.I_INF

    trusted = _run(common, "masked_flat_gather", x, idx, None, 0.0, True)
    untrusted = _run(common, "masked_flat_gather", x, idx, None, 0.0, False)
    gathered = _np_flat_gather(x, np.where(present, idx, 0))  # sentinel is no column
    expected = np.where(present, gathered, np.float32(0.0))
    np.testing.assert_array_equal(trusted, expected)
    np.testing.assert_array_equal(untrusted, expected)
    assert (trusted[~present] == 0.0).all()

    # the default fill (NaN) through a closure that bakes it into the trace
    default = _run_bound(common, "masked_flat_gather", x, idx, trusted=True)
    np.testing.assert_allclose(
        default,
        np.where(present, gathered, np.float32("nan")),
        equal_nan=True,
    )
    assert np.isnan(default[~present]).all()
    assert not np.isnan(default[present]).any()


def test_masked_flat_scatter_trusted_keeps_the_dump_redirect(common):
    """Invariant: ``trusted`` drops the row-bound redirect, never the dump redirect.

    Oracle: :func:`_np_masked_scatter` (a write loop that routes masked writes to
    the trailing column ``R`` of a ``(P, R + 1)`` destination). The masked cell
    may hold ``I_INF``; routing it to the backend would raise, so both spellings
    must agree and the dump column must absorb the write.
    """
    dst = np.arange(POP_SIZE * (ROW_LEN + 1), dtype=np.float32).reshape(
        POP_SIZE, ROW_LEN + 1
    )
    idx = _in_range_block((2,))
    idx[0, 0] = common.tools.I_INF
    mask = idx != common.tools.I_INF
    updates = (np.arange(POP_SIZE * 2).reshape(POP_SIZE, 2) + 50).astype(np.float32)

    trusted = _run(common, "masked_flat_scatter", dst, idx, updates, mask, True)
    untrusted = _run(common, "masked_flat_scatter", dst, idx, updates, mask, False)

    expected = _np_masked_scatter(dst, idx, updates, mask)
    np.testing.assert_array_equal(trusted, expected)
    np.testing.assert_array_equal(untrusted, expected)
    assert trusted.shape == dst.shape
    # the dropped write sits in the dump column; the cell it named is untouched
    assert float(trusted[0, ROW_LEN]) == float(updates[0, 0])
    assert float(trusted[0, 0]) == float(dst[0, 0])
    assert not np.array_equal(trusted[0, :ROW_LEN], dst[0, :ROW_LEN])  # the other write landed


# --------------------------------------------------------------------------- #
# batch_offsets: the other half of the same flat-index contract
# --------------------------------------------------------------------------- #
def test_batch_offsets_is_the_row_major_table(common):
    """Invariant: ``batch_offsets(P, R)`` is the literal ``[[0], [R], [2R], ...]``.

    Oracle: the literal, plus ``np.arange(P)[:, None] * R``. :mod:`test_tools`
    pins the same helper against the same expression; this test states the table
    the *prepared* index arithmetic adds to a column block.
    """
    out = _run(common, "batch_offsets", POP_SIZE, ROW_LEN)
    np.testing.assert_array_equal(out, np.array([[0], [6], [12], [18]], dtype=np.int32))
    np.testing.assert_array_equal(
        out, np.arange(POP_SIZE, dtype=np.int32)[:, None] * np.int32(ROW_LEN)
    )
    assert out.shape == (POP_SIZE, 1)
    assert out.dtype == np.int32


def test_batch_offsets_rejects_an_int32_overflowing_flat_range_with_its_message(common):
    """Invariant: ``P * R >= I_INF`` raises ``ConfigError`` naming both statics.

    Oracle: the documented message of ``batch_offsets`` — the raised error must
    say *which* product overflowed, because a flat index that wrapped would read
    another row's data instead of failing.
    """
    pop_size, row_len = 2**16, 2**15
    assert pop_size * row_len >= common.tools.I_INF  # the bound really is crossed
    with pytest.raises(common.errors.ConfigError) as excinfo:
        _run(common, "batch_offsets", pop_size, row_len)
    message = str(excinfo.value)
    assert "pop_size * row_len must stay below I_INF" in message
    assert f"pop_size={pop_size}" in message
    assert f"row_len={row_len}" in message


# --------------------------------------------------------------------------- #
# the prepared pair propagates the backend error for an escaping block
# --------------------------------------------------------------------------- #
def test_prepared_gather_and_scatter_propagate_the_out_of_range_error(common):
    """Invariant: a hand-built escaping block raises ``evoxc.ShapeError`` unchanged.

    Oracle: the same exception class ``test_tools`` observes for the one-shot
    wrappers (the numpy backend raises where JAX clamped). ``P * R`` is the value
    the redirect itself writes, and ``P * R + 3`` is any other out-of-payload
    index; a block that stays inside works, so the raise is about the index and
    not about the call.
    """
    x = _payload()
    updates = np.full((POP_SIZE, 2), 7.0, dtype=np.float32)
    # the flat indices of columns (0, 1) of every row, i.e. a block the redirect
    # would leave alone
    in_range = np.array([[0, 1], [6, 7], [12, 13], [18, 19]], dtype=np.int32)
    columns = np.tile(np.array([[0, 1]], dtype=np.int32), (POP_SIZE, 1))
    np.testing.assert_array_equal(
        in_range, _np_prepare_flat_index(POP_SIZE, ROW_LEN, columns, trusted=True)
    )
    np.testing.assert_array_equal(
        _run(common, "flat_gather_prepared", x, in_range, (POP_SIZE, 2)),
        _np_flat_gather(x, columns),
    )
    for escaped in (POP_SIZE * ROW_LEN, POP_SIZE * ROW_LEN + 3):
        block = in_range.copy()
        block[1, 1] = escaped
        assert escaped >= POP_SIZE * ROW_LEN  # outside the flattened payload
        with pytest.raises(evoxc.ShapeError):
            _run(common, "flat_gather_prepared", x, block, (POP_SIZE, 2))
        with pytest.raises(evoxc.ShapeError):
            _run(common, "flat_scatter_prepared", x, block, updates)
