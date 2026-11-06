"""Regression tests for ``./ir_ops.py``, the one traced-IR op walker this node counts with.

What this module proves
-----------------------
``ir_ops`` is the single walk behind every op count quoted in ``./``: the route
proofs (the host leg scatters and the accelerator leg does not, the host speciate
graph carries ``if`` ops and the accelerator one carries none) and the cost
budgets (one ``measure`` call's marginal op count).  A walk that is subtly wrong
reports a *plausible* number instead of failing, so it is pinned here on two
hand-written ``evoxc`` graphs rather than through a library kernel:

* **region-free** -- a four-op elementwise graph: ``count_ops`` counts every op of
  the block except the terminator, ``op_names`` agrees on the total, and the
  histogram is exactly the ops written in the source;
* **nested region** -- an ``evoxc.while_loop`` whose body runs a second
  ``while_loop``: the walk descends ``op.regions -> region.blocks`` two levels
  down, so the inner loop's own ``while``, ``less`` and ``add`` ops are counted.
  This is the regression: a walk that handed the ``Region`` back to itself raises
  ``AttributeError`` on the first region-carrying op, because a region is an
  ordered list of *blocks* and carries no ``.ops`` of its own -- it never reports
  a number at all;
* **terminators** -- the ``"return"`` op is not counted unless a caller asks for
  it, and the count of terminators is the number of blocks the walk reached (five
  for the nested graph, one for the flat one), which is what shows the descent is
  *complete* rather than merely non-zero.

Oracles used here (never the walker)
------------------------------------
The graphs are written by hand in this module, one ``evoxc`` op per construction
step, so the op names and their nesting are known from the source text; the block
structure is read back from the traced IR by hand (``entry_block.ops``,
``op.regions[...].blocks``) and asserted separately from the walker's answer.

Gotchas this module encodes
---------------------------
* ``op.regions`` yields ``Region`` objects; their bodies are ``region.blocks``,
  each an ``evoxc.ir.Block`` with its own ``"return"`` terminator.  The descent
  is the contract under test, so the tests assert both that the walker reaches a
  nested body and that a ``Region`` carries no ``.ops``.
* A ``while_loop`` init leaf must be a ``SymbolicTensor``: ``evoxc.zeros`` is a
  concrete ``Tensor`` outside a graph, so every loop counter here is derived from
  the graph's own input (``reduce_sum(x) * 0.0``) and stays symbolic.
* Only ``evoxc.trace`` is called -- tracing is neither a build nor a run -- and
  the input spec is spelled directly (``evoxc.TensorSpec``) because nothing here
  executes.  Backend is irrelevant, there is no GPU, no ``iree``, no
  ``jax``/``torch``.
"""

from __future__ import annotations

import collections

import pytest

import evoxc

from .ir_ops import TERMINATOR, count_graph_ops, count_ops, op_names

#: Input width of both probe graphs; only the *static* shape matters here.
SHAPE = (3,)


# --------------------------------------------------------------------------- #
# the two hand-written graphs
# --------------------------------------------------------------------------- #
def _flat(x):
    """A region-free graph: ``x * 2 + 1``, one ``multiply`` and one ``add``."""
    return evoxc.add(evoxc.multiply(x, 2.0), 1.0)


def _nested(x):
    """A ``while`` whose body runs a second ``while``: five blocks, one graph.

    The outer loop carries ``(counter, value)``; its body derives a fresh 0-d
    counter for the inner loop from the carried value, runs the inner loop, and
    adds the inner trip count back into the carried value.
    """
    zero = evoxc.multiply(evoxc.reduce_sum(x), 0.0)

    def outer_cond(carried):
        counter, _value = carried
        return evoxc.less(counter, 3.0)

    def outer_body(carried):
        counter, value = carried
        inner_zero = evoxc.multiply(evoxc.reduce_sum(value), 0.0)

        def inner_cond(inner):
            inner_counter, _inner_value = inner
            return evoxc.less(inner_counter, 2.0)

        def inner_body(inner):
            inner_counter, inner_value = inner
            return evoxc.add(inner_counter, 1.0), evoxc.multiply(inner_value, 2.0)

        trips, inner_value = evoxc.while_loop(
            inner_cond, inner_body, (inner_zero, value)
        )
        return evoxc.add(counter, 1.0), evoxc.add(inner_value, trips)

    _counter, result = evoxc.while_loop(outer_cond, outer_body, (zero, x))
    return result


@pytest.fixture(scope="module")
def flat():
    """The region-free graph, traced once for this module."""
    return evoxc.trace(_flat, evoxc.TensorSpec(SHAPE, evoxc.float32))


@pytest.fixture(scope="module")
def nested():
    """The two-level ``while`` graph, traced once for this module."""
    return evoxc.trace(_nested, evoxc.TensorSpec(SHAPE, evoxc.float32))


# --------------------------------------------------------------------------- #
# a graph without control flow
# --------------------------------------------------------------------------- #
def test_a_region_free_graph_counts_every_op_by_name(flat):
    """Every op of a flat block is counted once, the terminator aside.

    Oracle: the block's own op list, read back from the IR. The source writes one
    ``multiply`` and one ``add``; the two ``constant`` ops are its literal
    operands, so the assertions stay on the two names the construction spells and
    on the block length, not on how evoxc materialises its constants.

    Assertion: the block ends in the terminator, both written op names appear
    exactly once, ``count_ops`` equals the block's non-terminator length, and the
    graph-level count and the histogram total agree with it.
    """
    (function,) = flat.module.functions
    block = function.entry_block
    assert block.ops[-1].name == TERMINATOR, (
        f"the flat block must end in its terminator, got {[op.name for op in block.ops]}"
    )
    counts = op_names(flat)
    assert counts["multiply"] == 1, f"the histogram is {counts}"
    assert counts["add"] == 1, f"the histogram is {counts}"
    assert count_ops(block) == len(block.ops) - 1, (
        "count_ops must count every op of the block except its terminator: "
        f"{count_ops(block)} against {len(block.ops)} block ops"
    )
    assert count_graph_ops(flat) == count_ops(block) == sum(counts.values()), (
        "the whole-graph total, the block total and the histogram total must be "
        f"one number: {count_graph_ops(flat)}, {count_ops(block)}, {sum(counts.values())}"
    )


# --------------------------------------------------------------------------- #
# the descent through region blocks -- the regression
# --------------------------------------------------------------------------- #
def test_count_ops_descends_into_a_nested_while_body(nested):
    """The walk reaches the *inner* loop's ops: the region recursion is fixed.

    Oracle: the hand-written construction. The outer block holds one ``while`` op
    beside its operands; the body region that ``while`` owns runs a second
    ``while_loop``, and each loop's condition region holds one ``less``. So the
    whole-graph count exceeds the outer block's own op count, and the inner
    ``while``, both ``less`` ops and the body's ``add`` are all in the histogram.

    Assertion: strictly more ops than the outer block holds, exactly two ``while``
    ops and two ``less`` ops (one per loop), an ``add`` from the inner body, and
    the histogram total equal to the graph total. A walk that stopped at the outer
    block would report one ``while`` and one ``less``; the pre-fix walk raised
    ``AttributeError`` before reporting anything.
    """
    (function,) = nested.module.functions
    block = function.entry_block
    outer_only = sum(1 for op in block.ops if op.name != TERMINATOR)
    assert any(op.name == "while" for op in block.ops), (
        f"the outer block must carry the loop op, got {[op.name for op in block.ops]}"
    )
    counts = op_names(nested)
    assert count_graph_ops(nested) == sum(counts.values()), (
        f"the graph total {count_graph_ops(nested)} disagrees with the histogram "
        f"total {sum(counts.values())} ({counts})"
    )
    assert sum(counts.values()) > outer_only, (
        f"the walk counted {sum(counts.values())} ops, no more than the outer "
        f"block's {outer_only}: the nested while body was not descended into ({counts})"
    )
    assert counts["while"] == 2, (
        f"the nested loop must add a second while op, histogram {counts}"
    )
    assert counts["less"] == 2, (
        f"each loop's condition region holds one less op, histogram {counts}"
    )
    assert counts["add"] >= 3, (
        "the inner body's add op must be counted as well (outer body 2, inner body "
        f"1), histogram {counts}"
    )


def test_a_region_is_a_list_of_blocks_and_has_no_ops_of_its_own(nested):
    """The descent is ``op.regions -> region.blocks``; a region is not a block.

    Oracle: the traced IR of the nested graph. ``op.regions`` yields ``Region``
    objects -- ordered lists of blocks, ``region.blocks[0]`` being the entry block
    -- and a ``Region`` carries no ``.ops`` attribute, so the walker descends
    through the blocks and a walk that handed the region back to itself raises
    ``AttributeError``.

    Assertion: the outer body region holds the nested ``while`` op, ``count_ops``
    on that body block counts more than the block's own op list (it follows the
    inner loop in), and reading ``.ops`` off the region raises -- so if evoxc ever
    gives a ``Region`` an ``.ops`` attribute this test fails and the contract is
    re-read instead of silently assumed.
    """
    (function,) = nested.module.functions
    (while_op,) = [op for op in function.entry_block.ops if op.name == "while"]
    body = while_op.regions[1]
    assert isinstance(body.blocks, list) and body.blocks, (
        f"a region body must be a non-empty list of blocks, got {body.blocks!r}"
    )
    inner_names = [op.name for inner_block in body.blocks for op in inner_block.ops]
    assert "while" in inner_names, (
        f"the outer body region must carry the nested loop, got {inner_names}"
    )
    body_block = body.blocks[0]
    assert count_ops(body_block) > sum(
        1 for op in body_block.ops if op.name != TERMINATOR
    ), (
        "count_ops on the body block must follow its nested regions: "
        f"{count_ops(body_block)} against the block's own "
        f"{sum(1 for op in body_block.ops if op.name != TERMINATOR)} non-terminator ops"
    )
    with pytest.raises(AttributeError):
        body.ops  # a Region, not a Block -- the walk must go through .blocks


# --------------------------------------------------------------------------- #
# terminators
# --------------------------------------------------------------------------- #
def test_the_return_terminator_is_not_counted_unless_requested(flat, nested):
    """``"return"`` is excluded by default and countable on request.

    Oracle: the hand-read block structure. Every block ends in exactly one
    ``return``, so asking for terminators adds one histogram entry whose count is
    the number of blocks the walk reached -- one for the flat graph and five for
    the nested one (entry block, the outer loop's condition and body regions, the
    inner loop's condition and body regions). A walk that stopped at the
    top-level block would report one for both.

    Assertion: the default histogram carries no ``"return"`` key, the requested
    one differs from it by exactly ``Counter({"return": blocks})``, and both agree
    on every other name -- so the exclusion is a filter, not a different walk.
    """
    for graph, blocks in ((flat, 1), (nested, 5)):
        counts = op_names(graph)
        assert TERMINATOR not in counts, (
            f"the terminator must not be counted by default, histogram {counts}"
        )
        requested = op_names(graph, include_terminators=True)
        assert requested[TERMINATOR] == blocks, (
            f"the walk reached {requested[TERMINATOR]} terminator(s), expected one "
            f"per block ({blocks}): {requested}"
        )
        assert requested - counts == collections.Counter({TERMINATOR: blocks})
        assert counts == collections.Counter(
            {
                name: count
                for name, count in requested.items()
                if name != TERMINATOR
            }
        ), f"the requested histogram differs on non-terminator names: {requested}"
