"""One op walker for the traced-IR counts the unit suites quote.

Several suites in this node prove that a *route* was really taken by counting the
``op.name`` of a traced ``evoxc`` graph instead of comparing values: the host
schedule leg scatters and the accelerator leg does not, the gated speciate graph
carries four ``if`` ops where the dense and accelerator ones carry none, the
static level sweep has no ``while`` op while the loop leg has exactly one, and one
``measure`` call stays inside its op budget.  A walk that is subtly wrong reports
a *plausible* number rather than failing, so the walk lives here once: every
count in this node -- including the cost-model figures quoted in the child
``CONTEXT.md`` files -- comes out of this module.

The two shapes of a traced graph
--------------------------------
``graph.module.functions[i].entry_block`` is a :class:`evoxc.ir.Block`:

* ``block.ops`` is the ordered op list, and its last op is that block's
  terminator (``"return"`` in this tree's evoxc);
* ``op.regions`` is a tuple of :class:`evoxc.ir.Region` objects -- the nested
  bodies of ``if``/``while``/``scan``.  A ``Region`` is an *ordered list of
  blocks* (``region.blocks``, with ``region.blocks[0]`` the entry block), **not**
  a block: it carries no ``.ops`` of its own.  Handing a region back to the walk
  therefore raises ``AttributeError`` on the first region-carrying op, so the
  descent is ``op.regions -> region.blocks -> block.ops`` -- never
  ``op.regions -> block.ops``.

What every walker here shares
-----------------------------
The ``"return"`` terminator is not an executed op, so it is not counted: a total
that included it would depend on how many blocks a kernel happens to have, and an
op budget would then move with an unrelated refactor.  A caller that compares
*whole* histograms of two graphs wants the terminators too -- a differing block
count must not cancel out -- and asks for them explicitly with
``op_names(graph, include_terminators=True)``.

No optional dependency: this module reads the IR of an already-traced graph, so it
imports nothing beyond the standard library and works on every backend.
"""

from __future__ import annotations

import collections
from typing import Any

__all__ = ["TERMINATOR", "count_graph_ops", "count_ops", "op_names"]

#: The terminator op name v1 emits at the end of every block (``evoxc.ir.Block``).
TERMINATOR = "return"


def _walk(
    block: Any, counts: collections.Counter, include_terminators: bool
) -> None:
    """Add ``block``'s op names to ``counts``, descending into every region body.

    Args:
        block: An ``evoxc.ir.Block`` -- a function's ``entry_block`` or one block
            of a region body.
        counts: The accumulator, ``op name -> count``, updated in place.
        include_terminators: Count ``"return"`` ops too. A cost total passes
            ``False`` (the terminator is not executed), a whole-histogram
            comparison passes ``True`` (block structure must not cancel out).

    Notes:
        The descent is ``op.regions -> region.blocks``: ``op.regions`` yields
        ``Region`` objects, which are ordered lists of *blocks* and have no
        ``.ops``, so recursing into the region itself would raise
        ``AttributeError`` on the first ``if``/``while``/``scan`` op.
    """
    for op in block.ops:
        if op.name == TERMINATOR and not include_terminators:
            continue
        counts[op.name] += 1
        for region in getattr(op, "regions", ()) or ():
            for inner in region.blocks:
                _walk(inner, counts, include_terminators)


def count_ops(block: Any) -> int:
    """Executed ops of one block, every nested region body included.

    Args:
        block: An ``evoxc.ir.Block`` (``entry_block`` or a region body).

    Returns:
        int: Non-terminator ops of the block plus every ``op.regions`` body --
        the same recursive walk the cost tables in ``CONTEXT.md`` are counted
        with.

    Notes:
        The ``"return"`` terminator is not an executed op, so counting it would
        make a budget depend on how many blocks a kernel happens to have.
    """
    counts: collections.Counter = collections.Counter()
    _walk(block, counts, False)
    return sum(counts.values())


def count_graph_ops(graph: Any) -> int:
    """Executed ops over every function of a traced ``evoxc`` graph.

    Args:
        graph: A traced ``evoxc.Graph`` (what ``evoxc.trace`` returns).

    Returns:
        int: :func:`count_ops` summed over each function's ``entry_block``.
    """
    return sum(count_ops(function.entry_block) for function in graph.module.functions)


def op_names(graph: Any, *, include_terminators: bool = False) -> collections.Counter:
    """Op-name histogram of every function of a traced ``evoxc`` graph.

    Args:
        graph: A traced ``evoxc.Graph`` (what ``evoxc.trace`` returns).
        include_terminators: Count each block's ``"return"`` terminator as well.
            Default ``False``: a histogram used as a cost table or compared
            against a route's op list must not grow with the block count. Pass
            ``True`` only when two graphs' *whole* histograms are compared and a
            structural difference must show up in the table.

    Returns:
        collections.Counter: ``op name -> count`` over the whole graph, nested
        region bodies included.
    """
    counts: collections.Counter = collections.Counter()
    for function in graph.module.functions:
        _walk(function.entry_block, counts, include_terminators)
    return counts
