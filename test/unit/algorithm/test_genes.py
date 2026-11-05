"""Unit tests for the two HyperNEAT substrate gene kernels (``hyperneat/genes.py``).

What this module proves
-----------------------
``HyperNEATNode`` and ``HyperNEATConn`` are the only two genes of the tree that are never
evolved: the substrate payload is repainted from the decoded CPPN weights on every
``HyperNEAT.transform``, so these classes implement the *execution* half of the gene
contract -- ``forward`` (and, for the node gene, the ``combine`` hook) -- and nothing of the
genetic-operator surface.  They had no behavioural test before this module; every clause
below is pinned on a hand-built payload:

* the declared tables (widths, option columns, the ``weight`` spec) and the constructor
  validation of the node gene's two option callables;
* ``HyperNEATNode.combine`` is the identity (a substrate node has no bias/response map) and
  ``HyperNEATNode.forward`` is the inherited node algebra: a masked reduction over the
  incoming-source axis (a dead slot contributes exactly ``0.0``, an all-dead row reduces to
  exactly ``0.0``, never ``NaN``) followed by the fixed activation, which is *not* applied
  to output rows;
* ``HyperNEATConn.forward`` is one multiply by the ``"weight"`` column, in both documented
  spellings (raw ``(..., K, CA)`` rows and the ``ConnOperands`` bundle the genome passes);
* the NaN provenance of that pair: the bundle's columns are already cleaned (a ``NaN``
  weight row reads as ``0``), the raw rows are not, and the kernel guards no ``NaN`` of its
  own -- masking has a single owner, the node gene's ``valid_mask``;
* the host-only ``repr``/``to_dict`` hooks, whose option/column names must come from the
  configured tables rather than from literals.

Oracles used here (never the kernel under test)
-----------------------------------------------
* closed forms written in this file.  The masked reduction is a left-to-right ``float32``
  accumulation from ``+0.0`` -- the order ``AGG.sum``/``AGG.mean`` document for a window
  narrower than eight -- and the activation is ``1 / (1 + exp(-z))`` in ``float32``, the
  expression order ``ACT.sigmoid`` documents.  Both reproduce the kernel bit for bit, so
  the comparisons are ``rtol=0, atol=0``.
* a per-individual Python loop over ``(P, N, K)``, so every row of a batch is checked on its
  own instead of against a second vectorised kernel.
* literals restated from the documentation (``fixed_attrs``, widths, the ``weight`` spec's
  five distribution fields, the registry names ``"sum"``/``"max"``/``"sigmoid"``/``"tanh"``,
  the ``-1`` identity name) so a change in ``genes.py`` fails a test instead of silently
  redefining the expectation.

Gotchas this module encodes
---------------------------
* A dead slot carries ``NaN`` in ``messages`` on purpose: the *mask* is what neutralises
  it, not the data.  A kernel that reduced the raw window would return ``NaN`` here, so an
  all-dead row must come back exactly ``0.0``.
* An output row is **not** activated by the node gene: ``BaseNode.prepare`` forces its
  activation id to the ``-1`` identity sentinel, so the value is the bare aggregate.
* ``HyperNEATConn.forward`` reads its column through ``column_of``, resolved by *name*.
  The two spellings are not interchangeable for ``NaN`` data: the raw rows carry it, the
  prepared bundle has already cleaned it.
* A ``(P, K)`` source array is not reduced by the connection gene -- the message of an edge
  is elementwise and the destination node aggregates.  ``K`` is the in-degree window here,
  not a node axis.
* The gene payload's option columns are painted ``0`` by ``HyperNEAT._assemble_substrate``
  and both option tables hold exactly one element, so ``0`` selects the configured pair in
  every row.  ``_node_attrs`` below paints exactly that.

Backend is ``"numpy"`` throughout, through ``./hyperneat_kernel.run_method`` only; no
``jax``/``torch``, no network, no GPU, ``P <= 3``, ``N <= 3``, ``K <= 3``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorneat.algorithm.hyperneat import HyperNEATConn, HyperNEATNode
from tensorneat.common import ACT, AGG, State
from tensorneat.common.errors import ConfigError
from tensorneat.genome.gene.base import GeneAttr

from .hyperneat_kernel import run_method

# --------------------------------------------------------------------------- #
# oracles: the documented algebra, written here (never read off the kernel)
# --------------------------------------------------------------------------- #


def _fold_sum(values) -> np.float32:
    """The masked sum as a ``float32`` accumulation, documented order.

    ``AGG.sum`` emits, for a window narrower than eight slots, ``+0.0`` followed by one
    left-to-right add per live slot (see the ``_sum_last`` notes), so this reproduces it
    bit for bit -- including the empty case, which is exactly ``+0.0``.

    Args:
        values: The live slot values, in slot order (already masked).

    Returns:
        ``() float32`` sum, starting from ``np.float32(0.0)``.
    """
    total = np.float32(0.0)
    for value in values:
        total = np.float32(total + np.float32(value))
    return total


def _sigmoid(z) -> np.float32:
    """``ACT.sigmoid`` in ``float32``: ``1 / (1 + exp(-z))``, documented expression order.

    Args:
        z: Pre-activation value.

    Returns:
        ``() float32`` activation.
    """
    z = np.float32(z)
    return np.float32(np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(-z))))


def _aggregates(messages, valid, aggregation: str) -> np.ndarray:
    """``(P, N)`` masked aggregation, re-derived with one Python loop per individual.

    Args:
        messages: ``(P, N, K)`` float32 incoming-message window; dead slots may hold NaN.
        valid: ``(P, N, K)`` bool, True on live slots.
        aggregation: ``"sum"`` or ``"mean"`` -- the two self-neutral options the substrate
            constructor accepts; ``mean`` divides by ``max(count, 1)``.

    Returns:
        ``(P, N)`` float32 aggregates.

    Raises:
        AssertionError: If ``aggregation`` is not one of the two documented options.
    """
    pop_size, num_nodes, width = messages.shape
    out = np.zeros((pop_size, num_nodes), dtype=np.float32)
    for p in range(pop_size):
        for n in range(num_nodes):
            live = [messages[p, n, k] for k in range(width) if valid[p, n, k]]
            total = _fold_sum(live)
            if aggregation == "mean":
                total = np.float32(total / np.float32(max(len(live), 1)))
            elif aggregation != "sum":
                raise AssertionError(f"no oracle for aggregation {aggregation!r}")
            out[p, n] = total
    return out


def _node_oracle(messages, is_output, valid, aggregation: str = "sum") -> np.ndarray:
    """``(P, N)`` documented ``HyperNEATNode.forward``: aggregate, then activate.

    Args:
        messages: ``(P, N, K)`` float32 window.
        is_output: ``(P, N)`` bool, True on the read-out rows (left unactivated).
        valid: ``(P, N, K)`` bool live-slot mask.
        aggregation: ``"sum"`` or ``"mean"``.

    Returns:
        ``(P, N)`` float32 node values.
    """
    aggregated = _aggregates(messages, valid, aggregation)
    out = aggregated.copy()
    for p, n in zip(*np.nonzero(~is_output)):
        out[p, n] = _sigmoid(aggregated[p, n])
    return out


def _node_attrs(pop_size: int, num_nodes: int) -> np.ndarray:
    """The ``(P, N, 2)`` option-index rows ``_assemble_substrate`` paints: all zero.

    Both option tables of a ``HyperNEATNode`` hold exactly one element, so index ``0`` is
    the only legal value in each column and selects the constructor's callables.

    Args:
        pop_size: Population size ``P``.
        num_nodes: Node capacity ``N``.

    Returns:
        ``(P, N, 2)`` float32 zero array (column 0 ``aggregation``, column 1 ``activation``).
    """
    return np.zeros((pop_size, num_nodes, 2), dtype=np.float32)


def _run_forward(node, attrs, messages, is_output, valid):
    """Trace and run ``HyperNEATNode.forward`` on host arrays.

    Args:
        node: The gene under test.
        attrs: ``(P, N, 2)`` attribute rows, or the ``NodeOperands`` bundle of ``prepare``.
        messages: ``(P, N, K)`` window.
        is_output: ``(P, N)`` bool output-row mask.
        valid: ``(P, N, K)`` bool live-slot mask.

    Returns:
        ``(P, N)`` float32 host numpy node values.
    """
    return run_method(
        node, "forward", attrs, messages, is_output, valid, state=State()
    )


# --------------------------------------------------------------------------- #
# the shared payload: one live slot, a mixture, and all-dead rows in one batch
# --------------------------------------------------------------------------- #

#: ``(P, N, K) = (3, 3, 3)`` incoming-message window.  Dead slots carry ``NaN`` so that a
#: kernel which reduced the raw window instead of the masked one would be caught.
#: Row by row: ``(0,0)`` one live slot, ``(0,1)`` one live slot, ``(0,2)`` two live slots,
#: ``(1,0)`` all dead, ``(1,1)`` two live slots, ``(1,2)`` one live slot, ``(2,0)`` two live
#: slots, ``(2,1)`` all dead, ``(2,2)`` all dead.
MESSAGES = np.array(
    [
        [[np.nan, 1.5, np.nan], [np.nan, np.nan, -2.0], [-1.0, np.nan, 4.0]],
        [[np.nan, np.nan, np.nan], [2.0, 3.0, np.nan], [-0.5, np.nan, np.nan]],
        [[0.25, np.nan, -3.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    ],
    dtype=np.float32,
)

#: The live-slot mask: a slot is live iff its message is not ``NaN`` (the mask is data
#: here, never read off the implementation).
VALID = ~np.isnan(MESSAGES)

#: Output rows ``(P, N)``: ``(1,1)`` and ``(2,0)`` are read-out rows with live slots and
#: ``(2,1)`` is a read-out row with no live slot at all (it must still be exactly ``0.0``).
IS_OUTPUT = np.array(
    [[False, False, False], [False, True, False], [True, True, False]], dtype=bool
)

#: A ``(P, N, K) = (2, 3, 3)`` window with **exactly one** live slot per row: ``sum`` and
#: ``mean`` must agree on it element for element.
ONE_LIVE_MESSAGES = np.array(
    [
        [[1.5, np.nan, np.nan], [np.nan, -2.0, np.nan], [np.nan, np.nan, 0.75]],
        [[np.nan, -0.5, np.nan], [3.0, np.nan, np.nan], [np.nan, 2.5, np.nan]],
    ],
    dtype=np.float32,
)

#: Live mask of :data:`ONE_LIVE_MESSAGES`.
ONE_LIVE_VALID = ~np.isnan(ONE_LIVE_MESSAGES)

#: Output rows of :data:`ONE_LIVE_MESSAGES`: one activated row and one bare-aggregate row
#: per individual.
ONE_LIVE_OUTPUT = np.array([[False, True, False], [True, False, False]], dtype=bool)


# --------------------------------------------------------------------------- #
# HyperNEATNode: declared contract
# --------------------------------------------------------------------------- #


def test_node_declares_the_documented_option_table():
    """``HyperNEATNode`` declares the identity key and the two single-option int columns.

    The table is the gene's *payload layout*, so it is asserted as literals restated from
    the documentation: one ``int32`` key column (``index``) and two ``float32`` option
    columns (``aggregation`` then ``activation``), each with the single legal index ``0``
    and ``replace_rate == 0.1``.
    """
    node = HyperNEATNode()

    assert HyperNEATNode.fixed_attrs == ("index",), (
        "a substrate node's only key column is its identity index, got "
        f"{HyperNEATNode.fixed_attrs!r}"
    )
    assert node.key_width == 1, (
        f"key_width must be 1 (fixed_attrs == ('index',)), got {node.key_width}"
    )
    assert node.attr_width == 2, (
        f"attr_width must be 2 (the two option columns), got {node.attr_width}"
    )

    specs = node.attr_specs
    assert [spec.name for spec in specs] == ["aggregation", "activation"], (
        "the payload column order is aggregation then activation, got "
        f"{[spec.name for spec in specs]!r}"
    )
    assert [spec.kind for spec in specs] == ["int", "int"], (
        f"both option columns are int specs, got {[spec.kind for spec in specs]!r}"
    )
    assert [tuple(spec.options) for spec in specs] == [(0,), (0,)], (
        "each option column admits exactly one index, 0, so a substrate row can never "
        f"select a callable the gene was not built for; got {[spec.options for spec in specs]!r}"
    )
    assert [spec.replace_rate for spec in specs] == [0.1, 0.1], (
        "both option specs declare replace_rate == 0.1, got "
        f"{[spec.replace_rate for spec in specs]!r}"
    )

    def layout(table):
        return [(spec.name, spec.kind, spec.options, spec.replace_rate) for spec in table]

    assert layout(specs) == layout(HyperNEATNode.attr_specs), (
        "the table __init__ installs per instance must have the class-level layout, got "
        f"{layout(specs)!r} against {layout(HyperNEATNode.attr_specs)!r}"
    )

    assert node.aggregation is AGG.sum, (
        f"the default aggregation is AGG.sum, got {node.aggregation!r}"
    )
    assert node.activation is ACT.sigmoid, (
        f"the default activation is ACT.sigmoid, got {node.activation!r}"
    )
    assert tuple(node.aggregation_options) == (AGG.sum,) and len(
        node.aggregation_options
    ) == 1, (
        "aggregation_options must be the single-element tuple (aggregation,), got "
        f"{node.aggregation_options!r}"
    )
    assert tuple(node.activation_options) == (ACT.sigmoid,) and len(
        node.activation_options
    ) == 1, (
        "activation_options must be the single-element tuple (activation,), got "
        f"{node.activation_options!r}"
    )


def test_node_rejects_a_non_callable_option():
    """A non-callable ``aggregation``/``activation`` is a host-side ``ConfigError``.

    The option cascade would otherwise fail only once a graph is traced, so the constructor
    validates both arguments and names the offending one.
    """
    with pytest.raises(ConfigError, match="aggregation must be a callable"):
        HyperNEATNode(aggregation=0)
    with pytest.raises(ConfigError, match="activation must be a callable"):
        HyperNEATNode(activation="sigmoid")

    # A callable that is not the default is accepted and becomes the only option.
    node = HyperNEATNode(aggregation=AGG.max, activation=ACT.tanh)
    assert node.aggregation_options[0] is AGG.max
    assert node.activation_options[0] is ACT.tanh


# --------------------------------------------------------------------------- #
# HyperNEATNode: combine and forward
# --------------------------------------------------------------------------- #


def test_node_combine_returns_the_aggregated_value_unchanged():
    """``combine`` is the identity: a substrate node has no bias/response affine map.

    ``DefaultNode`` adds ``bias + response * aggregated`` and ``BiasNode`` adds ``bias``;
    a substrate node has neither column, so the aggregated (and even a ``NaN``) value must
    come back untouched.
    """
    node = HyperNEATNode()
    aggregated = np.array(
        [[0.0, -1.5, 3.25], [np.nan, -0.0, 1.0e6]], dtype=np.float32
    )

    got = run_method(
        node, "combine", _node_attrs(2, 3), aggregated, state=State()
    )

    assert got.shape == aggregated.shape, (
        f"combine must preserve the (P, N) shape of the aggregate, got {got.shape} for "
        f"{aggregated.shape}"
    )
    np.testing.assert_allclose(
        got, aggregated, rtol=0, atol=0, equal_nan=True,
        err_msg=(
            "combine must return the aggregated value unchanged (no bias/response map), so "
            "it can only differ from the hand-written array above by not being the identity"
        ),
    )
    assert np.isnan(got[1, 0]), (
        "combine must not sanitise a NaN aggregate: a substrate node has no op there, got "
        f"{got[1, 0]!r}"
    )


def test_node_forward_sums_the_live_slots_then_activates_non_output_rows():
    """The documented sum aggregation, per individual, on the mixed live/dead batch.

    The oracle is :func:`_node_oracle`, a per-individual Python loop over the hand-written
    window; the five anchors below are the same values derived by hand in the comments.
    """
    node = HyperNEATNode(aggregation=AGG.sum, activation=ACT.sigmoid)
    got = _run_forward(node, _node_attrs(3, 3), MESSAGES, IS_OUTPUT, VALID)

    assert got.shape == (3, 3), (
        f"forward must return one (P, N) value per node row, got {got.shape}"
    )
    assert got.dtype == np.float32, (
        f"forward must return float32 node values, got {got.dtype}"
    )
    np.testing.assert_allclose(
        got,
        _node_oracle(MESSAGES, IS_OUTPUT, VALID, "sum"),
        rtol=0,
        atol=0,
        err_msg=(
            "every individual's node row must be the masked sum of its live slots (a dead "
            "slot contributes exactly 0.0, never its NaN) followed by sigmoid on non-output "
            "rows; the oracle is the per-individual loop of _node_oracle"
        ),
    )

    anchors = {
        # (p, n): (hand aggregate, hand value after sigmoid)
        (0, 0): (1.5, 0.8175745),  # one live slot: 1.5 -> sigmoid(1.5)
        (0, 1): (-2.0, 0.119202934),  # one live slot: -2.0 -> sigmoid(-2.0)
        (0, 2): (3.0, 0.95257413),  # mixture: -1.0 + 4.0 -> sigmoid(3.0)
        (1, 2): (-0.5, 0.37754068),  # one live slot: -0.5 -> sigmoid(-0.5)
        (2, 0): (-2.75, -2.75),  # mixture on an OUTPUT row: 0.25 - 3.0, unactivated
    }
    for (p, n), (aggregate, activated) in anchors.items():
        assert got[p, n] == np.float32(activated), (
            f"node ({p}, {n}) must aggregate to {aggregate} and activate to "
            f"{np.float32(activated)!r}, got {got[p, n]!r}"
        )


def test_node_forward_returns_exactly_zero_for_an_all_dead_window():
    """A row with no live slot aggregates to exactly ``0.0`` (never ``NaN``).

    An output row is that bare ``0.0``; a non-output row is ``sigmoid(0.0) == 0.5``.  The
    row's slots all carry ``NaN`` in the window, so this also proves the mask -- not the
    data -- is what neutralises a dead slot.
    """
    node = HyperNEATNode(aggregation=AGG.sum, activation=ACT.sigmoid)
    got = _run_forward(node, _node_attrs(3, 3), MESSAGES, IS_OUTPUT, VALID)

    dead_rows = ~VALID.any(axis=-1)
    assert dead_rows.tolist() == [
        [False, False, False],
        [True, False, False],
        [False, True, True],
    ], f"the payload's all-dead rows are not the intended three, got {dead_rows.tolist()}"

    for p, n in zip(*np.nonzero(dead_rows)):
        expected = np.float32(0.0) if IS_OUTPUT[p, n] else np.float32(0.5)
        why = "output row: unactivated" if IS_OUTPUT[p, n] else "sigmoid(0.0)"
        assert got[p, n] == expected, (
            f"node ({p}, {n}) has no live slot, so it must aggregate to exactly 0.0 and "
            f"be {expected!r} ({why}), got {got[p, n]!r}"
        )
    assert not np.isnan(got).any(), (
        "an all-dead window must never produce NaN: the mask neutralises every slot, got "
        f"{got!r}"
    )


def test_node_forward_does_not_activate_output_rows():
    """An output row equals its aggregate exactly; every other row is activated.

    ``BaseNode.prepare`` forces an output row's activation id to the ``-1`` identity
    sentinel, so the read-out is left for the genome-level ``output_transform``.
    """
    node = HyperNEATNode(aggregation=AGG.sum, activation=ACT.sigmoid)
    got = _run_forward(node, _node_attrs(3, 3), MESSAGES, IS_OUTPUT, VALID)
    aggregated = _aggregates(MESSAGES, VALID, "sum")

    for p, n in zip(*np.nonzero(IS_OUTPUT)):
        assert got[p, n] == aggregated[p, n], (
            f"output row ({p}, {n}) must be its bare aggregate "
            f"{aggregated[p, n]!r}, got {got[p, n]!r}"
        )
        assert got[p, n] != _sigmoid(aggregated[p, n]), (
            f"output row ({p}, {n}) must NOT be activated: sigmoid({aggregated[p, n]!r}) is "
            f"{_sigmoid(aggregated[p, n])!r}, got {got[p, n]!r}"
        )

    for p, n in zip(*np.nonzero(~IS_OUTPUT)):
        assert got[p, n] == _sigmoid(aggregated[p, n]), (
            f"non-output row ({p}, {n}) must be sigmoid(aggregate) == "
            f"{_sigmoid(aggregated[p, n])!r}, got {got[p, n]!r}"
        )


def test_node_forward_mean_divides_by_the_live_slot_count():
    """``AGG.mean`` is a supported substrate aggregation: sum over ``max(count, 1)``.

    Hand values: ``(0, 2)`` is ``(-1.0 + 4.0) / 2 == 1.5``; ``(1, 1)`` is the bare output
    aggregate ``(2.0 + 3.0) / 2 == 2.5``; ``(2, 0)`` is the bare output aggregate
    ``(0.25 - 3.0) / 2 == -1.375``; the all-dead row ``(1, 0)`` is
    ``0.0 / max(0, 1) == 0.0``, i.e. ``sigmoid(0.0) == 0.5``.
    """
    node = HyperNEATNode(aggregation=AGG.mean, activation=ACT.sigmoid)
    got = _run_forward(node, _node_attrs(3, 3), MESSAGES, IS_OUTPUT, VALID)

    np.testing.assert_allclose(
        got,
        _node_oracle(MESSAGES, IS_OUTPUT, VALID, "mean"),
        rtol=0,
        atol=0,
        err_msg=(
            "AGG.mean must divide the masked sum by max(live count, 1); the oracle is the "
            "per-individual loop of _node_oracle"
        ),
    )

    anchors = {
        (0, 2): 0.8175745,  # mean of (-1.0, 4.0) == 1.5 -> sigmoid(1.5)
        (1, 1): 2.5,  # output row: mean of (2.0, 3.0), unactivated
        (2, 0): -1.375,  # output row: mean of (0.25, -3.0), unactivated
        (1, 0): 0.5,  # all dead: 0.0 / max(0, 1) -> sigmoid(0.0)
    }
    for (p, n), activated in anchors.items():
        assert got[p, n] == np.float32(activated), (
            f"AGG.mean node ({p}, {n}) must be {np.float32(activated)!r}, got {got[p, n]!r}"
        )


def test_sum_and_mean_agree_when_every_node_has_one_live_slot():
    """With one live slot per row, the ``AGG.sum`` and ``AGG.mean`` batches are identical.

    ``sum`` divides by nothing and ``mean`` by ``max(1, 1) == 1``, so the two aggregations
    -- the two self-neutral options the substrate supports -- must produce the same
    ``(P, N)`` batch bit for bit, and both must match the one-live-slot oracle.
    """
    attrs = _node_attrs(2, 3)
    sum_out = _run_forward(
        HyperNEATNode(aggregation=AGG.sum, activation=ACT.sigmoid),
        attrs,
        ONE_LIVE_MESSAGES,
        ONE_LIVE_OUTPUT,
        ONE_LIVE_VALID,
    )
    mean_out = _run_forward(
        HyperNEATNode(aggregation=AGG.mean, activation=ACT.sigmoid),
        attrs,
        ONE_LIVE_MESSAGES,
        ONE_LIVE_OUTPUT,
        ONE_LIVE_VALID,
    )
    expected = _node_oracle(
        ONE_LIVE_MESSAGES, ONE_LIVE_OUTPUT, ONE_LIVE_VALID, "sum"
    )

    assert (ONE_LIVE_VALID.sum(axis=-1) == 1).all(), (
        "the payload must give every node exactly one live slot, got live counts "
        f"{ONE_LIVE_VALID.sum(axis=-1).tolist()}"
    )
    np.testing.assert_array_equal(
        sum_out,
        mean_out,
        err_msg=(
            "with one live slot per node, sum and mean are the same reduction (mean divides "
            "by max(1, 1)), so the two batches must be bit-identical"
        ),
    )
    np.testing.assert_allclose(
        sum_out,
        expected,
        rtol=0,
        atol=0,
        err_msg=(
            "the one-live-slot batch must equal the masked sum followed by the activation "
            "rule, per the per-individual loop of _node_oracle"
        ),
    )


def test_node_forward_accepts_the_prepared_operand_bundle():
    """The ``NodeOperands`` spelling of ``attrs`` gives the same values as the raw rows.

    ``BaseGenome._node_step`` passes the bundle of ``BaseNode.prepare``, so the two
    spellings must be interchangeable -- including the output-row rule the bundle carries
    in its activation masks.
    """
    node = HyperNEATNode(aggregation=AGG.sum, activation=ACT.sigmoid)
    attrs = _node_attrs(3, 3)
    bundle = run_method(node, "prepare", attrs, IS_OUTPUT)

    raw = _run_forward(node, attrs, MESSAGES, IS_OUTPUT, VALID)
    prepared = _run_forward(node, bundle, MESSAGES, IS_OUTPUT, VALID)

    np.testing.assert_array_equal(
        prepared,
        raw,
        err_msg=(
            "forward on the NodeOperands bundle must equal forward on the raw (P, N, 2) "
            "rows it was prepared from"
        ),
    )
    np.testing.assert_allclose(
        prepared,
        _node_oracle(MESSAGES, IS_OUTPUT, VALID, "sum"),
        rtol=0,
        atol=0,
        err_msg=(
            "the bundle spelling must still be the documented aggregate-then-activate "
            "algebra of _node_oracle"
        ),
    )


# --------------------------------------------------------------------------- #
# HyperNEATNode: host-only presentation
# --------------------------------------------------------------------------- #


def test_node_to_dict_and_repr_read_the_configured_option_names():
    """``to_dict``/``repr`` resolve the option ids through the configured callables.

    ``AGG.max``/``ACT.tanh`` are used precisely so the rendered names cannot be the
    constructor defaults hard-coded somewhere: the row's id ``0`` must resolve to whatever
    the gene was built with.
    """
    node = HyperNEATNode(aggregation=AGG.max, activation=ACT.tanh)
    state = State()
    key_row = np.array([3], dtype=np.int32)
    attr_row = np.array([0.0, 0.0], dtype=np.float32)

    assert node.to_dict(state, key_row, attr_row) == {
        "idx": 3,
        "agg": "max",
        "act": "tanh",
    }, (
        "to_dict must name the configured options (AGG.max/ACT.tanh) for id 0, got "
        f"{node.to_dict(state, key_row, attr_row)!r}"
    )

    line = node.repr(state, key_row, attr_row)
    assert line == "HyperNEATNode(idx=3, aggregation=max, activation=tanh)", (
        f"repr must be the documented one-line summary, got {line!r}"
    )
    assert "\n" not in line, f"repr must be a single line, got {line!r}"

    default_node = HyperNEATNode()
    assert default_node.to_dict(state, key_row, attr_row) == {
        "idx": 3,
        "agg": "sum",
        "act": "sigmoid",
    }, (
        "the documented default pair must render as sum/sigmoid, got "
        f"{default_node.to_dict(state, key_row, attr_row)!r}"
    )


def test_node_to_dict_resolves_the_identity_sentinel_and_rejects_a_foreign_id():
    """The host reader maps the ``-1`` act id to ``"identity"`` and rejects an unknown id.

    Both rules are documented on ``to_dict``: ``-1`` is the identity sentinel rather than an
    option, and an id outside the configured table raises ``ConfigError`` instead of being
    clamped to a real option.
    """
    node = HyperNEATNode(aggregation=AGG.sum, activation=ACT.sigmoid)
    state = State()
    key_row = np.array([1], dtype=np.int32)

    identity = node.to_dict(state, key_row, np.array([0.0, -1.0], dtype=np.float32))
    assert identity["act"] == "identity", (
        f"activation id -1 is the identity sentinel and must render as 'identity', got "
        f"{identity['act']!r}"
    )
    assert identity["agg"] == "sum", (
        f"the aggregation column must still resolve to 'sum', got {identity['agg']!r}"
    )

    with pytest.raises(ConfigError, match="aggregation id 1 is outside"):
        node.to_dict(state, key_row, np.array([1.0, 0.0], dtype=np.float32))

    with pytest.raises(ConfigError, match="expected 2 attribute columns"):
        node.to_dict(state, key_row, np.array([0.0], dtype=np.float32))


# --------------------------------------------------------------------------- #
# HyperNEATConn: declared contract
# --------------------------------------------------------------------------- #


def test_conn_declares_the_two_key_columns_and_the_weight_spec():
    """``HyperNEATConn`` carries the two endpoint keys and exactly one float ``weight``.

    The five distribution fields of the spec are restated from the documentation, so a
    change in them fails here rather than silently changing a repaint distribution nothing
    else reads.
    """
    gene = HyperNEATConn()

    assert gene.fixed_attrs == ("input_index", "output_index"), (
        "a connection's key columns are its source and destination node keys, got "
        f"{gene.fixed_attrs!r}"
    )
    assert gene.key_width == 2, (
        f"HyperNEATConn.key_width must be 2 (the two endpoint columns), got {gene.key_width}"
    )
    assert gene.attr_width == 1, (
        f"HyperNEATConn.attr_width must be 1 (the weight alone), got {gene.attr_width}"
    )
    assert gene.uses_historical_marker is False, (
        "HyperNEATConn aligns genes by (in, out) alone and declares no historical_marker "
        f"key column, got {gene.uses_historical_marker!r}"
    )

    specs = gene.attr_specs
    assert len(specs) == 1, f"exactly one attribute spec is declared, got {specs!r}"
    spec = specs[0]
    assert (spec.name, spec.kind) == ("weight", "float"), (
        f"the single attribute column is the float 'weight', got {(spec.name, spec.kind)!r}"
    )
    assert (
        spec.init_mean,
        spec.init_std,
        spec.mutate_power,
        spec.mutate_rate,
        spec.replace_rate,
    ) == (0.0, 1.0, 0.15, 0.8, 0.1), (
        "the weight spec must keep the documented init mean/std and the mutation/replace "
        "rates, got a different five-tuple"
    )

    names = [item.name for item in gene.attr_specs]
    assert gene.weight_index == names.index("weight"), (
        "weight_index must be the position of 'weight' in attr_specs, never a literal: "
        f"attr_specs names are {names!r}, weight_index is {gene.weight_index}"
    )
    assert gene.weight_index == 0, (
        f"a one-spec table puts 'weight' at index 0, got {gene.weight_index}"
    )


def test_conn_weight_index_follows_the_spec_table_not_an_offset():
    """Inserting a column in front of ``weight`` moves ``weight_index`` to ``1``.

    The shipped gene has one attribute column, so the "resolved by name" contract is only
    observable on a table with a column before the weight: the very same ``__init__``
    resolves ``1`` there, and ``forward``/``to_dict`` then read that column.
    """

    class _TwoColumnConn(HyperNEATConn):
        """A weight column at position 1, preceded by an unrelated float column."""

        attr_specs = (
            GeneAttr(name="bias", kind="float"),
            GeneAttr(name="weight", kind="float", mutate_rate=0.8, replace_rate=0.1),
        )

    gene = _TwoColumnConn()
    names = [spec.name for spec in gene.attr_specs]
    assert gene.attr_width == 2
    assert gene.weight_index == names.index("weight") == 1, (
        "weight_index must follow the spec table (1 here), not the shipped gene's 0, got "
        f"{gene.weight_index} for names {names!r}"
    )

    # ``weight`` in column 1, the whole ``bias`` column NaN: the read is by name.
    attrs = np.array(
        [[[np.nan, 2.0], [np.nan, -0.5], [np.nan, 0.0]]], dtype=np.float32
    )
    source = np.array([[1.0, 2.0, -3.0]], dtype=np.float32)

    got = run_method(gene, "forward", attrs, source, state=State())
    np.testing.assert_allclose(
        got,
        np.array([[2.0, -1.0, -0.0]], dtype=np.float32),
        rtol=0,
        atol=0,
        err_msg=(
            "forward must multiply by the column named 'weight' (column 1 here: "
            "[2.0, -0.5, 0.0]) and ignore the all-NaN column in front of it"
        ),
    )

    bundle = run_method(gene, "prepare", attrs)
    prepared = run_method(gene, "forward", bundle, source, state=State())
    np.testing.assert_array_equal(
        prepared,
        got,
        err_msg=(
            "on a two-column table the bundle spelling must read the same named column as "
            "the raw spelling"
        ),
    )

    conn_dict = gene.to_dict(
        State(), np.array([1, 4], dtype=np.int32), np.array([7.5, 0.5], dtype=np.float32)
    )
    assert conn_dict == {"in": 1, "out": 4, "weight": 0.5}, (
        "to_dict must read the weight at weight_index (column 1) and not at offset 0, got "
        f"{conn_dict!r}"
    )


# --------------------------------------------------------------------------- #
# HyperNEATConn: forward
# --------------------------------------------------------------------------- #

#: ``(P, K)`` source values of one destination node's incoming window (``P = 2``, ``K = 3``).
SOURCE = np.array([[1.0, 2.0, -3.0], [0.5, -0.25, 4.0]], dtype=np.float32)

#: The three edge weights, one per slot, exactly representable in float32.
WEIGHTS = np.array([2.0, -0.5, 0.0], dtype=np.float32)

#: The hand-computed products ``SOURCE * WEIGHTS``.
PRODUCTS = np.array([[2.0, -1.0, -0.0], [1.0, 0.125, 0.0]], dtype=np.float32)


def _conn_attrs(weights) -> np.ndarray:
    """``(P, K, 1)`` attribute rows of one destination node, all rows carrying ``weights``.

    Args:
        weights: ``(K,)`` float32 weight row.

    Returns:
        ``(P, K, 1)`` float32 attribute batch.
    """
    row = np.asarray(weights, dtype=np.float32).reshape(1, -1, 1)
    return np.broadcast_to(row, (SOURCE.shape[0], row.shape[1], 1)).copy()


def test_conn_forward_multiplies_the_raw_attribute_rows_by_the_source_values():
    """The raw ``(..., K, CA)`` spelling: one multiply per slot, no reduction.

    Hand products ``[2.0, -1.0, 0.0]`` for the first individual and ``[1.0, 0.125, 0.0]``
    for the second (``0.5 * 2.0``, ``-0.25 * -0.5``, ``4.0 * 0.0``).
    """
    gene = HyperNEATConn()
    got = run_method(gene, "forward", _conn_attrs(WEIGHTS), SOURCE, state=State())

    assert got.shape == SOURCE.shape, (
        f"forward must return one message per (individual, slot): expected {SOURCE.shape}, "
        f"got {got.shape}"
    )
    assert got.dtype == np.float32, f"messages must be float32, got {got.dtype}"
    np.testing.assert_allclose(
        got,
        PRODUCTS,
        rtol=0,
        atol=0,
        err_msg=(
            "each slot must be source_values * weight with weights [2.0, -0.5, 0.0]; the "
            "expected products are [2.0, -1.0, 0.0] and [1.0, 0.125, 0.0]"
        ),
    )
    assert got[0, 0] == np.float32(2.0), f"1.0 * 2.0 must be 2.0, got {got[0, 0]!r}"
    assert got[1, 1] == np.float32(0.125), (
        f"-0.25 * -0.5 must be 0.125 exactly, got {got[1, 1]!r}"
    )
    assert got[1, 2] == np.float32(0.0), f"4.0 * 0.0 must be 0.0, got {got[1, 2]!r}"


def test_conn_forward_accepts_the_prepared_operand_bundle():
    """The ``ConnOperands`` spelling the genome passes gives the same messages as raw rows.

    ``BaseGenome._hit_conn_attrs`` builds the bundle once per forward pass and hands
    ``hit_conn_attrs.conn`` to the gene, so the two spellings must agree on clean data.
    """
    gene = HyperNEATConn()
    attrs = _conn_attrs(WEIGHTS)
    bundle = run_method(gene, "prepare", attrs)

    assert hasattr(bundle, "columns") and len(bundle.columns) == 1, (
        "prepare must return a ConnOperands bundle with one (K,) column per attribute spec, "
        f"got {bundle!r}"
    )
    np.testing.assert_array_equal(
        bundle.columns[0],
        np.broadcast_to(WEIGHTS, (SOURCE.shape[0], WEIGHTS.size)),
        err_msg="the bundle's single weight column must be the payload's weight rows",
    )

    prepared = run_method(gene, "forward", bundle, SOURCE, state=State())
    np.testing.assert_array_equal(
        prepared,
        PRODUCTS,
        err_msg=(
            "the bundle spelling must produce the same hand-computed products [2.0, -1.0, "
            "0.0] / [1.0, 0.125, 0.0] as the raw spelling"
        ),
    )
    raw = run_method(gene, "forward", attrs, SOURCE, state=State())
    np.testing.assert_array_equal(
        prepared,
        raw,
        err_msg="the two documented attrs spellings must be interchangeable on clean data",
    )


def test_conn_reads_the_bundle_s_nan_cleaned_weight_column():
    """A ``NaN`` weight row is an absent row: the bundle cleans it to ``0``, raw rows do not.

    ``BaseConn.prepare`` replaces a ``NaN`` attribute with ``0`` (the padding policy), and
    ``HyperNEATConn.forward`` reads that cleaned column when it is given the bundle.  The
    cleaning is per cell, so a ``NaN`` neighbour never disturbs a real weight.
    """
    gene = HyperNEATConn()
    # weights: individual 0 [2.0, NaN, 0.0]; individual 1 [NaN, -0.5, 3.0]
    attrs = np.array(
        [[[2.0], [np.nan], [0.0]], [[np.nan], [-0.5], [3.0]]], dtype=np.float32
    )

    raw = run_method(gene, "forward", attrs, SOURCE, state=State())
    assert np.isnan(raw[0, 1]), (
        "the raw (K, 1) spelling is not cleaned by the kernel, so a NaN weight row must "
        f"keep its NaN message, got {raw[0, 1]!r}"
    )
    assert raw[0, 0] == np.float32(2.0) and raw[1, 1] == np.float32(0.125), (
        "a NaN weight cell must not disturb its neighbours: 1.0 * 2.0 == 2.0 and "
        f"-0.25 * -0.5 == 0.125, got {raw[0, 0]!r} / {raw[1, 1]!r}"
    )

    bundle = run_method(gene, "prepare", attrs)
    np.testing.assert_array_equal(
        bundle.columns[0],
        np.array([[2.0, 0.0, 0.0], [0.0, -0.5, 3.0]], dtype=np.float32),
        err_msg=(
            "prepare must clean each NaN weight cell to 0 (the absent-row policy) and leave "
            "the real cells alone"
        ),
    )

    prepared = run_method(gene, "forward", bundle, SOURCE, state=State())
    np.testing.assert_allclose(
        prepared,
        np.array([[2.0, 0.0, -0.0], [0.0, 0.125, 12.0]], dtype=np.float32),
        rtol=0,
        atol=0,
        err_msg=(
            "with the cleaned column the messages are [2.0, 0.0, 0.0] and "
            "[0.0, 0.125, 12.0] (NaN weights contribute 0, the real weight 3.0 gives "
            "4.0 * 3.0 == 12.0)"
        ),
    )


def test_conn_does_no_nan_guard_and_the_node_gene_owns_the_masking():
    """A ``NaN`` source value propagates through the multiply; ``valid_mask`` is the mask.

    ``HyperNEATConn.forward`` has no mask argument and no ``isnan`` of its own: masking has
    a single owner, the destination node's ``valid_mask``, which replaces the dead slot with
    the aggregation's neutral value before reducing.
    """
    gene = HyperNEATConn()
    source = np.array([[np.nan, 2.0, -3.0]], dtype=np.float32)
    weights = np.array([[[2.0], [-0.5], [0.0]]], dtype=np.float32)

    messages = run_method(gene, "forward", weights, source, state=State())
    assert np.isnan(messages[0, 0]), (
        "the conn kernel must not guard a NaN source value (masking belongs to the node "
        f"gene): 2.0 * NaN must stay NaN, got {messages[0, 0]!r}"
    )
    assert messages[0, 1] == np.float32(-1.0), (
        f"2.0 * -0.5 must be -1.0, got {messages[0, 1]!r}"
    )

    # The single owner of masking: the node gene over the same window, output row, with the
    # NaN slot marked dead -> 2.0 * -0.5 == -1.0 and 0.0 * -3.0 == 0.
    node = HyperNEATNode(aggregation=AGG.sum, activation=ACT.sigmoid)
    value = _run_forward(
        node,
        _node_attrs(1, 1),
        messages.reshape(1, 1, 3),
        np.array([[True]], dtype=bool),
        np.array([[[False, True, True]]], dtype=bool),
    )
    assert value[0, 0] == np.float32(-1.0), (
        "the node gene's valid_mask must neutralise the dead NaN slot, leaving the "
        f"aggregate -1.0 + 0.0, got {value[0, 0]!r}"
    )


# --------------------------------------------------------------------------- #
# HyperNEATConn: host-only presentation
# --------------------------------------------------------------------------- #


def test_conn_to_dict_and_repr_render_the_key_row_and_the_weight():
    """``to_dict`` is ``{"in", "out", "weight"}`` and ``repr`` its one-line form.

    Both read the key columns through ``_key_int`` and the weight at ``weight_index``, so
    the rendered numbers are the row's own values.
    """
    gene = HyperNEATConn()
    state = State()
    key_row = np.array([1, 4], dtype=np.int32)
    attr_row = np.array([0.5], dtype=np.float32)

    assert gene.to_dict(state, key_row, attr_row) == {
        "in": 1,
        "out": 4,
        "weight": 0.5,
    }, f"to_dict must render in/out/weight, got {gene.to_dict(state, key_row, attr_row)!r}"

    line = gene.repr(state, key_row, attr_row)
    assert line == "HyperNEATConn(in: 1, out: 4, weight: 0.50)", (
        f"repr must be the documented one-line summary, got {line!r}"
    )
    assert "\n" not in line, f"repr must be a single line, got {line!r}"

    assert gene.to_dict(state, key_row, np.array([-0.5], dtype=np.float32))["weight"] == (
        -0.5
    ), "a negative weight must be rendered with its sign"
    assert gene.repr(
        state, key_row, np.array([2.5], dtype=np.float32), precision=3
    ) == "HyperNEATConn(in: 1, out: 4, weight: 2.500)", (
        "the precision argument must reach the rendered weight"
    )
