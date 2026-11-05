"""Unit tests for the HyperNEAT substrate geometry — plain host NumPy, no graph.

What this module proves
-----------------------
A substrate is the *geometry* of a HyperNEAT phenotype: a neuron layout becomes the
fixed topology the CPPN is queried over, one query row per phenotype connection.
Everything here is host-side and runs outside any trace, so every assertion is a
plain NumPy comparison: no ``evoxc`` graph, no backend, no optional dependency.
Covered, in test order: the shared helpers of ``base.py`` (key blocks, the *source
axis slowest* row order of :func:`cartesian_product`, the inclusive ``x`` spread of
:func:`layer_coors`, dtypes); the ``DefaultSubstrate`` container (normalised copies,
legacy ``(n, 1)`` node column, array-free repr, re-runnable ``validate``); the
``FullSubstrate`` mesh (``input→hidden``, ``hidden→hidden`` self-loops included,
``hidden→output``, in that row-block order, hidden keys *after* the output keys);
the ``MLPSubstrate`` grid (layer pair → source → target order, the documented ``y``
interpolation, the ``len(layers) == 2`` case); and every branch of the
constructor-time validation matrix.

Oracles used here (never the implementation)
--------------------------------------------
* **Hand-computed arrays.** Every expected ``query_coors``/``conn_index``/``nodes``
  is written out literally from the documented key blocks and row order, in
  coordinates exact in ``float32``, so the comparisons are exact, not tolerance-based.
* **A hand-written key → coordinate map.** :func:`_endpoint_rows` rebuilds
  ``query_coors`` from ``conn_index`` with a Python loop — the alignment rule
  restated without the ``np.repeat``/``np.tile`` it implements.
* **Closed forms** (``si*sh + sh*sh + sh*so``, ``sum(layers[i]*layers[i+1])``,
  ``np.sort(nodes) == arange(n)``) and ``match=`` substrings of the documented
  messages, distinctive enough that no neighbouring branch can satisfy a test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorneat.algorithm.hyperneat.substrate import (
    BaseSubstrate,
    DefaultSubstrate,
    FullSubstrate,
    MLPSubstrate,
    as_coor_array,
    cartesian_product,
    check_same_coordinate_dim,
    layer_coors,
    node_key_blocks,
)
from tensorneat.algorithm.hyperneat.substrate.full import (
    analysis_substrate as full_analysis_substrate,
)
from tensorneat.algorithm.hyperneat.substrate.mlp import (
    analysis_substrate as mlp_analysis_substrate,
)
from tensorneat.common.errors import TensorNEATError

# Hand-written topologies.  ``DefaultSubstrate``: inputs 0/1, output 2, hidden 3.
DEFAULT_NODES = np.array([0, 1, 2, 3], dtype=np.int32)
DEFAULT_CONN_INDEX = np.array([[0, 3], [3, 2]], dtype=np.int32)
DEFAULT_QUERY_COORS = np.array(
    [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0]], dtype=np.float32
)
#: Three query rows for the row-count/endpoint cases (one more than conn_index).
THREE_ROW_QUERIES = np.zeros((3, 4), dtype=np.float32)
#: Endpoint key 9 is not a node key; the second one repeats ``(0, 3)``.
UNKNOWN_ENDPOINT_CONN = np.array([[0, 3], [3, 2], [0, 9]], dtype=np.int32)
DUPLICATE_ROW_CONN = np.array([[0, 3], [3, 2], [0, 3]], dtype=np.int32)
# ``FullSubstrate`` tiny case: si=2, sh=1, so=1 -> 4 nodes, 4 connections.
FULL_INPUTS = ((0.0, 0.0), (1.0, 0.0))
FULL_HIDDEN = ((0.0, 1.0),)
FULL_OUTPUTS = ((0.0, 2.0),)
FULL_NODES = np.array([0, 1, 2, 3], dtype=np.int32)  # inputs 0/1, output 2, hidden 3
# Block 1 input->hidden (0,3),(1,3); block 2 hidden->hidden self-loop (3,3);
# block 3 hidden->output (3,2).  Source varies slowest inside a block.
FULL_CONN_INDEX = np.array([[0, 3], [1, 3], [3, 3], [3, 2]], dtype=np.int32)
# [coor(src), coor(dst)] per row: (0,0)+(0,1), (1,0)+(0,1), (0,1)+(0,1), (0,1)+(0,2).
FULL_QUERY_COORS = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
    ],
    dtype=np.float32,
)
# ``MLPSubstrate`` grid [3, 2, 1] in (x_min, x_max, y_min, y_max) = (0, 2, 0, 1):
# inputs 0..2, output 3, hidden 4/5 (hidden block last).
MLP_LAYERS = (3, 2, 1)
MLP_COOR_RANGE = (0.0, 2.0, 0.0, 1.0)
MLP_NODES = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
# Layer pair 0->1 (3x2 rows, source slowest), then 1->2 (2x1 rows).
MLP_CONN_INDEX = np.array(
    [[0, 4], [0, 5], [1, 4], [1, 5], [2, 4], [2, 5], [4, 3], [5, 3]], dtype=np.int32
)
# Layer 0 over x in [0,2] (3 neurons -> 0,1,2), layer 1 at the interpolated y=0.5
# (2 neurons -> 0,2), layer -1 at y=1 (1 neuron -> centred at (0+2)/2 = 1).
MLP_LAYER_COORDS = (
    ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
    ((0.0, 0.5), (2.0, 0.5)),
    ((1.0, 1.0),),
)
MLP_QUERY_COORS = np.array(
    [
        [0.0, 0.0, 0.0, 0.5],
        [0.0, 0.0, 2.0, 0.5],
        [1.0, 0.0, 0.0, 0.5],
        [1.0, 0.0, 2.0, 0.5],
        [2.0, 0.0, 0.0, 0.5],
        [2.0, 0.0, 2.0, 0.5],
        [0.0, 0.5, 1.0, 1.0],
        [2.0, 0.5, 1.0, 1.0],
    ],
    dtype=np.float32,
)


def _default_kwargs(**overrides):
    """One valid ``DefaultSubstrate`` argument set plus the per-test overrides."""
    kwargs = {
        "num_inputs": 2,
        "num_outputs": 1,
        "query_coors": DEFAULT_QUERY_COORS.copy(),
        "nodes": DEFAULT_NODES.copy(),
        "conn_index": DEFAULT_CONN_INDEX.copy(),
    }
    kwargs.update(overrides)
    return kwargs


def _key_coors(layer_coords, sizes):
    """Map every node key to its hand-written coordinate, in TensorNEAT key order.

    Keys are inputs ``0 .. sizes[0]-1``, outputs next (``sizes[-1]`` of them), then
    the hidden layers in ascending layer order — the hidden block sits *after* the
    output block even though it is the conceptual middle layer.
    """
    si, so = sizes[0], sizes[-1]
    mapping = {
        idx: tuple(float(value) for value in coor)
        for idx, coor in enumerate(layer_coords[0])
    }
    mapping.update(
        {
            si + idx: tuple(float(value) for value in coor)
            for idx, coor in enumerate(layer_coords[-1])
        }
    )
    offset = 0
    for coors in layer_coords[1:-1]:
        for idx, coor in enumerate(coors):
            mapping[si + so + offset + idx] = tuple(float(v) for v in coor)
        offset += len(coors)
    return mapping


def _endpoint_rows(conn_index, key_coors):
    """Rebuild ``query_coors``: row ``q`` is ``[coor(src), coor(dst)]`` of row ``q``."""
    rows = [
        list(key_coors[int(in_key)]) + list(key_coors[int(out_key)])
        for in_key, out_key in np.asarray(conn_index).tolist()
    ]
    return np.array(rows, dtype=np.float32)


def _assert_key_density(nodes, nodes_cnt):
    """``nodes`` is a dense permutation of ``0 .. nodes_cnt-1``."""
    np.testing.assert_array_equal(np.sort(nodes), np.arange(nodes_cnt, dtype=np.int32))
    assert len(set(nodes.tolist())) == nodes_cnt


# --- base.py: node_key_blocks ----------------------------------------------
def test_node_key_blocks_concatenates_hidden_layers_in_layer_order():
    """``(2, 1, (2, 3))``: inputs [0,1], output [2], hidden [3..7] layer by layer."""
    input_keys, output_keys, hidden_keys = node_key_blocks(2, 1, (2, 3))

    np.testing.assert_array_equal(input_keys, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(output_keys, np.array([2], dtype=np.int32))
    # 2 + 3 hidden neurons -> five keys starting right after the output block.
    np.testing.assert_array_equal(
        hidden_keys, np.array([3, 4, 5, 6, 7], dtype=np.int32)
    )
    assert (input_keys.dtype, output_keys.dtype, hidden_keys.dtype) == (
        np.int32,
        np.int32,
        np.int32,
    )
    assert (input_keys.shape, output_keys.shape, hidden_keys.shape) == (
        (2,),
        (1,),
        (5,),
    )
    # Documented slice rule: layer 0 owns hidden_keys[0:2], layer 1 [2:5] —
    # concatenation, not interleaving.
    assert np.cumsum((2, 3))[:-1].tolist() == [2]
    np.testing.assert_array_equal(hidden_keys[:2], np.array([3, 4], dtype=np.int32))
    np.testing.assert_array_equal(hidden_keys[2:], np.array([5, 6, 7], dtype=np.int32))


def test_node_key_blocks_zero_sized_and_absent_hidden_layers_contribute_no_key():
    """A legal ``0`` size shifts no key, and no hidden sizes give an empty block."""
    input_keys, output_keys, hidden_keys = node_key_blocks(2, 2, (0, 3))
    np.testing.assert_array_equal(input_keys, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(output_keys, np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(hidden_keys, np.array([4, 5, 6], dtype=np.int32))
    # Layer 0 owns hidden_keys[0:0], so layer 1 starts at hidden_keys[0].
    assert hidden_keys[:0].shape == (0,)
    _, _, no_hidden = node_key_blocks(3, 2)
    assert no_hidden.shape == (0,) and no_hidden.dtype == np.int32


# --- base.py: cartesian_product --------------------------------------------
def test_cartesian_product_varies_the_source_axis_slowest():
    """Row ``i*len(keys2)+j`` pairs ``keys1[i]`` with ``keys2[j]``, coors joined."""
    coors, key_pairs = cartesian_product(
        np.array([0, 1], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    )

    # Rows: (0,2), (0,3), (1,2), (1,3) — the source block repeats, the target tiles.
    np.testing.assert_array_equal(
        key_pairs, np.array([[0, 2], [0, 3], [1, 2], [1, 3]], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        coors,
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert coors.shape == (4, 2 * 2)
    assert key_pairs.shape == (4, 2)
    assert (coors.dtype, key_pairs.dtype) == (np.float32, np.int32)
    # The same rule read off the pairs: the source changes only every len(keys2).
    np.testing.assert_array_equal(key_pairs[:, 0], np.array([0, 0, 1, 1], np.int32))
    np.testing.assert_array_equal(key_pairs[:, 1], np.array([2, 3, 2, 3], np.int32))


@pytest.mark.parametrize("empty_side", ["source", "target"])
def test_cartesian_product_with_an_empty_side_keeps_width_and_dtypes(empty_side):
    """An empty side yields zero rows with the full ``2*cd`` / ``2`` widths."""
    empty_keys = np.zeros(0, dtype=np.int32)
    empty_coors = np.zeros((0, 2), dtype=np.float32)
    keys = np.array([2, 3], dtype=np.int32)
    coors = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    if empty_side == "source":
        result_coors, key_pairs = cartesian_product(empty_keys, keys, empty_coors, coors)
    else:
        result_coors, key_pairs = cartesian_product(keys, empty_keys, coors, empty_coors)

    assert result_coors.shape == (0, 2 * 2)
    assert key_pairs.shape == (0, 2)
    assert (result_coors.dtype, key_pairs.dtype) == (np.float32, np.int32)


# --- base.py: layer_coors --------------------------------------------------
@pytest.mark.parametrize(
    ("neuron_cnt", "x_min", "x_max", "y_coor", "expected"),
    [
        pytest.param(1, 0.0, 2.0, 5.0, [[1.0, 5.0]], id="single-neuron-centred"),
        pytest.param(3, -1.0, 1.0, 0.0, [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            id="both-edges"),
        pytest.param(
            4,
            0.0,
            3.0,
            -2.0,
            [[0.0, -2.0], [1.0, -2.0], [2.0, -2.0], [3.0, -2.0]],
            id="integer-steps",
        ),
    ],
)
def test_layer_coors_spans_the_range_inclusively(
    neuron_cnt, x_min, x_max, y_coor, expected
):
    """First ``x`` is ``x_min``, last ``x_max``, every ``y`` is ``y_coor``."""
    rows = layer_coors(neuron_cnt, x_min, x_max, y_coor)

    np.testing.assert_array_equal(rows, np.array(expected, dtype=np.float32))
    assert rows.shape == (neuron_cnt, 2)
    assert rows.dtype == np.float32


# --- base.py: as_coor_array ------------------------------------------------
def test_as_coor_array_rejects_a_one_dimensional_input_and_names_the_field():
    """A 1-D layout is never promoted to ``(1, n)``; the message names the field."""
    with pytest.raises(TensorNEATError, match="input_coors must be a 2-D"):
        as_coor_array([0.0, 1.0], "input_coors")


@pytest.mark.parametrize(
    ("coors", "match"),
    [
        pytest.param(np.zeros((0, 2)), "must hold at least one coordinate", id="no-rows"),
        pytest.param(np.zeros((2, 0)), "at least one coordinate dimension", id="zero-width"),
        pytest.param(((0.0, np.nan),), "must hold finite coordinates", id="nan"),
        pytest.param(((0.0, np.inf),), "must hold finite coordinates", id="infinite"),
        # A ragged layout cannot be a coordinate array: it would misalign rows.
        pytest.param(((0.0, 0.0), (1.0,)), "rectangular", id="ragged"),
    ],
)
def test_as_coor_array_rejects_degenerate_coordinate_sets(coors, match):
    """Empty, zero-width and non-finite sets are configuration errors."""
    with pytest.raises(TensorNEATError, match=match):
        as_coor_array(coors)


def test_as_coor_array_copies_and_coerces_to_float32():
    """Integral input becomes a fresh ``float32`` ``(n, cd)`` copy."""
    original = np.array([[0, 0], [1, 2]], dtype=np.int64)

    array = as_coor_array(original)

    np.testing.assert_array_equal(array, np.array([[0.0, 0.0], [1.0, 2.0]], np.float32))
    assert (array.dtype, array.shape) == (np.float32, (2, 2))
    array[0, 0] = 9.0  # a copy, not a view
    assert original[0, 0] == 0


# --- base.py: check_same_coordinate_dim ------------------------------------
def test_check_same_coordinate_dim_returns_the_shared_dimension():
    """Agreeing sets report their common ``cd``; a single pair is legal."""
    shared = check_same_coordinate_dim(
        ("input_coors", np.zeros((2, 2), np.float32)),
        ("hidden_coors", np.zeros((3, 2), np.float32)),
        ("output_coors", np.zeros((1, 2), np.float32)),
    )

    assert shared == 2 and isinstance(shared, int)
    assert check_same_coordinate_dim(("only", np.zeros((4, 3)))) == 3


def test_check_same_coordinate_dim_rejects_disagreement():
    """Two sets in different coordinate spaces cannot form one connection row."""
    with pytest.raises(TensorNEATError, match="coordinate space"):
        check_same_coordinate_dim(
            ("input_coors", np.zeros((2, 2), np.float32)),
            ("output_coors", np.zeros((2, 3), np.float32)),
        )


@pytest.mark.parametrize(
    ("named_coors", "match"),
    [
        pytest.param((), "needs at least one", id="no-pairs"),
        pytest.param((("input_coors", np.zeros(4, np.float32)),), "must be a 2-D", id="one-dim"),
        pytest.param((("input_coors", np.zeros((2, 0), np.float32)),), "with cd >= 1",
            id="cd-zero"),
    ],
)
def test_check_same_coordinate_dim_rejects_degenerate_pairs(named_coors, match):
    """A pair must be ``(n, cd)`` with ``cd >= 1``, and at least one pair is needed."""
    with pytest.raises(TensorNEATError, match=match):
        check_same_coordinate_dim(*named_coors)


# --- DefaultSubstrate ------------------------------------------------------
@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_default_substrate_stores_verbatim_normalised_arrays(dtype):
    """Storage is verbatim but ``int32``/``float32`` normalized, keys in order."""
    substrate = DefaultSubstrate(
        num_inputs=2,
        num_outputs=1,
        query_coors=DEFAULT_QUERY_COORS.astype(dtype),
        nodes=DEFAULT_NODES.astype(dtype),
        conn_index=DEFAULT_CONN_INDEX.astype(dtype),
    )

    np.testing.assert_array_equal(substrate.query_coors, DEFAULT_QUERY_COORS)
    np.testing.assert_array_equal(substrate.nodes, DEFAULT_NODES)
    np.testing.assert_array_equal(substrate.conn_index, DEFAULT_CONN_INDEX)
    assert substrate.query_coors.dtype == np.float32
    assert substrate.nodes.dtype == np.int32
    assert substrate.conn_index.dtype == np.int32
    assert (substrate.num_inputs, substrate.num_outputs) == (2, 1)
    assert (substrate.nodes_cnt, substrate.conns_cnt) == (4, 2)
    assert substrate.connection_type == "recurrent"
    assert substrate.validate() is None
    # The pre-rewrite ``(nodes_cnt, 1)`` column is flattened to ``(nodes_cnt,)``.
    column = DefaultSubstrate(**_default_kwargs(nodes=DEFAULT_NODES.reshape(-1, 1)))
    np.testing.assert_array_equal(column.nodes, DEFAULT_NODES)
    assert (column.nodes.shape, column.nodes.dtype) == ((4,), np.int32)


def test_default_substrate_copies_the_callers_arrays():
    """HyperNEAT freezes these as constants, so the substrate must own its arrays."""
    kwargs = _default_kwargs()
    substrate = DefaultSubstrate(**kwargs)
    for name in ("query_coors", "nodes", "conn_index"):
        kwargs[name][...] = 7  # mutate the caller's arrays in place

    np.testing.assert_array_equal(substrate.query_coors, DEFAULT_QUERY_COORS)
    np.testing.assert_array_equal(substrate.nodes, DEFAULT_NODES)
    np.testing.assert_array_equal(substrate.conn_index, DEFAULT_CONN_INDEX)


def test_default_substrate_repr_summarises_without_the_arrays():
    """One line: class name, ``connection_type``, the four counts, no array dump."""
    text = repr(DefaultSubstrate(**_default_kwargs()))

    assert text.startswith("DefaultSubstrate(")
    assert "connection_type='recurrent'" in text
    assert all(
        f"{field}={value}" in text
        for field, value in (
            ("num_inputs", 2),
            ("num_outputs", 1),
            ("nodes_cnt", 4),
            ("conns_cnt", 2),
        )
    )
    assert "\n" not in text
    assert "[" not in text and "array(" not in text  # no list/array repr


def test_default_substrate_validate_rechecks_replaced_fields():
    """``validate`` is public so a subclass may re-run it after rewriting storage."""
    substrate = DefaultSubstrate(**_default_kwargs())

    substrate._nodes = np.array([0, 0, 2, 3], dtype=np.int32)
    with pytest.raises(TensorNEATError, match="nodes must hold unique keys"):
        substrate.validate()


# --- BaseSubstrate protocol ------------------------------------------------
@pytest.mark.parametrize(
    "member",
    ["num_inputs", "num_outputs", "nodes_cnt", "conns_cnt", "query_coors", "nodes",
     "conn_index", "validate"],
)
def test_base_substrate_members_are_abstract(member):
    """Every count/array/validate member is ``NotImplementedError`` on the base."""
    base = BaseSubstrate()

    with pytest.raises(NotImplementedError):
        attribute = getattr(base, member)  # a property raises on access
        if callable(attribute):
            attribute()  # ``validate`` raises when called


def test_base_substrate_names_no_connection_type_and_cannot_print():
    """No executor requirement and no counts, so the base sets neither."""
    assert BaseSubstrate.connection_type is None
    assert BaseSubstrate().connection_type is None
    with pytest.raises(NotImplementedError):
        repr(BaseSubstrate())  # __repr__ reads the abstract count properties


# --- FullSubstrate ---------------------------------------------------------
def test_full_substrate_tiny_case_geometry_is_aligned():
    """``si=2``, ``sh=1``, ``so=1``: 4 nodes, 4 rows in three ordered blocks."""
    substrate = FullSubstrate(
        input_coors=FULL_INPUTS, hidden_coors=FULL_HIDDEN, output_coors=FULL_OUTPUTS
    )

    assert substrate.connection_type == "recurrent"
    assert (substrate.num_inputs, substrate.num_outputs) == (2, 1)
    assert (substrate.nodes_cnt, substrate.conns_cnt) == (4, 4)
    # Block 1 input->hidden (2 rows), block 2 hidden->hidden incl. the self-loop
    # (1 row), block 3 hidden->output (1 row).
    assert substrate.conns_cnt == 2 * 1 + 1 * 1 + 1 * 1

    np.testing.assert_array_equal(substrate.nodes, FULL_NODES)
    np.testing.assert_array_equal(substrate.conn_index, FULL_CONN_INDEX)
    np.testing.assert_array_equal(substrate.query_coors, FULL_QUERY_COORS)
    _assert_key_density(substrate.nodes, substrate.nodes_cnt)

    assert (substrate.nodes.dtype, substrate.conn_index.dtype) == (np.int32, np.int32)
    assert substrate.query_coors.dtype == np.float32
    assert substrate.query_coors.shape == (4, 2 * 2)
    assert substrate.conn_index.shape == (4, 2)
    assert substrate.conns_cnt == substrate.query_coors.shape[0]
    assert substrate.conns_cnt == substrate.conn_index.shape[0]

    # The alignment rule, row by row, from a hand-written key -> coordinate map.
    key_coors = _key_coors((FULL_INPUTS, FULL_HIDDEN, FULL_OUTPUTS), (2, 1, 1))
    assert key_coors == {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 2.0), 3: (0.0, 1.0)}
    np.testing.assert_array_equal(
        substrate.query_coors, _endpoint_rows(FULL_CONN_INDEX, key_coors)
    )
    for row, (in_key, out_key) in enumerate(FULL_CONN_INDEX.tolist()):
        np.testing.assert_array_equal(
            substrate.query_coors[row],
            np.array(key_coors[in_key] + key_coors[out_key], dtype=np.float32),
        )

    # The pure function is the same geometry: it diffs without instantiating.
    pure = full_analysis_substrate(FULL_INPUTS, FULL_HIDDEN, FULL_OUTPUTS)
    assert [array.dtype for array in pure] == [np.float32, np.int32, np.int32]
    for produced, stored in zip(
        pure, (substrate.query_coors, substrate.nodes, substrate.conn_index)
    ):
        np.testing.assert_array_equal(produced, stored)


def test_full_substrate_supports_one_dimensional_coordinates():
    """Any ``cd >= 1`` works: the query width follows the caller's arrays."""
    substrate = FullSubstrate(
        input_coors=((0.0,), (1.0,)), hidden_coors=((2.0,),), output_coors=((3.0,),)
    )

    # Same topology as the tiny case: keys 0/1 inputs, 2 output, 3 hidden.
    np.testing.assert_array_equal(substrate.nodes, FULL_NODES)
    np.testing.assert_array_equal(substrate.conn_index, FULL_CONN_INDEX)
    np.testing.assert_array_equal(
        substrate.query_coors,
        np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [2.0, 3.0]], dtype=np.float32),
    )
    assert substrate.query_coors.shape == (4, 2 * 1)


def test_full_substrate_default_layout_keeps_the_bias_input_last():
    """The default 3x3x1 mesh: 7 nodes, 21 connections, bias is input key ``2``."""
    substrate = FullSubstrate()

    assert (substrate.num_inputs, substrate.num_outputs) == (3, 1)
    assert (substrate.nodes_cnt, substrate.conns_cnt) == (7, 21)
    assert substrate.conns_cnt == 3 * 3 + 3 * 3 + 3 * 1
    np.testing.assert_array_equal(substrate.nodes, np.arange(7, dtype=np.int32))
    # Inputs 0/1/2 at (-1,-1), (0,-1), (1,-1); output 3 at (0,1); hidden 4/5/6
    # at (-1,0), (0,0), (1,0).
    np.testing.assert_array_equal(
        substrate.query_coors[0], np.array([-1.0, -1.0, -1.0, 0.0], np.float32)
    )
    # Row 9 opens the hidden->hidden block (si*sh = 9 rows in), a self-loop.
    assert substrate.conn_index[8].tolist() == [2, 6]
    assert substrate.conn_index[9].tolist() == [4, 4]
    np.testing.assert_array_equal(
        substrate.query_coors[9], np.array([-1.0, 0.0, -1.0, 0.0], np.float32)
    )
    assert substrate.query_coors.shape == (21, 4)


@pytest.mark.parametrize(
    ("input_coors", "hidden_coors", "output_coors"),
    [
        pytest.param(FULL_INPUTS, FULL_HIDDEN, FULL_OUTPUTS, id="2-1-1"),
        pytest.param(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), ((0.0, 1.0), (1.0, 1.0)),
            ((1.0, 2.0),), id="3-2-1"),
        pytest.param(((0.0,),), ((1.0,), (2.0,), (3.0,)), ((4.0,), (5.0,)), id="1-3-2-cd-1"),
    ],
)
def test_full_substrate_connection_count_and_key_invariants(
    input_coors, hidden_coors, output_coors
):
    """``conns_cnt == si*sh + sh*sh + sh*so`` and every row is an aligned pair."""
    si, sh, so = len(input_coors), len(hidden_coors), len(output_coors)
    cd = len(np.asarray(input_coors)[0])

    substrate = FullSubstrate(
        input_coors=input_coors, hidden_coors=hidden_coors, output_coors=output_coors
    )
    key_coors = _key_coors((input_coors, hidden_coors, output_coors), (si, sh, so))

    assert (substrate.num_inputs, substrate.num_outputs) == (si, so)
    assert substrate.nodes_cnt == si + sh + so
    assert substrate.conns_cnt == si * sh + sh * sh + sh * so
    assert substrate.query_coors.shape == (si * sh + sh * sh + sh * so, 2 * cd)
    assert substrate.conn_index.shape == (substrate.conns_cnt, 2)
    _assert_key_density(substrate.nodes, substrate.nodes_cnt)
    np.testing.assert_array_equal(
        substrate.query_coors, _endpoint_rows(substrate.conn_index, key_coors)
    )


# --- MLPSubstrate ----------------------------------------------------------
def test_mlp_substrate_layer_grid_geometry():
    """``[3,2,1]``: 6 nodes, 8 rows layer pair -> source -> target, 2-D queries."""
    substrate = MLPSubstrate(layers=list(MLP_LAYERS), coor_range=MLP_COOR_RANGE)

    assert substrate.connection_type == "feedforward"
    assert substrate.layers == MLP_LAYERS
    assert substrate.coor_range == MLP_COOR_RANGE
    assert (substrate.num_inputs, substrate.num_outputs) == (3, 1)
    assert (substrate.nodes_cnt, substrate.conns_cnt) == (6, 8)
    assert substrate.conns_cnt == 3 * 2 + 2 * 1

    np.testing.assert_array_equal(substrate.nodes, MLP_NODES)
    np.testing.assert_array_equal(substrate.conn_index, MLP_CONN_INDEX)
    np.testing.assert_array_equal(substrate.query_coors, MLP_QUERY_COORS)
    _assert_key_density(substrate.nodes, substrate.nodes_cnt)
    assert (substrate.nodes.dtype, substrate.conn_index.dtype) == (np.int32, np.int32)
    assert substrate.query_coors.dtype == np.float32
    # The grid is always 2-D, so the CPPN query width is always 4 (not 2*cd).
    assert substrate.query_coors.shape == (8, 4)
    assert substrate.conns_cnt == substrate.query_coors.shape[0]
    assert substrate.conns_cnt == substrate.conn_index.shape[0]

    # The pure function is the same geometry: it diffs without instantiating.
    for produced, stored in zip(
        mlp_analysis_substrate(list(MLP_LAYERS), MLP_COOR_RANGE),
        (substrate.query_coors, substrate.nodes, substrate.conn_index),
    ):
        np.testing.assert_array_equal(produced, stored)


def test_mlp_substrate_coordinates_follow_the_layer_formula():
    """Layer 0 at ``y_min``, layer -1 at ``y_max``, each layer spread over ``x``."""
    substrate = MLPSubstrate(layers=list(MLP_LAYERS), coor_range=MLP_COOR_RANGE)
    key_coors = _key_coors(MLP_LAYER_COORDS, MLP_LAYERS)

    # Key -> coordinate per neuron: inputs 0/1/2 on layer 0, output 3 on layer -1,
    # hidden 4/5 on the interpolated layer 1.
    assert key_coors == {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
        3: (1.0, 1.0),
        4: (0.0, 0.5),
        5: (2.0, 0.5),
    }
    np.testing.assert_array_equal(
        substrate.query_coors, _endpoint_rows(MLP_CONN_INDEX, key_coors)
    )
    # Layer 0 sits on y_min = 0, hidden layer 1 on y = 0 + (1-0)*1/2 = 0.5, the
    # output layer on y_max = 1.
    assert {substrate.query_coors[row, 1] for row in range(6)} == {0.0}
    assert {substrate.query_coors[row, 3] for row in range(6)} == {0.5}
    assert {substrate.query_coors[row, 1] for row in (6, 7)} == {0.5}
    assert {substrate.query_coors[row, 3] for row in (6, 7)} == {1.0}
    # Layer 0 spreads over [0, 2] -> x = 0, 1, 2; layer 1 -> the range edges.
    np.testing.assert_array_equal(
        substrate.query_coors[[0, 2, 4], 0], np.array([0.0, 1.0, 2.0], np.float32)
    )
    np.testing.assert_array_equal(
        substrate.query_coors[[6, 7], 0], np.array([0.0, 2.0], np.float32)
    )


def test_mlp_substrate_interpolates_hidden_y_and_centres_a_single_neuron():
    """Four layers on ``y`` 0..3: hidden ``i`` at ``y_min + i*(range)/(n-1)``."""
    substrate = MLPSubstrate(layers=[2, 2, 2, 1], coor_range=(0.0, 3.0, 0.0, 3.0))

    # Layers on y = 0, 1, 2, 3; two neurons per layer spread over [0, 3] -> x =
    # 0 and 3; the single output neuron is centred at (0+3)/2 = 1.5.
    np.testing.assert_array_equal(
        substrate.nodes, np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        substrate.conn_index,
        np.array(
            [[0, 3], [0, 4], [1, 3], [1, 4], [3, 5], [3, 6], [4, 5], [4, 6],
             [5, 2], [6, 2]],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        substrate.query_coors,
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 3.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 3.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 1.0, 3.0, 2.0],
                [3.0, 1.0, 0.0, 2.0],
                [3.0, 1.0, 3.0, 2.0],
                [0.0, 2.0, 1.5, 3.0],
                [3.0, 2.0, 1.5, 3.0],
            ],
            dtype=np.float32,
        ),
    )
    assert substrate.conns_cnt == 2 * 2 + 2 * 2 + 2 * 1
    assert substrate.query_coors.shape == (10, 4)


def test_mlp_substrate_is_feedforward_with_no_self_loops():
    """Every edge goes to a strictly later *layer*, and no row is a self-loop.

    Key-space caveat encoded here: ``conn_index`` keys are **not** monotone when
    hidden neurons exist, because the hidden key block follows the output block
    (``nodes == [0..5]`` with output key ``3`` and hidden keys ``4``/``5``), so the
    hidden->output rows read ``(4, 3)``/``(5, 3)``.  "Feedforward" is therefore
    the layer-index property below — the one the executor actually needs.
    """
    substrate = MLPSubstrate(layers=list(MLP_LAYERS), coor_range=MLP_COOR_RANGE)
    # Hand-written key -> layer index for [3, 2, 1].
    key_layer = {0: 0, 1: 0, 2: 0, 3: 2, 4: 1, 5: 1}

    for in_key, out_key in substrate.conn_index.tolist():
        assert key_layer[in_key] < key_layer[out_key], (in_key, out_key)
        assert in_key != out_key
    # No edge is listed twice: every CPPN query row is one distinct weight.
    assert len({tuple(row) for row in substrate.conn_index.tolist()}) == (
        substrate.conns_cnt
    )
    # Row order inside a block: layer pair ascending, then source, then target.
    assert substrate.conn_index[:3, 0].tolist() == [0, 0, 1]
    assert substrate.conn_index[:3, 1].tolist() == [4, 5, 4]


def test_mlp_substrate_two_layers_wire_inputs_straight_to_outputs():
    """``len(layers) == 2`` is a real no-hidden case; one-neuron layers centre."""
    substrate = MLPSubstrate(layers=[2, 3], coor_range=(-1.0, 1.0, -1.0, 1.0))

    assert (substrate.num_inputs, substrate.num_outputs) == (2, 3)
    assert (substrate.nodes_cnt, substrate.conns_cnt) == (5, 6)
    # Inputs 0/1, outputs 2/3/4: no hidden block, so every edge's keys increase.
    np.testing.assert_array_equal(
        substrate.nodes, np.array([0, 1, 2, 3, 4], dtype=np.int32)
    )
    assert np.all(substrate.conn_index[:, 0] < substrate.conn_index[:, 1])
    np.testing.assert_array_equal(
        substrate.conn_index,
        np.array([[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]], dtype=np.int32),
    )
    # Layer 0 on y_min = -1 spread over [-1, 1] -> x = -1, 1; layer 1 on
    # y_max = 1 spread over three neurons -> x = -1, 0, 1.
    np.testing.assert_array_equal(
        substrate.query_coors,
        np.array(
            [
                [-1.0, -1.0, -1.0, 1.0],
                [-1.0, -1.0, 0.0, 1.0],
                [-1.0, -1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0, 1.0],
                [1.0, -1.0, 0.0, 1.0],
                [1.0, -1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert substrate.query_coors.shape == (6, 4)

    # One-neuron layers sit at the middle of ``[x_min, x_max]`` (via layer_coors).
    centred = MLPSubstrate(layers=[1, 1])
    np.testing.assert_array_equal(centred.conn_index, np.array([[0, 1]], np.int32))
    np.testing.assert_array_equal(
        centred.query_coors, np.array([[0.0, -1.0, 0.0, 1.0]], np.float32)
    )


# --- validation matrix: DefaultSubstrate -----------------------------------
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"num_inputs": True}, "num_inputs must be an int", id="in-bool"),
        pytest.param({"num_inputs": 1.0}, "num_inputs must be an int", id="in-float"),
        pytest.param({"num_inputs": 0}, "num_inputs must be >= 1", id="in-zero"),
        pytest.param({"num_inputs": -2}, "num_inputs must be >= 1", id="in-negative"),
        pytest.param({"num_outputs": True}, "num_outputs must be an int", id="out-bool"),
        pytest.param({"num_outputs": 2.5}, "num_outputs must be an int", id="out-float"),
        pytest.param({"num_outputs": 0}, "num_outputs must be >= 1", id="out-zero"),
    ],
)
def test_default_substrate_rejects_bad_counts(overrides, match):
    """A count must be an ``int >= 1``; ``bool`` would silently pass as ``1``."""
    with pytest.raises(TensorNEATError, match=match):
        DefaultSubstrate(**_default_kwargs(**overrides))


@pytest.mark.parametrize(
    ("query_coors", "match"),
    [
        pytest.param(np.zeros((2, 3), np.float32), "even, non-zero trailing width",
            id="odd-width"),
        pytest.param(np.zeros((2, 0), np.float32), "even, non-zero trailing width",
            id="zero-width"),
        pytest.param(np.zeros(4, np.float32), "query_coors must be a 2-D", id="one-dim"),
        pytest.param(np.zeros((2, 2, 2), np.float32), "query_coors must be a 2-D", id="three-dim"),
        pytest.param(np.zeros((0, 4), np.float32), "at least one connection row", id="no-rows"),
        pytest.param([[0, 0, 0, 0], [1]], "rectangular", id="ragged"),
    ],
)
def test_default_substrate_rejects_bad_query_coors(query_coors, match):
    """A query row must hold two whole coordinates: 2-D, even, non-zero width."""
    with pytest.raises(TensorNEATError, match=match):
        DefaultSubstrate(**_default_kwargs(query_coors=query_coors))


@pytest.mark.parametrize(
    ("nodes", "match"),
    [
        pytest.param(np.array([], np.int32), "at least one node key", id="empty"),
        pytest.param(np.zeros((4, 2), np.int32), "nodes must be a 1-D", id="two-columns"),
        pytest.param(np.array([0.0, 1.0, 2.0, 3.5]), "must hold integer keys", id="fractional"),
        pytest.param(np.array([0, 0, 2, 3], np.int32), "nodes must hold unique keys",
            id="duplicate"),
        pytest.param(np.array([0, 1, 2, 4], np.int32), "must be a permutation of 0",
            id="not-a-permutation"),
        pytest.param(np.array([0, 2, 1, 3], np.int32), "must be the input block 0",
            id="wrong-input-block"),
        pytest.param(np.array([0, 1, 3, 2], np.int32), "must be the output block 2",
            id="wrong-output-block"),
    ],
)
def test_default_substrate_rejects_bad_nodes(nodes, match):
    """Keys must be integral, unique and in the TensorNEAT block order."""
    with pytest.raises(TensorNEATError, match=match):
        DefaultSubstrate(**_default_kwargs(nodes=nodes))


@pytest.mark.parametrize(
    ("conn_index", "match"),
    [
        pytest.param(np.zeros((2, 3), np.int32), "conn_index must be an", id="three-columns"),
        pytest.param(np.array([0, 3], np.int32), "conn_index must be an", id="one-dim"),
        pytest.param(np.zeros((0, 2), np.int32), "at least one connection row", id="empty"),
        pytest.param(np.array([[0.0, 3.5], [3.0, 2.0]]), "must hold integer keys",
            id="fractional"),
    ],
)
def test_default_substrate_rejects_bad_conn_index(conn_index, match):
    """Endpoints travel as ``(conns_cnt, 2)`` integral rows, at least one of them."""
    with pytest.raises(TensorNEATError, match=match):
        DefaultSubstrate(**_default_kwargs(conn_index=conn_index))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"num_inputs": 3, "num_outputs": 2}, "cannot hold 3 input and 2 output",
            id="io-past-nodes"),
        pytest.param({"query_coors": THREE_ROW_QUERIES},
            "query row q must describe connection q", id="row-count"),
        pytest.param({"query_coors": THREE_ROW_QUERIES, "conn_index": UNKNOWN_ENDPOINT_CONN},
            "conn_index endpoints must be node keys", id="unknown-key"),
        pytest.param({"query_coors": THREE_ROW_QUERIES, "conn_index": DUPLICATE_ROW_CONN},
            "conn_index rows must be unique", id="duplicate-row"),
    ],
)
def test_default_substrate_rejects_cross_field_inconsistencies(overrides, match):
    """Row counts must agree, endpoints must be keys, one edge has one weight."""
    with pytest.raises(TensorNEATError, match=match):
        DefaultSubstrate(**_default_kwargs(**overrides))


# --- validation matrix: FullSubstrate --------------------------------------
@pytest.mark.parametrize(
    ("field", "match"),
    [
        pytest.param("input_coors", "input_coors must hold at least one coordinate", id="inputs"),
        pytest.param("hidden_coors", "hidden_coors must hold at least one coordinate",
            id="hidden"),
        pytest.param("output_coors", "output_coors must hold at least one coordinate",
            id="outputs"),
    ],
)
def test_full_substrate_rejects_empty_coordinate_sets(field, match):
    """``hidden_coors`` is the only path to the outputs, so none may be empty."""
    kwargs = {
        "input_coors": FULL_INPUTS,
        "hidden_coors": FULL_HIDDEN,
        "output_coors": FULL_OUTPUTS,
    }
    kwargs[field] = np.zeros((0, 2), dtype=np.float32)

    with pytest.raises(TensorNEATError, match=match):
        FullSubstrate(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"input_coors": (0.0, 1.0)}, "input_coors must be a 2-D", id="one-dim"),
        pytest.param({"input_coors": ()}, "input_coors must be a 2-D", id="empty-tuple"),
        pytest.param({"input_coors": np.zeros((2, 0), np.float32)},
            "at least one coordinate dimension", id="zero-width"),
        pytest.param({"input_coors": ((0.0, np.nan),)}, "must hold finite coordinates", id="nan"),
        pytest.param({"output_coors": ((0.0, 1.0, 2.0),)}, "coordinate space",
            id="coordinate-dim-disagreement"),
    ],
)
def test_full_substrate_rejects_bad_coordinates(kwargs, match):
    """All three sets must be finite 2-D arrays sharing one ``cd >= 1``."""
    full_kwargs = {
        "input_coors": FULL_INPUTS,
        "hidden_coors": FULL_HIDDEN,
        "output_coors": FULL_OUTPUTS,
    }
    full_kwargs.update(kwargs)

    with pytest.raises(TensorNEATError, match=match):
        FullSubstrate(**full_kwargs)


# --- validation matrix: MLPSubstrate ---------------------------------------
@pytest.mark.parametrize(
    ("layers", "match"),
    [
        pytest.param([3], "at least two layers", id="one-layer"),
        pytest.param([], "at least two layers", id="no-layers"),
        pytest.param(5, "must be a sequence of layer sizes", id="not-a-sequence"),
        pytest.param([True, 2], r"layers\[0\] must be an int", id="inputs-bool"),
        pytest.param([3, True], r"layers\[1\] must be an int", id="outputs-bool"),
        pytest.param([3, 2.0], r"layers\[1\] must be an int", id="float-count"),
        pytest.param([3, "2"], r"layers\[1\] must be an int", id="string-count"),
        pytest.param([3, 0], r"layers\[1\] must be >= 1", id="zero-count"),
        pytest.param([3, -1], r"layers\[1\] must be >= 1", id="negative-count"),
    ],
)
def test_mlp_substrate_rejects_bad_layers(layers, match):
    """At least two layers, every count a positive non-``bool`` integer."""
    with pytest.raises(TensorNEATError, match=match):
        MLPSubstrate(layers=layers)


@pytest.mark.parametrize(
    ("coor_range", "match"),
    [
        pytest.param((0, 1, 0), "must hold four numbers", id="too-short"),
        pytest.param((0, 1, 0, 1, 2), "must hold four numbers", id="too-long"),
        pytest.param(5, "coor_range must be", id="not-a-sequence"),
        pytest.param((0, np.inf, 0, 1), r"coor_range\[1\] \(x_max\) must be finite",
            id="infinite"),
        pytest.param((0, 1, 0, np.nan), r"coor_range\[3\] \(y_max\) must be finite", id="nan"),
        pytest.param((True, 1, 0, 1), r"coor_range\[0\] \(x_min\) must be a real number",
            id="bool"),
        pytest.param(("a", 1, 0, 1), r"coor_range\[0\] \(x_min\) must be a real number",
            id="string"),
        pytest.param((1, 1, 0, 1), "x axis must be strictly increasing", id="flat-x"),
        pytest.param((2, 1, 0, 1), "x axis must be strictly increasing", id="reversed-x"),
        pytest.param((0, 1, 1, 1), "y axis must be strictly increasing", id="flat-y"),
        pytest.param((0, 1, 2, 1), "y axis must be strictly increasing", id="reversed-y"),
    ],
)
def test_mlp_substrate_rejects_bad_coor_range(coor_range, match):
    """The layout box holds four finite numbers with strictly increasing axes."""
    with pytest.raises(TensorNEATError, match=match):
        MLPSubstrate(layers=[2, 1], coor_range=coor_range)


# --- validation matrix: helper-level branches ------------------------------
@pytest.mark.parametrize(
    ("keys1", "keys2", "coors1", "coors2", "match"),
    [
        pytest.param(np.zeros((2, 2), np.int32), np.array([2], np.int32),
                     np.zeros((2, 2)), np.zeros((1, 2)),
                     "keys1 must be a 1-D key array", id="source-keys-2d"),
        pytest.param(np.array([0], np.int32), np.zeros((1, 2), np.int32),
                     np.zeros((1, 2)), np.zeros((1, 2)),
                     "keys2 must be a 1-D key array", id="target-keys-2d"),
        pytest.param(np.array([0], np.int32), np.array([2], np.int32),
                     np.zeros(2), np.zeros((1, 2)),
                     "coors1 must be a 2-D", id="source-coors-1d"),
        pytest.param(np.array([0, 1], np.int32), np.array([2], np.int32),
                     np.zeros((3, 2)), np.zeros((1, 2)),
                     "rows but its key array has", id="source-length-mismatch"),
        pytest.param(np.array([0], np.int32), np.array([2], np.int32),
                     np.zeros((1, 2)), np.zeros((1, 3)),
                     "both ends of a connection", id="coordinate-dim-mismatch"),
    ],
)
def test_cartesian_product_validates_its_inputs(keys1, keys2, coors1, coors2, match):
    """A misaligned block would silently misalign every weight, so it raises."""
    with pytest.raises(TensorNEATError, match=match):
        cartesian_product(keys1, keys2, coors1, coors2)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        pytest.param((0, 0.0, 1.0, 0.0), "neuron_cnt must be >= 1", id="zero-count"),
        pytest.param((True, 0.0, 1.0, 0.0), "neuron_cnt must be an int", id="bool-count"),
        pytest.param((2, np.nan, 1.0, 0.0), "x_min must be finite", id="nan-x-min"),
        pytest.param((2, 0.0, np.inf, 0.0), "x_max must be finite", id="infinite-x-max"),
        pytest.param((2, 0.0, 1.0, True), "y_coor must be a real number", id="bool-y"),
    ],
)
def test_layer_coors_validates_its_arguments(args, match):
    """A layer needs a positive count and finite edges, like the substrates."""
    with pytest.raises(TensorNEATError, match=match):
        layer_coors(*args)


@pytest.mark.parametrize(
    ("hidden_sizes", "match"),
    [
        pytest.param((1.5,), r"hidden_sizes\[0\] must be an int", id="float-size"),
        pytest.param((True,), r"hidden_sizes\[0\] must be an int", id="bool-size"),
        pytest.param((-1,), r"hidden_sizes\[0\] must be >= 0", id="negative-size"),
    ],
)
def test_node_key_blocks_validates_hidden_sizes(hidden_sizes, match):
    """A hidden size is a non-negative integer; ``0`` is the only allowed zero."""
    with pytest.raises(TensorNEATError, match=match):
        node_key_blocks(2, 1, hidden_sizes)
