"""Behavioural tests of the HyperNEAT CPPN -> substrate decode and its protocol surface.

``src/tensorneat/algorithm/hyperneat/hyperneat.py`` has two halves and this module tests
both: the *decode* (``_query_coors``, ``_expand_leaf`` / ``_expand_schedule`` with their
``p * Q + q`` order, the ``(P, Q, C) -> (P*Q, C)`` flattening, ``_assemble_substrate`` /
``_conn_keys`` and the two static-bound builders) and the *protocol* (the arities,
``ask`` / ``setup`` / ``tell`` / ``show_details``, the ``SubstrateSchedule`` aliases,
``transform`` and ``forward`` / ``prepare_forward`` / ``forward_prepared``).

Every expectation is hand-computed, never produced by calling the implementation again:
``DEFAULT_NODES`` / ``DEFAULT_CONN_INDEX`` / ``DEFAULT_QUERY`` are transcribed from the
documented ``FullSubstrate`` block order (input -> hidden, hidden -> hidden self-loop,
hidden -> output, source slowest), ``_tiled_by_hand`` is ``np.repeat(leaf, Q, axis=0)``,
the ``*_expected_rows`` helpers evaluate the documented tick algebra in float64, and the
reshape test drives the decode with a CPPN that is linear in the query coordinates.

Kernels go through ``./hyperneat_kernel.run_method``; the one exception is
:func:`_run_prepared_pair`, explained there. The harness geometry is 2 inputs (input 0 and
the trailing bias) and one hidden neuron: ``N = 4`` nodes (keys ``0, 1`` inputs, ``2``
output, ``3`` hidden) and ``Q = 4`` connections.
:func:`test_forward_reads_the_substrate_output_neurons` pins a source defect (strict
xfail). No network, no GPU, ``backend="numpy"``."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import evoxc

from support import payloads
from support.runner import run_defn, to_numpy

from tensorneat.algorithm.hyperneat import (
    DefaultSubstrate,
    FullSubstrate,
    HyperNEAT,
    HyperNEATConn,
    HyperNEATFeedForward,
    HyperNEATNode,
    MLPSubstrate,
    SubstrateSchedule,
)
from tensorneat.common import ACT, AGG, State
from tensorneat.common.errors import TensorNEATError
from tensorneat.common.tensorization import Transformed
from tensorneat.genome.utils import GenomeTensors

from .hyperneat_kernel import (
    CONNS_CNT,
    HIDDEN_COORS,
    INPUT_COORS,
    NODES_CNT,
    OUTPUT_COORS,
    POP_SIZE,
    QUERY_WIDTH,
    cppn_payload,
    make_cppn_genome,
    make_hyper,
    make_neat,
    make_state,
    run_method,
)

# Hand-transcribed geometry of the shared substrate: ``I = 2`` (input 0 + trailing bias),
# ``O = 1`` (key 2), hidden key 3, and ``Q = 4`` connections.
DEFAULT_NODES = np.array([0, 1, 2, 3], dtype=np.int32)
DEFAULT_CONN_INDEX = np.array([[0, 3], [1, 3], [3, 3], [3, 2]], dtype=np.int32)
DEFAULT_QUERY = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],  # (input 0) -> (hidden)
        [1.0, 0.0, 0.0, 1.0],  # (input 1, the bias) -> (hidden)
        [0.0, 1.0, 0.0, 1.0],  # (hidden) -> (hidden), the self-loop
        [0.0, 1.0, 0.0, 2.0],  # (hidden) -> (output)
    ],
    dtype=np.float32,
)

# The no-hidden substrate the ``forward`` tests run on: rows ``0 = input, 1 = bias,
# 2 = output``, so the read-out row coincides with the output *neuron*. The output neuron's
# self-loop makes the result depend on ``activate_time`` and pins the tick-1 "a ``NaN``
# source slot is not live" rule.
FLAT_NODES = np.array([0, 1, 2], dtype=np.int32)
FLAT_CONN_INDEX = np.array([[0, 2], [1, 2], [2, 2]], dtype=np.int32)
FLAT_QUERY = np.array(
    [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0]], dtype=np.float32
)

# One row per individual; column ``q`` is the weight of ``FLAT_CONN_INDEX[q]``. All three
# differ, so a bias column appended at the front or a dropped self-loop changes the value.
FLAT_WEIGHTS = np.array(
    [[0.5, -1.0, 0.25], [-0.75, 0.5, 1.5], [1.0, 0.25, -0.5], [0.0, -2.0, 0.75]],
    dtype=np.float32,
)

# Weights of ``DEFAULT_CONN_INDEX`` (shared by every row) for the documented read-out test.
HIDDEN_WEIGHTS = np.array([0.5, -1.0, 0.25, 2.0], dtype=np.float32)


# --------------------------------------------------------------------------- helpers


def _sigmoid(values):
    """``ACT.sigmoid`` as a float64 closed form.
    ``ACT.sigmoid`` as a float64 closed form."""
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def _tiled_by_hand(transformed: Transformed, n_query: int, pop_size: int) -> dict:
    """Expected expansion of every field: ``np.repeat(leaf, Q, axis=0)``.

    A leaf whose axis 0 is not ``pop_size`` is a shared table and stays unchanged."""
    expected = {}
    for name in type(transformed)._fields:
        leaf = np.asarray(getattr(transformed, name))
        population_leading = leaf.ndim >= 1 and int(leaf.shape[0]) == pop_size
        expected[name] = np.repeat(leaf, n_query, axis=0) if population_leading else leaf
    return expected


def _flat_tick_values(inputs, weights, ticks: int):
    """Documented value of the flat substrate before ``sigmoid``.

    The bias is injected last and the tick-1 self-loop source is ``NaN`` (not live)."""
    values = np.asarray(inputs, dtype=np.float64)
    w0, w1, w2 = (float(weight) for weight in weights)
    value = None
    for _ in range(ticks):
        value = w0 * values + w1 * 1.0 + (0.0 if value is None else w2 * value)
    return value


def _flat_expected_rows(inputs, weights, samples: int, ticks: int):
    """``forward``'s documented ``(B, 1)`` output; row ``p*samples + j`` is individual p.
    ``forward``'s documented ``(B, 1)`` output; row ``p*samples + j`` is individual p."""
    out = np.empty((inputs.shape[0], 1), dtype=np.float64)
    for row in range(inputs.shape[0]):
        value = _flat_tick_values(inputs[row, 0], weights[row // samples], ticks)
        out[row, 0] = _sigmoid(value)
    return out


def _hidden_expected_rows(inputs, weights, ticks: int):
    """Documented output of the shared geometry's output neuron (key 2).

    The hidden neuron aggregates ``w0*a + w1*bias + w2*h`` and is ``ACT.tanh`` applied, the
    output neuron aggregates ``w3*h`` unactivated, so the result is ``sigmoid(w3*h_{T-1})``;
    on tick 1 the self-loop source is ``NaN``, so the output row stays ``NaN``."""
    values = np.asarray(inputs, dtype=np.float64).reshape(-1)
    w0, w1, w2, w3 = (float(weight) for weight in weights)
    hidden = output = None
    for _ in range(ticks):
        hidden_next = np.tanh(
            w0 * values + w1 * 1.0 + (0.0 if hidden is None else w2 * hidden)
        )
        output_next = np.full(values.shape, np.nan) if hidden is None else w3 * hidden
        hidden, output = hidden_next, output_next
    return _sigmoid(output).reshape(-1, 1)


def _rescheduled(decoded: SubstrateSchedule, weights) -> SubstrateSchedule:
    """The decoded schedule with hand weights and zeroed node option columns.

    Option 0 is the constructor's aggregation/activation, and the schedule is derived from the
    keys alone, so only the attribute rows change."""
    weights = np.asarray(weights, dtype=np.float32)
    pop_size, conns_cnt = weights.shape
    assert conns_cnt == decoded.conn_attrs.shape[1], f"one weight column per connection, got {conns_cnt}"
    assert pop_size == decoded.node_keys.shape[0], f"{pop_size} weight rows vs {decoded.node_keys.shape[0]} keys"
    return SubstrateSchedule(
        schedule=decoded.schedule,
        node_keys=decoded.node_keys,
        node_attrs=np.zeros((pop_size, decoded.node_keys.shape[1], 2), np.float32),
        conn_keys=decoded.conn_keys,
        conn_attrs=weights.reshape(pop_size, conns_cnt, 1).copy(),
    )


def _run_prepared_pair(algorithm, state, decoded: SubstrateSchedule, inputs, samples: int):
    """Run ``prepare_forward`` + ``forward_prepared`` on one payload in one graph.

    ``prepare_forward`` returns a ``_StepStatics`` of *traced* tensors, which ``evoxc`` rejects
    as a graph output, so the two halves can only be compared inside one closure."""
    fields = type(decoded.schedule)._fields

    def kernel(*values):
        tail = values[len(fields) :]
        payload = SubstrateSchedule(type(decoded.schedule)(*values[: len(fields)]), *tail[:4])
        prepared = algorithm.prepare_forward(state, payload, None, samples)
        return algorithm.forward_prepared(state, payload, prepared, tail[4])

    leaves = [getattr(decoded.schedule, name) for name in fields]
    leaves += [decoded.node_keys, decoded.node_attrs, decoded.conn_keys, decoded.conn_attrs]
    return run_defn(kernel, *leaves, np.asarray(inputs, dtype=np.float32))


def _flat_substrate() -> DefaultSubstrate:
    """The no-hidden substrate of the ``forward`` tests (see ``FLAT_*``).
    The no-hidden substrate of the ``forward`` tests (see ``FLAT_*``)."""
    return DefaultSubstrate(2, 1, FLAT_QUERY, FLAT_NODES, FLAT_CONN_INDEX)


def _pass_through_cppn(neat, weights_by_individual):
    """A hand-built CPPN payload whose output is linear in the query coordinates.

    Neutral node rows and one live ``j -> output`` connection per query column weight the output
    by ``weights_by_individual[p, j]``, so it is ``sum_j weight_j * query[j]``."""
    genome = neat.genome
    output_row = int(genome.max_nodes) - 1
    payload = payloads.new_payload(
        len(weights_by_individual), int(genome.max_nodes), int(genome.max_conns)
    )
    neutral = (0.0, 1.0, 0.0, -1.0)
    for individual, row_weights in enumerate(weights_by_individual):
        for column in range(int(genome.num_inputs)):
            payloads.set_node(payload, individual, column, column, neutral)
        payloads.set_node(payload, individual, output_row, output_row, neutral)
        for column, weight in enumerate(row_weights):
            payloads.set_conn(payload, individual, column, column, output_row, (float(weight),))
    return GenomeTensors(*payload.args())


def _tampered_substrate(**overrides) -> DefaultSubstrate:
    """A ``DefaultSubstrate`` whose stored tables are replaced after construction.

    ``__init__`` validates its own tables, so a malformed table can only be injected this way
    without the constructor, rather than ``HyperNEAT._validate``, becoming the failing code."""
    substrate = DefaultSubstrate(2, 1, DEFAULT_QUERY, DEFAULT_NODES, DEFAULT_CONN_INDEX)
    for name, value in overrides.items():
        setattr(substrate, name, value)
    return substrate


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def hyper() -> HyperNEAT:
    """The shared decoder over the harness geometry (2 inputs, 1 hidden, 1 output).
    The shared decoder over the harness geometry (2 inputs, 1 hidden, 1 output)."""
    return make_hyper()


@pytest.fixture(scope="module")
def state(hyper):
    """Generation-0 state of ``hyper``.
    Generation-0 state of ``hyper``."""
    return make_state(hyper, seed=0)


@pytest.fixture(scope="module")
def decoded(hyper, state) -> SubstrateSchedule:
    """One full-population decode (``P = 4``).
    One full-population decode (``P = 4``)."""
    return run_method(hyper, "transform", cppn_payload(state), state=state)


@pytest.fixture(scope="module")
def flat() -> HyperNEAT:
    """A decoder whose read-out rows *are* its output neurons (no hidden block).
    A decoder whose read-out rows *are* its output neurons (no hidden block)."""
    return make_hyper(substrate=_flat_substrate())


@pytest.fixture(scope="module")
def flat_state(flat):
    """Generation-0 state of ``flat``.
    Generation-0 state of ``flat``."""
    return make_state(flat, seed=1)


@pytest.fixture(scope="module")
def flat_decoded(flat, flat_state) -> SubstrateSchedule:
    """One full-population decode of the flat substrate.
    One full-population decode of the flat substrate."""
    return run_method(flat, "transform", cppn_payload(flat_state), state=flat_state)


# --------------------------------------------------------------------------- B. grid


def test_query_coors_broadcasts_the_substrate_grid(hyper):
    """``_query_coors(P)`` is ``(P, Q, C)`` and every block is the hand coordinate table.

    Row ``q`` must be ``[coor(source_q), coor(target_q)]`` of ``DEFAULT_CONN_INDEX[q]``, the 1:1
    contract between ``query_coors`` and ``conn_index``."""
    grid = run_method(hyper, "_query_coors", POP_SIZE)
    assert grid.shape == (POP_SIZE, CONNS_CNT, QUERY_WIDTH), f"{(POP_SIZE, CONNS_CNT, QUERY_WIDTH)} != {grid.shape}"
    assert grid.dtype == np.float32, f"the query grid must be float32, got {grid.dtype}"
    for individual in range(POP_SIZE):
        npt.assert_array_equal(grid[individual], DEFAULT_QUERY, err_msg=f"block {individual}")


def test_query_coors_repeats_the_grid_without_a_per_individual_loop(hyper):
    """The grid is a pure broadcast: ``P = 1`` and ``P = 5`` agree block for block.

    A per-individual loop, a ``tile`` on the wrong axis or a grid sized from ``self.pop_size``
    instead of the argument would make one of the two batches disagree."""
    one = run_method(hyper, "_query_coors", 1)
    five = run_method(hyper, "_query_coors", 5)
    assert one.shape == (1, CONNS_CNT, QUERY_WIDTH), f"P = 1 must keep a leading axis, got {one.shape}"
    assert five.shape == (5, CONNS_CNT, QUERY_WIDTH), f"P = 5 must give 5 blocks, got {five.shape}"
    npt.assert_array_equal(one[0], five[0], err_msg="block 0 must not depend on the batch size")
    npt.assert_array_equal(one[0], five[4], err_msg="block P-1 must equal the P = 1 block")
    for individual in range(1, 5):
        npt.assert_array_equal(five[individual], five[0], err_msg=f"block {individual} is a broadcast")


# --------------------------------------------------------------------------- C. expand


def test_expand_leaf_repeats_every_row_n_query_times_individual_slowest(hyper):
    """A population-leading leaf ``(P, ...)`` becomes ``(P*Q, ...)`` with row ``p*Q+q``.

    The hand oracle ``P = Q = 2`` over ``[[0, 1, 2], [3, 4, 5]]`` is written literally, so an
    interleaved ``q*P + p`` order, a ``tile`` on the wrong axis or a lost trailing axis fails."""
    leaf = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    expected = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], np.float32)
    expanded = run_method(hyper, "_expand_leaf", leaf, 2, 2)
    assert expanded.shape == (4, 3), f"P*Q rows, trailing axes kept: (4, 3) != {expanded.shape}"
    npt.assert_array_equal(expanded, expected, err_msg="row p*Q+q must be leaf row p")


def test_expand_leaf_returns_a_leaf_without_the_population_axis_unchanged(hyper):
    """A leaf whose axis 0 is not ``P`` carries no population axis and comes back as it is.

    That shape test is the only guard between a shared ordering table and a wrong ``(P*Q, ...)``
    reshape, so it is checked at both ``3 > P`` and ``1 < P``."""
    fewer = np.arange(6, dtype=np.int32).reshape(3, 2)
    more = np.arange(3, dtype=np.int32).reshape(1, 3)
    npt.assert_array_equal(run_method(hyper, "_expand_leaf", fewer, 2, 2), fewer, err_msg="(3, 2) leaf, P = 2")
    npt.assert_array_equal(run_method(hyper, "_expand_leaf", more, 2, 2), more, err_msg="(1, 3) leaf, P = 2")


def test_expand_schedule_tiles_every_population_leaf_of_a_real_cppn_schedule(hyper, state):
    """``_expand_schedule`` tiles every field of a real ``neat.transform`` by ``p*Q + q``.

    Each of the twelve fields is compared against ``_tiled_by_hand`` (``np.repeat(leaf, Q,
    axis=0)``), so a field left un-expanded or expanded on the wrong axis is caught by name."""
    cppn_schedule = run_method(hyper.neat, "transform", cppn_payload(state), state=state)
    expanded = run_method(hyper, "_expand_schedule", cppn_schedule, CONNS_CNT, POP_SIZE)
    expected = _tiled_by_hand(cppn_schedule, CONNS_CNT, POP_SIZE)
    assert set(type(expanded)._fields) == set(expected), f"fields {type(expanded)._fields}"
    for name, want in expected.items():
        got = np.asarray(getattr(expanded, name))
        assert got.shape == want.shape, f"field {name}: {want.shape} expected, got {got.shape}"
        npt.assert_array_equal(got, want, err_msg=f"field {name} must be np.repeat(leaf, Q, axis=0)")


def test_expand_schedule_keeps_a_shared_leaf_of_the_schedule_untouched(hyper):
    """A schedule leaf whose axis 0 is not ``P`` is a shared table and must not be tiled.

    The real CPPN schedule is population-leading everywhere, so this builds a synthetic
    ``Transformed`` whose ``output_slot`` field is a ``(3, 2)`` table while ``P = 2``."""
    pop_size, n_query = 2, 3
    shared = np.arange(6, dtype=np.int32).reshape(3, 2)
    fields = {name: np.zeros((pop_size, 2), np.int32) for name in Transformed._fields}
    fields["output_slot"] = shared
    schedule = Transformed(**fields)

    expanded = run_method(hyper, "_expand_schedule", schedule, n_query, pop_size)
    npt.assert_array_equal(np.asarray(expanded.output_slot), shared, err_msg="shared (3, 2) leaf, P = 2")
    for name, want in _tiled_by_hand(schedule, n_query, pop_size).items():
        npt.assert_array_equal(np.asarray(getattr(expanded, name)), want, err_msg=f"field {name}")


def test_query_substrate_flattens_the_query_grid_in_individual_major_order():
    """The ``(P, Q, C) -> (P*Q, C)`` flattening uses the same ``p*Q + q`` order.

    A pass-through CPPN tags grid row ``q`` with ``0.25 * (q + 1)`` and individual ``p`` weights
    it by ``p + 1``, so ``P = 2`` / ``Q = 3`` must give ``[[0.25, 0.5, 0.75], [0.5, 1.0, 1.5]]``;
    the second half is the all-zero CPPN, which must give zeros on every grid row."""
    neat = make_neat(genome=make_cppn_genome(output_transform=None), pop_size=2)
    assert neat.genome.node_gene.aggregation_options[0] is AGG.sum, "option 0 must be AGG.sum"
    tags = np.array([[0.25, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.75, 0.0, 0.0, 0.0]], np.float32)
    substrate = DefaultSubstrate(2, 1, tags, FLAT_NODES, FLAT_CONN_INDEX)
    algorithm = make_hyper(substrate=substrate, neat=neat, activation=ACT.tanh)
    host_state = State()  # the decode reads no entry of the state

    payload = _pass_through_cppn(neat, [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    query = run_method(algorithm, "_query_substrate", payload, state=host_state)
    assert query.shape == (2, 3), f"one raw output per (individual, connection): (2, 3) != {query.shape}"
    npt.assert_array_equal(
        query,
        np.array([[0.25, 0.5, 0.75], [0.5, 1.0, 1.5]], np.float32),
        err_msg="individual p weighted by p+1 over grid row q (the p*Q+q order)",
    )

    zero_payload = _pass_through_cppn(neat, [[0.0] * 4, [0.0] * 4])
    zero_query = run_method(algorithm, "_query_substrate", zero_payload, state=host_state)
    npt.assert_array_equal(zero_query, np.zeros((2, 3), np.float32), err_msg="an all-zero CPPN")


# --------------------------------------------------------------------------- D. payload


def test_assemble_substrate_paints_the_fixed_topology_with_the_decoded_weights(hyper):
    """The payload is the substrate's tables plus one weight column, exactly.

    ``node_keys``/``conn_keys`` are ``DEFAULT_NODES``/``DEFAULT_CONN_INDEX`` broadcast verbatim,
    ``node_attrs`` is ``(P, N, 2)`` zeros (option 0 of both ``HyperNEATNode`` columns) and
    ``conn_attrs[:, :, 0]`` holds the distinct hand weights (``HyperNEATConn`` has one column)."""
    weights = np.array(
        [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [-0.5, -1.5, -2.5, -3.5], [0.25, 0.5, 0.75, 1.0]],
        np.float32,
    )
    payload = run_method(hyper, "_assemble_substrate", weights)

    assert type(payload) is GenomeTensors, f"the decode emits GenomeTensors, got {type(payload).__name__}"
    assert payload._fields == ("node_keys", "node_attrs", "conn_keys", "conn_attrs"), (
        f"GenomeTensors' field order is genome.forward's parameter order, got {payload._fields}"
    )

    assert payload.node_keys.shape == (POP_SIZE, NODES_CNT), f"{payload.node_keys.shape}"
    assert payload.node_keys.dtype == np.int32, f"{payload.node_keys.dtype}"
    npt.assert_array_equal(
        payload.node_keys,
        np.broadcast_to(DEFAULT_NODES[None, :], (POP_SIZE, NODES_CNT)),
        err_msg="node_keys must broadcast the substrate's node table",
    )

    assert payload.node_attrs.shape == (POP_SIZE, NODES_CNT, 2), f"{payload.node_attrs.shape}"
    assert payload.node_attrs.dtype == np.float32, f"{payload.node_attrs.dtype}"
    npt.assert_array_equal(
        payload.node_attrs,
        np.zeros((POP_SIZE, NODES_CNT, 2), np.float32),
        err_msg="no node attribute is decoded, so every option column is 0",
    )

    assert payload.conn_keys.shape == (POP_SIZE, CONNS_CNT, 2), f"{payload.conn_keys.shape}"
    assert payload.conn_keys.dtype == np.int32, f"{payload.conn_keys.dtype}"
    npt.assert_array_equal(
        payload.conn_keys[:, :, 0],
        np.broadcast_to(DEFAULT_CONN_INDEX[None, :, 0], (POP_SIZE, CONNS_CNT)),
        err_msg="conn_keys[..., 0] must be the substrate's source keys",
    )
    npt.assert_array_equal(
        payload.conn_keys[:, :, 1],
        np.broadcast_to(DEFAULT_CONN_INDEX[None, :, 1], (POP_SIZE, CONNS_CNT)),
        err_msg="conn_keys[..., 1] must be the substrate's destination keys",
    )

    assert payload.conn_attrs.shape == (POP_SIZE, CONNS_CNT, 1), f"{payload.conn_attrs.shape}"
    assert payload.conn_attrs.dtype == np.float32, f"{payload.conn_attrs.dtype}"
    assert hyper.hyper_genome.conn_gene.attr_width == 1, (
        "HyperNEATConn has exactly one attribute column, so no other column must be zero; "
        f"got {hyper.hyper_genome.conn_gene.attr_width}"
    )
    npt.assert_array_equal(
        payload.conn_attrs[:, :, 0], weights, err_msg="conn_attrs[..., 0] must be the decoded weight"
    )


def test_conn_keys_appends_the_row_id_to_every_extra_key_column(hyper):
    """``_conn_keys(P, Q, key_width)`` fills columns ``>= 2`` with the unique row id ``1..Q``.

    The shipped gene has ``key_width == 2`` and no extra column, so the documented
    ``OriginConn``-style fallback is exercised by asking for two extra columns at a time."""
    pop_size, conns_cnt, key_width = 2, CONNS_CNT, 4
    keys = run_method(hyper, "_conn_keys", pop_size, conns_cnt, key_width)
    assert keys.shape == (pop_size, conns_cnt, key_width), f"{keys.shape}"
    assert keys.dtype == np.int32, f"{keys.dtype}"

    row_ids = np.arange(1, conns_cnt + 1, dtype=np.int32)
    expected = np.concatenate(
        [
            np.broadcast_to(DEFAULT_CONN_INDEX[None, :, :], (pop_size, conns_cnt, 2)),
            np.broadcast_to(row_ids[None, :, None], (pop_size, conns_cnt, 2)),
        ],
        axis=2,
    )
    npt.assert_array_equal(
        keys, expected, err_msg="columns 0/1 are the endpoints, every extra column the id 1..Q"
    )
    for individual in range(pop_size):
        assert len(set(keys[individual, :, 2].tolist())) == conns_cnt, (
            f"the extra key column must be unique per row, got {keys[individual, :, 2].tolist()}"
        )


# --------------------------------------------------------------------------- E. bounds


def test_static_bounds_of_the_default_full_substrate(hyper):
    """``FullSubstrate(2 inputs, 1 hidden, 1 output)`` needs ``K = 3`` and no depth bound.

    The hand in-degrees of ``DEFAULT_CONN_INDEX`` are ``[0, 0, 1, 3]``, so
    ``max(bincount(dst).max(), 1, num_inputs) = max(3, 1, 2) = 3``; the recurrent executor ticks
    ``activate_time`` instead of unrolling a depth, so ``max_depth`` must be absent."""
    in_degree = np.bincount(DEFAULT_CONN_INDEX[:, 1], minlength=NODES_CNT)
    assert in_degree.tolist() == [0, 0, 1, 3], f"hand in-degree count, got {in_degree.tolist()}"
    assert hyper._static_bounds() == {"max_in_degree": 3}, f"{hyper._static_bounds()}"
    assert hyper._required_connection_type is None, "the recurrent variant accepts any connection_type"


def test_static_bounds_of_a_wider_full_substrate():
    """A ``FullSubstrate`` with 3 inputs and 2 hidden neurons needs ``K = 5``.

    Each hidden neuron receives from the 3 inputs and from both hidden neurons (its own
    self-loop included), above the ``num_inputs = 3`` floor."""
    substrate = FullSubstrate(
        input_coors=((-1, -1), (0, -1), (1, -1)), hidden_coors=((-1, 0), (0, 0)), output_coors=((0, 1),)
    )
    algorithm = HyperNEAT(substrate, make_neat())
    in_degree = np.bincount(np.asarray(substrate.conn_index)[:, 1], minlength=6)
    assert int(in_degree.max()) == 5, f"hand in-degree count peaks at 5, got {in_degree.tolist()}"
    assert algorithm._static_bounds() == {"max_in_degree": 5}, f"{algorithm._static_bounds()}"
    assert algorithm.hyper_genome.max_in_degree == 5, f"{algorithm.hyper_genome.max_in_degree}"


def test_static_bounds_of_an_mlp_substrate_include_the_depth():
    """``MLPSubstrate(layers=[3, 2, 1])`` needs ``K = 3`` and ``max_depth == 2``.

    Rows are ``0, 1, 2`` (inputs), ``3`` (output) and ``4, 5`` (hidden), wired input -> hidden
    then hidden -> output: a hidden neuron receives the 3 inputs and the longest input-to-node
    path is input -> hidden -> output. Only the feedforward executor unrolls that depth."""
    substrate = MLPSubstrate(layers=[3, 2, 1])
    assert np.asarray(substrate.conn_index).tolist() == [
        [0, 4], [0, 5], [1, 4], [1, 5], [2, 4], [2, 5], [4, 3], [5, 3],
    ], f"hand topology of MLPSubstrate([3, 2, 1]), got {np.asarray(substrate.conn_index).tolist()}"
    feedforward = HyperNEATFeedForward(substrate, make_neat())
    assert feedforward._static_bounds() == {"max_in_degree": 3, "max_depth": 2}, (
        f"{feedforward._static_bounds()}"
    )
    assert feedforward.hyper_genome.max_depth == 2, f"{feedforward.hyper_genome.max_depth}"
    recurrent = HyperNEAT(substrate, make_neat())
    assert "max_depth" not in recurrent._static_bounds(), (
        f"the recurrent executor takes a feedforward substrate but passes no depth: {recurrent._static_bounds()}"
    )


def test_make_hyper_genome_mirrors_the_substrate_and_the_configured_genes():
    """The executor genome is dense over the substrate and carries the fixed gene options.

    ``max_nodes``/``max_conns`` are the substrate's counts (no NaN padding rows), the widths are
    the *substrate's* and the genes are ``HyperNEATNode`` (two option columns) /
    ``HyperNEATConn`` (two endpoint columns, one ``weight`` column at index 0)."""
    algorithm = make_hyper()
    genome, substrate = algorithm.hyper_genome, algorithm.substrate
    assert genome.max_nodes == substrate.nodes_cnt == NODES_CNT, f"{genome.max_nodes}"
    assert genome.max_conns == substrate.conns_cnt == CONNS_CNT, f"{genome.max_conns}"
    assert genome.num_inputs == substrate.num_inputs == 2, f"{genome.num_inputs}"
    assert genome.num_outputs == substrate.num_outputs == 1, f"{genome.num_outputs}"

    node_gene = genome.node_gene
    assert isinstance(node_gene, HyperNEATNode), f"{type(node_gene).__name__}"
    assert node_gene.aggregation is AGG.sum, f"{node_gene.aggregation!r}"
    assert node_gene.activation is ACT.tanh, f"{node_gene.activation!r}"
    assert node_gene.attr_width == 2 and node_gene.key_width == 1, (
        f"attr_width {node_gene.attr_width} / key_width {node_gene.key_width}"
    )

    conn_gene = genome.conn_gene
    assert isinstance(conn_gene, HyperNEATConn), f"{type(conn_gene).__name__}"
    assert conn_gene.key_width == 2 and conn_gene.attr_width == 1, (
        f"key_width {conn_gene.key_width} / attr_width {conn_gene.attr_width}"
    )
    assert conn_gene.weight_index == 0, (
        f"the weight column is resolved by spec name and is the only one, got {conn_gene.weight_index}"
    )

    recurrent = make_hyper(substrate=_flat_substrate())
    assert recurrent._genome_extra_kwargs() == {"activate_time": recurrent.activate_time}, (
        f"the recurrent variant forwards activate_time, got {recurrent._genome_extra_kwargs()}"
    )
    feedforward = HyperNEATFeedForward(MLPSubstrate(layers=[3, 2, 1]), make_neat())
    assert feedforward._genome_extra_kwargs() == {}, f"{feedforward._genome_extra_kwargs()}"


# --------------------------------------------------------------------------- F. validate


def test_validate_accepts_the_untampered_geometry():
    """Control: the table the tampering helper starts from is accepted.
    Control: the table the tampering helper starts from is accepted."""
    assert HyperNEAT(_tampered_substrate(), make_neat()).substrate.nodes_cnt == NODES_CNT


def test_validate_rejects_a_cppn_query_width_mismatch():
    """The CPPN input width must equal ``query_coors.shape[1]`` (the legacy wording).
    The CPPN input width must equal ``query_coors.shape[1]`` (the legacy wording)."""
    with pytest.raises(TensorNEATError, match="Query coors of Substrate should be equal to NEAT input size"):
        HyperNEAT(_tampered_substrate(), make_neat(genome=make_cppn_genome(num_inputs=QUERY_WIDTH - 1)))


@pytest.mark.parametrize("threshold", [-0.1, 1.0, 2.5])
def test_validate_rejects_a_weight_threshold_outside_the_unit_interval(threshold):
    """``0 <= weight_threshold < 1``: the weight map divides by ``1 - weight_threshold``.
    ``0 <= weight_threshold < 1``: the weight map divides by ``1 - weight_threshold``."""
    with pytest.raises(TensorNEATError, match="weight_threshold must be in"):
        HyperNEAT(_tampered_substrate(), make_neat(), weight_threshold=threshold)


@pytest.mark.parametrize(
    "kwargs",
    [{"weight_threshold": True}, {"weight_threshold": "0.3"}, {"weight_threshold": None}]
    + [{"max_weight": True}, {"max_weight": "5.0"}],
)
def test_validate_rejects_a_non_real_threshold_or_weight(kwargs):
    """A ``bool`` (an ``int`` subclass), a string or ``None`` fails the real-number check.
    A ``bool`` (an ``int`` subclass), a string or ``None`` fails the real-number check."""
    with pytest.raises(TensorNEATError, match="must be a real number"):
        HyperNEAT(_tampered_substrate(), make_neat(), **kwargs)


@pytest.mark.parametrize("max_weight", [0.0, -2.0])
def test_validate_rejects_a_non_positive_max_weight(max_weight):
    """``max_weight`` is a saturating magnitude, so it must be strictly positive.
    ``max_weight`` is a saturating magnitude, so it must be strictly positive."""
    with pytest.raises(TensorNEATError, match="max_weight must be > 0"):
        HyperNEAT(_tampered_substrate(), make_neat(), max_weight=max_weight)


def test_validate_rejects_a_substrate_without_a_bias_input():
    """``substrate.num_inputs >= 2``: the last substrate input is the bias neuron.

    ``DefaultSubstrate(1, 1, ...)`` is a legal substrate, so the rejection is ``_validate``'s."""
    substrate = DefaultSubstrate(
        1, 1, np.zeros((1, QUERY_WIDTH), np.float32), np.array([0, 1]), np.array([[0, 1]])
    )
    with pytest.raises(TensorNEATError, match="substrate.num_inputs must be >= 2"):
        HyperNEAT(substrate, make_neat())


@pytest.mark.parametrize("kwargs", [{"activate_time": 0}, {"activate_time": -3}])
def test_validate_rejects_a_non_positive_activate_time(kwargs):
    """``activate_time >= 1``: zero ticks would read out the ``NaN`` seed.
    ``activate_time >= 1``: zero ticks would read out the ``NaN`` seed."""
    with pytest.raises(TensorNEATError, match="activate_time must be >= 1"):
        HyperNEAT(_tampered_substrate(), make_neat(), **kwargs)


@pytest.mark.parametrize("activate_time", [True, "3"])
def test_validate_rejects_a_non_integer_activate_time(activate_time):
    """A ``bool`` or a string fails ``_require_real`` ("a real number").
    A ``bool`` or a string fails ``_require_real`` ("a real number")."""
    with pytest.raises(TensorNEATError, match="activate_time must be a real number"):
        HyperNEAT(_tampered_substrate(), make_neat(), activate_time=activate_time)


def test_validate_rejects_a_fractional_activate_time():
    """``1.5`` is real, so the *genome* layer rejects it: ticks are a positive int.

    ``_require_real`` lets it through and the genome's ``_require_positive_int`` raises, so the
    two layers split the "real" and "integral" halves of the documented contract."""
    with pytest.raises(TensorNEATError, match="activate_time must be a positive int"):
        HyperNEAT(_tampered_substrate(), make_neat(), activate_time=1.5)


def test_validate_rejects_a_multi_output_cppn():
    """A CPPN has exactly one output -- the connection weight.
    A CPPN has exactly one output -- the connection weight."""
    with pytest.raises(TensorNEATError, match="a HyperNEAT CPPN must have exactly one output"):
        HyperNEAT(_tampered_substrate(), make_neat(genome=make_cppn_genome(num_outputs=2)))


@pytest.mark.parametrize(
    "tampering, match",
    [
        ({"_query_coors": np.zeros((3, QUERY_WIDTH), np.float32)}, "query_coors has 3 rows"),
        ({"_conn_index": np.zeros((CONNS_CNT, 3), np.int32)}, "conn_index must be"),
        ({"_nodes": np.zeros((NODES_CNT, 1), np.int32)}, "substrate.nodes must be"),
        ({"_nodes": np.array([0, 1, 2, 4], np.int32)}, "must be a permutation of"),
    ],
)
def test_validate_rejects_inconsistent_substrate_tables(tampering, match):
    """Each case tampers with one table, so the failure names that table.
    Each case tampers with one table, so the failure names that table."""
    with pytest.raises(TensorNEATError, match=match):
        HyperNEAT(_tampered_substrate(**tampering), make_neat())


def test_validate_rejects_a_wrong_node_key_block_order():
    """The node keys must be ``inputs 0..I-1`` then ``outputs I..I+O-1``, hidden after.

    ``[0, 2, 1, 3]`` swaps output and hidden, and the genome's ``is_output_node`` plus its
    ``input_idx``/``output_idx`` ranges depend on that block order."""
    with pytest.raises(TensorNEATError, match="substrate node keys must be the block order"):
        HyperNEAT(_tampered_substrate(_nodes=np.array([0, 2, 1, 3], np.int32)), make_neat())


def test_validate_rejects_a_substrate_of_the_wrong_connection_type():
    """``HyperNEATFeedForward`` requires ``connection_type == "feedforward"``; the harness
    ``FullSubstrate`` is recurrent, so ``_required_connection_type`` is the failing code."""
    substrate = FullSubstrate(
        input_coors=INPUT_COORS, hidden_coors=HIDDEN_COORS, output_coors=OUTPUT_COORS
    )
    assert substrate.connection_type == "recurrent", f"{substrate.connection_type!r}"
    with pytest.raises(
        TensorNEATError,
        match="HyperNEATFeedForward requires a substrate with connection_type='feedforward'",
    ):
        HyperNEATFeedForward(substrate, make_neat())


# --------------------------------------------------------------------------- G. protocol


def test_protocol_arities_come_from_the_substrate(hyper):
    """``num_inputs = I - 1`` (bias removed), ``num_outputs = O``, ``pop_size`` a property.
    ``num_inputs = I - 1`` (bias removed), ``num_outputs = O``, ``pop_size`` a property."""
    assert hyper.num_inputs == hyper.substrate.num_inputs - 1 == 1, f"{hyper.num_inputs}"
    assert hyper.num_outputs == hyper.substrate.num_outputs == 1, f"{hyper.num_outputs}"
    assert hyper.pop_size == hyper.neat.pop_size == POP_SIZE, f"{hyper.pop_size}"
    assert isinstance(type(hyper).pop_size, property), (
        "pop_size must be the base protocol's read-only property, not an instance attribute"
    )


def test_ask_returns_the_cppn_population_unchanged(hyper, state):
    """``ask`` is a zero-op delegation: the very same payload object comes back.
    ``ask`` is a zero-op delegation: the very same payload object comes back."""
    asked = hyper.ask(state)
    assert asked is state.pop, f"ask must return the state's payload, got a new {type(asked).__name__}"
    assert hyper.neat.ask(state) is asked, "ask must be a pure delegation to neat.ask"
    for name in GenomeTensors._fields:
        npt.assert_array_equal(
            to_numpy(getattr(asked, name)), to_numpy(getattr(state.pop, name)), err_msg=name
        )
        assert to_numpy(getattr(asked, name)).dtype == to_numpy(getattr(state.pop, name)).dtype, name


def test_setup_returns_a_state_identical_to_the_cppn_setup(hyper):
    """``setup`` chains ``neat.setup``; the substrate adds no state of its own.

    Two runs of one seed must agree array for array, including the advanced ``randkey``."""
    first = hyper.setup(State(randkey=evoxc.random.key(0)))
    second = hyper.setup(State(randkey=evoxc.random.key(0)))
    cppn_only = hyper.neat.setup(State(randkey=evoxc.random.key(0)))
    assert isinstance(first, State), f"setup must return a State, got {type(first).__name__}"
    for name in GenomeTensors._fields:
        npt.assert_array_equal(
            to_numpy(getattr(first.pop, name)),
            to_numpy(getattr(second.pop, name)),
            err_msg=f"two setups from one seed must agree on pop.{name}",
        )
        npt.assert_array_equal(
            to_numpy(getattr(first.pop, name)),
            to_numpy(getattr(cppn_only.pop, name)),
            err_msg=f"pop.{name} must be the CPPN setup's",
        )
    npt.assert_array_equal(to_numpy(first.generation), to_numpy(cppn_only.generation))
    npt.assert_array_equal(to_numpy(first.randkey), to_numpy(cppn_only.randkey))


def test_tell_delegates_to_the_cppn_algorithm(hyper, state, monkeypatch):
    """``tell`` forwards ``(state, fitness)`` to ``neat.tell`` and returns its result unchanged.

    ``NEAT.tell``'s own graph recipe lives in ``./test_neat_tell.py``; the delegation is pinned
    here so a ``HyperNEAT.tell`` that re-implements selection would be caught."""
    sentinel = object()
    seen = {}

    def fake_tell(current, fitness):
        seen["state"], seen["fitness"] = current, fitness
        return sentinel

    monkeypatch.setattr(hyper.neat, "tell", fake_tell)
    fitness = np.array([0.1, 0.5, 0.2, 0.9], dtype=np.float32)
    assert hyper.tell(state, fitness) is sentinel, "tell must return neat.tell's result unchanged"
    assert seen["state"] is state and seen["fitness"] is fitness, "tell must forward both arguments"


def test_show_details_delegates_to_the_cppn_algorithm(hyper, state, monkeypatch):
    """``show_details`` forwards to ``neat.show_details``: the CPPN owns the report.
    ``show_details`` forwards to ``neat.show_details``: the CPPN owns the report."""
    seen = {}

    def fake_show(current, fitness):
        seen["state"], seen["fitness"] = current, fitness

    monkeypatch.setattr(hyper.neat, "show_details", fake_show)
    fitness = np.array([0.1, 0.5, 0.2, 0.9], dtype=np.float32)
    assert hyper.show_details(state, fitness) is None, "show_details is host-side and returns None"
    assert seen["state"] is state and seen["fitness"] is fitness, "it must forward both arguments"


def test_substrate_schedule_level_and_overflow_delegate_to_the_transformed(decoded):
    """``SubstrateSchedule.level`` / ``.overflow`` alias the wrapped ``Transformed``'s own.

    ``level`` is the very same ``node_level`` object and ``overflow`` is ``edge_overflow > 0``,
    so neither is a recomputed copy."""
    schedule = decoded.schedule
    assert isinstance(decoded, SubstrateSchedule), f"{type(decoded).__name__}"
    assert decoded.level is schedule.level, "level must be the very same node_level object"
    assert decoded.level is schedule.node_level, "Transformed.level is node_level"
    npt.assert_array_equal(
        decoded.overflow, np.asarray(schedule.edge_overflow) > 0, err_msg="overflow = edge_overflow > 0"
    )
    assert np.asarray(schedule.edge_overflow).tolist() == [0] * POP_SIZE, (
        f"a dense substrate payload drops no connection, got {np.asarray(schedule.edge_overflow).tolist()}"
    )
    assert not np.any(decoded.overflow), "a lossless payload has no overflow"


# --------------------------------------------------------------------------- H. transform


def test_transform_returns_a_population_leading_substrate_schedule(decoded):
    """Every payload and schedule leaf keeps the population on axis 0.

    ``node_keys (P, N)``, ``node_attrs (P, N, 2)``, ``conn_keys (P, Q, 2)``, ``conn_attrs
    (P, Q, 1)``; the substrate is dense, so no NaN padding row exists."""
    expected = {
        "node_keys": (POP_SIZE, NODES_CNT),
        "node_attrs": (POP_SIZE, NODES_CNT, 2),
        "conn_keys": (POP_SIZE, CONNS_CNT, 2),
        "conn_attrs": (POP_SIZE, CONNS_CNT, 1),
    }
    for name, shape in expected.items():
        got = np.asarray(getattr(decoded, name)).shape
        assert got == shape, f"{name} must be batch-first at P = {POP_SIZE}: {shape} != {got}"
    for name in type(decoded.schedule)._fields:
        leaf = np.asarray(getattr(decoded.schedule, name))
        assert leaf.ndim >= 1 and int(leaf.shape[0]) == POP_SIZE, f"schedule field {name}: {leaf.shape}"
    npt.assert_array_equal(
        np.asarray(decoded.node_keys),
        np.broadcast_to(DEFAULT_NODES[None, :], (POP_SIZE, NODES_CNT)),
        err_msg="the decoded node rows are the substrate's node table",
    )


def test_transform_schedule_implies_the_substrate_topology(decoded):
    """The schedule describes exactly the hand-transcribed topology.

    Per connection row ``q``: ``src[q] = index(source)`` and ``dst[q] = index(target)`` in
    ``DEFAULT_NODES``, plus ``input_slot`` naming the slots of keys ``0, 1``. ``output_slot``'s
    *value* is the defect pinned in the forward section."""
    schedule = decoded.schedule
    slots = {int(key): slot for slot, key in enumerate(DEFAULT_NODES)}
    expected_src = np.array([slots[int(source)] for source, _ in DEFAULT_CONN_INDEX])
    expected_dst = np.array([slots[int(target)] for _, target in DEFAULT_CONN_INDEX])
    for individual in range(POP_SIZE):
        npt.assert_array_equal(
            np.asarray(schedule.src)[individual, :CONNS_CNT], expected_src, err_msg=f"row {individual} src"
        )
        npt.assert_array_equal(
            np.asarray(schedule.dst)[individual, :CONNS_CNT], expected_dst, err_msg=f"row {individual} dst"
        )
        assert np.all(np.asarray(schedule.node_exists)[individual]), "a dense payload has no padding row"
    npt.assert_array_equal(
        np.asarray(schedule.input_slot)[0],
        np.array([slots[0], slots[1]]),
        err_msg="input_slot names the substrate's input keys 0, 1",
    )
    assert np.asarray(schedule.output_slot).shape == (POP_SIZE, 1), (
        f"output_slot names one node per substrate output, got {np.asarray(schedule.output_slot).shape}"
    )


def test_transform_decodes_a_single_individual_exactly_like_its_population_row(hyper, state):
    """A one-individual decode equals that individual's rows of a ``P = 4`` decode.

    That is the ``Pipeline.show`` contract: the batch is read off the payload, so a decode sized
    from ``self.pop_size`` could not satisfy both."""
    payload = cppn_payload(state)
    full = run_method(hyper, "transform", payload, state=state)
    for individual in range(POP_SIZE):
        single_payload = GenomeTensors(
            *(np.asarray(getattr(payload, name))[individual : individual + 1] for name in GenomeTensors._fields)
        )
        single = run_method(hyper, "transform", single_payload, state=state)
        for name in GenomeTensors._fields:
            got = np.asarray(getattr(single, name))
            assert got.shape[0] == 1, f"a one-individual decode keeps a leading axis of 1 on {name}: {got.shape}"
            npt.assert_array_equal(
                got[0], np.asarray(getattr(full, name))[individual], err_msg=f"{name} of individual {individual}"
            )
        for name in type(full.schedule)._fields:
            npt.assert_array_equal(
                np.asarray(getattr(single.schedule, name))[0],
                np.asarray(getattr(full.schedule, name))[individual],
                err_msg=f"schedule field {name} of individual {individual}",
            )


# --------------------------------------------------------------------------- I. forward


def test_forward_appends_the_trailing_bias_and_runs_the_documented_ticks(flat, flat_state, flat_decoded):
    """``forward`` reproduces the hand tick algebra of the flat substrate.

    The bias column of ones is appended **last**, the tick-1 self-loop source is ``NaN`` (not
    live), so ``v3 = w0*a + w1 + w2*(w0*a + w1 + w2*(w0*a + w1))`` and the output is
    ``sigmoid(v3)`` -- with ``sigmoid(-1.3125)`` pinned for ``a = 0`` and ``(0.5, -1.0, 0.25)``."""
    assert flat.hyper_genome.activate_time == 3, f"{flat.hyper_genome.activate_time}"
    assert flat.activation is ACT.tanh, f"{flat.activation!r}"
    payload = _rescheduled(flat_decoded, FLAT_WEIGHTS)
    inputs = np.array([[0.0], [0.7], [-0.3], [1.5]], dtype=np.float32)
    got = run_method(flat, "forward", payload, inputs, state=flat_state)

    assert got.shape == (POP_SIZE, 1), f"forward returns (B, O) = {(POP_SIZE, 1)}, got {got.shape}"
    npt.assert_allclose(
        got,
        _flat_expected_rows(inputs, FLAT_WEIGHTS, samples=1, ticks=3),
        rtol=1e-6,
        atol=0,
        err_msg="forward must equal the documented tick algebra, bias column appended last",
    )
    npt.assert_allclose(
        got[0, 0],
        1.0 / (1.0 + np.exp(1.3125)),
        rtol=1e-6,
        atol=0,
        err_msg="a = 0 with (0.5, -1.0, 0.25) must give sigmoid(-1.3125)",
    )


def test_prepare_forward_and_forward_prepared_reproduce_forward_bit_for_bit(flat, flat_state, flat_decoded):
    """The prepared pair is ``forward`` minus its prelude: the same bits, same hand values.

    Both halves run in one graph (:func:`_run_prepared_pair`) because the prepared record is not
    a legal graph output."""
    payload = _rescheduled(flat_decoded, FLAT_WEIGHTS)
    inputs = np.array([[0.0], [0.7], [-0.3], [1.5]], dtype=np.float32)
    direct = run_method(flat, "forward", payload, inputs, state=flat_state)
    prepared = np.asarray(_run_prepared_pair(flat, flat_state, payload, inputs, samples=1))

    assert prepared.shape == (POP_SIZE, 1), f"forward_prepared returns (B, O), got {prepared.shape}"
    npt.assert_array_equal(prepared, direct, err_msg="prepare_forward + forward_prepared == forward")
    npt.assert_allclose(
        prepared,
        _flat_expected_rows(inputs, FLAT_WEIGHTS, samples=1, ticks=3),
        rtol=1e-6,
        atol=0,
        err_msg="the prepared pair must also match the hand tick algebra",
    )


def test_forward_tiles_the_population_axis_individual_major(flat, flat_state, flat_decoded):
    """``B = P * samples`` is individual-major: row ``p*samples + j`` uses individual p's weights.

    The weights differ per individual, so a sample-major (``j*P + p``) tiling cannot match, and
    the prepared pair must agree bit for bit at that width too."""
    payload = _rescheduled(flat_decoded, FLAT_WEIGHTS)
    samples = 2
    inputs = np.array([[0.0], [0.7], [-0.3], [1.5], [1.0], [-1.0], [0.5], [2.0]], dtype=np.float32)
    assert inputs.shape[0] == POP_SIZE * samples, "the tiled batch is pop_size * samples rows"
    got = run_method(flat, "forward", payload, inputs, state=flat_state)

    assert got.shape == (POP_SIZE * samples, 1), f"{(POP_SIZE * samples, 1)} != {got.shape}"
    npt.assert_allclose(
        got,
        _flat_expected_rows(inputs, FLAT_WEIGHTS, samples=samples, ticks=3),
        rtol=1e-6,
        atol=0,
        err_msg="row p*samples + j must use individual p's weights (individual-major tiling)",
    )
    prepared = np.asarray(_run_prepared_pair(flat, flat_state, payload, inputs, samples=samples))
    npt.assert_array_equal(prepared, got, err_msg="the prepared pair at a tiled width too")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "src defect: HyperNEAT.forward reads the node rows max_nodes-num_outputs..max_nodes-1 "
        "(genome/base.py output_idx, genome/step.py _output_values), but HyperNEAT._validate "
        "(hyperneat.py:491) requires the substrate node order inputs, outputs, hidden and "
        "_assemble_substrate (hyperneat.py:798) broadcasts that order verbatim -- so with a "
        "hidden block the read-out row is a hidden neuron, the real output neuron is "
        "activated like a hidden one, and the last connection's weight never reaches the output"
    ),
)
def test_forward_reads_the_substrate_output_neurons(hyper, state, decoded):
    """``forward`` must return the substrate's output neurons (keys ``I .. I+O-1``).

    Documented: the hidden neuron takes ``ACT.tanh`` and the output neuron is unactivated, so the
    result is ``sigmoid(w3 * tanh(w0*a + w1 + w2*tanh(w0*a + w1)))``."""
    payload = _rescheduled(decoded, np.broadcast_to(HIDDEN_WEIGHTS, (POP_SIZE, HIDDEN_WEIGHTS.size)))
    inputs = np.array([[0.7], [-0.3], [0.0], [1.5]], dtype=np.float32)
    slots = {int(key): slot for slot, key in enumerate(DEFAULT_NODES)}
    npt.assert_array_equal(
        np.asarray(decoded.schedule.output_slot)[0],
        np.array([slots[2]]),
        err_msg="root cause: output_slot must name the output block's node (key 2), not a hidden row",
    )
    got = run_method(hyper, "forward", payload, inputs, state=state)
    npt.assert_allclose(
        got,
        _hidden_expected_rows(inputs, HIDDEN_WEIGHTS, ticks=3),
        rtol=1e-6,
        atol=0,
        err_msg=(
            "forward must read out the substrate output neuron (key 2). Observed instead the "
            "hidden row's unactivated self-loop sum w0*a + w1 + w2*(w0*a + w1 + w2*(w0*a + w1)) "
            "-- sigmoid(-1.3125) = 0.2120688 at a = 0 -- with w3 never applied"
        ),
    )
