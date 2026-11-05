"""The seam half of the ``tensorization`` suites: the ``acyclic`` switch and every refusal.

``derive_schedule`` has a host-side *refusal block* (five statics, the three tensor
shapes, the two key-vector lengths and the key dtype) and two legal spellings of its
level derivation — ``acyclic=False`` (the default, which detects a live cycle and
reports every node on or downstream of it as ``I_INF``) and ``acyclic=True`` (which
asserts the population is cycle-free and drops the test plus its fallback campaign).
Neither seam is reachable through :mod:`unit.common.test_tensorization`, which only
ever calls the default path on a well-formed payload, so all of them live here.

What this module proves:

* on a cycle-free population — the canonical chain, the bypassed-node topology and
  a two-member chain — the two spellings derive the *same* twelve fields bit for
  bit, and both equal ``support.oracle.schedule``: ``acyclic=True`` drops the
  cycle-detection work, never the answer;
* on a cyclic population (the recurrent ring) the two disagree exactly where the
  documentation says they must. ``acyclic=False`` reports ``I_INF`` for both ring
  nodes and for the output row downstream of them; ``acyclic=True`` hands those
  rows a saturated *finite* level (each capped relaxation step raises a level by
  one, so ``max_nodes`` bounds it) and a ``topo_key`` computed from it. The result
  is a *silently wrong* schedule — it still passes ``validate_schedule`` — which is
  why ``acyclic=True`` is only legal for a genome family that guarantees
  acyclicity (``DefaultGenome``, whose structural gates refuse a cycle) and never
  for ``RecurrentGenome``, which is defined by having one;
* ``derive_schedule`` refuses every documented bad static, bad shape, oversized
  key vector and mixed key spelling with a ``ConfigError`` that names the offending
  argument — and each of those is a ``TensorNEATError`` (the hierarchy);
* ``node_levels`` refuses, in its own words, a missing ``(src, dst, edge_valid)``
  triple, an edge array whose population axis disagrees with ``in_conn``, one
  whose connection axis disagrees with ``src``, and an ``in_conn`` whose node axis
  disagrees with ``max_nodes``.

Oracles (never the kernel itself): the ``acyclic=False`` schedule of the sibling
module, compared field by field, and ``support.oracle.schedule`` — the independent
numpy derivation, whose level rule *is* the cycle-detecting one, so it agrees with
``acyclic=False`` on every payload (the ring included) and dissents on the ring
exactly where ``acyclic=True`` does; the payload's decoded edge list for the
"a saturated level admits a downhill edge" check; hand enumeration
(``[0, 0, 1, 2]``) and the payload's own statics for the ``node_levels`` guard
cases; and the documented message of each refusal, matched verbatim.

Gotchas: ``derive_schedule`` is an ``@evoxc.defn`` whose ``__call__`` refuses eager
use and whose statics are positional, so :func:`_call` is this module's single raw
call site (the sibling's ``_schedule`` derives every static from the payload, so it
cannot express a *deliberately* wrong one, and it takes no ``acyclic`` flag). The
refusal block raises at trace time, before anything compiles. ``node_levels`` is a
plain helper, not a ``@evoxc.defn``: it only runs inside an active trace, and its
edge triple must cross the graph boundary as tensor *arguments* — a host numpy
``src`` closed over instead reaches ``evoxc.minimum`` as a bare array and raises a
``TypeError`` before any guard can fire (:func:`_levels` forwards them). Every
refusal case changes exactly one argument and is paired with the untouched call
(the ``fixed_call`` / ``levels_control`` fixtures), so none can pass on an
unrelated error.

Scope: this is the *third* module of the ``tensorization`` family — the semantics
live in :mod:`unit.common.test_tensorization` and the host checker in
:mod:`unit.common.test_tensorization_validate`. It owns no fixture the other two
use; it imports their payload fixtures and helpers so the payloads and the derived
schedules are shared, not rebuilt. Backend is ``"numpy"``, no
``jax``/``torch`` import, no network, and the shapes stay small (``P <= 2``,
``max_nodes == 8`` where a real payload is used, ``N == 4`` for the hand-built
``node_levels`` tables).
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from support.runner import run_defn

# Shared fixtures, constants and the two conversion helpers, imported from the
# semantics half of the family: pytest resolves a fixture's own dependencies in the
# namespace of the module that *uses* it, so the payload fixtures travel with them.
from .test_tensorization import (
    DOCUMENTED_FIELDS,
    NUM_INPUTS,
    NUM_OUTPUTS,
    _decode,
    _float_keys,
    _key_span,
    chain_payload,
    chain_schedule,
    ring_payload,
    ring_schedule,
    skip_payload,
    skip_schedule,
)

#: The canonical chain's levels, hand-enumerated from its row geometry: two inputs at
#: level ``0`` and one level per hidden row after them (``payloads.chain_payload``).
_CHAIN_LEVELS = np.array([[0, 0, 1, 2, 3, 4, 5, 6]], dtype=np.int32)

#: The five trace-time statics whose positive-int check shares one message.
_POSITIVE_STATICS = ("pop_size", "max_nodes", "max_conns", "max_in_degree", "key_span")


# ---- helpers: the raw positional call and the traced `node_levels` call site -----------------


def _call(common, payload, *, acyclic=False, **overrides):
    """Call ``derive_schedule`` once on ``payload``, every static at its payload value.

    ``derive_schedule`` is an ``@evoxc.defn``, so its own ``__call__`` refuses eager
    use, and its five statics are positional-or-keyword: this is the module's one raw
    call site, mirroring ``derive_schedule``'s signature order. The sibling's
    ``_schedule`` derives every static from the payload (that is what the semantics
    tests want) and therefore cannot express a deliberately wrong one; the refusal
    table below needs exactly that, and the ``acyclic`` seam needs the eleventh
    argument. On the chain payload the two build the same call — the equivalence test
    asserts that field for field.

    Args:
        common: The ``tensorneat.common`` package fixture.
        payload: One ``support.payloads.Payload`` (a whole population).
        acyclic: The static switch under test, baked into the trace as a Python bool.
        **overrides: Replacement values for named positional arguments (the five
            arrays and the five statics), handed to ``run_defn`` in signature order.

    Returns:
        The ``Transformed`` schedule, as host numpy arrays.
    """
    i_inf = common.I_INF
    node_keys = np.asarray(payload.node_keys)
    conn_keys = np.asarray(payload.conn_keys)
    arguments = {
        "node_keys": node_keys,
        "conn_in": conn_keys[..., 0].copy(),
        "conn_out": conn_keys[..., 1].copy(),
        "input_keys": node_keys[0, :NUM_INPUTS].copy(),
        "output_keys": node_keys[0, payload.max_nodes - NUM_OUTPUTS :].copy(),
        "pop_size": payload.pop_size,
        "max_nodes": payload.max_nodes,
        "max_conns": payload.max_conns,
        "max_in_degree": payload.max_nodes,
        "key_span": _key_span(payload, i_inf),
    }
    unknown = set(overrides) - set(arguments)
    assert not unknown, f"no such derive_schedule argument: {sorted(unknown)}"
    arguments.update(overrides)
    return run_defn(
        common.tensorization.derive_schedule, *arguments.values(), acyclic
    )


def _levels(common, tables, *, acyclic=False):
    """Run ``node_levels`` inside a traced closure, the way ``derive_schedule`` calls it.

    ``node_levels`` is a plain helper, not an ``@evoxc.defn``: it only works during an
    active trace, so the closure is its call site. The ``(src, dst, edge_valid)``
    triple must arrive as tensor *arguments* — a host numpy ``src`` closed over
    instead reaches ``evoxc.minimum`` as a bare array and raises ``TypeError`` before
    any guard could fire — so the closure forwards whatever edge arrays it is given.

    Args:
        common: The ``tensorneat.common`` package fixture.
        tables: A ``SimpleNamespace`` with ``in_conn``/``in_src``/``input_slot`` (the
            traced inputs), ``max_nodes`` (a static) and ``edges`` — the
            ``(src, dst, edge_valid)`` triple, or ``None``.
        acyclic: The static switch, as in ``node_levels``.

    Returns:
        The ``(P, N)`` ``int32`` levels, as a host array.
    """

    def graph(in_conn, in_src, input_slot, *edge_arrays):
        triple = tuple(edge_arrays) if edge_arrays else None
        return common.tensorization.node_levels(
            in_conn, in_src, input_slot, tables.max_nodes, acyclic, triple
        )

    arguments = [tables.in_conn, tables.in_src, tables.input_slot]
    arguments.extend(tables.edges or ())
    return run_defn(graph, *arguments)


def _variant(tables, **overrides):
    """A copy of ``tables`` with named fields replaced — one field per guard case."""
    fields = dict(vars(tables))
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _stack(array):
    """Two copies of ``array`` along the population axis — a ``P == 2`` tamper."""
    return np.concatenate([array, array], axis=0)


# ---- fixtures: the well-formed control and the hand-built `node_levels` tables ----------------


@pytest.fixture(scope="module")
def fixed_call(common, chain_payload):
    """The chain payload's call with *no* argument overridden — the refusal control.

    Every refusal below changes exactly one argument of :func:`_call`, so this is
    that call with the offending argument changed back. Its levels are the chain's
    hand-enumerated ``[0, 0, 1, 2, 3, 4, 5, 6]`` and ``validate_schedule`` accepts
    it, so a refusal cannot be an unrelated error.
    """
    schedule = _call(common, chain_payload)
    assert common.tensorization.validate_schedule(schedule) is None
    assert np.array_equal(schedule.node_level, _CHAIN_LEVELS)
    return schedule


@pytest.fixture(scope="module")
def levels_tables(common):
    """The hand-built ``(P=1, N=4, K=2)`` chain ``node_levels`` consumes.

    Topology ``0 → 2``, ``1 → 2``, ``2 → 3`` with ``input_slot == [0, 1]``: node
    ``2``'s window is dense in connection-row order (rows 0 and 1, sources 0 and 1)
    and node ``3``'s is ``(row 2, source 2)``, every other cell ``I_INF`` — the
    padding the kernel documents. Oracle: hand enumeration, so the levels are
    ``[0, 0, 1, 2]``.
    """
    i_inf = common.I_INF
    in_conn = np.full((1, 4, 2), i_inf, dtype=np.int32)
    in_src = np.full((1, 4, 2), i_inf, dtype=np.int32)
    in_conn[0, 2, :] = (0, 1)
    in_src[0, 2, :] = (0, 1)
    in_conn[0, 3, 0] = 2
    in_src[0, 3, 0] = 2
    return SimpleNamespace(
        in_conn=in_conn,
        in_src=in_src,
        input_slot=np.array([[0, 1]], dtype=np.int32),
        max_nodes=4,
        edges=(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[2, 2, 3]], dtype=np.int32),
            np.array([[True, True, True]]),
        ),
    )


@pytest.fixture(scope="module")
def levels_control(common, levels_tables):
    """The untouched ``node_levels`` answer on the hand-built tables: ``[[0, 0, 1, 2]]``.

    The control for the guard cases: the same call with the one tampered argument
    changed back, pinned against hand enumeration rather than against the kernel's
    other spelling.
    """
    expected = np.array([[0, 0, 1, 2]], dtype=np.int32)
    levels = _levels(common, levels_tables)
    assert np.array_equal(levels, expected)
    return expected


# ---- the `acyclic` seam ----------------------------------------------------------------------


def test_acyclic_true_derives_the_same_schedule_as_the_cycle_detecting_path(
    common, oracle, chain_payload, chain_schedule, skip_payload, skip_schedule
):
    """Invariant: on a cycle-free payload the two spellings are the same twelve fields.

    Oracle: two independent statements of the same schedule — the sibling module's
    ``acyclic=False`` build (which this test first proves *is* the same raw call,
    argument for argument) and ``support.oracle.schedule``, the numpy derivation
    whose level rule is the cycle-detecting one. ``acyclic=True`` may drop the cycle
    test and its fallback campaign, but not a single cell of the answer, on the
    chain (every row reachable) and on the bypass topology (a node whose consumer
    reads an input directly, i.e. the case where connection-row order and level
    order disagree).
    """
    i_inf = common.I_INF
    for payload, caught in ((chain_payload, chain_schedule), (skip_payload, skip_schedule)):
        same_call = _call(common, payload)
        fast = _call(common, payload, acyclic=True)

        node_keys = np.asarray(payload.node_keys)
        conn_keys = np.asarray(payload.conn_keys)
        reference = oracle.schedule(
            node_keys,
            conn_keys[..., 0],
            conn_keys[..., 1],
            node_keys[0, :NUM_INPUTS].copy(),
            node_keys[0, -NUM_OUTPUTS:].copy(),
            max_in_degree=payload.max_nodes,
            key_span=_key_span(payload, i_inf),
        )

        for name in DOCUMENTED_FIELDS:
            expected = getattr(reference, name)
            assert np.array_equal(
                getattr(same_call, name), getattr(caught, name)
            ), f"{payload.shape}: {name} differs from the sibling's acyclic=False build"
            assert getattr(fast, name).dtype == expected.dtype, f"{payload.shape}: {name} dtype"
            assert np.array_equal(
                getattr(fast, name), expected
            ), f"{payload.shape}: {name} differs from support.oracle.schedule"
            assert np.array_equal(
                getattr(fast, name), getattr(caught, name)
            ), f"{payload.shape}: {name} differs between acyclic=True and acyclic=False"
        assert common.tensorization.validate_schedule(fast) is None

    # The bypass topology's whole point: row 3 reads input 0 directly, so it sits at
    # level 1 and *not* at level 2, which a naive row-order schedule would give it.
    bypass = _call(common, skip_payload, acyclic=True)
    assert int(bypass.node_level[0, 3]) == 1
    assert int(bypass.node_level[0, 2]) == 1


def test_acyclic_true_is_a_population_wide_switch(common, payloads, oracle):
    """Invariant: on a ``P == 2`` population the flag still changes nothing.

    Oracle: the two spellings plus ``support.oracle.schedule`` on a two-member chain.
    The cycle test is a whole-population reduce in the source, so a member boundary
    is exactly where a per-member fast path would go wrong; both members must come
    out at the chain's hand-enumerated levels.
    """
    payload = payloads.chain_payload(pop_size=2)
    caught = _call(common, payload)
    fast = _call(common, payload, acyclic=True)

    node_keys = np.asarray(payload.node_keys)
    conn_keys = np.asarray(payload.conn_keys)
    reference = oracle.schedule(
        node_keys,
        conn_keys[..., 0],
        conn_keys[..., 1],
        node_keys[0, :NUM_INPUTS].copy(),
        node_keys[0, -NUM_OUTPUTS:].copy(),
        max_in_degree=payload.max_nodes,
        key_span=_key_span(payload, common.I_INF),
    )
    assert fast.node_level.shape[0] == 2
    for name in DOCUMENTED_FIELDS:
        assert np.array_equal(getattr(fast, name), getattr(caught, name)), name
        assert np.array_equal(getattr(fast, name), getattr(reference, name)), name
    for member in range(2):
        assert np.array_equal(fast.node_level[member], _CHAIN_LEVELS[0])


def test_acyclic_true_saturates_a_cycle_the_detecting_path_reports_as_i_inf(
    common, oracle, ring_payload, ring_schedule
):
    """Invariant: on the ring ``acyclic=True`` returns finite *wrong* levels, silently.

    ``acyclic=False`` is the documented answer and the oracle's: ``I_INF`` for the
    two ring nodes and for the output row downstream of them ``(2, 3, 7)``.
    ``acyclic=True`` asserts what it cannot know, so those rows get a saturated
    finite level instead — bounded by ``max_nodes``, because the capped relaxation
    raises a level by at most one per step — and a ``topo_key`` derived from it.
    Oracles: the cycle-detecting schedule, ``support.oracle.schedule`` (same answer),
    the payload's decoded edge list (the saturated levels admit a *downhill* live
    edge, i.e. exactly the violation ``_cycle_violation`` tests for), and
    ``validate_schedule``, which accepts the wrong schedule — nothing raises.

    This is precisely why ``acyclic=True`` is only legal for a genome family that
    guarantees acyclicity: ``DefaultGenome``'s structural gates refuse a cycle, so a
    feed-forward genome may pass ``True``; ``RecurrentGenome`` exists to evolve one,
    so it must keep the default. The failure mode is not an exception but a schedule
    whose level sweep executes a recurrent edge as if it were feed-forward.
    """
    i_inf = common.I_INF
    max_nodes = ring_payload.max_nodes
    caught = ring_schedule
    fast = _call(common, ring_payload, acyclic=True)

    # The rows the two spellings disagree on: present, and I_INF on the safe path.
    present = caught.node_exists[0]
    dead = (caught.node_level[0] == i_inf) & present
    rows = np.flatnonzero(dead)
    assert rows.tolist() == [2, 3, 7], "the ring's divergent rows are the cycle and its read-out"

    # acyclic=False reports the cycle, and the oracle agrees with it.
    assert np.all(caught.node_level[0, rows] == i_inf)
    assert np.all(caught.topo_key[0, rows] == i_inf)
    node_keys = np.asarray(ring_payload.node_keys)
    conn_keys = np.asarray(ring_payload.conn_keys)
    reference = oracle.schedule(
        node_keys,
        conn_keys[..., 0],
        conn_keys[..., 1],
        node_keys[0, :NUM_INPUTS].copy(),
        node_keys[0, -NUM_OUTPUTS:].copy(),
        max_in_degree=ring_payload.max_nodes,
        key_span=_key_span(ring_payload, i_inf),
    )
    assert np.array_equal(reference.node_level, caught.node_level)
    assert not np.array_equal(reference.node_level, fast.node_level)

    # acyclic=True saturates those rows instead, inside the capped loop's bound.
    assert np.all(fast.node_level[0, rows] != i_inf)
    assert np.all(fast.node_level[0, rows] >= 0)
    assert np.all(fast.node_level[0, rows] <= max_nodes)
    assert np.array_equal(
        fast.topo_key[0, rows], fast.node_level[0, rows] * max_nodes + rows
    ), "the saturated rows still get topo_key = level * N + slot"
    assert np.all(fast.topo_key[0, rows] != i_inf)

    # The disagreement is exactly the level/topo_key pair; everything else is shared.
    for name in DOCUMENTED_FIELDS:
        if name in ("node_level", "topo_key"):
            continue
        assert np.array_equal(getattr(fast, name), getattr(caught, name)), name

    # A saturated level is not merely different: it executes a live edge backwards.
    key_to_slot, edges = _decode(ring_payload, i_inf)
    downhill = [
        (key_to_slot[src_key], key_to_slot[dst_key])
        for _row, src_key, dst_key in edges
        if fast.node_level[0, key_to_slot[src_key]] >= fast.node_level[0, key_to_slot[dst_key]]
    ]
    assert (3, 2) in downhill, f"the cycle's back edge must become a downhill edge, got {downhill}"

    # And nothing raises: a wrong schedule is structurally valid.
    assert common.tensorization.validate_schedule(fast) is None


# ---- the refusal block of `derive_schedule` --------------------------------------------------


@pytest.mark.parametrize("name", _POSITIVE_STATICS)
@pytest.mark.parametrize("value", (0, -1), ids=("zero", "negative"))
def test_derive_schedule_refuses_a_non_positive_static(
    common, chain_payload, fixed_call, name, value
):
    """Invariant: each of the five statics must be a positive int, named in the error.

    Oracle: the documented ``Raises`` clause ("a non-positive or shape-inconsistent
    static") and ``_positive_static``'s single message, matched verbatim — a static
    is baked into the trace, so it is checked on the host before anything is built.
    ``fixed_call`` is the same call with the argument at its payload value.
    """
    expected = f"{name} must be a positive int, got {value}"
    with pytest.raises(common.ConfigError, match=re.escape(expected)) as info:
        _call(common, chain_payload, **{name: value})
    assert isinstance(info.value, common.TensorNEATError)
    assert np.array_equal(fixed_call.node_level, _CHAIN_LEVELS)


@pytest.mark.parametrize("name", ("node_keys", "conn_in", "conn_out"))
def test_derive_schedule_refuses_a_shape_disagreement(common, chain_payload, fixed_call, name):
    """Invariant: each tensor must match its static shape; the error names it and the shape.

    Oracle: the documented shapes read off the payload — ``(P, max_nodes)`` for
    ``node_keys`` and ``(P, max_conns)`` for ``conn_in``/``conn_out`` — against the
    same call with only that tensor one column short, so no other check can fire.
    """
    node_keys = np.asarray(chain_payload.node_keys)
    conn_keys = np.asarray(chain_payload.conn_keys)
    tampered = {
        "node_keys": node_keys[:, :-1],
        "conn_in": conn_keys[..., 0][:, :-1],
        "conn_out": conn_keys[..., 1][:, :-1],
    }[name]
    expected_shape = (
        (chain_payload.pop_size, chain_payload.max_nodes)
        if name == "node_keys"
        else (chain_payload.pop_size, chain_payload.max_conns)
    )
    expected = f"{name} must have shape {expected_shape}, got {tuple(tampered.shape)}"

    with pytest.raises(common.ConfigError, match=re.escape(expected)) as info:
        _call(common, chain_payload, **{name: tampered})
    assert isinstance(info.value, common.TensorNEATError)
    assert np.array_equal(fixed_call.node_level, _CHAIN_LEVELS)


@pytest.mark.parametrize(
    ("name", "role"),
    (("input_keys", "num_inputs"), ("output_keys", "num_outputs")),
)
def test_derive_schedule_refuses_a_key_vector_longer_than_max_nodes(
    common, chain_payload, fixed_call, name, role
):
    """Invariant: a key vector wider than ``max_nodes`` is refused, naming its role.

    Oracle: the documented check ``num_inputs``/``num_outputs`` against ``max_nodes``
    — one key vector of length ``max_nodes + 1``, the smallest length a slot vector
    could not index. The other key vector stays at its payload length, so only one
    refusal can fire.
    """
    too_long = np.zeros(chain_payload.max_nodes + 1, dtype=np.int32)
    expected = f"{role} {too_long.size} exceeds max_nodes {chain_payload.max_nodes}"

    with pytest.raises(common.ConfigError, match=re.escape(expected)) as info:
        _call(common, chain_payload, **{name: too_long})
    assert isinstance(info.value, common.TensorNEATError)
    assert np.array_equal(fixed_call.node_level, _CHAIN_LEVELS)


@pytest.mark.parametrize("name", ("conn_in", "conn_out", "input_keys", "output_keys"))
def test_derive_schedule_refuses_a_key_array_in_the_other_spelling(
    common, chain_payload, fixed_call, name
):
    """Invariant: mixing the two key spellings is refused; ``node_keys`` is the reference.

    Oracle: the documented dtype rule — the kernel dispatches on ``node_keys``' static
    dtype, so every other key tensor must share it — applied to the conversion helper
    of the sibling module (``float32``/``NaN`` for ``int32``/``I_INF``). Exactly one
    tensor is re-spelled, so the message names it and not one of the other three.
    """
    i_inf = common.I_INF
    node_keys = np.asarray(chain_payload.node_keys)
    conn_keys = np.asarray(chain_payload.conn_keys)
    tampered = {
        "conn_in": _float_keys(conn_keys[..., 0], i_inf),
        "conn_out": _float_keys(conn_keys[..., 1], i_inf),
        "input_keys": _float_keys(node_keys[0, :NUM_INPUTS], i_inf),
        "output_keys": _float_keys(node_keys[0, -NUM_OUTPUTS:], i_inf),
    }[name]
    expected = f"{name} dtype {tampered.dtype} must match node_keys dtype {node_keys.dtype}"

    with pytest.raises(common.ConfigError, match=re.escape(expected)) as info:
        _call(common, chain_payload, **{name: tampered})
    assert isinstance(info.value, common.TensorNEATError)
    assert np.array_equal(fixed_call.node_level, _CHAIN_LEVELS)


# ---- the refusal block of `node_levels` ------------------------------------------------------


#: One guard case each: a single-field tamper of the hand-built tables plus the exact
#: ``ConfigError`` message that field must provoke. The messages are the documented
#: ``Raises`` clause of ``node_levels``, hard-coded so a reworded guard fails here.
#: ``src`` has a population-axis case only: the connection-axis rule compares every
#: array against ``src``'s width, so a short ``src`` cannot disagree with itself.
_LEVEL_GUARDS = (
    pytest.param(
        lambda tables: _variant(tables, edges=None),
        "node_levels: acyclic=False needs the (src, dst, edge_valid) triple of the "
        "connection rows for its in-graph cycle test",
        id="missing-edge-triple",
    ),
    pytest.param(
        lambda tables: _variant(
            tables, edges=(_stack(tables.edges[0]), tables.edges[1], tables.edges[2])
        ),
        "node_levels: src population axis 2 does not match in_conn 1",
        id="src-population-axis",
    ),
    pytest.param(
        lambda tables: _variant(
            tables, edges=(tables.edges[0], _stack(tables.edges[1]), tables.edges[2])
        ),
        "node_levels: dst population axis 2 does not match in_conn 1",
        id="dst-population-axis",
    ),
    pytest.param(
        lambda tables: _variant(
            tables, edges=(tables.edges[0], tables.edges[1], _stack(tables.edges[2]))
        ),
        "node_levels: edge_valid population axis 2 does not match in_conn 1",
        id="edge_valid-population-axis",
    ),
    pytest.param(
        lambda tables: _variant(
            tables, edges=(tables.edges[0], tables.edges[1][:, :2], tables.edges[2])
        ),
        "node_levels: dst connection axis 2 does not match src 3",
        id="dst-connection-axis",
    ),
    pytest.param(
        lambda tables: _variant(
            tables,
            edges=(tables.edges[0], tables.edges[1], np.ones((1, 4), dtype=np.bool_)),
        ),
        "node_levels: edge_valid connection axis 4 does not match src 3",
        id="edge_valid-connection-axis",
    ),
    pytest.param(
        lambda tables: _variant(tables, max_nodes=tables.max_nodes + 1),
        "in_conn node axis 4 does not match max_nodes 5",
        id="in_conn-node-axis",
    ),
    pytest.param(
        lambda tables: _variant(tables, max_nodes=0),
        "max_nodes must be a positive int, got 0",
        id="max_nodes-non-positive",
    ),
)


@pytest.mark.parametrize(("tamper", "expected"), _LEVEL_GUARDS)
def test_node_levels_refuses_a_bad_static_or_a_disagreeing_edge_triple(
    common, levels_tables, levels_control, tamper, expected
):
    """Invariant: every guard of ``node_levels`` raises its own ``ConfigError``.

    Oracles: the exact documented message, matched verbatim, and ``levels_control`` —
    the same call with the one tampered field changed back, which the fixture pins
    against hand enumeration ``[0, 0, 1, 2]``. Each case alters exactly one field, so
    a refusal cannot be an unrelated error; the in-graph ``node_levels`` call site is
    exercised (not a host re-implementation of the checks).
    """
    with pytest.raises(common.ConfigError, match=re.escape(expected)) as info:
        _levels(common, tamper(levels_tables))
    assert isinstance(info.value, common.TensorNEATError)
    assert np.array_equal(levels_control, np.array([[0, 0, 1, 2]], dtype=np.int32))


def test_node_levels_acyclic_needs_no_edge_triple(common, levels_tables, levels_control):
    """Invariant: ``acyclic=True`` drops the triple requirement and the answer is unchanged.

    Oracle: hand enumeration, via ``levels_control`` (which the default spelling
    produces on the same tables), so the fast path is not merely *accepted* without
    an ``edges`` argument but yields the identical levels on a cycle-free genome.
    """
    fast = _levels(common, levels_tables, acyclic=True)
    assert np.array_equal(fast, levels_control)
