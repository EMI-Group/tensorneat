"""The accelerator formulation of the execution schedule, proved against the host one.

``derive_schedule`` has two *formulations*, selected at trace time by the plain
Python static ``state.kernel_mode``: ``"host"`` (the default) writes the dense
key->slot table and inverts the entry order with one flat ``scatter`` each, while
``"accelerator"`` replaces both writes with an equality template
(:func:`~tensorneat.common.schedule_accelerator.key_template`) and a direct count
(:func:`~tensorneat.common.schedule_accelerator.cumulative_indegree`), groups the
level relaxation ``tensorization._ACCELERATOR_UNROLL`` steps per trip (with a
static remainder tail) and runs the cycle-detecting fallback counter-only. Both
legs are documented to return **byte-identical** ``Transformed`` values; this
module is their first committed test.

What this module proves
-----------------------
* the two legs derive the same schedule **field for field** — all twelve
  ``Transformed`` leaves, shape and dtype included — through the *public* genome
  API (``Genome.transform``) on ``P = 100``, for three payloads that between them
  cover both genome families and three accelerator geometries;
* ``validate_schedule`` accepts both results and still rejects a one-cell tamper,
  so the acceptance is not vacuous;
* each leg equals the independent numpy derivation ``support.oracle.schedule``,
  driven with the genome's own statics;
* the payloads really are three accelerator paths — ``shallow`` settles inside
  the first group of ``rounds = min(4, cap)`` steps, ``short`` is one trip
  (``rounds == cap == max_nodes``, no tail), ``cyclic`` pays the counter-only
  fallback at ``cap = 2 * max_nodes`` — confirmed by hand-enumerated levels;
* the formulations are really *routed differently*, proved on the traced IR and
  not on the numbers (equal output is also what a leg that ran twice would
  produce): the host leg traces ``scatter`` op(s), the accelerator leg none, both
  keep exactly one ``argsort``, and the accelerator leg adds the template's
  ``equal`` comparisons;
* a ``State`` without the entry traces the host route, op for op;
* ``_cycle_violation``'s ``evoxc.cond`` stays data-dependent on purpose — a
  recurrent payload traces exactly one ``if`` on *both* legs, a feed-forward one
  zero on both (``acyclic=True`` drops it, not the formulation) — and each leg
  keeps its relaxation ``while_loop``(s);
* the resolver contract: ``kernel_mode`` reads the trace-time entry (``"host"``
  when absent), ``resolve_mode``/``is_accelerator`` agree with it, and an unknown
  spelling raises ``ConfigError`` (a ``TensorNEATError``) naming both accepted
  values — from ``resolve_mode`` and from a *traced* ``transform``.

Oracles used here (never the implementation under test)
-------------------------------------------------------
the **other leg** (each formulation is the other's differential oracle);
``support.oracle.schedule``, whose level rule has the cycle-detecting semantics
and therefore dissents on the ring; **hand enumeration** of one member's levels
and ``node_exists``; the **traced IR** (``op.name`` counts over the function and
every nested ``op.regions`` block); and the documented campaign arithmetic
(``rounds = min(_ACCELERATOR_UNROLL, cap)``, ``bound = cap // rounds``,
``tail = cap % rounds``), used only to show the payloads are distinct paths.

Gotchas
-------
* ``evoxc.build`` turns only an ``ndarray``/``Tensor`` argument into a graph
  input, so the host ``State`` is **closed over** in the traced closure — a
  static Python value baked into the trace, as
  ``./unit/algorithm/test_distance_dense.py`` does it; that is what makes a leg a
  property of the *trace*.
* ``evoxc.Tensor`` has no ``__array__``: read results through
  ``support.runner.to_numpy`` (``run_defn`` already does), never ``np.asarray``.
* pytest hangs when an exception escapes a trace under ``--tb=auto``; use
  ``--tb=short`` while debugging.
* ``Transformed`` is a namedtuple of twelve leaves in a documented order, with
  ``level``/``overflow`` as read-only *properties*: compare leaf by leaf
  (``_fields`` + ``np.array_equal``), never with ``==``.
* ``evoxc.trace`` is the only direct ``evoxc`` call here — not
  ``build``/``run``/``lower``/``compile``/``load`` (the runner's monopoly) — and
  the counters are cached beside the schedules.
* ``_ACCELERATOR_UNROLL`` is a *private* constant (``tensorization.py``); the
  geometry test pins it, because every note here is derived from ``4``.

Scope: the schedule family's fourth module (semantics in
``./test_tensorization.py``, the host checker in
``./test_tensorization_validate.py``, the ``acyclic`` seam in
``./test_tensorization_seams.py``), ``"numpy"`` backend throughout: no GPU, no
``iree``, no optional extra, no ``requires_iree`` marker, no ``jax``/``torch``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest

from support import runner

from ..ir_ops import op_names

# Shared structural constant: the documented field order of ``Transformed`` (the
# seam module imports its fixtures and helpers the same way).
from .test_tensorization import DOCUMENTED_FIELDS

#: The population every schedule in this module is derived for.
POP = 100

#: The two formulation spellings ``state.kernel_mode`` accepts, in documentation order.
MODES = ("host", "accelerator")


class Case(NamedTuple):
    """One (genome, payload) pair plus the hand-written expectations for it.

    ``genome`` is a ``DefaultGenome`` (feed-forward) or ``RecurrentGenome`` whose
    own ``max_in_degree``/``key_span``/``max_conns`` are the statics ``transform``
    hands the schedule; ``levels``/``node_exists`` are one member's rows by hand
    (``None`` spells ``I_INF``); ``acyclic`` is the static the family passes.
    """

    name: str
    genome: Any
    payload: Any
    acyclic: bool
    levels: tuple
    node_exists: tuple


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def cases(payloads):
    """The three (genome, payload) pairs, each an accelerator path of its own.

    * ``cyclic`` — ``RecurrentGenome`` over ``ring_payload``: rows 2 and 3 feed
      each other, so both, the output row downstream of them (7) and the absent
      padding rows (4-6) are ``I_INF``. Being recurrent it keeps
      ``acyclic=False``, which puts ``_cycle_violation``'s ``cond`` in the trace
      and makes the accelerator fallback the counter-only ``while_loop``.
    * ``shallow`` — ``DefaultGenome`` over a five-row chain (levels ``0, 0, 1, 2,
      3``): the deepest level is below ``rounds = min(4, 5) = 4``, so the
      relaxation settles inside the FIRST accelerator group.
    * ``short`` — ``DefaultGenome`` over a three-row chain: ``rounds == cap ==
      max_nodes == 3``, so ``bound`` is 1 and the remainder tail is 0 — the
      grouped trip is the whole campaign.
    """
    def case(name, genome, payload, levels, node_exists):
        return Case(
            name=name,
            genome=genome,
            payload=payload,
            acyclic=genome.network_type == "feedforward",
            levels=tuple(levels),
            node_exists=tuple(node_exists),
        )

    return (
        case("cyclic",
             payloads.recurrent_genome(max_nodes=8, max_conns=16),
             payloads.ring_payload(pop_size=POP, max_nodes=8, max_conns=16),
             (0, 0, None, None, None, None, None, None),
             (True, True, True, True, False, False, False, True)),
        case("shallow",
             payloads.default_genome(max_nodes=5, max_conns=8),
             payloads.chain_payload(pop_size=POP, max_nodes=5, max_conns=8),
             (0, 0, 1, 2, 3),
             (True,) * 5),
        case("short",
             payloads.default_genome(max_nodes=3, max_conns=4),
             payloads.chain_payload(pop_size=POP, max_nodes=3, max_conns=4),
             (0, 0, 1),
             (True,) * 3),
    )


@pytest.fixture(scope="module", params=("cyclic", "shallow", "short"))
def case(request, cases):
    """One :class:`Case`, parametrized so every test runs all three geometries."""
    return {item.name: item for item in cases}[request.param]


@pytest.fixture(scope="module")
def schedules(common, cases):
    """``{(case name, mode): Transformed}`` — one built executable per leg.

    The host ``State`` is rebuilt per leg and closed over by the traced closure,
    so the two modes are two different graphs of one payload; each is traced and
    run exactly once (the pop-100 traces are the expensive part here).
    """
    built = {}
    for item in cases:
        for mode in MODES:
            built[(item.name, mode)] = _transform(common, item, mode)
    return built


@pytest.fixture(scope="module")
def traces(cases):
    """``{(case name, mode): Counter}`` — the traced ``op.name`` table of each leg.

    ``"default"`` is a third key: the same closure with a ``State`` that does not
    carry the entry at all, i.e. the spelling a caller who never registers
    ``kernel_mode`` traces.
    """
    return {
        (item.name, mode): _op_counts(_trace(item, mode))
        for item in cases
        for mode in (*MODES, "default")
    }


@pytest.fixture(scope="module")
def accelerator():
    """``tensorneat.common.schedule_accelerator`` — the resolver module under test."""
    from tensorneat.common import schedule_accelerator

    return schedule_accelerator


# --------------------------------------------------------------------------- helpers


def _transform(common, case, mode):
    """Derive one payload's schedule through the public genome API on one leg."""
    state = common.State(kernel_mode=mode)
    return runner.run_defn(
        lambda *arrays: case.genome.transform(state, *arrays), *case.payload.args()
    )


def _trace(case, mode):
    """Trace one leg and return the ``evoxc.Graph`` (no build, no run)."""
    # The one direct evoxc call in this module: tracing is neither build nor run,
    # and support.runner exports no trace helper (see the module docstring).
    import evoxc
    from tensorneat.common import State

    state = State() if mode == "default" else State(kernel_mode=mode)
    return evoxc.trace(
        lambda *arrays: case.genome.transform(state, *arrays),
        *(runner.spec_like(value) for value in case.payload.args()),
    )


def _op_counts(graph):
    """``op.name -> count`` over the graph, descending into every nested region.

    Terminators included on purpose: this module compares *whole* histograms of
    two legs (``default == traces["host"]``), where a differing block count must
    not cancel out. The walk itself is the shared one in ``..ir_ops``, whose
    descent is ``op.regions -> region.blocks`` (a region is an ordered list of
    blocks, not a block).
    """
    return op_names(graph, include_terminators=True)


def _campaign(cap, unroll, detect_cycles=False):
    """``(rounds, bound, tail)`` of one campaign, from the documented rule.

    ``detect_cycles`` is the fallback campaign, which the accelerator leg runs
    counter-only (one step per trip, no ``changed`` leaf), so it never groups.
    """
    rounds = 1 if detect_cycles else min(unroll, cap)
    return rounds, cap // rounds, cap % rounds


def test_the_payloads_and_schedules_are_hundred_member_batches(cases, schedules):
    """Premise: every case is a ``P = 100`` population and every schedule covers it.

    Oracle: ``Payload.pop_size`` and the leading axis of the derived
    ``node_level``. Agreeing for a *whole population* is the point, so a payload
    that had silently shrunk to one member would weaken every other test here
    without failing.
    """
    assert tuple(item.name for item in cases) == ("cyclic", "shallow", "short")
    for item in cases:
        assert item.payload.pop_size == POP
        for mode in MODES:
            schedule = schedules[(item.name, mode)]
            assert schedule.node_level.shape[0] == POP, (
                f"the {mode} leg of {item.name} covers {schedule.node_level.shape[0]} "
                f"members, expected {POP}"
            )


# --------------------------------------------------------------------------- the equality


def test_host_and_accelerator_transform_agree_leaf_for_leaf(common, case, schedules):
    """All twelve ``Transformed`` leaves are identical on the two legs.

    Oracle: the *other* leg. ``level``/``overflow`` are compatibility properties,
    not leaves, so the loop runs over ``_fields`` (checked against the documented
    order) and compares shape, dtype and value: the formulations are documented
    to be bit-identical, not merely close.
    """
    host = schedules[(case.name, "host")]
    accel = schedules[(case.name, "accelerator")]
    leaves = tuple(common.Transformed._fields)
    assert leaves == DOCUMENTED_FIELDS, (
        "Transformed's leaf list changed; this test and the documented order disagree"
    )
    for name in leaves:
        expected = getattr(host, name)
        actual = getattr(accel, name)
        assert actual.shape == expected.shape, (
            f"{name} shape {actual.shape} on the accelerator leg, {expected.shape} on "
            f"the host leg ({case.name})"
        )
        assert actual.dtype == expected.dtype, (
            f"{name} dtype {actual.dtype} on the accelerator leg, {expected.dtype} on "
            f"the host leg ({case.name})"
        )
        assert np.array_equal(actual, expected), (
            f"{name} differs between the two legs of {case.name}: the accelerator "
            f"formulation is documented to return a byte-identical schedule"
        )


def test_validate_schedule_accepts_both_legs(common, case, schedules):
    """``validate_schedule`` accepts both legs — and still rejects a tampered one.

    Oracle: the host checker's own invariants. The tamper (``edge_overflow - 1``,
    a count that cannot go negative) proves the acceptance is not what an
    unconditionally-``None`` checker would produce.
    """
    for mode in MODES:
        schedule = schedules[(case.name, mode)]
        assert common.tensorization.validate_schedule(schedule) is None, (
            f"the {mode} leg of {case.name} violates a Transformed invariant"
        )
    host = schedules[(case.name, "host")]
    with pytest.raises(common.ConfigError):
        common.tensorization.validate_schedule(
            host._replace(edge_overflow=host.edge_overflow - 1)
        )


def test_each_leg_matches_the_numpy_oracle(oracle, case, schedules):
    """Every leaf of both legs equals ``support.oracle.schedule``.

    Oracle: the independent numpy derivation (dense key->slot table, in-group
    rank, counting ``cc``, longest-path levels), driven with the very statics
    ``transform`` hands the kernel — so this is the only *third* opinion on the
    accelerator leg's output.
    """
    node_keys = np.asarray(case.payload.node_keys)
    conn_keys = np.asarray(case.payload.conn_keys)
    reference = oracle.schedule(
        node_keys,
        conn_keys[..., 0],
        conn_keys[..., 1],
        node_keys[0, : case.genome.num_inputs].copy(),
        node_keys[0, list(case.genome.output_idx)].copy(),
        max_in_degree=case.genome.max_in_degree,
        key_span=case.genome.key_span,
    )
    for mode in MODES:
        schedule = schedules[(case.name, mode)]
        for name in DOCUMENTED_FIELDS:
            expected = getattr(reference, name)
            actual = getattr(schedule, name)
            assert actual.dtype == expected.dtype, (
                f"{name} dtype {actual.dtype} on the {mode} leg, oracle says "
                f"{expected.dtype} ({case.name})"
            )
            assert np.array_equal(actual, expected), (
                f"{name} of the {mode} leg disagrees with support.oracle.schedule "
                f"on {case.name}"
            )


def test_levels_are_hand_enumerated_on_both_legs(common, case, schedules):
    """One member's ``node_level`` and ``node_exists`` are the hand-written rows.

    Oracle: hand enumeration from each payload's topology — chain levels grow by
    one per row from the pinned inputs, and on the ring the two cycle nodes, the
    output row downstream of them and the absent padding rows are all ``I_INF``
    while ``node_exists`` still tells padding from cyclic rows.
    """
    # One member's two hand-written rows, in the kernel's own spellings.
    expected_levels = np.array(
        [common.I_INF if value is None else value for value in case.levels], dtype=np.int32
    )
    expected_exists = np.array(case.node_exists, dtype=np.bool_)
    for mode in MODES:
        schedule = schedules[(case.name, mode)]
        assert np.all(schedule.node_level == expected_levels), (
            f"node_level of the {mode} leg of {case.name} is not the hand-enumerated "
            f"row {expected_levels.tolist()}: got {schedule.node_level[0].tolist()}"
        )
        assert np.all(schedule.node_exists == expected_exists), (
            f"node_exists of the {mode} leg of {case.name} is not the hand-enumerated "
            f"row {expected_exists.tolist()}: got {schedule.node_exists[0].tolist()}"
        )


def test_the_payloads_cover_three_accelerator_geometries(common, cases):
    """The three payloads hit three distinct accelerator campaign geometries.

    Oracle: ``rounds = min(_ACCELERATOR_UNROLL, cap)``, ``bound = cap // rounds``
    and ``tail = cap % rounds`` recomputed from the module constant and each
    payload's ``max_nodes``, plus the hand-enumerated levels — the coverage proof
    for the rest of the module. The constant is pinned because every note in it
    is derived from ``4``.
    """
    unroll = common.tensorization._ACCELERATOR_UNROLL
    assert unroll == 4, (
        f"_ACCELERATOR_UNROLL moved to {unroll}; re-derive the geometry of every "
        "payload in this module"
    )
    by_name = {item.name: item for item in cases}
    fast = {
        name: _campaign(item.payload.max_nodes, unroll) for name, item in by_name.items()
    }
    assert fast == {
        "cyclic": (4, 2, 0),
        "shallow": (4, 1, 1),
        "short": (3, 1, 0),
    }, f"the payload geometries changed: {fast}"

    # The shallow chain's deepest level (3) is below its group size (4), so the
    # data-dependent exit fires after the first trip.
    shallow = by_name["shallow"]
    deepest = max(value for value in shallow.levels if value is not None)
    assert deepest < fast["shallow"][0], (
        f"{shallow.name} no longer settles inside the first group: deepest level "
        f"{deepest} against rounds {fast['shallow'][0]}"
    )
    # The short chain is one trip: rounds == cap, so bound is 1 and the tail 0.
    short = by_name["short"]
    assert fast["short"][0] == short.payload.max_nodes
    # The recurrent fallback campaign is counter-only and runs at 2 * max_nodes.
    cyclic = by_name["cyclic"]
    fallback_cap = 2 * cyclic.payload.max_nodes
    assert fallback_cap == 16
    assert _campaign(fallback_cap, unroll, detect_cycles=True) == (1, 16, 0)


# --------------------------------------------------------------------------- the route proof


def test_only_the_host_leg_traces_the_flat_scatters(case, traces):
    """The host leg traces ``scatter`` op(s); the accelerator leg traces none.

    Oracle: the traced IR (``op.name`` counted over the top-level block *and*
    every nested region), because equal output arrays are also what two runs of
    the host route would produce. Both legs keep the single stable ``argsort``
    the two routes share, and the accelerator leg carries the equality template's
    extra ``equal`` comparisons, so a leg that silently ran the host route fails
    here even though its schedule would validate.
    """
    host = traces[(case.name, "host")]
    accel = traces[(case.name, "accelerator")]
    assert sum(host.values()) > 0, "the host trace is empty; the scatter count is vacuous"
    assert sum(accel.values()) > 0, "the accelerator trace is empty; its zero is vacuous"
    assert host["scatter"] >= 1, (
        f"the host leg of {case.name} traced no scatter: the dense key table and the "
        f"inverted entry order are each one flat scatter"
    )
    assert accel["scatter"] == 0, (
        f"the accelerator leg of {case.name} traced {accel['scatter']} scatter op(s): "
        f"the equality template and the counted cc are documented to be scatter-free"
    )
    assert host["argsort"] == accel["argsort"] == 1, (
        f"both routes keep exactly one stable argsort; host {host['argsort']}, "
        f"accelerator {accel['argsort']}"
    )
    assert accel["equal"] > host["equal"], (
        f"the accelerator leg of {case.name} traced {accel['equal']} equal op(s) "
        f"against the host leg's {host['equal']}: the key->slot equality template is "
        f"missing from the trace"
    )


def test_a_state_without_the_entry_traces_the_host_route(cases, traces):
    """A ``State`` with no ``kernel_mode`` entry traces the host leg, op for op.

    Oracle: the explicit ``"host"`` trace. Comparing the whole ``op.name`` table
    (not just the scatter count, not just the output arrays) is what rules out a
    silent switch of the default to the accelerator route.
    """
    for item in cases:
        default = traces[(item.name, "default")]
        assert default["scatter"] >= 1, (
            f"a State without the entry traced no scatter on {item.name}: the route "
            f"assumed by a caller who never registers kernel_mode is the host one"
        )
        assert default == traces[(item.name, "host")], (
            f"the entry-less trace of {item.name} differs from the explicit host "
            f"trace: {default} against {traces[(item.name, 'host')]}"
        )


def test_the_control_flow_shape_is_the_same_on_both_legs(case, traces):
    """Both legs keep the cycle ``cond`` and the relaxation ``while_loop``(s).

    Oracle: the ``acyclic`` static each genome family passes. A feed-forward
    genome asserts a cycle-free population, so ``acyclic=True`` drops the cycle
    test and its fallback — **zero** ``if`` ops on *both* legs, which this test
    states rather than blaming on the formulation — and one ``while`` remains. A
    recurrent genome keeps ``acyclic=False``, so ``_cycle_violation`` stays a
    data-dependent ``evoxc.cond`` on purpose (**one** ``if`` on both legs) and
    the fallback campaign sits nested inside its region as a second ``while``
    (which is why the walk over ``op.regions`` matters).
    """
    expected_if = 0 if case.acyclic else 1
    expected_while = 1 if case.acyclic else 2
    for mode in MODES:
        counts = traces[(case.name, mode)]
        assert counts["if"] == expected_if, (
            f"the {mode} leg of {case.name} traced {counts['if']} if op(s), expected "
            f"{expected_if} (acyclic={case.acyclic}): the cycle test is selected by "
            f"the genome family's acyclic static, never by the kernel_mode formulation"
        )
        assert counts["while"] >= expected_while, (
            f"the {mode} leg of {case.name} traced {counts['while']} while op(s), "
            f"expected at least {expected_while} (the fast campaign"
            f"{'' if case.acyclic else ' plus the cycle-detection fallback'})"
        )


# --------------------------------------------------------------------------- the resolver


def test_the_resolver_contract(common, accelerator):
    """``kernel_mode`` reads the trace-time entry; the two helpers agree with it.

    Oracle: the documented selection rule and the module's ``KERNEL_MODES``
    table. A ``State`` that never carries the key must not raise —
    ``State.__getattr__`` raises for a missing entry, so the resolver's
    ``getattr`` default is what keeps an entry-less state on the host route — and
    ``is_accelerator`` is ``resolve_mode(mode) == "accelerator"``.
    """
    assert accelerator.KERNEL_MODE_KEY == "kernel_mode"
    assert accelerator.kernel_mode(common.State()) == "host"
    assert accelerator.kernel_mode(common.State(kernel_mode="host")) == "host"
    assert accelerator.kernel_mode(common.State(kernel_mode="accelerator")) == "accelerator"
    assert tuple(accelerator.KERNEL_MODES) == ("host", "accelerator")
    for mode in accelerator.KERNEL_MODES:
        assert accelerator.resolve_mode(mode) == mode
        assert accelerator.is_accelerator(mode) is (mode == "accelerator")


def test_an_unknown_mode_is_refused_with_the_accepted_values_named(common, accelerator):
    """An unknown formulation raises ``ConfigError`` naming both accepted values.

    Oracle: the documented message (``kernel_mode must be one of ('host',
    'accelerator')``) and the error hierarchy (``ConfigError`` is a
    ``TensorNEATError``). Each spelling probed here differs from an accepted one
    in exactly one way, so a resolver that only checked ``"host"`` would still
    fail on the ``"accelerator"`` half of the message.
    """
    for bad in ("auto", "Host", "", None, 3, True):
        with pytest.raises(common.ConfigError) as raised:
            accelerator.resolve_mode(bad)
        message = str(raised.value)
        assert "kernel_mode" in message, f"{bad!r} raised without naming the entry: {message}"
        assert "host" in message and "accelerator" in message, (
            f"{bad!r} raised without naming both accepted values: {message}"
        )
        assert isinstance(raised.value, common.TensorNEATError)


def test_an_unknown_mode_is_refused_inside_a_traced_transform(case):
    """The same refusal fires while a ``transform`` graphs, not only on the host.

    Oracle: ``ConfigError`` raised out of the *traced* call. ``transform``
    resolves the static inside ``derive_schedule`` — after the host-side static
    checks and before any schedule work — so the failure is the documented loud
    refusal rather than a silently defaulted route. A trace is the only place
    this can be observed, because the method's first steps already build graph
    ops.
    """
    from tensorneat.common import ConfigError, State

    state = State(kernel_mode="auto")
    with pytest.raises(ConfigError) as raised:
        runner.run_defn(
            lambda *arrays: case.genome.transform(state, *arrays), *case.payload.args()
        )
    assert "kernel_mode" in str(raised.value)
