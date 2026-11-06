"""Regression tests for the *accelerator* spelling of ``DefaultGenome``'s level sweep.

What this module proves
-----------------------
``prepare_pass`` picks the level spelling from two host statics
(``genome/default.py``)::

    sweep = is_accelerator(kernel_mode(state)) and samples == 1

``True`` selects ``_level_sweep`` — the same ``_node_step`` statically unrolled over
``level = 1 .. max_depth``, with **no** ``evoxc.while_loop`` and neither the depth
bound (``_depth_limit``'s ``reduce_max``) nor the rank-0 loop counter. Every other
combination (the host leg at any ``samples``, the accelerator leg at ``samples > 1``)
selects ``_level_loop`` — one ``evoxc.while_loop`` bounded by that depth limit.

The two spellings must be the *same function*, bit for bit, and this module is the
first committed regression test of that claim. It proves, on hand-built payloads:

1. ``samples == 1``: the accelerator sweep and the host loop return **exactly** the
   same array on a shallow chain (whose closed form is also checked), on a chain
   deeper than ``max_depth`` (both stay ``NaN``, the documented depth limit), on a
   topology whose node is bypassed by its consumer, and on an unconnected node whose
   ``NaN`` must cross two hops into the read-out;
2. ``samples == 4``: the accelerator leg falls back to the loop and still agrees with
   the host leg exactly, and a batch whose four rows per individual are identical
   repeats that individual's ``samples == 1`` result — the individual-major tiling;
3. by **traced op names**: the accelerator ``samples == 1`` graph holds no ``while``
   op (and no ``reduce_max``) while the host graph holds exactly one of each, and the
   accelerator unroll is *more* ops than the loop; a tiled accelerator batch holds
   the ``while`` op again. That is the proof the two spellings really ran: a
   value-only test would also pass if both legs took the same path.

Oracles used here (never the kernel itself)
-------------------------------------------
* the **other leg** — exact, ``NaN``-aware equality
  (``np.array_equal(..., equal_nan=True)`` plus identical shape and dtype). A
  tolerance would hide a level-visitation bug, so none is used: both legs run the
  same ``_node_step`` on the same operands, and any difference is a real one. A
  difference found here is reported upward instead of being loosened;
* the **hand-known depth rule** — ``chain_payload(max_nodes=8)`` puts its output row
  at level 6, so ``max_depth=3`` leaves the read-out ``NaN`` by construction, while
  the same payload evaluated with ``max_depth=8`` is finite on both legs (the
  control that tells a bound from a broken leg);
* a **closed form** on the chain (``weight=1``, ``bias=0``, ``response=1``, the
  default gene's single ``sigmoid`` activation option, output rows unactivated): five
  logistic compositions of ``x0 + x1``, computed in ``float64`` and compared with a
  float32-accumulation tolerance — this one assertion is deliberately *not* exact,
  because the claim it backs is "the legs agree", not "float64 == float32";
* the **traced IR** — the op-name walker below, for the route-selected proof.

Gotchas this module encodes
---------------------------
* The schedule is passed **field by field** and rebuilt *inside* the traced closure
  (``test/unit/genome/conftest.py``): a ``Transformed`` namedtuple handed to the
  runner as one argument is baked into the trace as a static whose fields are plain
  numpy arrays, and every ``transformed.in_conn`` then fails with a ``TypeError``.
* The shared executable cache is keyed by the identity of the genome **and** of the
  host ``state``, so the host and accelerator states of one genome never share an
  entry; the two schedules are also derived per state, because the accelerator
  formulation encodes the incoming windows differently.
* ``prepare_pass`` reads no ``samples`` argument from ``forward``: it derives
  ``samples = len(inputs) // pop_size``. The tiled leg is therefore the same genome
  and payload with a ``(P * 4, num_inputs)`` batch of the same rows.
* The sweep is selected only for ``samples == 1`` on the accelerator leg on purpose:
  a tiled dataset keeps the loop (the unrolled form multiplies a ``P * samples``-row
  batch by ``max_depth``). Never "simplify" the gate to the formulation static alone.
* The op counts come from the shared walker ``..ir_ops``, which descends a region's
  **blocks** (``op.regions -> region.blocks -> block.ops``): a ``Region`` is an
  ordered list of blocks and has no ``.ops`` of its own, so recursing into the
  region itself would raise ``AttributeError`` on the first ``while``.
* Debug with ``--tb=short``: a trace-time exception inside a fixture surfaces as a
  build error, and tracing this node's graphs takes about ten times a run.

``max_depth=8`` (the chain's real depth is 6) and ``max_depth=3`` are the only
build-time statics varied here. Backend is ``"numpy"`` throughout, through the shared
runner only: no GPU, no ``iree``, no optional extra, no ``jax``/``torch``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..ir_ops import op_names

# --------------------------------------------------------------------------- #
# helpers: one genome factory, the two formulation states, one evaluator
# --------------------------------------------------------------------------- #
#: Three distinct input rows, tiled four times each by the tiled test. ``float32``
#: because that is what ``forward`` documents, and one row is all zeros so a
#: level-step that mishandles a zero input cannot pass unnoticed.
INPUTS = np.array([[0.3, -0.2], [-1.5, 0.75], [0.0, 0.0]], dtype=np.float32)


def _genome(
    genome,
    *,
    num_inputs: int = 2,
    num_outputs: int = 1,
    max_nodes: int = 8,
    max_conns: int = 16,
    max_in_degree: int = 4,
    **kwargs,
):
    """A ``DefaultGenome`` whose static shapes match the payloads of this module.

    ``max_in_degree=4`` is the window width ``K``; every payload here keeps its real
    in-degree at or below ``4`` (the chain's first hidden row reads both inputs, the
    rest read one), so no edge is dropped into ``edge_overflow``.

    Args:
        genome: The ``tensorneat.genome`` package (the module-scoped fixture).
        num_inputs: Input node count / input width.
        num_outputs: Output node count / read-out width.
        max_nodes: Static node capacity ``N``; must equal the payload's.
        max_conns: Static connection capacity ``E``; must equal the payload's.
        max_in_degree: Window width ``K`` of every node.
        **kwargs: Extra ``DefaultGenome`` keywords (``max_depth`` here).

    Returns:
        The constructed genome.
    """
    return genome.DefaultGenome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=max_nodes,
        max_conns=max_conns,
        max_in_degree=max_in_degree,
        **kwargs,
    )


def _node_row(bias=0.0, response=1.0, aggregation=0.0, activation=0.0):
    """One ``DefaultNode`` attribute row, in the documented column order.

    ``support.payloads`` reads ``(bias, response, aggregation id, activation id)``,
    and both ids are *positions* in the gene's configured option lists (``-1`` is
    identity and is not an option index).

    Args:
        bias: Additive term of ``bias + response * agg``.
        response: Multiplicative term of ``bias + response * agg``.
        aggregation: Aggregation option position.
        activation: Activation option position, or ``-1`` for identity.

    Returns:
        The 4-tuple :func:`support.payloads.set_node` writes.
    """
    return (float(bias), float(response), float(aggregation), float(activation))


def _states(common, genome_object):
    """The two formulation states of one genome: the shipped default and the accelerator.

    Oracle: the two documented formulations — ``"host"`` is the absent entry's
    default and ``"accelerator"`` is this module's subject. The assertions here keep
    a *value* test from silently running both legs on the same spelling: a lost entry
    would make ``prepare_pass`` return the same ``sweep`` twice and every equality
    below vacuous.

    Args:
        common: The ``tensorneat.common`` package (``State``, ``kernel_mode``).
        genome_object: The genome whose configuration the state extends.

    Returns:
        ``{"host": State, "accelerator": State}``.
    """
    host = genome_object.setup(common.State(kernel_mode="host"))
    accelerator = genome_object.setup(common.State(kernel_mode="accelerator"))
    assert common.kernel_mode(host) == "host"
    assert common.kernel_mode(accelerator) == "accelerator"
    return {"host": host, "accelerator": accelerator}


def _legs(run_transform, genome_object, states, payload):
    """``{leg: (state, schedule)}`` for one payload, on already-built states.

    Notes:
        The schedule is derived per state, because the accelerator formulation
        encodes the incoming windows differently, and it must never be rebuilt by
        hand inside a test: ``genome.transform`` is the only source of a
        ``Transformed``.

    Args:
        run_transform: The ``run(genome_object, state, payload) -> Transformed``
            fixture of ``./conftest.py``.
        genome_object: The genome whose statics match ``payload``.
        states: Output of :func:`_states`.
        payload: The payload batch to derive the schedule of.

    Returns:
        A dict whose values are ``(state, transformed)`` pairs.
    """
    return {
        leg: (state, run_transform(genome_object, state, payload))
        for leg, state in states.items()
    }


def _forward(run_forward, genome_object, legs, leg, payload, inputs):
    """Evaluate one leg's forward pass on one batch.

    Args:
        run_forward: The ``run(genome_object, state, transformed, payload, inputs)``
            fixture of ``./conftest.py``.
        genome_object: The genome under test.
        legs: Output of :func:`_legs`.
        leg: ``"host"`` or ``"accelerator"``.
        payload: The payload batch.
        inputs: ``(P * samples, num_inputs) float32`` rows, individual-major.

    Returns:
        ``(P * samples, num_outputs)`` numpy read-out.
    """
    state, schedule = legs[leg]
    return run_forward(genome_object, state, schedule, payload, inputs)


def _assert_same(host, accelerator, label):
    """Assert two legs returned the same array, shape and dtype included.

    Oracle: the other leg. The comparison is exact and ``NaN``-aware because both
    spellings run the same ``_node_step`` on the same operands: a tolerance would
    hide a level that one spelling visited and the other skipped, and
    ``equal_nan=True`` keeps a poisoned read-out comparable instead of demanding a
    value.

    Args:
        host: Output of the host leg.
        accelerator: Output of the accelerator leg.
        label: What the two arrays were evaluated on, for the failure message.
    """
    assert host.shape == accelerator.shape, (
        f"{label}: the two legs returned different shapes, {host.shape} vs "
        f"{accelerator.shape}"
    )
    assert host.dtype == accelerator.dtype, (
        f"{label}: the two legs returned different dtypes, {host.dtype} vs "
        f"{accelerator.dtype}"
    )
    assert np.array_equal(host, accelerator, equal_nan=True), (
        f"{label}: the accelerator sweep and the host loop disagree: "
        f"{host.ravel()} vs {accelerator.ravel()}"
    )


def _sigmoid(values):
    """The reference logistic sigmoid (the default gene's ``ACT.sigmoid`` option)."""
    return 1.0 / (1.0 + np.exp(-values))


def _total(counts):
    """Total traced ops of one route, ``while`` bodies included."""
    return sum(counts.values())


# --------------------------------------------------------------------------- #
# fixtures: one shared chain, and one trace of each of the three routes
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def chain(common, genome, payloads, run_transform):
    """A shallow chain payload, its ``max_depth=8`` genome, and both formulation legs.

    ``payloads.chain_payload(3, 8, 16)`` wires ``0, 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7``
    on three identical individuals, so the read-out (row ``7``) sits at level ``6``:
    inside a ``max_depth=8`` unroll, past a ``max_depth=3`` one.

    Returns:
        ``(payload, genome_object, states, legs)`` — see :func:`_states` and
        :func:`_legs`.
    """
    payload = payloads.chain_payload(3, 8, 16)
    genome_object = _genome(genome, max_depth=8)
    states = _states(common, genome_object)
    return payload, genome_object, states, _legs(
        run_transform, genome_object, states, payload
    )


@pytest.fixture(scope="module")
def traced(runner, common, chain):
    """Op-name counts of the three forward routes, one ``evoxc.trace`` each.

    Returns:
        ``{"host": counts, "sweep": counts, "tiled": counts}``. ``host`` is the host
        formulation at ``samples == 1``, ``sweep`` the accelerator formulation at
        ``samples == 1`` (the statically unrolled spelling) and ``tiled`` the
        accelerator formulation on a ``(P * 4, num_inputs)`` batch, which
        ``forward`` derives as ``samples == 4`` (the ``while_loop`` fallback). Each
        value maps an op name to its count over the whole traced module, nested
        region bodies included.

    Notes:
        The three calls share one genome and one payload and differ only in the two
        statics ``prepare_pass`` reads — the formulation on the state and the sample
        count derived from the batch width — so the trace is the proof that the
        routes diverged rather than a comparison of two builds of the same graph.
        ``forward`` is called on a closure here (a bound method is not traceable
        directly), with the schedule rebuilt field by field inside it.
    """
    import evoxc

    payload, genome_object, _states_by_leg, legs = chain
    width = len(common.Transformed._fields)
    tiled = np.repeat(INPUTS, 4, axis=0)
    routes = (
        ("host", legs["host"], INPUTS),
        ("sweep", legs["accelerator"], INPUTS),
        ("tiled", legs["accelerator"], tiled),
    )
    counts = {}
    for route, (state, schedule), inputs in routes:
        args = (*schedule, *payload.args(), inputs)
        specs = [runner.spec_like(array) for array in args]

        def forward_route(*arrays, state=state, schedule=schedule):
            """One ``forward`` call with the schedule rebuilt inside the trace."""
            return genome_object.forward(
                state, common.Transformed(*arrays[:width]), *arrays[width:]
            )

        counts[route] = op_names(evoxc.trace(forward_route, *specs))
    return counts


# --------------------------------------------------------------------------- #
# values: samples == 1, the sweep against the loop
# --------------------------------------------------------------------------- #
def test_shallow_chain_agrees_exactly_with_its_closed_form(
    chain, run_forward
):
    """Invariant: on a chain within ``max_depth`` the sweep equals the loop and the algebra.

    Oracle: the other leg (exact), then the hand-written closed form. Rows
    ``2..6`` are hidden, each activated by the default gene's ``sigmoid``, every
    edge has ``weight=1`` and every attribute row is ``(bias=0, response=1)``, so::

        v2 = sigmoid(x0 + x1),  v3 = sigmoid(v2),  ...,  v6 = sigmoid(v5)
        out = v6                      # the output row is never activated

    Assertion: the two legs are array-equal (shape, dtype and ``NaN`` pattern
    included), every output is finite and at least one is non-zero — so the
    equality is not the vacuous ``NaN == NaN`` of a network that never ran — and the
    accelerator read-out matches the five-composition closed form within float32
    accumulation. The chain is longer than one level and its three input rows differ
    (one negative pair, one mixed pair, one all-zero row), so a sweep that skipped or
    reordered a level cannot reproduce the value.
    """
    payload, genome_object, _states_by_leg, legs = chain
    host = _forward(run_forward, genome_object, legs, "host", payload, INPUTS)
    accelerator = _forward(
        run_forward, genome_object, legs, "accelerator", payload, INPUTS
    )
    _assert_same(host, accelerator, "shallow chain, samples=1")
    assert host.shape == (INPUTS.shape[0], 1)
    assert np.isfinite(host).all(), (
        f"the chain must compute every read-out at max_depth=8, got {host.ravel()}"
    )
    assert np.abs(host).max() > 0.0, (
        "every read-out is zero: the equality above would be vacuous"
    )
    expected = INPUTS[:, 0].astype(np.float64) + INPUTS[:, 1].astype(np.float64)
    for _ in range(5):  # hidden rows 2..6, one sigmoid each; row 7 is read out raw
        expected = _sigmoid(expected)
    np.testing.assert_allclose(accelerator[:, 0], expected, rtol=1e-5, atol=1e-6)


def test_a_chain_deeper_than_max_depth_stays_nan_on_both_legs(
    common, genome, run_transform, run_forward, chain
):
    """Invariant: past ``max_depth`` neither spelling visits a level, and the value is ``NaN``.

    Oracle: the documented depth limit plus the control build. The chain's output
    row is at level ``6``; with ``max_depth=3`` only levels ``1..3`` run, so the
    read-out keeps its ``NaN`` seed on *both* legs — ``NaN``, not ``0``: the bound
    never writes the row, and a falsy ``0`` would mean the sweep had computed it. The
    same payload under ``max_depth=8`` is finite on both legs and both agree, which
    is what tells the depth bound from a broken spelling.

    Assertion: the two shallow legs are array-equal and all ``NaN``; the shallow
    accelerator read-out is *not* the deep build's finite value (a sweep that
    silently unrolled every level would return it); the deep legs agree and are
    finite.
    """
    payload, genome_object, _states_by_leg, deep_legs = chain
    deep_host = _forward(run_forward, genome_object, deep_legs, "host", payload, INPUTS)
    deep_accelerator = _forward(
        run_forward, genome_object, deep_legs, "accelerator", payload, INPUTS
    )
    _assert_same(deep_host, deep_accelerator, "chain within max_depth=8")
    assert np.isfinite(deep_host).all(), (
        f"the control build must compute the whole chain, got {deep_host.ravel()}"
    )
    shallow_genome = _genome(genome, max_depth=3)
    shallow_legs = _legs(
        run_transform, shallow_genome, _states(common, shallow_genome), payload
    )
    host = _forward(run_forward, shallow_genome, shallow_legs, "host", payload, INPUTS)
    accelerator = _forward(
        run_forward, shallow_genome, shallow_legs, "accelerator", payload, INPUTS
    )
    _assert_same(host, accelerator, "chain past max_depth=3")
    assert np.isnan(host).all(), (
        "the level-6 output row was computed by the host loop at max_depth=3: "
        f"{host.ravel()}"
    )
    assert np.isnan(accelerator).all(), (
        "the level-6 output row was computed by the accelerator sweep at "
        f"max_depth=3, so the unroll ran past the bound: {accelerator.ravel()}"
    )
    assert not np.array_equal(accelerator, deep_accelerator), (
        "the shallow sweep returned the deep build's finite read-out "
        f"{deep_accelerator.ravel()}: the unroll visited levels past max_depth"
    )


def test_the_bypassed_node_never_reaches_the_read_out_on_either_leg(
    payloads, run_transform, run_forward, chain
):
    """Invariant: a node its consumer bypasses is dead on the sweep too, whatever it holds.

    Oracle: the other leg, plus a perturbation argument on the payload itself.
    ``payloads.skip_payload`` wires ``0, 1 -> 2`` and ``0 -> 3``, ``3 -> 7``: row
    ``2`` receives the whole input signal but reaches no output, and the schedule
    puts row ``3`` at level ``1`` — not at ``2`` — which is the level assignment a
    level-synchronous sweep must reproduce.

    Assertion: the two legs agree exactly on a finite, non-zero read-out; row ``2``
    is a real present node; and poisoning row ``2``'s value columns with ``NaN``
    leaves the *accelerator* read-out identical to the unperturbed one (the
    perturbation is applied only there, so the host leg's value is the independent
    side of the comparison). Option-id columns stay finite: a ``NaN`` id would emit a
    ``NaN``-to-``int32`` cast warning for a row that never matters.
    """
    _payload, genome_object, states, _chain_legs = chain
    base = payloads.skip_payload(3, 8, 16)
    legs = _legs(run_transform, genome_object, states, base)
    host = _forward(run_forward, genome_object, legs, "host", base, INPUTS)
    accelerator = _forward(run_forward, genome_object, legs, "accelerator", base, INPUTS)
    _assert_same(host, accelerator, "skipped-node topology")
    assert np.isfinite(host).all(), (
        f"the skipped-node read-out must be finite, got {host.ravel()}"
    )
    assert np.abs(host).max() > 0.0, "every read-out is zero: nothing was computed"
    assert (base.node_keys[:, 2] != payloads.I_INF).all(), (
        "row 2 must be a present node for the bypass to mean anything"
    )
    poisoned = payloads.skip_payload(3, 8, 16)
    for individual in range(poisoned.pop_size):
        payloads.set_node(
            poisoned, individual, 2, 2, _node_row(np.nan, np.nan, 0.0, 0.0)
        )
    poisoned_legs = _legs(run_transform, genome_object, states, poisoned)
    poisoned_accelerator = _forward(
        run_forward, genome_object, poisoned_legs, "accelerator", poisoned, INPUTS
    )
    assert np.array_equal(poisoned_accelerator, accelerator, equal_nan=True), (
        "poisoning the bypassed row changed the accelerator read-out, so the "
        f"sweep let it reach an output: {poisoned_accelerator.ravel()} vs "
        f"{accelerator.ravel()}"
    )


def test_an_unconnected_node_stays_nan_into_the_read_out_on_both_legs(
    payloads, run_transform, run_forward, chain
):
    """Invariant: an unconnected node keeps ``NaN`` and poisons a read-out with no other source.

    Oracle: the documented poisoning rule — a node with no live incoming slot never
    enters an ``update_mask``, so its value stays the ``NaN`` seed, and its consumer
    then loses its only live source and keeps ``NaN`` as well. Both spellings
    recompute the mask per level, so both must show it.

    Two payloads, one per half of the rule. ``payloads.isolated_payload`` leaves row
    ``2`` present but edge-less next to a dense input-to-output layer, so the network
    still reads out and both legs must agree on that finite value. The second layout
    wires ``2 -> 3 -> 7`` with finite attribute rows *and* with a huge
    ``bias``/``response`` on the isolated row: attributes cannot rescue a node that
    never runs, so the read-out is ``NaN`` on both legs after two hops.
    """
    _payload, genome_object, states, _chain_legs = chain
    isolated = payloads.isolated_payload(3, 8, 16)
    isolated_legs = _legs(run_transform, genome_object, states, isolated)
    host = _forward(run_forward, genome_object, isolated_legs, "host", isolated, INPUTS)
    accelerator = _forward(
        run_forward, genome_object, isolated_legs, "accelerator", isolated, INPUTS
    )
    _assert_same(host, accelerator, "isolated node next to a dense layer")
    assert np.isfinite(host).all(), (
        f"the dense input-to-output layer must read out, got {host.ravel()}"
    )
    assert np.abs(host).max() > 0.0, "every read-out is zero: nothing was computed"
    assert (isolated.node_keys[:, 2] != payloads.I_INF).all(), (
        "row 2 must be a present node for it to be an unconnected one"
    )

    def poisoned(row):
        """Rows ``0, 1`` in, row ``2`` isolated, row ``3`` its consumer, row ``7`` out."""
        batch = payloads.new_payload(3, 8, 16)
        for individual in range(batch.pop_size):
            payloads.set_node(batch, individual, 0, 0, _node_row())
            payloads.set_node(batch, individual, 1, 1, _node_row())
            payloads.set_node(batch, individual, 2, 2, row)
            payloads.set_node(batch, individual, 3, 3, _node_row(0.5, 1.0, 0.0, 0.0))
            payloads.set_node(batch, individual, 7, 7, _node_row())
            payloads.set_conn(batch, individual, 0, 2, 3, (1.0,))
            payloads.set_conn(batch, individual, 1, 3, 7, (1.0,))
        return batch

    for label, row in (
        ("finite attributes", _node_row(0.5, 1.0, 0.0, 0.0)),
        ("huge attributes", _node_row(1e9, 1e9, 0.0, 0.0)),
    ):
        batch = poisoned(row)
        legs = _legs(run_transform, genome_object, states, batch)
        host = _forward(run_forward, genome_object, legs, "host", batch, INPUTS)
        accelerator = _forward(
            run_forward, genome_object, legs, "accelerator", batch, INPUTS
        )
        _assert_same(host, accelerator, f"unconnected node, {label}")
        assert np.isnan(accelerator).all(), (
            f"the accelerator sweep turned the unconnected node's NaN ({label}) into "
            f"a finite read-out: {accelerator.ravel()}"
        )
        assert np.isnan(host).all(), (
            f"the host loop turned the unconnected node's NaN ({label}) into a "
            f"finite read-out: {host.ravel()}"
        )


# --------------------------------------------------------------------------- #
# values: samples == 4, the tiled fallback
# --------------------------------------------------------------------------- #
def test_a_tiled_batch_matches_the_host_leg_and_repeats_each_individual(
    chain, run_forward
):
    """Invariant: a tiled batch routes both legs through the loop and tiles individual-major.

    Oracle: the other leg, then the *same* individual's ``samples == 1`` result.
    ``forward`` has no ``samples`` argument — it derives ``samples = len(inputs) //
    pop_size`` — so a ``(P * 4, num_inputs)`` batch of the same rows is the tiled
    spelling, and ``prepare_pass`` keeps the ``while_loop`` for it on *both*
    formulations (the unroll would multiply a ``P * samples``-row batch by
    ``max_depth``).

    Assertion: the two legs agree exactly on the ``(P * 4, num_outputs)`` read-out;
    every row is finite; and because each individual's four input rows are identical,
    the four output rows of individual ``p`` repeat that individual's ``samples == 1``
    read-out — which the accelerator takes through the *sweep*, so this also
    cross-checks the two spellings per individual and pins the individual-major tiling
    (row ``p * samples + j`` is individual ``p``, not row ``j * P + p``).
    """
    payload, genome_object, _states_by_leg, legs = chain
    samples = 4
    tiled = np.repeat(INPUTS, samples, axis=0)
    assert tiled.shape == (INPUTS.shape[0] * samples, INPUTS.shape[1])
    host = _forward(run_forward, genome_object, legs, "host", payload, tiled)
    accelerator = _forward(
        run_forward, genome_object, legs, "accelerator", payload, tiled
    )
    _assert_same(host, accelerator, "tiled chain, samples=4")
    assert accelerator.shape == (INPUTS.shape[0] * samples, 1)
    assert np.isfinite(accelerator).all(), (
        f"the tiled chain must compute every row, got {accelerator.ravel()}"
    )
    single = _forward(run_forward, genome_object, legs, "accelerator", payload, INPUTS)
    grouped = accelerator.reshape(INPUTS.shape[0], samples, 1)
    expected = np.repeat(single[:, None, :], samples, axis=1)
    assert np.array_equal(grouped, expected, equal_nan=True), (
        "the four tiled rows of an individual do not repeat its samples==1 "
        f"read-out, so the tiling is not individual-major: {grouped.ravel()} vs "
        f"{expected.ravel()}"
    )


# --------------------------------------------------------------------------- #
# the route-selected proof: traced op names
# --------------------------------------------------------------------------- #
def test_the_accelerator_sweep_traces_no_while_loop_and_unrolls(traced):
    """Invariant: the accelerator ``samples == 1`` route is the statically unrolled sweep.

    Oracle: the traced IR. ``_level_sweep`` is a Python loop over ``level = 1 ..
    max_depth`` at trace time, so its graph holds ``max_depth`` copies of
    :meth:`._node_step` and **no** region op at all — and, because it reads neither
    the depth bound nor the loop counter, no ``reduce_max`` dispatch either. The host
    graph of the same genome and payload holds exactly one of each.

    Assertion: zero ``while`` ops, zero ``reduce_max`` ops, and strictly more total
    ops than the host leg (the unroll is more work than one loop, which is why the
    numpy interpreter does not take it). This is the proof that the two spellings
    really ran: the value tests above would also pass if both legs took the same
    path.
    """
    sweep, host = traced["sweep"], traced["host"]
    assert _total(sweep) > 0, "the traced sweep graph is empty"
    assert sweep.get("while", 0) == 0, (
        "the accelerator sweep traced a region op; the inner level step must be "
        f"statically unrolled, got {sweep.get('while', 0)} while op(s)"
    )
    assert sweep.get("reduce_max", 0) == 0, (
        "the accelerator sweep derived the loop's depth bound "
        f"({sweep.get('reduce_max', 0)} reduce_max op(s)); it reads no limit"
    )
    assert _total(sweep) > _total(host), (
        f"the unrolled sweep traced {_total(sweep)} ops against the host leg's "
        f"{_total(host)}: an unroll should cost more than one loop body"
    )


def test_the_host_leg_traces_the_while_loop(traced):
    """Invariant: the host formulation keeps the data-dependent ``while_loop``.

    Oracle: the traced IR plus the source's own contract — ``_level_loop`` walks
    ``level = 1 .. limit`` in one ``evoxc.while_loop``, where ``limit`` is
    ``_depth_limit``'s ``reduce_max`` clamped to ``[1, max_depth]``.

    Assertion: exactly one ``while`` op (the op name of ``evoxc.while_loop`` in this
    tree's evoxc version) and exactly one ``reduce_max`` — the bound the loop reads
    and the sweep does not.
    """
    host = traced["host"]
    assert _total(host) > 0, "the traced host graph is empty"
    assert host.get("while", 0) == 1, (
        "the host leg must walk the levels in one while_loop, got "
        f"{host.get('while', 0)} while op(s)"
    )
    assert host.get("reduce_max", 0) == 1, (
        "the host loop must derive its trip count from reduce_max, got "
        f"{host.get('reduce_max', 0)} such op(s)"
    )


def test_a_tiled_accelerator_batch_keeps_the_while_loop(traced):
    """Invariant: ``samples == 1`` — not the formulation alone — selects the unroll.

    Oracle: the traced IR of the same genome, payload and state at ``samples == 4``.
    A tiled batch keeps ``_level_loop`` on purpose: each unrolled level would cost
    ``P * samples`` rows, and the loop's per-level host readback is amortised over one
    query per generation instead of one per rollout timestep.

    Assertion: exactly one ``while`` op and one ``reduce_max`` (the loop's own bound),
    and fewer total ops than the accelerator ``samples == 1`` sweep of the same graph —
    the loop holds one level step, not ``max_depth`` of them. A gate reduced to the
    formulation static alone fails this test.
    """
    tiled, sweep = traced["tiled"], traced["sweep"]
    assert _total(tiled) > 0, "the traced tiled graph is empty"
    assert tiled.get("while", 0) == 1, (
        "the tiled accelerator batch must fall back to the loop, got "
        f"{tiled.get('while', 0)} while op(s)"
    )
    assert tiled.get("reduce_max", 0) == 1, (
        "the tiled loop must derive its trip count from reduce_max, got "
        f"{tiled.get('reduce_max', 0)} such op(s)"
    )
    assert _total(tiled) < _total(sweep), (
        f"the tiled accelerator leg traced {_total(tiled)} ops against the "
        f"samples==1 sweep's {_total(sweep)}: the tiled batch was unrolled"
    )
