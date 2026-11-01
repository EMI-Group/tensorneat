"""Unit tests for ``RecurrentGenome.forward`` — the tick-unrolled Jacobi pass.

What this module proves
-----------------------
A recurrent genome has no depth: the schedule's incoming-edge table is the same,
but the network is unrolled ``activate_time`` times and *every* node is
recomputed from the **previous** tick's values. The tests pin down the four
properties that make that pass what it is:

* **synchronous (Jacobi) updates** — a node never reads a value produced earlier
  in the same tick (that would be Gauss-Seidel), so a cycle neither settles in
  one tick nor propagates one row per tick;
* **raw input re-injection** — the input rows hold ``inputs`` at the start of
  every tick, and ``input_transform`` is deliberately **not** applied (legacy
  parity; HyperNEAT uses this genome as a CPPN, where the transform belongs to the
  caller);
* **the tick count is the only depth knob** — ``activate_time`` applications of
  the tick map, so ``T+1`` ticks equal one more application of the same map, and
  on an oscillating ring the read-out really changes from one tick to the next
  instead of converging immediately;
* **NaN poisoning is permanent** — a node that never had a live incoming edge
  keeps ``NaN`` on every tick, and a descendant whose only source is such a node
  keeps ``NaN`` too.

Oracles used here (never the kernel itself)
-------------------------------------------
* ``support.oracle.forward(..., network_type="recurrent", activate_time=T)`` —
  the independent numpy rollout of the same documented algorithm (seed, step,
  repeat; ``NaN`` sanitization; neutral-padded aggregation);
* closed forms on hand-built payloads whose weights/activations make the tick
  map arithmetic exact:
  ``ring_payload`` (rows ``0, 1`` inputs, ``2``/``3`` the ring, ``7`` the output;
  edges ``0->2``, ``1->2``, ``2->3``, ``3->2``, ``3->7``) with identity activation
  on the ring rows, a self-connected accumulator, a two-step network and an
  isolated-row chain (each builder below spells its own recurrence out).

Gotchas this module encodes
---------------------------
* ``RecurrentGenome`` ignores ``node_level`` — the schedule of a cyclic genome
  reports ``level == I_INF`` for every node on or downstream of the cycle, so a
  test that reused a level-driven mask would produce all ``NaN``. The update mask
  is ``node_exists & (valid_count > 0)`` instead.
* ``input_transform`` is accepted by the constructor and then never used. That is
  not a bug to fix: it is the documented parity behaviour, and the only
  observable form of it is a *non-idempotent* transform (apply it twice and the
  answer changes), which is what the test uses.
* Poisoning is *starvation*, not contagion: the aggregation neutralises a dead
  slot with its option's neutral element, so a ``NaN`` *sibling* does not poison
  a node — only losing every live source does. The ``NaN`` test therefore reads
  out through a chain whose only source is the isolated row, and a control chain
  pins the contrast.
* ``activate_time`` is a static Python int, part of the shape signature; changing
  it forces a rebuild. The tick count is therefore a configuration error, not a
  runtime value, when it is zero or negative.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# payloads whose tick map is arithmetic
# --------------------------------------------------------------------------- #

#: ``(bias, response, aggregation id 0 = sum, activation id -1 = identity)``; the
#: identity option is what makes the recurrences of this module exact arithmetic.
_IDENTITY_ROW = (0.0, 1.0, 0.0, -1.0)


def _ring(payloads):
    """``payloads.ring_payload`` with identity activation on the two ring rows.

    Edges ``0->2``, ``1->2``, ``2->3``, ``3->2``, ``3->7``, all ``weight 1``;
    rows ``0, 1`` inputs, ``2``/``3`` the ring, ``7`` the output. With ``S = x0 +
    x1`` the tick map is therefore ``v2(t+1) = S + v3(t)`` and ``v3(t+1) =
    v2(t)``, and the read-out is ``v3`` one tick late.
    """
    payload = payloads.ring_payload()
    for row in (2, 3):
        payloads.set_node(payload, 0, row, row, attrs=_IDENTITY_ROW)
    return payload


def _oscillating_ring(payloads):
    """A 2-ring with a negative recurrent weight, so its orbit never settles.

    Rows ``0, 1`` inputs (``S = x0 + x1``), rows ``2``/``3`` the ring with
    identity activation, row ``7`` the output: ``0,1 -> 2`` (``weight 1``),
    ``2 -> 3`` (``weight 1``), ``3 -> 2`` (``weight -1``), ``3 -> 7`` (``weight
    1``). The Jacobi map ``v2(t+1) = S - v3(t)``, ``v3(t+1) = v2(t)`` has a
    period-four orbit, so consecutive tick counts differ.
    """
    payload = payloads.new_payload(1, 8, 16)
    for row in (0, 1, 2, 3, 7):
        attrs = _IDENTITY_ROW if row in (2, 3) else None
        payloads.set_node(payload, 0, row, row, attrs=attrs)
    payloads.set_conn(payload, 0, 0, 0, 2)
    payloads.set_conn(payload, 0, 1, 1, 2)
    payloads.set_conn(payload, 0, 2, 2, 3)
    payloads.set_conn(payload, 0, 3, 3, 2, attrs=(-1.0,))
    payloads.set_conn(payload, 0, 4, 3, 7)
    return payload


def _self_loop_accumulator(payloads):
    """A row that only ever sees the raw inputs plus its own previous value.

    Rows ``0, 1`` inputs, row ``2`` hidden with identity activation, row ``7``
    the output: ``0,1 -> 2`` (``weight 1``), the self-loop ``2 -> 2`` (``weight
    1``), ``2 -> 7`` (``weight 1``). The tick map is ``v2(t+1) = S + v2(t)`` with
    ``S = x0 + x1``, so the read-out grows by ``S`` per tick exactly while the
    raw inputs are re-injected on every tick.
    """
    payload = payloads.new_payload(1, 8, 16)
    for row in (0, 1, 7):
        payloads.set_node(payload, 0, row, row)
    payloads.set_node(payload, 0, 2, 2, attrs=_IDENTITY_ROW)
    payloads.set_conn(payload, 0, 0, 0, 2)
    payloads.set_conn(payload, 0, 1, 1, 2)
    payloads.set_conn(payload, 0, 2, 2, 2)
    payloads.set_conn(payload, 0, 3, 2, 7)
    return payload


def _two_step_payload(payloads):
    """A network whose read-out is ``3`` after one tick and ``6`` after two.

    Rows ``0, 1`` inputs, row ``2`` hidden (identity activation) fed by the
    inputs with ``weight 0.5``, row ``7`` the output fed by the inputs with
    ``weight 1`` and by row ``2`` with ``weight 2``. For ``x = (1, 2)``: one tick
    reads only the input slots (``1 + 2 = 3``), two ticks also read the live row
    ``2`` (``3 + 2 * 0.5 * 3 = 6``).
    """
    payload = payloads.new_payload(1, 8, 16)
    for row in (0, 1, 7):
        payloads.set_node(payload, 0, row, row)
    payloads.set_node(payload, 0, 2, 2, attrs=_IDENTITY_ROW)
    payloads.set_conn(payload, 0, 0, 0, 2, attrs=(0.5,))
    payloads.set_conn(payload, 0, 1, 1, 2, attrs=(0.5,))
    payloads.set_conn(payload, 0, 2, 0, 7)
    payloads.set_conn(payload, 0, 3, 1, 7)
    payloads.set_conn(payload, 0, 4, 2, 7, attrs=(2.0,))
    return payload


def _isolated_chain(payloads, *, from_island: bool):
    """A `2 -> 3 -> 7` chain in which row ``2`` is present but has no incoming edge.

    With ``from_island=True`` row ``3`` is fed by the isolated row ``2`` only, so
    it can never go live and the read-out stays ``NaN``. With ``from_island=False``
    row ``3`` is fed by the input rows instead and the isolated row is a dead
    *sibling* slot — the aggregation neutral-fills it and the read-out is finite
    from tick two on.
    """
    payload = payloads.new_payload(1, 8, 16)
    for row in (0, 1, 2, 3, 7):
        payloads.set_node(payload, 0, row, row)
    if from_island:
        payloads.set_conn(payload, 0, 0, 2, 3)
    else:
        payloads.set_conn(payload, 0, 0, 0, 3)
        payloads.set_conn(payload, 0, 1, 1, 3)
    payloads.set_conn(payload, 0, 2, 3, 7)
    return payload


def _evaluate(
    payloads, run_transform, run_forward, payload, inputs, activate_time, **genome_kwargs
):
    """Run one payload through ``RecurrentGenome.forward`` with ``activate_time`` ticks.

    Args:
        payloads: The ``support.payloads`` module fixture.
        run_transform: The ``genome.transform`` graph runner fixture.
        run_forward: The ``genome.forward`` graph runner fixture.
        payload: The hand-built payload to evaluate.
        inputs: ``(P, num_inputs) float32`` network inputs.
        activate_time: Static tick count of the genome under test.
        **genome_kwargs: Any other ``RecurrentGenome`` keyword (``input_transform``,
            ``output_transform``, ...).

    Returns:
        ``(outputs, transformed)`` — the ``(P, num_outputs)`` read-out and the
        schedule the oracle needs to replay the same rollout.
    """
    genome = payloads.recurrent_genome(activate_time=activate_time, **genome_kwargs)
    state = payloads.genome_state(genome)
    transformed = run_transform(genome, state, payload)
    return run_forward(genome, state, transformed, payload, inputs), transformed


#: The inputs every closed form below is written for: ``S = x0 + x1 = 3``.
_INPUTS = np.array([[1.0, 2.0]], dtype=np.float32)
#: ``S`` of :data:`_INPUTS`.
_S = 3.0


# --------------------------------------------------------------------------- #
# synchronous (Jacobi) updates
# --------------------------------------------------------------------------- #


def test_same_tick_values_are_synchronous_jacobi(payloads, run_transform, run_forward):
    """Invariant: in one tick, a node reads only the *previous* tick's values.

    Oracle: closed form on :func:`_ring` with ``S = x0 + x1 = 3``. The documented
    Jacobi map is ``v2(t+1) = S + v3(t)`` and ``v3(t+1) = v2(t)`` starting from
    the ``NaN`` seed, so ``v2 = 3, 3, 6`` and ``v3 = NaN, 3, 3`` for ``t = 1, 2,
    3``; the read-out is ``v3(t-1)``, i.e. ``NaN, NaN, 3, 3, 6`` for ``t = 1..5``.
    A Gauss-Seidel implementation (row ``3`` reading row ``2``'s *new* value in
    the same tick) would give ``v3(1) = v2(1) = 3`` and therefore a read-out of
    ``3`` already at the second tick, which the ``NaN`` assertion catches.
    """
    payload = _ring(payloads)
    readout = {
        ticks: _evaluate(payloads, run_transform, run_forward, payload, _INPUTS, ticks)[0]
        for ticks in (1, 2, 3, 4, 5)
    }

    # One tick: row 2 has two live input slots, row 3 has only the NaN seed.
    assert np.isnan(readout[1]).all()
    # Two ticks: row 2 is S, row 3 would be S under a Gauss-Seidel sweep.
    assert np.isnan(readout[2]).all()
    # Three ticks: the read-out finally sees row 3's tick-2 value, i.e. row 2's 3.
    assert readout[3].ravel().tolist() == [_S]
    # Four ticks: v3(3) = v2(2) = S + v3(1) (dead) = S.
    assert readout[4].ravel().tolist() == [_S]
    # Five ticks: v3(4) = v2(3) = S + v3(2) = 2S.
    assert readout[5].ravel().tolist() == [2.0 * _S]


def test_more_ticks_apply_the_tick_map_once_more_on_a_ring(
    payloads, oracle, run_transform, run_forward
):
    """Invariant: ``activate_time = T + 1`` is the ``T`` rollout plus one more tick.

    Oracle: :func:`_oscillating_ring`, whose Jacobi map ``v2(t+1) = S - v3(t)``,
    ``v3(t+1) = v2(t)`` has a period-four orbit: the read-out is ``NaN, NaN, 3,
    3, 0, 0`` for ``t = 1..6``, so consecutive tick counts *differ* and a genome
    that silently ran one tick (or ``max_depth`` ticks) cannot pass. Each count is
    also compared with ``support.oracle.forward(..., "recurrent",
    activate_time=t)``, which by construction replays ``t-1`` ticks and then
    applies the tick map exactly once more.
    """
    payload = _oscillating_ring(payloads)
    closed_form = {1: np.nan, 2: np.nan, 3: _S, 4: _S, 5: 0.0, 6: 0.0}
    readout = {}
    for ticks in sorted(closed_form):
        output, transformed = _evaluate(
            payloads, run_transform, run_forward, payload, _INPUTS, ticks
        )
        readout[ticks] = output
        expected = oracle.forward(
            payload, transformed, _INPUTS, network_type="recurrent", activate_time=ticks
        )
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(np.isnan(output), np.isnan(expected))
        want = closed_form[ticks]
        if np.isnan(want):
            assert np.isnan(output).all(), f"{ticks} ticks should still read the NaN seed"
        else:
            assert output.ravel().tolist() == [want], f"{ticks} ticks"

    # The ring oscillates, so one further tick really changes the read-out.
    assert not np.allclose(readout[4], readout[5])


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #


def test_raw_inputs_are_reinjected_on_every_tick(payloads, oracle, run_transform, run_forward):
    """Invariant: the input rows hold the raw ``inputs`` at the start of every tick.

    Oracle: :func:`_self_loop_accumulator`, whose only non-recurrent source is the
    input block: its tick map is ``v2(t+1) = S + v2(t)`` with ``S = x0 + x1 = 3``,
    so the read-out is ``NaN, 3, 6, 9`` for ``t = 1..4`` — the raw input is added
    on *every* tick. A kernel that injected the inputs once and then let the rows
    go stale (or overwrote them with ``NaN``) would show the input contribution at
    most once, i.e. a constant read-out from tick two on.
    ``support.oracle.forward(..., "recurrent")`` seeds the input slots before each
    step and must reproduce the same arithmetic progression.
    """
    payload = _self_loop_accumulator(payloads)
    for ticks in (1, 2, 3, 4):
        output, transformed = _evaluate(
            payloads, run_transform, run_forward, payload, _INPUTS, ticks
        )
        expected = oracle.forward(
            payload, transformed, _INPUTS, network_type="recurrent", activate_time=ticks
        )
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(np.isnan(output), np.isnan(expected))
        if ticks == 1:
            assert np.isnan(output).all()
        else:
            assert output.ravel().tolist() == [(ticks - 1) * _S], f"{ticks} ticks"


def test_input_transform_is_never_applied(payloads, oracle, run_transform, run_forward):
    """Invariant: a configured ``input_transform`` changes nothing for this genome.

    Oracle: equality against the same genome with ``input_transform=None`` fed the
    raw inputs. The transform used is non-idempotent (``x -> x + 1``), so applying
    it once, or once per tick, would both be visible; not applying it at all gives
    bit-identical outputs. The counterfactual is
    ``support.oracle.forward(..., network_type="recurrent")`` fed ``inputs + 1``,
    which is what a transform-applying genome would read out and which does differ
    from the kernel's read-out. This is the documented legacy-parity behaviour of
    ``RecurrentGenome``, not an oversight.
    """
    payload = payloads.ring_payload()
    plain, transformed = _evaluate(payloads, run_transform, run_forward, payload, _INPUTS, 3)
    shifted, _ = _evaluate(
        payloads,
        run_transform,
        run_forward,
        payload,
        _INPUTS,
        3,
        input_transform=lambda x: x + 1.0,
    )

    assert np.isfinite(plain).all(), (
        "the read-out must be finite to make the comparison meaningful"
    )
    np.testing.assert_array_equal(shifted, plain)

    applied = oracle.forward(
        payload,
        transformed,
        _INPUTS + np.float32(1.0),
        network_type="recurrent",
        activate_time=3,
    )
    assert not np.allclose(applied, plain), (
        "the transform would change the read-out if it were applied, so this "
        "payload does pin the 'never applied' contract"
    )


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #


def test_output_transform_is_applied_after_the_last_tick(payloads, run_transform, run_forward):
    """Invariant: ``output_transform`` maps the ``(P, num_outputs)`` read-out of the final tick.

    Oracle: composition — the output equals the transform applied elementwise to
    the untransformed read-out of the same payload and tick count. The transform is
    non-idempotent (``x -> x + 1``) and ``activate_time = 3``, so applying it per
    tick instead of once would give ``readout + 3`` rather than ``readout + 1``,
    which the final assertion rejects.
    """
    payload = payloads.ring_payload()
    ticks = 3
    plain, _ = _evaluate(payloads, run_transform, run_forward, payload, _INPUTS, ticks)
    shifted, _ = _evaluate(
        payloads,
        run_transform,
        run_forward,
        payload,
        _INPUTS,
        ticks,
        output_transform=lambda x: x + 1.0,
    )

    assert np.isfinite(plain).all(), "the read-out must be finite to compare the transform"
    np.testing.assert_allclose(shifted, plain + np.float32(1.0), rtol=1e-6, atol=1e-6)
    assert not np.allclose(shifted, plain + np.float32(ticks)), (
        "the transform must be applied once, after the loop — not once per tick"
    )


# --------------------------------------------------------------------------- #
# NaN semantics
# --------------------------------------------------------------------------- #


def test_nan_poisoning_survives_every_tick(payloads, oracle, run_transform, run_forward):
    """Invariant: a starved node keeps ``NaN`` on every tick, and so does its descendant.

    Oracle: ``support.oracle.forward(..., network_type="recurrent")`` on
    :func:`_isolated_chain` with ``from_island=True`` — row ``2`` is present but
    has no edge at all (the unconnected-row case ``payloads.isolated_payload``
    also builds), so it never satisfies ``node_exists & (valid_count > 0)``, row
    ``3``'s only slot therefore never goes live, and the output row's only slot
    is row ``3``. The read-out is ``NaN`` at every tick, not just at tick one.

    Poisoning is *starvation*, not contagion: the aggregation neutral-fills a dead
    slot, so the control chain (the same isolated row, with row ``3`` fed by the
    inputs instead) must read out a finite value from tick two on.
    """
    poisoned = _isolated_chain(payloads, from_island=True)
    for ticks in (1, 2, 3, 4):
        output, transformed = _evaluate(
            payloads, run_transform, run_forward, poisoned, _INPUTS, ticks
        )
        expected = oracle.forward(
            poisoned, transformed, _INPUTS, network_type="recurrent", activate_time=ticks
        )
        assert np.isnan(output).all(), f"{ticks} ticks: the starved chain must stay NaN"
        assert np.isnan(expected).all(), f"{ticks} ticks: oracle disagreement"

    control = _isolated_chain(payloads, from_island=False)
    output, transformed = _evaluate(payloads, run_transform, run_forward, control, _INPUTS, 3)
    expected = oracle.forward(
        control, transformed, _INPUTS, network_type="recurrent", activate_time=3
    )
    assert np.isfinite(output).all(), (
        "an isolated row is a dead slot, not a poison: a sibling with a live source "
        "must still produce a value"
    )
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# against the independent oracle
# --------------------------------------------------------------------------- #


def test_ring_matches_the_numpy_oracle(payloads, oracle, rng, run_transform, run_forward):
    """Invariant: the cyclic ring equals ``support.oracle.forward(..., "recurrent")``.

    Oracle: the independent numpy rollout at the same ``activate_time``, on
    ``payloads.ring_payload`` with random ``weight`` per edge (drawn from the
    seeded ``rng`` fixture) and random ``bias``/``response`` per present row, and
    the single default activation option. Both the values (``allclose``) and the
    ``NaN`` pattern are compared, so a schedule that ordered the ring differently
    from the oracle cannot hide behind a tolerance.
    """
    payload = payloads.ring_payload()
    weights = rng.normal(size=payload.max_conns).astype(np.float32)
    live_edges = np.flatnonzero(payload.conn_keys[0, :, 0] != payloads.I_INF)
    for index in live_edges:
        payloads.set_conn(
            payload,
            0,
            int(index),
            int(payload.conn_keys[0, index, 0]),
            int(payload.conn_keys[0, index, 1]),
            attrs=(float(weights[index]),),
        )
    biases = rng.normal(size=payload.max_nodes).astype(np.float32)
    responses = rng.uniform(0.2, 2.0, size=payload.max_nodes).astype(np.float32)
    for row in np.flatnonzero(payload.node_keys[0] != payloads.I_INF):
        payloads.set_node(
            payload,
            0,
            int(row),
            int(payload.node_keys[0, row]),
            attrs=(float(biases[row]), float(responses[row]), 0.0, 0.0),
        )

    inputs = np.array([[0.5, -0.25]], dtype=np.float32)
    readout = {}
    for ticks in (1, 2, 3, 4):
        output, transformed = _evaluate(
            payloads, run_transform, run_forward, payload, inputs, ticks
        )
        readout[ticks] = output
        expected = oracle.forward(
            payload, transformed, inputs, network_type="recurrent", activate_time=ticks
        )
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(np.isnan(output), np.isnan(expected))

    # The first tick reads only the input rows; the read-out row's only source
    # (the second ring row) is still at the NaN seed, so it cannot be finite yet.
    assert np.isnan(readout[1]).all()
    assert np.isfinite(readout[4]).all()


def test_activate_time_of_one_evaluates_exactly_one_tick(
    payloads, oracle, run_transform, run_forward
):
    """Invariant: ``activate_time=1`` is *one* tick, not zero and not ``max_depth``.

    Oracle: closed form on :func:`_two_step_payload` — after one tick the hidden
    row is not computed yet, so the read-out is ``x0 + x1 = 3``; after two ticks
    it is ``x0 + x1 + 2 * 0.5 * (x0 + x1) = 6``. Zero ticks would read out the
    ``NaN`` seed (a loop that starts at one, or that unrolls ``range(1, T)``,
    fails here while every ``T >= 2`` test still passes) and ``max_depth`` ticks
    would already read the two-tick value.
    ``support.oracle.forward(..., activate_time=1)`` must agree at each count.
    """
    payload = _two_step_payload(payloads)
    one, one_schedule = _evaluate(payloads, run_transform, run_forward, payload, _INPUTS, 1)
    two, two_schedule = _evaluate(payloads, run_transform, run_forward, payload, _INPUTS, 2)

    assert one.ravel().tolist() == [_S]
    assert two.ravel().tolist() == [2.0 * _S]
    for ticks, output, schedule in ((1, one, one_schedule), (2, two, two_schedule)):
        expected = oracle.forward(
            payload, schedule, _INPUTS, network_type="recurrent", activate_time=ticks
        )
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(np.isnan(output), np.isnan(expected))


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("invalid", [0, -1, -7, 1.5, 2.0, "3", None, True])
def test_activate_time_must_be_a_positive_int(payloads, common, invalid):
    """Invariant: an invalid ``activate_time`` raises ``TensorNEATError`` at construction.

    Oracle: the documented configuration-error contract. Zero ticks would read out
    the ``NaN`` seed (a silently dead population), a negative count would unroll
    nothing, and a non-int is not a static shape fact at all — so all of them must
    be reported as ``TensorNEATError`` (the package hierarchy), not as a bare
    ``ValueError``/``TypeError`` from deep inside a trace. ``True`` is rejected even
    though it is an ``int`` subclass, because ``range(True)`` silently means one tick.
    """
    with pytest.raises(common.TensorNEATError):
        payloads.recurrent_genome(activate_time=invalid)


@pytest.mark.parametrize("ticks", [1, 5, np.int64(7)])
def test_a_positive_activate_time_is_recorded(payloads, ticks):
    """Invariant: a positive ``int`` (including a numpy scalar) is stored as a host ``int``.

    Oracle: the constructor contract — the tick count is a static Python ``int``
    baked into the trace, so ``range(self.activate_time)`` must receive an ``int``
    rather than the caller's numpy scalar.
    """
    genome = payloads.recurrent_genome(activate_time=ticks)
    assert genome.activate_time == int(ticks)
    assert isinstance(genome.activate_time, int)
