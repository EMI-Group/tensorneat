"""Unit tests for ``RecurrentGenome.forward`` — the tick-unrolled Jacobi pass.

What this module proves
-----------------------
A recurrent genome has no depth: the schedule's incoming-edge table is the same,
but the network is unrolled ``activate_time`` times and *every* node is
recomputed from the **previous** tick's values. The tests pin down the four
properties that make that pass what it is:

* **synchronous (Jacobi) updates** — two nodes that feed each other exchange
  their values in one tick; they do not settle, and a node never reads a value
  produced earlier in the same tick (that would be Gauss-Seidel);
* **raw input re-injection** — the input rows hold ``inputs`` at the start of
  every tick, and ``input_transform`` is deliberately **not** applied (legacy
  parity; HyperNEAT uses this genome as a CPPN, where the transform belongs to the
  caller);
* **the tick count is the only depth knob** — ``activate_time`` applications of
  the tick map, so ``T+1`` ticks equal one more application of the same map, and
  on an oscillating ring the read-out really changes from one tick to the next
  instead of converging immediately;
* **NaN poisoning is permanent** — a node that never had a live incoming edge
  keeps ``NaN`` on every tick, and so does everything downstream of it.

Oracles used here (never the kernel itself)
-------------------------------------------
* ``support.oracle.forward(..., network_type="recurrent", activate_time=T)`` —
  the independent numpy rollout of the same documented algorithm (seed, step,
  repeat; ``NaN`` sanitization; neutral-padded aggregation);
* closed forms on ``payloads.ring_payload`` (rows ``0, 1`` inputs, ``2``/``3`` the
  ring, ``7`` the output; edges ``0->2``, ``1->2``, ``2->3``, ``3->2``, ``3->7``),
  where the identity activation makes one tick an exchange of the two ring rows.

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
* ``activate_time`` is a static Python int, part of the shape signature; changing
  it forces a rebuild. The tick count is therefore a configuration error, not a
  runtime value, when it is zero or negative.

Backend is ``"numpy"`` throughout, through the shared runner only; no
``jax``/``torch``, no network, no GPU.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# synchronous (Jacobi) updates
# --------------------------------------------------------------------------- #


def test_same_tick_values_are_synchronous_jacobi(payloads, run_transform, run_forward):
    """Invariant: in one tick, the two rows of a ring exchange their values.

    Oracle: closed form on ``payloads.ring_payload`` with the identity activation
    and ``weight=1``: ``v2(t+1) = v3(t)`` and ``v3(t+1) = v2(t)``. After the
    first tick the pair is swapped; a Gauss-Seidel implementation (row ``3``
    reading row ``2``'s *new* value) would instead give ``v3(t+1) = v2(t+1) =
    v3(t)``, i.e. a constant, which the assertion catches.
    """
    pytest.skip(
        "not implemented: one tick reads only the previous tick's values "
        "(owner: genome/recurrent.RecurrentGenome.forward, genome/base.BaseGenome._node_step)"
    )


def test_more_ticks_apply_the_tick_map_once_more_on_a_ring(payloads, run_transform, run_forward):
    """Invariant: ``T+1`` ticks equal one further application of the tick map.

    Oracle: the host recurrence. Running the same payload with ``activate_time =
    T`` and with ``activate_time = T + 1`` must differ by exactly one tick — the
    ``T+1`` read-out equals the tick map applied to the ``T`` values — and on the
    oscillating ring consecutive tick counts must *differ*, so a genome that
    silently ran one tick (or ``max_depth`` ticks) cannot pass.
    """
    pytest.skip(
        "not implemented: activate_time is the number of unrolled ticks "
        "(owner: genome/recurrent.RecurrentGenome.forward)"
    )


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #


def test_raw_inputs_are_reinjected_on_every_tick(payloads, oracle, run_transform, run_forward):
    """Invariant: the input rows hold the raw ``inputs`` at the start of every tick.

    Oracle: ``support.oracle.forward(..., network_type="recurrent")``, which seeds
    the input slots before each step. A node fed only by input rows therefore sees
    ``f(inputs)`` at every tick, never an accumulator of previous ticks — a kernel
    that seeded once and read a stale row would drift, and one that accumulated
    would change from tick to tick on a constant input.
    """
    pytest.skip(
        "not implemented: inputs are written into the input rows on every tick "
        "(owner: genome/recurrent.RecurrentGenome.forward)"
    )


def test_input_transform_is_never_applied(payloads, run_transform, run_forward):
    """Invariant: a configured ``input_transform`` changes nothing for this genome.

    Oracle: equality against the same genome with ``input_transform=None`` fed the
    raw inputs. The transform used is non-idempotent (``x -> x + 1``), so
    applying it once, or once per tick, would both be visible; not applying it at
    all gives bit-identical outputs. This is the documented legacy-parity
    behaviour of ``RecurrentGenome``, not an oversight.
    """
    pytest.skip(
        "not implemented: RecurrentGenome ignores input_transform "
        "(owner: genome/recurrent.RecurrentGenome.forward)"
    )


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #


def test_output_transform_is_applied_after_the_last_tick(payloads, run_transform, run_forward):
    """Invariant: ``output_transform`` maps the ``(P, num_outputs)`` read-out of the final tick.

    Oracle: composition — the output equals the transform applied elementwise to
    the untransformed read-out of the same payload and tick count. Applying it
    per tick (a common mistake when the read-out is taken inside the loop) gives a
    different value after two or more ticks, which is what the chosen
    non-idempotent transform exposes.
    """
    pytest.skip(
        "not implemented: output_transform applies once, after the final tick "
        "(owner: genome/recurrent.RecurrentGenome.forward, genome/base.BaseGenome._output_values)"
    )


# --------------------------------------------------------------------------- #
# NaN semantics
# --------------------------------------------------------------------------- #


def test_nan_poisoning_survives_every_tick(payloads, oracle, run_transform, run_forward):
    """Invariant: a node with no live incoming edge keeps ``NaN`` on every tick.

    Oracle: ``support.oracle.forward(..., network_type="recurrent")`` plus the
    documented mask rule. A present-but-isolated row (``payloads.isolated_payload``,
    row ``2``) never satisfies ``node_exists & (valid_count > 0)``, so it stays at
    the ``NaN`` seed for all ``T`` ticks; adding one edge from it to the output row
    must make the read-out ``NaN`` too, at every tick — not just at tick one.
    """
    pytest.skip(
        "not implemented: isolated rows stay NaN across all ticks and poison descendants "
        "(owner: genome/base.BaseGenome._node_step, genome/recurrent.RecurrentGenome.forward)"
    )


# --------------------------------------------------------------------------- #
# against the independent oracle
# --------------------------------------------------------------------------- #


def test_ring_matches_the_numpy_oracle(payloads, oracle, run_transform, run_forward):
    """Invariant: the cyclic ring equals ``support.oracle.forward(..., "recurrent")``.

    Oracle: the independent numpy rollout at the same ``activate_time``, on
    ``payloads.ring_payload`` with random weights, random ``bias``/``response``
    and one activation option. Both the values (``allclose``) and the ``NaN``
    pattern are compared, so a schedule that ordered the ring differently from the
    oracle cannot hide behind a tolerance.
    """
    pytest.skip(
        "not implemented: recurrent ring equals the numpy oracle "
        "(owner: genome/recurrent.RecurrentGenome.forward)"
    )


def test_activate_time_of_one_evaluates_exactly_one_tick(
    payloads, oracle, run_transform, run_forward
):
    """Invariant: ``activate_time=1`` is *one* window reduce, not zero and not ``max_depth``.

    Oracle: ``support.oracle.forward(..., activate_time=1)`` on the dense initial
    network of ``payloads.recurrent_genome()`` and on a hand-built payload. This
    is the boundary case of the tick loop (a loop that starts at one, or that
    unrolls ``range(1, T)``, fails here while every ``T >= 2`` test still passes).
    """
    pytest.skip(
        "not implemented: activate_time=1 evaluates exactly one tick "
        "(owner: genome/recurrent.RecurrentGenome.forward)"
    )


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_activate_time_must_be_a_positive_int(payloads):
    """Invariant: ``activate_time <= 0`` raises ``TensorNEATError`` at construction.

    Oracle: the documented configuration-error contract. Zero ticks would read out
    the ``NaN`` seed (a silently dead population) and a negative count would
    unroll nothing, so both are configuration errors — and they must be reported
    as ``TensorNEATError`` (the package hierarchy), not as a bare ``ValueError``
    from deep inside a trace.
    """
    pytest.skip(
        "not implemented: activate_time <= 0 raises TensorNEATError "
        "(owner: genome/recurrent.RecurrentGenome.__init__)"
    )
