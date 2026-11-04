"""Unit tests for the batched Pendulum rollout of ``problem/rl/pendulum.py``.

What this module proves
-----------------------
``Pendulum`` is a fixed-horizon batched Euler rollout in evoxc ops with **no
termination**: one ``evoxc.while_loop`` over ``max_step`` timesteps, carry
``(obs (B, 2), total (B,), step ())`` with ``B = P * repeat_times`` in
individual-major order, and a reward of ``-cost`` read from the
**pre-transition** state, where ::

    u          = max_torque * tanh(logit)
    cost       = angle_normalize(theta)**2 + 0.1*theta_dot**2 + 0.001*u**2
    theta_dot' = clip(theta_dot + (gravity_term*sin(theta)
                                   + torque_term*u) * dt, +/-max_speed)
    theta'     = theta + theta_dot' * dt          # the *clipped* velocity

Fitness is the mean over ``repeat_times`` of the episode returns, then
``finalize_fitness``, so every return lives in ``[-max_step * max_cost, 0]``.
The legacy RL problems were deleted, so there is no parity oracle: the oracle
**is** the transcription below (:func:`_pendulum_rollout`), and the closed forms
in :func:`test_the_velocity_is_clipped_before_the_angle_integrates`, in
:func:`_unwrapped_return` and in the documented cost ceiling keep it honest.
Unlike ``CartPole`` there is no ``reset_scale`` knob, so the graph's own
``_reset`` is traced in a one-op graph (:func:`_reset_observations`) and the
resulting ``(B, 2)`` block is the host ``obs0`` the oracle starts from.

Gotchas this module encodes
---------------------------
* The two coefficients are precomputed **host-side** (``gravity_term =
  3*gravity/(2*length)``, ``torque_term = 3/(mass*length**2)``); the oracle must
  use the same ``float32`` values or bit-equality is lost to rounding.
* ``angle_normalize`` is the **floor-mod** spelling
  ``x - two_pi*floor((x + pi)/two_pi)``, not ``evoxc.remainder`` (``fmod`` on
  stablehlo/iree): a row that spins several revolutions would otherwise have its
  return inflated by two orders of magnitude.
* The velocity is clipped *before* the angle integrates: from ``theta_dot = 7.9``
  at maximum torque, ``theta`` advances by ``8.0*dt``, not ``8.2*dt``.
* There is no NaN census: ``tanh(NaN)`` makes the control penalty NaN, the
  carried accumulator NaN, and ``finalize_fitness`` maps it to ``-inf``.

Everything runs on the numpy backend with small batches (``pop <= 4``,
``max_step <= 200``): the interpreter pays per traced op, not per element.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# genome-free policies: plain ``enp`` closures, inlined into the trace
# --------------------------------------------------------------------------- #


def _constant_logits(enp, value):
    """A policy emitting the same single logit for every row and every step."""

    def act_fn(batch):
        n = batch.shape[0]
        return enp.full((n, 1), value)

    return act_fn


def _linear_logits(enp, k_theta, k_speed):
    """``logit = k_theta*theta + k_speed*theta_dot`` -- plain arithmetic on purpose.

    :func:`_linear_logit_matrix` replays the same expression host-side; keeping it
    free of transcendental ops keeps the two bit-equal.
    """

    def act_fn(batch):
        cols = enp.split(batch, 2, axis=1)
        return k_theta * cols[0] + k_speed * cols[1]

    return act_fn


def _nan_policy(enp, split_rows=None, tilt=None):
    """NaN logits on the rows before ``split_rows``, or once ``|theta| > tilt``."""

    def act_fn(batch):
        n = batch.shape[0]
        theta = enp.split(batch, 2, axis=1)[0]
        if tilt is not None:
            poisoned = enp.greater(enp.abs(theta), tilt)
        else:
            poisoned = enp.expand_dims(enp.less(enp.arange(n), split_rows), 1)
        return enp.where(
            poisoned, enp.full((n, 1), float("nan")), enp.full((n, 1), 1.0)
        )

    return act_fn


# --------------------------------------------------------------------------- #
# the numpy oracle: an independent transcription of the documented recurrence
# --------------------------------------------------------------------------- #


def _physics(pendulum):
    """The constants bundle as the trace derives it: both coefficients on the host."""
    return {
        "gravity_term": np.float32(3.0 * pendulum.gravity / (2.0 * pendulum.length)),
        "torque_term": np.float32(3.0 / (pendulum.mass * pendulum.length**2)),
        "dt": pendulum.dt,
        "max_speed": pendulum.max_speed,
        "max_torque": pendulum.max_torque,
    }


def _angle_normalize(angles, real):
    """Wrap angles into ``[-pi, pi)`` with the floor-mod form the kernel uses."""
    pi = real.type(np.pi)
    two_pi = real.type(2.0 * np.pi)
    return angles - two_pi * np.floor((angles + pi) / two_pi)


def _pendulum_rollout(
    logits,
    obs0,
    max_step,
    *,
    dtype=np.float32,
    gravity_term,
    torque_term,
    dt=0.05,
    max_speed=8.0,
    max_torque=2.0,
):
    """Roll out ``obs0`` for ``max_step`` steps; return ``(total (B,), states (T, B, 2))``.

    ``states`` holds the pre-transition observations :func:`_unwrapped_return`
    reads. The accumulator is summed **sequentially**: a vectorized ``sum``
    rounds differently and breaks bit-exactness at 200 steps.
    """
    real = np.dtype(dtype)
    obs = np.array(obs0, dtype=real)
    logits = np.asarray(logits, dtype=real)
    total = np.zeros(obs.shape[0], dtype=real)
    states = np.empty((max_step,) + obs.shape, dtype=real)
    for index in range(max_step):
        states[index] = obs
        theta, theta_dot = obs[:, 0], obs[:, 1]
        torque = real.type(max_torque) * np.tanh(logits[:, index])
        angle = _angle_normalize(theta, real)
        cost = np.square(angle) + real.type(0.1) * np.square(theta_dot)
        cost += real.type(0.001) * np.square(torque)
        drive = real.type(gravity_term) * np.sin(theta) + real.type(torque_term) * torque
        next_speed = theta_dot + drive * real.type(dt)
        next_speed = np.clip(next_speed, real.type(-max_speed), real.type(max_speed))
        obs = np.stack([theta + next_speed * real.type(dt), next_speed], axis=1)
        total = total - cost
    return total, states


def _unwrapped_return(states, logits, *, dtype=np.float32, max_torque=2.0):
    """The same trajectory scored with the **raw** angle, i.e. with no wrapping.

    Deliberately the *wrong* reference: it is what the return would be if the
    kernel forgot ``angle_normalize``, so asserting the graph is far from it is
    what proves the wrap is exercised by the policy under test.
    """
    real = np.dtype(dtype)
    logits = np.asarray(logits, dtype=real)
    total = np.zeros(states.shape[1], dtype=real)
    for index in range(states.shape[0]):
        theta, theta_dot = states[index, :, 0], states[index, :, 1]
        torque = real.type(max_torque) * np.tanh(logits[:, index])
        total = total - (
            np.square(theta)
            + real.type(0.1) * np.square(theta_dot)
            + real.type(0.001) * np.square(torque)
        )
    return total


def _linear_logit_matrix(obs0, max_step, k_theta, k_speed, physics):
    """The ``(B, T)`` logits the policy emitted, replayed on the oracle's own trajectory."""
    logits = np.zeros((obs0.shape[0], max_step), dtype=np.float32)
    obs = np.array(obs0, dtype=np.float32)
    for index in range(max_step):
        logits[:, index] = np.float32(k_theta) * obs[:, 0] + np.float32(k_speed) * obs[:, 1]
        torque = np.float32(physics["max_torque"]) * np.tanh(logits[:, index])
        next_speed = obs[:, 1] + (
            physics["gravity_term"] * np.sin(obs[:, 0]) + physics["torque_term"] * torque
        ) * np.float32(physics["dt"])
        speed = np.float32(physics["max_speed"])
        next_speed = np.clip(next_speed, -speed, speed)
        obs = np.stack([obs[:, 0] + next_speed * np.float32(physics["dt"]), next_speed], axis=1)
    return logits


def _reset_observations(run_in_graph, evoxc, pendulum, batch, key_seed):
    """Recover the ``(batch, 2)`` initial states the graph draws, host-side."""

    def kernel(_unused):
        return pendulum._reset(evoxc.random.key(key_seed), pendulum._constants(), batch)

    obs0 = run_in_graph(kernel, np.zeros((1,), dtype=np.float32))
    assert obs0.shape == (batch, 2)
    assert obs0.dtype == np.float32
    return obs0


# --------------------------------------------------------------------------- #
# agreement with the numpy oracle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("max_step", [20, 60, 200])
@pytest.mark.parametrize("logit", [0.0, 2.0, -2.0], ids=["zero_torque", "push", "pull"])
def test_a_constant_torque_policy_matches_the_float32_oracle(
    state, problem, enp, evoxc, run_in_graph, run_evaluate, fake_schedule, max_step, logit
):
    """Invariant: a constant torque reproduces the float32 transcription bit-for-bit.

    Oracle: :func:`_pendulum_rollout` at the same horizon, fed the very initial
    states the graph draws and the coefficients :func:`_physics` precomputes the
    way the constructor chain does. Measured ``max_abs = 0.0`` on every row at
    every horizon, so the assertion is exact equality, not a tolerance.

    Notes:
        The three logits are the three regimes of a constant policy -- no torque
        (the all-zero-logit policy) and saturated torque in both directions
        (``u = +/-2*tanh(2) = +/-1.93``); nothing terminates, so the return
        strictly decreases with ``max_step``.
    """
    pendulum = problem.Pendulum(max_step=max_step)
    fitness = run_evaluate(
        pendulum, state, _constant_logits(enp, logit), fake_schedule(2), key_seed=0
    )
    obs0 = _reset_observations(run_in_graph, evoxc, pendulum, 2, 0)
    logits = np.full((2, max_step), logit, dtype=np.float32)
    expected, _ = _pendulum_rollout(logits, obs0, max_step, **_physics(pendulum))

    assert fitness.shape == (2,)
    assert fitness.dtype == np.float32
    np.testing.assert_array_equal(fitness, expected)
    assert np.all(np.isfinite(fitness)) and np.all(fitness < 0.0)


@pytest.mark.parametrize("max_step", [20, 60, 200])
def test_a_state_dependent_policy_agrees_with_both_oracles(
    state, problem, enp, evoxc, run_in_graph, run_evaluate, fake_schedule, max_step
):
    """Invariant: a closed-loop policy is bit-equal to the float32 and ``~1e-5`` to the float64 oracle.

    Oracle: two transcriptions of the same recurrence. In ``float32`` the graph
    must match exactly (measured ``max_abs = 0.0``), which proves the rounding
    order -- precomputed coefficients, clip before the angle update, sequential
    accumulator. In ``float64`` it may differ only by float32 rounding of the
    trajectory: measured ``max_rel`` 1.9e-7 at 20 steps, 1.2e-7 at 60 and 3.6e-6
    at 200, so ``rtol=1e-5`` is a documented rounding bound, not slack.
    """
    pendulum = problem.Pendulum(max_step=max_step)
    fitness = run_evaluate(
        pendulum, state, _linear_logits(enp, 1.5, 0.5), fake_schedule(2), key_seed=3
    )
    obs0 = _reset_observations(run_in_graph, evoxc, pendulum, 2, 3)
    physics = _physics(pendulum)
    logits = _linear_logit_matrix(obs0, max_step, 1.5, 0.5, physics)

    tight, _ = _pendulum_rollout(logits, obs0, max_step, **physics)
    loose, _ = _pendulum_rollout(
        logits, obs0.astype(np.float64), max_step, dtype=np.float64, **physics
    )

    np.testing.assert_array_equal(fitness, tight)
    np.testing.assert_allclose(fitness, loose, rtol=1e-5)
    assert np.all(fitness < 0.0)


# --------------------------------------------------------------------------- #
# the return range
# --------------------------------------------------------------------------- #


def test_the_return_stays_in_the_documented_negative_range(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: ``-max_step*max_cost <= fitness <= 0`` for any policy and key.

    Oracle: the documented algebra -- each step subtracts ``angle**2 +
    0.1*theta_dot**2 + 0.001*u**2``, bounded by ``pi**2``, ``0.1*max_speed**2``
    and ``0.001*max_torque**2`` (``16.2736`` per step at the defaults), and
    nothing is ever added. This is the range the ``fitness_target <= 0``
    rejection rests on; the lower bound carries a ``1e-2`` absolute slack because
    the accumulator is summed in float32.
    """
    max_step = 60
    pendulum = problem.Pendulum(max_step=max_step)
    ceiling = np.pi**2 + 0.1 * pendulum.max_speed**2 + 0.001 * pendulum.max_torque**2
    lower = -max_step * ceiling

    for name, policy in {
        "zero_torque": _constant_logits(enp, 0.0),
        "saturated": _constant_logits(enp, 30.0),
        "linear": _linear_logits(enp, 1.5, 0.5),
    }.items():
        for key_seed in range(3):
            fitness = run_evaluate(pendulum, state, policy, fake_schedule(4), key_seed=key_seed)
            assert fitness.shape == (4,), name
            assert fitness.dtype == np.float32, name
            assert np.all(np.isfinite(fitness)), name
            assert np.all(fitness <= 0.0), name
            assert np.all(fitness >= lower - 1e-2), name


# --------------------------------------------------------------------------- #
# repeat_times, randomness, NaN
# --------------------------------------------------------------------------- #


def test_repeat_times_averages_individual_major_rows(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: ``B = P * R`` is individual-major and the fitness is the mean over ``R``.

    Oracle: an identity between two runs, not a number. ``P = 4, R = 1`` exposes
    the four per-episode returns; ``P = 2, R = 2`` draws *the same* four initial
    states from the same key -- the draw is one ``(B, 2)`` block, so it depends on
    ``B`` alone -- and must pair them ``[(r0+r1)/2, (r2+r3)/2]``; a repeat-major
    layout would pair ``(r0, r2)`` and ``(r1, r3)``.

    Notes:
        The four returns are distinct because the four initial states are distinct
        draws, asserted first so the layout check cannot pass vacuously; the
        *mean* rather than the sum is pinned by the row-sum comparison.
    """
    max_step = 60
    policy = _linear_logits(enp, 1.5, 0.5)
    per_row = run_evaluate(
        problem.Pendulum(max_step=max_step), state, policy, fake_schedule(4), key_seed=5
    )
    paired = run_evaluate(
        problem.Pendulum(max_step=max_step, repeat_times=2),
        state,
        policy,
        fake_schedule(2),
        key_seed=5,
    )

    assert per_row.shape == (4,)
    assert len({float(value) for value in per_row}) == 4
    assert per_row[1] != per_row[2]

    individual_major = np.array(
        [(per_row[0] + per_row[1]) / 2, (per_row[2] + per_row[3]) / 2], dtype=np.float32
    )
    repeat_major = np.array(
        [(per_row[0] + per_row[2]) / 2, (per_row[1] + per_row[3]) / 2], dtype=np.float32
    )
    summed = np.array([per_row[0] + per_row[1], per_row[2] + per_row[3]], dtype=np.float32)

    assert not np.allclose(repeat_major, individual_major)
    np.testing.assert_allclose(paired, individual_major)
    assert not np.allclose(paired, repeat_major)
    assert not np.allclose(paired, summed)


def test_the_same_key_reproduces_the_run_and_another_key_diverges(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: the rollout is a pure function of ``(key, config, shape)``.

    Oracle: run-to-run comparison. The same ``key_seed`` twice must give
    bit-identical fitness; across five seeds the vectors must not all be equal
    ("more than one distinct vector", which cannot pass by coincidence the way a
    single pair could). The reset is one ``evoxc.random.uniform(key, (B, 2))``
    and the key is never threaded through the loop, so this is what would catch a
    reintroduced global ``np.random``.
    """
    pendulum = problem.Pendulum(max_step=60)
    schedule = fake_schedule(4)
    policy = _linear_logits(enp, 1.5, 0.5)

    np.testing.assert_array_equal(
        run_evaluate(pendulum, state, policy, schedule, key_seed=7),
        run_evaluate(pendulum, state, policy, schedule, key_seed=7),
    )
    runs = [
        run_evaluate(pendulum, state, policy, schedule, key_seed=key_seed)
        for key_seed in range(5)
    ]
    for run in runs:
        assert run.shape == (4,)
        assert np.all(np.isfinite(run))
    assert len({tuple(float(value) for value in run) for run in runs}) > 1


def test_nan_logits_poison_the_fitness_to_negative_infinity(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: a NaN logit poisons the episode through the control penalty, to ``-inf``.

    Oracle: closed form on the documented propagation. ``tanh(NaN) = NaN``, so
    ``0.001*u**2`` is NaN, the step's cost is NaN and the carried accumulator
    becomes NaN -- there is **no** NaN census, the poison rides the arithmetic.
    A NaN emitted at *any* step poisons the rest of the episode too, because the
    NaN state is carried; ``finalize_fitness`` then maps it to ``-inf``.

    Notes:
        Three independent cases plus a finite control, so ``-inf`` cannot come
        from a rollout that failed for another reason: a permanently NaN policy,
        one that only turns NaN after ``|theta| > 0.5`` (proving the poison
        survives the carry), and a row-dependent policy where only the poisoned
        rows are demoted -- the finite rows equal the control's rows.
    """
    nan = float("nan")
    pendulum = problem.Pendulum(max_step=20)
    permanently = run_evaluate(
        pendulum, state, _constant_logits(enp, nan), fake_schedule(2), key_seed=0
    )
    assert permanently.shape == (2,)
    assert np.all(np.isneginf(permanently))

    late = run_evaluate(
        pendulum, state, _nan_policy(enp, tilt=0.5), fake_schedule(2), key_seed=0
    )
    assert np.all(np.isneginf(late))

    control = run_evaluate(
        pendulum, state, _constant_logits(enp, 1.0), fake_schedule(3), key_seed=0
    )
    assert np.all(np.isfinite(control))
    mixed = run_evaluate(
        pendulum, state, _nan_policy(enp, split_rows=1), fake_schedule(3), key_seed=0
    )
    np.testing.assert_array_equal(
        mixed, np.array([-np.inf, control[1], control[2]], dtype=np.float32)
    )


# --------------------------------------------------------------------------- #
# Gym-semantics spot checks: the angle wrap and the velocity clip
# --------------------------------------------------------------------------- #


def test_the_angle_wrap_is_the_floor_mod_form(
    state, problem, enp, evoxc, run_in_graph, run_evaluate, fake_schedule
):
    """Invariant: angles are wrapped by whole turns, and a spinning policy is scored on the wrapped angle.

    Oracle: (a) the closed form ``x - 2*pi*floor((x + pi)/(2*pi))`` in
    ``float32``, checked on the helper at ``0``, ``+/-1``, ``+/-pi``,
    ``+/-3pi +/- 0.1``, ``+/-7pi +/- 0.25``, ``1000 rad`` and ``+/-pi`` one ulp
    inside the boundary; (b) the same recurrence end to end, where
    :func:`_pendulum_rollout` must equal the graph bit-for-bit while
    :func:`_unwrapped_return` -- the *same* trajectory scored with the raw angle
    -- must be far larger in magnitude on a row that spins.

    Notes:
        ``+/-pi`` maps to ``-pi`` (the window is half-open on the right, like
        Gym) and ``3pi + 0.1`` maps to ``-(pi - 0.1)``. A ``remainder`` spelling
        would keep the sign of the dividend and return ``-3pi`` for ``-3pi`` on
        the stablehlo/iree path, which is why the kernel uses ``evoxc.floor``. At
        ``max_step = 200`` one row of this key spins ~12 turns (``|theta| ~ 75``
        rad, unwrapped cost ~226x the wrapped one) while another stays near the
        window (~1.1x apart), so the gap is asserted on the rows that really left
        it, plus the universal "a wrap never grows a cost".
    """
    pendulum = problem.Pendulum(max_step=20)
    pi32 = np.float32(np.pi)
    angles = np.array(
        [
            0.0, 1.0, -1.0, pi32, -pi32, 3.0 * np.pi + 0.1, -3.0 * np.pi - 0.1,
            7.0 * np.pi + 0.25, -7.0 * np.pi - 0.25, 1e3,
            pi32 - np.float32(1e-6), -pi32 + np.float32(1e-6),
        ],
        dtype=np.float32,
    ).reshape(-1, 1)

    def kernel(values):
        return pendulum._angle_normalize(values, pendulum._constants())

    got = run_in_graph(kernel, angles)
    np.testing.assert_array_equal(got, _angle_normalize(angles, np.dtype(np.float32)))
    assert got.shape == angles.shape
    assert np.all(got >= -np.pi) and np.all(got < np.pi)
    assert float(got[3, 0]) == pytest.approx(-np.pi, abs=1e-6)  # +pi folds onto -pi
    assert float(got[5, 0]) == pytest.approx(-(np.pi - 0.1), abs=1e-5)
    assert float(got[7, 0]) == pytest.approx(-(np.pi - 0.25), abs=1e-5)

    max_step = 200
    spinning = problem.Pendulum(max_step=max_step)
    fitness = run_evaluate(
        spinning, state, _constant_logits(enp, 30.0), fake_schedule(2), key_seed=0
    )
    obs0 = _reset_observations(run_in_graph, evoxc, spinning, 2, 0)
    logits = np.full((2, max_step), 30.0, dtype=np.float32)
    expected, states = _pendulum_rollout(logits, obs0, max_step, **_physics(spinning))
    unwrapped = _unwrapped_return(states, logits, max_torque=spinning.max_torque)
    spun = np.abs(states[:, :, 0]).max(axis=0) > 7.0 * np.pi

    np.testing.assert_array_equal(fitness, expected)
    assert spun.any()  # at least one row spins past a whole turn
    assert np.all(np.abs(unwrapped) >= np.abs(fitness))  # a wrap never grows a cost
    assert np.all(np.abs(unwrapped[spun]) > 100.0 * np.abs(fitness[spun]))


def test_the_velocity_is_clipped_before_the_angle_integrates(state, problem, run_in_graph):
    """Invariant: ``theta_dot'`` is clipped to ``+/-max_speed`` and ``theta'`` uses the clipped value.

    Oracle: closed-form one-step arithmetic through ``Pendulum._advance(obs,
    logit, _constants())``. From ``(0, 7.9)`` at ``logit = 30`` (``u = 2``) the
    unclipped velocity would be ``7.9 + (15*sin(0) + 3*2)*0.05 = 8.2``; the
    carried velocity is exactly ``8.0`` and the angle advances by exactly
    ``8.0*0.05 = 0.4``. ``(0, -7.9)`` mirrors it and an absurd ``100`` is clipped
    the same way.

    Notes:
        The cost of the same step is closed-form on the **pre-transition** state:
        ``0.1*7.9**2 + 0.001*2**2 = 6.245``, and ``1000`` for the third row -- an
        order of magnitude above the clip bound, which is only possible if the
        cost is read before the clip.
    """
    pendulum = problem.Pendulum(max_step=20)
    obs = np.array([[0.0, 7.9], [0.0, -7.9], [0.0, 100.0]], dtype=np.float32)
    logits = np.array([[30.0], [-30.0], [0.0]], dtype=np.float32)

    def kernel(values, actions):
        return pendulum._advance(values, actions, pendulum._constants())

    next_obs, cost = run_in_graph(kernel, obs, logits)

    np.testing.assert_array_equal(
        next_obs, np.array([[0.4, 8.0], [-0.4, -8.0], [0.4, 8.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        cost, np.array([6.245, 6.245, 1000.0], dtype=np.float32), rtol=1e-6
    )


# --------------------------------------------------------------------------- #
# configuration errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"max_step": 0}, id="max_step_zero"),
        pytest.param({"max_step": -1}, id="max_step_negative"),
        pytest.param({"max_step": True}, id="max_step_bool"),
        pytest.param({"max_step": 1.5}, id="max_step_float"),
        pytest.param({"repeat_times": 0}, id="repeat_times_zero"),
        pytest.param({"repeat_times": False}, id="repeat_times_bool"),
        pytest.param({"repeat_times": 2.0}, id="repeat_times_float"),
        pytest.param({"gravity": 0.0}, id="gravity_zero"),
        pytest.param({"gravity": -1.0}, id="gravity_negative"),
        pytest.param({"gravity": float("nan")}, id="gravity_nan"),
        pytest.param({"mass": 0.0}, id="mass_zero"),
        pytest.param({"length": -1.0}, id="length_negative"),
        pytest.param({"max_speed": 0.0}, id="max_speed_zero"),
        pytest.param({"max_torque": 0.0}, id="max_torque_zero"),
        pytest.param({"dt": 0.0}, id="dt_zero"),
        pytest.param({"fitness_target": 0.5}, id="fitness_target_positive"),
    ],
)
def test_out_of_domain_configuration_raises_tensorneat_error(problem, common, kwargs):
    """Invariant: every value outside its documented domain raises ``TensorNEATError``.

    Oracle: the ``Raises:`` clause of the constructor docstring -- ``max_step``
    and ``repeat_times`` must be ints (a ``bool`` is rejected explicitly, being
    an ``int`` in Python but never a horizon), every physics scalar must be
    strictly positive (a NaN included, the check being ``not number > 0.0``), and
    ``fitness_target`` may not be positive. One case per knob, and the message
    must name the offending knob.
    """
    with pytest.raises(common.TensorNEATError) as excinfo:
        problem.Pendulum(**kwargs)
    assert next(iter(kwargs)) in str(excinfo.value)


def test_in_domain_configuration_is_accepted(problem):
    """Invariant: the defaults and the boundary values construct and are stored as given.

    Oracle: the class docstring's attribute table plus the constructor body.
    ``fitness_target = 0.0`` is the hard ceiling and is legal (only a *positive*
    value is rejected), ``None`` disables the check, and a negative target is
    stored verbatim. Arity: two observations in, one torque logit out.
    """
    defaults = problem.Pendulum()
    assert (defaults.max_step, defaults.repeat_times) == (200, 1)
    assert (defaults.gravity, defaults.mass, defaults.length) == (10.0, 1.0, 1.0)
    assert (defaults.max_speed, defaults.max_torque) == (8.0, 2.0)
    assert defaults.dt == pytest.approx(0.05)
    assert defaults.fitness_target is None
    assert (defaults.input_shape, defaults.output_shape) == ((2,), (1,))

    configured = problem.Pendulum(
        max_step=1, repeat_times=3, max_speed=4.0, fitness_target=-300.0
    )
    assert (configured.max_step, configured.repeat_times) == (1, 3)
    assert configured.max_speed == 4.0
    assert configured.fitness_target == -300.0
    assert problem.Pendulum(fitness_target=0.0).fitness_target == 0.0
