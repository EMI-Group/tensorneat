"""Unit tests for the batched MountainCarContinuous rollout of ``problem/rl/mountaincar.py``.

``MountainCarContinuous`` is a fixed-horizon batched Euler rollout in evoxc ops:
one ``evoxc.while_loop`` over ``max_step`` steps, carry ``(obs (B, 2), alive (B,),
total (B,), step ())`` with ``B = P * repeat_times`` individual-major, and Gym's
transition, in Gym's order, read off the **pre-transition** state ::

    a          = tanh(logit)                    # in [-1, 1]; the problem squashes
    force      = clip(a, -1, 1)                 # never changes a, see the gotchas
    velocity'  = clip(velocity + force*power - gravity*cos(3*position), +/-max_speed)
    position'  = clip(position + velocity', min_position, max_position)
    velocity'  = 0 where position' == min_position and velocity' < 0
    done       = position' >= goal_position and velocity' >= goal_velocity
    reward     = (100 if done else 0) - 0.1*a**2

The accumulator is masked by the **pre-step** ``alive`` flag and the state is
frozen with ``where(alive, next_obs, obs)``, so the ``100`` bonus is paid once;
fitness is the mean over ``repeat_times``, then ``finalize_fitness``. There is no
``reset_scale`` knob (unlike ``CartPole``), so the graph's own ``_reset`` is traced in
a one-op graph and its ``(B, 2)`` block seeds :func:`_mountaincar_rollout`.

Gotchas: ``-0.1 * max_step`` is only the *approximate* saturated floor (``-2.000000238``
at 20, ``-5.9999967`` at 60, ``-99.89905`` at 999); the idle policy returns exactly ``0.0``
and is the deceptive optimum the example converges to; the right wall does **not** zero the
velocity while the left wall does, by exact float equality; ``|tanh| <= 1`` makes
``clip(a, -1, 1)`` unobservable; a NaN poisons a live episode to ``-inf``, dropped once the
episode is done; ``ATTAINABLE_CEILING`` is ``93.0`` (the nominal ``100`` bonus is paid
once while the penalty is charged on every scoring step), and ``show`` prints ``n/a`` for a
non-finite return instead of claiming the goal was missed.

The long-horizon oracle is the transcription :func:`_mountaincar_rollout`; because it shares
its recurrence with nothing else, :func:`_one_step_by_hand` derives single transitions from
the published equations (explicit coefficients, numeric literals, ``position != 0``) and
:func:`test_one_step_matches_hand_derived_values_at_non_zero_position` pins the kernel
against it. Numpy backend, ``pop <= 4``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


#: A shorter, slower, steeper track: every physical constant of the kernel differs.
CUSTOM_GEOMETRY = {
    "min_position": -1.0, "max_position": 0.5, "max_speed": 0.05, "goal_position": 0.4,
    "goal_velocity": 0.0, "power": 0.002, "gravity": 0.003,
}

#: The oracle cases: (horizon, geometry) at the defaults and on the custom track.
ORACLE_CASES = {"default_20": (20, {}), "default_60": (60, {}),
                "custom_100": (100, CUSTOM_GEOMETRY)}

#: The knobs :func:`_physics` reads off a configuration.
PHYSICS_KEYS = ("min_position", "max_position", "max_speed", "goal_position",
                "goal_velocity", "power", "gravity")


# --------------------------------------------------------------------------- #
# genome-free policies: plain ``enp`` closures, inlined into the trace
# --------------------------------------------------------------------------- #


def _constant_logits(enp, value):
    """A policy emitting the same single logit for every row (NaN gives the poisoned one)."""

    def act_fn(batch):
        return enp.full((batch.shape[0], 1), value)

    return act_fn


def _sign_velocity_logits(enp, magnitude=30.0):
    """``a = sign(velocity)`` -- the bang-bang controller that pumps the car to the goal."""

    def act_fn(batch):
        n = batch.shape[0]
        velocity = enp.split(batch, 2, axis=1)[1]
        head, tail = enp.full((n, 1), magnitude), enp.full((n, 1), -magnitude)
        return enp.where(enp.greater(velocity, 0.0), head,
                         enp.where(enp.less(velocity, 0.0), tail, enp.full((n, 1), 0.0)))

    return act_fn


def _split_logits(enp, split_rows, head_value, tail_value):
    """Rows before ``split_rows`` emit ``head_value``, the rest ``tail_value``.

    Two families, so a per-episode row is traceable: a sign-symmetric policy cannot pin
    the ``(P, R)`` layout, because ``a = +1`` and ``a = -1`` both return ``-3.4801521`` at
    60 steps. ``head_value`` is ``30.0`` (a saturated run, ``-5.9999967``), ``0.0`` or
    ``NaN`` for the poisoning tests.
    """

    def act_fn(batch):
        n = batch.shape[0]
        head = enp.expand_dims(enp.less(enp.arange(n), split_rows), 1)
        return enp.where(head, enp.full((n, 1), head_value), enp.full((n, 1), tail_value))

    return act_fn


def _state_logits(enp, k_position, k_velocity, bias):
    """``logit = k_position*position + k_velocity*velocity + bias``, a closed-loop policy."""

    def act_fn(batch):
        n = batch.shape[0]
        position, velocity = (enp.reshape(c, (n,)) for c in enp.split(batch, 2, axis=1))
        return enp.reshape(k_position * position + k_velocity * velocity + bias, (n, 1))

    return act_fn


def _physics(mountaincar):
    """The geometry of one configuration, as the constructor stores it."""
    return {key: getattr(mountaincar, key) for key in PHYSICS_KEYS}


def _mountaincar_rollout(obs0, max_step, logit_fn, *, dtype=np.float32, **physics):
    """Transcribe Gym's transition for ``max_step`` steps and return the ``(B,)`` returns.

    The documented order, the ``100`` bonus masked by the **pre-step** alive flag, the
    state frozen once done, the accumulator summed sequentially as the carried ``total``
    is; ``logit_fn(obs) -> (B,)`` runs on the oracle's own trajectory and ``dtype``
    selects the bit-exact (``float32``) or the rounding cross-check (``float64``) run.
    """
    real = np.dtype(dtype)
    obs = np.array(obs0, dtype=real)
    total = np.zeros(obs.shape[0], dtype=real)
    alive = np.ones(obs.shape[0], dtype=bool)
    zero, one = real.type(0.0), real.type(1.0)
    speed, low, high = (real.type(physics[k]) for k in
                        ("max_speed", "min_position", "max_position"))
    for _ in range(max_step):
        action = np.tanh(np.asarray(logit_fn(obs), dtype=real))
        position, velocity = obs[:, 0], obs[:, 1]
        drive = np.clip(action, -one, one) * real.type(physics["power"])
        next_velocity = velocity + drive - real.type(physics["gravity"]) * np.cos(
            real.type(3.0) * position
        )
        next_velocity = np.clip(next_velocity, -speed, speed)
        next_position = np.clip(position + next_velocity, low, high)
        reset = (next_position == low) & (next_velocity < zero)
        next_velocity = np.where(reset, zero, next_velocity)
        done = (next_position >= real.type(physics["goal_position"])) & (
            next_velocity >= real.type(physics["goal_velocity"])
        )
        penalty = real.type(0.1) * np.square(action)
        total = total + np.where(alive, np.where(done, real.type(100.0), zero) - penalty, zero)
        alive = alive & ~done
        obs = np.where(alive[:, None], np.stack([next_position, next_velocity], 1), obs)
    return total


def _reset_observations(run_in_graph, evoxc, mountaincar, batch, key_seed):
    """Recover the ``(B, 2)`` initial states the graph draws, host-side."""

    def kernel(_unused):
        return mountaincar._reset(evoxc.random.key(key_seed), mountaincar._constants(), batch)

    obs0 = run_in_graph(kernel, np.zeros((1,), dtype=np.float32))
    assert (obs0.shape, obs0.dtype) == ((batch, 2), np.dtype(np.float32))
    return obs0


def _drive_body(run_in_graph, evoxc, enp, mountaincar, policy, obs, alive, calls=1):
    """Run ``_loop_body`` ``calls`` times from a hand-built carry; return its leaves per call.

    Every leaf returns in the order ``(obs, alive, total, index)``, so the freeze, the
    masked bonus and the advancing counter are all assertable from one graph.
    """

    def kernel(obs_value, alive_value):
        body = mountaincar._loop_body(policy, mountaincar._constants())
        carry = (obs_value, enp.greater(alive_value, 0.5),
                 enp.zeros((obs_value.shape[0],), dtype=evoxc.float32),
                 enp.zeros((), dtype=evoxc.int32))
        flat = ()
        for _ in range(calls):
            carry = body(carry)
            flat = flat + tuple(carry)
        return flat

    return run_in_graph(kernel, obs, alive)


# --------------------------------------------------------------------------- #
# agreement with the numpy oracle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("max_step", [20, 60])
@pytest.mark.parametrize(
    "logit", [0.0, 1.0, 30.0, -30.0], ids=["idle", "partial", "saturated", "saturated_negative"]
)
def test_a_constant_policy_matches_the_float32_oracle(
    state, problem, enp, evoxc, run_in_graph, run_evaluate, fake_schedule, max_step, logit
):
    """Invariant: a constant policy reproduces the float32 transcription bit-for-bit.

    Oracle: :func:`_mountaincar_rollout` on the block ``_reset`` draws; measured
    ``max_abs = 0.0`` for all eight cases, so this is exact equality. The logits are the
    three regimes of a constant policy -- idle (``a = 0``, exactly ``0.0``), partial
    (``a = tanh(1)``) and both saturated directions (``a = +/-1``), neither reaching the goal.
    """
    mountaincar = problem.MountainCarContinuous(max_step=max_step)
    fitness = run_evaluate(
        mountaincar, state, _constant_logits(enp, logit), fake_schedule(2), key_seed=0
    )
    obs0 = _reset_observations(run_in_graph, evoxc, mountaincar, 2, 0)
    expected = _mountaincar_rollout(
        obs0, max_step, lambda obs: np.full(obs.shape[0], logit), **_physics(mountaincar)
    )

    assert (fitness.shape, fitness.dtype) == ((2,), np.dtype(np.float32))
    np.testing.assert_array_equal(fitness, expected)
    assert np.all(np.isfinite(fitness)) and np.all(fitness <= 0.0)


@pytest.mark.parametrize("max_step,geometry", ORACLE_CASES.values(), ids=list(ORACLE_CASES))
def test_the_float64_oracle_agrees_within_float32_rounding(
    state, problem, enp, evoxc, run_in_graph, run_evaluate, fake_schedule, max_step, geometry
):
    """Invariant: the same recurrence in ``float64`` agrees to ``rtol=1e-5``.

    Oracle: :func:`_mountaincar_rollout` twice on the recovered ``obs0``, in ``float32``
    (bit-equal to the graph) and in ``float64``: measured ``max_rel`` 2.3e-7 at 20 and 5.6e-7
    at 60 steps, and 9.6e-7 for :data:`CUSTOM_GEOMETRY`, which changes all five physical
    constants at once -- a kernel reading a default instead of a configured knob fails here.
    """
    mountaincar = problem.MountainCarContinuous(max_step=max_step, **geometry)
    physics = _physics(mountaincar)
    for logit in (1.0, 30.0):
        fitness = run_evaluate(
            mountaincar, state, _constant_logits(enp, logit), fake_schedule(2), key_seed=0
        )
        obs0 = _reset_observations(run_in_graph, evoxc, mountaincar, 2, 0)

        def logit_fn(obs):
            return np.full(obs.shape[0], logit)

        tight = _mountaincar_rollout(obs0, max_step, logit_fn, **physics)
        loose = _mountaincar_rollout(
            obs0.astype(np.float64), max_step, logit_fn, dtype=np.float64, **physics
        )
        np.testing.assert_array_equal(fitness, tight)
        np.testing.assert_allclose(fitness, loose, rtol=1e-5)
        assert np.all(fitness < 0.0)


# --------------------------------------------------------------------------- #
# the reset draw, repeat_times, randomness, NaN
# --------------------------------------------------------------------------- #


def test_the_reset_draws_inside_the_valley_at_rest(problem, evoxc, run_in_graph):
    """Invariant: ``position ~ U(-0.6, -0.4)`` and ``velocity == 0`` for the whole batch.

    Oracle: Gym's reset distribution -- every position inside the interval, every velocity
    bitwise zero, another key drawing another block. The draw is one ``U(-1, 1)`` block
    rescaled by ``-0.5 + 0.1 * raw``, so a wrong centre or half-width lands outside it.
    """
    mountaincar = problem.MountainCarContinuous(max_step=20)
    first = _reset_observations(run_in_graph, evoxc, mountaincar, 32, 0)
    second = _reset_observations(run_in_graph, evoxc, mountaincar, 32, 1)

    assert np.all(first[:, 0] >= np.float32(-0.6)) and np.all(first[:, 0] <= np.float32(-0.4))
    np.testing.assert_array_equal(first[:, 1], np.zeros(32, dtype=np.float32))
    assert np.any(first != second)


def test_repeat_times_averages_individual_major_rows(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: ``B = P * R`` is individual-major and the fitness is the mean over ``R``.

    Oracle: an identity between two runs. ``P = 4, R = 1`` exposes the four per-episode
    returns; ``P = 2, R = 2`` draws *the same* four states (the draw depends on ``B`` alone)
    and must pair them ``[(r0+r1)/2, (r2+r3)/2]`` -- a repeat-major layout would average the
    two policy families into two equal ``-2.9999983`` rows instead of ``[-5.9999967, 0.0]``.
    """
    per_row = run_evaluate(
        problem.MountainCarContinuous(max_step=60), state, _split_logits(enp, 2, 30.0, 0.0),
        fake_schedule(4), key_seed=5,
    )
    paired = run_evaluate(
        problem.MountainCarContinuous(max_step=60, repeat_times=2), state,
        _split_logits(enp, 2, 30.0, 0.0), fake_schedule(2), key_seed=5,
    )

    np.testing.assert_array_equal(
        per_row, np.array([-5.9999967, -5.9999967, 0.0, 0.0], dtype=np.float32)
    )
    individual_major = np.array([per_row[0], per_row[2]], dtype=np.float32)
    repeat_major = np.array(
        [(per_row[0] + per_row[2]) / 2, (per_row[1] + per_row[3]) / 2], dtype=np.float32
    )
    assert not np.allclose(repeat_major, individual_major)
    np.testing.assert_allclose(paired, individual_major, rtol=1e-6)


def test_the_same_key_reproduces_the_run_and_another_key_diverges(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: the rollout is a pure function of ``(key, config, shape)``.

    Oracle: run-to-run comparison. The same ``key_seed`` twice must give bit-identical
    fitness; across five seeds the vectors must not all be equal -- a single pair could
    match by coincidence, five cannot. The key is never threaded through the loop, so this
    would catch a reintroduced global ``np.random``; the policy is state-dependent, because
    an idle one would return ``0.0`` everywhere.
    """
    mountaincar = problem.MountainCarContinuous(max_step=60)
    schedule, policy = fake_schedule(4), _state_logits(enp, 6.0, 2.0, 3.0)

    np.testing.assert_array_equal(
        run_evaluate(mountaincar, state, policy, schedule, key_seed=7),
        run_evaluate(mountaincar, state, policy, schedule, key_seed=7),
    )
    runs = [run_evaluate(mountaincar, state, policy, schedule, key_seed) for key_seed in range(5)]
    assert all(run.shape == (4,) and np.all(np.isfinite(run)) for run in runs)
    assert len({tuple(float(value) for value in run) for run in runs}) > 1


def test_nan_logits_poison_a_live_episode_but_not_a_finished_one(
    state, problem, enp, evoxc, run_in_graph, run_evaluate, fake_schedule
):
    """Invariant: a live NaN is ``-inf`` and only for its row; a post-goal NaN is dropped.

    Oracle: closed form on the documented propagation. ``tanh(NaN) = NaN``, so ``0.1*a**2``
    is NaN and the accumulator becomes NaN -- there is **no** NaN census here, unlike
    ``CartPole`` -- while the goal test is false against NaN, so a poisoned episode never
    terminates. The mixed run keeps the finite rows equal to the control (measured
    ``[-inf, -1.1600516, -1.1600516]`` at ``max_step = 20``, ``key_seed = 0``), and a body
    call from a carry already ``alive = False`` leaves ``total`` finite and unchanged (the
    ``where(alive, reward, 0)`` mask) while a live ``(0, 0)`` carry turns it NaN.
    """
    mountaincar = problem.MountainCarContinuous(max_step=20)
    nan_policy = _constant_logits(enp, float("nan"))
    permanent = run_evaluate(mountaincar, state, nan_policy, fake_schedule(2), key_seed=0)
    assert np.all(np.isneginf(permanent))

    control = run_evaluate(
        mountaincar, state, _constant_logits(enp, 1.0), fake_schedule(3), key_seed=0
    )
    mixed = run_evaluate(
        mountaincar, state, _split_logits(enp, 1, float("nan"), 1.0), fake_schedule(3), key_seed=0
    )
    assert np.all(np.isfinite(control))
    np.testing.assert_array_equal(
        mixed, np.array([-np.inf, control[1], control[2]], dtype=np.float32)
    )

    finished = _drive_body(
        run_in_graph, evoxc, enp, mountaincar, nan_policy,
        np.array([[0.5, 0.001]], dtype=np.float32), np.array([0.0], dtype=np.float32),
    )
    assert float(np.asarray(finished[2])[0]) == 0.0
    assert not bool(np.asarray(finished[1])[0])
    np.testing.assert_array_equal(np.asarray(finished[0]), np.array([[0.5, 0.001]], np.float32))

    live = _drive_body(
        run_in_graph, evoxc, enp, mountaincar, nan_policy,
        np.zeros((1, 2), dtype=np.float32), np.array([1.0], dtype=np.float32),
    )
    assert np.isnan(float(np.asarray(live[2])[0]))
    assert bool(np.asarray(live[1])[0])  # NaN never terminates


# --------------------------------------------------------------------------- #
# the return range and the idle optimum
# --------------------------------------------------------------------------- #


def test_the_idle_policy_returns_exactly_zero_at_every_horizon(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: ``logit = 0`` gives fitness exactly ``0.0`` -- no bonus, no penalty.

    Oracle: closed form. ``a = tanh(0) = 0``, so every step's reward is ``0 - 0.1*0**2 = 0``
    and the accumulator keeps the exact zero it was initialized with; the car only rolls
    downhill from rest and never reaches ``goal_position``. This is the **deceptive optimum**
    the shipped example converges to (``fitness_target = -0.5`` at generation 1), so a later
    "improvement" cannot turn standing still into a positive return.
    """
    for max_step, pop_size in ((20, 3), (60, 3), (999, 1)):
        fitness = run_evaluate(
            problem.MountainCarContinuous(max_step=max_step), state,
            _constant_logits(enp, 0.0), fake_schedule(pop_size), key_seed=0,
        )
        assert fitness.dtype == np.float32
        np.testing.assert_array_equal(fitness, np.zeros(fitness.shape, np.float32))


def test_the_return_range_and_the_saturated_floor(
    state, problem, enp, evoxc, run_in_graph, run_evaluate, fake_schedule
):
    """Invariant: ``-0.1*max_step <= return <= 100``, and a saturated policy sits at that floor.

    Oracle: the documented algebra -- the ``100`` bonus is paid once, every other scoring
    step subtracts ``0.1*a**2`` with ``|a| <= 1`` -- plus the float32 transcription
    bit-for-bit for ``logit = +/-30``, where ``tanh`` saturates to exactly ``1``. Measured
    ``-2.000000238`` (20), ``-5.9999967`` (60), ``-20.00004`` (200), ``-99.89905`` (999):
    within ``1e-3`` relative of ``-0.1*max_step``, never equal to it. The range loop covers
    four policies -- idle, saturated, bang-bang, state-dependent -- times three keys.
    """
    mountaincar = problem.MountainCarContinuous(max_step=60)
    for name, policy in {
        "idle": _constant_logits(enp, 0.0),
        "saturated": _constant_logits(enp, 30.0),
        "bang_bang": _sign_velocity_logits(enp),
        "state": _state_logits(enp, 6.0, 2.0, 3.0),
    }.items():
        for key_seed in range(3):
            fitness = run_evaluate(mountaincar, state, policy, fake_schedule(4), key_seed)
            assert np.all(np.isfinite(fitness)), name
            assert np.all(fitness <= 100.0), name
            assert np.all(fitness >= -6.0 - 1e-2), name

    returns = {}
    for horizon in (20, 60, 200, 999):
        saturated = problem.MountainCarContinuous(max_step=horizon)
        obs0 = _reset_observations(run_in_graph, evoxc, saturated, 1, 0)
        for logit in (30.0, -30.0):
            fitness = run_evaluate(
                saturated, state, _constant_logits(enp, logit), fake_schedule(1), key_seed=0
            )
            expected = _mountaincar_rollout(
                obs0, horizon, lambda obs: np.full(obs.shape[0], logit), **_physics(saturated)
            )
            np.testing.assert_array_equal(fitness, expected)
            assert fitness[0] == pytest.approx(-0.1 * horizon, rel=1e-3)
            returns[(horizon, logit)] = float(fitness[0])

    assert returns[(20, 30.0)] == returns[(20, -30.0)]
    assert returns[(999, 30.0)] == pytest.approx(-99.89905, rel=1e-6)
    assert float(np.float32(np.tanh(np.float32(30.0)))) == 1.0


# --------------------------------------------------------------------------- #
# Gym-semantics spot checks: the bonus, the freeze, the walls, the clips
# --------------------------------------------------------------------------- #


def test_the_goal_bonus_is_paid_once(state, problem, enp, run_evaluate, fake_schedule):
    """Invariant: a reaching episode returns the same value at every horizon past the goal.

    Oracle: closed form on the masking rule. The bang-bang policy (``a = sign(velocity)``)
    reaches the goal between step 100 and 140 from this key's start, so ``max_step = 100``
    returns ``-9.800001`` (penalty only) while ``140``, ``160``, ``200`` and ``999`` all
    return ``89.299995`` -- the ``100`` bonus minus the ``0.1*a**2`` of the ~106 steps the
    climb took. The frozen post-goal state still satisfies ``done``, so an unmasked bonus
    would grow the return with the horizon.
    """
    returns = {}
    for max_step in (100, 140, 160, 200, 999):
        fitness = run_evaluate(problem.MountainCarContinuous(max_step=max_step), state,
                               _sign_velocity_logits(enp), fake_schedule(1), key_seed=0)
        returns[max_step] = float(fitness[0])

    assert returns[100] == pytest.approx(-9.800001, rel=1e-6)
    assert returns[100] < 0.0  # not at the goal yet: no bonus at all
    for max_step in (140, 160, 200, 999):
        assert returns[max_step] == pytest.approx(89.299995, rel=1e-6)
    assert 0.0 < returns[140] < 100.0


def test_the_body_pays_the_bonus_once_on_the_squashed_action(problem, enp, evoxc, run_in_graph):
    """Invariant: one body step scores ``bonus - 0.1*tanh(logit)**2`` and then freezes.

    Oracle: closed form on ``_loop_body``. From ``(0.5, 0.0)`` with ``logit = 1`` the
    transition reaches the goal, so ``total = 100 - 0.1*tanh(1)**2 = 99.94199`` and ``alive``
    becomes false; a second call from that carry leaves ``total`` and ``obs`` bit-identical
    while the step counter advances -- the freeze plus the pre-step mask, which is what
    keeps the bonus from being re-awarded.

    Notes:
        The penalty is charged on ``a = tanh(logit)``: from the live state ``(0, 0)`` (no
        bonus available) the body returns ``0.1*a**2`` over a logit grid -- ``-0.0580025``
        at ``1``, ``-0.0990134`` at ``3``, ``-0.1`` at ``30``. A live step is used because a
        negative force from ``(0.5, 0)`` puts the velocity below ``goal_velocity``, and
        ``|a| <= 1`` is why ``clip(a, -1, 1)`` stays unobservable.
    """
    mountaincar = problem.MountainCarContinuous(max_step=20)
    obs, alive = np.array([[0.5, 0.0]], dtype=np.float32), np.array([1.0], dtype=np.float32)
    policy = _constant_logits(enp, 1.0)
    carry = _drive_body(run_in_graph, evoxc, enp, mountaincar, policy, obs, alive, calls=2)
    first_obs, first_alive, first_total, first_index = (np.asarray(x) for x in carry[0:4])
    second_obs, _, second_total, second_index = (np.asarray(x) for x in carry[4:8])

    assert not bool(first_alive[0])
    assert float(first_total[0]) == pytest.approx(
        100.0 - 0.1 * float(np.float32(np.tanh(np.float32(1.0)))) ** 2, rel=1e-6
    )
    np.testing.assert_array_equal(first_obs, obs)  # frozen at the goal
    np.testing.assert_array_equal(second_obs, first_obs)
    np.testing.assert_array_equal(second_total, first_total)
    assert (int(first_index), int(second_index)) == (1, 2)

    for logit in (-30.0, -3.0, 0.0, 1.0, 3.0, 30.0):
        live = _drive_body(
            run_in_graph, evoxc, enp, mountaincar, _constant_logits(enp, logit),
            np.zeros((1, 2), dtype=np.float32), alive,
        )
        action = float(np.float32(np.tanh(np.float32(logit))))
        assert abs(action) <= 1.0
        assert float(np.asarray(live[2])[0]) == pytest.approx(-0.1 * action**2, abs=1e-7)
        assert bool(np.asarray(live[1])[0])


def test_the_step_clips_the_velocity_and_tests_the_goal(problem, run_in_graph):
    """Invariant: Gym's Euler step after both clips, the wall reset, and the goal predicate.

    Oracle: closed-form arithmetic through ``_advance`` on hand-built rows, the only way to
    place the car at an exact state. ``(-1.2, -0.05)`` with ``a = -1`` would move to
    ``-1.2493``; the position clips to ``-1.2`` and the velocity is reset to exactly ``0.0``
    (``clip`` returns the bound verbatim, so that equality holds). The same push from
    ``(-0.5, -0.05)`` keeps its ``-0.050176844``: the reset is not unconditional.
    ``(0.59, 0.05)`` with ``a = 1`` clamps the position at ``0.6``, keeps ``0.051994722``
    (**only** the left wall zeroes the velocity) and reports ``done``; ``(-0.5, 0.0695)``
    and ``(0, -0.0695)`` land on exactly ``+/-max_speed`` and integrate the *clipped*
    velocity (``-0.5 + 0.07 = -0.43``, not the ``-0.4292`` of the unclipped ``0.0708``).
    ``_terminated`` pins the goal as a conjunction: ``(0.45, 0)`` and ``(0.5, 0)`` are done,
    ``(0.4499, 0)`` and ``(0.5, -0.001)`` are not (rolling backwards), the wall is not.
    """
    mountaincar = problem.MountainCarContinuous(max_step=20)
    obs = np.float32([[-1.2, -0.05], [-0.5, -0.05], [0.59, 0.05], [-0.5, 0.0695], [0, -0.0695]])
    action = np.float32([[-1.0], [0.0], [1.0], [1.0], [-1.0]])

    def kernel(obs_value, action_value):
        return mountaincar._advance(obs_value, action_value, mountaincar._constants())

    next_obs, done = run_in_graph(kernel, obs, action)
    np.testing.assert_array_equal(
        next_obs,
        np.float32([
            [-1.2, 0.0],  # left wall: position clipped, velocity zeroed
            [-0.5501768589019775, -0.050176844000816345],  # free: velocity kept
            [0.6, 0.051994722336530685],  # right wall: clamped, velocity kept
            [-0.43, 0.07],  # positive velocity clip, then integrates
            [-0.07, -0.07],  # negative velocity clip
        ]),
    )
    np.testing.assert_array_equal(done, np.array([False, False, True, False, False]))
    assert float(next_obs[3, 1]) == float(np.float32(0.07))

    at_goal = np.float32(
        [[0.45, 0.0], [0.4499, 0.0], [0.5, -0.001], [0.5, 0.0], [0.6, 0.0], [-1.2, 0.0]]
    )
    terminated = run_in_graph(
        lambda obs_value: mountaincar._terminated(obs_value, mountaincar._constants()), at_goal
    )
    np.testing.assert_array_equal(terminated, np.array([True, False, False, True, True, False]))


# --------------------------------------------------------------------------- #
# one Euler step derived by hand from the published equations
# --------------------------------------------------------------------------- #


def _one_step_by_hand(
    position,
    velocity,
    action,
    *,
    min_position=-1.2,
    max_position=0.6,
    max_speed=0.07,
    goal_position=0.45,
    goal_velocity=0.0,
    power=0.0015,
    gravity=0.0025,
):
    """Advance one MountainCar state with Gym's equations, written out by hand.

    Independent of the kernel and of ``_mountaincar_rollout``: the coefficients
    are the published ``power``/``gravity``/``cos(3 * position)`` terms spelled as
    plain Python ``float`` arithmetic (float64), with the two clips, the left-wall
    velocity reset and the goal conjunction in the documented order, so a wrong
    coefficient or a reordered clip moves the expected value instead of moving
    both sides together.

    Args:
        position: Car position on the track.
        velocity: Car velocity in units/step.
        action: The **pre-clip** action ``tanh(logit)``.

    Returns:
        ``(next_position, next_velocity, done)``.
    """
    force = min(max(action, -1.0), 1.0)
    next_velocity = velocity + force * power - gravity * math.cos(3.0 * position)
    next_velocity = min(max(next_velocity, -max_speed), max_speed)
    next_position = min(max(position + next_velocity, min_position), max_position)
    if next_position == min_position and next_velocity < 0.0:
        next_velocity = 0.0
    done = next_position >= goal_position and next_velocity >= goal_velocity
    return next_position, next_velocity, done


@pytest.mark.parametrize(
    "position, velocity, action, expected",
    [
        # At a=tanh(1)=0.761594156, from (-0.5, 0):
        # drive = 0.761594156*0.0015 - 0.0025*cos(-1.5)
        #       = 0.001142391 - 0.000176843 = 0.000965548, so
        # position' = -0.5 + 0.000965548 = -0.499034452.
        (-0.5, 0.0, 0.7615941559557649, (-0.499034451770, 0.000965548229764, False)),
        # A push back down the slope from (-0.3, 0.06):
        # v' = 0.06 - 0.00075 - 0.0025*cos(-0.9)
        #    = 0.06 - 0.00075 - 0.001554025 = 0.057695975, pos' = -0.242304025.
        (-0.3, 0.06, -0.5, (-0.242304024921, 0.0576959750793, False)),
        # Just short of the goal, rolling forward at full force:
        # v' = 0.03 + 0.0015 - 0.0025*cos(1.32) = 0.030879561 (no clip), so
        # pos' = 0.470879561 >= 0.45 with v' >= 0 -> done.
        (0.44, 0.03, 1.0, (0.470879561371, 0.0308795613709, True)),
        # A gentle forward push while rolling backwards:
        # v' = -0.02 + 0.000375 - 0.0025*cos(-0.3) = -0.022013341, pos' = -0.122013341.
        (-0.1, -0.02, 0.25, (-0.122013341223, -0.0220133412228, False)),
    ],
    ids=["from_rest", "down_slope", "reaches_goal", "rolling_back"],
)
def test_one_step_matches_hand_derived_values_at_non_zero_position(
    problem, run_in_graph, position, velocity, action, expected
):
    """Invariant: ``_advance`` reproduces hand-computed transitions off the centre.

    Oracle: :func:`_one_step_by_hand`, Gym's transition transcribed in this module
    from the published equations, plus the numeric literals in the parametrization
    -- the two are asserted to agree with each other (float64) and the float32
    kernel with both (``rtol=1e-6``).

    Notes:
        The ``dtype``-parameterized rollout oracle runs one recurrence twice, so a
        wrong coefficient cancels out there; here it cannot. Every row has
        ``position != 0``, where ``cos(3 * position)`` is neither ``1`` nor
        degenerate, and the four rows cover a free step, a downhill push, a goal
        transition and a step with no clip at all.
    """
    assert position != 0.0, "cos(3 * position) is at its trivial value at the centre"

    next_position, next_velocity, done = _one_step_by_hand(position, velocity, action)
    np.testing.assert_allclose(
        [next_position, next_velocity], expected[:2], rtol=1e-10, atol=0
    )
    assert done is expected[2]

    mountaincar = problem.MountainCarContinuous(max_step=20)
    obs = np.array([[position, velocity]], dtype=np.float32)
    actions = np.array([[action]], dtype=np.float32)

    def kernel(values, action_value):
        return mountaincar._advance(values, action_value, mountaincar._constants())

    kernel_obs, kernel_done = run_in_graph(kernel, obs, actions)
    np.testing.assert_allclose(kernel_obs[0], [next_position, next_velocity], rtol=1e-6)
    assert bool(np.asarray(kernel_done)[0]) is done


# --------------------------------------------------------------------------- #
# the host-side report is finite-safe
# --------------------------------------------------------------------------- #


def test_show_reports_a_nan_poisoned_policy_instead_of_raising(
    state, problem, enp, fake_schedule, evoxc_key, capsys
):
    """Invariant: a NaN-poisoned rollout prints an honest report and never raises.

    Oracle: the documented rule ("a non-finite return (a NaN-poisoned policy)
    compares false, so the goal reads as ``n/a`` with the honest return value
    instead of claiming the goal was missed"). The NaN poisoning test above
    proves ``evaluate`` carries NaN into the return; this test drives the same
    poisoned ``act_fn`` through ``show`` -- a host-side call that builds its own
    one-off graph -- and asserts the report reaches stdout for every repeat plus
    the mean line.

    Notes:
        ``value > 0.0`` on a NaN return is ``False``, so a report that did not
        special-case non-finite values would claim "goal not reached" for a
        poisoned episode: a wrong statement, not just a missing one. The idle
        policy is the finite control and must still print ``goal not reached``
        with no ``nan`` in the report.
    """
    mountaincar = problem.MountainCarContinuous(max_step=10, repeat_times=2)
    schedule = fake_schedule(pop_size=1)

    returned = mountaincar.show(
        state, evoxc_key, _constant_logits(enp, float("nan")), schedule
    )
    report = capsys.readouterr().out

    assert returned is None
    assert report.count("MountainCarContinuous repeat") == 2
    assert report.count("return nan, goal n/a (NaN return)") == 2
    assert report.count("final state: position=nan, velocity=nan") == 2
    assert "MountainCarContinuous mean return: nan" in report
    assert "goal not reached" not in report

    healthy = mountaincar.show(state, evoxc_key, _constant_logits(enp, 0.0), schedule)
    finite = capsys.readouterr().out

    assert healthy is None
    assert finite.count("MountainCarContinuous repeat") == 2
    assert finite.count("goal not reached") == 2
    assert "nan" not in finite


# --------------------------------------------------------------------------- #
# configuration errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [{"max_step": 0}, {"max_step": -1}, {"max_step": True}, {"max_step": 2.5},
     {"repeat_times": 0}, {"repeat_times": False}, {"repeat_times": 2.0},
     {"max_speed": 0.0}, {"power": 0.0}, {"gravity": -0.1}, {"min_position": 0.6},
     {"goal_position": 0.7}, {"goal_position": -1.3}, {"goal_velocity": 0.08},
     {"fitness_target": 100.5}],
    ids=lambda kwargs: "=".join(str(item) for item in next(iter(kwargs.items()))),
)
def test_out_of_domain_configuration_raises_tensorneat_error(problem, common, kwargs):
    """Invariant: every value outside its documented domain raises ``TensorNEATError``.

    Oracle: the ``Raises:`` clause of the constructor docstring -- the horizon and the
    repeat count must be ints (a ``bool`` is rejected explicitly: an ``int``, never a
    horizon), ``max_speed``/``power`` strictly positive, ``gravity`` non-negative, the track
    of positive width, the goal on the clipped track and reachable at ``max_speed``, and
    ``fitness_target`` no higher than :data:`ATTAINABLE_CEILING` (``93.0``), the *attainable*
    ceiling rather than the nominal ``100`` bonus. One case per knob, and the message must
    name the offending knob.
    """
    with pytest.raises(common.TensorNEATError, match=next(iter(kwargs))):
        problem.MountainCarContinuous(**kwargs)


def test_in_domain_configuration_is_accepted(problem, common):
    """Invariant: Gym's defaults construct, are stored as given, and fix the arity.

    Oracle: the constructor signature and the class docstring's attribute table.
    ``gravity = 0.0`` is legal (``require_non_negative``), ``fitness_target =
    ATTAINABLE_CEILING`` (``93.0``) is the hard ceiling and is legal while ``None``
    disables the check and a negative value is stored verbatim, and the arity is two
    observations in and one action logit out.

    Notes:
        The ceiling is the *attainable* one, not the nominal ``100`` goal bonus: the
        bonus is paid once while the ``0.1 * a**2`` penalty is charged on every scoring
        step of the climb, so a reaching policy returns about ``89``-``91``. The
        boundary is asserted from both sides -- ``93.0`` accepted, ``93.5`` rejected --
        so a regression to the nominal ``100`` ceiling fails here rather than silently
        validating an unreachable target.
    """
    assert problem.rl.mountaincar.ATTAINABLE_CEILING == 93.0

    defaults = problem.MountainCarContinuous()
    assert (defaults.max_step, defaults.repeat_times) == (999, 1)
    assert (defaults.min_position, defaults.max_position) == (-1.2, 0.6)
    assert (defaults.goal_position, defaults.goal_velocity) == (0.45, 0.0)
    assert defaults.max_speed == pytest.approx(0.07) and defaults.fitness_target is None
    assert (defaults.power, defaults.gravity) == (pytest.approx(0.0015), pytest.approx(0.0025))
    assert (defaults.input_shape, defaults.output_shape) == ((2,), (1,))

    configured = problem.MountainCarContinuous(
        max_step=1, repeat_times=3, gravity=0.0, max_speed=0.05, goal_velocity=0.04,
        fitness_target=93.0,
    )
    assert (configured.max_step, configured.repeat_times) == (1, 3)
    assert (configured.gravity, configured.max_speed) == (0.0, 0.05)
    assert (configured.goal_velocity, configured.fitness_target) == (0.04, 93.0)
    assert problem.MountainCarContinuous(fitness_target=-0.5).fitness_target == -0.5

    with pytest.raises(common.TensorNEATError, match="attainable ceiling"):
        problem.MountainCarContinuous(fitness_target=93.5)
