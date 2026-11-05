"""Unit tests for the batched CartPole rollout of ``problem/rl/cartpole.py``.

What this module proves
-----------------------
``CartPole`` is a fixed-horizon, batched Euler rollout written in evoxc ops: one
``evoxc.scan`` over ``max_step`` timesteps, carry ``(obs (B, 4), alive (B,),
poisoned (B,))`` with ``B = P * repeat_times`` in individual-major order, reward
``alive`` **before** the transition (the failing step still scores, as in Gym),
death on ``|x| > 2.4`` or ``|theta| > 12 deg``, and the state *frozen* once dead
so an episode's return is exactly its survival time. Fitness is the mean of the
``repeat_times`` returns, then ``finalize_fitness``.

The legacy RL problems were deleted, so there is no parity oracle for this
environment: :func:`support.oracle.cartpole_rollout` -- an independent numpy
transcription of the same documented recurrence -- **is** the oracle, and the
closed forms below (the push-right survival count, the return range) are the
second opinion that keeps the oracle itself honest.

Two properties make the whole suite reproducible without knowing the rollout's
random draw:

* ``reset_scale = 0.0`` is a legal configuration (the validator only rejects a
  *negative* scale) and makes every initial state exactly ``(0, 0, 0, 0)``, so
  the initial states the graph draws are known host-side and the numpy oracle
  can be fed the same ``obs0``. With the default ``reset_scale`` the draw comes
  from ``evoxc.random`` and is *not* reproducible on the host -- tests that use
  it assert relations between runs (determinism, divergence, averaging) instead
  of absolute values.
* ``B = P * repeat_times`` depends only on the product, so ``P = 4, R = 1`` and
  ``P = 2, R = 2`` describe the same four episodes. With ``reset_scale = 0.0``
  every episode starts at rest, so a row-dependent policy alone fixes each
  row's return; that is what pins the individual-major layout.

Gotchas this module encodes
---------------------------
* Both constant policies ("always push right", "always push left") survive
  exactly **9** steps from rest, because the dynamics are symmetric under
  ``(x, theta, force) -> (-x, -theta, -force)``. A test that means to
  distinguish two policies must therefore differ in more than the sign: the
  layout test's second policy is a stabilising ``2 * theta + 3 * theta_dot > 0``
  controller, which survives the whole horizon.
* ``theta_threshold`` is a **degrees** value on the host (``12.0``) and radians
  in the trace (``0.2094``); :func:`test_death_is_a_strict_inequality_at_the_documented_limits`
  checks the radians side.
* A NaN logit still acts as "push left" *inside* the rollout (``enp.greater`` is
  false against NaN), but the scan carries a poison flag, so **any** individual
  whose policy emitted a NaN at any step gets a NaN fitness, which
  ``finalize_fitness`` maps to ``-inf`` -- see
  :func:`test_nan_logits_poison_the_fitness_to_negative_infinity`.
* A constant-folding trap: the tie (equal logits) must push **left**, matching
  ``argmax`` index 0, so the tie test asserts the always-left force.

``max_step`` stays at or below 50 everywhere (the numpy interpreter is
interpreter-bound, not batch-bound), the backend is always ``"numpy"``, and no
``jax``/``torch``, network or GPU is involved.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# genome-free policies: plain ``enp`` closures, inlined into the trace
# --------------------------------------------------------------------------- #

#: The documented limit in radians, as the trace sees it (float32 graph constant).
THETA_LIMIT = float(np.deg2rad(12.0))
#: The documented cart-position limit.
X_LIMIT = 2.4
#: Survival count of a constant push-right (or push-left) policy from rest,
#: derived below in :func:`test_a_constant_push_right_policy_survives_nine_steps`.
CONSTANT_POLICY_SURVIVAL = 9
#: Horizon the stabilising controller below survives in full (it never dies).
STABILISED_SURVIVAL = 20


def _constant_logits(enp, left, right):
    """A policy emitting the same two logits for every row.

    Args:
        enp: The ``evoxc.numpy`` op namespace.
        left: Value of logit 0; the cart is pushed right iff ``right > left``
            (a tie pushes left).
        right: Value of logit 1.

    Returns:
        ``act_fn(batch) -> (B, 2) float32``.
    """

    def act_fn(batch):
        n = batch.shape[0]
        return enp.stack([enp.full((n,), left), enp.full((n,), right)], axis=1)

    return act_fn


def _split_row_policy(enp, split_rows):
    """Rows before ``split_rows`` push right constantly; the rest stabilise the pole.

    Args:
        enp: The ``evoxc.numpy`` op namespace.
        split_rows: Row index at which the policy switches.

    Returns:
        ``act_fn(batch) -> (B, 2) float32``. Rows ``< split_rows`` emit
        ``(0, 1)`` -- the constant push-right policy, which dies after
        :data:`CONSTANT_POLICY_SURVIVAL` steps. The remaining rows emit ``(0, 1)``
        when ``2 * theta + 3 * theta_dot > 0`` and ``(1, 0)`` otherwise, i.e. a
        state-dependent controller that balances the pole for the whole horizon
        (:data:`STABILISED_SURVIVAL` steps at ``max_step = 20``).

    Notes:
        The row index is the tiled episode index, so this policy is how a test
        observes the ``B = P * repeat_times`` layout without relying on the
        random draw (see :func:`test_repeat_times_averages_individual_major_rows`).
        From rest both branches of the controller would be needed to keep the
        pole upright, so the two row families really do return different
        survival times -- a sign-flipped constant policy would return ``9`` for
        both (see the module docstring).
    """

    def act_fn(batch):
        n = batch.shape[0]
        head = enp.less(enp.arange(n), split_rows)
        cols = enp.split(batch, 4, axis=1)
        theta = enp.reshape(cols[2], (n,))
        theta_dot = enp.reshape(cols[3], (n,))
        balanced = enp.greater(2.0 * theta + 3.0 * theta_dot, 0.0)
        push_right = enp.logical_or(head, balanced)
        return enp.stack(
            [
                enp.where(push_right, enp.full((n,), 0.0), enp.full((n,), 1.0)),
                enp.where(push_right, enp.full((n,), 1.0), enp.full((n,), 0.0)),
            ],
            axis=1,
        )

    return act_fn


def _leading_rows_nan_policy(enp, split_rows):
    """Rows before ``split_rows`` emit a NaN logit; the rest push right.

    Args:
        enp: The ``evoxc.numpy`` op namespace.
        split_rows: Row index at which the NaN policy stops.

    Returns:
        ``act_fn(batch) -> (B, 2) float32``. Rows ``< split_rows`` return
        ``(NaN, 1.0)`` (a poisoned but otherwise well-formed logit pair); the
        rest return ``(0, 1)``, the constant push-right policy.
    """

    def act_fn(batch):
        n = batch.shape[0]
        poisoned = enp.less(enp.arange(n), split_rows)
        left = enp.where(
            poisoned, enp.full((n,), float("nan")), enp.full((n,), 0.0)
        )
        return enp.stack([left, enp.full((n,), 1.0)], axis=1)

    return act_fn


def _nan_after_tilt_policy(enp, tilt):
    """A push-right policy that starts emitting NaN once ``|theta| > tilt``.

    Args:
        enp: The ``evoxc.numpy`` op namespace.
        tilt: Absolute pole-angle threshold (radians) after which the logits
            become NaN.

    Returns:
        ``act_fn(batch) -> (B, 2) float32``: ``(0, 1)`` while the pole is
        near-upright, ``(NaN, NaN)`` afterwards.

    Notes:
        This is the "a NaN emitted at *any* step poisons the episode" case: the
        first steps of the rollout are perfectly finite, so a poison flag that
        only inspected the last step (or that were reset in the carry) would let
        the episode score normally.
    """

    def act_fn(batch):
        n = batch.shape[0]
        theta = enp.reshape(enp.split(batch, 4, axis=1)[2], (n,))
        late = enp.greater(enp.abs(theta), tilt)
        nan = enp.full((n,), float("nan"))
        zero = enp.full((n,), 0.0)
        one = enp.full((n,), 1.0)
        return enp.stack(
            [enp.where(late, nan, zero), enp.where(late, nan, one)], axis=1
        )

    return act_fn


def _constant_actions(rows, max_step, push_right):
    """The oracle's ``(B, max_step)`` action matrix for a constant policy.

    Args:
        rows: Number of episodes ``B``.
        max_step: Horizon ``T``.
        push_right: ``True`` for ``1`` (``+force_mag``), ``False`` for ``0``.

    Returns:
        ``(B, T)`` int32 actions.
    """
    return np.full((rows, max_step), 1 if push_right else 0, dtype=np.int32)


# --------------------------------------------------------------------------- #
# agreement with the numpy oracle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("max_step", [20, 50])
@pytest.mark.parametrize("direction", [1, 0], ids=["push_right", "push_left"])
def test_a_constant_policy_matches_the_numpy_oracle(
    state, problem, oracle, enp, run_evaluate, fake_schedule, max_step, direction
):
    """Invariant: a constant policy reproduces ``support.oracle.cartpole_rollout`` exactly.

    Oracle: ``cartpole_rollout`` with the same actions -- ``actions = direction``
    everywhere, ``obs0 = zeros`` -- and ``max_step`` identical, on a population
    of two individuals and ``repeat_times = 1``. ``reset_scale = 0.0`` is what
    makes the comparison possible: the graph's initial states are then exactly
    zero, so the host oracle and the traced rollout start from the same state.
    The closed form (the crash step of a constant policy, ``9``) is asserted
    alongside, so a shared slip between the kernel and the oracle shows up.

    Notes:
        The two horizons are the point of the parametrization: a fixed policy's
        return must not change once it exceeds the death step (see
        :func:`test_an_episode_that_dies_at_step_k_never_scores_more_than_k`).
        ``direction = 0`` (push left) is compared with a zero action matrix, so
        the ``out[1] > out[0]`` decision rule is exercised in both senses.
    """
    cartpole = problem.CartPole(max_step=max_step, reset_scale=0.0)
    act_fn = _constant_logits(
        enp,
        0.0 if direction == 1 else 1.0,
        1.0 if direction == 1 else 0.0,
    )
    fitness = run_evaluate(cartpole, state, act_fn, fake_schedule(2), key_seed=0)

    expected = oracle.cartpole_rollout(
        _constant_actions(2, max_step, direction == 1),
        np.zeros((2, 4), dtype=np.float32),
        max_step=max_step,
        params=oracle.cartpole_params(),
    )

    assert fitness.shape == (2,)
    assert fitness.dtype == np.float32
    np.testing.assert_allclose(fitness, expected, rtol=1e-5)
    np.testing.assert_array_equal(
        fitness, np.full((2,), CONSTANT_POLICY_SURVIVAL, dtype=np.float32)
    )


def test_a_constant_push_right_policy_survives_nine_steps(
    state, problem, oracle, enp, run_evaluate, fake_schedule
):
    """Invariant: from rest, "always push right" returns exactly ``9`` for any horizon >= 9.

    Oracle: hand arithmetic on the documented Euler recurrence, with
    ``force_mag = 10``, ``total_mass = 1.1``, ``polemass_length = 0.05``,
    ``length = 0.5`` and ``dt = 0.02``. The first step only fills the
    velocities (``theta`` and ``x`` start at rest), after which the pole angle
    grows quadratically:

    ``temp = 10 / 1.1 = 9.0909``, ``thetaacc = -9.0909 / (0.5 * (4/3 - 0.1/1.1))
    = -14.634``, hence ``theta_k ~= -14.634 * dt**2 * k * (k - 1) / 2`` --
    ``-0.164 rad`` at ``k = 8`` and ``-0.211 rad`` at ``k = 9``. The ``12 deg``
    (``0.2094 rad``) limit is first crossed by the ninth transition, so the
    episode is rewarded nine times and returns ``9``.

    Notes:
        ``max_step = 8`` is the witness that the crash really happens at step
        ``9`` and not earlier: a horizon below the death step returns the
        horizon. The same ``9`` is asserted for ``max_step`` 9, 20 and 50 (a
        longer horizon cannot add reward to a dead episode), and the numpy
        oracle is cross-checked on every configuration so a slip in this
        arithmetic shows up as a disagreement between two references.
    """
    horizons = (8, 9, 20, 50)
    returns = {}
    for max_step in horizons:
        cartpole = problem.CartPole(max_step=max_step, reset_scale=0.0)
        fitness = run_evaluate(
            cartpole,
            state,
            _constant_logits(enp, 0.0, 1.0),
            fake_schedule(2),
            key_seed=0,
        )
        expected = oracle.cartpole_rollout(
            _constant_actions(2, max_step, True),
            np.zeros((2, 4), dtype=np.float32),
            max_step=max_step,
            params=oracle.cartpole_params(),
        )
        np.testing.assert_allclose(fitness, expected, rtol=1e-5)
        returns[max_step] = float(fitness[0])

    assert returns[8] == 8  # still alive through the eighth transition
    assert returns[9] == CONSTANT_POLICY_SURVIVAL
    assert returns[20] == CONSTANT_POLICY_SURVIVAL
    assert returns[50] == CONSTANT_POLICY_SURVIVAL


def test_the_two_logits_select_a_signed_force_and_a_tie_pushes_left(
    state, problem, run_in_graph
):
    """Invariant: ``out[1] > out[0]`` is ``+force_mag``, everything else (ties included) ``-force_mag``.

    Oracle: closed form on ``CartPole._force_from_logits(logits,
    CartPole._constants())``, the documented decision rule. ``(0, 1)`` must give
    ``+force_mag``; ``(1, 0)``, the tie ``(0, 0)``, the negated tie ``(-1, -1)``
    and a tie at a third value ``(0.25, 0.25)`` must give ``-force_mag`` -- a tie
    pushes left, matching ``argmax`` index 0.

    Notes:
        The decision is checked on the force and not through ``evaluate``,
        because from a symmetric start the two constant policies return the same
        survival time (see the module docstring): the *force sign* is the only
        place the rule is observable in isolation. ``_constants`` must be built
        inside the trace, which is why the closure calls it per invocation.
    """
    cartpole = problem.CartPole(max_step=20, reset_scale=0.0)
    logits = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [-1.0, -1.0], [0.25, 0.25]],
        dtype=np.float32,
    )

    def force(logits_value):
        return cartpole._force_from_logits(logits_value, cartpole._constants())

    forces = run_in_graph(force, logits)

    assert forces.shape == (5, 1)
    np.testing.assert_array_equal(
        forces[:, 0],
        np.array([10.0, -10.0, -10.0, -10.0, -10.0], dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# termination and the frozen state
# --------------------------------------------------------------------------- #


def test_death_is_a_strict_inequality_at_the_documented_limits(
    state, problem, run_in_graph
):
    """Invariant: ``|x| > 2.4`` and ``|theta| > 12 deg``, strictly.

    Oracle: closed form, checked on hand-built observations through
    ``CartPole._terminated(obs, CartPole._constants())`` -- the only way to place
    an episode at an exact state, since ``evaluate`` draws its own ``obs0``.
    A value *equal* to a limit is alive; one ``nextafter`` above it is dead, in
    both signs; and ``theta = 0.3 rad`` is dead, which is what proves the
    constant is in radians (``0.3 < 12`` would pass a degrees comparison).

    Notes:
        ``theta_threshold`` is given in **degrees** on the host and converted
        before it enters the trace, so the comparison value is
        ``float32(deg2rad(12.0)) = 0.20944``. The boundary pair is expressed with
        ``np.nextafter`` in ``float32`` so "equal is alive" is exact rather than
        approximate.
    """
    cartpole = problem.CartPole(max_step=20, reset_scale=0.0)
    theta_limit = np.float32(np.deg2rad(cartpole.theta_threshold))
    x_limit = np.float32(cartpole.x_threshold)
    assert float(theta_limit) == pytest.approx(THETA_LIMIT, rel=0, abs=1e-7)
    assert float(x_limit) == pytest.approx(X_LIMIT)

    theta_above = np.nextafter(theta_limit, np.float32(np.inf))
    theta_below = np.nextafter(theta_limit, np.float32(0.0))
    x_above = np.nextafter(x_limit, np.float32(np.inf))
    obs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # rest: alive
            [x_limit, 0.0, 0.0, 0.0],  # exactly at the limit: alive
            [x_above, 0.0, 0.0, 0.0],  # one float32 above: dead
            [-x_above, 0.0, 0.0, 0.0],  # ... in the other sign: dead
            [0.0, 0.0, theta_limit, 0.0],  # exactly at the limit: alive
            [0.0, 0.0, -theta_limit, 0.0],  # ... other sign: alive
            [0.0, 0.0, theta_above, 0.0],  # one float32 above: dead
            [0.0, 0.0, -theta_above, 0.0],  # ... other sign: dead
            [0.0, 0.0, theta_below, 0.0],  # still inside: alive
            [0.0, 0.0, 0.3, 0.0],  # 0.3 rad = 17.2 deg is dead; 0.3 < 12 is not
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [False, False, True, True, False, False, True, True, False, True]
    )

    def terminated(obs_value):
        return cartpole._terminated(obs_value, cartpole._constants())

    dead = run_in_graph(terminated, obs)

    assert dead.shape == (obs.shape[0],)
    np.testing.assert_array_equal(dead, expected)


def test_an_episode_that_dies_at_step_k_never_scores_more_than_k(
    state, problem, oracle, enp, run_evaluate, fake_schedule
):
    """Invariant: a frozen episode's return is its survival time, independent of the horizon.

    Oracle: closed form + the numpy oracle. The constant push-right policy from
    rest dies after :data:`CONSTANT_POLICY_SURVIVAL` steps, so its fitness is
    ``9`` for ``max_step = 9`` (the horizon ends first), ``20`` and ``50`` --
    identical in all three cases, and strictly below the two longer horizons.

    Notes:
        This is the observable consequence of freezing the state with
        ``enp.where(alive, next_obs, obs)``: nothing after the death step scores,
        which is why ``max_step`` is a hard fitness ceiling.
    """
    horizon_returns = {}
    for max_step in (9, 20, 50):
        cartpole = problem.CartPole(max_step=max_step, reset_scale=0.0)
        fitness = run_evaluate(
            cartpole,
            state,
            _constant_logits(enp, 0.0, 1.0),
            fake_schedule(2),
            key_seed=0,
        )
        expected = oracle.cartpole_rollout(
            _constant_actions(2, max_step, True),
            np.zeros((2, 4), dtype=np.float32),
            max_step=max_step,
            params=oracle.cartpole_params(),
        )
        np.testing.assert_allclose(fitness, expected, rtol=1e-5)
        np.testing.assert_array_equal(
            fitness, np.full((2,), CONSTANT_POLICY_SURVIVAL, dtype=np.float32)
        )
        horizon_returns[max_step] = float(fitness[0])

    assert horizon_returns[9] == horizon_returns[20] == horizon_returns[50]
    assert horizon_returns[20] < 20
    assert horizon_returns[50] < 50


# --------------------------------------------------------------------------- #
# repeat_times: the (P, R) layout and the averaging
# --------------------------------------------------------------------------- #


def test_repeat_times_averages_individual_major_rows(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: ``B = P * R`` is individual-major and the fitness is the mean over ``R``.

    Oracle: an identity between two runs, not a number. ``P = 4, R = 1`` exposes
    the four per-row returns; ``P = 2, R = 2`` describes *the same* four episodes
    in the same row order, so it must produce the individual-major means
    ``[(r0 + r1) / 2, (r2 + r3) / 2]`` -- a repeat-major layout would instead
    pair ``(r0, r2)`` and ``(r1, r3)``.

    Notes:
        ``reset_scale = 0.0`` makes every row start at rest, so the
        row-dependent :func:`_split_row_policy` alone fixes each return: rows
        ``0, 1`` push right constantly and crash at
        :data:`CONSTANT_POLICY_SURVIVAL`, rows ``2, 3`` balance the pole for the
        whole :data:`STABILISED_SURVIVAL`-step horizon. The two families return
        different values (asserted first, so the layout check cannot pass
        vacuously), which is exactly what distinguishes the two pairings. The
        *mean* (rather than the sum) is pinned by also asserting the result
        differs from the row sums.
    """
    max_step = 20
    single = problem.CartPole(max_step=max_step, repeat_times=1, reset_scale=0.0)
    doubled = problem.CartPole(max_step=max_step, repeat_times=2, reset_scale=0.0)
    per_row = run_evaluate(
        single, state, _split_row_policy(enp, 2), fake_schedule(4), key_seed=0
    )
    paired = run_evaluate(
        doubled, state, _split_row_policy(enp, 2), fake_schedule(2), key_seed=0
    )

    np.testing.assert_array_equal(
        per_row,
        np.array(
            [CONSTANT_POLICY_SURVIVAL] * 2 + [STABILISED_SURVIVAL] * 2,
            dtype=np.float32,
        ),
    )
    assert per_row[1] != per_row[2]  # otherwise the parities coincide

    individual_major = np.array(
        [(per_row[0] + per_row[1]) / 2, (per_row[2] + per_row[3]) / 2],
        dtype=np.float32,
    )
    repeat_major = np.array(
        [(per_row[0] + per_row[2]) / 2, (per_row[1] + per_row[3]) / 2],
        dtype=np.float32,
    )
    summed = np.array(
        [per_row[0] + per_row[1], per_row[2] + per_row[3]], dtype=np.float32
    )

    assert paired.shape == (2,)
    assert not np.allclose(repeat_major, individual_major)
    np.testing.assert_allclose(paired, individual_major)
    assert not np.allclose(paired, repeat_major)
    assert not np.allclose(paired, summed)


# --------------------------------------------------------------------------- #
# randomness: determinism, divergence and the fitness range
# --------------------------------------------------------------------------- #


def test_the_same_key_reproduces_the_run_and_another_key_diverges(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: the rollout is a pure function of ``(key, config, shape)``.

    Oracle: run-to-run comparison. The same ``key_seed`` twice must give
    bit-identical fitness for a state-dependent policy; across five distinct
    seeds the fitness vectors must not all be equal (asserted as "more than one
    distinct value", which cannot fail by coincidence the way a single pair of
    seeds could).

    Notes:
        The default ``reset_scale`` is used on purpose: the draw is one
        ``evoxc.random.uniform(key, (B, 4), low=-reset_scale, high=reset_scale)``,
        never ``np.random``, so reproducibility is a property of the key alone --
        this is the test that would catch a reintroduced global RNG or a key
        threaded through the scan.
    """
    cartpole = problem.CartPole(max_step=20)
    schedule = fake_schedule(4)
    policy = _split_row_policy(enp, 2)

    first = run_evaluate(cartpole, state, policy, schedule, key_seed=7)
    again = run_evaluate(cartpole, state, policy, schedule, key_seed=7)
    np.testing.assert_array_equal(first, again)

    runs = [
        run_evaluate(cartpole, state, policy, schedule, key_seed=key_seed)
        for key_seed in range(5)
    ]
    for run in runs:
        assert run.shape == (4,)
        assert np.all(np.isfinite(run))
    distinct = {tuple(float(value) for value in run) for run in runs}
    assert len(distinct) > 1


def test_the_return_is_an_integer_between_zero_and_max_step(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: ``0 <= fitness <= max_step`` and integral, for any key.

    Oracle: the documented algebra -- the reward is the ``alive`` mask, so a
    return counts steps and is an integer; the state freezes after death, so it
    can never exceed the horizon. Checked for several keys and for both a
    constant and a state-dependent policy; the closed form (the constant policy
    from rest returns ``9``) is asserted on a ``reset_scale = 0.0`` instance,
    since the default reset draw is not reproducible host-side.

    Notes:
        This is also why ``CartPole`` documents that ``fitness_target`` must be
        ``<= max_step``: a target above the ceiling is unreachable and the run
        would hit ``generation_limit`` instead of converging. The constructor
        rejects such a target (asserted in
        :func:`test_out_of_domain_configuration_raises_tensorneat_error`), and
        this test pins the range that rejection rests on.
    """
    max_step = 20
    cartpole = problem.CartPole(max_step=max_step)
    schedule = fake_schedule(4)
    policies = {
        "push_right": _constant_logits(enp, 0.0, 1.0),
        "stabilising": _split_row_policy(enp, 2),
    }

    for name, policy in policies.items():
        for key_seed in range(3):
            fitness = run_evaluate(cartpole, state, policy, schedule, key_seed=key_seed)
            assert fitness.shape == (4,), name
            assert fitness.dtype == np.float32, name
            assert np.all(np.isfinite(fitness)), name
            assert np.all(fitness >= 0.0), name
            assert np.all(fitness <= float(max_step)), name
            np.testing.assert_array_equal(np.round(fitness), fitness, err_msg=name)

    from_rest = problem.CartPole(max_step=max_step, reset_scale=0.0)
    constant = run_evaluate(
        from_rest, state, _constant_logits(enp, 0.0, 1.0), schedule, key_seed=0
    )
    np.testing.assert_array_equal(
        constant, np.full((4,), CONSTANT_POLICY_SURVIVAL, dtype=np.float32)
    )


def test_nan_logits_poison_the_fitness_to_negative_infinity(
    state, problem, enp, run_in_graph, run_evaluate, fake_schedule
):
    """Invariant: a NaN logit acts as "push left" but poisons the fitness to ``-inf``.

    Oracle: closed form at the decision site plus ``finalize_fitness``'s NaN to
    ``-inf`` mapping. ``enp.greater(nan, x)`` is ``False``, so
    ``_force_from_logits`` must still return ``-force_mag`` -- the rollout keeps
    running with the documented "push left" branch -- while ``_nan_rows`` feeds
    the scan's poison flag, and **any** NaN at **any** step turns that
    individual's fitness into ``NaN``, which the last op of ``evaluate`` maps to
    ``-inf``.

    Notes:
        This is the contract of ``src/tensorneat/CONTEXT.md`` reconciliation #7
        ("``CartPole`` ... demotes NaN logits to ``-inf`` fitness") and of
        ``CartPole.evaluate``/``_nan_rows``/``_scan_body``. Three cases pin it
        independently: a permanently NaN policy, a policy that only becomes NaN
        after the pole has tilted (so the poison must be carried across steps),
        and a row-dependent policy where only the poisoned rows are demoted. The
        finite control policy is asserted in the same test, so ``-inf`` cannot
        come from a rollout that failed for an unrelated reason.
    """
    nan = float("nan")

    # 1. The decision site still reads a NaN as "push left".
    cartpole = problem.CartPole(max_step=20, reset_scale=0.0)

    def force(logits_value):
        return cartpole._force_from_logits(logits_value, cartpole._constants())

    forces = run_in_graph(
        force,
        np.array([[nan, nan], [0.0, nan], [nan, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        forces[:, 0], np.full((3,), -10.0, dtype=np.float32)
    )

    # 2. A permanently NaN policy is demoted for every individual ...
    poisoned = run_evaluate(
        cartpole,
        state,
        _constant_logits(enp, nan, nan),
        fake_schedule(2),
        key_seed=0,
    )
    assert poisoned.shape == (2,)
    assert np.all(np.isneginf(poisoned))

    # ... and so is one that only turns NaN after the pole has tilted, which is
    # what proves the poison flag is carried across scan steps.
    late = run_evaluate(
        cartpole,
        state,
        _nan_after_tilt_policy(enp, 0.05),
        fake_schedule(2),
        key_seed=0,
    )
    assert np.all(np.isneginf(late))

    # Control: a finite policy on the same configuration scores normally.
    control = run_evaluate(
        cartpole,
        state,
        _constant_logits(enp, 0.0, 1.0),
        fake_schedule(2),
        key_seed=0,
    )
    assert np.all(np.isfinite(control))
    np.testing.assert_array_equal(
        control, np.full((2,), CONSTANT_POLICY_SURVIVAL, dtype=np.float32)
    )

    # 3. Poisoning is per individual: only the rows that emitted a NaN.
    mixed = run_evaluate(
        cartpole,
        state,
        _leading_rows_nan_policy(enp, 2),
        fake_schedule(4),
        key_seed=0,
    )
    np.testing.assert_array_equal(
        mixed,
        np.array(
            [-np.inf, -np.inf, CONSTANT_POLICY_SURVIVAL, CONSTANT_POLICY_SURVIVAL],
            dtype=np.float32,
        ),
    )

    # ... and one poisoned repeat demotes the whole individual (_any_repeat).
    doubled = problem.CartPole(max_step=20, repeat_times=2, reset_scale=0.0)
    one_repeat = run_evaluate(
        doubled,
        state,
        _leading_rows_nan_policy(enp, 1),
        fake_schedule(2),
        key_seed=0,
    )
    np.testing.assert_array_equal(
        one_repeat,
        np.array([-np.inf, CONSTANT_POLICY_SURVIVAL], dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# the host-side report is finite-safe
# --------------------------------------------------------------------------- #


def test_show_reports_a_nan_poisoned_policy_instead_of_raising(
    state, problem, enp, fake_schedule, evoxc_key, capsys
):
    """Invariant: a NaN-poisoned rollout prints an honest report and never raises.

    Oracle: the documented rule ("a NaN-poisoned policy is reported, not crashed
    on: ... a non-finite return has no integer survival length, so the step count
    is printed as ``n/a`` with the honest return value"). The poisoning test above
    already proves ``evaluate`` carries NaN into the return; this test drives the
    same poisoned ``act_fn`` through ``show`` -- whose one-off graph is built
    outside any trace -- and asserts the report reaches stdout for every repeat
    plus the mean line.

    Notes:
        The failure this guards is concrete: ``int(round(nan))`` raises
        ``ValueError``, so a report that formatted the survival count
        unconditionally would crash on exactly the individual a user most wants
        to inspect. The finite control run in the same test pins the other
        branch, so ``n/a`` cannot be printed for a healthy policy either.
    """
    cartpole = problem.CartPole(max_step=10, repeat_times=2, reset_scale=0.0)
    schedule = fake_schedule(pop_size=1)

    returned = cartpole.show(
        state, evoxc_key, _constant_logits(enp, float("nan"), float("nan")), schedule
    )
    report = capsys.readouterr().out

    assert returned is None
    assert report.count("CartPole repeat") == 2
    assert report.count("survived n/a (NaN return)") == 2
    assert "return nan" in report
    assert "CartPole mean return: nan" in report
    assert "survived 9/10 steps" not in report

    healthy = cartpole.show(state, evoxc_key, _constant_logits(enp, 0.0, 1.0), schedule)
    finite = capsys.readouterr().out

    assert healthy is None
    assert finite.count(f"survived {CONSTANT_POLICY_SURVIVAL}/10 steps") == 2
    assert "n/a" not in finite


# --------------------------------------------------------------------------- #
# configuration errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"max_step": 0}, id="max_step_zero"),
        pytest.param({"max_step": -1}, id="max_step_negative"),
        pytest.param({"repeat_times": 0}, id="repeat_times_zero"),
        pytest.param({"gravity": 0.0}, id="gravity_zero"),
        pytest.param({"masscart": -1.0}, id="masscart_negative"),
        pytest.param({"masspole": 0.0}, id="masspole_zero"),
        pytest.param({"length": 0.0}, id="length_zero"),
        pytest.param({"force_mag": 0.0}, id="force_mag_zero"),
        pytest.param({"dt": 0.0}, id="dt_zero"),
        pytest.param({"theta_threshold": 0.0}, id="theta_threshold_zero"),
        pytest.param({"x_threshold": -0.1}, id="x_threshold_negative"),
        pytest.param({"reset_scale": -0.1}, id="reset_scale_negative"),
        pytest.param(
            {"max_step": 20, "fitness_target": 21},
            id="fitness_target_above_max_step",
        ),
    ],
)
def test_out_of_domain_configuration_raises_tensorneat_error(problem, common, kwargs):
    """Invariant: every value outside its documented domain raises ``TensorNEATError``.

    Oracle: the ``Raises:`` clause of the constructor docstring -- ``max_step``
    and ``repeat_times`` must be positive; every mass, ``length``, ``dt`` and
    ``force_mag`` must be strictly positive; ``theta_threshold`` must be
    positive; ``x_threshold`` and ``reset_scale`` must be non-negative; and
    ``fitness_target`` may not exceed ``max_step`` (the hard fitness ceiling,
    since a frozen episode's return is its survival time). One case per knob, so
    a partially-validating constructor is caught.

    Notes:
        ``reset_scale = 0.0`` is explicitly *not* an error (the suite relies on
        it), which is why that knob's negative case is the one asserted. The
        error message must name the offending knob, so a constructor that
        rejects the wrong field is caught too.
    """
    with pytest.raises(common.TensorNEATError) as excinfo:
        problem.CartPole(**kwargs)
    offending = "fitness_target" if "fitness_target" in kwargs else next(iter(kwargs))
    assert offending in str(excinfo.value)


def test_in_domain_configuration_is_accepted(problem):
    """Invariant: the documented boundary values construct and are stored as given.

    Oracle: the constructor docstring and the attributes section of the class.
    ``reset_scale = 0.0``, ``x_threshold = 0.0`` and ``fitness_target`` exactly
    equal to ``max_step`` are in domain; ``fitness_target = None`` disables the
    ceiling check. The per-sample arity is the other half of the contract: ``4``
    observations in, ``2`` logits out.
    """
    cartpole = problem.CartPole(
        max_step=20,
        repeat_times=2,
        x_threshold=0.0,
        reset_scale=0.0,
        fitness_target=20,
    )
    assert cartpole.max_step == 20
    assert cartpole.repeat_times == 2
    assert cartpole.x_threshold == 0.0
    assert cartpole.reset_scale == 0.0
    assert cartpole.fitness_target == 20
    assert cartpole.theta_threshold == pytest.approx(12.0)
    assert problem.CartPole().fitness_target is None
    assert cartpole.input_shape == (4,)
    assert cartpole.output_shape == (2,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"max_step": 1.5},
            "CartPole: max_step must be an int, got 1.5",
            id="non_int_names_type_and_value",
        ),
        pytest.param(
            {"max_step": True},
            "CartPole: max_step must be an int, got True",
            id="bool_is_not_a_valid_int",
        ),
        pytest.param(
            {"max_step": 0},
            "CartPole: max_step must be positive, got 0",
            id="non_positive_int_names_the_value",
        ),
        pytest.param(
            {"gravity": 0.0},
            "CartPole: gravity must be positive, got 0.0",
            id="non_positive_float_names_the_value",
        ),
        pytest.param(
            {"x_threshold": -0.1},
            "CartPole: x_threshold must be non-negative, got -0.1",
            id="negative_float_names_the_value",
        ),
    ],
)
def test_configuration_errors_keep_the_exact_owner_prefixed_message(
    problem, common, kwargs, message
):
    """Invariant: each rejected knob raises ``<Owner>: <name> must be ...`` verbatim.

    Oracle: the constructor's ``Raises:`` clause plus the shared messages of
    ``problem.rl._common`` -- the three validators live once in that module, so
    the only thing ``CartPole`` still owns is the ``CartPole:`` owner prefix and
    the per-knob wording. The comparison is exact because the pinned text is the
    visible half of the shared helper's contract: a drifted prefix, a swapped
    knob name or a ``got`` value formatted differently would all be silent
    otherwise. ``True`` is in the list because it is an ``int`` in Python but
    never a meaningful horizon.
    """
    with pytest.raises(common.TensorNEATError) as excinfo:
        problem.CartPole(**kwargs)
    assert str(excinfo.value) == message
