"""Unit tests for the batched CartPole rollout of ``problem/rl/cartpole.py``.

What this module proves
-----------------------
``CartPole`` is a fixed-horizon, batched Euler rollout written in evoxc ops: one
``evoxc.scan`` over ``max_step`` timesteps, carry ``(obs (B, 4), alive (B,))``
with ``B = P * repeat_times`` in individual-major order, reward ``alive``
**before** the transition (the failing step still scores, as in Gym), death on
``|x| > 2.4`` or ``|theta| > 12 deg``, and the state *frozen* once dead so an
episode's return is exactly its survival time. Fitness is the mean of the
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
  ``P = 2, R = 2`` draw the **same four** initial states from the same key. That
  is what pins the individual-major layout: the paired run must average exactly
  the two rows of each individual.

Gotchas this module encodes
---------------------------
* Both constant policies ("always push right", "always push left") survive
  exactly **9** steps from rest, because the dynamics are symmetric under
  ``(x, theta, force) -> (-x, -theta, -force)``. A test that means to
  distinguish two policies must therefore differ in more than the sign.
* ``theta_threshold`` is a **degrees** value on the host (``12.0``) and radians
  in the trace (``0.2094``); :func:`test_death_is_a_strict_inequality_at_the_documented_limits`
  checks the radians side.
* ``enp.greater(NaN, NaN)`` is ``False``, so a NaN-logit policy acts as
  "push left" rather than poisoning the rollout -- see
  :func:`test_nan_logits_fall_back_to_push_left` for what that means for the
  NaN-to-``-inf`` fitness rule.
* A constant-folding trap: the tie (equal logits) must push **left**, matching
  ``argmax`` index 0, so the tie test asserts the always-left return.

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


def _mixed_row_policy(enp, split_rows):
    """Rows before ``split_rows`` always push right; the rest damp the cart.

    Args:
        enp: The ``evoxc.numpy`` op namespace.
        split_rows: Row index at which the policy switches.

    Returns:
        ``act_fn(batch) -> (B, 2) float32``. Rows ``< split_rows`` emit
        ``(0, 1)`` (push right, a constant policy that dies after
        :data:`CONSTANT_POLICY_SURVIVAL` steps); the remaining rows emit
        ``(1, 0)`` when ``theta_dot >= 0`` and ``(0, 1)`` otherwise, i.e. they
        push *against* the pole's angular velocity -- a state-dependent policy
        with a different survival time.

    Notes:
        The row index is the tiled episode index, so this policy is how a test
        observes the ``B = P * repeat_times`` layout without relying on the
        random draw (see :func:`test_repeat_times_averages_individual_major_rows`).
    """

    def act_fn(batch):
        n = batch.shape[0]
        head = enp.less(enp.arange(n), split_rows)
        theta_dot = enp.reshape(enp.split(batch, 4, axis=1)[3], (n,))
        damping = enp.less(theta_dot, 0.0)
        push_right = enp.logical_or(head, damping)
        return enp.stack(
            [
                enp.where(push_right, enp.full((n,), 0.0), enp.full((n,), 1.0)),
                enp.where(push_right, enp.full((n,), 1.0), enp.full((n,), 0.0)),
            ],
            axis=1,
        )

    return act_fn


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

    Notes:
        The two horizons are the point of the parametrization: a fixed policy's
        return must not change once it exceeds the death step (see
        :func:`test_an_episode_that_dies_at_step_k_never_scores_more_than_k`).
        ``direction = 0`` (push left) is compared with a zero action matrix, so
        the ``out[1] > out[0]`` decision rule is exercised in both senses.
    """
    pytest.skip(
        "not implemented: a constant policy reproduces the numpy oracle rollout "
        "(owner: problem/rl/cartpole.CartPole.evaluate)"
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
        The same number is asserted for ``max_step`` 20 and 50 (a horizon
        shorter than the death step returns the horizon instead), and the numpy
        oracle is cross-checked on the same configuration so a slip in this
        arithmetic shows up as a disagreement between two references.
    """
    pytest.skip(
        "not implemented: a constant push-right policy from rest returns 9 "
        "(owner: problem/rl/cartpole.CartPole.evaluate)"
    )


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
        place the rule is observable in isolation.
    """
    pytest.skip(
        "not implemented: out[1] > out[0] selects +force_mag, a tie selects -force_mag "
        "(owner: problem/rl/cartpole.CartPole._force_from_logits)"
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
    pytest.skip(
        "not implemented: death is |x| > 2.4 or |theta| > 12deg, strictly "
        "(owner: problem/rl/cartpole.CartPole._terminated/_constants)"
    )


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
    pytest.skip(
        "not implemented: the state is frozen after death, so the return is the "
        "survival time (owner: problem/rl/cartpole.CartPole.evaluate scan carry)"
    )


# --------------------------------------------------------------------------- #
# repeat_times: the (P, R) layout and the averaging
# --------------------------------------------------------------------------- #


def test_repeat_times_averages_individual_major_rows(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: ``B = P * R`` is individual-major and the fitness is the mean over ``R``.

    Oracle: an identity between two runs, not a number. ``P = 4, R = 1`` and
    ``P = 2, R = 2`` draw the **same four** initial states from the same key
    (``B = 4`` in both), so with a row-dependent policy the first run exposes the
    four per-row returns ``[a, a, b, b]`` and the second must produce
    ``[(a + a) / 2, (b + b) / 2]``: row ``p * R + r`` belongs to individual ``p``.
    A repeat-major layout would instead pair ``(a, b)`` in both individuals and
    produce two equal values.

    Notes:
        ``reset_scale = 0.0`` keeps the comparison exact (identical ``obs0`` for
        every row), and the policy distinguishes rows by index: rows ``0, 1``
        push right constantly, rows ``2, 3`` push against the pole's angular
        velocity, so ``a != b``. The test asserts ``a != b`` first, so the layout
        check cannot pass vacuously; the mean (rather than the sum) is pinned by
        also asserting the result is not ``2 * [a, b]``.
    """
    pytest.skip(
        "not implemented: B = P*R rows are individual-major and the fitness means "
        "over the repeat axis (owner: problem/rl/cartpole.CartPole._finish/evaluate)"
    )


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
        The draw is one ``evoxc.random.uniform(key, (B, 4), low=-reset_scale,
        high=reset_scale)``, never ``np.random``, so reproducibility is a
        property of the key alone -- this is the test that would catch a
        reintroduced global RNG or a key threaded through the scan.
    """
    pytest.skip(
        "not implemented: the reset draw is one evoxc.random.uniform on the key, "
        "and the rollout is deterministic per key (owner: problem/rl/cartpole.CartPole.evaluate)"
    )


def test_the_return_is_an_integer_between_zero_and_max_step(
    state, problem, enp, run_evaluate, fake_schedule
):
    """Invariant: ``0 <= fitness <= max_step`` and integral, for any key.

    Oracle: the documented algebra -- the reward is the ``alive`` mask, so a
    return counts steps and is an integer; the state freezes after death, so it
    can never exceed the horizon. Checked for several keys, both constant and
    state-dependent policies.

    Notes:
        This is also why ``CartPole`` documents that ``fitness_target`` must be
        ``<= max_step``: a target above the ceiling is unreachable and the run
        would hit ``generation_limit`` instead of converging. No shipped code
        validates that today (``Pipeline`` only compares ``max(fitnesses)``
        against the target), so the test asserts the reachable range, which is
        the fact the guidance rests on, and leaves the missing validation to the
        report to the parent.
    """
    pytest.skip(
        "not implemented: the return is an integer in [0, max_step] "
        "(owner: problem/rl/cartpole.CartPole.evaluate + _finish)"
    )


def test_nan_logits_fall_back_to_push_left_and_keep_the_fitness_finite(
    state, problem, oracle, enp, run_in_graph, run_evaluate, fake_schedule
):
    """Invariant: a NaN-logit policy acts as "push left"; the fitness stays finite.

    Oracle: closed form at the decision site (``enp.greater(nan, nan)`` is
    ``False``, so ``_force_from_logits`` must return ``-force_mag``) plus the
    numpy oracle's always-left return through ``evaluate`` -- with
    ``reset_scale = 0.0`` the two are comparable.

    Notes:
        **Doc conflict, reported to the parent.** CartPole's fitness is a mean of
        0/1 rewards, so it can never be ``NaN`` and ``finalize_fitness`` never
        turns it into ``-inf``: a NaN-poisoned network scores *normally* here,
        unlike the func-fit family, where "a NaN prediction gives ``-inf``
        fitness". Poisoning NaN logits would need an explicit ``evoxc.isnan``
        guard in ``_force_from_logits``; that is a ``src`` decision, so this test
        pins the behaviour the documented algebra actually produces and must be
        flipped if ``src`` adds the guard.
    """
    pytest.skip(
        "not implemented: NaN logits compare False and act as push-left, so the "
        "fitness stays finite (owner: problem/rl/cartpole.CartPole._force_from_logits)"
    )


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
    ],
)
def test_out_of_domain_configuration_raises_tensorneat_error(problem, common, kwargs):
    """Invariant: every value outside its documented domain raises ``TensorNEATError``.

    Oracle: the ``Raises:`` clause of the constructor docstring -- ``max_step``
    and ``repeat_times`` must be positive; every mass, ``length``, ``dt`` and
    ``force_mag`` must be strictly positive; ``theta_threshold`` must be
    positive; ``x_threshold`` and ``reset_scale`` must be non-negative. One case
    per knob, so a partially-validating constructor is caught.

    Notes:
        ``reset_scale = 0.0`` is explicitly *not* an error (the suite relies on
        it), which is why that knob's negative case is the one asserted.
    """
    pytest.skip(
        "not implemented: out-of-domain CartPole configuration raises TensorNEATError "
        "(owner: problem/rl/cartpole.CartPole.__init__)"
    )
