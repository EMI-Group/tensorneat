"""Batched Pendulum rollout on evoxc: Gym dynamics, fixed horizon, return fitness."""

from typing import Any, Callable, NamedTuple

import numpy as np

import evoxc
import evoxc.numpy as enp

from ..base import BaseProblem, finalize_fitness, static_batch_size
from ...common.errors import TensorNEATError
from ._common import require_positive, require_positive_int, returns_of, to_numpy

__all__ = ["Pendulum"]

#: Number of state variables in an observation ``(theta, theta_dot)``.
_OBS_DIM = 2
#: Number of logits a policy must emit for this problem.
_ACTION_DIM = 1


class _PendulumConstants(NamedTuple):
    """The physics bundle of one ``Pendulum`` configuration, as graph scalars.

    Every field is a rank-0 ``SymbolicTensor`` created inside the active trace
    with ``enp.full((), value)``, so no Python literal (a float would
    materialize as f64 on the stablehlo/iree path) ever reaches an op.

    The bundle is built **once per rollout, outside the timestep loop**, so a
    constant the dynamics needs every step (``two_pi``, ``neg_max_speed``, the
    penalties) is one Constant op per run instead of one per timestep -- the
    reference interpreter pays per traced op, not per element.

    Attributes:
        gravity_term: ``3 * gravity / (2 * length)``, the coefficient of
            ``sin(theta)`` in the angular acceleration (precomputed on the host
            so the body never divides by ``2 * length``).
        torque_term: ``3.0 / (mass * length**2)``, the coefficient of the
            applied torque in the angular acceleration.
        dt: Euler integration step.
        max_speed: ``|theta_dot|`` clip, applied **before** the angle update.
        neg_max_speed: ``-max_speed``, the lower clip bound (precomputed so the
            loop never negates a constant).
        max_torque: Bound of the ``tanh``-squashed action.
        pi: Half turn in radians, the centre of the angle-normalisation window.
        two_pi: One full turn in radians, the modulus of that window.
        neg_pi: ``-pi``, the offset that moves the normalised angle back into
            ``[-pi, pi)``.
        speed_penalty: Weight of the angular-velocity cost (``0.1``).
        torque_penalty: Weight of the control cost (``0.001``).
        one: The float32 ``1`` (the upper bound of the reset draw).
        neg_one: ``-1``, the lower bound of the reset draw.
        one_step: The int32 ``1`` the loop counter advances by.
        horizon: The int32 ``max_step`` the loop counter is compared against.
    """

    gravity_term: Any
    torque_term: Any
    dt: Any
    max_speed: Any
    neg_max_speed: Any
    max_torque: Any
    pi: Any
    two_pi: Any
    neg_pi: Any
    speed_penalty: Any
    torque_penalty: Any
    one: Any
    neg_one: Any
    one_step: Any
    horizon: Any


class Pendulum(BaseProblem):
    """Classic Gym ``Pendulum-v1`` with Euler integration, written in evoxc ops.

    No simulator is imported: the dynamics are the algebra below, executed as one
    ``evoxc.while_loop`` over a **fixed** horizon. There is no terminal state, so
    every episode runs the whole ``max_step`` and an individual's fitness is the
    mean of its (negative) cumulative costs, i.e. a value in
    ``[-max_step * max_cost, 0]``: the return is never positive, which is why
    ``fitness_target`` may only be ``<= 0``.

    The action is continuous. A genome used with this problem must emit **one**
    logit and must leave ``output_transform`` unset (identity): this problem
    owns the squashing, ``u = max_torque * tanh(logit)``, so a bounded transform
    such as ``ACT.tanh`` would rescale the logit before it ever arrives here and
    the effective torque range would silently shrink.

    Attributes:
        max_step: Number of timesteps simulated per episode.
        repeat_times: Independent initial states averaged per individual.
        gravity: Gravity constant (default ``10.0``).
        mass: Mass of the pendulum bob (default ``1.0``).
        length: Length of the pendulum (default ``1.0``).
        max_speed: ``|theta_dot|`` clip (default ``8.0``).
        max_torque: Bound of the applied torque (default ``2.0``).
        dt: Integration timestep (default ``0.05``).
        fitness_target: Optional fitness target used only for validation; any
            positive value is unreachable and is rejected.
    """

    def __init__(
        self,
        max_step: int = 200,
        repeat_times: int = 1,
        gravity: float = 10.0,
        mass: float = 1.0,
        length: float = 1.0,
        max_speed: float = 8.0,
        max_torque: float = 2.0,
        dt: float = 0.05,
        fitness_target: float | None = None,
    ) -> None:
        """Configure the environment's physics and rollout shape.

        The physics follows Gym's ``Pendulum-v1``. The two derived coefficients
        are computed here, on the host, so the kernel only ever multiplies:
        ``gravity_term = 3 * gravity / (2 * length)`` and
        ``torque_term = 3.0 / (mass * length**2)``.

        Args:
            max_step: Fixed loop horizon, i.e. the number of timesteps scored.
            repeat_times: Independent initial states averaged per individual.
            gravity: Gravity constant.
            mass: Mass of the pendulum bob.
            length: Length of the pendulum.
            max_speed: ``theta_dot`` is clipped to ``[-max_speed, max_speed]``
                after the velocity update and before the angle update.
            max_torque: The action logit is squashed to
                ``[-max_torque, max_torque]``.
            dt: Euler integration step.
            fitness_target: Optional fitness target; ``None`` disables the
                check. Every Pendulum reward is ``-cost <= 0``, so the mean
                episode return can never exceed ``0`` and a positive target is
                unreachable.

        Raises:
            TensorNEATError: If ``max_step <= 0``, ``repeat_times <= 0``, or any
                of ``gravity``/``mass``/``length``/``max_speed``/``max_torque``/
                ``dt`` is not positive.
            TensorNEATError: If ``fitness_target`` is positive.
        """
        self.max_step = require_positive_int("Pendulum", "max_step", max_step)
        self.repeat_times = require_positive_int(
            "Pendulum", "repeat_times", repeat_times
        )
        self.gravity = require_positive("Pendulum", "gravity", gravity)
        self.mass = require_positive("Pendulum", "mass", mass)
        self.length = require_positive("Pendulum", "length", length)
        self.max_speed = require_positive("Pendulum", "max_speed", max_speed)
        self.max_torque = require_positive("Pendulum", "max_torque", max_torque)
        self.dt = require_positive("Pendulum", "dt", dt)
        if fitness_target is not None:
            target = float(fitness_target)
            if target > 0.0:
                raise TensorNEATError(
                    f"Pendulum: fitness_target ({target}) is positive, but every "
                    "Pendulum reward is -cost <= 0, so the mean episode return can "
                    "never exceed 0 -- such a target can never be reached"
                )
        self.fitness_target = fitness_target

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Observation shape ``(2,)`` = ``(theta, theta_dot)``.

        ``theta`` is the pendulum angle in **radians** measured from the upright
        equilibrium, ``theta_dot`` its angular velocity.

        Returns:
            The per-step observation shape.
        """
        return (_OBS_DIM,)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Action shape ``(1,)``: one torque logit.

        The problem owns the squashing, ``u = max_torque * tanh(logit)``, so the
        genome must not apply an ``output_transform`` of its own -- a bounded
        transform would rescale the logit before this kernel ever sees it.

        Returns:
            The per-step output shape.
        """
        return (_ACTION_DIM,)

    def evaluate(
        self,
        state: Any,
        keys: Any,
        act_fn: Callable,
        transformed: Any,
    ) -> Any:
        """Roll out ``B = P * repeat_times`` episodes and return ``(P,)`` fitness.

        The algebra:

        1. ``P = static_batch_size(transformed)`` and ``B = P * repeat_times``,
           individual-major: row ``p * R + r`` is repeat ``r`` of individual
           ``p``.
        2. Initial states come from **one** draw,
           ``evoxc.random.uniform(keys, (B, 2), low=-1, high=1)``, rescaled per
           column: ``theta = raw_0 * pi`` and ``theta_dot = raw_1 * max_speed``.
           The key is not threaded through the loop and ``np.random`` is never
           used.
        3. One ``evoxc.while_loop`` over ``max_step`` timesteps, carry
           ``(obs (B, 2), total (B,), step ())``: the return accumulator rides in
           the carry instead of being gathered from a scan output stack, which
           is what keeps the per-timestep cost free of any ``B * max_step``
           array traffic.
        4. Step body, following Gym's ``Pendulum-v1`` order and evaluating every
           term on the **pre-transition** state::

               u         = max_torque * tanh(logit)
               cost      = angle_normalize(theta)**2
                         + 0.1 * theta_dot**2
                         + 0.001 * u**2
               new_thdot = theta_dot + (gravity_term * sin(theta)
                                        + torque_term * u) * dt
               new_thdot = clip(new_thdot, -max_speed, max_speed)
               new_theta = theta + new_thdot * dt
               total     = total - cost

           with ``angle_normalize(x) = remainder(x + pi, 2 * pi) - pi``.
        5. Fitness = mean over the ``repeat_times`` axis of the episode return
           (a value in ``[-max_step * max_cost, 0]`` per episode), then
           ``finalize_fitness``.

        There is no termination and no alive mask: the trip count is the static
        ``max_step``, so every episode is scored over the same horizon.

        A NaN logit propagates on its own: ``tanh(NaN) = NaN``, so
        ``0.001 * u**2`` is NaN, the step's cost is NaN, the accumulator turns
        NaN and stays NaN for the rest of the episode. The mean over
        ``repeat_times`` keeps it NaN and ``finalize_fitness`` maps it to
        ``-inf``, so a poisoned policy can never win selection.

        Trace hygiene: every arithmetic constant is an in-trace
        ``enp.full((), v)`` so it stays float32 on the stablehlo/iree path, and
        all axes are non-negative.

        Args:
            state: Host container; unused, kept for protocol uniformity.
            keys: One evoxc key, used for the single reset draw.
            act_fn: Callable mapping ``(B, 2)`` observations to ``(B, 1)``
                logits.
            transformed: Batch-first execution schedule holding ``P``
                individuals.

        Returns:
            ``(P,)`` fitness = mean negated cumulative cost, NaN mapped to
            ``-inf``.

        Raises:
            TensorNEATError: If ``transformed`` carries no static population
                axis (raised by :func:`static_batch_size`).
        """
        pop_size = static_batch_size(transformed)
        total, _ = self._rollout(keys, act_fn, pop_size)
        return finalize_fitness(
            enp.mean(returns_of(total, pop_size, self.repeat_times), axis=1)
        )

    def show(
        self,
        state: Any,
        key: Any,
        act_fn: Callable,
        transformed: Any,
    ) -> None:
        """Print the outcome of a single rollout.

        Host-side only: never called from a trace, and it builds nothing at
        module scope. The report runs its own graph of one individual -- one
        data argument (the rank-0 key) and ``pop_size = 1`` -- and reads both the
        return and the final observation out of the rollout carry, so no extra
        stacked output is needed.

        Args:
            state: Host container with configuration.
            key: One evoxc key for the reset draw.
            act_fn: Callable mapping observations to logits.
            transformed: A schedule holding a single individual.

        Returns:
            ``None``; the report goes to stdout.
        """
        key_spec = evoxc.TensorSpec(tuple(key.shape), key.dtype)

        def report(key_value):
            """Roll out one individual and return its per-episode results.

            Args:
                key_value: The rank-0 reset key, as a graph value.

            Returns:
                ``(returns (R,), final_obs (R, 2))`` for the single individual.
            """
            total, final_obs = self._rollout(key_value, act_fn, 1)
            return enp.reshape(total, (self.repeat_times,)), final_obs

        executable = evoxc.build(report, key_spec)
        returns, final_obs = evoxc.run(executable, key)
        returns = to_numpy(returns)
        final_obs = to_numpy(final_obs)
        for repeat in range(self.repeat_times):
            theta, theta_dot = (float(value) for value in final_obs[repeat])
            print(
                f"Pendulum repeat {repeat}: return {float(returns[repeat]):.2f}"
            )
            print(
                f"\tfinal state: theta={theta:.4f} rad, "
                f"theta_dot={theta_dot:.4f}"
            )
        print(f"Pendulum mean return: {float(np.mean(returns)):.2f}")
        return None

    def _rollout(
        self,
        keys: Any,
        act_fn: Callable,
        pop_size: int,
    ) -> tuple[Any, Any]:
        """Roll out ``pop_size * repeat_times`` episodes for one key.

        Must be called inside the active trace: it draws the initial states with
        ``evoxc.random.uniform`` and runs one ``evoxc.while_loop`` over
        ``max_step`` timesteps, carry ``(obs (B, 2), total (B,), step ())`` with
        ``B = pop_size * repeat_times`` in individual-major order.

        Args:
            keys: One rank-0 evoxc key for the single reset draw.
            act_fn: Callable mapping ``(B, 2)`` observations to ``(B, 1)``
                logits.
            pop_size: Number of individuals ``P`` (static).

        Returns:
            The pair ``(total (B,), final_obs (B, 2))``, where ``total`` is the
            accumulated (negated) cost of each episode -- a value in
            ``[-max_step * max_cost, 0]``, NaN once that episode's policy
            emitted a NaN logit -- and ``final_obs`` the last observation the
            carry holds.
        """
        consts = self._constants()
        batch = pop_size * self.repeat_times
        obs0 = self._reset(keys, consts, batch)
        init = (
            obs0,
            enp.zeros((batch,), dtype=evoxc.float32),
            enp.zeros((), dtype=evoxc.int32),
        )
        obs, total, _ = evoxc.while_loop(
            self._loop_cond(consts),
            self._loop_body(act_fn, consts),
            init,
        )
        return total, obs

    def _reset(
        self,
        keys: Any,
        consts: _PendulumConstants,
        batch: int,
    ) -> Any:
        """Draw the initial observations of ``batch`` episodes from one key.

        The draw is a single ``U(-1, 1)`` block of shape ``(B, 2)``; the two
        columns are then scaled by different factors (``pi`` for the angle,
        ``max_speed`` for the angular velocity), which is done with
        ``enp.split`` + ``enp.concatenate`` rather than by indexing, because a
        traced tensor has no ``__getitem__``.

        Args:
            keys: One rank-0 evoxc key.
            consts: The bundle from :meth:`_constants`.
            batch: Number of episodes ``B`` (static).

        Returns:
            ``(B, 2)`` initial observations: ``theta`` in ``[-pi, pi)``,
            ``theta_dot`` in ``[-max_speed, max_speed]``.
        """
        raw = evoxc.random.uniform(
            keys,
            (batch, _OBS_DIM),
            low=consts.neg_one,
            high=consts.one,
        )
        raw_theta, raw_speed = enp.split(raw, _OBS_DIM, axis=1)
        return enp.concatenate(
            (
                raw_theta * consts.pi,
                raw_speed * consts.max_speed,
            ),
            axis=1,
        )

    def _loop_cond(self, consts: _PendulumConstants) -> Callable:
        """Build the trip condition of the rollout loop.

        The horizon is static config, so the loop runs exactly ``max_step``
        iterations: the condition counts up from ``0`` instead of testing a
        terminal flag, which Pendulum does not have anyway.

        Args:
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``cond(carry) -> () bool`` -- true while ``step < max_step``.
        """

        def cond(carry):
            """Test the loop counter against the static horizon.

            Args:
                carry: ``(obs, total, step)``.

            Returns:
                The rank-0 boolean ``step < max_step``.
            """
            return enp.less(carry[2], consts.horizon)

        return cond

    def _loop_body(
        self,
        act_fn: Callable,
        consts: _PendulumConstants,
    ) -> Callable:
        """Build the per-timestep body of the rollout loop.

        One step forwards the policy, converts the logit into a bounded torque,
        scores the **pre-transition** state, integrates, and subtracts the cost
        from the carried return accumulator, so the carried types stay constant
        across iterations (what ``evoxc.while_loop`` requires).

        The accumulator rides in the carry rather than being emitted as a scan
        output, so no ``(max_step, B)`` stack is ever allocated or scattered
        into. There is no alive mask: Pendulum has no terminal state, so every
        episode is scored over the full horizon.

        Args:
            act_fn: Callable mapping ``(B, 2)`` observations to ``(B, 1)``
                logits.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``step(carry) -> (obs, total, step)`` -- the next carry.
        """

        def step(carry):
            """Advance every episode by one timestep.

            Args:
                carry: ``(obs (B, 2), total (B,), step ())``.

            Returns:
                The next carry, same tree and leaf types.
            """
            obs, total, index = carry
            logit = act_fn(obs)
            next_obs, cost = self._advance(obs, logit, consts)
            return (
                next_obs,
                total - cost,
                index + consts.one_step,
            )

        return step

    def _constants(self) -> _PendulumConstants:
        """Bundle the physics constants as float32 graph scalars.

        Must be called inside the trace: ``enp.full((), v)`` is a graph
        constructor, and a bare Python float would materialize as f64 on the
        stablehlo path and break iree type unification.

        The caller builds this bundle **once per rollout**, outside the timestep
        loop, so a scalar the body needs every step (``two_pi``,
        ``neg_max_speed``, the penalties) costs one Constant op per run rather
        than one per timestep.

        Returns:
            A structure holding ``gravity_term`` (``3 * gravity / (2 * length)``)
            and ``torque_term`` (``3.0 / (mass * length**2)``) precomputed on the
            host, plus every other scalar the rollout needs as a float32 graph
            value, and the int32 loop scalars ``one_step`` and ``horizon`` -- so
            neither the reset draw nor the trip condition sees an f64 constant.
        """
        return _PendulumConstants(
            gravity_term=enp.full((), 3.0 * self.gravity / (2.0 * self.length)),
            torque_term=enp.full((), 3.0 / (self.mass * self.length**2)),
            dt=enp.full((), self.dt),
            max_speed=enp.full((), self.max_speed),
            neg_max_speed=enp.full((), -self.max_speed),
            max_torque=enp.full((), self.max_torque),
            pi=enp.full((), float(np.pi)),
            two_pi=enp.full((), float(2.0 * np.pi)),
            neg_pi=enp.full((), float(-np.pi)),
            speed_penalty=enp.full((), 0.1),
            torque_penalty=enp.full((), 0.001),
            one=enp.full((), 1.0),
            neg_one=enp.full((), -1.0),
            one_step=enp.full((), 1, dtype=evoxc.int32),
            horizon=enp.full((), self.max_step, dtype=evoxc.int32),
        )

    def _advance(
        self,
        obs: Any,
        logit: Any,
        consts: _PendulumConstants,
    ) -> tuple[Any, Any]:
        """Advance the ``(B, 2)`` observations by one Euler step and score them.

        With ``u = max_torque * tanh(logit)``, ``theta``/``theta_dot`` split out
        of the observation row and ``angle_normalize`` as in
        :meth:`_angle_normalize`, the step is exactly Gym's ordering: the cost is
        evaluated on the **pre-transition** state, the velocity is integrated and
        clipped, and only then is the angle integrated with the clipped velocity.

        The clip is ``enp.clip``, i.e. ``minimum(maximum(x, -max_speed),
        max_speed)``, and both bounds are symbolic graph scalars like every other
        constant in the trace.

        Args:
            obs: Current observations of shape ``(B, 2)``.
            logit: Policy output of shape ``(B, 1)``.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``(next_obs (B, 2), cost (B,))`` -- the transitioned observations and
            the (non-negative) cost of the state they came from, reshaped to
            rank 1 so it can be subtracted from the rank-1 return accumulator
            (``(B,) - (B, 1)`` would broadcast into a ``(B, B)`` matrix).
        """
        theta, theta_dot = enp.split(obs, _OBS_DIM, axis=1)
        torque = consts.max_torque * enp.tanh(logit)
        angle = self._angle_normalize(theta, consts)
        cost = (
            enp.square(angle)
            + consts.speed_penalty * enp.square(theta_dot)
            + consts.torque_penalty * enp.square(torque)
        )
        next_speed = theta_dot + (
            consts.gravity_term * enp.sin(theta) + consts.torque_term * torque
        ) * consts.dt
        next_speed = enp.clip(next_speed, consts.neg_max_speed, consts.max_speed)
        next_theta = theta + next_speed * consts.dt
        next_obs = enp.concatenate((next_theta, next_speed), axis=1)
        return next_obs, enp.reshape(cost, (obs.shape[0],))

    def _angle_normalize(self, x: Any, consts: _PendulumConstants) -> Any:
        """Wrap an angle in radians into ``[-pi, pi)``.

        This is exactly Gym's ``((x + pi) % (2 * pi)) - pi``.
        ``evoxc.remainder`` is the floor-modulo of the numpy ``remainder``
        contract (the result carries the sign of the *divisor*, verified
        empirically on the numpy backend), which is what makes the wrapping
        agree with Gym rather than with ``fmod``; ``enp`` has no ``remainder``,
        so the top-level op is the escape hatch.

        Args:
            x: Angles in radians, any shape.
            consts: The bundle from :meth:`_constants`.

        Returns:
            The same-shape angle in ``[-pi, pi)``.
        """
        return evoxc.remainder(x + consts.pi, consts.two_pi) + consts.neg_pi
