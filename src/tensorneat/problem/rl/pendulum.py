"""Batched Pendulum rollout on evoxc: Gym dynamics, fixed horizon, return fitness."""

from typing import Any, Callable, NamedTuple

import numpy as np

import evoxc
import evoxc.numpy as enp
from evoxc import ops

from ..base import BaseProblem, build_options, finalize_fitness, static_batch_size
from ...common.errors import TensorNEATError
from ...common.schedule_accelerator import kernel_mode
from ._common import (
    horizon_cond,
    prepared_query,
    require_finite,
    require_positive,
    require_positive_int,
    returns_of,
    to_numpy,
)

__all__ = ["Pendulum"]

#: Number of state variables in an observation ``(theta, theta_dot)``.
_OBS_DIM = 2
#: Number of logits a policy must emit for this problem.
_ACTION_DIM = 1


class _PendulumConstants(NamedTuple):
    """The physics bundle of one ``Pendulum`` configuration, as graph scalars.

    Every *scalar* field is a rank-0 ``SymbolicTensor`` created inside the
    active trace with ``enp.full((), value)``, so no Python literal (a float
    would materialize as f64 on the stablehlo/iree path) ever reaches an op.
    The two row-shaped fields (``obs_columns``, ``reset_scale``) are boolean /
    float32 rows baked with ``evoxc.constant(evoxc.from_numpy(...))``, because
    ``enp.full`` only fills a scalar; they are data, never arithmetic, so they
    introduce no float64 constant.

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
        speed_penalty: Weight of the angular-velocity cost (``0.1``).
        torque_penalty: Weight of the control cost (``0.001``).
        one: The float32 ``1`` (the upper bound of the reset draw).
        neg_one: ``-1``, the lower bound of the reset draw.
        one_step: The int32 ``1`` the loop counter advances by.
        horizon: The int32 ``max_step`` the loop counter is compared against.
        obs_columns: The ``(1, 2)`` boolean ``[True, False]`` that assembles the
            policy's observation row from the two carried state leaves; one
            elementwise select against it is one dispatch, where an
            ``enp.concatenate`` of two ``(B, 1)`` columns lowers to one strided
            copy per column.
        reset_scale: The ``(1, 2)`` float32 ``[pi, max_speed]`` the single
            ``U(-1, 1)`` reset draw is scaled by, so the initial observation is
            one elementwise multiply instead of a split, two multiplies and a
            concatenate. Each element is ``raw * scale`` with the same operand
            the per-column spelling used, so the draw is bit-identical.
    """

    gravity_term: Any
    torque_term: Any
    dt: Any
    max_speed: Any
    neg_max_speed: Any
    max_torque: Any
    pi: Any
    two_pi: Any
    speed_penalty: Any
    torque_penalty: Any
    one: Any
    neg_one: Any
    one_step: Any
    horizon: Any
    obs_columns: Any
    reset_scale: Any


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
            positive value is unreachable and is rejected, and so is one that is
            not a finite number.
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
                ``dt`` is not a finite, positive number.
            TensorNEATError: If ``fitness_target`` is not a finite number, or is
                positive.
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
            target = require_finite("Pendulum", "fitness_target", fitness_target)
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
        3. One ``evoxc.while_loop`` over ``max_step`` timesteps. Two carry
           formulations ship, chosen by the trace-time static
           ``state.kernel_mode`` (see :meth:`_rollout`): the host carry
           ``(theta (B, 1), theta_dot (B, 1), total (B, 1), step ())`` and, on
           ``"accelerator"``, the packed ``(row (B, 3), step ())`` with
           ``row = (theta, theta_dot, total)``. Either way the return
           accumulator rides in the carry instead of being gathered from a scan
           output stack, which is what keeps the per-timestep cost free of any
           ``B * max_step`` array traffic.
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

           with ``angle_normalize(x) = x - 2*pi*floor((x + pi) / (2*pi))``,
           the backend-portable spelling of Gym's ``((x + pi) % (2*pi)) - pi``.
           The algebra is :meth:`_transition_rows` plus the ``new_theta`` line,
           shared verbatim by both carry formulations.
        5. Fitness = mean over the ``repeat_times`` axis of the episode return
           (a value in ``[-max_step * max_cost, 0]`` per episode), then
           ``finalize_fitness``. The shipped ``repeat_times=1`` already makes
           ``total`` one return per individual, so that path skips the
           ``returns_of`` reshape and the mean entirely (two ops on the
           identity); only ``repeat_times > 1`` pays them.

        There is no termination and no alive mask: the trip count is the static
        ``max_step``, so every episode is scored over the same horizon.

        A NaN logit propagates on its own: ``tanh(NaN) = NaN``, so
        ``0.001 * u**2`` is NaN, the step's cost is NaN, the accumulator turns
        NaN and stays NaN for the rest of the episode. The mean over
        ``repeat_times`` keeps it NaN and ``finalize_fitness`` maps it to
        ``-inf``, so a poisoned policy can never win selection. The packed
        accelerator carry additionally drags that NaN into the ``theta`` column
        (via ``NaN * 0``), which is outcome-identical because the fitness reads
        the accumulator column only.

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
        total, _ = self._rollout(keys, act_fn, pop_size, kernel_mode(state))
        if self.repeat_times == 1:
            # ``repeat_times`` is host-side config, so this is a trace-time
            # branch and never a branch on a traced value: with one repeat
            # ``total`` already holds one return per individual, and the
            # ``returns_of`` reshape plus the mean would be two ops on the
            # identity.
            return finalize_fitness(total)
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

        That graph is built with the options the run registered on ``state``
        (``backend``/``device``/``compile_options``, see
        :func:`tensorneat.problem.base.build_options`), so
        ``Pipeline(backend="iree").show`` reports from the backend the training
        ran on, while a problem used directly -- a state registering no backend
        -- reports from the numpy reference interpreter. There is no silent
        fallback: a backend that cannot run the report raises.

        The report also traces the carry formulation the run trained with
        (``state.kernel_mode``, see :meth:`_rollout`), so a state on the
        accelerator leg reports from the packed carry formulation instead of
        the host one.

        A non-finite return (a NaN-poisoned policy) prints as ``nan`` in the
        return columns rather than raising: unlike ``CartPole``'s survival
        count, nothing here needs an integer.

        Args:
            state: Host container with configuration.
            key: One evoxc key for the reset draw.
            act_fn: Callable mapping observations to logits.
            transformed: A schedule holding a single individual.

        Returns:
            ``None``; the report goes to stdout.
        """
        key_spec = evoxc.TensorSpec(tuple(key.shape), key.dtype)
        mode = kernel_mode(state)

        def report(key_value):
            """Roll out one individual and return its per-episode results.

            Args:
                key_value: The rank-0 reset key, as a graph value.

            Returns:
                ``(returns (R,), final_obs (R, 2))`` for the single individual.
            """
            total, final_obs = self._rollout(key_value, act_fn, 1, mode)
            return enp.reshape(total, (self.repeat_times,)), final_obs

        executable = evoxc.build(report, key_spec, **build_options(state))
        returns, final_obs = evoxc.run(executable, key)
        returns = to_numpy(returns)
        final_obs = to_numpy(final_obs)
        for repeat in range(self.repeat_times):
            theta, theta_dot = (float(entry) for entry in final_obs[repeat])
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
        mode: str,
    ) -> tuple[Any, Any]:
        """Roll out ``pop_size * repeat_times`` episodes for one key.

        Must be called inside the active trace: it draws the initial states with
        ``evoxc.random.uniform`` and runs one ``evoxc.while_loop`` over
        ``max_step`` timesteps with ``B = pop_size * repeat_times`` episodes in
        individual-major order. The policy query is prepared once for the whole
        rollout, so the per-pass operands are not re-derived per timestep.

        Two carry formulations ship, chosen by the trace-time static ``mode``
        (``state.kernel_mode``, a host string, so a graph contains exactly one of
        them):

        * ``"host"`` -- the four-leaf carry
          ``(theta (B, 1), theta_dot (B, 1), total (B, 1), step ())``. The two
          state variables ride as separate leaves rather than as one ``(B, 2)``
          observation row, which saves the two ``slice`` ops a per-step
          ``enp.split`` costs: the row is only built where it is actually
          consumed (the policy query), as one elementwise select against the
          baked ``(1, 2)`` boolean ``obs_columns``, and the accumulator rides at
          the cost's own rank so no step needs a ``reshape``.
        * ``"accelerator"`` -- the packed carry ``(row (B, 3), step ())`` with
          ``row = (theta, theta_dot, total)``, see :meth:`_packed_rollout`.

        Why the packing is gated rather than unconditional: on a compiled
        backend a dispatch is one loop-carried float root region, so collapsing
        three roots into one carried row reaches a single fused dispatch in a
        standalone rollout graph (against three elementwise ones for the
        three-leaf host carry). Two costs keep it off the host leg:

        * the packed step is 4 ops wider on numpy -- 33 against 29 in an
          env-only trace, from five slices and two extra multiplies in place of
          one select -- and accelerator mode costs **121 218** executed ops on
          one generation against **26 190** for host mode, because that mode
          also unrolls the genome's level sweep inside the rollout (a 565-op
          body per timestep where host mode runs 36 twice). The host leg
          therefore keeps the byte-identical host kernel: same per-step op list,
          same fitness;
        * in the **assembled generation graph** the packed body's column slices
          are materialised as their own copies instead of fusing into the
          carried region, so the whole-step dispatch win of the standalone body
          does not survive graph assembly; what survives is the per-query
          region, because the packed body slices its observation out of the
          carried row instead of selecting it out of two leaves.

        Args:
            keys: One rank-0 evoxc key for the single reset draw.
            act_fn: Callable mapping ``(B, 2)`` observations to ``(B, 1)``
                logits, or a prepared-query object exposing ``prepare``.
            pop_size: Number of individuals ``P`` (static).
            mode: Resolved ``state.kernel_mode``: ``"host"`` or
                ``"accelerator"`` (already validated by
                :func:`~tensorneat.common.schedule_accelerator.kernel_mode`).

        Returns:
            The pair ``(total (B,), final_obs (B, 2))``, where ``total`` is the
            accumulated (negated) cost of each episode -- a value in
            ``[-max_step * max_cost, 0]``, NaN once that episode's policy
            emitted a NaN logit -- and ``final_obs`` the last observation the
            carry holds.
        """
        consts = self._constants()
        batch = pop_size * self.repeat_times
        query = prepared_query(act_fn, batch)
        obs0 = self._reset(keys, consts, batch)
        theta0, theta_dot0 = enp.split(obs0, _OBS_DIM, axis=1)
        if mode == "accelerator":
            return self._packed_rollout(
                query, consts, batch, theta0, theta_dot0
            )
        init = (
            theta0,
            theta_dot0,
            enp.zeros((batch, 1), dtype=evoxc.float32),
            enp.zeros((), dtype=evoxc.int32),
        )
        # The trip condition is the shared fixed-horizon rule of
        # ``._common.horizon_cond`` at the host carry's counter slot: exactly
        # ``max_step`` iterations, counted up from ``0`` rather than tested
        # against a terminal flag (Pendulum has none).
        theta, theta_dot, total, _ = evoxc.while_loop(
            horizon_cond(consts),
            self._loop_body(query, consts),
            init,
        )
        return (
            enp.reshape(total, (batch,)),
            enp.concatenate((theta, theta_dot), axis=1),
        )

    def _packed_rollout(
        self,
        query: Callable,
        consts: _PendulumConstants,
        batch: int,
        theta0: Any,
        theta_dot0: Any,
    ) -> tuple[Any, Any]:
        """Roll out the packed step of the accelerator formulation.

        The carry is ``(row (B, 3), step ())`` with ``row = (theta, theta_dot,
        total)``: one float root instead of three, which is what a compiled
        backend fuses furthest (:meth:`_rollout` holds the measured caveat that
        the assembled generation graph materialises the column slices). The step
        is :meth:`_loop_body_packed`; everything after the loop is three
        contiguous column slices of the final row, so the packing costs no op
        outside the body.

        Args:
            query: The prepared policy query, as in :meth:`_rollout`.
            consts: The bundle from :meth:`_constants`.
            batch: Number of episodes ``B`` (static).
            theta0: Initial angles, ``(B, 1)``.
            theta_dot0: Initial angular velocities, ``(B, 1)``.

        Returns:
            The pair ``(total (B,), final_obs (B, 2))``, as :meth:`_rollout`.
        """
        init = (
            enp.concatenate(
                (
                    theta0,
                    theta_dot0,
                    enp.zeros((batch, 1), dtype=evoxc.float32),
                ),
                axis=1,
            ),
            enp.zeros((), dtype=evoxc.int32),
        )
        # Same fixed-horizon trip condition, at the packed ``(row, step)``
        # carry's counter slot.
        row, _ = evoxc.while_loop(
            horizon_cond(consts, 1),
            self._loop_body_packed(query, consts, batch),
            init,
        )
        return (
            enp.reshape(ops.slice(row, (0, 2), (batch, 1)), (batch,)),
            ops.slice(row, (0, 0), (batch, _OBS_DIM)),
        )

    def _reset(
        self,
        keys: Any,
        consts: _PendulumConstants,
        batch: int,
    ) -> Any:
        """Draw the initial observations of ``batch`` episodes from one key.

        The draw is a single ``U(-1, 1)`` block of shape ``(B, 2)``, scaled per
        column by the baked ``(1, 2)`` row ``reset_scale``: ``theta = raw_0 *
        pi`` and ``theta_dot = raw_1 * max_speed``. One elementwise multiply
        replaces the ``enp.split`` + two multiplies + ``enp.concatenate`` a
        per-column spelling costs, because a traced tensor has no
        ``__getitem__``; each element is ``raw * scale`` with exactly the
        operand the per-column spelling used, so the draw is bit-identical.

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
        return raw * consts.reset_scale

    def _loop_body(
        self,
        query: Callable,
        consts: _PendulumConstants,
    ) -> Callable:
        """Build the per-timestep body of the rollout loop.

        This is the **host** formulation; the accelerator leg uses the packed
        carry of :meth:`_loop_body_packed` instead, and :meth:`_rollout` picks
        one at trace time on ``state.kernel_mode``.

        One step forwards the policy, converts the logit into a bounded torque,
        scores the **pre-transition** state, integrates, and subtracts the cost
        from the carried return accumulator, so the carried types stay constant
        across iterations (what ``evoxc.while_loop`` requires).

        The accumulator rides in the carry rather than being emitted as a scan
        output, so no ``(max_step, B)`` stack is ever allocated or scattered
        into. There is no alive mask: Pendulum has no terminal state, so every
        episode is scored over the full horizon.

        The state rides as two ``(B, 1)`` leaves instead of one ``(B, 2)``
        observation row (see :meth:`_rollout`): the row is built for the policy
        query and nowhere else, as one select against ``obs_columns``, and the
        accumulator is carried at the cost's own rank.

        Args:
            query: The policy query, prepared once for the whole rollout by
                :func:`._common.prepared_query` so the per-pass operands are not
                re-derived per timestep; maps ``(B, 2)`` observations to
                ``(B, 1)`` logits.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``step(carry) -> (theta, theta_dot, total, step)`` -- the next carry.
        """

        def step(carry):
            """Advance every episode by one timestep.

            The policy is the only consumer of the ``(B, 2)`` observation row, so
            that row is built here, once, and handed straight to the query; the
            transition works on the two ``(B, 1)`` state leaves
            (:meth:`_advance_rows`) and returns the next pair, so no step ever
            splits a row it just built.

            The row is one elementwise select against the baked ``(1, 2)``
            boolean ``obs_columns`` (``[True, False]``), which broadcasts the two
            ``(B, 1)`` leaves into a ``(B, 2)`` row. That is the same op count as
            the ``enp.concatenate`` it replaced -- one -- but a concatenate of two
            columns lowers to one serialised copy per column on a compiled
            backend, while the select fuses into the observation producer.

            Args:
                carry: ``(theta (B, 1), theta_dot (B, 1), total (B, 1),
                    step ())``.

            Returns:
                The next carry, same tree and leaf types.
            """
            theta, theta_dot, total, index = carry
            obs = enp.where(consts.obs_columns, theta, theta_dot)
            next_theta, next_theta_dot, cost = self._advance_rows(
                theta, theta_dot, query(obs), consts
            )
            return (
                next_theta,
                next_theta_dot,
                total - cost,
                index + consts.one_step,
            )

        return step

    def _loop_body_packed(
        self,
        query: Callable,
        consts: _PendulumConstants,
        batch: int,
    ) -> Callable:
        """Build the per-timestep body of the packed (accelerator) carry.

        One step is the same algebra as :meth:`_loop_body` -- :meth:`_transition_rows`
        then ``theta' = theta + theta_dot' * dt`` and ``total' = total - cost``
        -- but it reads and writes one ``(B, 3)`` row ``(theta, theta_dot,
        total)`` instead of three float leaves, so the step has one carried
        float root instead of three (one dispatch per carried root on a
        compiled backend).

        The row is split with static slices (:func:`evoxc.ops.slice`): the
        leading ``(B, 2)`` window is the observation the policy consumes
        (``theta``, ``theta_dot``), and the two single-column slices are the
        state variables the transition reads. The next row is one fused
        expression::

            next_row = row * keep_dims + next_speed * speed_row - cost * total_dim

        with the three ``(1, 3)`` float32 rows
        ``keep_dims = [[1, 0, 1]]``, ``speed_row = [[dt, 1, 0]]`` and
        ``total_dim = [[0, 0, 1]]``, so column ``0`` is literally
        ``theta + next_speed * dt``, column ``1`` literally ``next_speed`` and
        column ``2`` literally ``total - cost`` -- bit-identical to the host
        carry for finite values. Both products against a ``0`` entry are exact
        (``0`` or the operand), and the ``- 0`` the theta column carries is an
        exact subtraction, so the packed step reproduces the host step op for
        op; a NaN cost does drag the ``theta`` column to NaN (``NaN * 0``),
        which the host carry does not, but the fitness reads the accumulator
        column only and any NaN there is already ``-inf``.

        One carried float root instead of three is what a compiled backend
        fuses furthest: in a standalone rollout graph the assembly above is a
        single dispatch region against three elementwise regions for the
        three-leaf host carry. In the assembled generation graph the column
        slices are materialised as their own copies rather than fused into the
        carried region, so the step block there holds those copies plus the
        fused region -- about the host formulation's count -- and the surviving
        win is the per-query region (see :meth:`_rollout`).

        The three baked rows are built **once per rollout, outside the loop**
        (here, at trace time, not inside ``step``), so the body pays no
        constant materialisation per timestep.

        Args:
            query: The policy query, prepared once for the whole rollout by
                :func:`._common.prepared_query`; maps ``(B, 2)`` observations to
                ``(B, 1)`` logits.
            consts: The bundle from :meth:`_constants`.
            batch: Number of episodes ``B`` (static), the row width.

        Returns:
            ``step(carry) -> (row, step)`` -- the next packed carry.
        """
        keep_dims = evoxc.constant(
            evoxc.from_numpy(np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32))
        )
        speed_row = evoxc.constant(
            evoxc.from_numpy(
                np.asarray([[self.dt, 1.0, 0.0]], dtype=np.float32)
            )
        )
        total_dim = evoxc.constant(
            evoxc.from_numpy(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32))
        )

        def step(carry):
            """Advance every episode by one packed timestep.

            The policy is the only consumer of a ``(B, 2)`` observation row, so
            that row is the leading two-column slice of the carried row, taken
            once and handed straight to the query; the transition then runs on
            the two single-column slices (:meth:`_transition_rows`).

            Slicing the observation out instead of building it with the
            ``obs_columns`` select of the host formulation is what keeps the
            per-query region at its host width: the select would need both
            columns of a three-column row, while the leading slice is one
            narrower.

            Args:
                carry: ``(row (B, 3), index ())`` with ``row`` holding
                    ``(theta, theta_dot, total)``.

            Returns:
                The next carry, same tree and leaf types.
            """
            row, index = carry
            obs = ops.slice(row, (0, 0), (batch, _OBS_DIM))
            logit = query(obs)
            theta = ops.slice(row, (0, 0), (batch, 1))
            theta_dot = ops.slice(row, (0, 1), (batch, 1))
            next_speed, cost = self._transition_rows(
                theta, theta_dot, logit, consts
            )
            next_row = (
                row * keep_dims
                + next_speed * speed_row
                - cost * total_dim
            )
            return next_row, index + consts.one_step

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

        The accelerator formulation's three ``(1, 3)`` selector rows
        (``keep_dims``/``speed_row``/``total_dim``) are deliberately *not* part
        of this bundle: only the packed body reads them, so
        :meth:`_loop_body_packed` bakes them per rollout and a host-mode graph
        carries no trace of the other formulation.

        Returns:
            A structure holding ``gravity_term`` (``3 * gravity / (2 * length)``)
            and ``torque_term`` (``3.0 / (mass * length**2)``) precomputed on the
            host, plus every other scalar the rollout needs as a float32 graph
            value, the int32 loop scalars ``one_step`` and ``horizon`` -- so
            neither the reset draw nor the trip condition sees an f64 constant --
            and the two baked rows ``obs_columns`` and ``reset_scale``.
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
            speed_penalty=enp.full((), 0.1),
            torque_penalty=enp.full((), 0.001),
            one=enp.full((), 1.0),
            neg_one=enp.full((), -1.0),
            one_step=enp.full((), 1, dtype=evoxc.int32),
            horizon=enp.full((), self.max_step, dtype=evoxc.int32),
            obs_columns=evoxc.constant(
                evoxc.from_numpy(np.asarray([[True, False]]))
            ),
            reset_scale=evoxc.constant(
                evoxc.from_numpy(
                    np.asarray([[np.pi, self.max_speed]], dtype=np.float32)
                )
            ),
        )

    def _advance(
        self,
        obs: Any,
        logit: Any,
        consts: _PendulumConstants,
    ) -> tuple[Any, Any]:
        """Advance the ``(B, 2)`` observations by one Euler step and score them.

        The row-shaped face of :meth:`_advance_rows`: it splits the observation
        row into its two state variables, runs the algebra, and re-joins the
        result, so it costs the three ops (two ``slice``, one ``concatenate``)
        that a caller holding an observation row necessarily pays.

        The rollout loop never calls it -- its carry holds the two state
        variables separately, so it calls :meth:`_advance_rows` directly and
        builds the row only for the policy query (one select) -- but the
        ``(B, 2)`` spelling is the environment's documented one-step contract
        (``obs``, ``logit`` in; next ``obs``, cost out) and is exercised as such.

        Args:
            obs: Current observations of shape ``(B, 2)``.
            logit: Policy output of shape ``(B, 1)``.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``(next_obs (B, 2), cost (B,))`` -- the transitioned observations and
            the (non-negative) cost of the state they came from, flattened to
            rank 1 so it can be subtracted from a rank-1 return accumulator
            (``(B,) - (B, 1)`` would broadcast into a ``(B, B)`` matrix).
        """
        theta, theta_dot = enp.split(obs, _OBS_DIM, axis=1)
        next_theta, next_theta_dot, cost = self._advance_rows(
            theta, theta_dot, logit, consts
        )
        return (
            enp.concatenate((next_theta, next_theta_dot), axis=1),
            enp.reshape(cost, (obs.shape[0],)),
        )

    def _advance_rows(
        self,
        theta: Any,
        theta_dot: Any,
        logit: Any,
        consts: _PendulumConstants,
    ) -> tuple[Any, Any, Any]:
        """Advance the two ``(B, 1)`` state leaves by one Euler step and score them.

        :meth:`_transition_rows` plus the angle integration, in Gym's order:
        with ``u = max_torque * tanh(logit)``, ``angle`` as in
        :meth:`_angle_normalize`, the step is exactly

        - ``cost = angle**2 + 0.1 * theta_dot**2 + 0.001 * u**2``, evaluated on
          the **pre-transition** state;
        - ``theta_dot' = clip(theta_dot + (gravity_term * sin(theta)
          + torque_term * u) * dt, -max_speed, max_speed)``: the velocity is
          integrated and clipped **before** the angle integrates;
        - ``theta' = theta + theta_dot' * dt``.

        The first two lines are :meth:`_transition_rows`, shared verbatim with
        the packed accelerator step so the two formulations cannot drift.

        The cost keeps the rank of the states it is computed from (``(B, 1)``),
        which is what lets a caller accumulating at that rank subtract it with
        one op; :meth:`_advance` flattens it for the row-shaped contract.

        Args:
            theta: Pendulum angle in radians, ``(B, 1)``.
            theta_dot: Angular velocity, ``(B, 1)``.
            logit: Policy output of shape ``(B, 1)``.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``(next_theta (B, 1), next_theta_dot (B, 1), cost (B, 1))``.
        """
        next_speed, cost = self._transition_rows(
            theta, theta_dot, logit, consts
        )
        next_theta = theta + next_speed * consts.dt
        return next_theta, next_speed, cost

    def _transition_rows(
        self,
        theta: Any,
        theta_dot: Any,
        logit: Any,
        consts: _PendulumConstants,
    ) -> tuple[Any, Any]:
        """Score the pre-transition state and integrate the velocity.

        The shared half of :meth:`_advance_rows`, in the exact operand order the
        environment has always used:

        - ``u = max_torque * tanh(logit)``;
        - ``cost = angle**2 + 0.1 * theta_dot**2 + 0.001 * u**2``, on the
          **pre-transition** state;
        - ``next_speed = clip(theta_dot + (gravity_term * sin(theta)
          + torque_term * u) * dt, -max_speed, max_speed)`` -- the clip happens
          here, i.e. before the angle integrates.

        The clip is ``enp.clip``, i.e. ``minimum``/``maximum`` against
        ``max_speed``/``neg_max_speed``, and both bounds are symbolic graph
        scalars like every other constant in the trace.

        Args:
            theta: Pendulum angle in radians, ``(B, 1)``.
            theta_dot: Angular velocity, ``(B, 1)``.
            logit: Policy output of shape ``(B, 1)``.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``(next_speed (B, 1), cost (B, 1))``.
        """
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
        return next_speed, cost

    def _angle_normalize(self, x: Any, consts: _PendulumConstants) -> Any:
        """Wrap an angle in radians into ``[-pi, pi)``.

        This is Gym's ``((x + pi) % (2 * pi)) - pi`` written as the floor-mod
        ``x - two_pi * floor((x + pi) / two_pi)``: subtracting ``pi`` after a
        non-negative remainder is the same as subtracting the whole turn from
        ``x`` itself. It costs two traced ops more than a ``remainder`` spelling
        (five against three), which buys the portability described below.

        ``evoxc.remainder`` is **not** a portable way to spell this: the numpy
        backend implements floor-mod, while ``stablehlo.remainder`` (and hence
        ``iree``) is ``fmod``, i.e. it carries the sign of the dividend, so a
        negative angle wrapped to ``-3pi`` instead of ``-pi`` and the cost
        ballooned. ``evoxc.floor`` is an exact IEEE op on every backend, so the
        floor form is the portable one; ``enp`` has no ``floor``/``remainder``,
        which is why the top-level ops are the escape hatch.

        Args:
            x: Angles in radians, any shape.
            consts: The bundle from :meth:`_constants`.

        Returns:
            The same-shape angle in ``[-pi, pi)``.
        """
        turns = evoxc.floor((x + consts.pi) / consts.two_pi)
        return x - consts.two_pi * turns
