"""Batched ``MountainCarContinuous-v0`` on evoxc: Gym dynamics, fixed horizon."""

import math
from typing import Any, Callable, NamedTuple

import numpy as np

import evoxc
import evoxc.numpy as enp

from ..base import BaseProblem, build_options, finalize_fitness, static_batch_size
from ...common.errors import TensorNEATError
from ._common import (
    prepared_query,
    require_finite,
    require_non_negative,
    require_positive,
    require_positive_int,
    returns_of,
    to_numpy,
)

__all__ = ["ATTAINABLE_CEILING", "MountainCarContinuous"]

#: Message prefix shared by every host-side validation error of this problem.
_OWNER = "MountainCarContinuous"

#: Number of state variables in an observation ``(position, velocity)``.
_OBS_DIM = 2
#: Number of logits a policy must emit for this problem.
_ACTION_DIM = 1

#: Lower bound of Gym's reset distribution for the car position, ``U(-0.6, -0.4)``.
_RESET_LOW = -0.6
#: Upper bound of Gym's reset distribution for the car position.
_RESET_HIGH = -0.4

#: Highest return a reaching policy can be defended to attain: ``93.0``.
#:
#: ``100`` is the **nominal** bonus, not an attainable return, because the
#: ``0.1 * a**2`` penalty is charged on the pre-clip action of *every* scoring
#: step of the climb and the bonus is paid once. Derivation, with
#: ``power = 0.0015``, ``gravity = 0.0025``, ``max_speed = 0.07`` and
#: ``goal_position = 0.45``:
#:
#: * the per-step velocity gain is at most ``power + gravity = 0.004`` (full
#:   force with the slope term at its most favourable value,
#:   ``cos(3 * position) = -1``), so reaching the ``max_speed`` clip from rest
#:   takes at least ``ceil(0.07 / 0.004) = 18`` scored steps;
#: * the shortest remaining travel is from the highest reset position
#:   (``_RESET_HIGH = -0.4``) to the goal, ``0.85``, at no more than
#:   ``max_speed = 0.07`` per step, i.e. at least ``ceil(0.85 / 0.07) = 13``
#:   more scored steps, each charged up to ``0.1 * 1**2 = 0.1``;
#: * a real trajectory cannot hold that idealised ramp: the slope term
#:   oscillates, the car has to pump, and the measured climb of a solving
#:   policy is ~106 steps, at which the accumulated penalty is ~10.6.
#:
#: The best reaching policy measured on this box -- the bang-bang
#: ``a = sign(velocity)`` at ``P = 8`` -- returns ``90.8`` (min ``89.1``) at both
#: ``max_step = 999`` and ``max_step = 200``, so no policy can approach ``100``
#: and this constant is a small margin over the measured best: a
#: ``fitness_target`` above it validates as "reachable" and then never
#: terminates.
ATTAINABLE_CEILING = 93.0


class _MountainCarConstants(NamedTuple):
    """The physics bundle of one ``MountainCarContinuous`` configuration.

    Every field is a rank-0 ``SymbolicTensor`` created inside the active trace
    with ``enp.full((), value)``, so no Python literal (a float would
    materialize as f64 on the stablehlo/iree path) ever reaches an op.

    The bundle is built **once per rollout, outside the timestep loop**, so a
    constant the dynamics needs every step (``power``, ``three``, the ``+/-1``
    clip bounds) is one Constant op per run instead of one per timestep -- the
    reference interpreter pays per traced op, not per element.

    Attributes:
        min_position: Left end of the track (``-1.2``), the lower position clip.
        max_position: Right end of the track (``0.6``), the upper position clip.
        max_speed: Velocity clip bound (``0.07``).
        neg_max_speed: ``-max_speed``, the lower velocity clip bound.
        goal_position: Position at or above which the goal counts (``0.45``).
        goal_velocity: Velocity at or above which the goal counts (``0.0``).
        power: Force-to-acceleration factor per unit action (``0.0015``).
        gravity: Gravity constant of the slope term (``0.0025``).
        three: The ``3`` of the ``cos(3 * position)`` slope factor.
        penalty: The ``0.1`` charged per step on ``a**2``.
        goal_reward: The one-time ``100`` terminus bonus.
        zero: A float32 ``0``.
        one: A float32 ``1``, the upper action clip bound.
        neg_one: A float32 ``-1``, the lower action clip bound.
        false: A bool ``False``, the value the alive mask is reset to once the
            goal has been reached.
        reset_center: Centre of the reset draw, ``-0.5``.
        reset_half_width: Half-width of the reset draw, ``0.1``.
        one_step: The int32 ``1`` the loop counter advances by.
        horizon: The int32 ``max_step`` the loop counter is compared against.
    """

    min_position: Any
    max_position: Any
    max_speed: Any
    neg_max_speed: Any
    goal_position: Any
    goal_velocity: Any
    power: Any
    gravity: Any
    three: Any
    penalty: Any
    goal_reward: Any
    zero: Any
    one: Any
    neg_one: Any
    false: Any
    reset_center: Any
    reset_half_width: Any
    one_step: Any
    horizon: Any


class MountainCarContinuous(BaseProblem):
    """Gym ``MountainCarContinuous-v0`` with Euler integration, written in evoxc ops.

    No simulator is imported: the dynamics are the algebra below, executed as one
    ``evoxc.while_loop`` over a fixed horizon. The car starts at rest inside the
    valley (``position ~ U(-0.6, -0.4)``, ``velocity = 0``) and has to pump
    enough speed to climb to ``goal_position``; the episode ends the first time
    the car is at the goal, and the return is the one-time ``100`` bonus minus
    ``0.1 * a**2`` for every step that scored.

    A genome used with this problem emits **one** logit and must leave
    ``output_transform`` unset (identity): the problem owns the squashing
    ``a = tanh(logit) in (-1, 1)``, so the action stays a traced elementwise op.

    Termination differs from ``CartPole``: there, dying is a failure and the
    failure step is the last one that scores, while here the terminal transition
    is a *success* that scores the ``100``. Afterwards the state is frozen (so
    the frozen post-goal state keeps satisfying ``done``) and no reward is
    accumulated any more: the bonus is masked by the pre-step alive flag, since
    unmasked it would be re-awarded on every remaining timestep.

    Attributes:
        max_step: Number of timesteps simulated, Gym's episode limit by default
            (``999``).
        repeat_times: Independent initial states averaged per individual.
        min_position: Left end of the track (default ``-1.2``).
        max_position: Right end of the track (default ``0.6``).
        max_speed: Velocity clip bound (default ``0.07``).
        goal_position: Position at or above which the goal counts (``0.45``).
        goal_velocity: Velocity at or above which the goal counts (``0.0``).
        power: Acceleration per unit action (default ``0.0015``).
        gravity: Slope gravity constant (default ``0.0025``).
        fitness_target: Optional fitness target used only for validation; a
            value above :data:`ATTAINABLE_CEILING` (``93.0``) is unreachable and
            is rejected, and so is one that is not a finite number.
    """

    def __init__(
        self,
        max_step: int = 999,
        repeat_times: int = 1,
        min_position: float = -1.2,
        max_position: float = 0.6,
        max_speed: float = 0.07,
        goal_position: float = 0.45,
        goal_velocity: float = 0.0,
        power: float = 0.0015,
        gravity: float = 0.0025,
        fitness_target: float | None = None,
    ) -> None:
        """Configure the environment's physics and rollout shape.

        The defaults are Gym's ``MountainCarContinuous-v0`` exactly, including
        the ``999``-step time limit and ``goal_velocity = 0``.

        The geometry is validated here, not inside the kernel: it is host-side
        configuration, and a track of zero width, a goal outside the clipped part
        of the track (so it can never be reached) or a goal velocity above
        ``max_speed`` would make every episode fail silently.

        Args:
            max_step: Fixed rollout horizon; also the penalty ceiling.
            repeat_times: Independent initial states averaged per individual.
            min_position: Left end of the track.
            max_position: Right end of the track.
            max_speed: Velocity clip bound.
            goal_position: Position at or above which the goal counts.
            goal_velocity: Velocity at or above which the goal counts; the
                Gym default ``0.0`` also accepts a car rolling forward.
            power: Acceleration per unit action.
            gravity: Slope gravity constant.
            fitness_target: Optional fitness target; ``None`` disables the
                check. The one-time goal bonus is ``100``, but it is paid on a
                single step while the ``0.1 * a**2`` penalty is charged on every
                scoring step of the climb, so an attaining policy returns about
                ``89``-``91``: a target above :data:`ATTAINABLE_CEILING`
                (``93.0``) can never be reached and is rejected. A target below
                it can still be out of reach on a short horizon -- a
                structurally minimal climb needs ``ceil(max_speed /
                (power + gravity)) + ceil((goal_position - _RESET_HIGH) /
                max_speed) = 18 + 13 = 31`` scoring steps, a reaching policy
                needs ~106, so pick ``max_step`` well above that for a positive
                target.

        Raises:
            TensorNEATError: If ``max_step`` or ``repeat_times`` is not a
                positive int, ``max_speed``/``power`` is not a finite, positive
                number, ``gravity`` is not a finite, non-negative number, or any
                of ``min_position``/``max_position``/``goal_position``/
                ``goal_velocity`` is not a finite number.
            TensorNEATError: If ``min_position >= max_position``, if
                ``goal_position`` lies outside ``[min_position, max_position]``,
                or if ``goal_velocity > max_speed``.
            TensorNEATError: If ``fitness_target`` is not a finite number, or is
                greater than :data:`ATTAINABLE_CEILING`.
        """
        self.max_step = require_positive_int(_OWNER, "max_step", max_step)
        self.repeat_times = require_positive_int(_OWNER, "repeat_times", repeat_times)
        self.min_position = require_finite(_OWNER, "min_position", min_position)
        self.max_position = require_finite(_OWNER, "max_position", max_position)
        self.max_speed = require_positive(_OWNER, "max_speed", max_speed)
        self.goal_position = require_finite(_OWNER, "goal_position", goal_position)
        self.goal_velocity = require_finite(_OWNER, "goal_velocity", goal_velocity)
        self.power = require_positive(_OWNER, "power", power)
        self.gravity = require_non_negative(_OWNER, "gravity", gravity)
        if self.min_position >= self.max_position:
            raise TensorNEATError(
                f"{_OWNER}: min_position ({self.min_position}) must be below "
                f"max_position ({self.max_position}) -- the position is clipped "
                "to that interval, so a track of zero (or negative) width leaves "
                "the car unable to move"
            )
        if not self.min_position <= self.goal_position <= self.max_position:
            raise TensorNEATError(
                f"{_OWNER}: goal_position ({self.goal_position}) lies outside "
                f"[min_position, max_position] = [{self.min_position}, "
                f"{self.max_position}] -- the position is clipped to that "
                "interval, so such a goal can never be reached"
            )
        if self.goal_velocity > self.max_speed:
            raise TensorNEATError(
                f"{_OWNER}: goal_velocity ({self.goal_velocity}) exceeds "
                f"max_speed ({self.max_speed}) -- the velocity is clipped to "
                "[-max_speed, max_speed], so the goal could never be reached"
            )
        if fitness_target is not None:
            target = require_finite(_OWNER, "fitness_target", fitness_target)
            if target > ATTAINABLE_CEILING:
                raise TensorNEATError(
                    f"{_OWNER}: fitness_target ({target}) exceeds the attainable "
                    f"ceiling {ATTAINABLE_CEILING} -- the nominal 100 goal bonus "
                    "is paid once, on the single step that reaches the goal, "
                    "while the 0.1 * a**2 penalty is charged on the pre-clip "
                    "action a in (-1, 1) of every scoring step of the ~106-step "
                    "climb, so a reaching policy returns about 89-91 (the best "
                    f"measured one returns 90.8) and no target above "
                    f"{ATTAINABLE_CEILING} can ever be reached"
                )
        self.fitness_target = fitness_target

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Observation shape ``(2,)`` = ``(position, velocity)``.

        Returns:
            The per-step observation shape.
        """
        return (_OBS_DIM,)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Action shape ``(1,)``: one logit.

        The problem squashes the logit with ``tanh`` into a continuous force in
        ``(-1, 1)``, so a genome used here must leave ``output_transform``
        unset (identity) -- an ``argmax`` or threshold transform would destroy
        the continuous control this environment needs.

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
        2. Initial states come from **one** draw, ``U(-1, 1)`` of shape
           ``(B, 1)``, rescaled to ``position = -0.5 + 0.1 * raw`` (i.e.
           ``U(-0.6, -0.4)``) and concatenated with a zero velocity; the key is
           not threaded through the loop and ``np.random`` is never used.
        3. One ``evoxc.while_loop`` over ``max_step`` timesteps, carry
           ``(obs (B, 2), alive (B,), total (B,), step ())``: the return
           accumulator rides in the carry instead of being gathered from a
           scan output stack, which keeps the per-timestep cost free of any
           ``B * max_step`` array traffic.
        4. Step body: forward the policy, squash the single logit with ``tanh``,
           integrate one Euler step in Gym's exact order, accumulate the
           reward of the still-alive episodes, update the alive mask and freeze
           the state of every episode that just reached the goal.
        5. Fitness = mean over the ``repeat_times`` axis of the episode return
           (in ``[-(0.1 * max_step), 100]`` per episode), then ``finalize_fitness``.

        NaN handling: ``a = tanh(NaN) = NaN``, so the clip does not fold it
        (numpy ``clip`` is ``minimum(maximum(...))``, and both propagate NaN),
        the ``0.1 * a**2`` penalty is NaN, and the reward of that live step is
        NaN, which poisons ``total`` for the rest of the rollout. The return is
        then NaN and ``finalize_fitness`` maps it to ``-inf``, so no separate
        NaN census is needed. A NaN arriving only *after* an episode finished is
        dropped, because accumulation stops once ``alive`` is false -- that is
        intended: the episode is over and already scored.

        Args:
            state: Host container; unused, kept for protocol uniformity.
            keys: One evoxc key, used for the single reset draw.
            act_fn: Callable mapping ``(B, 2)`` observations to ``(B, 1)``
                logits.
            transformed: Batch-first execution schedule holding ``P``
                individuals.

        Returns:
            ``(P,)`` fitness = mean episode return, NaN mapped to ``-inf``.

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

        Host-side only: never called from a trace. Builds one graph of one
        individual -- one data argument (the rank-0 key) and ``pop_size = 1`` --
        and reads both the return and the final observation out of the rollout
        carry, so no extra stacked output is needed.

        The goal indicator is ``return > 0`` rather than a test against the
        ``100`` bonus: a successful episode pays the ``100`` **once** but is
        charged ``0.1 * a**2`` on every step it took to get there, so a reaching
        policy (about 106 steps from rest) returns roughly ``89``-``91``, not
        ``100`` -- which is why :data:`ATTAINABLE_CEILING` is ``93`` and not
        ``100``. A failed episode's return is at most ``0`` (the bonus was never
        paid), so a positive return means the bonus was paid. The indicator is
        exact for any horizon below ``1000`` steps (the default is ``999``) and
        can only under-report above it. A non-finite return (a NaN-poisoned
        policy) compares false, so the goal reads as ``n/a`` with the honest
        return value instead of claiming the goal was missed.

        The reported final state is the one the carry holds: for an episode that
        reached the goal that is the state *before* the terminal transition
        (the same freeze rule as ``CartPole``), so its position sits just short
        of ``goal_position``.

        The report graph is built with the options the run registered on
        ``state`` (``backend``/``device``/``compile_options``, see
        :func:`tensorneat.problem.base.build_options`), so
        ``Pipeline(backend="iree").show`` reports from the backend the training
        ran on, while a problem used directly -- a state registering no backend
        -- reports from the numpy reference interpreter. There is no silent
        fallback: a backend that cannot run the report raises.

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

        executable = evoxc.build(report, key_spec, **build_options(state))
        returns, final_obs = evoxc.run(executable, key)
        returns = to_numpy(returns)
        final_obs = to_numpy(final_obs)
        for repeat in range(self.repeat_times):
            position, velocity = (float(entry) for entry in final_obs[repeat])
            value = float(returns[repeat])
            if math.isfinite(value):
                goal = "reached" if value > 0.0 else "not reached"
            else:
                kind = "NaN" if math.isnan(value) else "non-finite"
                goal = f"n/a ({kind} return)"
            print(
                f"MountainCarContinuous repeat {repeat}: return "
                f"{value:.2f}, goal {goal}"
            )
            print(
                f"\tfinal state: position={position:.4f}, "
                f"velocity={velocity:.4f}"
            )
        print(f"MountainCarContinuous mean return: {float(np.mean(returns)):.2f}")
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
        ``max_step`` timesteps, carry ``(obs (B, 2), alive (B,), total (B,),
        step ())`` with ``B = pop_size * repeat_times`` in individual-major
        order.
        The policy query is prepared once for the whole rollout, so the per-pass
        operands are not re-derived per timestep.

        Args:
            keys: One rank-0 evoxc key for the single reset draw.
            act_fn: Callable mapping ``(B, 2)`` observations to ``(B, 1)``
                logits, or a prepared-query object exposing ``prepare``.
            pop_size: Number of individuals ``P`` (static).

        Returns:
            The pair ``(total (B,), final_obs (B, 2))``, where ``total`` is the
            accumulated reward of each episode (in ``[-(0.1 * max_step), 100]``,
            NaN once that episode's policy emitted a NaN logit) and
            ``final_obs`` the last observation the carry holds (the state before
            the terminal transition once the episode is over).
        """
        consts = self._constants()
        batch = pop_size * self.repeat_times
        query = prepared_query(act_fn, batch)
        obs0 = self._reset(keys, consts, batch)
        init = (
            obs0,
            enp.ones((batch,), dtype=evoxc.bool_),
            enp.zeros((batch,), dtype=evoxc.float32),
            enp.zeros((), dtype=evoxc.int32),
        )
        obs, _, total, _ = evoxc.while_loop(
            self._loop_cond(consts),
            self._loop_body(query, consts),
            init,
        )
        return total, obs

    def _reset(
        self,
        keys: Any,
        consts: _MountainCarConstants,
        batch: int,
    ) -> Any:
        """Draw the ``(B, 2)`` initial observations.

        Gym's reset is ``position ~ U(-0.6, -0.4)`` and ``velocity = 0``. It is
        built from **one** rank-0 key and one draw, ``raw ~ U(-1, 1)`` of shape
        ``(B, 1)``, rescaled column-wise to ``reset_center + reset_half_width *
        raw``; the velocity column is an exact zero block, and the two columns
        are joined with ``enp.concatenate(axis=1)`` (a traced tensor has no
        ``__getitem__``, so no indexing and no assignment).

        Args:
            keys: One rank-0 evoxc key.
            consts: The bundle from :meth:`_constants`.
            batch: Number of episodes ``B = P * repeat_times``.

        Returns:
            ``(B, 2)`` observations ``(position, velocity)``.
        """
        raw = evoxc.random.uniform(
            keys, (batch, 1), low=consts.neg_one, high=consts.one
        )
        position = consts.reset_center + consts.reset_half_width * raw
        velocity = enp.zeros((batch, 1), dtype=evoxc.float32)
        return enp.concatenate((position, velocity), axis=1)

    def _loop_cond(self, consts: _MountainCarConstants) -> Callable:
        """Build the trip condition of the rollout loop.

        The horizon is static config, so the loop runs exactly ``max_step``
        iterations: the condition counts up from ``0`` instead of testing
        whether any episode is still alive, which would make the cost depend on
        the population's behaviour and would let a NaN logit arriving after
        every episode finished escape the accumulator.

        Args:
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``cond(carry) -> () bool`` -- true while ``step < max_step``.
        """

        def cond(carry):
            """Test the loop counter against the static horizon.

            Args:
                carry: ``(obs, alive, total, step)``.

            Returns:
                The rank-0 boolean ``step < max_step``.
            """
            return enp.less(carry[3], consts.horizon)

        return cond

    def _loop_body(
        self, query: Callable, consts: _MountainCarConstants
    ) -> Callable:
        """Build the per-timestep body of the rollout loop.

        One step squashes the policy's single logit into an action, integrates
        one Euler step, rewards the episodes that were still alive *before* the
        transition, updates the alive mask and freezes the state of every
        episode that just reached the goal, so the carried types stay constant
        across iterations (what ``evoxc.while_loop`` requires).

        The reward of a step is ``100`` when the transition reached the goal
        minus ``0.1 * a**2``, and it is masked by the **pre-step** alive flag:
        the frozen post-goal state still satisfies ``done``, so an unmasked
        bonus would be re-awarded on every remaining timestep.

        Rank hygiene: ``done`` comes back rank-1 from :meth:`_advance`, and the
        action is reshaped to rank-1 before it is squared, so no ``(B, 1)``
        operand ever meets the ``(B,)`` accumulator (numpy-style broadcasting
        would silently produce a ``(B, B)`` total).

        The alive mask is retired with one ``enp.where(done, consts.false,
        alive)`` -- the single op that says "an episode that just reached the
        goal is no longer live" -- instead of a ``logical_not`` feeding a
        ``logical_and`` (two ops for the same value). ``consts.false`` is the
        in-trace rank-0 bool ``False`` from :meth:`_constants`; a Python
        ``False`` or a numpy bool scalar is a concrete value rather than a graph
        operand, and ``enp.where`` would promote the result away from ``bool``
        instead of returning the mask at its own dtype.

        Args:
            query: The policy query, prepared once for the whole rollout by
                :func:`._common.prepared_query` so the per-pass operands are not
                re-derived per timestep; maps ``(B, 2)`` observations to
                ``(B, 1)`` logits.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``step(carry) -> (obs, alive, total, step)`` -- the next carry.
        """

        def step(carry):
            """Advance every episode by one timestep.

            Args:
                carry: ``(obs (B, 2), alive (B,), total (B,), step ())``.

            Returns:
                The next carry, same tree and leaf types.
            """
            obs, alive, total, index = carry
            logits = query(obs)
            action = self._action_from_logits(logits)
            next_obs, done = self._advance(obs, action, consts)
            bonus = enp.where(done, consts.goal_reward, consts.zero)
            penalty = consts.penalty * enp.square(
                enp.reshape(action, (obs.shape[0],))
            )
            reward = bonus - penalty
            total = total + enp.where(alive, reward, consts.zero)
            alive = enp.where(done, consts.false, alive)
            obs = enp.where(enp.expand_dims(alive, 1), next_obs, obs)
            return (obs, alive, total, index + consts.one_step)

        return step

    def _constants(self) -> _MountainCarConstants:
        """Bundle the physics constants as float32 graph scalars.

        Must be called inside the trace: ``enp.full((), v)`` is a graph
        constructor, and a bare Python float would materialize as f64 on the
        stablehlo path and break iree type unification.

        The caller builds this bundle **once per rollout**, outside the timestep
        loop, so the constants the loop needs every step (``power``, ``three``,
        ``penalty``, ``goal_reward``, ``zero``, ``false``, the clip bounds) cost
        one Constant op per run rather than one per timestep.

        ``reset_center`` and ``reset_half_width`` are derived from the module
        literals ``_RESET_LOW``/``_RESET_HIGH`` so the reset draw cannot drift
        from Gym's ``U(-0.6, -0.4)``: ``0.5 * (low + high) = -0.5`` and
        ``0.5 * (high - low) = 0.1``, both exact in float32.

        Returns:
            A structure holding ``min_position``, ``max_position``,
            ``max_speed``, ``neg_max_speed``, ``goal_position``,
            ``goal_velocity``, ``power``, ``gravity``, ``three``, ``penalty``,
            ``goal_reward``, ``zero``, ``one``, ``neg_one``, ``reset_center``,
            ``reset_half_width`` (all float32), the bool ``false`` the alive mask
            is reset to, plus the int32 loop scalars ``one_step`` and ``horizon``
            -- so neither the reset draw nor the trip condition ever sees an f64
            constant.
        """
        return _MountainCarConstants(
            min_position=enp.full((), self.min_position),
            max_position=enp.full((), self.max_position),
            max_speed=enp.full((), self.max_speed),
            neg_max_speed=enp.full((), -self.max_speed),
            goal_position=enp.full((), self.goal_position),
            goal_velocity=enp.full((), self.goal_velocity),
            power=enp.full((), self.power),
            gravity=enp.full((), self.gravity),
            three=enp.full((), 3.0),
            penalty=enp.full((), 0.1),
            goal_reward=enp.full((), 100.0),
            zero=enp.full((), 0.0),
            one=enp.full((), 1.0),
            neg_one=enp.full((), -1.0),
            false=enp.zeros((), dtype=evoxc.bool_),
            reset_center=enp.full((), 0.5 * (_RESET_LOW + _RESET_HIGH)),
            reset_half_width=enp.full((), 0.5 * (_RESET_HIGH - _RESET_LOW)),
            one_step=enp.full((), 1, dtype=evoxc.int32),
            horizon=enp.full((), self.max_step, dtype=evoxc.int32),
        )

    def _action_from_logits(self, logits: Any) -> Any:
        """Squash the policy's single logit into an action in ``(-1, 1)``.

        ``a = tanh(logit)`` is an elementwise op, so no ``clip`` or ``argmax``
        dtype handling is needed: the problem owns the squashing and the genome
        is expected to emit a raw logit (``output_transform`` unset).

        The returned action is the **pre-clip** action, which is what Gym
        charges the ``0.1 * a**2`` penalty on; :meth:`_advance` clips it to
        ``[-1, 1]`` before it becomes a force, which ``|tanh| <= 1`` makes a
        no-op kept for faithfulness with the reference implementation.

        Args:
            logits: Policy outputs of shape ``(B, 1)``.

        Returns:
            Actions of shape ``(B, 1)``, each in ``(-1, 1)``.
        """
        return enp.tanh(logits)

    def _advance(self, obs: Any, action: Any, consts: Any) -> tuple[Any, Any]:
        """Advance the ``(B, 2)`` observations by one Euler step and test the goal.

        Gym's exact ordering, all on the **pre-transition** state:

        - ``force = clip(a, -1, 1)``
        - ``velocity' = velocity + force * power - gravity * cos(3 * position)``
        - ``velocity' = clip(velocity', -max_speed, max_speed)``
        - ``position' = clip(position + velocity', min_position, max_position)``
        - ``velocity' = 0`` where ``position' == min_position and velocity' < 0``
          (the car came to rest against the left wall; the equality is exact
          because ``clip`` returns the bound verbatim)
        - ``done = position' >= goal_position and velocity' >= goal_velocity``

        The goal test is folded in here because the transition already has
        ``position'`` and ``velocity'`` in hand: testing them separately would
        split the same observation row a second time per timestep.

        Args:
            obs: Current observations of shape ``(B, 2)``.
            action: Pre-clip actions of shape ``(B, 1)``.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``(next_obs (B, 2), done (B,))`` -- the Euler step and the goal mask
            of the *resulting* state, already rank-1 so it can be combined with
            the ``(B,)`` accumulator.
        """
        position, velocity = enp.split(obs, _OBS_DIM, axis=1)
        # ``enp.clip`` is ``maximum(minimum(x, a_max), a_min)``, so the symbolic
        # bundle scalars are legal bounds: no Python literal and no f64 constant
        # is involved, and NaN propagates through both ops (what the NaN policy
        # of ``evaluate`` relies on).
        force = enp.clip(action, consts.neg_one, consts.one)
        next_velocity = (
            velocity
            + force * consts.power
            - consts.gravity * enp.cos(consts.three * position)
        )
        next_velocity = enp.clip(
            next_velocity, consts.neg_max_speed, consts.max_speed
        )
        next_position = enp.clip(
            position + next_velocity, consts.min_position, consts.max_position
        )
        next_velocity = enp.where(
            enp.logical_and(
                enp.equal(next_position, consts.min_position),
                enp.less(next_velocity, consts.zero),
            ),
            consts.zero,
            next_velocity,
        )
        next_obs = enp.concatenate((next_position, next_velocity), axis=1)
        done = enp.reshape(
            self._goal_mask(next_position, next_velocity, consts), (obs.shape[0],)
        )
        return next_obs, done

    def _goal_mask(self, position: Any, velocity: Any, consts: Any) -> Any:
        """The goal predicate shared by ``_advance`` and ``_terminated``.

        An episode is done when ``position >= goal_position`` **and**
        ``velocity >= goal_velocity``, both non-strict, exactly as Gym's
        ``terminated``. Comparisons against NaN are false, so a NaN-poisoned
        rollout never terminates and keeps accumulating its NaN penalty.

        Args:
            position: Car position of shape ``(B, 1)``.
            velocity: Car velocity of shape ``(B, 1)``.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``(B, 1)`` boolean mask, ``True`` where the goal has been reached.
        """
        reached = enp.greater_equal(position, consts.goal_position)
        fast_enough = enp.greater_equal(velocity, consts.goal_velocity)
        return enp.logical_and(reached, fast_enough)

    def _terminated(self, obs: Any, consts: Any) -> Any:
        """Return the goal mask of an observation.

        ``_advance`` does not call this on its own transitions -- it tests the
        components it already has -- but the predicate stays available (and
        rank-1, as documented) for host-side introspection and tests.

        Args:
            obs: Observations of shape ``(B, 2)``.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``(B,)`` boolean mask, ``True`` where the goal has been reached.
        """
        position, velocity = enp.split(obs, _OBS_DIM, axis=1)
        return enp.reshape(
            self._goal_mask(position, velocity, consts), (obs.shape[0],)
        )
