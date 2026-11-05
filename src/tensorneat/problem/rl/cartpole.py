"""Batched CartPole rollout on evoxc: Gym dynamics, fixed horizon, return fitness."""

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

__all__ = ["CartPole"]

#: Number of state variables in an observation ``(x, x_dot, theta, theta_dot)``.
_OBS_DIM = 4
#: Number of logits a policy must emit for this problem.
_ACTION_DIM = 2


class _CartPoleConstants(NamedTuple):
    """The physics bundle of one ``CartPole`` configuration, as graph scalars.

    Every field is a rank-0 ``SymbolicTensor`` created inside the active trace
    with ``enp.full((), value)``, so no Python literal (a float would
    materialize as f64 on the stablehlo/iree path) ever reaches an op.

    The bundle is built **once per rollout, outside the timestep loop**, so a
    constant the dynamics needs every step (``four_thirds``, ``neg_force_mag``,
    the NaN scalar) is one Constant op per run instead of one per timestep --
    the reference interpreter pays per traced op, not per element.

    Attributes:
        gravity: Gravity constant.
        masscart: Cart mass.
        masspole: Pole mass.
        polemass_length: ``masspole * length``.
        total_mass: ``masscart + masspole``.
        length: Half-pole length.
        force_mag: Magnitude of the applied force.
        neg_force_mag: ``-force_mag``, the push-left force (precomputed so the
            loop never negates a constant).
        dt: Euler integration step.
        x_threshold: ``|x|`` limit.
        theta_threshold: ``|theta|`` limit in **radians**.
        reset_scale: Half-width of the uniform reset distribution.
        neg_reset_scale: ``-reset_scale``, the lower bound of the reset draw.
        four_thirds: The ``4/3`` of the pole-moment denominator.
        nan: A float32 NaN, the poison value accumulated into the return.
        one_step: The int32 ``1`` the loop counter advances by.
        horizon: The int32 ``max_step`` the loop counter is compared against.
    """

    gravity: Any
    masscart: Any
    masspole: Any
    polemass_length: Any
    total_mass: Any
    length: Any
    force_mag: Any
    neg_force_mag: Any
    dt: Any
    x_threshold: Any
    theta_threshold: Any
    reset_scale: Any
    neg_reset_scale: Any
    four_thirds: Any
    nan: Any
    one_step: Any
    horizon: Any


class CartPole(BaseProblem):
    """Classic Gym ``CartPole-v1`` with Euler integration, written in evoxc ops.

    No simulator is imported: the dynamics are the algebra below, executed as one
    ``evoxc.scan`` over a fixed horizon. Because a dead episode's state is
    frozen, an individual's return is exactly its survival time, so ``max_step``
    is a hard fitness ceiling and ``fitness_target`` must be ``<= max_step``.

    A genome used with this problem must emit two logits and must **not** apply
    an ``argmax``-style ``output_transform``: the problem owns the discrete
    decision, so that the branch stays a traced ``enp.greater`` instead of a
    host-side dtype conversion.

    Attributes:
        max_step: Number of timesteps simulated (the fitness ceiling).
        repeat_times: Independent initial states averaged per individual.
        gravity: Gravity constant (default ``9.8``).
        masscart: Cart mass (default ``1.0``).
        masspole: Pole mass (default ``0.1``).
        length: Half-pole length (default ``0.5``).
        force_mag: Magnitude of the applied force (default ``10.0``).
        dt: Integration timestep (default ``0.02``).
        x_threshold: Cart position limit (default ``2.4``).
        theta_threshold: Pole angle limit in **degrees** (default ``12.0``,
            i.e. ``0.2095`` rad).
        reset_scale: Reset states are ``U(-reset_scale, reset_scale)^4``
            (default ``0.05``).
        fitness_target: Optional fitness target used only for validation; a
            value above ``max_step`` is unreachable and is rejected, and so is
            one that is not a finite number.
    """

    def __init__(
        self,
        max_step: int = 500,
        repeat_times: int = 1,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        dt: float = 0.02,
        x_threshold: float = 2.4,
        theta_threshold: float = 12.0,
        reset_scale: float = 0.05,
        fitness_target: float | None = None,
    ) -> None:
        """Configure the environment's physics and rollout shape.

        The physics follows Gym's ``CartPole-v1``, with
        ``polemass_length = masspole * length`` and
        ``total_mass = masscart + masspole``. ``theta_threshold`` is given in
        degrees and converted to radians (``12`` deg is ``0.2095`` rad) before
        it enters the trace, so the kernel only ever compares radians.

        Args:
            max_step: Fixed scan horizon; also the fitness ceiling.
            repeat_times: Independent initial states averaged per individual.
            gravity: Gravity constant.
            masscart: Cart mass.
            masspole: Pole mass.
            length: Half-pole length.
            force_mag: Magnitude of the ``+/-`` force applied each step.
            dt: Euler integration step.
            x_threshold: ``|x|`` above this terminates the episode.
            theta_threshold: ``|theta|`` limit in **degrees**.
            reset_scale: Half-width of the uniform reset distribution.
            fitness_target: Optional fitness target; ``None`` disables the
                check. Because a frozen episode's return is its survival time,
                ``max_step`` is a hard ceiling and a larger target can never be
                reached.

        Raises:
            TensorNEATError: If ``max_step <= 0``, ``repeat_times <= 0``, any
                mass/length/dt/force is not finite and positive,
                ``theta_threshold`` is not finite and positive, or
                ``x_threshold``/``reset_scale`` is not finite and non-negative.
            TensorNEATError: If ``fitness_target`` is not a finite number, or is
                greater than ``max_step``.
        """
        self.max_step = require_positive_int("CartPole", "max_step", max_step)
        self.repeat_times = require_positive_int(
            "CartPole", "repeat_times", repeat_times
        )
        self.gravity = require_positive("CartPole", "gravity", gravity)
        self.masscart = require_positive("CartPole", "masscart", masscart)
        self.masspole = require_positive("CartPole", "masspole", masspole)
        self.length = require_positive("CartPole", "length", length)
        self.force_mag = require_positive("CartPole", "force_mag", force_mag)
        self.dt = require_positive("CartPole", "dt", dt)
        self.x_threshold = require_non_negative(
            "CartPole", "x_threshold", x_threshold
        )
        self.theta_threshold = require_positive(
            "CartPole", "theta_threshold", theta_threshold
        )
        self.reset_scale = require_non_negative(
            "CartPole", "reset_scale", reset_scale
        )
        if fitness_target is not None:
            target = require_finite("CartPole", "fitness_target", fitness_target)
            if target > self.max_step:
                raise TensorNEATError(
                    f"CartPole: fitness_target ({target}) exceeds max_step "
                    f"({self.max_step}), the hard fitness ceiling -- a frozen "
                    "episode's return is its survival time, so such a target can "
                    "never be reached"
                )
        self.fitness_target = fitness_target

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Observation shape ``(4,)`` = ``(x, x_dot, theta, theta_dot)``.

        Returns:
            The per-step observation shape.
        """
        return (_OBS_DIM,)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Action shape ``(2,)``: two logits.

        The cart is pushed right iff ``out[1] > out[0]``; a tie pushes left,
        which matches ``argmax`` index ``0``.

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
           ``evoxc.random.uniform(keys, (B, 4), low=-reset_scale,
           high=reset_scale)``; the key is not threaded through the loop and
           ``np.random`` is never used.
        3. One ``evoxc.while_loop`` over ``max_step`` timesteps, carry
           ``(obs (B, 4), alive (B,), total (B,), step ())``: the return
           accumulator rides in the carry instead of being gathered from a
           scan output stack, which is what keeps the per-timestep cost free of
           any ``B * max_step`` array traffic.
        4. Step body: forward the policy, convert the two logits into a
           ``(B, 1)`` ``+/-force_mag`` force, add ``alive`` to ``total``
           **before** the transition (the failing step still scores, as in
           Gym), integrate with Euler, mark death on ``|x| > x_threshold or
           |theta| > theta_threshold``, then freeze the state with
           ``enp.where(enp.expand_dims(alive, 1), next_obs, obs)`` (a traced
           tensor has no ``__getitem__``, so the rank-1 mask is broadcast with
           ``expand_dims``).
        5. Fitness = mean over the ``repeat_times`` axis of the episode return
           (a count in ``[0, max_step]`` per episode), then
           ``finalize_fitness``.

        A NaN logit still acts as "push left" inside the rollout (``enp.greater``
        is false against NaN), but the step accumulates a NaN into ``total``
        instead of a live reward, so any individual whose policy produced a NaN
        logit at any step gets a NaN fitness -- which survives the mean over
        ``repeat_times`` and is mapped to ``-inf`` by ``finalize_fitness``.

        Trace hygiene: every arithmetic constant is an in-trace
        ``enp.full((), v)`` so it stays float32 on the stablehlo/iree path, and
        all axes are non-negative.

        Args:
            state: Host container; unused, kept for protocol uniformity.
            keys: One evoxc key, used for the single reset draw.
            act_fn: Callable mapping ``(B, 4)`` observations to ``(B, 2)``
                logits.
            transformed: Batch-first execution schedule holding ``P``
                individuals.

        Returns:
            ``(P,)`` fitness = mean survival time, NaN mapped to ``-inf``.

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

        Host-side only: never called from a trace. Prints the survival length,
        the total return and the final state of one episode.

        The report runs its own graph of one individual -- one data argument
        (the rank-0 key) and ``pop_size = 1`` -- and reads both the return and
        the final observation out of the rollout carry, so no extra stacked
        output is needed. That graph is built with the options the run
        registered on ``state`` (``backend``/``device``/``compile_options``,
        see :func:`tensorneat.problem.base.build_options`), so
        ``Pipeline(backend="iree").show`` reports from the backend the training
        ran on, while a problem used directly -- a state registering no backend
        -- reports from the numpy reference interpreter. There is no silent
        fallback: a backend that cannot run the report raises.

        A NaN-poisoned policy is reported, not crashed on: the rollout
        deliberately accumulates NaN into the return (see :meth:`evaluate`), and
        a non-finite return has no integer survival length, so the step count is
        printed as ``n/a`` with the honest return value instead.

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
                ``(returns (R,), final_obs (R, 4))`` for the single individual.
            """
            total, final_obs = self._rollout(key_value, act_fn, 1)
            return enp.reshape(total, (self.repeat_times,)), final_obs

        executable = evoxc.build(report, key_spec, **build_options(state))
        returns, final_obs = evoxc.run(executable, key)
        returns = to_numpy(returns)
        final_obs = to_numpy(final_obs)
        for repeat in range(self.repeat_times):
            value = float(returns[repeat])
            x, x_dot, theta, theta_dot = (float(entry) for entry in final_obs[repeat])
            if math.isfinite(value):
                survived = f"survived {int(round(value))}/{self.max_step} steps"
            else:
                kind = "NaN" if math.isnan(value) else "non-finite"
                survived = f"survived n/a ({kind} return)"
            print(
                f"CartPole repeat {repeat}: {survived}, "
                f"return {value:.2f}"
            )
            print(
                f"\tfinal state: x={x:.4f}, x_dot={x_dot:.4f}, "
                f"theta={theta:.4f} rad, theta_dot={theta_dot:.4f}"
            )
        print(f"CartPole mean return: {float(np.mean(returns)):.2f}")
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
        ``max_step`` timesteps, carry ``(obs (B, 4), alive (B,), total (B,),
        step ())`` with ``B = pop_size * repeat_times`` in individual-major
        order.
        The policy query is prepared once for the whole rollout, so the per-pass
        operands are not re-derived per timestep.

        Args:
            keys: One rank-0 evoxc key for the single reset draw.
            act_fn: Callable mapping ``(B, 4)`` observations to ``(B, 2)``
                logits, or a prepared-query object exposing ``prepare``.
            pop_size: Number of individuals ``P`` (static).

        Returns:
            The pair ``(total (B,), final_obs (B, 4))``, where ``total`` is the
            accumulated reward of each episode (a count in ``[0, max_step]``,
            NaN once that episode's policy emitted a NaN logit) and
            ``final_obs`` the last observation the carry holds (the state before
            death once the episode is over).
        """
        consts = self._constants()
        batch = pop_size * self.repeat_times
        query = prepared_query(act_fn, batch)
        obs0 = evoxc.random.uniform(
            keys,
            (batch, _OBS_DIM),
            low=consts.neg_reset_scale,
            high=consts.reset_scale,
        )
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

    def _loop_cond(self, consts: _CartPoleConstants) -> Callable:
        """Build the trip condition of the rollout loop.

        The horizon is static config, so the loop runs exactly ``max_step``
        iterations: the condition counts up from ``0`` instead of testing
        whether any episode is still alive, which would make the cost depend on
        the population's behaviour and the reported per-horizon numbers
        meaningless.

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
        self, query: Callable, consts: _CartPoleConstants
    ) -> Callable:
        """Build the per-timestep body of the rollout loop.

        One step rewards the still-alive episodes, converts the policy's logits
        into a signed force, integrates, updates the alive mask and freezes the
        state of every episode that just died, so the carried types stay
        constant across iterations (what ``evoxc.while_loop`` requires).

        The return accumulator rides in the carry rather than being emitted as
        a scan output, so no ``(max_step, B)`` stack is ever allocated or
        scattered into; a NaN logit poisons the accumulator instead of a
        separate flag, which is why no post-loop poison reduction is needed.

        Args:
            query: The policy query, prepared once for the whole rollout by
                :func:`._common.prepared_query` so the per-pass operands are not
                re-derived per timestep; maps ``(B, 4)`` observations to
                ``(B, 2)`` logits.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``step(carry) -> (obs, alive, total, step)`` -- the next carry.
        """

        def step(carry):
            """Advance every episode by one timestep.

            Args:
                carry: ``(obs (B, 4), alive (B,), total (B,), step ())``.

            Returns:
                The next carry, same tree and leaf types.
            """
            obs, alive, total, index = carry
            logits = query(obs)
            force = self._force_from_logits(logits, consts)
            next_obs, dead = self._advance(obs, force, consts)
            reward = enp.where(
                self._nan_rows(logits),
                consts.nan,
                enp.astype(alive, evoxc.float32),
            )
            total = total + reward
            alive = enp.logical_and(alive, enp.logical_not(dead))
            obs = enp.where(enp.expand_dims(alive, 1), next_obs, obs)
            return (obs, alive, total, index + consts.one_step)

        return step

    def _constants(self) -> _CartPoleConstants:
        """Bundle the physics constants as float32 graph scalars.

        Must be called inside the trace: ``enp.full((), v)`` is a graph
        constructor, and a bare Python float would materialize as f64 on the
        stablehlo path and break iree type unification.

        The caller builds this bundle **once per rollout**, outside the timestep
        loop, so the derived scalars the loop needs every step
        (``neg_force_mag``, ``four_thirds``, ``nan``) cost one Constant op per
        run rather than one per timestep.

        Returns:
            A structure holding ``gravity``, ``masscart``, ``masspole``,
            ``polemass_length`` (``masspole * length``), ``total_mass``
            (``masscart + masspole``), ``length``, ``force_mag``,
            ``neg_force_mag``, ``dt``, ``x_threshold``, ``theta_threshold`` (in
            radians), ``reset_scale``, ``neg_reset_scale``, ``four_thirds``,
            ``nan`` (all float32), plus the int32 loop scalars ``one_step`` and
            ``horizon`` -- so neither the reset draw nor the trip condition ever
            sees an f64 constant.
        """
        return _CartPoleConstants(
            gravity=enp.full((), self.gravity),
            masscart=enp.full((), self.masscart),
            masspole=enp.full((), self.masspole),
            polemass_length=enp.full((), self.masspole * self.length),
            total_mass=enp.full((), self.masscart + self.masspole),
            length=enp.full((), self.length),
            force_mag=enp.full((), self.force_mag),
            neg_force_mag=enp.full((), -self.force_mag),
            dt=enp.full((), self.dt),
            x_threshold=enp.full((), self.x_threshold),
            theta_threshold=enp.full((), float(np.deg2rad(self.theta_threshold))),
            reset_scale=enp.full((), self.reset_scale),
            neg_reset_scale=enp.full((), -self.reset_scale),
            four_thirds=enp.full((), 4.0 / 3.0),
            nan=enp.full((), float("nan")),
            one_step=enp.full((), 1, dtype=evoxc.int32),
            horizon=enp.full((), self.max_step, dtype=evoxc.int32),
        )

    def _advance(self, obs: Any, force: Any, consts: Any) -> tuple[Any, Any]:
        """Advance the ``(B, 4)`` observations by one Euler step and test death.

        With ``c = cos(theta)``, ``s = sin(theta)`` and
        ``temp = (force + polemass_length * theta_dot**2 * s) / total_mass``,
        the four updates are

        - ``thetaacc = (gravity * s - c * temp) /
          (length * (4/3 - masspole * c**2 / total_mass))``
        - ``xacc = temp - polemass_length * thetaacc * c / total_mass``
        - ``x' = x + dt * x_dot``, ``theta' = theta + dt * theta_dot``
        - ``x_dot' = x_dot + dt * xacc``, ``theta_dot' = theta_dot + dt *
          thetaacc``

        The death test is folded in here because the transition already has
        ``x'`` and ``theta'`` in hand: testing them separately would split the
        same observation row a second time per timestep.

        Args:
            obs: Current observations of shape ``(B, 4)``.
            force: Applied force of shape ``(B, 1)``.
            consts: The bundle from ``_constants``.

        Returns:
            ``(next_obs (B, 4), dead (B,))`` -- the Euler step and the death
            mask of the *resulting* state.
        """
        x, x_dot, theta, theta_dot = enp.split(obs, _OBS_DIM, axis=1)
        cos = enp.cos(theta)
        sin = enp.sin(theta)
        temp = (
            force + consts.polemass_length * enp.square(theta_dot) * sin
        ) / consts.total_mass
        thetaacc = (consts.gravity * sin - cos * temp) / (
            consts.length
            * (consts.four_thirds - consts.masspole * enp.square(cos) / consts.total_mass)
        )
        xacc = temp - consts.polemass_length * thetaacc * cos / consts.total_mass
        next_x = x + consts.dt * x_dot
        next_theta = theta + consts.dt * theta_dot
        next_obs = enp.concatenate(
            (
                next_x,
                x_dot + consts.dt * xacc,
                next_theta,
                theta_dot + consts.dt * thetaacc,
            ),
            axis=1,
        )
        dead = enp.reshape(
            self._death_mask(next_x, next_theta, consts), (obs.shape[0],)
        )
        return next_obs, dead

    def _death_mask(self, x: Any, theta: Any, consts: Any) -> Any:
        """The failure predicate shared by ``_advance`` and ``_terminated``.

        An episode is dead when ``|x| > x_threshold`` or ``|theta| >
        theta_threshold`` (radians), both strict.

        Args:
            x: Cart position of shape ``(B, 1)``.
            theta: Pole angle in radians of shape ``(B, 1)``.
            consts: The bundle from ``_constants``.

        Returns:
            ``(B, 1)`` boolean mask, ``True`` where the episode has failed.
        """
        too_far = enp.greater(enp.abs(x), consts.x_threshold)
        too_tilted = enp.greater(enp.abs(theta), consts.theta_threshold)
        return enp.logical_or(too_far, too_tilted)

    def _terminated(self, obs: Any, consts: Any) -> Any:
        """Return the death mask of an observation.

        An episode is dead when ``|x| > x_threshold`` or ``|theta| >
        theta_threshold`` (radians). The rollout does not call this on its own
        transitions -- :meth:`_advance` tests the components it already has --
        but the predicate stays available (and rank-1, as documented) for
        host-side introspection and tests.

        Args:
            obs: Observations of shape ``(B, 4)``.
            consts: The bundle from ``_constants``.

        Returns:
            ``(B,)`` boolean mask, ``True`` where the episode has failed.
        """
        x, _, theta, _ = enp.split(obs, _OBS_DIM, axis=1)
        return enp.reshape(
            self._death_mask(x, theta, consts), (obs.shape[0],)
        )

    def _force_from_logits(self, logits: Any, consts: Any) -> Any:
        """Turn two logits per row into a signed force.

        The decision is ``enp.split`` + ``enp.greater`` (push right iff
        ``out[1] > out[0]``, a tie pushes left), so there is no integer or bool
        control flow and no ``argmax`` dtype handling.

        A NaN logit compares false against everything, so it selects the push
        left branch; :meth:`evaluate` poisons the individual's fitness instead
        of letting that happen silently.

        Args:
            logits: Policy outputs of shape ``(B, 2)``.
            consts: The bundle from ``_constants``.

        Returns:
            Force of shape ``(B, 1)``, each row ``+force_mag`` or
            ``-force_mag``.
        """
        left, right = enp.split(logits, _ACTION_DIM, axis=1)
        push_right = enp.greater(right, left)
        return enp.where(push_right, consts.force_mag, consts.neg_force_mag)

    def _nan_rows(self, logits: Any) -> Any:
        """Mark the rows of ``logits`` that contain a NaN.

        ``evoxc`` has no ``isnan`` in the traced-op namespace, so the test comes
        from the top-level ``evoxc.isnan``, applied to the row sum: NaN is
        absorbing under addition, so a row sum is NaN iff either logit is NaN.
        That is one reduce plus one unary op, where splitting the NaN mask into
        two columns and folding them costs two slices, a boolean or and a
        reshape.

        Args:
            logits: Policy outputs of shape ``(B, 2)``.

        Returns:
            ``(B,)`` boolean mask, ``True`` where either logit is NaN.
        """
        return evoxc.isnan(enp.sum(logits, axis=1))

