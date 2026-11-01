"""Batched CartPole rollout on evoxc: Gym dynamics, fixed horizon, return fitness."""

from typing import Any, Callable, NamedTuple

import numpy as np

import evoxc
import evoxc.numpy as enp

from ..base import BaseProblem, finalize_fitness, static_batch_size
from ...common.errors import TensorNEATError

__all__ = ["CartPole"]

#: Number of state variables in an observation ``(x, x_dot, theta, theta_dot)``.
_OBS_DIM = 4
#: Number of logits a policy must emit for this problem.
_ACTION_DIM = 2


class _CartPoleConstants(NamedTuple):
    """The physics bundle of one ``CartPole`` configuration, as graph scalars.

    Every field is a float32 rank-0 ``SymbolicTensor`` created inside the
    active trace with ``enp.full((), value)``, so no Python float (which would
    materialize as f64 on the stablehlo/iree path) ever reaches an op.

    Attributes:
        gravity: Gravity constant.
        masscart: Cart mass.
        masspole: Pole mass.
        polemass_length: ``masspole * length``.
        total_mass: ``masscart + masspole``.
        length: Half-pole length.
        force_mag: Magnitude of the applied force.
        dt: Euler integration step.
        x_threshold: ``|x|`` limit.
        theta_threshold: ``|theta|`` limit in **radians**.
        reset_scale: Half-width of the uniform reset distribution.
    """

    gravity: Any
    masscart: Any
    masspole: Any
    polemass_length: Any
    total_mass: Any
    length: Any
    force_mag: Any
    dt: Any
    x_threshold: Any
    theta_threshold: Any
    reset_scale: Any


def _to_numpy(value: Any) -> np.ndarray:
    """Return a host numpy view of a concrete ``evoxc`` tensor or an array.

    Host-side helper for ``show`` only: a concrete ``evoxc.Tensor`` exposes its
    data through ``numpy()`` (zero-copy for the numpy-backed reference
    backend), while everything else is coerced with ``np.asarray``.

    Args:
        value: An ``evoxc.Tensor``, a numpy array, or a numpy scalar.

    Returns:
        A numpy array (or 0-d array) holding the same values.
    """
    if isinstance(value, evoxc.Tensor):
        return value.numpy()
    return np.asarray(value)


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
            value above ``max_step`` is unreachable and is rejected.
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
                mass/length/dt/force is not positive, ``theta_threshold <= 0``,
                or ``x_threshold``/``reset_scale`` is negative.
            TensorNEATError: If ``fitness_target`` is greater than ``max_step``.
        """
        self.max_step = self._require_positive_int("max_step", max_step)
        self.repeat_times = self._require_positive_int("repeat_times", repeat_times)
        self.gravity = self._require_positive("gravity", gravity)
        self.masscart = self._require_positive("masscart", masscart)
        self.masspole = self._require_positive("masspole", masspole)
        self.length = self._require_positive("length", length)
        self.force_mag = self._require_positive("force_mag", force_mag)
        self.dt = self._require_positive("dt", dt)
        self.x_threshold = self._require_non_negative("x_threshold", x_threshold)
        self.theta_threshold = self._require_positive(
            "theta_threshold", theta_threshold
        )
        self.reset_scale = self._require_non_negative("reset_scale", reset_scale)
        if fitness_target is not None:
            target = float(fitness_target)
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
           high=reset_scale)``; the key is not threaded through the scan and
           ``np.random`` is never used.
        3. One ``evoxc.scan`` over ``max_step`` timesteps with
           ``xs = enp.zeros((max_step, 1))`` and carry ``(obs (B, 4),
           alive (B,))``.
        4. Step body: forward the policy, convert the two logits into a
           ``(B, 1)`` ``+/-force_mag`` force, reward ``alive`` **before** the
           transition (the failing step still scores, as in Gym), integrate with
           Euler, mark death on ``|x| > x_threshold or |theta| >
           theta_threshold``, then freeze the state with
           ``enp.where(enp.expand_dims(alive, 1), next_obs, obs)`` so no
           ``cond`` and no data-dependent trip count is needed (a traced tensor
           has no ``__getitem__``, so the rank-1 mask is broadcast with
           ``expand_dims``).
        5. The scan emits the rank-1 ``alive`` mask as its output; a rank-3
           stacked output would need a ``scatter`` the stablehlo exporter
           rejects.
        6. Fitness = mean over the ``repeat_times`` axis of the summed reward
           (a return in ``[0, max_step]`` per episode), then
           ``finalize_fitness``.

        A NaN logit still acts as "push left" inside the rollout (``enp.greater``
        is false against NaN), but the scan carries a poison flag for it, so any
        individual whose policy produced a NaN logit at any step gets a NaN
        fitness, which ``finalize_fitness`` maps to ``-inf``.

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
        returns, _, poisoned = self._rollout(keys, act_fn, pop_size)
        fitness = self._finish(returns, pop_size, self.repeat_times)
        poisoned_individual = self._any_repeat(poisoned, pop_size, self.repeat_times)
        nan = enp.full((), float("nan"))
        return finalize_fitness(evoxc.select(poisoned_individual, nan, fitness))

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
        (the rank-0 key) and ``pop_size = 1`` -- and takes the final
        observation out of the scan's **carry**, so no extra stacked output is
        needed.

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
            returns, final_obs, _ = self._rollout(key_value, act_fn, 1)
            return returns, final_obs

        executable = evoxc.build(report, key_spec)
        returns, final_obs = evoxc.run(executable, key)
        returns = _to_numpy(returns)
        final_obs = _to_numpy(final_obs)
        for repeat in range(self.repeat_times):
            survived = int(round(float(returns[repeat])))
            x, x_dot, theta, theta_dot = (float(value) for value in final_obs[repeat])
            print(
                f"CartPole repeat {repeat}: survived {survived}/{self.max_step} steps, "
                f"return {float(returns[repeat]):.2f}"
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
    ) -> tuple[Any, Any, Any]:
        """Roll out ``pop_size * repeat_times`` episodes for one key.

        Must be called inside the active trace: it draws the initial states with
        ``evoxc.random.uniform`` and runs the ``evoxc.scan`` over
        ``max_step`` steps, carry ``(obs (B, 4), alive (B,), poisoned (B,))``
        with ``B = pop_size * repeat_times`` in individual-major order.

        Args:
            keys: One rank-0 evoxc key for the single reset draw.
            act_fn: Callable mapping ``(B, 4)`` observations to ``(B, 2)``
                logits.
            pop_size: Number of individuals ``P`` (static).

        Returns:
            The triple ``(returns (B,), final_obs (B, 4), poisoned (B,))``,
            where ``returns`` is the summed reward of each episode (an integer
            in ``[0, max_step]``), ``final_obs`` the last observation the carry
            holds (the state before death once the episode is over) and
            ``poisoned`` marks the episodes whose policy emitted a NaN logit.
        """
        consts = self._constants()
        batch = pop_size * self.repeat_times
        obs0 = evoxc.random.uniform(
            keys,
            (batch, _OBS_DIM),
            low=enp.negative(consts.reset_scale),
            high=consts.reset_scale,
        )
        init = (
            obs0,
            enp.ones((batch,), dtype=evoxc.bool_),
            enp.zeros((batch,), dtype=evoxc.bool_),
        )
        xs = enp.zeros((self.max_step, 1))
        carry, rewards = evoxc.scan(self._scan_body(act_fn, consts), init, xs)
        obs, _, poisoned = carry
        returns = enp.sum(rewards, axis=0)
        return returns, obs, poisoned

    def _scan_body(self, act_fn: Callable, consts: _CartPoleConstants) -> Callable:
        """Build the per-timestep body of the rollout scan.

        One step rewards the still-alive episodes, converts the policy's logits
        into a signed force, integrates, updates the alive mask and freezes the
        state of every episode that just died, so the carry types stay constant
        across iterations (what ``evoxc.scan``'s ``while_loop`` desugaring
        requires).

        Args:
            act_fn: Callable mapping ``(B, 4)`` observations to ``(B, 2)``
                logits.
            consts: The bundle from :meth:`_constants`.

        Returns:
            ``step(carry, _) -> ((obs, alive, poisoned), reward)`` with
            ``reward`` of shape ``(B,)`` float32 -- the rank-1 scan output.
        """

        def step(carry, _):
            """Advance every episode by one timestep.

            Args:
                carry: ``(obs (B, 4), alive (B,), poisoned (B,))``.
                _: The unused scan input slice (the scan only counts steps).

            Returns:
                The next carry and the ``(B,)`` float32 reward of this step.
            """
            obs, alive, poisoned = carry
            logits = act_fn(obs)
            force = self._force_from_logits(logits, consts)
            nan_rows = self._nan_rows(logits)
            reward = enp.astype(alive, evoxc.float32)
            next_obs = self._euler_step(obs, force, consts)
            dead = self._terminated(next_obs, consts)
            alive = enp.logical_and(alive, enp.logical_not(dead))
            obs = enp.where(enp.expand_dims(alive, 1), next_obs, obs)
            poisoned = enp.logical_or(poisoned, nan_rows)
            return (obs, alive, poisoned), reward

        return step

    def _constants(self) -> _CartPoleConstants:
        """Bundle the physics constants as float32 graph scalars.

        Must be called inside the trace: ``enp.full((), v)`` is a graph
        constructor, and a bare Python float would materialize as f64 on the
        stablehlo path and break iree type unification.

        Returns:
            A structure holding ``gravity``, ``masscart``, ``masspole``,
            ``polemass_length`` (``masspole * length``), ``total_mass``
            (``masscart + masspole``), ``length``, ``force_mag``, ``dt``,
            ``x_threshold``, ``theta_threshold`` (in radians) and
            ``reset_scale``, all float32 -- the reset bounds are float32 graph
            scalars too, so ``evoxc.random.uniform`` never sees an f64 constant.
        """
        return _CartPoleConstants(
            gravity=enp.full((), self.gravity),
            masscart=enp.full((), self.masscart),
            masspole=enp.full((), self.masspole),
            polemass_length=enp.full((), self.masspole * self.length),
            total_mass=enp.full((), self.masscart + self.masspole),
            length=enp.full((), self.length),
            force_mag=enp.full((), self.force_mag),
            dt=enp.full((), self.dt),
            x_threshold=enp.full((), self.x_threshold),
            theta_threshold=enp.full((), float(np.deg2rad(self.theta_threshold))),
            reset_scale=enp.full((), self.reset_scale),
        )

    def _euler_step(self, obs: Any, force: Any, consts: Any) -> Any:
        """Advance the ``(B, 4)`` observations by one Euler step.

        With ``c = cos(theta)``, ``s = sin(theta)`` and
        ``temp = (force + polemass_length * theta_dot**2 * s) / total_mass``,
        the four updates are

        - ``thetaacc = (gravity * s - c * temp) /
          (length * (4/3 - masspole * c**2 / total_mass))``
        - ``xacc = temp - polemass_length * thetaacc * c / total_mass``
        - ``x' = x + dt * x_dot``, ``theta' = theta + dt * theta_dot``
        - ``x_dot' = x_dot + dt * xacc``, ``theta_dot' = theta_dot + dt *
          thetaacc``

        Args:
            obs: Current observations of shape ``(B, 4)``.
            force: Applied force of shape ``(B, 1)``.
            consts: The bundle from ``_constants``.

        Returns:
            The next observations, shape ``(B, 4)``.
        """
        x, x_dot, theta, theta_dot = enp.split(obs, _OBS_DIM, axis=1)
        cos = enp.cos(theta)
        sin = enp.sin(theta)
        temp = (
            force + consts.polemass_length * enp.square(theta_dot) * sin
        ) / consts.total_mass
        four_thirds = enp.full((), 4.0 / 3.0)
        thetaacc = (consts.gravity * sin - cos * temp) / (
            consts.length
            * (four_thirds - consts.masspole * enp.square(cos) / consts.total_mass)
        )
        xacc = temp - consts.polemass_length * thetaacc * cos / consts.total_mass
        return enp.concatenate(
            (
                x + consts.dt * x_dot,
                x_dot + consts.dt * xacc,
                theta + consts.dt * theta_dot,
                theta_dot + consts.dt * thetaacc,
            ),
            axis=1,
        )

    def _terminated(self, obs: Any, consts: Any) -> Any:
        """Return the death mask after a transition.

        An episode is dead when ``|x| > x_threshold`` or ``|theta| >
        theta_threshold`` (radians).

        Args:
            obs: Observations of shape ``(B, 4)``.
            consts: The bundle from ``_constants``.

        Returns:
            ``(B,)`` boolean mask, ``True`` where the episode has failed.
        """
        x, _, theta, _ = enp.split(obs, _OBS_DIM, axis=1)
        too_far = enp.greater(enp.abs(x), consts.x_threshold)
        too_tilted = enp.greater(enp.abs(theta), consts.theta_threshold)
        dead = enp.logical_or(too_far, too_tilted)
        return enp.reshape(dead, (obs.shape[0],))

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
        return enp.where(push_right, consts.force_mag, enp.negative(consts.force_mag))

    def _nan_rows(self, logits: Any) -> Any:
        """Mark the rows of ``logits`` that contain a NaN.

        ``evoxc`` has no ``isnan`` in the traced-op namespace, so the test comes
        from the top-level ``evoxc.isnan`` and is folded over the two logit
        columns.

        Args:
            logits: Policy outputs of shape ``(B, 2)``.

        Returns:
            ``(B,)`` boolean mask, ``True`` where either logit is NaN.
        """
        left, right = enp.split(evoxc.isnan(logits), _ACTION_DIM, axis=1)
        return enp.reshape(enp.logical_or(left, right), (logits.shape[0],))

    def _any_repeat(self, flags: Any, pop_size: int, repeat_times: int) -> Any:
        """Reduce a per-episode flag to one flag per individual.

        The episodes are individual-major, so reshaping ``(P * R,)`` to
        ``(P, R)`` and summing over axis 1 yields a count of poisoned repeats
        that is positive exactly when one of them was poisoned.

        Args:
            flags: Per-episode boolean flags of shape ``(P * R,)``.
            pop_size: Population size ``P``.
            repeat_times: Repeats ``R`` per individual.

        Returns:
            ``(P,)`` boolean mask, ``True`` where any repeat was flagged.
        """
        floats = enp.astype(flags, evoxc.float32)
        grouped = enp.reshape(floats, (pop_size, repeat_times))
        return enp.greater(enp.sum(grouped, axis=1), enp.full((), 0.0))

    def _finish(
        self,
        returns: Any,
        pop_size: int,
        repeat_times: int,
    ) -> Any:
        """Average the episode returns of each individual.

        Reshapes ``(P * R,)`` to ``(P, R)`` and means over axis 1.

        Args:
            returns: Per-episode returns of shape ``(P * R,)`` in
                individual-major order.
            pop_size: Population size ``P``.
            repeat_times: Repeats ``R`` per individual.

        Returns:
            ``(P,)`` mean return, one entry per individual.
        """
        grouped = enp.reshape(returns, (pop_size, repeat_times))
        return enp.mean(grouped, axis=1)

    @staticmethod
    def _require_positive_int(name: str, value: Any) -> int:
        """Validate a strictly positive integer configuration value.

        Args:
            name: Configuration name, used in the error message.
            value: The value to validate; a Python or numpy integer.

        Returns:
            The value as a Python ``int``.

        Raises:
            TensorNEATError: If ``value`` is not an ``int`` or is not positive.
        """
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TensorNEATError(f"CartPole: {name} must be an int, got {value!r}")
        value = int(value)
        if value <= 0:
            raise TensorNEATError(f"CartPole: {name} must be positive, got {value}")
        return value

    @staticmethod
    def _require_positive(name: str, value: Any) -> float:
        """Validate a strictly positive scalar configuration value.

        Args:
            name: Configuration name, used in the error message.
            value: The value to validate.

        Returns:
            The value as a Python ``float``.

        Raises:
            TensorNEATError: If ``value`` is not positive.
        """
        number = float(value)
        if not number > 0.0:
            raise TensorNEATError(f"CartPole: {name} must be positive, got {number}")
        return number

    @staticmethod
    def _require_non_negative(name: str, value: Any) -> float:
        """Validate a non-negative scalar configuration value.

        Args:
            name: Configuration name, used in the error message.
            value: The value to validate.

        Returns:
            The value as a Python ``float``.

        Raises:
            TensorNEATError: If ``value`` is negative.
        """
        number = float(value)
        if number < 0.0:
            raise TensorNEATError(
                f"CartPole: {name} must be non-negative, got {number}"
            )
        return number
