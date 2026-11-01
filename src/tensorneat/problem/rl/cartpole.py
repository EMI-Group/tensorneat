"""Batched CartPole rollout on evoxc: Gym dynamics, fixed horizon, return fitness."""

from typing import Any, Callable

import numpy as np

import evoxc
import evoxc.numpy as enp

from ..base import BaseProblem, finalize_fitness, static_batch_size
from ...common.errors import TensorNEATError

__all__ = ["CartPole"]


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

        Raises:
            TensorNEATError: If ``max_step <= 0``, ``repeat_times <= 0``, any
                mass/length/dt/force is not positive, ``theta_threshold <= 0``,
                or ``x_threshold``/``reset_scale`` is negative.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Observation shape ``(4,)`` = ``(x, x_dot, theta, theta_dot)``.

        Returns:
            The per-step observation shape.
        """
        raise NotImplementedError

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Action shape ``(2,)``: two logits.

        The cart is pushed right iff ``out[1] > out[0]``; a tie pushes left,
        which matches ``argmax`` index ``0``.

        Returns:
            The per-step output shape.
        """
        raise NotImplementedError

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
           ``enp.where(alive[:, None], next_obs, obs)`` so no ``cond`` and no
           data-dependent trip count is needed.
        5. The scan emits the rank-1 ``alive`` mask as its output; a rank-3
           stacked output would need a ``scatter`` the stablehlo exporter
           rejects.
        6. Fitness = mean over the ``repeat_times`` axis of the summed reward
           (a return in ``[0, max_step]`` per episode), then
           ``finalize_fitness``.

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
            NotImplementedError: Always, in the skeleton.
        """
        raise NotImplementedError

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

        Args:
            state: Host container with configuration.
            key: One evoxc key for the reset draw.
            act_fn: Callable mapping observations to logits.
            transformed: A schedule holding a single individual.

        Returns:
            ``None``; the report goes to stdout.
        """
        raise NotImplementedError

    def _constants(self) -> Any:
        """Bundle the physics constants as float32 graph scalars.

        Must be called inside the trace: ``enp.full((), v)`` is a graph
        constructor, and a bare Python float would materialize as f64 on the
        stablehlo path and break iree type unification.

        Returns:
            A structure holding ``gravity``, ``masscart``, ``masspole``,
            ``polemass_length`` (``masspole * length``), ``total_mass``
            (``masscart + masspole``), ``length``, ``force_mag``, ``dt``,
            ``x_threshold`` and ``theta_threshold`` (in radians), all float32.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def _force_from_logits(self, logits: Any, consts: Any) -> Any:
        """Turn two logits per row into a signed force.

        The decision is ``enp.split`` + ``enp.greater`` (push right iff
        ``out[1] > out[0]``, a tie pushes left), so there is no integer or bool
        control flow and no ``argmax`` dtype handling.

        Args:
            logits: Policy outputs of shape ``(B, 2)``.
            consts: The bundle from ``_constants``.

        Returns:
            Force of shape ``(B, 1)``, each row ``+force_mag`` or
            ``-force_mag``.
        """
        raise NotImplementedError

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
        raise NotImplementedError
