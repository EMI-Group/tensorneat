"""Problem protocol for TensorNEAT on the EvoX Compiler (evoxc).

A problem turns a batch-first execution schedule into one scalar fitness per
individual. Fitness is always **maximized**: a loss is negated before it leaves
a problem, and NaN fitness is mapped to ``-inf``.

``evaluate`` is a plain helper that is inlined into the caller's trace, never an
``@evoxc.defn`` graph, because one ``@evoxc.defn`` may not call another and
``act_fn`` reaches the problem by closure.
"""

from typing import Any, Callable

import numpy as np

from ..common.errors import TensorNEATError
from ..common.state import State
from ..common.stateful_class import StatefulBaseClass

__all__ = ["BaseProblem", "finalize_fitness", "static_batch_size"]


def finalize_fitness(fitness: Any) -> Any:
    """Map NaN fitness to ``-inf`` so a NaN-poisoned network can never win.

    An unconnected or otherwise broken topology evaluates to NaN, and NaN
    compares false against every value, so it could silently survive sorting.
    Mapping it to ``-inf`` makes such an individual the worst in the population.

    The body runs inside an active trace, so it uses ``evoxc.numpy`` only;
    ``evoxc.isnan`` lives at the top level because ``enp`` has no ``isnan``.

    Args:
        fitness: Fitness batch of shape ``(P,)``, one entry per individual.

    Returns:
        ``(P,)`` float32 fitness with every NaN replaced by ``-inf``.

    Notes:
        Every shipped problem applies this as its last operation, before the
        values are handed back to the algorithm's selection step.
    """
    raise NotImplementedError


def static_batch_size(transformed: Any) -> int:
    """Return the static population size of a batch-first execution schedule.

    ``transformed`` is the pytree produced by ``Genome.transform``; every leaf
    carries the population on axis 0. The size is recovered by walking the
    pytree leaves and taking their static leading dimension -- inside a trace
    ``SymbolicTensor.shape[0]`` is a plain ``int``, so it stays host-visible.

    A problem must not introspect the schedule's schema beyond this call, so the
    genome is free to change its execution representation without touching any
    problem.

    Args:
        transformed: Batch-first execution schedule from ``Genome.transform``.

    Returns:
        The population size ``P`` as a Python ``int``.

    Raises:
        TensorNEATError: If no leaf exposes a static leading dimension, or if
            the leaves disagree about it.
    """
    raise NotImplementedError


class BaseProblem(StatefulBaseClass):
    """Base class for every problem (task / fitness layer).

    A problem is pure host-side configuration -- construction-time values plus
    plain-Python kernel helpers that get inlined into the caller's graph -- with
    no traced state of its own. The legacy ``jitable`` flag is intentionally
    absent: a problem is always traced, and whether a kernel compiles is the
    backend's business (``evoxc.BackendError`` propagates unchanged, never a
    silent host fallback).

    Subclasses implement ``evaluate`` plus the ``input_shape``/``output_shape``
    declarations; ``setup`` is inherited from ``StatefulBaseClass``.
    """

    def evaluate(
        self,
        state: State,
        keys: Any,
        act_fn: Callable,
        transformed: Any,
    ) -> Any:
        """Score a whole population in one batched, in-trace evaluation.

        Args:
            state: Immutable host container; accepted for protocol uniformity
                (a user problem may register normalization statistics in
                ``setup``). No shipped problem registers keys today.
            keys: **One** evoxc key -- a rank-0 int64 from ``evoxc.random.key``
                -- never a ``(P,)`` batch: evoxc has no batched key type and
                random ops are not vmappable, so a population draws a whole
                ``(P, ...)`` block from a single key.
            act_fn: Policy/value callable closed over by the problem;
                ``act_fn(batch, *input_shape) -> (batch, *output_shape)`` with
                the batch on axis 0. ``batch`` is not necessarily ``P``: func
                fit tiles the dataset (``P * S``) and rollouts repeat episodes
                (``P * R``).
            transformed: Batch-first execution schedule from
                ``Genome.transform``. The only thing a problem may read from it
                is ``static_batch_size(transformed)``, which is schema
                agnostic.

        Returns:
            ``(P,)`` fitness, one entry per individual, **maximized**.

        Raises:
            NotImplementedError: Always, in the skeleton.
            TensorNEATError: Configuration errors detected by a subclass.

        Notes:
            The body runs inside an active trace, so it may not use Python
            control flow over traced values and may not call ``np.*`` on them.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Per-sample input shape of the problem.

        ``(2,)`` for XOR, ``(4,)`` for CartPole's observation. It is the shape
        of **one** sample, not of the dataset, so ``input_shape[-1]`` equals
        ``algorithm.num_inputs``; the legacy dataset-shaped tuples are gone.

        Returns:
            The per-sample input shape.
        """
        raise NotImplementedError

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Per-sample output shape of the problem.

        ``(1,)`` for XOR, ``(2,)`` for the two CartPole logits. It is the shape
        of **one** sample, not of the dataset, so ``output_shape[-1]`` equals
        ``algorithm.num_outputs``; the legacy dataset-shaped tuples are gone.

        Returns:
            The per-sample output shape.
        """
        raise NotImplementedError

    def show(
        self,
        state: State,
        key: Any,
        act_fn: Callable,
        transformed: Any,
    ) -> None:
        """Print a human-facing report of one individual's behaviour.

        Host-side only: never called from a trace. It evaluates a batch of one
        and prints a readable summary (for func fit one row per sample plus the
        loss, for CartPole the survival length and the return).

        Args:
            state: Immutable host container with configuration.
            key: One evoxc key, used only by stochastic problems.
            act_fn: Callable mapping a batch to a batch of predictions.
            transformed: A schedule holding a single individual.

        Returns:
            ``None``; the report goes to stdout.
        """
        raise NotImplementedError

    def show_details(
        self,
        state: State,
        key: Any,
        act_fn: Callable,
        transformed: Any,
    ) -> None:
        """Host-side hook called by ``Pipeline.auto_run`` for extra diagnostics.

        Invoked once per generation when ``Pipeline.auto_run`` runs with
        ``show_problem_details=True``. The default is a no-op; a problem that
        wants to publish running statistics overrides it.

        Args:
            state: Immutable host container with configuration.
            key: One evoxc key, usable for a diagnostic rollout.
            act_fn: Callable mapping a batch to a batch of predictions.
            transformed: The population's batch-first execution schedule.

        Returns:
            ``None``.
        """
        pass
