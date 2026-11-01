"""Supervised function fitting: dataset tiling, loss reduction, host-side report.

``FuncFit`` owns the one non-trivial part of this family -- how a small fixed
dataset becomes a single batched evaluation of the whole population -- and each
concrete problem is just a dataset plus two shapes.
"""

from typing import Any, Callable

import numpy as np

import evoxc
import evoxc.numpy as enp

from ..base import BaseProblem, finalize_fitness, static_batch_size
from ...common.errors import TensorNEATError

__all__ = ["FuncFit"]


class FuncFit(BaseProblem):
    """Fit a genome to a fixed dataset; fitness is the negated loss.

    The dataset is genuinely static, so it is baked into the compiled artifact
    as an evoxc constant at trace time instead of being threaded through
    ``State`` or uploaded per call.

    Attributes:
        ERROR_METHODS: The supported per-individual loss reductions.
    """

    ERROR_METHODS: tuple[str, ...] = ("mse", "rmse", "mae", "mape")

    def __init__(self, error_method: str = "mse") -> None:
        """Configure the loss reduction.

        Args:
            error_method: One of ``"mse"``, ``"rmse"``, ``"mae"``, ``"mape"``,
                i.e. a member of ``ERROR_METHODS``.

        Raises:
            TensorNEATError: If ``error_method`` is not in ``ERROR_METHODS``.
        """
        raise NotImplementedError

    @property
    def inputs(self) -> np.ndarray:
        """Host-side dataset inputs of shape ``(S, I)``.

        Returns:
            Float32 numpy array with one row per sample.
        """
        raise NotImplementedError

    @property
    def targets(self) -> np.ndarray:
        """Host-side dataset targets of shape ``(S, O)``.

        Returns:
            Float32 numpy array with one row per sample.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Per-sample input shape ``(I,)``.

        Returns:
            The shape of one dataset row, not of the whole dataset.
        """
        raise NotImplementedError

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Per-sample output shape ``(O,)``.

        Returns:
            The shape of one target row, not of the whole dataset.
        """
        raise NotImplementedError

    def evaluate(
        self,
        state: Any,
        keys: Any,
        act_fn: Callable,
        transformed: Any,
    ) -> Any:
        """Score the population over every dataset sample in one batched call.

        The algebra, with ``P`` individuals, ``S`` samples, ``I`` inputs and
        ``O`` outputs (all static at build time):

        1. ``P = static_batch_size(transformed)`` -- the only thing read from
           the schedule.
        2. The dataset is baked at trace time as
           ``evoxc.constant(evoxc.from_numpy(...))``; it is genuinely static
           data, so nothing is uploaded per call.
        3. ``tiled = enp.reshape(enp.broadcast_to(inputs, (P, S, I)), (P * S, I))``
           so that row ``p * S + s`` is sample ``s`` of individual ``p``
           (individual-major, the layout ``act_fn`` must see).
        4. **One** ``act_fn`` call over the whole tiled batch; there is no vmap
           in evoxc (``evoxc.vmap`` over control flow or random is unsupported),
           so tiling replaces vmap-over-samples.
        5. ``pred = enp.reshape(pred, (P, S, O))`` and
           ``err = pred - enp.expand_dims(targets, 0)`` broadcast over the
           population axis.
        6. The per-individual loss is reduced over the sample **and** output
           axes (equivalent to the legacy ``jnp.mean`` over all elements), then
           negated for maximization: ``finalize_fitness(-loss)``.

        A NaN prediction (an unconnected or dead topology) yields a NaN loss and
        therefore ``-inf`` fitness.

        Args:
            state: Host container; unused, kept for protocol uniformity.
            keys: One evoxc key; **unused** -- function fitting is
                deterministic and draws no random numbers.
            act_fn: Callable mapping the ``(P * S, I)`` batch to
                ``(P * S, O)``.
            transformed: Batch-first execution schedule holding ``P``
                individuals.

        Returns:
            ``(P,)`` fitness, maximized (``-loss``), NaN mapped to ``-inf``.

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
        """Print one individual's predictions against the dataset.

        Host-side only: never called from a trace. Evaluates a batch of one and
        prints one ``input | target | predict`` row per sample, followed by the
        loss.

        Args:
            state: Host container with configuration.
            key: One evoxc key; unused, the problem is deterministic.
            act_fn: Callable mapping a batch to a batch of predictions.
            transformed: A schedule holding a single individual.

        Returns:
            ``None``; the report goes to stdout.
        """
        raise NotImplementedError

    def _dataset_constants(self) -> tuple[Any, Any]:
        """Bake the host dataset into graph constants.

        Must be called inside the trace, because ``evoxc.constant`` is a graph
        constructor. The dataset travels inside the compiled artifact, so keep
        ``S * I`` modest: evoxc warns above ~1 MiB of constant payload, i.e.
        roughly ``S * I > 250k`` floats.

        Returns:
            The pair ``(inputs, targets)`` as ``(S, I)`` and ``(S, O)`` graph
            constants.
        """
        raise NotImplementedError

    def _loss(self, err: Any, targets: Any) -> Any:
        """Reduce the ``(P, S, O)`` error tensor to a ``(P,)`` loss.

        The four reductions, each reduced over the sample and output axes:

        - ``mse = mean(err ** 2)``
        - ``rmse = sqrt(mse)``
        - ``mae = mean(abs(err))``
        - ``mape = mean(abs(err / targets))`` -- a zero target yields ``inf``,
          which is legacy behaviour and is kept.

        Args:
            err: Prediction error of shape ``(P, S, O)``.
            targets: Broadcastable targets of shape ``(S, O)``.

        Returns:
            ``(P,)`` float32 loss, not yet negated.
        """
        raise NotImplementedError
