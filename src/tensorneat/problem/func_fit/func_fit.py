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


def _host_array(value: Any) -> np.ndarray:
    """Return a host numpy view of one graph result.

    ``show`` runs outside a trace, so the values it prints come back from
    ``evoxc.evaluate`` as ``evoxc.Tensor`` leaves; a plain numpy value is
    accepted unchanged so the helper also serves a host-only schedule.

    Args:
        value: An ``evoxc.Tensor`` or anything ``np.asarray`` accepts.

    Returns:
        A numpy array holding the same values.
    """
    numpy_view = getattr(value, "numpy", None)
    if callable(numpy_view):
        return np.asarray(numpy_view())
    return np.asarray(value)


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
        if error_method not in self.ERROR_METHODS:
            raise TensorNEATError(
                f"error_method must be one of {self.ERROR_METHODS}, "
                f"got {error_method!r}"
            )
        self.error_method = error_method

    @property
    def inputs(self) -> np.ndarray:
        """Host-side dataset inputs of shape ``(S, I)``.

        Returns:
            Float32 numpy array with one row per sample.

        Raises:
            NotImplementedError: Always, in the base class; a concrete problem
                must provide the dataset.
        """
        raise NotImplementedError(
            "FuncFit.inputs must be implemented by the concrete problem: return "
            "the (S, I) float32 dataset inputs"
        )

    @property
    def targets(self) -> np.ndarray:
        """Host-side dataset targets of shape ``(S, O)``.

        Returns:
            Float32 numpy array with one row per sample.

        Raises:
            NotImplementedError: Always, in the base class; a concrete problem
                must provide the dataset.
        """
        raise NotImplementedError(
            "FuncFit.targets must be implemented by the concrete problem: return "
            "the (S, O) float32 dataset targets"
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Per-sample input shape ``(I,)``.

        Returns:
            The shape of one dataset row, not of the whole dataset.

        Raises:
            NotImplementedError: Always, in the base class; a concrete problem
                must declare its arity.
        """
        raise NotImplementedError(
            "FuncFit.input_shape must be implemented by the concrete problem: "
            "declare the per-sample input shape (I,)"
        )

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Per-sample output shape ``(O,)``.

        Returns:
            The shape of one target row, not of the whole dataset.

        Raises:
            NotImplementedError: Always, in the base class; a concrete problem
                must declare its arity.
        """
        raise NotImplementedError(
            "FuncFit.output_shape must be implemented by the concrete problem: "
            "declare the per-sample output shape (O,)"
        )

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
        3. ``tiled = enp.reshape(enp.broadcast_to(enp.expand_dims(inputs, 0),
           (P, S, I)), (P * S, I))`` so that row ``p * S + s`` is sample ``s``
           of individual ``p`` (individual-major, the layout ``act_fn`` must
           see). The ``expand_dims`` is required because
           ``enp.broadcast_to`` cannot introduce a new leading axis.
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
        """
        pop_size = static_batch_size(transformed)
        _, loss = self._predict_and_loss(act_fn, pop_size)
        return finalize_fitness(enp.negative(loss))

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

        The schedule's leaves are threaded into a small one-off graph as data
        arguments and the schedule container is rebuilt inside the trace, so a
        schedule backed by numpy arrays and one backed by ``evoxc.Tensor``
        leaves are both accepted.

        Args:
            state: Host container with configuration.
            key: One evoxc key; unused, the problem is deterministic.
            act_fn: Callable mapping a batch to a batch of predictions.
            transformed: A schedule holding a single individual.

        Returns:
            ``None``; the report goes to stdout.
        """
        schedule_type = type(transformed)
        leaves = tuple(evoxc.tree_leaves(transformed))

        def kernel(*arrays: Any) -> tuple[Any, Any]:
            """Rebuild the schedule inside the trace and score it.

            Args:
                *arrays: The schedule's leaves, one concrete tensor per leaf.

            Returns:
                ``(pred (P, S, O), loss (P,))`` graph values.
            """
            schedule = schedule_type(*arrays)
            pop_size = static_batch_size(schedule)
            return self._predict_and_loss(act_fn, pop_size)

        pred, loss = evoxc.evaluate(kernel, *leaves)
        pred = _host_array(pred)
        loss = _host_array(loss)
        inputs = np.asarray(self.inputs, dtype=np.float32)
        targets = np.asarray(self.targets, dtype=np.float32)
        for individual in range(pred.shape[0]):
            if pred.shape[0] > 1:
                print(f"individual: {individual}")
            for sample in range(inputs.shape[0]):
                print(
                    f"input: {inputs[sample]}, target: {targets[sample]}, "
                    f"predict: {pred[individual, sample]}"
                )
            print(f"loss: {loss[individual]}")

    def _predict_and_loss(self, act_fn: Callable, pop_size: int) -> tuple[Any, Any]:
        """Tile the baked dataset over the population and score every sample.

        Runs inside the active trace: the dataset comes from
        :meth:`_dataset_constants`, the only ``act_fn`` call receives the whole
        ``(pop_size * S, I)`` tiled batch, and the error tensor is reduced by
        :meth:`_loss`.

        Args:
            act_fn: Callable mapping the ``(pop_size * S, I)`` batch to
                ``(pop_size * S, O)``.
            pop_size: Static population size ``P``.

        Returns:
            The pair ``(pred, loss)`` of shapes ``(P, S, O)`` and ``(P,)``.
        """
        inputs, targets = self._dataset_constants()
        num_samples = int(inputs.shape[0])
        num_inputs = int(inputs.shape[1])
        num_outputs = int(targets.shape[1])
        tiled = enp.reshape(
            enp.broadcast_to(
                enp.expand_dims(inputs, 0),
                (pop_size, num_samples, num_inputs),
            ),
            (pop_size * num_samples, num_inputs),
        )
        pred = enp.reshape(act_fn(tiled), (pop_size, num_samples, num_outputs))
        err = enp.subtract(pred, enp.expand_dims(targets, 0))
        return pred, self._loss(err, targets)

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
        inputs = np.asarray(self.inputs, dtype=np.float32)
        targets = np.asarray(self.targets, dtype=np.float32)
        return (
            evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(inputs))),
            evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(targets))),
        )

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
        if self.error_method == "mse":
            return enp.mean(enp.square(err), axis=(1, 2))
        if self.error_method == "rmse":
            return enp.sqrt(enp.mean(enp.square(err), axis=(1, 2)))
        if self.error_method == "mae":
            return enp.mean(enp.abs(err), axis=(1, 2))
        # ``mape`` -- the only reduction left, since ``__init__`` validated the
        # name against ``ERROR_METHODS``.
        ratio = enp.divide(err, enp.expand_dims(targets, 0))
        return enp.mean(enp.abs(ratio), axis=(1, 2))
