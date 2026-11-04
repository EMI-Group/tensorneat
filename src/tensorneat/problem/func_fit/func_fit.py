"""Supervised function fitting: dataset tiling, loss reduction, host-side report.

``FuncFit`` owns the one non-trivial part of this family -- how a small fixed
dataset becomes a single batched evaluation of the whole population -- and each
concrete problem is just a dataset plus two shapes.
Tiling is one ``evoxc.tile`` block repetition (individual-major, ``row p*S+s``),
the ``(S, O)`` targets broadcast numpy-style against the ``(P, S, O)`` error
tensor, and the dataset itself is a compile-time constant bounded by
``FuncFit.MAX_DATASET_NBYTES``.
"""

from typing import Any, Callable

import numpy as np

import evoxc
import evoxc.numpy as enp

from ..base import BaseProblem, finalize_fitness, static_batch_size
from ...common import to_numpy
from ...common.errors import TensorNEATError

__all__ = ["FuncFit"]


class FuncFit(BaseProblem):
    """Fit a genome to a fixed dataset; fitness is the negated loss.

    The dataset is genuinely static, so it is baked into the compiled artifact
    as an evoxc constant at trace time instead of being threaded through
    ``State`` or uploaded per call.
    That constant is paid on every build, so its footprint is checked against
    ``MAX_DATASET_NBYTES`` on the host (at construction, in ``setup`` and again
    at bake time) rather than discovered as an evoxc warning mid-trace.

    Attributes:
        ERROR_METHODS: The supported per-individual loss reductions.
        MAX_DATASET_NBYTES: Compile budget for the two baked dataset constants.

    Notes:
        ``MAX_DATASET_NBYTES`` mirrors evoxc's large-constant threshold
        (``EVOXC_LARGE_CONSTANT_BYTES``, 1 MiB by default).
        A baked dataset is compiled into the artifact and the problem hands the
        network a ``(P * S, I)`` batch, so the constant's size is paid on every
        build; a subclass may raise the attribute deliberately.
    """

    ERROR_METHODS: tuple[str, ...] = ("mse", "rmse", "mae", "mape")
    #: evoxc's large-constant threshold (``EVOXC_LARGE_CONSTANT_BYTES``, 1 MiB
    #: by default); a baked dataset is compiled into the artifact, so its size
    #: is paid on every build, and a subclass may raise this deliberately.
    MAX_DATASET_NBYTES: int = 1 << 20

    def __init__(self, error_method: str = "mse") -> None:
        """Configure the loss reduction.

        Args:
            error_method: One of ``"mse"``, ``"rmse"``, ``"mae"``, ``"mape"``,
                i.e. a member of ``ERROR_METHODS``.

        Raises:
            TensorNEATError: If ``error_method`` is not in ``ERROR_METHODS``, or
                if the dataset already exists and exceeds
                ``MAX_DATASET_NBYTES``.
        """
        if error_method not in self.ERROR_METHODS:
            raise TensorNEATError(
                f"error_method must be one of {self.ERROR_METHODS}, "
                f"got {error_method!r}"
            )
        self.error_method = error_method
        try:
            self.validate_dataset()
        except NotImplementedError:
            # a subclass may build its dataset after calling super().__init__();
            # the check then runs in setup() and again at bake time.
            pass

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

    @property
    def dataset_nbytes(self) -> int:
        """Combined footprint of the two baked dataset constants, in bytes.

        Measured on the float32 host arrays that :meth:`_dataset_constants`
        bakes, so it is exactly what the compiled artifact carries; the dataset
        is never pre-tiled, so the population size does not enter it.

        Returns:
            ``inputs.nbytes + targets.nbytes`` for the float32 dataset.

        Raises:
            NotImplementedError: If the concrete problem has no dataset yet, via
                the base :attr:`inputs`/:attr:`targets` properties.
        """
        return self._dataset_footprint(*self._host_dataset())

    def validate_dataset(self) -> int:
        """Check the dataset footprint against ``MAX_DATASET_NBYTES``.

        The budget exists because evoxc already warns at this size
        (``EVOXC_LARGE_CONSTANT_BYTES``, 1 MiB by default), yet in function
        fitting the same ``S`` also multiplies every per-row allocation on the
        activation path and is paid again on every build, so an oversized
        dataset is a configuration mistake rather than a slow-but-valid run --
        which is why it is an error and not a warning.

        Returns:
            The combined footprint in bytes.

        Raises:
            NotImplementedError: If the concrete problem has no dataset yet, via
                the base :attr:`inputs`/:attr:`targets` properties.
            TensorNEATError: If the footprint exceeds ``MAX_DATASET_NBYTES``.
        """
        return self._check_dataset_budget(*self._host_dataset())

    def setup(self, state: Any = None) -> Any:
        """Validate the dataset budget, then delegate to the base ``setup``.

        Args:
            state: The incoming shared state, or ``None`` to start from a fresh
                empty ``State``.

        Returns:
            Whatever ``StatefulBaseClass.setup`` returns -- the same state,
            unchanged, since func fit registers no keys.

        Raises:
            TensorNEATError: If the dataset exceeds ``MAX_DATASET_NBYTES``.
        """
        self.validate_dataset()
        return super().setup(state)

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
        3. ``tiled = evoxc.tile(inputs, (P, 1))`` repeats the ``(S, I)`` block
           ``P`` times with numpy ``tile`` semantics, so row ``p * S + s`` is
           sample ``s`` of individual ``p`` (individual-major, the layout
           ``act_fn`` must see). One ``tile`` op replaces the former
           ``expand_dims`` + ``broadcast_to`` + ``reshape`` chain.
        4. **One** ``act_fn`` call over the whole tiled batch; there is no vmap
           in evoxc (``evoxc.vmap`` over control flow or random is unsupported),
           so tiling replaces vmap-over-samples.
        5. ``pred = enp.reshape(act_fn(...), (P, S, O))`` and
           ``err = enp.subtract(pred, targets)``: the ``(S, O)`` targets
           broadcast numpy-style over the population axis, so no
           ``expand_dims`` is needed.
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
        arguments and the schedule container is rebuilt inside the trace with
        evoxc's pytree machinery -- ``tree_unflatten(tree_structure(...),
        leaves)`` -- never with a positional constructor call. That matters
        because the container is opaque to a problem: a flat ``Transformed``
        namedtuple and HyperNEAT's nested ``SubstrateSchedule`` (whose first
        field is itself a schedule) must both round-trip, and only the
        structure does not care how many leaves a container has.

        Args:
            state: Host container with configuration.
            key: One evoxc key; unused, the problem is deterministic.
            act_fn: Callable mapping a batch to a batch of predictions.
            transformed: A batch-first schedule, normally holding a single
                individual; the report prints one block per individual, so any
                static ``P`` is accepted.

        Returns:
            ``None``; the report goes to stdout.
        """
        structure = evoxc.tree_structure(transformed)
        leaves = tuple(evoxc.tree_leaves(transformed))

        def kernel(*arrays: Any) -> tuple[Any, Any]:
            """Rebuild the schedule inside the trace and score it.

            Args:
                *arrays: The schedule's leaves, one concrete tensor per leaf.

            Returns:
                ``(pred (P, S, O), loss (P,))`` graph values.
            """
            schedule = evoxc.tree_unflatten(list(arrays), structure)
            pop_size = static_batch_size(schedule)
            return self._predict_and_loss(act_fn, pop_size)

        pred, loss = evoxc.evaluate(kernel, *leaves)
        pred = to_numpy(pred)
        loss = to_numpy(loss)
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
        tiled = evoxc.tile(inputs, (pop_size, 1))
        pred = enp.reshape(
            act_fn(tiled), (pop_size, num_samples, int(targets.shape[1]))
        )
        return pred, self._loss(enp.subtract(pred, targets), targets)

    def _host_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the float32 host dataset exactly as it will be baked.

        Host-side only; the single place the dtype coercion happens, so the
        budget arithmetic and the baked constants can never disagree.

        Returns:
            The pair ``(inputs (S, I), targets (S, O))`` of float32 arrays.

        Raises:
            NotImplementedError: If the concrete problem has no dataset yet, via
                the base :attr:`inputs`/:attr:`targets` properties.
        """
        return (
            np.asarray(self.inputs, dtype=np.float32),
            np.asarray(self.targets, dtype=np.float32),
        )

    @staticmethod
    def _dataset_footprint(inputs: np.ndarray, targets: np.ndarray) -> int:
        """Return the combined byte size of two already-materialised arrays.

        Args:
            inputs: Float32 inputs of shape ``(S, I)``.
            targets: Float32 targets of shape ``(S, O)``.

        Returns:
            ``inputs.nbytes + targets.nbytes`` as an ``int``.
        """
        return int(inputs.nbytes + targets.nbytes)

    def _check_dataset_budget(
        self, inputs: np.ndarray, targets: np.ndarray
    ) -> int:
        """Enforce the dataset-constant budget on materialised arrays.

        The single place the comparison lives: :meth:`validate_dataset` calls it
        on the host, and :meth:`_dataset_constants` calls it again at bake time
        so a direct ``evaluate`` is guarded too.

        Args:
            inputs: Float32 inputs of shape ``(S, I)``.
            targets: Float32 targets of shape ``(S, O)``.

        Returns:
            The combined footprint in bytes.

        Raises:
            TensorNEATError: If the footprint exceeds ``MAX_DATASET_NBYTES``.
        """
        nbytes = self._dataset_footprint(inputs, targets)
        if nbytes > self.MAX_DATASET_NBYTES:
            raise TensorNEATError(
                f"dataset constants cost {nbytes} bytes: inputs "
                f"{inputs.shape} ({inputs.nbytes} bytes) + targets "
                f"{targets.shape} ({targets.nbytes} bytes) exceed the "
                f"{self.MAX_DATASET_NBYTES}-byte budget "
                f"(FuncFit.MAX_DATASET_NBYTES); shrink the dataset (fewer "
                f"samples or a coarser grid) or raise "
                f"FuncFit.MAX_DATASET_NBYTES deliberately"
            )
        return nbytes

    def _dataset_constants(self) -> tuple[Any, Any]:
        """Bake the host dataset into graph constants.

        Must be called inside the trace, because ``evoxc.constant`` is a graph
        constructor.
        The constants stay ``(S, I)`` and ``(S, O)`` -- never pre-tiled, never
        scaled with the population -- because the tiling happens in the graph.
        The budget is enforced here as well as on the host, so an ``evaluate``
        called directly (as the parity harness does) is guarded too; the
        ``TensorNEATError`` propagates unchanged through ``evoxc.build``.

        Returns:
            The pair ``(inputs, targets)`` as ``(S, I)`` and ``(S, O)`` graph
            constants.

        Raises:
            TensorNEATError: If the two constants exceed
                :attr:`MAX_DATASET_NBYTES`.
        """
        inputs, targets = self._host_dataset()
        self._check_dataset_budget(inputs, targets)
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
        # name against ``ERROR_METHODS``. The ``(S, O)`` targets broadcast over
        # the population axis, so no ``expand_dims`` is needed.
        ratio = enp.divide(err, targets)
        return enp.mean(enp.abs(ratio), axis=(1, 2))
