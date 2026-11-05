"""Function fitting against a user-supplied function, sampled or gridded."""

from typing import Any, Callable

import numpy as np

from .func_fit import FuncFit
from ...common.errors import TensorNEATError

__all__ = ["CustomFuncFit", "cartesian_product"]


class CustomFuncFit(FuncFit):
    """Fit a genome to an arbitrary Python function over a box.

    The dataset is generated host-side at construction time and then baked into
    the graph, so it is a compile-time constant: its footprint is checked
    against ``FuncFit.MAX_DATASET_NBYTES`` (1 MiB) at construction, and a fine
    ``method="grid"`` step can exceed that and raises ``TensorNEATError``.
    """

    def __init__(
        self,
        func: Callable,
        low_bounds: Any,
        upper_bounds: Any,
        method: str = "sample",
        num_samples: int = 100,
        step_size: Any = None,
        seed: int | None = None,
        error_method: str = "mse",
    ) -> None:
        """Generate the dataset for ``func`` and configure the loss.

        ``func`` maps a 1-D point of shape ``(I,)`` to a scalar or an array of
        shape ``(O,)``; targets are built host-side with
        ``np.apply_along_axis(func, -1, inputs)``.

        Sampling modes:

        - ``method="sample"`` draws ``num_samples`` rows in total, each of them a
          uniform point of the box ``[low_bounds, upper_bounds]``, using
          ``np.random.default_rng(seed)``. The generator is host-side only: the
          legacy version mutated the global ``np.random`` state, which made runs
          irreproducible without also seeding the process globally.
        - ``method="grid"`` builds the cartesian product of
          ``np.arange(low_bounds[i], upper_bounds[i], step_size[i])`` over every
          axis, so ``step_size`` must have the same shape as the bounds and be
          strictly positive.

        Args:
            func: The target function, ``(I,) -> scalar | (O,)``.
            low_bounds: Lower corner of the box, one entry per input.
            upper_bounds: Upper corner of the box, same shape as ``low_bounds``.
            method: ``"sample"`` or ``"grid"``.
            num_samples: Number of sample rows drawn from the box for
                ``method="sample"``; ignored by ``method="grid"``.
            step_size: Per-axis grid step for ``method="grid"``; ignored
                otherwise.
            seed: Seed of the host-side ``np.random.default_rng`` for
                ``method="sample"``; ``None`` means a fresh entropy source.
            error_method: One of ``"mse"``, ``"rmse"``, ``"mae"``, ``"mape"``;
                forwarded to ``FuncFit``.

        Raises:
            TensorNEATError: If ``method`` is not ``"sample"``/``"grid"``, if
                ``num_samples <= 0``, if ``step_size`` is missing, does not
                match the bounds' shape or is not strictly positive, if
                ``low_bounds.shape != upper_bounds.shape``, if a bound is not
                finite, if any axis is not a proper interval
                (``low_bounds[i] < upper_bounds[i]``), or if
                ``func(low_bounds)`` raises.
        """
        low = np.asarray(low_bounds, dtype=np.float32)
        high = np.asarray(upper_bounds, dtype=np.float32)
        if low.shape != high.shape:
            raise TensorNEATError(
                f"low_bounds.shape {low.shape} must equal upper_bounds.shape "
                f"{high.shape}"
            )
        not_finite = np.flatnonzero(~(np.isfinite(low) & np.isfinite(high)))
        if not_finite.size:
            axis = int(not_finite[0])
            raise TensorNEATError(
                f"bounds must be finite, but axis {axis} has low_bounds "
                f"{low.ravel()[axis]} and upper_bounds {high.ravel()[axis]}; "
                f"inf/nan cannot define a sampling box"
            )
        not_proper = np.flatnonzero(~(low < high))
        if not_proper.size:
            axis = int(not_proper[0])
            raise TensorNEATError(
                f"low_bounds must be strictly below upper_bounds on every axis, "
                f"but axis {axis} has low_bounds {low.ravel()[axis]} and "
                f"upper_bounds {high.ravel()[axis]}; a zero-width or reversed "
                f"box yields an empty dataset (S = 0)"
            )
        try:
            func(low)
        except Exception as exc:  # noqa: BLE001 - re-raised as a config error
            raise TensorNEATError(
                f"func(low_bounds) raised {type(exc).__name__}: {exc}"
            ) from exc
        if method not in {"sample", "grid"}:
            raise TensorNEATError(
                f"method must be 'sample' or 'grid', got {method!r}"
            )
        if num_samples <= 0:
            raise TensorNEATError(
                f"num_samples must be positive, got {num_samples}"
            )
        if method == "grid":
            if step_size is None:
                raise TensorNEATError(
                    "step_size must be provided when method is 'grid'"
                )
            step = np.asarray(step_size, dtype=np.float32)
            if step.shape != low.shape:
                raise TensorNEATError(
                    f"step_size must have the same shape as the bounds "
                    f"{low.shape}, got {step.shape}"
                )
            if not np.all(step > 0):
                raise TensorNEATError(
                    f"step_size must be strictly positive, got {step}"
                )
            self.step_size = step
        else:
            self.step_size = step_size
        self.func = func
        self.low_bounds = low
        self.upper_bounds = high
        self.method = method
        self.num_samples = num_samples
        self.seed = seed
        self.generate_dataset()
        super().__init__(error_method=error_method)

    def generate_dataset(self) -> None:
        """Build the ``(S, I)`` inputs and ``(S, O)`` targets on the host.

        For ``method="sample"`` the inputs are ``num_samples`` rows drawn
        uniformly per axis from ``np.random.default_rng(seed)``, so ``S`` equals
        ``num_samples``. For ``method="grid"`` the inputs are the cartesian
        product of the per-axis ``np.arange`` grids, so ``S`` is the product of
        the per-axis grid lengths.

        Either way the targets are ``np.apply_along_axis(func, -1, inputs)``,
        producing a scalar per row promoted to ``(S, 1)`` when ``func`` returns
        a scalar.

        Returns:
            ``None``; the arrays are stored on the instance for ``inputs`` and
            ``targets``.
        """
        if self.method == "sample":
            generator = np.random.default_rng(self.seed)
            inputs = generator.uniform(
                low=self.low_bounds,
                high=self.upper_bounds,
                size=(self.num_samples, self.low_bounds.shape[0]),
            )
        else:
            inputs = np.zeros((1, 1))
            for axis in range(self.low_bounds.shape[0]):
                grid = np.arange(
                    self.low_bounds[axis],
                    self.upper_bounds[axis],
                    self.step_size[axis],
                )
                inputs = cartesian_product(inputs, grid[:, None])
            inputs = inputs[:, 1:]
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(
            np.apply_along_axis(self.func, -1, inputs), dtype=np.float32
        )
        if targets.ndim == 1:
            targets = targets[:, None]
        self._inputs = inputs
        self._targets = targets

    @property
    def inputs(self) -> np.ndarray:
        """The generated inputs of shape ``(S, I)``.

        Returns:
            Float32 array with one row per sample.
        """
        return self._inputs

    @property
    def targets(self) -> np.ndarray:
        """The generated targets of shape ``(S, O)``.

        Returns:
            Float32 array with one row per sample.
        """
        return self._targets

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Per-sample input shape ``(I,)``.

        Returns:
            The number of bounds, without the sample axis.
        """
        return (int(self.low_bounds.shape[0]),)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Per-sample output shape ``(O,)``.

        Returns:
            The shape ``func`` returns for one point, without the sample axis.
        """
        return (int(self._targets.shape[1]),)


def cartesian_product(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Column-wise cartesian product of two ``(n, k)`` arrays.

    Rows of ``arr1`` are repeated so that each of them is paired with every row
    of ``arr2``, then the two column blocks are concatenated -- the same
    contract as the legacy helper, used to grow the grid one axis at a time.

    Args:
        arr1: Array of shape ``(n1, k1)``.
        arr2: Array of shape ``(n2, k2)`` with ``arr1.ndim == arr2.ndim``.

    Returns:
        Array of shape ``(n1 * n2, k1 + k2)`` holding every pair of rows. Rank-1
        inputs are treated as single columns, so the result is
        ``(n1 * n2, 2)``.

    Raises:
        TensorNEATError: If the inputs disagree in rank or exceed rank 2.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if arr1.ndim != arr2.ndim:
        raise TensorNEATError(
            "arr1 and arr2 must have the same number of dimensions"
        )
    if arr1.ndim > 2:
        raise TensorNEATError("arr1 and arr2 must have at most 2 dimensions")
    len1 = arr1.shape[0]
    len2 = arr2.shape[0]
    repeated_arr1 = np.repeat(arr1, len2, axis=0)
    if arr1.ndim == 1:
        return np.stack([repeated_arr1, np.tile(arr2, len1)], axis=1)
    tiled_arr2 = np.tile(arr2, (len1, 1))
    return np.concatenate((repeated_arr1, tiled_arr2), axis=1)
