"""Function fitting against a user-supplied function, sampled or gridded."""

from typing import Any, Callable

import numpy as np

from .func_fit import FuncFit
from ...common.errors import TensorNEATError

__all__ = ["CustomFuncFit", "cartesian_product"]


class CustomFuncFit(FuncFit):
    """Fit a genome to an arbitrary Python function over a box.

    The dataset is generated host-side at construction time and then baked into
    the graph, so it is a compile-time constant: keep ``S * I`` under roughly
    250k floats, since evoxc warns above ~1 MiB of constant payload.
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

        - ``method="sample"`` draws ``num_samples`` points per axis from the box
          ``[low_bounds, upper_bounds]`` using ``np.random.default_rng(seed)``.
          The generator is host-side only: the legacy version mutated the global
          ``np.random`` state, which made runs irreproducible without also
          seeding the process globally.
        - ``method="grid"`` builds the cartesian product of
          ``np.arange(low_bounds[i], upper_bounds[i], step_size[i])`` over every
          axis, so ``step_size`` must have the same shape as the bounds and be
          strictly positive.

        Args:
            func: The target function, ``(I,) -> scalar | (O,)``.
            low_bounds: Lower corner of the box, one entry per input.
            upper_bounds: Upper corner of the box, same shape as ``low_bounds``.
            method: ``"sample"`` or ``"grid"``.
            num_samples: Number of samples per axis for ``method="sample"``.
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
                ``low_bounds.shape != upper_bounds.shape``, or if
                ``func(low_bounds)`` raises.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def inputs(self) -> np.ndarray:
        """The generated inputs of shape ``(S, I)``.

        Returns:
            Float32 array with one row per sample.
        """
        raise NotImplementedError

    @property
    def targets(self) -> np.ndarray:
        """The generated targets of shape ``(S, O)``.

        Returns:
            Float32 array with one row per sample.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Per-sample input shape ``(I,)``.

        Returns:
            The number of bounds, without the sample axis.
        """
        raise NotImplementedError

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Per-sample output shape ``(O,)``.

        Returns:
            The shape ``func`` returns for one point, without the sample axis.
        """
        raise NotImplementedError


def cartesian_product(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Column-wise cartesian product of two ``(n, k)`` arrays.

    Rows of ``arr1`` are repeated so that each of them is paired with every row
    of ``arr2``, then the two column blocks are concatenated -- the same
    contract as the legacy helper, used to grow the grid one axis at a time.

    Args:
        arr1: Array of shape ``(n1, k1)``.
        arr2: Array of shape ``(n2, k2)`` with ``arr1.ndim == arr2.ndim``.

    Returns:
        Array of shape ``(n1 * n2, k1 + k2)`` holding every pair of rows.

    Raises:
        AssertionError: If the inputs disagree in rank or exceed rank 2.
    """
    raise NotImplementedError
