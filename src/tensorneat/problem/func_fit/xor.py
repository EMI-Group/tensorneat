"""XOR over ``{0, 1}^2`` as a function-fitting problem."""

import numpy as np

from .func_fit import FuncFit

__all__ = ["XOR"]


class XOR(FuncFit):
    """Two-input XOR: four samples, two inputs, one output.

    Dataset literals (float32):

    - inputs ``[[0, 0], [0, 1], [1, 0], [1, 1]]`` (``S = 4``, ``I = 2``)
    - targets ``[[0], [1], [1], [0]]`` (``O = 1``)

    Shapes are per sample: ``input_shape = (2,)`` and ``output_shape = (1,)``.
    """

    def __init__(self, error_method: str = "mse") -> None:
        """Configure the loss reduction.

        Args:
            error_method: One of ``"mse"``, ``"rmse"``, ``"mae"``, ``"mape"``;
                forwarded to ``FuncFit``.

        Raises:
            TensorNEATError: If ``error_method`` is not supported.
        """
        super().__init__(error_method=error_method)

    @property
    def inputs(self) -> np.ndarray:
        """The four XOR inputs of shape ``(S, I) = (4, 2)``.

        Returns:
            Float32 array ``[[0, 0], [0, 1], [1, 0], [1, 1]]``.
        """
        return np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            dtype=np.float32,
        )

    @property
    def targets(self) -> np.ndarray:
        """The XOR targets of shape ``(S, O) = (4, 1)``.

        Returns:
            Float32 array ``[[0], [1], [1], [0]]``.
        """
        return np.array(
            [[0], [1], [1], [0]],
            dtype=np.float32,
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Per-sample input shape ``(2,)``.

        Returns:
            ``(2,)``, ``S`` is not part of this shape.
        """
        return (2,)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Per-sample output shape ``(1,)``.

        Returns:
            ``(1,)``, ``S`` is not part of this shape.
        """
        return (1,)
