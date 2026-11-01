"""3-bit parity as a function-fitting problem."""

import numpy as np

from .func_fit import FuncFit

__all__ = ["XOR3d"]


class XOR3d(FuncFit):
    """Three-input parity: eight samples, three inputs, one output.

    Dataset literals (float32):

    - inputs ``[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
      [1, 0, 1], [1, 1, 0], [1, 1, 1]]`` (``S = 8``, ``I = 3``)
    - targets ``[[0], [1], [1], [0], [1], [0], [0], [1]]`` (3-bit parity,
      ``O = 1``)

    Shapes are per sample: ``input_shape = (3,)`` and ``output_shape = (1,)``.
    """

    def __init__(self, error_method: str = "mse") -> None:
        """Configure the loss reduction.

        Args:
            error_method: One of ``"mse"``, ``"rmse"``, ``"mae"``, ``"mape"``;
                forwarded to ``FuncFit``.

        Raises:
            TensorNEATError: If ``error_method`` is not supported.
        """
        raise NotImplementedError

    @property
    def inputs(self) -> np.ndarray:
        """The eight parity inputs of shape ``(S, I) = (8, 3)``.

        Returns:
            Float32 array of all three-bit patterns in counting order.
        """
        raise NotImplementedError

    @property
    def targets(self) -> np.ndarray:
        """The 3-bit parity targets of shape ``(S, O) = (8, 1)``.

        Returns:
            Float32 array ``[[0], [1], [1], [0], [1], [0], [0], [1]]``.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Per-sample input shape ``(3,)``.

        Returns:
            ``(3,)``, ``S`` is not part of this shape.
        """
        raise NotImplementedError

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Per-sample output shape ``(1,)``.

        Returns:
            ``(1,)``, ``S`` is not part of this shape.
        """
        raise NotImplementedError
