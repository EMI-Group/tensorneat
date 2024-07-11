from typing import Callable, Union, List, Tuple, Sequence

import jax
from jax import vmap, Array, numpy as jnp
import numpy as np

from .func_fit import FuncFit


class CustomFuncFit(FuncFit):

    def __init__(
        self,
        func: Callable,
        low_bounds: Union[List, Tuple, Array],
        upper_bounds: Union[List, Tuple, Array],
        method: str = "sample",
        num_samples: int = 100,
        step_size: Array = None,
        *args,
        **kwargs,
    ):

        if isinstance(low_bounds, list) or isinstance(low_bounds, tuple):
            low_bounds = np.array(low_bounds, dtype=np.float32)
        if isinstance(upper_bounds, list) or isinstance(upper_bounds, tuple):
            upper_bounds = np.array(upper_bounds, dtype=np.float32)

        try:
            out = func(low_bounds)
        except Exception as e:
            raise ValueError(f"func(low_bounds) raise an exception: {e}")
        assert low_bounds.shape == upper_bounds.shape

        assert method in {"sample", "grid"}

        self.func = func
        self.low_bounds = low_bounds
        self.upper_bounds = upper_bounds

        self.method = method
        self.num_samples = num_samples
        self.step_size = step_size

        self.generate_dataset()

        super().__init__(*args, **kwargs)

    def generate_dataset(self):

        if self.method == "sample":
            assert (
                self.num_samples > 0
            ), f"num_samples must be positive, got {self.num_samples}"

            inputs = np.zeros(
                (self.num_samples, self.low_bounds.shape[0]), dtype=np.float32
            )
            for i in range(self.low_bounds.shape[0]):
                inputs[:, i] = np.random.uniform(
                    low=self.low_bounds[i],
                    high=self.upper_bounds[i],
                    size=(self.num_samples,),
                )
        elif self.method == "grid":
            assert (
                self.step_size is not None
            ), "step_size must be provided when method is 'grid'"
            assert (
                self.step_size.shape == self.low_bounds.shape
            ), "step_size must have the same shape as low_bounds"
            assert np.all(self.step_size > 0), "step_size must be positive"

            inputs = np.zeros((1, 1))
            for i in range(self.low_bounds.shape[0]):
                new_col = np.arange(
                    self.low_bounds[i], self.upper_bounds[i], self.step_size[i]
                )
                inputs = cartesian_product(inputs, new_col[:, None])
            inputs = inputs[:, 1:]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        outputs = vmap(self.func)(inputs)

        self.data_inputs = jnp.array(inputs)
        self.data_outputs = jnp.array(outputs)

    @property
    def inputs(self):
        return self.data_inputs

    @property
    def targets(self):
        return self.data_outputs

    @property
    def input_shape(self):
        return self.data_inputs.shape

    @property
    def output_shape(self):
        return self.data_outputs.shape


def cartesian_product(arr1, arr2):
    assert (
        arr1.ndim == arr2.ndim
    ), "arr1 and arr2 must have the same number of dimensions"
    assert arr1.ndim <= 2, "arr1 and arr2 must have at most 2 dimensions"

    len1 = arr1.shape[0]
    len2 = arr2.shape[0]

    repeated_arr1 = np.repeat(arr1, len2, axis=0)
    tiled_arr2 = np.tile(arr2, (len1, 1))

    new_arr = np.concatenate((repeated_arr1, tiled_arr2), axis=1)
    return new_arr
