from typing import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from config import ProblemConfig
from core import Problem, State


@dataclass(frozen=True)
class FuncFitConfig(ProblemConfig):
    error_method: str = 'mse'

    def __post_init__(self):
        assert self.error_method in {'mse', 'rmse', 'mae', 'mape'}


class FuncFit(Problem):

    jitable = True

    def __init__(self, config: FuncFitConfig = FuncFitConfig()):
        self.config = config
        super().__init__(config)

    def evaluate(self, randkey, state: State, act_func: Callable, params):

        predict = act_func(state, self.inputs, params)

        if self.config.error_method == 'mse':
            loss = jnp.mean((predict - self.targets) ** 2)

        elif self.config.error_method == 'rmse':
            loss = jnp.sqrt(jnp.mean((predict - self.targets) ** 2))

        elif self.config.error_method == 'mae':
            loss = jnp.mean(jnp.abs(predict - self.targets))

        elif self.config.error_method == 'mape':
            loss = jnp.mean(jnp.abs((predict - self.targets) / self.targets))

        else:
            raise NotImplementedError

        return -loss

    def show(self, randkey, state: State, act_func: Callable, params):
        predict = act_func(state, self.inputs, params)
        inputs, target, predict = jax.device_get([self.inputs, self.targets, predict])
        loss = -self.evaluate(randkey, state, act_func, params)
        msg = ""
        for i in range(inputs.shape[0]):
            msg += f"input: {inputs[i]}, target: {target[i]}, predict: {predict[i]}\n"
        msg += f"loss: {loss}\n"
        print(msg)

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError
