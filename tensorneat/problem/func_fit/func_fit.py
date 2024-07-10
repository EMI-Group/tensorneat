import jax
import jax.numpy as jnp

from tensorneat.common import State
from .. import BaseProblem


class FuncFit(BaseProblem):
    jitable = True

    def __init__(self, error_method: str = "mse", return_data: bool = False):
        super().__init__()

        assert error_method in {"mse", "rmse", "mae", "mape"}
        self.error_method = error_method
        self.return_data = return_data

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state, randkey, act_func, params):

        predict = jax.vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )

        if self.error_method == "mse":
            loss = jnp.mean((predict - self.targets) ** 2)

        elif self.error_method == "rmse":
            loss = jnp.sqrt(jnp.mean((predict - self.targets) ** 2))

        elif self.error_method == "mae":
            loss = jnp.mean(jnp.abs(predict - self.targets))

        elif self.error_method == "mape":
            loss = jnp.mean(jnp.abs((predict - self.targets) / self.targets))

        else:
            raise NotImplementedError

        if self.return_data:
            return -loss, self.inputs
        else:
            return -loss

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        predict = jax.vmap(act_func, in_axes=(None, None, 0))(
            state, params, self.inputs
        )
        inputs, target, predict = jax.device_get([self.inputs, self.targets, predict])
        if self.return_data:
            loss, _ = self.evaluate(state, randkey, act_func, params)
        else:
            loss = self.evaluate(state, randkey, act_func, params)
        loss = -loss

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
