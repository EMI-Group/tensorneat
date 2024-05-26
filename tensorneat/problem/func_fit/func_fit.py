import jax
import jax.numpy as jnp

from utils import State
from .. import BaseProblem


class FuncFit(BaseProblem):
    jitable = True

    def __init__(self,
                 error_method: str = 'mse'
                 ):
        super().__init__()

        assert error_method in {'mse', 'rmse', 'mae', 'mape'}
        self.error_method = error_method

    def setup(self, state: State = State()):
        return state

    def evaluate(self, randkey, state, act_func, params):

        state, predict = jax.vmap(act_func, in_axes=(None, 0, None), out_axes=(None, 0))(state, self.inputs, params)

        if self.error_method == 'mse':
            loss = jnp.mean((predict - self.targets) ** 2)

        elif self.error_method == 'rmse':
            loss = jnp.sqrt(jnp.mean((predict - self.targets) ** 2))

        elif self.error_method == 'mae':
            loss = jnp.mean(jnp.abs(predict - self.targets))

        elif self.error_method == 'mape':
            loss = jnp.mean(jnp.abs((predict - self.targets) / self.targets))

        else:
            raise NotImplementedError

        return state, -loss

    def show(self, randkey, state, act_func, params, *args, **kwargs):
        state, predict = jax.vmap(act_func, in_axes=(None, 0, None), out_axes=(None, 0))(state, self.inputs, params)
        inputs, target, predict = jax.device_get([self.inputs, self.targets, predict])
        state, loss = self.evaluate(randkey, state, act_func, params)
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
