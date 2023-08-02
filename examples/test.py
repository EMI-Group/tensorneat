from functools import partial
import jax

from utils import unflatten_conns, act, agg, Activation, Aggregation
from algorithm.neat.gene import RecurrentGeneConfig

config = RecurrentGeneConfig(
    activation_options=("tanh", "sigmoid"),
    activation_default="tanh",
)


class A:
    def __init__(self):
        self.act_funcs = [Activation.name2func[name] for name in config.activation_options]
        self.agg_funcs = [Aggregation.name2func[name] for name in config.aggregation_options]
        self.isTrue = False

    @partial(jax.jit, static_argnums=(0,))
    def step(self):
        i = jax.numpy.array([0, 1])
        z = jax.numpy.array([
            [1, 1],
            [2, 2]
        ])
        print(self.act_funcs)
        return jax.vmap(act, in_axes=(0, 0, None))(i, z, self.act_funcs)


AA = A()
print(AA.step())
