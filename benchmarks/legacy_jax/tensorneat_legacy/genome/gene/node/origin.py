import jax, jax.numpy as jnp
from .default import DefaultNode


class OriginNode(DefaultNode):
    """
    Implementation of nodes in origin NEAT Paper.
    Details at https://github.com/EMI-Group/tensorneat/issues/11.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def crossover(self, state, randkey, attrs1, attrs2):
        # random pick one of attrs, without attrs exchange
        return jnp.where(
            # origin code, generate multiple random numbers, without attrs exchange
            # jax.random.normal(randkey, attrs1.shape) > 0,
            jax.random.normal(randkey)
            > 0,  # generate one random number, without attrs exchange
            attrs1,
            attrs2,
        )
