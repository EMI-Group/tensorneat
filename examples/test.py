from functools import partial
import jax


class A:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.isTrue = False

    @partial(jax.jit, static_argnums=(0,))
    def step(self):
        if self.isTrue:
            return self.a + 1
        else:
            return self.b + 1


AA = A()
print(AA.step(), hash(AA))
print(AA.step(), hash(AA))
print(AA.step(), hash(AA))
AA.a = (2, 3, 4)
print(AA.step(), hash(AA))
