from functools import partial
import numpy as np
import jax.numpy as jnp
from typing import Union, Callable
import sympy as sp


class FunctionManager:

    def __init__(self, name2jnp, name2sympy):
        self.name2jnp = name2jnp
        self.name2sympy = name2sympy
        for name, func in name2jnp.items():
            setattr(self, name, func)

    def get_all_funcs(self):
        all_funcs = []
        for name in self.name2jnp:
            all_funcs.append(getattr(self, name))
        return all_funcs

    def add_func(self, name, func):
        if not callable(func):
            raise ValueError("The provided function is not callable")
        if name in self.name2jnp:
            raise ValueError(f"The provided name={name} is already in use")

        self.name2jnp[name] = func
        setattr(self, name, func)

    def update_sympy(self, name, sympy_cls: sp.Function):
        self.name2sympy[name] = sympy_cls

    def obtain_sympy(self, func: Union[str, Callable]):
        if isinstance(func, str):
            if func not in self.name2sympy:
                raise ValueError(f"Func {func} doesn't have a sympy representation.")
            return self.name2sympy[func]

        elif isinstance(func, Callable):
            # try to find name
            for name, f in self.name2jnp.items():
                if f == func:
                    return self._obtain_sympy_by_name(name)
            raise ValueError(f"Func {func} doesn't not registered.")

        else:
            raise ValueError(f"Func {func} need be a string or callable.")

    def _obtain_sympy_by_name(self, name: str):
        if name not in self.name2sympy:
            raise ValueError(f"Func {name} doesn't have a sympy representation.")
        return self.name2sympy[name]

    def sympy_module(self, backend: str):
        assert backend in ["jax", "numpy"]
        if backend == "jax":
            backend = jnp
        elif backend == "numpy":
            backend = np
        module = {}
        for sympy_cls in self.name2sympy.values():
            if hasattr(sympy_cls, "numerical_eval"):
                module[sympy_cls.__name__] = partial(sympy_cls.numerical_eval, backend)

        return module
