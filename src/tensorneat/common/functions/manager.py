from typing import Union, Callable
import sympy as sp

class FunctionManager:

    def __init__(self, name2jnp, name2sympy):
        self.name2jnp = name2jnp
        self.name2sympy = name2sympy

    def get_all_funcs(self):
        all_funcs = []
        for name in self.names:
            all_funcs.append(getattr(self, name))
        return all_funcs

    def __getattribute__(self, name: str):
        return self.name2jnp[name]

    def add_func(self, name, func):
        if not callable(func):
            raise ValueError("The provided function is not callable")
        if name in self.names:
            raise ValueError(f"The provided name={name} is already in use")

        self.name2jnp[name] = func

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
