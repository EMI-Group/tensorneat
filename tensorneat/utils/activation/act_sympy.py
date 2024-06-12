from typing import Union
import sympy as sp
import numpy as np


class SympyClip(sp.Function):
    @classmethod
    def eval(cls, val, min_val, max_val):
        if val.is_Number and min_val.is_Number and max_val.is_Number:
            return sp.Piecewise(
                (min_val, val < min_val), (max_val, val > max_val), (val, True)
            )
        return None

    @staticmethod
    def numerical_eval(val, min_val, max_val):
        return np.clip(val, min_val, max_val)

    def _sympystr(self, printer):
        return f"clip({self.args[0]}, {self.args[1]}, {self.args[2]})"

    def _latex(self, printer):
        return rf"\mathrm{{clip}}\left({sp.latex(self.args[0])}, {self.args[1]}, {self.args[2]}\right)"


class SympySigmoid(sp.Function):
    @classmethod
    def eval(cls, z):
        if z.is_Number:
            z = SympyClip(5 * z, -10, 10)
            return 1 / (1 + sp.exp(-z))
        return None

    @staticmethod
    def numerical_eval(z):
        z = np.clip(5 * z, -10, 10)
        return 1 / (1 + np.exp(-z))

    def _sympystr(self, printer):
        return f"sigmoid({self.args[0]})"

    def _latex(self, printer):
        return rf"\mathrm{{sigmoid}}\left({sp.latex(self.args[0])}\right)"


class SympyTanh(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.tanh(0.6 * z)

    @staticmethod
    def numerical_eval(z):
        return np.tanh(0.6 * z)


class SympySin(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.sin(z)

    @staticmethod
    def numerical_eval(z):
        return np.sin(z)


class SympyRelu(sp.Function):
    @classmethod
    def eval(cls, z):
        if z.is_Number:
            return sp.Piecewise((z, z > 0), (0, True))
        return None

    @staticmethod
    def numerical_eval(z):
        return np.maximum(z, 0)

    def _sympystr(self, printer):
        return f"relu({self.args[0]})"

    def _latex(self, printer):
        return rf"\mathrm{{relu}}\left({sp.latex(self.args[0])}\right)"


class SympyLelu(sp.Function):
    @classmethod
    def eval(cls, z):
        if z.is_Number:
            leaky = 0.005
            return sp.Piecewise((z, z > 0), (leaky * z, True))
        return None

    @staticmethod
    def numerical_eval(z):
        leaky = 0.005
        return np.maximum(z, leaky * z)

    def _sympystr(self, printer):
        return f"lelu({self.args[0]})"

    def _latex(self, printer):
        return rf"\mathrm{{lelu}}\left({sp.latex(self.args[0])}\right)"


class SympyIdentity(sp.Function):
    @classmethod
    def eval(cls, z):
        return z

    @staticmethod
    def numerical_eval(z):
        return z


class SympyClamped(sp.Function):
    @classmethod
    def eval(cls, z):
        return SympyClip(z, -1, 1)

    @staticmethod
    def numerical_eval(z):
        return np.clip(z, -1, 1)


class SympyInv(sp.Function):
    @classmethod
    def eval(cls, z):
        if z.is_Number:
            z = sp.Piecewise((sp.Max(z, 1e-7), z > 0), (sp.Min(z, -1e-7), True))
            return 1 / z
        return None

    @staticmethod
    def numerical_eval(z):
        z = np.maximum(z, 1e-7)
        return 1 / z

    def _sympystr(self, printer):
        return f"1 / {self.args[0]}"

    def _latex(self, printer):
        return rf"\frac{{1}}{{{sp.latex(self.args[0])}}}"


class SympyLog(sp.Function):
    @classmethod
    def eval(cls, z):
        if z.is_Number:
            z = sp.Max(z, 1e-7)
            return sp.log(z)
        return None

    @staticmethod
    def numerical_eval(z):
        z = np.maximum(z, 1e-7)
        return np.log(z)

    def _sympystr(self, printer):
        return f"log({self.args[0]})"

    def _latex(self, printer):
        return rf"\mathrm{{log}}\left({sp.latex(self.args[0])}\right)"


class SympyExp(sp.Function):
    @classmethod
    def eval(cls, z):
        if z.is_Number:
            z = SympyClip(z, -10, 10)
            return sp.exp(z)
        return None

    @staticmethod
    def numerical_eval(z):
        z = np.clip(z, -10, 10)
        return np.exp(z)

    def _sympystr(self, printer):
        return f"exp({self.args[0]})"

    def _latex(self, printer):
        return rf"\mathrm{{exp}}\left({sp.latex(self.args[0])}\right)"


class SympyAbs(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Abs(z)

    @staticmethod
    def numerical_eval(z):
        return np.abs(z)