import numpy as np
import sympy as sp


class SympySum(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Add(*z)


class SympyProduct(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Mul(*z)


class SympyMax(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Max(*z)


class SympyMin(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Min(*z)


class SympyMaxabs(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Max(*z, key=sp.Abs)


class SympyMean(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Add(*z) / len(z)


class SympyMedian(sp.Function):
    @classmethod
    def eval(cls, args):

        if all(arg.is_number for arg in args):
            sorted_args = sorted(args)
            n = len(sorted_args)
            if n % 2 == 1:
                return sorted_args[n // 2]
            else:
                return (sorted_args[n // 2 - 1] + sorted_args[n // 2]) / 2

        return None

    def _sympystr(self, printer):
        return f"median({', '.join(map(str, self.args))})"

    def _latex(self, printer):
        return (
            r"\mathrm{median}\left(" + ", ".join(map(sp.latex, self.args)) + r"\right)"
        )
