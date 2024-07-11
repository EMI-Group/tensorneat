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
