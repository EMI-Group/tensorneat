"""Supervised function fitting: a fixed dataset plus a loss reduction."""

from .func_fit import FuncFit
from .xor import XOR
from .xor3d import XOR3d
from .custom import CustomFuncFit

__all__ = ["FuncFit", "XOR", "XOR3d", "CustomFuncFit"]
