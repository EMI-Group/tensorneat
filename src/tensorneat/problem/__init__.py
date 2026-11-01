"""Problem (task / fitness) layer: function fitting and the evoxc CartPole rollout."""

from .base import BaseProblem
from .func_fit import CustomFuncFit, FuncFit, XOR, XOR3d
from .rl import CartPole

__all__ = ["BaseProblem", "FuncFit", "XOR", "XOR3d", "CustomFuncFit", "CartPole"]
