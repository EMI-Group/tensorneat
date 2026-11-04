"""Problem (task / fitness) layer: function fitting plus the evoxc RL rollouts."""

from .base import BaseProblem
from .func_fit import CustomFuncFit, FuncFit, XOR, XOR3d
from .rl import CartPole, MountainCarContinuous, Pendulum

__all__ = [
    "BaseProblem",
    "FuncFit",
    "XOR",
    "XOR3d",
    "CustomFuncFit",
    "CartPole",
    "MountainCarContinuous",
    "Pendulum",
]
