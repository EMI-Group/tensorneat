from typing import Callable

from config import ProblemConfig
from core.state import State


class BaseProblem:

    jitable = None

    def __init__(self):
        pass

    def setup(self, randkey, state: State = State()):
        """initialize the state of the problem"""
        raise NotImplementedError

    def evaluate(self, randkey, state: State, act_func: Callable, params):
        """evaluate one individual"""
        raise NotImplementedError

    @property
    def input_shape(self):
        """
        The input shape for the problem to evaluate
        In RL problem, it is the observation space
        In function fitting problem, it is the input shape of the function
        """
        raise NotImplementedError

    @property
    def output_shape(self):
        """
        The output shape for the problem to evaluate
        In RL problem, it is the action space
        In function fitting problem, it is the output shape of the function
        """
        raise NotImplementedError

    def show(self, randkey, state: State, act_func: Callable, params, *args, **kwargs):
        """
        show how a genome perform in this problem
        """
        raise NotImplementedError
