from typing import Callable

from utils import State


class BaseProblem:
    jitable = None

    def setup(self, state: State = State()):
        """initialize the state of the problem"""
        return state

    def evaluate(self, state: State, randkey, act_func: Callable, params):
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

    def show(self, state: State, randkey, act_func: Callable, params, *args, **kwargs):
        """
        show how a genome perform in this problem
        """
        raise NotImplementedError
