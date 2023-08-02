from typing import Callable
from config import ProblemConfig
from state import State


class Problem:

    def __init__(self, config: ProblemConfig):
        raise NotImplementedError

    def setup(self, state=State()):
        raise NotImplementedError

    def evaluate(self, state: State, act_func: Callable, params):
        raise NotImplementedError
