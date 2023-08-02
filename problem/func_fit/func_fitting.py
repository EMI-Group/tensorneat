from dataclasses import dataclass
from typing import Callable

from config import ProblemConfig
from core import Problem, State


@dataclass(frozen=True)
class FuncFitConfig:
    pass


class FuncFit(Problem):
    def __init__(self, config: ProblemConfig):
        self.config = ProblemConfig

    def setup(self, state=State()):
        pass

    def evaluate(self, state: State, act_func: Callable, params):
        pass