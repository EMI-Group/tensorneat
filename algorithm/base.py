from typing import Callable

from .state import State

EMPTY = lambda *args: args


class Algorithm:

    def __init__(self):
        self.tell: Callable = EMPTY
        self.ask: Callable = EMPTY
        self.forward: Callable = EMPTY
        self.forward_transform: Callable = EMPTY

    def setup(self, randkey, state=State()):
        pass
