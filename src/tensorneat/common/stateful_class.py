from typing import Optional
from . import State
import pickle
import datetime
import warnings


class StatefulBaseClass:
    def setup(self, state=State()):
        return state

    def save(self, state: Optional[State] = None, path: Optional[str] = None):
        if path is None:
            time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            path = f"./{self.__class__.__name__} {time}.pkl"
        if state is not None:
            self.__dict__["aux_for_state"] = state
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __getstate__(self):
        # only pickle the picklable attributes
        state = self.__dict__.copy()
        non_picklable_keys = []
        for key, value in state.items():
            try:
                pickle.dumps(value)
            except Exception:
                non_picklable_keys.append(key)

        for key in non_picklable_keys:
            state.pop(key)

        return state

    def show_config(self, registered_objects=None):
        if registered_objects is None:  # root call
            registered_objects = []

        config = {}
        for key, value in self.__dict__.items():
            if isinstance(value, StatefulBaseClass) and value not in registered_objects:
                registered_objects.append(value)
                config[str(key)] = value.show_config(registered_objects)

            else:
                config[str(key)] = str(value)
        return config

    @classmethod
    def load(cls, path: str, with_state: bool = False, warning: bool = True):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if with_state:
            if "aux_for_state" not in obj.__dict__:
                if warning:
                    warnings.warn(
                        "This object does not have state to load, return empty state",
                        category=UserWarning,
                    )
                return obj, State()
            state = obj.__dict__["aux_for_state"]
            del obj.__dict__["aux_for_state"]
            return obj, state
        else:
            if "aux_for_state" in obj.__dict__:
                if warning:
                    warnings.warn(
                        "This object has state to load, ignore it",
                        category=UserWarning,
                    )
                del obj.__dict__["aux_for_state"]
            return obj
