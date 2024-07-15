import pickle

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class State:
    def __init__(self, **kwargs):
        self.__dict__["state_dict"] = kwargs

    def registered_keys(self):
        return self.state_dict.keys()

    def register(self, **kwargs):
        for key in kwargs:
            if key in self.registered_keys():
                raise ValueError(f"Key {key} already exists in state")
        return State(**{**self.state_dict, **kwargs})

    def update(self, **kwargs):
        for key in kwargs:
            if key not in self.registered_keys():
                raise ValueError(f"Key {key} does not exist in state")
        return State(**{**self.state_dict, **kwargs})

    def __getattr__(self, name):
        return self.state_dict[name]

    def __setattr__(self, name, value):
        raise AttributeError("State is immutable")

    def __repr__(self):
        return f"State ({self.state_dict})"

    def __getstate__(self):
        return self.state_dict.copy()

    def __setstate__(self, state):
        self.__dict__["state_dict"] = state

    def __contains__(self, item):
        return item in self.state_dict

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def tree_flatten(self):
        children = list(self.state_dict.values())
        aux_data = list(self.state_dict.keys())
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))
