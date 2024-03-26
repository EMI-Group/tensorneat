from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class State:

    def __init__(self, **kwargs):
        self.__dict__['state_dict'] = kwargs

    def update(self, **kwargs):
        return State(**{**self.state_dict, **kwargs})

    def __getattr__(self, name):
        return self.state_dict[name]

    def __setattr__(self, name, value):
        raise AttributeError("State is immutable")

    def __repr__(self):
        return f"State ({self.state_dict})"

    def tree_flatten(self):
        children = list(self.state_dict.values())
        aux_data = list(self.state_dict.keys())
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))