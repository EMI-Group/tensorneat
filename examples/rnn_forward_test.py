import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Genome:
    def __init__(self, nodes, conns):
        self.nodes = nodes
        self.conns = conns

    def update_nodes(self, nodes):
        return Genome(nodes, self.conns)

    def update_conns(self, conns):
        return Genome(self.nodes, conns)

    def tree_flatten(self):
        children = self.nodes, self.conns
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self):
        return f"Genome ({self.nodes}, \n\t{self.conns})"

    @jax.jit
    def add_node(self, a: int):
        nodes = self.nodes.at[0, :].set(a)
        return self.update_nodes(nodes)


nodes, conns = jnp.array([[1, 2, 3, 4, 5]]), jnp.array([[1, 2, 3, 4]])
g = Genome(nodes, conns)
print(g)

g = g.add_node(1)
print(g)

g = jax.jit(g.add_node)(2)
print(g)
