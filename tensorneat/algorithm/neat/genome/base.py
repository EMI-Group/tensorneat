import numpy as np
import jax, jax.numpy as jnp
from ..gene import BaseNodeGene, BaseConnGene
from ..ga import BaseMutation, BaseCrossover
from utils import State, StatefulBaseClass, topological_sort_python


class BaseGenome(StatefulBaseClass):
    network_type = None

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes: int,
        max_conns: int,
        node_gene: BaseNodeGene,
        conn_gene: BaseConnGene,
        mutation: BaseMutation,
        crossover: BaseCrossover,
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_idx = np.arange(num_inputs)
        self.output_idx = np.arange(num_inputs, num_inputs + num_outputs)
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self.node_gene = node_gene
        self.conn_gene = conn_gene
        self.mutation = mutation
        self.crossover = crossover

    def setup(self, state=State()):
        state = self.node_gene.setup(state)
        state = self.conn_gene.setup(state)
        state = self.mutation.setup(state)
        state = self.crossover.setup(state)
        return state

    def transform(self, state, nodes, conns):
        raise NotImplementedError

    def restore(self, state, transformed):
        raise NotImplementedError

    def forward(self, state, transformed, inputs):
        raise NotImplementedError

    def execute_mutation(self, state, randkey, nodes, conns, new_node_key):
        return self.mutation(state, randkey, self, nodes, conns, new_node_key)

    def execute_crossover(self, state, randkey, nodes1, conns1, nodes2, conns2):
        return self.crossover(state, randkey, self, nodes1, conns1, nodes2, conns2)

    def initialize(self, state, randkey):
        """
        Default initialization method for the genome.
        Add an extra hidden node.
        Make all input nodes and output nodes connected to the hidden node.
        All attributes will be initialized randomly using gene.new_random_attrs method.

        For example, a network with 2 inputs and 1 output, the structure will be:
        nodes:
            [
                [0, attrs0],  # input node 0
                [1, attrs1],  # input node 1
                [2, attrs2],  # output node 0
                [3, attrs3],  # hidden node
                [NaN, NaN],  # empty node
            ]
        conns:
            [
                [0, 3, attrs0],  # input node 0 -> hidden node
                [1, 3, attrs1],  # input node 1 -> hidden node
                [3, 2, attrs2], # hidden node -> output node 0
                [NaN, NaN],
                [NaN, NaN],
            ]
        """

        k1, k2 = jax.random.split(randkey)  # k1 for nodes, k2 for conns
        # initialize nodes
        new_node_key = (
            max([*self.input_idx, *self.output_idx]) + 1
        )  # the key for the hidden node
        node_keys = jnp.concatenate(
            [self.input_idx, self.output_idx, jnp.array([new_node_key])]
        )  # the list of all node keys

        # initialize nodes and connections with NaN
        nodes = jnp.full((self.max_nodes, self.node_gene.length), jnp.nan)
        conns = jnp.full((self.max_conns, self.conn_gene.length), jnp.nan)

        # set keys for input nodes, output nodes and hidden node
        nodes = nodes.at[node_keys, 0].set(node_keys)

        # generate random attributes for nodes
        node_keys = jax.random.split(k1, len(node_keys))
        random_node_attrs = jax.vmap(
            self.node_gene.new_random_attrs, in_axes=(None, 0)
        )(state, node_keys)
        nodes = nodes.at[: len(node_keys), 1:].set(random_node_attrs)

        # initialize conns
        # input-hidden connections
        input_conns = jnp.c_[
            self.input_idx, jnp.full_like(self.input_idx, new_node_key)
        ]
        conns = conns.at[self.input_idx, :2].set(input_conns)  # in-keys, out-keys

        # output-hidden connections
        output_conns = jnp.c_[
            jnp.full_like(self.output_idx, new_node_key), self.output_idx
        ]
        conns = conns.at[self.output_idx, :2].set(output_conns)  # in-keys, out-keys

        conn_keys = jax.random.split(k2, num=len(self.input_idx) + len(self.output_idx))
        # generate random attributes for conns
        random_conn_attrs = jax.vmap(
            self.conn_gene.new_random_attrs, in_axes=(None, 0)
        )(state, conn_keys)
        conns = conns.at[: len(conn_keys), 2:].set(random_conn_attrs)

        return nodes, conns

    def update_by_batch(self, state, batch_input, transformed):
        """
        Update the genome by a batch of data.
        """
        raise NotImplementedError

    def repr(self, state, nodes, conns, precision=2):
        nodes, conns = jax.device_get([nodes, conns])
        nodes_cnt, conns_cnt = self.valid_cnt(nodes), self.valid_cnt(conns)
        s = f"{self.__class__.__name__}(nodes={nodes_cnt}, conns={conns_cnt}):\n"
        s += f"\tNodes:\n"
        for node in nodes:
            if np.isnan(node[0]):
                break
            s += f"\t\t{self.node_gene.repr(state, node, precision=precision)}"
            node_idx = int(node[0])
            if np.isin(node_idx, self.input_idx):
                s += " (input)"
            elif np.isin(node_idx, self.output_idx):
                s += " (output)"
            s += "\n"

        s += f"\tConns:\n"
        for conn in conns:
            if np.isnan(conn[0]):
                break
            s += f"\t\t{self.conn_gene.repr(state, conn, precision=precision)}\n"
        return s

    @classmethod
    def valid_cnt(cls, arr):
        return jnp.sum(~jnp.isnan(arr[:, 0]))

    def get_conn_dict(self, state, conns):
        conns = jax.device_get(conns)
        conn_dict = {}
        for conn in conns:
            if np.isnan(conn[0]):
                continue
            cd = self.conn_gene.to_dict(state, conn)
            in_idx, out_idx = cd["in"], cd["out"]
            del cd["in"], cd["out"]
            conn_dict[(in_idx, out_idx)] = cd
        return conn_dict

    def get_node_dict(self, state, nodes):
        nodes = jax.device_get(nodes)
        node_dict = {}
        for node in nodes:
            if np.isnan(node[0]):
                continue
            nd = self.node_gene.to_dict(state, node)
            idx = nd["idx"]
            del nd["idx"]
            node_dict[idx] = nd
        return node_dict

    def network_dict(self, state, nodes, conns):
        return {
            "nodes": self.get_node_dict(state, nodes),
            "conns": self.get_conn_dict(state, conns),
        }

    def get_input_idx(self):
        return self.input_idx.tolist()

    def get_output_idx(self):
        return self.output_idx.tolist()

    def sympy_func(self, state, network, precision=3):
        raise NotImplementedError

    def visualize(
        self,
        network,
        rotate=0,
        reverse_node_order=False,
        size=(300, 300, 300),
        color=("blue", "blue", "blue"),
        save_path="network.svg",
        save_dpi=800,
        **kwargs,
    ):
        import networkx as nx
        from matplotlib import pyplot as plt

        nodes_list = list(network["nodes"])
        conns_list = list(network["conns"])
        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()
        topo_order, topo_layers = topological_sort_python(nodes_list, conns_list)
        node2layer = {
            node: layer for layer, nodes in enumerate(topo_layers) for node in nodes
        }
        if reverse_node_order:
            topo_order = topo_order[::-1]

        G = nx.DiGraph()

        if not isinstance(size, tuple):
            size = (size, size, size)
        if not isinstance(color, tuple):
            color = (color, color, color)

        for node in topo_order:
            if node in input_idx:
                G.add_node(node, subset=node2layer[node], size=size[0], color=color[0])
            elif node in output_idx:
                G.add_node(node, subset=node2layer[node], size=size[2], color=color[2])
            else:
                G.add_node(node, subset=node2layer[node], size=size[1], color=color[1])

        for conn in conns_list:
            G.add_edge(conn[0], conn[1])
        pos = nx.multipartite_layout(G)

        def rotate_layout(pos, angle):
            angle_rad = np.deg2rad(angle)
            cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
            rotated_pos = {}
            for node, (x, y) in pos.items():
                rotated_pos[node] = (
                    cos_angle * x - sin_angle * y,
                    sin_angle * x + cos_angle * y,
                )
            return rotated_pos

        rotated_pos = rotate_layout(pos, rotate)

        node_sizes = [n["size"] for n in G.nodes.values()]
        node_colors = [n["color"] for n in G.nodes.values()]

        nx.draw(
            G,
            with_labels=True,
            pos=rotated_pos,
            node_size=node_sizes,
            node_color=node_colors,
            **kwargs,
        )
        plt.savefig(save_path, dpi=save_dpi)
