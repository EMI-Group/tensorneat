import warnings

import jax
from jax import vmap, numpy as jnp
import numpy as np
import sympy as sp

from .base import BaseGenome
from .gene import DefaultNodeGene, DefaultConnGene
from .operations import DefaultMutation, DefaultCrossover, DefaultDistance
from .utils import unflatten_conns, extract_node_attrs, extract_conn_attrs

from tensorneat.common import (
    topological_sort,
    topological_sort_python,
    I_INF,
    attach_with_inf,
    SYMPY_FUNCS_MODULE_NP,
    SYMPY_FUNCS_MODULE_JNP,
)


class DefaultGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=50,
        max_conns=100,
        node_gene=DefaultNodeGene(),
        conn_gene=DefaultConnGene(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        output_transform=None,
        input_transform=None,
        init_hidden_layers=(),
    ):

        super().__init__(
            num_inputs,
            num_outputs,
            max_nodes,
            max_conns,
            node_gene,
            conn_gene,
            mutation,
            crossover,
            distance,
            output_transform,
            input_transform,
            init_hidden_layers,
        )

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        conn_exist = u_conns != I_INF

        seqs = topological_sort(nodes, conn_exist)

        return seqs, nodes, conns, u_conns

    def forward(self, state, transformed, inputs):

        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        cal_seqs, nodes, conns, u_conns = transformed

        ini_vals = jnp.full((self.max_nodes,), jnp.nan)
        ini_vals = ini_vals.at[self.input_idx].set(inputs)
        nodes_attrs = vmap(extract_node_attrs)(nodes)
        conns_attrs = vmap(extract_conn_attrs)(conns)

        def cond_fun(carry):
            values, idx = carry
            return (idx < self.max_nodes) & (
                cal_seqs[idx] != I_INF
            )  # not out of bounds and next node exists

        def body_func(carry):
            values, idx = carry
            i = cal_seqs[idx]

            def input_node():
                return values

            def otherwise():
                # calculate connections
                conn_indices = u_conns[:, i]
                hit_attrs = attach_with_inf(conns_attrs, conn_indices)  # fetch conn attrs
                ins = vmap(self.conn_gene.forward, in_axes=(None, 0, 0))(
                    state, hit_attrs, values
                )

                # calculate nodes
                z = self.node_gene.forward(
                    state,
                    nodes_attrs[i],
                    ins,
                    is_output_node=jnp.isin(nodes[i, 0], self.output_idx),  # nodes[0] -> the key of nodes
                )

                # set new value
                new_values = values.at[i].set(z)
                return new_values

            values = jax.lax.cond(jnp.isin(i, self.input_idx), input_node, otherwise)

            return values, idx + 1

        vals, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, 0))

        if self.output_transform is None:
            return vals[self.output_idx]
        else:
            return self.output_transform(vals[self.output_idx])

    def network_dict(self, state, nodes, conns):
        network = super().network_dict(state, nodes, conns)
        topo_order, topo_layers = topological_sort_python(
            set(network["nodes"]), set(network["conns"])
        )
        network["topo_order"] = topo_order
        network["topo_layers"] = topo_layers
        return network

    def sympy_func(
        self,
        state,
        network,
        sympy_input_transform=None,
        sympy_output_transform=None,
        backend="jax",
    ):

        assert backend in ["jax", "numpy"], "backend should be 'jax' or 'numpy'"
        module = SYMPY_FUNCS_MODULE_JNP if backend == "jax" else SYMPY_FUNCS_MODULE_NP

        if sympy_input_transform is None and self.input_transform is not None:
            warnings.warn(
                "genome.input_transform is not None but sympy_input_transform is None!"
            )

        if sympy_input_transform is None:
            sympy_input_transform = lambda x: x

        if sympy_input_transform is not None:
            if not isinstance(sympy_input_transform, list):
                sympy_input_transform = [sympy_input_transform] * self.num_inputs

        if sympy_output_transform is None and self.output_transform is not None:
            warnings.warn(
                "genome.output_transform is not None but sympy_output_transform is None!"
            )

        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()
        order = network["topo_order"]

        hidden_idx = [
            i for i in network["nodes"] if i not in input_idx and i not in output_idx
        ]
        symbols = {}
        for i in network["nodes"]:
            if i in input_idx:
                symbols[-i - 1] = sp.Symbol(f"i{i - min(input_idx)}")  # origin_i
                symbols[i] = sp.Symbol(f"norm{i - min(input_idx)}")
            elif i in output_idx:
                symbols[i] = sp.Symbol(f"o{i - min(output_idx)}")
            else:  # hidden
                symbols[i] = sp.Symbol(f"h{i - min(hidden_idx)}")

        nodes_exprs = {}
        args_symbols = {}
        for i in order:

            if i in input_idx:
                nodes_exprs[symbols[-i - 1]] = symbols[
                    -i - 1
                ]  # origin equal to its symbol
                nodes_exprs[symbols[i]] = sympy_input_transform[i - min(input_idx)](
                    symbols[-i - 1]
                )  # normed i

            else:
                in_conns = [c for c in network["conns"] if c[1] == i]
                node_inputs = []
                for conn in in_conns:
                    val_represent = symbols[conn[0]]
                    # a_s -> args_symbols
                    val, a_s = self.conn_gene.sympy_func(
                        state,
                        network["conns"][conn],
                        val_represent,
                    )
                    args_symbols.update(a_s)
                    node_inputs.append(val)
                nodes_exprs[symbols[i]], a_s = self.node_gene.sympy_func(
                    state,
                    network["nodes"][i],
                    node_inputs,
                    is_output_node=(i in output_idx),
                )
                args_symbols.update(a_s)

                if i in output_idx and sympy_output_transform is not None:
                    nodes_exprs[symbols[i]] = sympy_output_transform(
                        nodes_exprs[symbols[i]]
                    )

        input_symbols = [symbols[-i - 1] for i in input_idx]
        reduced_exprs = nodes_exprs.copy()
        for i in order:
            reduced_exprs[symbols[i]] = reduced_exprs[symbols[i]].subs(reduced_exprs)

        output_exprs = [reduced_exprs[symbols[i]] for i in output_idx]

        lambdify_output_funcs = [
            sp.lambdify(
                input_symbols + list(args_symbols.keys()),
                exprs,
                modules=[backend, module],
            )
            for exprs in output_exprs
        ]

        fixed_args_output_funcs = []
        for i in range(len(output_idx)):

            def f(inputs, i=i):
                return lambdify_output_funcs[i](*inputs, *args_symbols.values())

            fixed_args_output_funcs.append(f)

        forward_func = lambda inputs: jnp.array(
            [f(inputs) for f in fixed_args_output_funcs]
        )

        return (
            symbols,
            args_symbols,
            input_symbols,
            nodes_exprs,
            output_exprs,
            forward_func,
        )

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

        topo_order, topo_layers = network["topo_order"], network["topo_layers"]
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
            pos=rotated_pos,
            node_size=node_sizes,
            node_color=node_colors,
            **kwargs,
        )
        plt.savefig(save_path, dpi=save_dpi)
