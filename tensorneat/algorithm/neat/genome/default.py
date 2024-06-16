import warnings
from typing import Callable

import jax, jax.numpy as jnp
import sympy as sp
from utils import (
    unflatten_conns,
    topological_sort,
    topological_sort_python,
    I_INF,
    extract_node_attrs,
    extract_conn_attrs,
    set_node_attrs,
    set_conn_attrs,
    attach_with_inf,
    SYMPY_FUNCS_MODULE_NP,
    SYMPY_FUNCS_MODULE_JNP,
)
from . import BaseGenome
from ..gene import BaseNodeGene, BaseConnGene, DefaultNodeGene, DefaultConnGene
from ..ga import BaseMutation, BaseCrossover, DefaultMutation, DefaultCrossover


class DefaultGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=5,
        max_conns=4,
        node_gene: BaseNodeGene = DefaultNodeGene(),
        conn_gene: BaseConnGene = DefaultConnGene(),
        mutation: BaseMutation = DefaultMutation(),
        crossover: BaseCrossover = DefaultCrossover(),
        output_transform: Callable = None,
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
        )

        if output_transform is not None:
            try:
                _ = output_transform(jnp.zeros(num_outputs))
            except Exception as e:
                raise ValueError(f"Output transform function failed: {e}")
        self.output_transform = output_transform

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        conn_exist = u_conns != I_INF

        seqs = topological_sort(nodes, conn_exist)

        return seqs, nodes, conns, u_conns

    def restore(self, state, transformed):
        seqs, nodes, conns, u_conns = transformed
        return nodes, conns

    def forward(self, state, transformed, inputs):
        cal_seqs, nodes, conns, u_conns = transformed

        ini_vals = jnp.full((self.max_nodes,), jnp.nan)
        ini_vals = ini_vals.at[self.input_idx].set(inputs)
        nodes_attrs = jax.vmap(extract_node_attrs)(nodes)
        conns_attrs = jax.vmap(extract_conn_attrs)(conns)

        def cond_fun(carry):
            values, idx = carry
            return (idx < self.max_nodes) & (cal_seqs[idx] != I_INF)

        def body_func(carry):
            values, idx = carry
            i = cal_seqs[idx]

            def input_node():
                z = self.node_gene.input_transform(state, nodes_attrs[i], values[i])
                new_values = values.at[i].set(z)
                return new_values

            def otherwise():
                conn_indices = u_conns[:, i]
                hit_attrs = attach_with_inf(conns_attrs, conn_indices)
                ins = jax.vmap(self.conn_gene.forward, in_axes=(None, 0, 0))(
                    state, hit_attrs, values
                )

                z = self.node_gene.forward(
                    state,
                    nodes_attrs[i],
                    ins,
                    is_output_node=jnp.isin(i, self.output_idx),
                )

                new_values = values.at[i].set(z)
                return new_values

            values = jax.lax.cond(jnp.isin(i, self.input_idx), input_node, otherwise)

            return values, idx + 1

        vals, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, 0))

        if self.output_transform is None:
            return vals[self.output_idx]
        else:
            return self.output_transform(vals[self.output_idx])

    def update_by_batch(self, state, batch_input, transformed):
        cal_seqs, nodes, conns, u_conns = transformed

        batch_size = batch_input.shape[0]
        batch_ini_vals = jnp.full((batch_size, self.max_nodes), jnp.nan)
        batch_ini_vals = batch_ini_vals.at[:, self.input_idx].set(batch_input)
        nodes_attrs = jax.vmap(extract_node_attrs)(nodes)
        conns_attrs = jax.vmap(extract_conn_attrs)(conns)

        def cond_fun(carry):
            batch_values, nodes_attrs_, conns_attrs_, idx = carry
            return (idx < self.max_nodes) & (cal_seqs[idx] != I_INF)

        def body_func(carry):
            batch_values, nodes_attrs_, conns_attrs_, idx = carry
            i = cal_seqs[idx]

            def input_node():
                batch, new_attrs = self.node_gene.update_input_transform(
                    state, nodes_attrs_[i], batch_values[:, i]
                )
                return (
                    batch_values.at[:, i].set(batch),
                    nodes_attrs_.at[i].set(new_attrs),
                    conns_attrs_,
                )

            def otherwise():

                conn_indices = u_conns[:, i]
                hit_attrs = attach_with_inf(conns_attrs, conn_indices)
                batch_ins, new_conn_attrs = jax.vmap(
                    self.conn_gene.update_by_batch,
                    in_axes=(None, 0, 1),
                    out_axes=(1, 0),
                )(state, hit_attrs, batch_values)

                batch_z, new_node_attrs = self.node_gene.update_by_batch(
                    state,
                    nodes_attrs_[i],
                    batch_ins,
                    is_output_node=jnp.isin(i, self.output_idx),
                )

                return (
                    batch_values.at[:, i].set(batch_z),
                    nodes_attrs_.at[i].set(new_node_attrs),
                    conns_attrs_.at[conn_indices].set(new_conn_attrs),
                )

            # the val of input nodes is obtained by the task, not by calculation
            (batch_values, nodes_attrs_, conns_attrs_) = jax.lax.cond(
                jnp.isin(i, self.input_idx),
                input_node,
                otherwise,
            )

            return batch_values, nodes_attrs_, conns_attrs_, idx + 1

        batch_vals, nodes_attrs, conns_attrs, _ = jax.lax.while_loop(
            cond_fun, body_func, (batch_ini_vals, nodes_attrs, conns_attrs, 0)
        )

        nodes = jax.vmap(set_node_attrs)(nodes, nodes_attrs)
        conns = jax.vmap(set_conn_attrs)(conns, conns_attrs)

        new_transformed = (cal_seqs, nodes, conns, u_conns)

        if self.output_transform is None:
            return batch_vals[:, self.output_idx], new_transformed
        else:
            return (
                jax.vmap(self.output_transform)(batch_vals[:, self.output_idx]),
                new_transformed,
            )

    def sympy_func(self, state, network, sympy_output_transform=None, backend="jax"):

        assert backend in ["jax", "numpy"], "backend should be 'jax' or 'numpy'"
        module = SYMPY_FUNCS_MODULE_JNP if backend == "jax" else SYMPY_FUNCS_MODULE_NP

        if sympy_output_transform is None and self.output_transform is not None:
            warnings.warn(
                "genome.output_transform is not None but sympy_output_transform is None!"
            )

        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()
        order, _ = topological_sort_python(set(network["nodes"]), set(network["conns"]))
        hidden_idx = [i for i in network["nodes"] if i not in input_idx and i not in output_idx]
        symbols = {}
        for i in network["nodes"]:
            if i in input_idx:
                symbols[i] = sp.Symbol(f"i{i - min(input_idx)}")
            elif i in output_idx:
                symbols[i] = sp.Symbol(f"o{i - min(output_idx)}")
            else:  # hidden
                symbols[i] = sp.Symbol(f"h{i - min(hidden_idx)}")

        nodes_exprs = {}
        args_symbols = {}
        for i in order:

            if i in input_idx:
                nodes_exprs[symbols[i]] = symbols[i]
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

        input_symbols = [v for k, v in symbols.items() if k in input_idx]
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

        forward_func = lambda inputs: jnp.array([f(inputs) for f in fixed_args_output_funcs])

        return (
            symbols,
            args_symbols,
            input_symbols,
            nodes_exprs,
            output_exprs,
            forward_func,
        )
