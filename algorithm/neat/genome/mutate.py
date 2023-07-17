from typing import Dict, Tuple, Type

import numpy as np
import jax
from jax import Array, numpy as jnp, vmap

from algorithm import State
from .basic import add_node, add_connection, delete_node_by_idx, delete_connection_by_idx
from .graph import check_cycles
from ..utils import fetch_random, fetch_first, I_INT, unflatten_connections
from ..gene import BaseGene


def create_mutate(config: Dict, gene_type: Type[BaseGene]):
    """
    Create function to mutate the whole population
    """

    def mutate_structure(state: State, randkey, nodes, cons, new_node_key):
        def nothing(*args):
            return nodes, cons

        def mutate_add_node(key_):
            i_key, o_key, idx = choice_connection_key(key_, nodes, cons)

            def successful_add_node():
                # disable the connection
                aux_nodes, aux_cons = nodes, cons

                # set enable to false
                aux_cons = aux_cons.at[idx, 2].set(False)

                # add a new node
                aux_nodes, aux_cons = add_node(aux_nodes, aux_cons, new_node_key, gene_type.new_node_attrs(state))

                # add two new connections
                aux_nodes, aux_cons = add_connection(aux_nodes, aux_cons, i_key, new_node_key, True,
                                                     gene_type.new_conn_attrs(state))
                aux_nodes, aux_cons = add_connection(aux_nodes, aux_cons, new_node_key, o_key, True,
                                                     gene_type.new_conn_attrs(state))

                return aux_nodes, aux_cons

            # if from_idx == I_INT, that means no connection exist, do nothing
            return jax.lax.cond(idx == I_INT, nothing, successful_add_node)

        def mutate_delete_node(key_):
            # TODO: Do we really need to delete a node?
            # randomly choose a node
            key, idx = choice_node_key(key_, nodes, config['input_idx'], config['output_idx'],
                                       allow_input_keys=False, allow_output_keys=False)

            def successful_delete_node():
                # delete the node
                aux_nodes, aux_cons = delete_node_by_idx(nodes, cons, idx)

                # delete all connections
                aux_cons = jnp.where(((aux_cons[:, 0] == key) | (aux_cons[:, 1] == key))[:, None],
                                     jnp.nan, aux_cons)

                return aux_nodes, aux_cons

            return jax.lax.cond(idx == I_INT, nothing, successful_delete_node)

        def mutate_add_conn(key_):
            # randomly choose two nodes
            k1_, k2_ = jax.random.split(key_, num=2)
            i_key, from_idx = choice_node_key(k1_, nodes, config['input_idx'], config['output_idx'],
                                              allow_input_keys=True, allow_output_keys=True)
            o_key, to_idx = choice_node_key(k2_, nodes, config['input_idx'], config['output_idx'],
                                            allow_input_keys=False, allow_output_keys=True)

            con_idx = fetch_first((cons[:, 0] == i_key) & (cons[:, 1] == o_key))

            def successful():
                new_nodes, new_cons = add_connection(nodes, cons, i_key, o_key, True, gene_type.new_conn_attrs(state))
                return new_nodes, new_cons

            def already_exist():
                new_cons = cons.at[con_idx, 2].set(True)
                return nodes, new_cons

            is_already_exist = con_idx != I_INT

            if config['network_type'] == 'feedforward':
                u_cons = unflatten_connections(nodes, cons)
                is_cycle = check_cycles(nodes, u_cons, from_idx, to_idx)

                choice = jnp.where(is_already_exist, 0, jnp.where(is_cycle, 1, 2))
                return jax.lax.switch(choice, [already_exist, nothing, successful])

            elif config['network_type'] == 'recurrent':
                return jax.lax.cond(is_already_exist, already_exist, successful)

            else:
                raise ValueError(f"Invalid network type: {config['network_type']}")

        def mutate_delete_conn(key_):
            # randomly choose a connection
            i_key, o_key, idx = choice_connection_key(key_, nodes, cons)

            def successfully_delete_connection():
                return delete_connection_by_idx(nodes, cons, idx)

            return jax.lax.cond(idx == I_INT, nothing, successfully_delete_connection)

        k, k1, k2, k3, k4 = jax.random.split(randkey, num=5)
        r1, r2, r3, r4 = jax.random.uniform(k1, shape=(4,))

        nodes, cons = jax.lax.cond(r1 < config['node_add_prob'], mutate_add_node, nothing, k1)
        nodes, cons = jax.lax.cond(r2 < config['node_delete_prob'], mutate_delete_node, nothing, k2)
        nodes, cons = jax.lax.cond(r3 < config['conn_add_prob'], mutate_add_conn, nothing, k3)
        nodes, cons = jax.lax.cond(r4 < config['conn_delete_prob'], mutate_delete_conn, nothing, k4)
        return nodes, cons

    def mutate_values(state: State, randkey, nodes, conns):
        k1, k2 = jax.random.split(randkey, num=2)
        nodes_keys = jax.random.split(k1, num=nodes.shape[0])
        conns_keys = jax.random.split(k2, num=conns.shape[0])

        nodes_attrs, conns_attrs = nodes[:, 1:], conns[:, 3:]

        new_nodes_attrs = vmap(gene_type.mutate_node, in_axes=(None, 0, 0))(state, nodes_attrs, nodes_keys)
        new_conns_attrs = vmap(gene_type.mutate_conn, in_axes=(None, 0, 0))(state, conns_attrs, conns_keys)

        # nan nodes not changed
        new_nodes_attrs = jnp.where(jnp.isnan(nodes_attrs), jnp.nan, new_nodes_attrs)
        new_conns_attrs = jnp.where(jnp.isnan(conns_attrs), jnp.nan, new_conns_attrs)

        new_nodes = nodes.at[:, 1:].set(new_nodes_attrs)
        new_conns = conns.at[:, 3:].set(new_conns_attrs)

        return new_nodes, new_conns

    def mutate(state):
        pop_nodes, pop_conns = state.pop_nodes, state.pop_conns
        pop_size = pop_nodes.shape[0]

        new_node_keys = jnp.arange(pop_size) + state.next_node_key
        k1, k2, randkey = jax.random.split(state.randkey, num=3)
        structure_randkeys = jax.random.split(k1, num=pop_size)
        values_randkeys = jax.random.split(k2, num=pop_size)

        structure_func = jax.vmap(mutate_structure, in_axes=(None, 0, 0, 0, 0))
        pop_nodes, pop_conns = structure_func(state, structure_randkeys, pop_nodes, pop_conns, new_node_keys)

        values_func = jax.vmap(mutate_values, in_axes=(None, 0, 0, 0))
        pop_nodes, pop_conns = values_func(state, values_randkeys, pop_nodes, pop_conns)

        # update next node key
        all_nodes_keys = pop_nodes[:, :, 0]
        max_node_key = jnp.max(jnp.where(jnp.isnan(all_nodes_keys), -jnp.inf, all_nodes_keys))
        next_node_key = max_node_key + 1

        return state.update(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            next_node_key=next_node_key,
            randkey=randkey
        )

    return mutate


def choice_node_key(rand_key: Array, nodes: Array,
                    input_keys: Array, output_keys: Array,
                    allow_input_keys: bool = False, allow_output_keys: bool = False) -> Tuple[Array, Array]:
    """
    Randomly choose a node key from the given nodes. It guarantees that the chosen node not be the input or output node.
    :param rand_key:
    :param nodes:
    :param input_keys:
    :param output_keys:
    :param allow_input_keys:
    :param allow_output_keys:
    :return: return its key and position(idx)
    """

    node_keys = nodes[:, 0]
    mask = ~jnp.isnan(node_keys)

    if not allow_input_keys:
        mask = jnp.logical_and(mask, ~jnp.isin(node_keys, input_keys))

    if not allow_output_keys:
        mask = jnp.logical_and(mask, ~jnp.isin(node_keys, output_keys))

    idx = fetch_random(rand_key, mask)
    key = jnp.where(idx != I_INT, nodes[idx, 0], jnp.nan)
    return key, idx


def choice_connection_key(rand_key: Array, nodes: Array, cons: Array) -> Tuple[Array, Array, Array]:
    """
    Randomly choose a connection key from the given connections.
    :param rand_key:
    :param nodes:
    :param cons:
    :return: i_key, o_key, idx
    """

    idx = fetch_random(rand_key, ~jnp.isnan(cons[:, 0]))
    i_key = jnp.where(idx != I_INT, cons[idx, 0], jnp.nan)
    o_key = jnp.where(idx != I_INT, cons[idx, 1], jnp.nan)

    return i_key, o_key, idx
