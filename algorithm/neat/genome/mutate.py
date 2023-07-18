from typing import Dict, Tuple, Type

import jax
from jax import Array, numpy as jnp, vmap

from algorithm import State
from .basic import add_node, add_connection, delete_node_by_idx, delete_connection_by_idx, count
from .graph import check_cycles
from ..utils import fetch_random, fetch_first, I_INT, unflatten_connections
from ..gene import BaseGene


def create_mutate(config: Dict, gene_type: Type[BaseGene]):
    """
    Create function to mutate a single genome
    """

    def mutate_structure(state: State, randkey, nodes, conns, new_node_key):

        def mutate_add_node(key_, nodes_, conns_):
            i_key, o_key, idx = choice_connection_key(key_, nodes_, conns_)

            def nothing():
                return nodes_, conns_

            def successful_add_node():
                # disable the connection
                aux_nodes, aux_conns = nodes_, conns_

                # set enable to false
                aux_conns = aux_conns.at[idx, 2].set(False)

                # add a new node
                aux_nodes, aux_conns = add_node(aux_nodes, aux_conns, new_node_key, gene_type.new_node_attrs(state))

                # add two new connections
                aux_nodes, aux_conns = add_connection(aux_nodes, aux_conns, i_key, new_node_key, True,
                                                     gene_type.new_conn_attrs(state))
                aux_nodes, aux_conns = add_connection(aux_nodes, aux_conns, new_node_key, o_key, True,
                                                     gene_type.new_conn_attrs(state))

                return aux_nodes, aux_conns

            # if from_idx == I_INT, that means no connection exist, do nothing
            new_nodes, new_conns = jax.lax.cond(idx == I_INT, nothing, successful_add_node)

            return new_nodes, new_conns

        def mutate_delete_node(key_, nodes_, conns_):
            # TODO: Do we really need to delete a node?
            # randomly choose a node
            key, idx = choice_node_key(key_, nodes_, config['input_idx'], config['output_idx'],
                                       allow_input_keys=False, allow_output_keys=False)
            def nothing():
                return nodes_, conns_

            def successful_delete_node():
                # delete the node
                aux_nodes, aux_cons = delete_node_by_idx(nodes_, conns_, idx)

                # delete all connections
                aux_cons = jnp.where(((aux_cons[:, 0] == key) | (aux_cons[:, 1] == key))[:, None],
                                     jnp.nan, aux_cons)

                return aux_nodes, aux_cons

            return jax.lax.cond(idx == I_INT, nothing, successful_delete_node)

        def mutate_add_conn(key_, nodes_, conns_):
            # randomly choose two nodes
            k1_, k2_ = jax.random.split(key_, num=2)
            i_key, from_idx = choice_node_key(k1_, nodes_, config['input_idx'], config['output_idx'],
                                              allow_input_keys=True, allow_output_keys=True)
            o_key, to_idx = choice_node_key(k2_, nodes_, config['input_idx'], config['output_idx'],
                                            allow_input_keys=False, allow_output_keys=True)

            con_idx = fetch_first((conns_[:, 0] == i_key) & (conns_[:, 1] == o_key))

            def nothing():
                return nodes_, conns_

            def successful():
                new_nodes, new_cons = add_connection(nodes_, conns_, i_key, o_key, True, gene_type.new_conn_attrs(state))
                return new_nodes, new_cons

            def already_exist():
                new_cons = conns_.at[con_idx, 2].set(True)
                return nodes_, new_cons

            is_already_exist = con_idx != I_INT

            if config['network_type'] == 'feedforward':
                u_cons = unflatten_connections(nodes_, conns_)
                is_cycle = check_cycles(nodes_, u_cons, from_idx, to_idx)

                choice = jnp.where(is_already_exist, 0, jnp.where(is_cycle, 1, 2))
                return jax.lax.switch(choice, [already_exist, nothing, successful])

            elif config['network_type'] == 'recurrent':
                return jax.lax.cond(is_already_exist, already_exist, successful)

            else:
                raise ValueError(f"Invalid network type: {config['network_type']}")

        def mutate_delete_conn(key_, nodes_, conns_):
            # randomly choose a connection
            i_key, o_key, idx = choice_connection_key(key_, nodes_, conns_)

            def nothing():
                return nodes_, conns_

            def successfully_delete_connection():
                return delete_connection_by_idx(nodes_, conns_, idx)

            return jax.lax.cond(idx == I_INT, nothing, successfully_delete_connection)

        k1, k2, k3, k4 = jax.random.split(randkey, num=4)
        r1, r2, r3, r4 = jax.random.uniform(k1, shape=(4,))

        def no(k, n, c):
            return n, c

        nodes, conns = jax.lax.cond(r1 < config['node_add_prob'], mutate_add_node, no, k1, nodes, conns)

        nodes, conns = jax.lax.cond(r2 < config['node_delete_prob'], mutate_delete_node, no, k2, nodes, conns)

        nodes, conns = jax.lax.cond(r3 < config['conn_add_prob'], mutate_add_conn, no, k3, nodes, conns)

        nodes, conns = jax.lax.cond(r4 < config['conn_delete_prob'], mutate_delete_conn, no, k4, nodes, conns)

        return nodes, conns

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

    def mutate(state, randkey, nodes, conns, new_node_key):
        k1, k2 = jax.random.split(randkey)

        nodes, conns = mutate_structure(state, k1, nodes, conns, new_node_key)
        nodes, conns = mutate_values(state, k2, nodes, conns)

        return nodes, conns

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
