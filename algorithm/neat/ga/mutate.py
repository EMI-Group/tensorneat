from typing import Tuple

import jax
from jax import Array, numpy as jnp, vmap

from config import NeatConfig
from core import State, Gene, Genome
from utils import check_cycles, fetch_random, fetch_first, I_INT, unflatten_conns


def mutate(config: NeatConfig, gene: Gene, state: State, randkey, genome: Genome, new_node_key):
    """
    Mutate a population of genomes
    """
    k1, k2 = jax.random.split(randkey)

    genome = mutate_structure(config, gene, state, k1, genome, new_node_key)
    genome = mutate_values(gene, state, randkey, genome)

    return genome


def mutate_structure(config: NeatConfig, gene: Gene, state: State, randkey, genome: Genome, new_node_key):
    def mutate_add_node(key_, genome_: Genome):
        i_key, o_key, idx = choice_connection_key(key_, genome_.conns)

        def nothing():
            return genome_

        def successful_add_node():
            # disable the connection
            new_genome = genome_.update_conns(genome_.conns.at[idx, 2].set(False))

            # add a new node
            new_genome = new_genome.add_node(new_node_key, gene.new_node_attrs(state))

            # add two new connections
            new_genome = new_genome.add_conn(i_key, new_node_key, True, gene.new_conn_attrs(state))
            new_genome = new_genome.add_conn(new_node_key, o_key, True, gene.new_conn_attrs(state))

            return new_genome

        # if from_idx == I_INT, that means no connection exist, do nothing
        return jax.lax.cond(idx == I_INT, nothing, successful_add_node)

    def mutate_delete_node(key_, genome_: Genome):
        # TODO: Do we really need to delete a node?
        # randomly choose a node
        key, idx = choice_node_key(key_, genome_.nodes, state.input_idx, state.output_idx,
                                   allow_input_keys=False, allow_output_keys=False)

        def nothing():
            return genome_

        def successful_delete_node():
            # delete the node
            new_genome = genome_.delete_node_by_pos(idx)

            # delete all connections
            new_conns = jnp.where(((new_genome.conns[:, 0] == key) | (new_genome.conns[:, 1] == key))[:, None],
                                  jnp.nan, new_genome.conns)

            return new_genome.update_conns(new_conns)

        return jax.lax.cond(idx == I_INT, nothing, successful_delete_node)

    def mutate_add_conn(key_, genome_: Genome):
        # randomly choose two nodes
        k1_, k2_ = jax.random.split(key_, num=2)
        i_key, from_idx = choice_node_key(k1_, genome_.nodes, state.input_idx, state.output_idx,
                                          allow_input_keys=True, allow_output_keys=True)
        o_key, to_idx = choice_node_key(k2_, genome_.nodes, state.input_idx, state.output_idx,
                                        allow_input_keys=False, allow_output_keys=True)

        conn_pos = fetch_first((genome_.conns[:, 0] == i_key) & (genome_.conns[:, 1] == o_key))

        def nothing():
            return genome_

        def successful():
            return genome_.add_conn(i_key, o_key, True, gene.new_conn_attrs(state))

        def already_exist():
            return genome_.update_conns(genome_.conns.at[conn_pos, 2].set(True))

        is_already_exist = conn_pos != I_INT

        if config.network_type == 'feedforward':
            u_cons = unflatten_conns(genome_.nodes, genome_.conns)
            cons_exist = jnp.where(~jnp.isnan(u_cons[0, :, :]), True, False)
            is_cycle = check_cycles(genome_.nodes, cons_exist, from_idx, to_idx)

            choice = jnp.where(is_already_exist, 0, jnp.where(is_cycle, 1, 2))
            return jax.lax.switch(choice, [already_exist, nothing, successful])

        elif config.network_type == 'recurrent':
            return jax.lax.cond(is_already_exist, already_exist, successful)

        else:
            raise ValueError(f"Invalid network type: {config.network_type}")

    def mutate_delete_conn(key_, genome_: Genome):
        # randomly choose a connection
        i_key, o_key, idx = choice_connection_key(key_, genome_.conns)

        def nothing():
            return genome_

        def successfully_delete_connection():
            return genome_.delete_conn_by_pos(idx)

        return jax.lax.cond(idx == I_INT, nothing, successfully_delete_connection)

    k1, k2, k3, k4 = jax.random.split(randkey, num=4)
    r1, r2, r3, r4 = jax.random.uniform(k1, shape=(4,))

    def no(k, g):
        return g

    genome = jax.lax.cond(r1 < config.node_add, mutate_add_node, no, k1, genome)
    genome = jax.lax.cond(r2 < config.node_delete, mutate_delete_node, no, k2, genome)
    genome = jax.lax.cond(r3 < config.conn_add, mutate_add_conn, no, k3, genome)
    genome = jax.lax.cond(r4 < config.conn_delete, mutate_delete_conn, no, k4, genome)

    return genome


def mutate_values(gene: Gene, state: State, randkey, genome: Genome):
    k1, k2 = jax.random.split(randkey, num=2)
    nodes_keys = jax.random.split(k1, num=genome.nodes.shape[0])
    conns_keys = jax.random.split(k2, num=genome.conns.shape[0])

    nodes_attrs, conns_attrs = genome.nodes[:, 1:], genome.conns[:, 3:]

    new_nodes_attrs = vmap(gene.mutate_node, in_axes=(None, 0, 0))(state, nodes_keys, nodes_attrs)
    new_conns_attrs = vmap(gene.mutate_conn, in_axes=(None, 0, 0))(state, conns_keys, conns_attrs)

    # nan nodes not changed
    new_nodes_attrs = jnp.where(jnp.isnan(nodes_attrs), jnp.nan, new_nodes_attrs)
    new_conns_attrs = jnp.where(jnp.isnan(conns_attrs), jnp.nan, new_conns_attrs)

    new_nodes = genome.nodes.at[:, 1:].set(new_nodes_attrs)
    new_conns = genome.conns.at[:, 3:].set(new_conns_attrs)

    return genome.update(new_nodes, new_conns)


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


def choice_connection_key(rand_key: Array, conns: Array):
    """
    Randomly choose a connection key from the given connections.
    :return: i_key, o_key, idx
    """

    idx = fetch_random(rand_key, ~jnp.isnan(conns[:, 0]))
    i_key = jnp.where(idx != I_INT, conns[idx, 0], jnp.nan)
    o_key = jnp.where(idx != I_INT, conns[idx, 1], jnp.nan)

    return i_key, o_key, idx
