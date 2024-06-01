import jax, jax.numpy as jnp
from . import BaseMutation
from utils import (
    fetch_first,
    fetch_random,
    I_INF,
    unflatten_conns,
    check_cycles,
    add_node,
    add_conn,
    delete_node_by_pos,
    delete_conn_by_pos,
    extract_node_attrs,
    extract_conn_attrs,
    set_node_attrs,
    set_conn_attrs,
)


class DefaultMutation(BaseMutation):
    def __init__(
        self,
        conn_add: float = 0.2,
        conn_delete: float = 0,
        node_add: float = 0.2,
        node_delete: float = 0,
    ):
        self.conn_add = conn_add
        self.conn_delete = conn_delete
        self.node_add = node_add
        self.node_delete = node_delete

    def __call__(self, state, randkey, genome, nodes, conns, new_node_key):
        k1, k2 = jax.random.split(randkey)

        nodes, conns = self.mutate_structure(
            state, k1, genome, nodes, conns, new_node_key
        )
        nodes, conns = self.mutate_values(state, k2, genome, nodes, conns)

        return nodes, conns

    def mutate_structure(self, state, randkey, genome, nodes, conns, new_node_key):

        remain_node_space = jnp.isnan(nodes[:, 0]).sum()
        remain_conn_space = jnp.isnan(conns[:, 0]).sum()

        def mutate_add_node(key_, nodes_, conns_):
            """
            add a node while do not influence the output of the network
            """

            i_key, o_key, idx = self.choose_connection_key(
                key_, conns_
            )  # choose a connection

            def successful_add_node():
                # remove the original connection and record its attrs
                original_attrs = extract_conn_attrs(conns_[idx])
                new_conns = delete_conn_by_pos(conns_, idx)

                # add a new node with identity attrs
                new_nodes = add_node(
                    nodes_, new_node_key, genome.node_gene.new_identity_attrs(state)
                )

                # add two new connections
                # first is with identity attrs
                new_conns = add_conn(
                    new_conns,
                    i_key,
                    new_node_key,
                    genome.conn_gene.new_identity_attrs(state),
                )
                # second is with the origin attrs
                new_conns = add_conn(
                    new_conns,
                    new_node_key,
                    o_key,
                    original_attrs,
                )

                return new_nodes, new_conns

            return jax.lax.cond(
                (idx == I_INF) & (remain_node_space < 1) & (remain_conn_space < 2),
                lambda: (nodes_, conns_),  # do nothing
                successful_add_node,
            )

        def mutate_delete_node(key_, nodes_, conns_):
            """
            delete a node
            """

            # randomly choose a node
            key, idx = self.choose_node_key(
                key_,
                nodes_,
                genome.input_idx,
                genome.output_idx,
                allow_input_keys=False,
                allow_output_keys=False,
            )

            def successful_delete_node():
                # delete the node
                new_nodes = delete_node_by_pos(nodes_, idx)

                # delete all connections
                new_conns = jnp.where(
                    ((conns_[:, 0] == key) | (conns_[:, 1] == key))[:, None],
                    jnp.nan,
                    conns_,
                )

                return new_nodes, new_conns

            return jax.lax.cond(
                idx == I_INF,  # no available node to delete
                lambda: (nodes_, conns_),  # do nothing
                successful_delete_node,
            )

        def mutate_add_conn(key_, nodes_, conns_):
            """
            add a connection while do not influence the output of the network
            """

            # randomly choose two nodes
            k1_, k2_ = jax.random.split(key_, num=2)

            # input node of the connection can be any node
            i_key, from_idx = self.choose_node_key(
                k1_,
                nodes_,
                genome.input_idx,
                genome.output_idx,
                allow_input_keys=True,
                allow_output_keys=True,
            )

            # output node of the connection can be any node except input node
            o_key, to_idx = self.choose_node_key(
                k2_,
                nodes_,
                genome.input_idx,
                genome.output_idx,
                allow_input_keys=False,
                allow_output_keys=True,
            )

            conn_pos = fetch_first((conns_[:, 0] == i_key) & (conns_[:, 1] == o_key))
            is_already_exist = conn_pos != I_INF

            def nothing():
                return nodes_, conns_

            def successful():
                # add a connection with zero attrs
                return nodes_, add_conn(
                    conns_, i_key, o_key, genome.conn_gene.new_zero_attrs(state)
                )

            if genome.network_type == "feedforward":
                u_cons = unflatten_conns(nodes_, conns_)
                conns_exist = ~jnp.isnan(u_cons[0, :, :])
                is_cycle = check_cycles(nodes_, conns_exist, from_idx, to_idx)

                return jax.lax.cond(
                    is_already_exist | is_cycle | (remain_conn_space < 1),
                    nothing,
                    successful,
                )

            elif genome.network_type == "recurrent":
                return jax.lax.cond(
                    is_already_exist | (remain_conn_space < 1),
                    nothing,
                    successful,
                )

            else:
                raise ValueError(f"Invalid network type: {genome.network_type}")

        def mutate_delete_conn(key_, nodes_, conns_):
            # randomly choose a connection
            i_key, o_key, idx = self.choose_connection_key(key_, conns_)

            return jax.lax.cond(
                idx == I_INF,
                lambda: (nodes_, conns_),  # nothing
                lambda: (nodes_, delete_conn_by_pos(conns_, idx)),  # success
            )

        k1, k2, k3, k4 = jax.random.split(randkey, num=4)
        r1, r2, r3, r4 = jax.random.uniform(k1, shape=(4,))

        def nothing(_, nodes_, conns_):
            return nodes_, conns_

        if self.node_add > 0:
            nodes, conns = jax.lax.cond(
                r1 < self.node_add, mutate_add_node, nothing, k1, nodes, conns
            )

        if self.node_delete > 0:
            nodes, conns = jax.lax.cond(
                r2 < self.node_delete, mutate_delete_node, nothing, k2, nodes, conns
            )

        if self.conn_add > 0:
            nodes, conns = jax.lax.cond(
                r3 < self.conn_add, mutate_add_conn, nothing, k3, nodes, conns
            )

        if self.conn_delete > 0:
            nodes, conns = jax.lax.cond(
                r4 < self.conn_delete, mutate_delete_conn, nothing, k4, nodes, conns
            )

        return nodes, conns

    def mutate_values(self, state, randkey, genome, nodes, conns):
        k1, k2 = jax.random.split(randkey)
        nodes_randkeys = jax.random.split(k1, num=genome.max_nodes)
        conns_randkeys = jax.random.split(k2, num=genome.max_conns)

        node_attrs = jax.vmap(extract_node_attrs)(nodes)
        new_node_attrs = jax.vmap(genome.node_gene.mutate, in_axes=(None, 0, 0))(
            state, nodes_randkeys, node_attrs
        )
        new_nodes = jax.vmap(set_node_attrs)(nodes, new_node_attrs)

        conn_attrs = jax.vmap(extract_conn_attrs)(conns)
        new_conn_attrs = jax.vmap(genome.conn_gene.mutate, in_axes=(None, 0, 0))(
            state, conns_randkeys, conn_attrs
        )
        new_conns = jax.vmap(set_conn_attrs)(conns, new_conn_attrs)

        # nan nodes not changed
        new_nodes = jnp.where(jnp.isnan(nodes), jnp.nan, new_nodes)
        new_conns = jnp.where(jnp.isnan(conns), jnp.nan, new_conns)

        return new_nodes, new_conns

    def choose_node_key(
        self,
        key,
        nodes,
        input_idx,
        output_idx,
        allow_input_keys: bool = False,
        allow_output_keys: bool = False,
    ):
        """
        Randomly choose a node key from the given nodes. It guarantees that the chosen node not be the input or output node.
        :param key:
        :param nodes:
        :param input_idx:
        :param output_idx:
        :param allow_input_keys:
        :param allow_output_keys:
        :return: return its key and position(idx)
        """

        node_keys = nodes[:, 0]
        mask = ~jnp.isnan(node_keys)

        if not allow_input_keys:
            mask = jnp.logical_and(mask, ~jnp.isin(node_keys, input_idx))

        if not allow_output_keys:
            mask = jnp.logical_and(mask, ~jnp.isin(node_keys, output_idx))

        idx = fetch_random(key, mask)
        key = jnp.where(idx != I_INF, nodes[idx, 0], jnp.nan)
        return key, idx

    def choose_connection_key(self, key, conns):
        """
        Randomly choose a connection key from the given connections.
        :return: i_key, o_key, idx
        """

        idx = fetch_random(key, ~jnp.isnan(conns[:, 0]))
        i_key = jnp.where(idx != I_INF, conns[idx, 0], jnp.nan)
        o_key = jnp.where(idx != I_INF, conns[idx, 1], jnp.nan)

        return i_key, o_key, idx
