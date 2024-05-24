import jax, jax.numpy as jnp
from . import BaseMutation
from utils import fetch_first, fetch_random, I_INT, unflatten_conns, check_cycles


class DefaultMutation(BaseMutation):

    def __init__(
            self,
            conn_add: float = 0.4,
            conn_delete: float = 0,
            node_add: float = 0.2,
            node_delete: float = 0,
    ):
        self.conn_add = conn_add
        self.conn_delete = conn_delete
        self.node_add = node_add
        self.node_delete = node_delete

    def __call__(self, randkey, genome, nodes, conns, new_node_key):
        k1, k2 = jax.random.split(randkey)

        nodes, conns = self.mutate_structure(k1, genome, nodes, conns, new_node_key)
        nodes, conns = self.mutate_values(k2, genome, nodes, conns)

        return nodes, conns

    def mutate_structure(self, randkey, genome, nodes, conns, new_node_key):
        def mutate_add_node(key_, nodes_, conns_):
            i_key, o_key, idx = self.choice_connection_key(key_, conns_)

            def successful_add_node():
                # disable the connection
                new_conns = conns_.at[idx, 2].set(False)

                # add a new node
                new_nodes = genome.add_node(nodes_, new_node_key, genome.node_gene.new_custom_attrs())

                # add two new connections
                new_conns = genome.add_conn(new_conns, i_key, new_node_key, True, genome.conn_gene.new_custom_attrs())
                new_conns = genome.add_conn(new_conns, new_node_key, o_key, True, genome.conn_gene.new_custom_attrs())

                return new_nodes, new_conns

            return jax.lax.cond(
                idx == I_INT,
                lambda: (nodes_, conns_),  # do nothing
                successful_add_node
            )

        def mutate_delete_node(key_, nodes_, conns_):

            # randomly choose a node
            key, idx = self.choice_node_key(key_, nodes_, genome.input_idx, genome.output_idx,
                                            allow_input_keys=False, allow_output_keys=False)

            def successful_delete_node():
                # delete the node
                new_nodes = genome.delete_node_by_pos(nodes_, idx)

                # delete all connections
                new_conns = jnp.where(
                    ((conns_[:, 0] == key) | (conns_[:, 1] == key))[:, None],
                    jnp.nan,
                    conns_
                )

                return new_nodes, new_conns

            return jax.lax.cond(
                idx == I_INT,
                lambda: (nodes_, conns_),  # do nothing
                successful_delete_node
            )

        def mutate_add_conn(key_, nodes_, conns_):
            # randomly choose two nodes
            k1_, k2_ = jax.random.split(key_, num=2)

            # input node of the connection can be any node
            i_key, from_idx = self.choice_node_key(k1_, nodes_, genome.input_idx, genome.output_idx,
                                                   allow_input_keys=True, allow_output_keys=True)

            # output node of the connection can be any node except input node
            o_key, to_idx = self.choice_node_key(k2_, nodes_, genome.input_idx, genome.output_idx,
                                                 allow_input_keys=False, allow_output_keys=True)

            conn_pos = fetch_first((conns_[:, 0] == i_key) & (conns_[:, 1] == o_key))
            is_already_exist = conn_pos != I_INT

            def nothing():
                return nodes_, conns_

            def successful():
                return nodes_, genome.add_conn(conns_, i_key, o_key, True, genome.conn_gene.new_custom_attrs())

            def already_exist():
                return nodes_, conns_.at[conn_pos, 2].set(True)

            if genome.network_type == 'feedforward':
                u_cons = unflatten_conns(nodes_, conns_)
                cons_exist = ~jnp.isnan(u_cons[0, :, :])
                is_cycle = check_cycles(nodes_, cons_exist, from_idx, to_idx)

                return jax.lax.cond(
                    is_already_exist,
                    already_exist,
                    lambda:
                        jax.lax.cond(
                            is_cycle,
                            nothing,
                            successful
                        )
                )

            elif genome.network_type == 'recurrent':
                return jax.lax.cond(
                    is_already_exist,
                    already_exist,
                    successful
                )

            else:
                raise ValueError(f"Invalid network type: {genome.network_type}")

        def mutate_delete_conn(key_, nodes_, conns_):
            # randomly choose a connection
            i_key, o_key, idx = self.choice_connection_key(key_, conns_)

            def successfully_delete_connection():
                return nodes_, genome.delete_conn_by_pos(conns_, idx)

            return jax.lax.cond(
                idx == I_INT,
                lambda: (nodes_, conns_),  # nothing
                successfully_delete_connection
            )

        k1, k2, k3, k4 = jax.random.split(randkey, num=4)
        r1, r2, r3, r4 = jax.random.uniform(k1, shape=(4,))

        def no(key_, nodes_, conns_):
            return nodes_, conns_

        nodes, conns = jax.lax.cond(r1 < self.node_add, mutate_add_node, no, k1, nodes, conns)
        nodes, conns = jax.lax.cond(r2 < self.node_delete, mutate_delete_node, no, k2, nodes, conns)
        nodes, conns = jax.lax.cond(r3 < self.conn_add, mutate_add_conn, no, k3, nodes, conns)
        nodes, conns = jax.lax.cond(r4 < self.conn_delete, mutate_delete_conn, no, k4, nodes, conns)

        return nodes, conns

    def mutate_values(self, randkey, genome, nodes, conns):
        k1, k2 = jax.random.split(randkey, num=2)
        nodes_keys = jax.random.split(k1, num=nodes.shape[0])
        conns_keys = jax.random.split(k2, num=conns.shape[0])

        new_nodes = jax.vmap(genome.node_gene.mutate)(nodes_keys, nodes)
        new_conns = jax.vmap(genome.conn_gene.mutate)(conns_keys, conns)

        # nan nodes not changed
        new_nodes = jnp.where(jnp.isnan(nodes), jnp.nan, new_nodes)
        new_conns = jnp.where(jnp.isnan(conns), jnp.nan, new_conns)

        return new_nodes, new_conns

    def choice_node_key(self, rand_key, nodes, input_idx, output_idx,
                        allow_input_keys: bool = False, allow_output_keys: bool = False):
        """
        Randomly choose a node key from the given nodes. It guarantees that the chosen node not be the input or output node.
        :param rand_key:
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

        idx = fetch_random(rand_key, mask)
        key = jnp.where(idx != I_INT, nodes[idx, 0], jnp.nan)
        return key, idx

    def choice_connection_key(self, rand_key, conns):
        """
        Randomly choose a connection key from the given connections.
        :return: i_key, o_key, idx
        """

        idx = fetch_random(rand_key, ~jnp.isnan(conns[:, 0]))
        i_key = jnp.where(idx != I_INT, conns[idx, 0], jnp.nan)
        o_key = jnp.where(idx != I_INT, conns[idx, 1], jnp.nan)

        return i_key, o_key, idx
