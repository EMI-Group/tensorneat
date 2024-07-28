import jax
from jax import vmap, numpy as jnp
from .base import BaseCrossover
from ...utils import (
    extract_node_attrs,
    extract_conn_attrs,
    set_node_attrs,
    set_conn_attrs,
)


class DefaultCrossover(BaseCrossover):
    def __call__(self, state, genome, randkey, nodes1, conns1, nodes2, conns2):
        """
        use genome1 and genome2 to generate a new genome
        notice that genome1 should have higher fitness than genome2 (genome1 is winner!)
        """
        randkey1, randkey2 = jax.random.split(randkey, 2)
        randkeys1 = jax.random.split(randkey1, genome.max_nodes)
        randkeys2 = jax.random.split(randkey2, genome.max_conns)

        # crossover nodes
        keys1, keys2 = nodes1[:, 0], nodes2[:, 0]
        # make homologous genes align in nodes2 align with nodes1
        nodes2 = self.align_array(keys1, keys2, nodes2, is_conn=False)

        # For not homologous genes, use the value of nodes1(winner)
        # For homologous genes, use the crossover result between nodes1 and nodes2
        node_attrs1 = vmap(extract_node_attrs)(nodes1)
        node_attrs2 = vmap(extract_node_attrs)(nodes2)

        new_node_attrs = jnp.where(
            jnp.isnan(node_attrs1) | jnp.isnan(node_attrs2),  # one of them is nan
            node_attrs1,  # not homologous genes or both nan, use the value of nodes1(winner)
            vmap(genome.node_gene.crossover, in_axes=(None, 0, 0, 0))(
                state, randkeys1, node_attrs1, node_attrs2
            ),  # homologous or both nan
        )
        new_nodes = vmap(set_node_attrs)(nodes1, new_node_attrs)

        # crossover connections
        con_keys1, con_keys2 = conns1[:, :2], conns2[:, :2]
        conns2 = self.align_array(con_keys1, con_keys2, conns2, is_conn=True)

        conns_attrs1 = vmap(extract_conn_attrs)(conns1)
        conns_attrs2 = vmap(extract_conn_attrs)(conns2)

        new_conn_attrs = jnp.where(
            jnp.isnan(conns_attrs1) | jnp.isnan(conns_attrs2),
            conns_attrs1,  # not homologous genes or both nan, use the value of conns1(winner)
            vmap(genome.conn_gene.crossover, in_axes=(None, 0, 0, 0))(
                state, randkeys2, conns_attrs1, conns_attrs2
            ),  # homologous or both nan
        )
        new_conns = vmap(set_conn_attrs)(conns1, new_conn_attrs)

        return new_nodes, new_conns

    def align_array(self, seq1, seq2, ar2, is_conn: bool):
        """
        After I review this code, I found that it is the most difficult part of the code.
        Please consider carefully before change it!
        make ar2 align with ar1.
        :param seq1:
        :param seq2:
        :param ar2:
        :param is_conn:
        :return:
        align means to intersect part of ar2 will be at the same position as ar1,
        non-intersect part of ar2 will be set to Nan
        """
        seq1, seq2 = seq1[:, jnp.newaxis], seq2[jnp.newaxis, :]
        mask = (seq1 == seq2) & (~jnp.isnan(seq1))

        if is_conn:
            mask = jnp.all(mask, axis=2)

        intersect_mask = mask.any(axis=1)
        idx = jnp.arange(0, len(seq1))
        idx_fixed = jnp.dot(mask, idx)

        refactor_ar2 = jnp.where(
            intersect_mask[:, jnp.newaxis], ar2[idx_fixed], jnp.nan
        )

        return refactor_ar2
