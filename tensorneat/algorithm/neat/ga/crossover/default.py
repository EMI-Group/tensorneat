import jax, jax.numpy as jnp

from .base import BaseCrossover


class DefaultCrossover(BaseCrossover):

    def __call__(self, state, genome, nodes1, conns1, nodes2, conns2):
        """
        use genome1 and genome2 to generate a new genome
        notice that genome1 should have higher fitness than genome2 (genome1 is winner!)
        """
        randkey1, randkey2, randkey = jax.random.split(state.randkey, 3)

        # crossover nodes
        keys1, keys2 = nodes1[:, 0], nodes2[:, 0]
        # make homologous genes align in nodes2 align with nodes1
        nodes2 = self.align_array(keys1, keys2, nodes2, is_conn=False)

        # For not homologous genes, use the value of nodes1(winner)
        # For homologous genes, use the crossover result between nodes1 and nodes2
        new_nodes = jnp.where(jnp.isnan(nodes1) | jnp.isnan(nodes2), nodes1,
                              self.crossover_gene(randkey1, nodes1, nodes2, is_conn=False))

        # crossover connections
        con_keys1, con_keys2 = conns1[:, :2], conns2[:, :2]
        conns2 = self.align_array(con_keys1, con_keys2, conns2, is_conn=True)

        new_conns = jnp.where(jnp.isnan(conns1) | jnp.isnan(conns2), conns1,
                              self.crossover_gene(randkey2, conns1, conns2, is_conn=True))

        return state.update(randkey=randkey), new_nodes, new_conns

    def align_array(self, seq1, seq2, ar2, is_conn: bool):
        """
        After I review this code, I found that it is the most difficult part of the code. Please never change it!
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

        refactor_ar2 = jnp.where(intersect_mask[:, jnp.newaxis], ar2[idx_fixed], jnp.nan)

        return refactor_ar2

    def crossover_gene(self, rand_key, g1, g2, is_conn):
        r = jax.random.uniform(rand_key, shape=g1.shape)
        new_gene = jnp.where(r > 0.5, g1, g2)
        if is_conn:  # fix enabled
            enabled = jnp.where(
                g1[:, 2] + g2[:, 2] > 0,  # any of them is enabled
                1,
                0
            )
            new_gene = new_gene.at[:, 2].set(enabled)
        return new_gene
