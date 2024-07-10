import jax, jax.numpy as jnp

from .base import BaseSpecies
from tensorneat.common import (
    State,
    rank_elements,
    argmin_with_mask,
    fetch_first,
)
from tensorneat.genome.utils import (
    extract_conn_attrs,
    extract_node_attrs,
)
from tensorneat.genome import BaseGenome


"""
Core procedures of NEAT algorithm, contains the following steps:
1. Update the fitness of each species;
2. Decide which species will be stagnation;
3. Decide the number of members of each species in the next generation;
4. Choice the crossover pair for each species;
5. Divided the whole new population into different species;

This class use tensor operation to imitate the behavior of NEAT algorithm which implemented in NEAT-python.
The code may be hard to understand. Fortunately, we don't need to overwrite it in most cases.
"""


class DefaultSpecies(BaseSpecies):
    def __init__(
        self,
        genome: BaseGenome,
        pop_size,
        species_size,
        compatibility_disjoint: float = 1.0,
        compatibility_weight: float = 0.4,
        max_stagnation: int = 15,
        species_elitism: int = 2,
        spawn_number_change_rate: float = 0.5,
        genome_elitism: int = 2,
        survival_threshold: float = 0.2,
        min_species_size: int = 1,
        compatibility_threshold: float = 3.0,
    ):
        self.genome = genome
        self.pop_size = pop_size
        self.species_size = species_size

        self.compatibility_disjoint = compatibility_disjoint
        self.compatibility_weight = compatibility_weight
        self.max_stagnation = max_stagnation
        self.species_elitism = species_elitism
        self.spawn_number_change_rate = spawn_number_change_rate
        self.genome_elitism = genome_elitism
        self.survival_threshold = survival_threshold
        self.min_species_size = min_species_size
        self.compatibility_threshold = compatibility_threshold

        self.species_arange = jnp.arange(self.species_size)

    def setup(self, state=State()):
        state = self.genome.setup(state)
        k1, randkey = jax.random.split(state.randkey, 2)

        # initialize the population
        initialize_keys = jax.random.split(randkey, self.pop_size)
        pop_nodes, pop_conns = jax.vmap(self.genome.initialize, in_axes=(None, 0))(
            state, initialize_keys
        )

        species_keys = jnp.full(
            (self.species_size,), jnp.nan
        )  # the unique index (primary key) for each species
        best_fitness = jnp.full(
            (self.species_size,), jnp.nan
        )  # the best fitness of each species
        last_improved = jnp.full(
            (self.species_size,), jnp.nan
        )  # the last 1 that the species improved
        member_count = jnp.full(
            (self.species_size,), jnp.nan
        )  # the number of members of each species
        idx2species = jnp.zeros(self.pop_size)  # the species index of each individual

        # nodes for each center genome of each species
        center_nodes = jnp.full(
            (self.species_size, self.genome.max_nodes, self.genome.node_gene.length),
            jnp.nan,
        )

        # connections for each center genome of each species
        center_conns = jnp.full(
            (self.species_size, self.genome.max_conns, self.genome.conn_gene.length),
            jnp.nan,
        )

        species_keys = species_keys.at[0].set(0)
        best_fitness = best_fitness.at[0].set(-jnp.inf)
        last_improved = last_improved.at[0].set(0)
        member_count = member_count.at[0].set(self.pop_size)
        center_nodes = center_nodes.at[0].set(pop_nodes[0])
        center_conns = center_conns.at[0].set(pop_conns[0])

        pop_nodes, pop_conns = jax.device_put((pop_nodes, pop_conns))

        state = state.update(randkey=randkey)

        return state.register(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            member_count=member_count,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,
            next_species_key=jnp.float32(1),  # 0 is reserved for the first species
            generation=jnp.float32(0),
        )

    def ask(self, state):
        return state.pop_nodes, state.pop_conns

    def tell(self, state, fitness):
        k1, k2, randkey = jax.random.split(state.randkey, 3)

        state = state.update(generation=state.generation + 1, randkey=randkey)
        state, winner, loser, elite_mask = self.update_species(state, fitness)
        state = self.create_next_generation(state, winner, loser, elite_mask)
        state = self.speciate(state)

        return state

    def update_species(self, state, fitness):
        # update the fitness of each species
        state, species_fitness = self.update_species_fitness(state, fitness)

        # stagnation species
        state, species_fitness = self.stagnation(state, species_fitness)

        # sort species_info by their fitness. (also push nan to the end)
        sort_indices = jnp.argsort(species_fitness)[::-1]

        state = state.update(
            species_keys=state.species_keys[sort_indices],
            best_fitness=state.best_fitness[sort_indices],
            last_improved=state.last_improved[sort_indices],
            member_count=state.member_count[sort_indices],
            center_nodes=state.center_nodes[sort_indices],
            center_conns=state.center_conns[sort_indices],
        )

        # decide the number of members of each species by their fitness
        state, spawn_number = self.cal_spawn_numbers(state)

        k1, k2 = jax.random.split(state.randkey)
        # crossover info
        state, winner, loser, elite_mask = self.create_crossover_pair(
            state, spawn_number, fitness
        )

        return state.update(randkey=k2), winner, loser, elite_mask

    def update_species_fitness(self, state, fitness):
        """
        obtain the fitness of the species by the fitness of each individual.
        use max criterion.
        """

        def aux_func(idx):
            s_fitness = jnp.where(
                state.idx2species == state.species_keys[idx], fitness, -jnp.inf
            )
            val = jnp.max(s_fitness)
            return val

        return state, jax.vmap(aux_func)(self.species_arange)

    def stagnation(self, state, species_fitness):
        """
        stagnation species.
        those species whose fitness is not better than the best fitness of the species for a long time will be stagnation.
        elitism species never stagnation
        """

        def check_stagnation(idx):
            # determine whether the species stagnation
            st = (
                species_fitness[idx] <= state.best_fitness[idx]
            ) & (  # not better than the best fitness of the species
                state.generation - state.last_improved[idx] > self.max_stagnation
            )  # for a long time

            # update last_improved and best_fitness
            li, bf = jax.lax.cond(
                species_fitness[idx] > state.best_fitness[idx],
                lambda: (state.generation, species_fitness[idx]),  # update
                lambda: (
                    state.last_improved[idx],
                    state.best_fitness[idx],
                ),  # not update
            )

            return st, bf, li

        spe_st, best_fitness, last_improved = jax.vmap(check_stagnation)(
            self.species_arange
        )

        # elite species will not be stagnation
        species_rank = rank_elements(species_fitness)
        spe_st = jnp.where(
            species_rank < self.species_elitism, False, spe_st
        )  # elitism never stagnation

        # set stagnation species to nan
        def update_func(idx):
            return jax.lax.cond(
                spe_st[idx],
                lambda: (
                    jnp.nan,  # species_key
                    jnp.nan,  # best_fitness
                    jnp.nan,  # last_improved
                    jnp.nan,  # member_count
                    -jnp.inf,  # species_fitness
                    jnp.full_like(state.center_nodes[idx], jnp.nan),  # center_nodes
                    jnp.full_like(state.center_conns[idx], jnp.nan),  # center_conns
                ),  # stagnation species
                lambda: (
                    state.species_keys[idx],
                    best_fitness[idx],
                    last_improved[idx],
                    state.member_count[idx],
                    species_fitness[idx],
                    state.center_nodes[idx],
                    state.center_conns[idx],
                ),  # not stagnation species
            )

        (
            species_keys,
            best_fitness,
            last_improved,
            member_count,
            species_fitness,
            center_nodes,
            center_conns,
        ) = jax.vmap(update_func)(self.species_arange)

        return (
            state.update(
                species_keys=species_keys,
                best_fitness=best_fitness,
                last_improved=last_improved,
                member_count=member_count,
                center_nodes=center_nodes,
                center_conns=center_conns,
            ),
            species_fitness,
        )

    def cal_spawn_numbers(self, state):
        """
        decide the number of members of each species by their fitness rank.
        the species with higher fitness will have more members
        Linear ranking selection
            e.g. N = 3, P=10 -> probability = [0.5, 0.33, 0.17], spawn_number = [5, 3, 2]
        """

        species_keys = state.species_keys

        is_species_valid = ~jnp.isnan(species_keys)
        valid_species_num = jnp.sum(is_species_valid)
        denominator = (
            (valid_species_num + 1) * valid_species_num / 2
        )  # obtain 3 + 2 + 1 = 6

        rank_score = valid_species_num - self.species_arange  # obtain [3, 2, 1]
        spawn_number_rate = rank_score / denominator  # obtain [0.5, 0.33, 0.17]
        spawn_number_rate = jnp.where(
            is_species_valid, spawn_number_rate, 0
        )  # set invalid species to 0

        target_spawn_number = jnp.floor(
            spawn_number_rate * self.pop_size
        )  # calculate member

        # Avoid too much variation of numbers for a species
        previous_size = state.member_count
        spawn_number = (
            previous_size
            + (target_spawn_number - previous_size) * self.spawn_number_change_rate
        )
        spawn_number = spawn_number.astype(jnp.int32)

        # must control the sum of spawn_number to be equal to pop_size
        error = self.pop_size - jnp.sum(spawn_number)

        # add error to the first species to control the sum of spawn_number
        spawn_number = spawn_number.at[0].add(error)

        return state, spawn_number

    def create_crossover_pair(self, state, spawn_number, fitness):
        s_idx = self.species_arange
        p_idx = jnp.arange(self.pop_size)

        def aux_func(key, idx):
            members = state.idx2species == state.species_keys[idx]
            members_num = jnp.sum(members)

            members_fitness = jnp.where(members, fitness, -jnp.inf)
            sorted_member_indices = jnp.argsort(members_fitness)[::-1]

            survive_size = jnp.floor(self.survival_threshold * members_num).astype(
                jnp.int32
            )

            select_pro = (p_idx < survive_size) / survive_size
            fa, ma = jax.random.choice(
                key,
                sorted_member_indices,
                shape=(2, self.pop_size),
                replace=True,
                p=select_pro,
            )

            # elite
            fa = jnp.where(p_idx < self.genome_elitism, sorted_member_indices, fa)
            ma = jnp.where(p_idx < self.genome_elitism, sorted_member_indices, ma)
            elite = jnp.where(p_idx < self.genome_elitism, True, False)
            return fa, ma, elite

        randkey_, randkey = jax.random.split(state.randkey)
        fas, mas, elites = jax.vmap(aux_func)(
            jax.random.split(randkey_, self.species_size), s_idx
        )

        spawn_number_cum = jnp.cumsum(spawn_number)

        def aux_func(idx):
            loc = jnp.argmax(idx < spawn_number_cum)

            # elite genomes are at the beginning of the species
            idx_in_species = jnp.where(loc > 0, idx - spawn_number_cum[loc - 1], idx)
            return (
                fas[loc, idx_in_species],
                mas[loc, idx_in_species],
                elites[loc, idx_in_species],
            )

        part1, part2, elite_mask = jax.vmap(aux_func)(p_idx)

        is_part1_win = fitness[part1] >= fitness[part2]
        winner = jnp.where(is_part1_win, part1, part2)
        loser = jnp.where(is_part1_win, part2, part1)

        return state.update(randkey=randkey), winner, loser, elite_mask

    def speciate(self, state):
        # prepare distance functions
        o2p_distance_func = jax.vmap(
            self.distance, in_axes=(None, None, None, 0, 0)
        )  # one to population

        # idx to specie key
        idx2species = jnp.full(
            (self.pop_size,), jnp.nan
        )  # NaN means not assigned to any species

        # the distance between genomes to its center genomes
        o2c_distances = jnp.full((self.pop_size,), jnp.inf)

        # step 1: find new centers
        def cond_func(carry):
            # i, idx2species, center_nodes, center_conns, o2c_distances
            i, i2s, cns, ccs, o2c = carry

            return (i < self.species_size) & (
                ~jnp.isnan(state.species_keys[i])
            )  # current species is existing

        def body_func(carry):
            i, i2s, cns, ccs, o2c = carry

            distances = o2p_distance_func(
                state, cns[i], ccs[i], state.pop_nodes, state.pop_conns
            )

            # find the closest one
            closest_idx = argmin_with_mask(distances, mask=jnp.isnan(i2s))

            i2s = i2s.at[closest_idx].set(state.species_keys[i])
            cns = cns.at[i].set(state.pop_nodes[closest_idx])
            ccs = ccs.at[i].set(state.pop_conns[closest_idx])

            # the genome with closest_idx will become the new center, thus its distance to center is 0.
            o2c = o2c.at[closest_idx].set(0)

            return i + 1, i2s, cns, ccs, o2c

        _, idx2species, center_nodes, center_conns, o2c_distances = jax.lax.while_loop(
            cond_func,
            body_func,
            (0, idx2species, state.center_nodes, state.center_conns, o2c_distances),
        )

        state = state.update(
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,
        )

        # part 2: assign members to each species
        def cond_func(carry):
            # i, idx2species, center_nodes, center_conns, species_keys, o2c_distances, next_species_key
            i, i2s, cns, ccs, sk, o2c, nsk = carry

            current_species_existed = ~jnp.isnan(sk[i])
            not_all_assigned = jnp.any(jnp.isnan(i2s))
            not_reach_species_upper_bounds = i < self.species_size
            return not_reach_species_upper_bounds & (
                current_species_existed | not_all_assigned
            )

        def body_func(carry):
            i, i2s, cns, ccs, sk, o2c, nsk = carry

            _, i2s, cns, ccs, sk, o2c, nsk = jax.lax.cond(
                jnp.isnan(sk[i]),  # whether the current species is existing or not
                create_new_species,  # if not existing, create a new specie
                update_exist_specie,  # if existing, update the specie
                (i, i2s, cns, ccs, sk, o2c, nsk),
            )

            return i + 1, i2s, cns, ccs, sk, o2c, nsk

        def create_new_species(carry):
            i, i2s, cns, ccs, sk, o2c, nsk = carry

            # pick the first one who has not been assigned to any species
            idx = fetch_first(jnp.isnan(i2s))

            # assign it to the new species
            # [key, best score, last update generation, member_count]
            sk = sk.at[i].set(nsk)  # nsk -> next species key
            i2s = i2s.at[idx].set(nsk)
            o2c = o2c.at[idx].set(0)

            # update center genomes
            cns = cns.at[i].set(state.pop_nodes[idx])
            ccs = ccs.at[i].set(state.pop_conns[idx])

            # find the members for the new species
            i2s, o2c = speciate_by_threshold(i, i2s, cns, ccs, sk, o2c)

            return i, i2s, cns, ccs, sk, o2c, nsk + 1  # change to next new speciate key

        def update_exist_specie(carry):
            i, i2s, cns, ccs, sk, o2c, nsk = carry

            i2s, o2c = speciate_by_threshold(i, i2s, cns, ccs, sk, o2c)

            # turn to next species
            return i + 1, i2s, cns, ccs, sk, o2c, nsk

        def speciate_by_threshold(i, i2s, cns, ccs, sk, o2c):
            # distance between such center genome and ppo genomes
            o2p_distance = o2p_distance_func(
                state, cns[i], ccs[i], state.pop_nodes, state.pop_conns
            )

            close_enough_mask = o2p_distance < self.compatibility_threshold
            # when a genome is not assigned or the distance between its current center is bigger than this center
            catchable_mask = jnp.isnan(i2s) | (o2p_distance < o2c)

            mask = close_enough_mask & catchable_mask

            # update species info
            i2s = jnp.where(mask, sk[i], i2s)

            # update distance between centers
            o2c = jnp.where(mask, o2p_distance, o2c)

            return i2s, o2c

        # update idx2species
        (
            _,
            idx2species,
            center_nodes,
            center_conns,
            species_keys,
            _,
            next_species_key,
        ) = jax.lax.while_loop(
            cond_func,
            body_func,
            (
                0,
                state.idx2species,
                center_nodes,
                center_conns,
                state.species_keys,
                o2c_distances,
                state.next_species_key,
            ),
        )

        # if there are still some pop genomes not assigned to any species, add them to the last genome
        # this condition can only happen when the number of species is reached species upper bounds
        idx2species = jnp.where(jnp.isnan(idx2species), species_keys[-1], idx2species)

        # complete info of species which is created in this generation
        new_created_mask = (~jnp.isnan(species_keys)) & jnp.isnan(state.best_fitness)
        best_fitness = jnp.where(new_created_mask, -jnp.inf, state.best_fitness)
        last_improved = jnp.where(
            new_created_mask, state.generation, state.last_improved
        )

        # update members count
        def count_members(idx):
            return jax.lax.cond(
                jnp.isnan(species_keys[idx]),  # if the species is not existing
                lambda: jnp.nan,  # nan
                lambda: jnp.sum(
                    idx2species == species_keys[idx], dtype=jnp.float32
                ),  # count members
            )

        member_count = jax.vmap(count_members)(self.species_arange)

        return state.update(
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            member_count=member_count,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,
            next_species_key=next_species_key,
        )

    def distance(self, state, nodes1, conns1, nodes2, conns2):
        """
        The distance between two genomes
        """
        d = self.node_distance(state, nodes1, nodes2) + self.conn_distance(
            state, conns1, conns2
        )
        return d

    def node_distance(self, state, nodes1, nodes2):
        """
        The distance of the nodes part for two genomes
        """
        node_cnt1 = jnp.sum(~jnp.isnan(nodes1[:, 0]))
        node_cnt2 = jnp.sum(~jnp.isnan(nodes2[:, 0]))
        max_cnt = jnp.maximum(node_cnt1, node_cnt2)

        # align homologous nodes
        # this process is similar to np.intersect1d.
        nodes = jnp.concatenate((nodes1, nodes2), axis=0)
        keys = nodes[:, 0]
        sorted_indices = jnp.argsort(keys, axis=0)
        nodes = nodes[sorted_indices]
        nodes = jnp.concatenate(
            [nodes, jnp.full((1, nodes.shape[1]), jnp.nan)], axis=0
        )  # add a nan row to the end
        fr, sr = nodes[:-1], nodes[1:]  # first row, second row

        # flag location of homologous nodes
        intersect_mask = (fr[:, 0] == sr[:, 0]) & ~jnp.isnan(nodes[:-1, 0])

        # calculate the count of non_homologous of two genomes
        non_homologous_cnt = node_cnt1 + node_cnt2 - 2 * jnp.sum(intersect_mask)

        # calculate the distance of homologous nodes
        fr_attrs = jax.vmap(extract_node_attrs)(fr)
        sr_attrs = jax.vmap(extract_node_attrs)(sr)
        hnd = jax.vmap(self.genome.node_gene.distance, in_axes=(None, 0, 0))(
            state, fr_attrs, sr_attrs
        )  # homologous node distance
        hnd = jnp.where(jnp.isnan(hnd), 0, hnd)
        homologous_distance = jnp.sum(hnd * intersect_mask)

        val = (
            non_homologous_cnt * self.compatibility_disjoint
            + homologous_distance * self.compatibility_weight
        )

        val = jnp.where(max_cnt == 0, 0, val / max_cnt)  # normalize

        return val

    def conn_distance(self, state, conns1, conns2):
        """
        The distance of the conns part for two genomes
        """
        con_cnt1 = jnp.sum(~jnp.isnan(conns1[:, 0]))
        con_cnt2 = jnp.sum(~jnp.isnan(conns2[:, 0]))
        max_cnt = jnp.maximum(con_cnt1, con_cnt2)

        cons = jnp.concatenate((conns1, conns2), axis=0)
        keys = cons[:, :2]
        sorted_indices = jnp.lexsort(keys.T[::-1])
        cons = cons[sorted_indices]
        cons = jnp.concatenate(
            [cons, jnp.full((1, cons.shape[1]), jnp.nan)], axis=0
        )  # add a nan row to the end
        fr, sr = cons[:-1], cons[1:]  # first row, second row

        # both genome has such connection
        intersect_mask = jnp.all(fr[:, :2] == sr[:, :2], axis=1) & ~jnp.isnan(fr[:, 0])

        non_homologous_cnt = con_cnt1 + con_cnt2 - 2 * jnp.sum(intersect_mask)

        fr_attrs = jax.vmap(extract_conn_attrs)(fr)
        sr_attrs = jax.vmap(extract_conn_attrs)(sr)
        hcd = jax.vmap(self.genome.conn_gene.distance, in_axes=(None, 0, 0))(
            state, fr_attrs, sr_attrs
        )  # homologous connection distance
        hcd = jnp.where(jnp.isnan(hcd), 0, hcd)
        homologous_distance = jnp.sum(hcd * intersect_mask)

        val = (
            non_homologous_cnt * self.compatibility_disjoint
            + homologous_distance * self.compatibility_weight
        )

        val = jnp.where(max_cnt == 0, 0, val / max_cnt)  # normalize

        return val

    def create_next_generation(self, state, winner, loser, elite_mask):

        # find next node key
        all_nodes_keys = state.pop_nodes[:, :, 0]
        max_node_key = jnp.max(
            all_nodes_keys, where=~jnp.isnan(all_nodes_keys), initial=0
        )
        next_node_key = max_node_key + 1
        new_node_keys = jnp.arange(self.pop_size) + next_node_key

        # prepare random keys
        k1, k2, randkey = jax.random.split(state.randkey, 3)
        crossover_randkeys = jax.random.split(k1, self.pop_size)
        mutate_randkeys = jax.random.split(k2, self.pop_size)

        wpn, wpc = state.pop_nodes[winner], state.pop_conns[winner]
        lpn, lpc = state.pop_nodes[loser], state.pop_conns[loser]

        # batch crossover
        n_nodes, n_conns = jax.vmap(
            self.genome.execute_crossover, in_axes=(None, 0, 0, 0, 0, 0)
        )(
            state, crossover_randkeys, wpn, wpc, lpn, lpc
        )  # new_nodes, new_conns

        # batch mutation
        m_n_nodes, m_n_conns = jax.vmap(
            self.genome.execute_mutation, in_axes=(None, 0, 0, 0, 0)
        )(
            state, mutate_randkeys, n_nodes, n_conns, new_node_keys
        )  # mutated_new_nodes, mutated_new_conns

        # elitism don't mutate
        pop_nodes = jnp.where(elite_mask[:, None, None], n_nodes, m_n_nodes)
        pop_conns = jnp.where(elite_mask[:, None, None], n_conns, m_n_conns)

        return state.update(
            randkey=randkey,
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
        )
