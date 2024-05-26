import numpy as np
import jax, jax.numpy as jnp
from utils import State, rank_elements, argmin_with_mask, fetch_first
from ..genome import BaseGenome
from .base import BaseSpecies


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
        initialize_method: str = "one_hidden_node",
        # {'one_hidden_node', 'dense_hideen_layer', 'no_hidden_random'}
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
        self.initialize_method = initialize_method

        self.species_arange = jnp.arange(self.species_size)

    def setup(self, state=State()):
        state = self.genome.setup(state)
        k1, randkey = jax.random.split(state.randkey, 2)
        pop_nodes, pop_conns = initialize_population(
            self.pop_size, self.genome, k1, self.initialize_method
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

        return state.register(
            randkey=randkey,
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            member_count=member_count,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,
            next_species_key=jnp.array(1),  # 0 is reserved for the first species
        )

    def ask(self, state):
        return state, state.pop_nodes, state.pop_conns

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
        winner, loser, elite_mask = self.create_crossover_pair(
            state, k1, spawn_number, fitness
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

        return state(randkey=randkey), winner, loser, elite_mask

    def speciate(self, state):
        # prepare distance functions
        o2p_distance_func = jax.vmap(
            self.distance, in_axes=(None, None, 0, 0)
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
                cns[i], ccs[i], state.pop_nodes, state.pop_conns
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
                cns[i], ccs[i], state.pop_nodes, state.pop_conns
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

    def distance(self, nodes1, conns1, nodes2, conns2):
        """
        The distance between two genomes
        """
        d = self.node_distance(nodes1, nodes2) + self.conn_distance(conns1, conns2)
        return d

    def node_distance(self, nodes1, nodes2):
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
        hnd = jax.vmap(self.genome.node_gene.distance, in_axes=(0, 0))(fr, sr)
        hnd = jnp.where(jnp.isnan(hnd), 0, hnd)
        homologous_distance = jnp.sum(hnd * intersect_mask)

        val = (
            non_homologous_cnt * self.compatibility_disjoint
            + homologous_distance * self.compatibility_weight
        )

        return jnp.where(max_cnt == 0, 0, val / max_cnt)  # avoid zero division

    def conn_distance(self, conns1, conns2):
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
        hcd = jax.vmap(self.genome.conn_gene.distance, in_axes=(0, 0))(fr, sr)
        hcd = jnp.where(jnp.isnan(hcd), 0, hcd)
        homologous_distance = jnp.sum(hcd * intersect_mask)

        val = (
            non_homologous_cnt * self.compatibility_disjoint
            + homologous_distance * self.compatibility_weight
        )

        return jnp.where(max_cnt == 0, 0, val / max_cnt)


def initialize_population(pop_size, genome, randkey, init_method="default"):
    rand_keys = jax.random.split(randkey, pop_size)

    if init_method == "one_hidden_node":
        init_func = init_one_hidden_node
    elif init_method == "dense_hideen_layer":
        init_func = init_dense_hideen_layer
    elif init_method == "no_hidden_random":
        init_func = init_no_hidden_random
    else:
        raise ValueError("Unknown initialization method: {}".format(init_method))

    pop_nodes, pop_conns = jax.vmap(init_func, in_axes=(None, 0))(genome, rand_keys)

    return pop_nodes, pop_conns


# one hidden node
def init_one_hidden_node(genome, randkey):
    input_idx, output_idx = genome.input_idx, genome.output_idx
    new_node_key = max([*input_idx, *output_idx]) + 1

    nodes = jnp.full((genome.max_nodes, genome.node_gene.length), jnp.nan)
    conns = jnp.full((genome.max_conns, genome.conn_gene.length), jnp.nan)

    nodes = nodes.at[input_idx, 0].set(input_idx)
    nodes = nodes.at[output_idx, 0].set(output_idx)
    nodes = nodes.at[new_node_key, 0].set(new_node_key)

    rand_keys_nodes = jax.random.split(
        randkey, num=len(input_idx) + len(output_idx) + 1
    )
    input_keys, output_keys, hidden_key = (
        rand_keys_nodes[: len(input_idx)],
        rand_keys_nodes[len(input_idx) : len(input_idx) + len(output_idx)],
        rand_keys_nodes[-1],
    )

    node_attr_func = jax.vmap(genome.node_gene.new_attrs, in_axes=(None, 0))
    input_attrs = node_attr_func(input_keys)
    output_attrs = node_attr_func(output_keys)
    hidden_attrs = genome.node_gene.new_custom_attrs(hidden_key)

    nodes = nodes.at[input_idx, 1:].set(input_attrs)
    nodes = nodes.at[output_idx, 1:].set(output_attrs)
    nodes = nodes.at[new_node_key, 1:].set(hidden_attrs)

    input_conns = jnp.c_[input_idx, jnp.full_like(input_idx, new_node_key)]
    conns = conns.at[input_idx, 0:2].set(input_conns)
    conns = conns.at[input_idx, 2].set(True)

    output_conns = jnp.c_[jnp.full_like(output_idx, new_node_key), output_idx]
    conns = conns.at[output_idx, 0:2].set(output_conns)
    conns = conns.at[output_idx, 2].set(True)

    rand_keys_conns = jax.random.split(randkey, num=len(input_idx) + len(output_idx))
    input_conn_keys, output_conn_keys = (
        rand_keys_conns[: len(input_idx)],
        rand_keys_conns[len(input_idx) :],
    )

    conn_attr_func = jax.vmap(genome.conn_gene.new_random_attrs, in_axes=(None, 0))
    input_conn_attrs = conn_attr_func(input_conn_keys)
    output_conn_attrs = conn_attr_func(output_conn_keys)

    conns = conns.at[input_idx, 3:].set(input_conn_attrs)
    conns = conns.at[output_idx, 3:].set(output_conn_attrs)

    return nodes, conns


# random dense connections with 1 hidden layer
def init_dense_hideen_layer(genome, randkey, hiddens=20):
    k1, k2, k3 = jax.random.split(randkey, num=3)
    input_idx, output_idx = genome.input_idx, genome.output_idx
    input_size = len(input_idx)
    output_size = len(output_idx)

    hidden_idx = jnp.arange(
        input_size + output_size, input_size + output_size + hiddens
    )
    nodes = jnp.full(
        (genome.max_nodes, genome.node_gene.length), jnp.nan, dtype=jnp.float32
    )
    nodes = nodes.at[input_idx, 0].set(input_idx)
    nodes = nodes.at[output_idx, 0].set(output_idx)
    nodes = nodes.at[hidden_idx, 0].set(hidden_idx)

    total_idx = input_size + output_size + hiddens
    rand_keys_n = jax.random.split(k1, num=total_idx)
    input_keys = rand_keys_n[:input_size]
    output_keys = rand_keys_n[input_size : input_size + output_size]
    hidden_keys = rand_keys_n[input_size + output_size :]

    node_attr_func = jax.vmap(genome.conn_gene.new_random_attrs, in_axes=(0))
    input_attrs = node_attr_func(input_keys)
    output_attrs = node_attr_func(output_keys)
    hidden_attrs = node_attr_func(hidden_keys)

    nodes = nodes.at[input_idx, 1:].set(input_attrs)
    nodes = nodes.at[output_idx, 1:].set(output_attrs)
    nodes = nodes.at[hidden_idx, 1:].set(hidden_attrs)

    total_connections = input_size * hiddens + hiddens * output_size
    conns = jnp.full(
        (genome.max_conns, genome.conn_gene.length), jnp.nan, dtype=jnp.float32
    )

    rand_keys_c = jax.random.split(k2, num=total_connections)
    conns_attr_func = jax.vmap(genome.node_gene.new_random_attrs, in_axes=(0))
    conns_attrs = conns_attr_func(rand_keys_c)

    input_to_hidden_ids, hidden_ids = jnp.meshgrid(input_idx, hidden_idx, indexing="ij")
    hidden_to_output_ids, output_ids = jnp.meshgrid(
        hidden_idx, output_idx, indexing="ij"
    )

    conns = conns.at[: input_size * hiddens, 0].set(input_to_hidden_ids.flatten())
    conns = conns.at[: input_size * hiddens, 1].set(hidden_ids.flatten())
    conns = conns.at[input_size * hiddens : total_connections, 0].set(
        hidden_to_output_ids.flatten()
    )
    conns = conns.at[input_size * hiddens : total_connections, 1].set(
        output_ids.flatten()
    )
    conns = conns.at[: input_size * hiddens + hiddens * output_size, 2].set(True)
    conns = conns.at[: input_size * hiddens + hiddens * output_size, 3:].set(
        conns_attrs
    )

    return nodes, conns


# random sparse connections with no hidden nodes
def init_no_hidden_random(genome, randkey):
    k1, k2, k3 = jax.random.split(randkey, num=3)
    input_idx, output_idx = genome.input_idx, genome.output_idx

    nodes = jnp.full((genome.max_nodes, genome.node_gene.length), jnp.nan)
    nodes = nodes.at[input_idx, 0].set(input_idx)
    nodes = nodes.at[output_idx, 0].set(output_idx)

    total_idx = len(input_idx) + len(output_idx)
    rand_keys_n = jax.random.split(k1, num=total_idx)
    input_keys = rand_keys_n[: len(input_idx)]
    output_keys = rand_keys_n[len(input_idx) :]

    node_attr_func = jax.vmap(genome.node_gene.new_random_attrs, in_axes=(0))
    input_attrs = node_attr_func(input_keys)
    output_attrs = node_attr_func(output_keys)
    nodes = nodes.at[input_idx, 1:].set(input_attrs)
    nodes = nodes.at[output_idx, 1:].set(output_attrs)

    conns = jnp.full((genome.max_conns, genome.conn_gene.length), jnp.nan)

    num_connections_per_output = 4
    total_connections = len(output_idx) * num_connections_per_output

    def create_connections_for_output(key):
        permuted_inputs = jax.random.permutation(key, input_idx)
        selected_inputs = permuted_inputs[:num_connections_per_output]
        return selected_inputs

    conn_keys = jax.random.split(k2, num=len(output_idx))
    connections = jax.vmap(create_connections_for_output)(conn_keys)
    connections = connections.flatten()

    output_repeats = jnp.repeat(output_idx, num_connections_per_output)

    rand_keys_c = jax.random.split(k3, num=total_connections)
    conns_attr_func = jax.vmap(genome.conn_gene.new_random_attrs, in_axes=(0))
    conns_attrs = conns_attr_func(rand_keys_c)

    conns = conns.at[:total_connections, 0].set(connections)
    conns = conns.at[:total_connections, 1].set(output_repeats)
    conns = conns.at[:total_connections, 2].set(True)  # enabled
    conns = conns.at[:total_connections, 3:].set(conns_attrs)

    return nodes, conns
