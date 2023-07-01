import pickle

import jax
from jax import numpy as jnp, jit, vmap

import numpy as np

from configs import Configer
from algorithms.neat import initialize_genomes
from algorithms.neat import tell
from algorithms.neat import unflatten_connections, topological_sort, create_forward_function

jax.config.update("jax_disable_jit", True)

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
xor_outputs = np.array([[0], [1], [1], [0]], dtype=np.float32)

def evaluate(forward_func):
    """
    :param forward_func: (4: batch, 2: input size) -> (pop_size, 4: batch, 1: output size)
    :return:
    """
    outs = forward_func(xor_inputs)
    outs = jax.device_get(outs)
    fitnesses = 4 - np.sum((outs - xor_outputs) ** 2, axis=(1, 2))
    return fitnesses


def get_fitnesses(pop_nodes, pop_cons, pop_unflatten_connections, pop_topological_sort, forward_func):
    u_pop_cons = pop_unflatten_connections(pop_nodes, pop_cons)
    pop_seqs = pop_topological_sort(pop_nodes, u_pop_cons)
    func = lambda x: forward_func(x, pop_seqs, pop_nodes, u_pop_cons)

    return evaluate(func)




def equal(ar1, ar2):
    if ar1.shape != ar2.shape:
        return False

    nan_mask1 = jnp.isnan(ar1)
    nan_mask2 = jnp.isnan(ar2)

    return jnp.all((ar1 == ar2) | (nan_mask1 & nan_mask2))

def main():
    # initialize
    config = Configer.load_config("xor.ini")
    jit_config = Configer.create_jit_config(config)  # config used in jit-able functions

    P = config['pop_size']
    N = config['init_maximum_nodes']
    C = config['init_maximum_connections']
    S = config['init_maximum_species']
    randkey = jax.random.PRNGKey(6)
    np.random.seed(6)

    pop_nodes, pop_cons = initialize_genomes(N, C, config)
    species_info = np.full((S, 4), np.nan)
    species_info[0, :] = 0, -np.inf, 0, P
    idx2species = np.zeros(P, dtype=np.float32)
    center_nodes = np.full((S, N, 5), np.nan)
    center_cons = np.full((S, C, 4), np.nan)
    center_nodes[0, :, :] = pop_nodes[0, :, :]
    center_cons[0, :, :] = pop_cons[0, :, :]
    generation = 0

    pop_nodes, pop_cons, species_info, idx2species, center_nodes, center_cons = jax.device_put(
        [pop_nodes, pop_cons, species_info, idx2species, center_nodes, center_cons])

    pop_unflatten_connections = jit(vmap(unflatten_connections))
    pop_topological_sort = jit(vmap(topological_sort))
    forward = create_forward_function(config)


    while True:
        fitness = get_fitnesses(pop_nodes, pop_cons, pop_unflatten_connections, pop_topological_sort, forward)

        last_max = np.max(fitness)

        info = [fitness, randkey, pop_nodes, pop_cons, species_info, idx2species, center_nodes, center_cons, generation,
                jit_config]

        with open('list.pkl', 'wb') as f:
            # 使用pickle模块的dump函数来保存list
            pickle.dump(info, f)

        randkey, pop_nodes, pop_cons, species_info, idx2species, center_nodes, center_cons, generation = tell(
            fitness, randkey, pop_nodes, pop_cons, species_info, idx2species, center_nodes, center_cons, generation,
            jit_config)

        fitness = get_fitnesses(pop_nodes, pop_cons, pop_unflatten_connections, pop_topological_sort, forward)
        current_max = np.max(fitness)
        print(last_max, current_max)
        assert current_max >= last_max, f"current_max: {current_max}, last_max: {last_max}"


if __name__ == '__main__':
    # main()
    config = Configer.load_config("xor.ini")
    pop_unflatten_connections = jit(vmap(unflatten_connections))
    pop_topological_sort = jit(vmap(topological_sort))
    forward = create_forward_function(config)

    with open('list.pkl', 'rb') as f:
        # 使用pickle模块的dump函数来保存list
        fitness, randkey, pop_nodes, pop_cons, species_info, idx2species, center_nodes, center_cons, i, jit_config = pickle.load(
            f)

    print(np.max(fitness))
    randkey, pop_nodes, pop_cons, species_info, idx2species, center_nodes, center_cons, _ = tell(
        fitness, randkey, pop_nodes, pop_cons, species_info, idx2species, center_nodes, center_cons, i,
        jit_config)
    fitness = get_fitnesses(pop_nodes, pop_cons, pop_unflatten_connections, pop_topological_sort, forward)
    print(np.max(fitness))
