import jax, jax.numpy as jnp
from tensorneat.genome.operations import (
    DefaultMutation,
    DefaultDistance,
    DefaultCrossover,
)
from tensorneat.genome import (
    DefaultGenome,
    DefaultNode,
    DefaultConn,
    OriginNode,
    OriginConn,
)
from tensorneat.genome.utils import add_node, add_conn

origin_genome = DefaultGenome(
    node_gene=OriginNode(response_init_std=1),
    conn_gene=OriginConn(),
    mutation=DefaultMutation(conn_add=1, node_add=1, conn_delete=0, node_delete=0),
    crossover=DefaultCrossover(),
    distance=DefaultDistance(),
    num_inputs=3,
    num_outputs=1,
    max_nodes=6,
    max_conns=6,
)

default_genome = DefaultGenome(
    node_gene=DefaultNode(response_init_std=1),
    conn_gene=DefaultConn(),
    mutation=DefaultMutation(conn_add=1, node_add=1, conn_delete=0, node_delete=0),
    crossover=DefaultCrossover(),
    distance=DefaultDistance(),
    num_inputs=3,
    num_outputs=1,
    max_nodes=6,
    max_conns=6,
)

state = default_genome.setup()
state = origin_genome.setup(state)

randkey = jax.random.PRNGKey(42)


def mutation_default():
    nodes, conns = default_genome.initialize(state, randkey)
    print("old genome:\n", default_genome.repr(state, nodes, conns))

    nodes, conns = default_genome.execute_mutation(
        state,
        randkey,
        nodes,
        conns,
        new_node_key=jnp.asarray(10),
        new_conn_keys=jnp.array([20, 21, 22]),
    )

    # new_conn_keys is not used in default genome
    print("new genome:\n", default_genome.repr(state, nodes, conns))


def mutation_origin():
    nodes, conns = origin_genome.initialize(state, randkey)
    print(conns)
    print("old genome:\n", origin_genome.repr(state, nodes, conns))

    nodes, conns = origin_genome.execute_mutation(
        state,
        randkey,
        nodes,
        conns,
        new_node_key=jnp.asarray(10),
        new_conn_keys=jnp.array([20, 21, 22]),
    )
    print(conns)
    # new_conn_keys is used in origin genome
    print("new genome:\n", origin_genome.repr(state, nodes, conns))

def distance_default():
    nodes, conns = default_genome.initialize(state, randkey)
    nodes = add_node(
        nodes, 
        fix_attrs=jnp.asarray([10]),
        custom_attrs=default_genome.node_gene.new_identity_attrs(state)
    )
    conns1 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10]),  # in-idx, out-idx
        custom_attrs=default_genome.conn_gene.new_zero_attrs(state)
    )
    conns2 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10]),  # in-idx, out-idx
        custom_attrs=default_genome.conn_gene.new_random_attrs(state, randkey)
    )
    print("genome1:\n", default_genome.repr(state, nodes, conns1))
    print("genome2:\n", default_genome.repr(state, nodes, conns2))

    distance = default_genome.execute_distance(state, nodes, conns1, nodes, conns2)
    print("distance: ", distance)

def distance_origin_case1():
    """
    distance with different historical marker
    """
    nodes, conns = origin_genome.initialize(state, randkey)
    nodes = add_node(
        nodes, 
        fix_attrs=jnp.asarray([10]),
        custom_attrs=origin_genome.node_gene.new_identity_attrs(state)
    )
    conns1 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10, 99]),  # in-idx, out-idx, historical mark
        custom_attrs=origin_genome.conn_gene.new_zero_attrs(state)
    )
    conns2 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10, 88]),  # in-idx, out-idx, historical mark
        custom_attrs=origin_genome.conn_gene.new_random_attrs(state, randkey)
    )
    print("genome1:\n", origin_genome.repr(state, nodes, conns1))
    print("genome2:\n", origin_genome.repr(state, nodes, conns2))

    distance = origin_genome.execute_distance(state, nodes, conns1, nodes, conns2)
    print("distance: ", distance)

def distance_origin_case2():
    """
    distance with same historical marker
    """
    nodes, conns = origin_genome.initialize(state, randkey)
    nodes = add_node(
        nodes, 
        fix_attrs=jnp.asarray([10]),
        custom_attrs=origin_genome.node_gene.new_identity_attrs(state)
    )
    conns1 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10, 99]),  # in-idx, out-idx, historical mark
        custom_attrs=origin_genome.conn_gene.new_zero_attrs(state)
    )
    conns2 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10, 99]),  # in-idx, out-idx, historical mark
        custom_attrs=origin_genome.conn_gene.new_random_attrs(state, randkey)
    )
    print("genome1:\n", origin_genome.repr(state, nodes, conns1))
    print("genome2:\n", origin_genome.repr(state, nodes, conns2))

    distance = origin_genome.execute_distance(state, nodes, conns1, nodes, conns2)
    print("distance: ", distance)

def crossover_origin_case1():
    """
    crossover with different historical marker
    """
    nodes, conns = origin_genome.initialize(state, randkey)
    nodes = add_node(
        nodes, 
        fix_attrs=jnp.asarray([10]),
        custom_attrs=origin_genome.node_gene.new_identity_attrs(state)
    )
    conns1 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10, 99]),  # in-idx, out-idx, historical mark
        custom_attrs=origin_genome.conn_gene.new_zero_attrs(state)
    )
    conns2 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10, 88]),  # in-idx, out-idx, historical mark
        custom_attrs=origin_genome.conn_gene.new_random_attrs(state, randkey)
    )
    print("genome1:\n", origin_genome.repr(state, nodes, conns1))
    print("genome2:\n", origin_genome.repr(state, nodes, conns2))

    # (0, 10)'s weight must be 0 (disjoint gene, use fitter)
    child_nodes, child_conns = origin_genome.execute_crossover(state, randkey, nodes, conns1, nodes, conns2)
    print("child:\n", origin_genome.repr(state, child_nodes, child_conns))

def crossover_origin_case2():
    """
    crossover with same historical marker
    """
    nodes, conns = origin_genome.initialize(state, randkey)
    nodes = add_node(
        nodes, 
        fix_attrs=jnp.asarray([10]),
        custom_attrs=origin_genome.node_gene.new_identity_attrs(state)
    )
    conns1 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10, 99]),  # in-idx, out-idx, historical mark
        custom_attrs=origin_genome.conn_gene.new_zero_attrs(state)
    )
    conns2 = add_conn(
        conns,
        fix_attrs=jnp.array([0, 10, 99]),  # in-idx, out-idx, historical mark
        custom_attrs=origin_genome.conn_gene.new_random_attrs(state, randkey)
    )
    print("genome1:\n", origin_genome.repr(state, nodes, conns1))
    print("genome2:\n", origin_genome.repr(state, nodes, conns2))

    # (0, 10)'s weight might be random or zero (homologous gene)

    # zero case:
    child_nodes, child_conns = origin_genome.execute_crossover(state, jax.random.key(99), nodes, conns1, nodes, conns2)
    print("child_zero:\n", origin_genome.repr(state, child_nodes, child_conns))

    # random case:
    child_nodes, child_conns = origin_genome.execute_crossover(state, jax.random.key(0), nodes, conns1, nodes, conns2)
    print("child_random:\n", origin_genome.repr(state, child_nodes, child_conns))

def crossover_origin_case3():
    """
    test examine it use random gene rather than attribute exchange
    """
    nodes, conns = origin_genome.initialize(state, randkey)
    nodes1 = add_node(
        nodes, 
        fix_attrs=jnp.asarray([10]),
        custom_attrs=jnp.array([1, 2, 0, 0])
    )
    nodes2 = add_node(
        nodes, 
        fix_attrs=jnp.asarray([10]),
        custom_attrs=jnp.array([100, 200, 0, 0])
    )

    # [1, 2] case
    child_nodes, child_conns = origin_genome.execute_crossover(state, jax.random.key(99), nodes1, conns, nodes2, conns)
    print("child1:\n", origin_genome.repr(state, child_nodes, child_conns))

    # [100, 200] case
    child_nodes, child_conns = origin_genome.execute_crossover(state, jax.random.key(1), nodes1, conns, nodes2, conns)
    print("child2:\n", origin_genome.repr(state, child_nodes, child_conns))


if __name__ == "__main__":
    # mutation_origin()
    # distance_default()
    # distance_origin_case1()
    # distance_origin_case2()
    # crossover_origin_case1()
    # crossover_origin_case2()
    crossover_origin_case3()
