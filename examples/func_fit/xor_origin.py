from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, problem
from tensorneat.genome import OriginNode, OriginConn
from tensorneat.common import ACT

"""
Solving XOR-3d problem with OriginGene
See https://github.com/EMI-Group/tensorneat/issues/11
"""

algorithm = algorithm.NEAT(
    pop_size=10000,
    species_size=20,
    survival_threshold=0.01,
    genome=genome.DefaultGenome(
        node_gene=OriginNode(),
        conn_gene=OriginConn(),
        num_inputs=3,
        num_outputs=1,
        max_nodes=7,
        output_transform=ACT.sigmoid,
    ),
)
problem = problem.XOR3d()

pipeline = Pipeline(
    algorithm,
    problem,
    generation_limit=200,
    fitness_target=-1e-6,
    seed=42,
)
state = pipeline.setup()
# run until terminate
state, best = pipeline.auto_run(state)
# show result
pipeline.show(state, best)

# visualize the best individual
network = algorithm.genome.network_dict(state, *best)
print(algorithm.genome.repr(state, *best))
# algorithm.genome.visualize(network, save_path="./imgs/xor_network.svg")

# transform the best individual to latex formula
from tensorneat.common.sympy_tools import to_latex_code, to_python_code

sympy_res = algorithm.genome.sympy_func(
    state, network, sympy_output_transform=ACT.obtain_sympy(ACT.sigmoid)
)
latex_code = to_latex_code(*sympy_res)
print(latex_code)

# transform the best individual to python code
python_code = to_python_code(*sympy_res)
print(python_code)
