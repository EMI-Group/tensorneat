from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import NormalGene, NormalGeneConfig
from problem.func_fit import XOR, FuncFitConfig

def evaluate():
    pass



if __name__ == '__main__':
    config = Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=-1e-2,
            pop_size=10000
        ),
        neat=NeatConfig(
            max_nodes=50,
            max_conns=100,
            max_species=30,
            conn_add=0.8,
            conn_delete=0,
            node_add=0.4,
            node_delete=0,
            inputs=2,
            outputs=1
        ),
        gene=NormalGeneConfig(),
        problem=FuncFitConfig(
            error_method='rmse'
        )
    )

    algorithm = NEAT(config, NormalGene)
    pipeline = Pipeline(config, algorithm, XOR)
    state = pipeline.setup()
    pipeline.pre_compile(state)
    state, best = pipeline.auto_run(state)
    pipeline.show(state, best)
