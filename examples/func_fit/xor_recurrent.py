from config import *
from pipeline import Pipeline
from algorithm import NEAT
from algorithm.neat.gene import RecurrentGene, RecurrentGeneConfig
from problem.func_fit import XOR3d, FuncFitConfig


if __name__ == '__main__':
    config = Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=-1e-2,
            generation_limit=300,
            pop_size=1000
        ),
        neat=NeatConfig(
            network_type="recurrent",
            max_nodes=50,
            max_conns=100,
            max_species=30,
            conn_add=0.5,
            conn_delete=0.5,
            node_add=0.4,
            node_delete=0.4,
            inputs=3,
            outputs=1
        ),
        gene=RecurrentGeneConfig(
            activate_times=10
        ),
        problem=FuncFitConfig(
            error_method='rmse'
        )
    )

    algorithm = NEAT(config, RecurrentGene)
    pipeline = Pipeline(config, algorithm, XOR3d)
    state = pipeline.setup()
    pipeline.pre_compile(state)
    state, best = pipeline.auto_run(state)
    pipeline.show(state, best)
