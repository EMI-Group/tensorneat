from config import *
from pipeline_jitable_env import Pipeline
from algorithm.neat import NormalGene, NormalGeneConfig
from algorithm.hyperneat import HyperNEAT, NormalSubstrate, NormalSubstrateConfig
from problem.func_fit import XOR3d, FuncFitConfig
from utils import Act


if __name__ == '__main__':
    config = Config(
        basic=BasicConfig(
            seed=42,
            fitness_target=0,
            pop_size=1000
        ),
        neat=NeatConfig(
            max_nodes=50,
            max_conns=100,
            max_species=30,
            inputs=4,
            outputs=1
        ),
        hyperneat=HyperNeatConfig(
            inputs=3,
            outputs=1
        ),
        substrate=NormalSubstrateConfig(
            input_coors=((-1, -1), (-0.5, -1), (0.5, -1), (1, -1)),
        ),
        gene=NormalGeneConfig(
            activation_default=Act.tanh,
            activation_options=(Act.tanh, ),
        ),
        problem=FuncFitConfig()
    )

    algorithm = HyperNEAT(config, NormalGene, NormalSubstrate)
    pipeline = Pipeline(config, algorithm, XOR3d)
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    pipeline.show(state, best)
