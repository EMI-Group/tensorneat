from pipeline import Pipeline
from algorithm.neat import *

from problem.func_fit import XOR3d

if __name__ == '__main__':
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=3,
                    num_outputs=1,
                    max_nodes=50,
                    max_conns=100,
                ),
                pop_size=10000,
                species_size=10,
                compatibility_threshold=3.5,
            ),
        ),
        problem=XOR3d(),
        generation_limit=10000,
        fitness_target=-1e-8
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)
