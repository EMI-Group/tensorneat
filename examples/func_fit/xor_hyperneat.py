"""Recurrent HyperNEAT on 3-bit parity; run with `python examples/func_fit/xor_hyperneat.py`."""
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.algorithm.hyperneat import FullSubstrate, HyperNEAT
from tensorneat.common import ACT

if __name__ == "__main__":
    # The CPPN is queried with the coordinates of a substrate connection's two
    # endpoints (2 coords -> 4 inputs) and answers with that connection's weight.
    neat = algorithm.NEAT(
        pop_size=1000,
        species_size=10,
        survival_threshold=0.1,
        genome=genome.DefaultGenome(
            num_inputs=4,
            num_outputs=1,
            output_transform=ACT.tanh,
        ),
    )

    pipeline = Pipeline(
        algorithm=HyperNEAT(
            substrate=FullSubstrate(
                input_coors=((-1, -1), (-0.33, -1), (0.33, -1), (1, -1)),
                hidden_coors=((-1, 0), (0, 0), (1, 0)),
                output_coors=((0, 1),),
            ),
            neat=neat,
            activation=ACT.tanh,
            # The substrate is recurrent, so the phenotype is tick-unrolled 10 times.
            activate_time=10,
            output_transform=ACT.sigmoid,
        ),
        problem=problem.XOR3d(),
        seed=42,
        generation_limit=150,
        fitness_target=-1e-4,
    )

    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    pipeline.show(state, best)
