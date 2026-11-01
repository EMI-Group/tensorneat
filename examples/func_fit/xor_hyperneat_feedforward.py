"""Feedforward HyperNEAT on 3-bit parity; run with
`python examples/func_fit/xor_hyperneat_feedforward.py`."""
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.algorithm.hyperneat import HyperNEATFeedForward, MLPSubstrate
from tensorneat.common import ACT

if __name__ == "__main__":
    # The CPPN maps substrate endpoint coordinates (2 coords -> 4 inputs) to a weight.
    neat = algorithm.NEAT(
        # Demo budget: a real run uses pop_size in the thousands and a few hundred generations.
        pop_size=300,
        species_size=10,
        survival_threshold=0.1,
        genome=genome.DefaultGenome(
            num_inputs=4,
            num_outputs=1,
            output_transform=ACT.tanh,
        ),
    )

    # layers[0] counts the bias neuron: 3 inputs + bias -> 6 -> 6 -> 1 output.
    # A feedforward substrate must be paired with HyperNEATFeedForward (no activate_time).
    substrate = MLPSubstrate(layers=[4, 6, 6, 1], coor_range=(-5.0, 5.0, -5.0, 5.0))

    pipeline = Pipeline(
        algorithm=HyperNEATFeedForward(
            substrate=substrate,
            neat=neat,
            activation=ACT.tanh,
            output_transform=ACT.sigmoid,
        ),
        problem=problem.XOR3d(),
        seed=42,
        generation_limit=40,
        fitness_target=-1e-4,
    )

    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    pipeline.show(state, best)
