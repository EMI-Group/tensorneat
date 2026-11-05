"""Draw one genome's network as an SVG (host only)."""
from pathlib import Path

import evoxc

from tensorneat import genome
from tensorneat.common import ACT, AGG

if __name__ == "__main__":
    g = genome.DefaultGenome(
        num_inputs=3,
        num_outputs=2,
        max_nodes=10,
        max_conns=20,
        init_hidden_layers=(3,),
        output_transform=ACT.sigmoid,
        node_gene=genome.BiasNode(
            activation_options=ACT.tanh, aggregation_options=AGG.sum
        ),
    )
    state = g.setup()

    # initialize() is a trace-time kernel, so one host graph call draws one individual.
    exe = evoxc.build(
        lambda key: g.initialize(state, key, 1),
        evoxc.TensorSpec((), evoxc.int64),
        backend="numpy",
    )
    payload = evoxc.run(exe, evoxc.random.key(42))

    network = g.network_dict(state, payload)
    print(g.repr(state, payload))

    save_path = Path(__file__).with_name("network.svg")
    g.visualize(network, save_path=str(save_path))
    print(f"saved network figure to {save_path}")
