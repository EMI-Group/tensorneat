<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./imgs/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./imgs/evox_logo_light.png">
    <img alt="EvoX Logo" height="50" src="./imgs/evox_logo_light.png">
  </picture>
  <br>
</h1>

# TensorNEAT: Tensorized NEAT implementation in JAX

<p align="center">
  <a href="https://arxiv.org/">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="TensorRVEA Paper on arXiv">
  </a>
</p>

## Introduction
🚀TensorNEAT, a part of EvoX project, aims to enhance the NEAT (NeuroEvolution of Augmenting Topologies) algorithm by incorporating GPU acceleration. Utilizing JAX for parallel computations, it extends NEAT's capabilities to modern computational environments, making advanced neuroevolution accessible and fast.

## Requirements
TensorNEAT requires:
- jax (version >= 0.4.16)
- jaxlib (version >= 0.3.0)
- brax [optional]
- gymnax [optional]
  
## Example
Simple Example for XOR problem:
```python
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
```

Simple Example for RL envs in Brax (Ant):
```python
from pipeline import Pipeline
from algorithm.neat import *

from problem.rl_env import BraxEnv
from utils import Act

if __name__ == '__main__':
    pipeline = Pipeline(
        algorithm=NEAT(
            species=DefaultSpecies(
                genome=DefaultGenome(
                    num_inputs=27,
                    num_outputs=8,
                    max_nodes=50,
                    max_conns=100,
                    node_gene=DefaultNodeGene(
                        activation_options=(Act.tanh,),
                        activation_default=Act.tanh,
                    )
                ),
                pop_size=1000,
                species_size=10,
            ),
        ),
        problem=BraxEnv(
            env_name='ant',
        ),
        generation_limit=10000,
        fitness_target=5000
    )

    # initialize state
    state = pipeline.setup()
    # print(state)
    # run until terminate
    state, best = pipeline.auto_run(state)
```

more examples are in `tensorneat/examples`.

