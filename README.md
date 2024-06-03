<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./imgs/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./imgs/evox_logo_light.png">
      <img alt="EvoX Logo" height="50" src="./imgs/evox_logo_light.png">
  </picture>
  </a>
  <br>
</h1>

<p align="center">
🌟 TensorNEAT: Tensorized NEAT Implementation in JAX 🌟
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2404.01817">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="TensorNEAT Paper on arXiv">
  </a>
</p>

## Introduction
TensorNEAT is a JAX-based libaray for NeuroEvolution of Augmenting Topologies (NEAT) algorithms, focused on harnessing GPU acceleration to enhance the efficiency of evolving neural network structures for complex tasks. Its core mechanism involves the tensorization of network topologies, enabling parallel processing and significantly boosting computational speed and scalability by leveraging modern hardware accelerators. TensorNEAT is compatible with the [EvoX](https://github.com/EMI-Group/evox/) framewrok.

## Requirements
Due to the rapid iteration of JAX versions, configuring the runtime environment for TensorNEAT can be challenging. We recommend the following versions for the relevant libraries:

- jax (0.4.28)
- jaxlib (0.4.28+cuda12.cudnn89)
- brax (0.10.3)
- gymnax (0.0.8)
  
We provide detailed JAX-related environment references in [recommend_environment](recommend_environment.txt). If you encounter any issues while configuring the environment yourself, you can use this as a reference.

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

## Community & Support

- Engage in discussions and share your experiences on [GitHub Discussion Board](https://github.com/EMI-Group/evox/discussions).
- Join our QQ group (ID: 297969717).
  
## Citing TensorNEAT

If you use TensorNEAT in your research and want to cite it in your work, please use:
```
@article{tensorneat,
  title = {{Tensorized} {NeuroEvolution} of {Augmenting} {Topologies} for {GPU} {Acceleration}},
  author = {Wang, Lishuang and Zhao, Mengfei and Liu, Enyu and Sun, Kebin and Cheng, Ran},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  year = {2024}
}
