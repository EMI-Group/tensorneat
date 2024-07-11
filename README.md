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

## Key Features
- JAX-based network for neuroevolution:
    - **Batch inference** across networks with different architectures, GPU-accelerated.
    - Evolve networks with **irregular structures** and **fully customize** their behavior.
    - Visualize the network and represent it in **mathematical formulas**.

- GPU-accelerated NEAT implementation:
    - Run NEAT and HyperNEAT on GPUs.
    - Achieve **500x** speedup compared to CPU-based NEAT libraries.

- Rich in extended content:
    - Compatible with **EvoX** for multi-device and distributed support.
    - Test neuroevolution algorithms on advanced **RL tasks** (Brax, Gymnax).

## Basic API Usage
Start your journey with TensorNEAT in a few simple steps:

1. **Import necessary modules**:
```python
from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, problem, common
```

2. **Configure the NEAT algorithm and define a problem**:
```python
algorithm = algorithm.NEAT(
    pop_size=10000,
    species_size=20,
    survival_threshold=0.01,
    genome=genome.DefaultGenome(
        num_inputs=3,
        num_outputs=1,
        output_transform=common.ACT.sigmoid,
    ),
)
problem = problem.XOR3d()
```

3. **Initialize the pipeline and run**:
```python
pipeline = Pipeline(
    algorithm,
    problem,
    generation_limit=200,
    fitness_target=-1e-6,
    seed=42,
)
state = pipeline.setup()
# run until termination
state, best = pipeline.auto_run(state)
# show results
pipeline.show(state, best)
```

## Installation
Install `tensorneat` from the GitHub source code:
```
pip install git+https://github.com/EMI-Group/tensorneat.git
```


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
