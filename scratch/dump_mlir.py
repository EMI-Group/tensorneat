import sys
from tensorneat import Pipeline, algorithm, genome, problem

env = sys.argv[1]
out = sys.argv[2]
if env == "cartpole":
    prob = problem.CartPole(max_step=200, repeat_times=1)
    g = genome.DefaultGenome(num_inputs=4, num_outputs=2, max_nodes=50, max_conns=100)
elif env == "pendulum":
    prob = problem.Pendulum(max_step=200, repeat_times=1)
    g = genome.DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100)
elif env == "mountaincar":
    prob = problem.MountainCarContinuous(max_step=999, repeat_times=1)
    g = genome.DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=50, max_conns=100)
alg = algorithm.NEAT(genome=g, pop_size=100, species_size=20, survival_threshold=0.01)
pipe = Pipeline(algorithm=alg, problem=prob, seed=42, fitness_target=1e9,
                generation_limit=1, backend="iree", is_detailed=False)
state = pipe.setup()
pipe.step(state)
exe = list(pipe._executables.values())[0]
mlir = exe.backend_executable.artifact.payload["mlir_text"]
open(out, "w").write(mlir)
print("wrote", out, len(mlir), "bytes")
