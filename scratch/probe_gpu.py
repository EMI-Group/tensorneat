"""Scratch probe: compile+run the xor_p500 generation graph on iree/rocm."""
import sys, time
sys.path.insert(0, "src")
sys.path.insert(0, "benchmarks")
import numpy as np
import evoxc
from tensorneat import Pipeline
from cases import get_cases

POP, GENS = 500, 5
case = get_cases([f"xor_p{POP}"])[0]
import tensorneat
algo, prob = case.build(tensorneat, "new")
p = Pipeline(algo, prob, seed=42, generation_limit=1, backend="numpy",
             is_detailed=False, fitness_target=-1e9)
state = p.setup()
dev = evoxc.Device("rocm", 0)
p.backend = "iree"; p.device = "rocm"
state = state.update(backend="iree", device="rocm",
                     pop=tuple(t.to(dev) for t in state.pop),
                     randkey=state.randkey.to(dev),
                     generation=state.generation.to(dev),
                     species=evoxc.tree_map(lambda t: t.to(dev), state.species))
t1 = time.perf_counter()
try:
    state, best, fit = p.step(state)
except Exception as e:
    import traceback; traceback.print_exc(); sys.exit(1)
t2 = time.perf_counter()
print(f"build+first step {t2-t1:.2f}s")
for _ in range(GENS):
    state, best, fit = p.step(state)
t3 = time.perf_counter()
print(f"ms/gen={(t3-t2)/GENS*1000:.2f}")
