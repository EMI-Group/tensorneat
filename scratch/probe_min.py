"""Scratch probe: minimal evoxc graph on iree/rocm."""
import sys, time
sys.path.insert(0, "/root/evoxc")
import numpy as np
import evoxc

@evoxc.defn
def kern(a, b):
    return {"c": evoxc.reduce_sum(a * b, axes=(1,)), "d": evoxc.sqrt(evoxc.reduce_sum(a, axes=(1,)) + 1.0)}

rng = np.random.default_rng(0)
a = np.asarray(rng.standard_normal((512, 512)), dtype=np.float32)
b = np.asarray(rng.standard_normal((512, 512)), dtype=np.float32)
for tb in ("llvm-cpu", "rocm"):
    dev = "cpu" if tb == "llvm-cpu" else "rocm"
    opts = {} if tb == "llvm-cpu" else {"target_backends": ["rocm"]}
    t0 = time.perf_counter()
    ex = evoxc.build(kern, evoxc.TensorSpec((512, 512), evoxc.float32),
                     evoxc.TensorSpec((512, 512), evoxc.float32),
                     backend="iree", device=dev, **opts)
    t1 = time.perf_counter()
    d = evoxc.Device(dev, 0)
    ta = evoxc.Tensor(a).to(d) if dev != "cpu" else evoxc.Tensor(a)
    tb_ = evoxc.Tensor(b).to(d) if dev != "cpu" else evoxc.Tensor(b)
    out = evoxc.run(ex, ta, tb_)
    t2 = time.perf_counter()
    vals = [np.asarray(o.to(evoxc.Device("cpu", 0)).numpy() if hasattr(o, "to") else o.numpy()).ravel()[:2] for o in [out["c"], out["d"]]]
    print(tb, f"compile={t1-t0:.2f}s run={t2-t1:.3f}s", vals)
