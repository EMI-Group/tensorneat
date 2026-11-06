"""Probe: drive evoxc's xla adapter against a PJRT C API plugin (scratch)."""
import sys
import numpy as np
import evoxc


@evoxc.defn
def elem_reduce(a, b):
    return evoxc.sum(a * b + 1.0, axes=1)


@evoxc.defn
def loop_graph(x, i0):
    def cond(c):
        i, acc = c
        return i < 7

    def body(c):
        i, acc = c
        return i + 1, acc + x * 0.5

    return evoxc.while_loop(cond, body, (i0, x * 0.0))


def _host(v):
    for attr in ("numpy", "to_host"):
        if hasattr(v, attr):
            try:
                return np.asarray(getattr(v, attr)())
            except TypeError:
                pass
    if hasattr(v, "data"):
        return np.asarray(v.data)
    return np.asarray(v)


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "numpy"
    device = sys.argv[2] if len(sys.argv) > 2 else None
    plugin = sys.argv[3] if len(sys.argv) > 3 else None
    opts = {}
    if plugin:
        opts["plugin_path"] = plugin
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(8, 16)).astype(np.float32)
    x = rng.normal(size=(4,)).astype(np.float32)

    for name, fn, args in (("elem_reduce", elem_reduce, (a, b)), ("loop_graph", loop_graph, (x, np.array(0.0, np.float32)))):
        ref = evoxc.evaluate(fn, *args, backend="numpy")
        got = evoxc.evaluate(fn, *args, backend=backend, device=device, **opts)
        refs = [_host(r) for r in (ref if isinstance(ref, tuple) else (ref,))]
        gots = [_host(g) for g in (got if isinstance(got, tuple) else (got,))]
        ok = all(np.allclose(r, g, rtol=1e-4, atol=1e-4) for r, g in zip(refs, gots))
        print(f"{name}: match={ok}")
        for r in refs:
            print("  ref", r.ravel()[:4])
        for g in gots:
            print("  got", g.ravel()[:4])


if __name__ == "__main__":
    main()
