"""The single host-side bridge between pytest and ``evoxc``'s build/run pipeline.

Every test in this suite that executes a kernel goes through this module, for
two reasons:

* evoxc's contracts are easy to get subtly wrong — ``build`` takes *positional*
  ``TensorSpec``/static arguments mirroring the traced call, ``run`` must be
  handed the static values **again**, an ``Executable`` is not callable, and
  ``evoxc.Tensor`` has no ``__array__`` so ``np.asarray`` silently leaves it
  alone. Centralising the recipe means a test reads ``run_defn(kernel, ...)``
  and nothing else.
* ``genome.grad`` returns an ``evoxc.Graph`` (a graph *factory* result), which
  needs ``lower`` → ``compile`` → ``load`` instead of ``build``. That path is
  :func:`run_graph`.

Verified behaviours this module encodes (evoxc numpy reference backend):

* ``evoxc.build(fn, *args, backend="numpy")`` where a numpy array argument
  stands for ``TensorSpec(shape, dtype)`` and a plain Python leaf is a static
  value baked into the trace.
* ``evoxc.run(exe, *args)`` — the static leaves must be passed again with the
  same values, otherwise ``evoxc.TraceError``.
* ``evoxc.lower(graph, backend=...)`` → ``evoxc.compile(...)`` →
  ``evoxc.load(...)`` for a pre-traced ``Graph``.
* numpy dtypes map to ``evoxc.float32``/``int32``/``int64``/``bool_``
  (note: ``evoxc.bool_``, there is no ``evoxc.bool``).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

import evoxc

__all__ = [
    "DEFAULT_BACKEND",
    "build_defn",
    "compile_graph",
    "run_defn",
    "run_graph",
    "spec_like",
    "tensor",
    "to_numpy",
]

#: The backend every non-``requires_iree`` test runs on (the reference interpreter).
DEFAULT_BACKEND = "numpy"

#: numpy dtype -> evoxc dtype. Extend only when a payload actually needs it.
_DTYPES = {
    np.dtype("float32"): evoxc.float32,
    np.dtype("float64"): evoxc.float64,
    np.dtype("int32"): evoxc.int32,
    np.dtype("int64"): evoxc.int64,
    np.dtype("bool"): evoxc.bool_,
}


def tensor(value: Any) -> evoxc.Tensor:
    """Wrap a numpy array (or pass an ``evoxc.Tensor`` through) for ``evoxc.run``."""
    if isinstance(value, evoxc.Tensor):
        return value
    return evoxc.from_numpy(np.ascontiguousarray(value))


def spec_like(value: Any) -> evoxc.TensorSpec:
    """Build the ``TensorSpec`` of a concrete tensor, so a call site can just pass data.

    Accepts a numpy array or an already-wrapped ``evoxc.Tensor`` — the latter
    matters for RNG keys, whose rank-0 ``int64`` dtype cannot be recovered by
    ``np.asarray`` on a ``Tensor`` (it has no ``__array__``).
    """
    if isinstance(value, evoxc.Tensor):
        return evoxc.TensorSpec(tuple(value.shape), value.dtype)
    array = np.asarray(value)
    if array.dtype not in _DTYPES:  # a wrong dtype is a test bug, not a backend error
        raise TypeError(
            f"no evoxc dtype for numpy dtype {array.dtype}; extend support.runner._DTYPES"
        )
    # Derive the spec from the *wrapped* tensor: ``evoxc.from_numpy`` reports a
    # 0-d array as shape ``(1,)``, and the spec must agree with what ``run``
    # receives or evoxc raises a rank mismatch.
    wrapped = evoxc.from_numpy(np.ascontiguousarray(array))
    return evoxc.TensorSpec(tuple(wrapped.shape), wrapped.dtype)


def _is_data(value: Any) -> bool:
    """True for leaves that cross the graph boundary as tensors (vs. static values)."""
    return isinstance(value, (np.ndarray, evoxc.Tensor, np.generic))


def _split_arg(value: Any) -> tuple[Any, Any]:
    """Split one argument into ``(build arg, run arg)``, recursing into pytrees.

    A pytree of arrays (a ``Transformed`` namedtuple, a tuple of payload
    arrays) must be treated as data leaf by leaf: a container that holds at
    least one array is traversed, so the spec tree mirrors the runtime tree
    exactly — which is what ``evoxc.build``/``evoxc.run`` compare against.
    """
    if _is_data(value):
        return spec_like(value), tensor(value)
    if isinstance(value, tuple):
        pairs = [_split_arg(item) for item in value]
        builds = tuple(build for build, _ in pairs)
        runs = tuple(run for _, run in pairs)
        if type(value) is tuple:  # a plain tuple is not rebuilt by calling its type
            return builds, runs
        return type(value)(*builds), type(value)(*runs)
    if isinstance(value, list):
        pairs = [_split_arg(item) for item in value]
        return [build for build, _ in pairs], [run for _, run in pairs]
    if isinstance(value, dict):
        build, run = {}, {}
        for key, item in value.items():
            build[key], run[key] = _split_arg(item)
        return build, run
    return value, value


def _split_args(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Split one call's arguments into ``(build args, run args)``.

    Data leaves become a ``TensorSpec`` for ``build`` and the concrete tensor
    for ``run``; static leaves are passed through unchanged to both.
    """
    pairs = [_split_arg(argument) for argument in args]
    return tuple(build for build, _ in pairs), tuple(run for _, run in pairs)


def build_defn(
    fn: Callable[..., Any],
    *args: Any,
    backend: str = DEFAULT_BACKEND,
    **options: Any,
) -> evoxc.Executable:
    """Trace/compile ``fn`` on the arguments of one call and return the executable.

    Args:
        fn: A ``@evoxc.defn`` function, or any plain callable that closes over
            its host-side configuration (state, genome, problem) and takes only
            tensor/static arguments — a closure is how a test runs a *method*
            inside a graph without nesting two ``@evoxc.defn`` functions.
        *args: The arguments of one call. A ``numpy.ndarray`` leaf becomes a
            ``TensorSpec``; anything else (``int``/``float``/``bool``/``str``/
            ``None``) is a static value that is baked into the trace.
        backend: ``evoxc`` backend name; ``"numpy"`` is the reference interpreter.
        **options: Extra ``evoxc.build`` options (e.g. ``device``).

    Returns:
        A reusable ``evoxc.Executable``: call it through :func:`evoxc.run` with
        the same structure and the same static values, and any data values of
        the same shapes/dtypes.

    Raises:
        evoxc.TraceError, evoxc.ShapeError, evoxc.DTypeError: Propagated
            unchanged — a test must never launder an evoxc error.
    """
    build_args, _ = _split_args(args)
    return evoxc.build(fn, *build_args, backend=backend, **options)


def run_defn(
    fn: Callable[..., Any],
    *args: Any,
    backend: str = DEFAULT_BACKEND,
    **options: Any,
) -> Any:
    """Build and run ``fn`` on one call, returning host numpy leaves.

    Args:
        fn: As in :func:`build_defn`.
        *args: As in :func:`build_defn`.
        backend: As in :func:`build_defn`.
        **options: As in :func:`build_defn`.

    Returns:
        The graph output with every ``evoxc.Tensor`` leaf replaced by its
        ``numpy`` view; tuples/lists/dicts/namedtuples are rebuilt in place, so
        a ``Transformed`` comes back as a ``Transformed`` of numpy arrays.
    """
    build_args, run_args = _split_args(args)
    executable = evoxc.build(fn, *build_args, backend=backend, **options)
    return to_numpy(evoxc.run(executable, *run_args))


def compile_graph(graph: evoxc.Graph, backend: str = DEFAULT_BACKEND) -> evoxc.Executable:
    """Lower → compile → load an already traced ``evoxc.Graph`` (what ``genome.grad`` returns)."""
    return evoxc.load(evoxc.compile(evoxc.lower(graph, backend=backend)))


def run_graph(
    graph: evoxc.Graph,
    *arrays: Any,
    backend: str = DEFAULT_BACKEND,
) -> Any:
    """Run a traced graph (e.g. from ``genome.grad``) on numpy inputs.

    Unlike :func:`run_defn` this takes only data: a graph returned by
    ``evoxc.grad`` has its static values already specialised, and its inputs are
    exactly the differentiated/derived tensor arguments.

    Args:
        graph: A lowered-but-not-compiled ``evoxc.Graph``.
        *arrays: Concrete inputs, one per graph input, in order.
        backend: As in :func:`build_defn`.

    Returns:
        Host numpy leaves, as :func:`run_defn`.
    """
    executable = compile_graph(graph, backend=backend)
    return to_numpy(evoxc.run(executable, *(tensor(a) for a in arrays)))


def to_numpy(value: Any) -> Any:
    """Recursively replace ``evoxc.Tensor`` leaves with their numpy view.

    Tuples, lists, dicts and namedtuples are rebuilt (a namedtuple keeps its
    type, which is what makes ``Transformed`` readable in an assertion);
    anything else is returned unchanged.
    """
    if isinstance(value, evoxc.Tensor):
        return value.numpy()
    if isinstance(value, tuple):
        items = [to_numpy(item) for item in value]
        if type(value) is tuple:  # a plain tuple is not rebuilt by calling its type
            return tuple(items)
        return type(value)(*items)
    if isinstance(value, list):
        return [to_numpy(item) for item in value]
    if isinstance(value, dict):
        return {key: to_numpy(item) for key, item in value.items()}
    return value
