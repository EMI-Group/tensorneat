"""Fixtures shared by the ``./unit/genome`` suites: five modules and four graph runners.

Two kinds of fixture live here.

**Inputs, resolved at fixture time.** ``tensorneat.genome``, ``tensorneat.common``,
``support.payloads``, ``support.oracle`` and ``support.runner`` are each resolved
through :func:`support.lazy.require` inside the fixture body, never at module
scope: every error — a ``NotImplementedError`` included — propagates unchanged,
because ``require`` converts a pending implementation into a skip only under the
explicit ``allow_pending=True`` keyword and no fixture here passes it.

**Graph glue, one place.** ``support/runner.py`` is the only module allowed to
call ``evoxc.build``/``run``/``lower``/``compile``/``load``
(``../../support/CONTEXT.md``), and it takes *top-level* numpy leaves as data.
That single rule has two consequences these fixtures encode so no test module
re-derives them:

* a method must be executed through a closure —
  ``run_defn(lambda *arrays: obj.method(state, *arrays), ...)`` — because two
  nested ``@evoxc.defn`` functions may not call each other (:func:`run_kernel`);
* a ``Transformed`` schedule is a *namedtuple*, and the runner classifies only
  top-level array leaves as data, so passing the schedule as one argument would
  bake it into the trace as a static value whose fields are plain numpy arrays —
  every ``transformed.in_conn`` inside the kernel would then be an ndarray, not
  a symbolic tensor, and the trace fails with a ``TypeError``. The schedule is
  therefore passed **field by field** and rebuilt inside the traced closure,
  where constructing a namedtuple is a pure-Python operation
  (:func:`run_transform` produces the fields, :func:`run_forward` and
  :func:`run_prepared` consume them); see the report to the parent.

**One executable per graph, not one per call.** Tracing costs about ten times as
much as running, so :func:`run_kernel`, :func:`run_transform`,
:func:`run_forward` and :func:`run_prepared` share one module-scoped
``evoxc.Executable`` cache (``_executable_cache``) and reuse a built executable
for every later call that differs only in *data values*. The key is ``(graph,
*per_argument)``: ``graph`` names the traced closure — ``method.__func__`` plus
the identity of the genome (``method.__self__``) and of the host ``state`` for a
kernel, the genome/state plus the ``Transformed`` type for
:func:`run_forward`/``run_prepared`` — and each argument
contributes ``("data", shape, dtype)`` for a tensor leaf, ``("static", type
name, value)`` for a frozen scalar, ``("tuple", ...)`` for a tuple of those, or
``("object", identity)`` for anything else. **A different static value, a
different genome/state object or a different shape/dtype therefore rebuilds**;
data values, RNG key contents and non-scalar statics are never compared. The key
holds the genome/state objects themselves (through :class:`_Identity`), so an
address is never recycled into a false hit. A call that cannot be classified
with certainty — a mutable container baked into the trace, a callee that is not
a bound method, an unmapped dtype — **bypasses the cache and rebuilds every
time**. ``genome.grad`` graphs are never cached: each call returns a freshly
traced ``evoxc.Graph``, which keeps going through ``runner.run_graph``. Reusing
an executable needs a run-only entry point the runner does not export, so
:func:`_run_one` calls ``evoxc.run`` itself (imported locally) while every build
still goes through ``runner.build_defn``; a ``runner.run_built(executable,
*args)`` would let this cache move into ``support/`` (reported to the parent).

Nothing here imports ``jax``/``torch``, ``evoxc`` or ``numpy`` at module scope:
the fixtures return modules and closures, and the suites import what they need
from them. That keeps the per-test guard in ``../conftest.py`` (no optional
framework may enter ``sys.modules``) honest for every module in this directory.
"""

from __future__ import annotations

import pytest

#: Scalars whose *value* fully determines the trace, keyed by type name + value.
#: The type name is what keeps ``run(..., True)`` from reusing ``run(..., 1)``.
_FROZEN_SCALARS = (bool, int, float, complex, str, bytes, type(None))


class _Identity:
    """A cache-key component keyed on an object's identity that keeps it alive.

    ``__hash__`` is the object's address, which is sound only because this
    wrapper holds a *strong* reference: an entry cannot outlive its referent, so
    an address can never be recycled into a false hit.
    """

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        return isinstance(other, _Identity) and other.obj is self.obj


@pytest.fixture(scope="module")
def genome():
    """The ``tensorneat.genome`` package under test.

    Returns:
        The imported package, so a test reads ``genome.DefaultGenome`` /
        ``genome.RecurrentGenome`` / ``genome.DefaultNode`` /
        ``genome.GenomeTensors`` / ``genome.DefaultMutation`` and the rest of the
        public surface listed in ``tensorneat/genome/__init__.py``.

    Notes:
        The import is deferred (see the module docstring); ``require`` propagates
        every error by default, so a ``NotImplementedError`` raised by the library
        under test surfaces here as a failure, not a skip.
    """
    from support.lazy import require

    return require("tensorneat.genome")


@pytest.fixture(scope="module")
def common():
    """The ``tensorneat.common`` package (``State``, ``I_INF``, ``Transformed``, errors).

    Returns:
        The imported package: ``common.State`` to build the host container the
        genome registers its configuration on, ``common.I_INF`` for the ``int32``
        padding sentinel, ``common.Transformed`` to rebuild a schedule inside a
        trace, ``common.validate_schedule`` for the schedule oracle and
        ``common.TensorNEATError`` for the configuration-error assertions.
    """
    from support.lazy import require

    return require("tensorneat.common")


@pytest.fixture(scope="module")
def payloads():
    """The hand-built gene payloads of ``../../support/payloads.py``.

    Returns:
        The ``support.payloads`` module: ``new_payload``, ``set_node``,
        ``set_conn``, ``chain_payload``, ``skip_payload``, ``isolated_payload``,
        ``ring_payload``, ``default_genome``, ``recurrent_genome``,
        ``genome_state`` and ``I_INF``.

    Notes:
        Every topology here is written by hand, never drawn, so a test asserts a
        value it can print. ``support.payloads`` imports ``tensorneat.common.tools``
        at module scope, so an unfinished import there fails the fixture instead
        of skipping it.
    """
    from support.lazy import require

    return require("support.payloads")


@pytest.fixture(scope="module")
def oracle():
    """The independent numpy reference oracles of ``../../support/oracle.py``.

    Returns:
        The ``support.oracle`` module: ``schedule`` (the reference tensorization
        derivation), ``forward`` (the NaN-aware level-synchronous / tick-based
        rollout), ``default_node_attrs_of``/``default_conn_attr_of`` (the column
        readers a non-default gene layout overrides) and the activation /
        aggregation kernels re-exported from ``support.functions_ref``.

    Notes:
        Written from the documented algorithm, never by calling the kernel it
        checks — otherwise the assertion is a tautology.
    """
    from support.lazy import require

    return require("support.oracle")


@pytest.fixture(scope="module")
def runner():
    """The one sanctioned bridge to ``evoxc`` (``../../support/runner.py``).

    Returns:
        The ``support.runner`` module: ``run_defn`` (build + run one call),
        ``build_defn`` (a reusable executable), ``run_graph`` (lower → compile →
        load → run, for the ``evoxc.Graph`` that ``genome.grad`` returns) and
        ``to_numpy``.
    """
    from support.lazy import require

    return require("support.runner")


@pytest.fixture(scope="module")
def _executable_cache():
    """One built-executable cache per test module, shared by the three runners.

    Returns:
        A ``dict`` mapping a :func:`_cache_key` to an ``evoxc.Executable``,
        dropped (with the genome/state objects its keys keep alive) when the
        module's tests finish; the three runners use disjoint key heads, so
        sharing it cannot make one reuse another's graph.
    """
    return {}


def _static_component(value):
    """Key component of a non-tensor argument, or ``None`` if it cannot be keyed.

    A frozen scalar is keyed by *value* (a changed static forces a rebuild), a
    tuple element-wise, and any other object — a callable, a configuration
    object — by identity, which is what makes "a different genome/state object
    rebuilds" true. A mutable container is not keyed at all: its contents are
    baked into the trace, so an in-place edit after a build could hide behind a
    hit.
    """
    if isinstance(value, _FROZEN_SCALARS):
        return ("static", type(value).__name__, value)
    if isinstance(value, tuple):
        items = tuple(_static_component(item) for item in value)
        if any(item is None for item in items):
            return None
        return ("tuple", items)
    if isinstance(value, (list, dict, set, frozenset, bytearray)):
        return None
    return ("object", _Identity(value))


def _cache_key(runner_module, graph, args):
    """The full cache key of one call, or ``None`` when the call must not be cached.

    Each argument contributes the component documented in the module docstring.
    A tensor leaf is keyed by ``runner.spec_like`` — the very call the build
    performs — so the key can never be *coarser* than the ``TensorSpec`` that
    reaches ``evoxc.build``; a dtype the runner cannot map (``TypeError``) makes
    the call bypass the cache, and ``runner.run_defn`` then raises its own error.
    """
    if graph is None:
        return None
    components = []
    for value in args:
        # The runner's own data/static split, so the key's notion of a leaf is
        # exactly the one ``build``/``run`` will use.
        if not runner_module._is_data(value):
            component = _static_component(value)
        else:
            try:
                spec = runner_module.spec_like(value)
            except TypeError:
                return None
            component = ("data", tuple(spec.shape), str(spec.dtype))
        if component is None:
            return None
        components.append(component)
    return (graph, tuple(components))


def _run_one(runner_module, cache, graph, fn, args):
    """Build ``fn`` once per :func:`_cache_key` and run it on this call's data.

    Args:
        runner_module: The ``support.runner`` module.
        cache: The module-scoped executable cache.
        graph: The key head (``None`` disables caching for this call).
        fn: The traced closure, already closing over genome/state.
        args: The arguments of the call; tensor leaves are data, the rest are
            statics that must be handed to ``evoxc.run`` again.

    Returns:
        Host numpy leaves, as ``runner.run_defn``.

    Notes:
        Running a *built* executable is the one step not expressible on
        ``support.runner``'s public surface (it exposes ``run_defn``, not a
        run-only entry point), so ``evoxc.run`` is imported and called here with
        exactly the arguments ``runner.run_defn`` would have used; every build
        still goes through ``runner.build_defn``. A ``runner.run_built`` would
        remove this last exception and let the cache live in ``test/support/``.
    """
    key = _cache_key(runner_module, graph, args)
    if key is None:
        return runner_module.run_defn(fn, *args)
    executable = cache.get(key)
    if executable is None:
        executable = runner_module.build_defn(fn, *args)
        cache[key] = executable
    import evoxc

    _, run_args = runner_module._split_args(args)
    return runner_module.to_numpy(evoxc.run(executable, *run_args))


@pytest.fixture(scope="module")
def run_kernel(runner, _executable_cache):
    """Execute one genome kernel (a bound method) inside a single traced closure.

    Returns:
        ``run(method, state, *args) -> host numpy``: ``method`` is any genome
        kernel that takes the host ``state`` first and tensors afterwards
        (``genome.forward``, ``genome.execute_mutation``,
        ``genome.execute_crossover``, ``genome.execute_distance``,
        ``genome.transform`` ...), ``state`` is static configuration, ``*args``
        are the tensors/statics of the call in the method's own order.

    Notes:
        A *method* cannot be traced directly: it is not an ``@evoxc.defn``
        function, and nesting two defns is not allowed, so the fixtures close
        over the bound method and the host objects. The executable is cached per
        ``(__func__, identity of __self__, identity of state, argument
        shapes/dtypes)``; a callee that is not a bound method is not cached, its
        closure being invisible from here.
    """

    def run(method, state, *args):
        owner = getattr(method, "__self__", None)
        function = getattr(method, "__func__", None)
        graph = None
        if owner is not None and function is not None:
            graph = ("kernel", function, _Identity(owner), _Identity(state))
        return _run_one(
            runner,
            _executable_cache,
            graph,
            lambda *arrays: method(state, *arrays),
            args,
        )

    return run


@pytest.fixture(scope="module")
def run_transform(runner, _executable_cache):
    """Derive the execution schedule of a whole payload through ``genome.transform``.

    Returns:
        ``run(genome_object, state, payload) -> Transformed`` of numpy arrays
        (``support.runner.to_numpy`` rebuilds the namedtuple, so the result keeps
        its field names and can be compared field by field against
        ``support.oracle.schedule``).

    Notes:
        ``transform`` is the public entry point, so the tests never name a
        ``Transformed`` field that the derivation might rename; the schedule's
        *contents* are checked against the oracle instead. Cached per
        ``(identity of the genome, identity of the state, shapes/dtypes of
        ``node_keys``/``conn_keys``)``.
    """

    def run(genome_object, state, payload):
        return _run_one(
            runner,
            _executable_cache,
            ("transform", _Identity(genome_object), _Identity(state)),
            lambda node_keys, conn_keys: genome_object.transform(
                state, node_keys, conn_keys
            ),
            (payload.node_keys, payload.conn_keys),
        )

    return run


@pytest.fixture(scope="module")
def run_forward(common, runner, _executable_cache):
    """Evaluate one payload's networks with an already-derived schedule.

    Returns:
        ``run(genome_object, state, transformed, payload, inputs) -> (P,
        num_outputs)`` numpy output of ``genome_object.forward``.

    Notes:
        The schedule is passed **field by field** and the ``Transformed`` is
        rebuilt inside the traced closure, for the reason spelled out in the
        module docstring. ``len(common.Transformed._fields)`` is read instead of
        a literal, so adding a field to the schedule does not silently shift the
        payload arguments. The same helper serves ``DefaultGenome.forward`` (one
        step per level) and ``RecurrentGenome.forward`` (one step per tick). The
        executable is cached per ``(identity of the genome, identity of the
        state, the Transformed type, shapes/dtypes of the schedule fields, of the
        payload and of the inputs)``, so a different ``max_depth``/transform or
        network family can never share an entry.
    """
    width = len(common.Transformed._fields)

    def run(genome_object, state, transformed, payload, inputs):
        fields = tuple(transformed)
        if len(fields) != width:
            raise ValueError(
                f"expected the {width} fields of Transformed, got {len(fields)}"
            )
        return _run_one(
            runner,
            _executable_cache,
            (
                "forward",
                _Identity(genome_object),
                _Identity(state),
                _Identity(common.Transformed),
            ),
            lambda *arrays: genome_object.forward(
                state, common.Transformed(*arrays[:width]), *arrays[width:]
            ),
            (*fields, *payload.args(), inputs),
        )

    return run


@pytest.fixture(scope="module")
def run_prepared(common, runner, _executable_cache):
    """Evaluate one payload through the two halves of ``forward``.

    Returns:
        ``run(genome_object, state, transformed, payload, inputs, samples) ->
        (P * samples, num_outputs)`` numpy output of
        ``genome_object.forward_prepared`` on a record from
        ``genome_object.prepare_pass(..., samples)``.

    Notes:
        The schedule is passed **field by field** and rebuilt inside the traced
        closure, for the reason in the module docstring. ``samples`` is a plain
        Python int, which the runner bakes into the trace as a static value and
        the cache keys by value, so ``S = 1`` and ``S = 4`` never share an
        executable. The call is exactly the composition ``forward`` documents:
        ``prepare_pass`` derives the loop-invariant operands for this payload and
        tiling factor, and ``forward_prepared`` runs the per-query half on
        ``inputs``. Comparing the result with :func:`run_forward` is therefore a
        check that the split is numerically neutral, not a second implementation.
    """
    width = len(common.Transformed._fields)

    def run(genome_object, state, transformed, payload, inputs, samples):
        fields = tuple(transformed)
        if len(fields) != width:
            raise ValueError(
                f"expected the {width} fields of Transformed, got {len(fields)}"
            )

        def traced(*arrays):
            """Prepare one pass, then run its per-query half on the same batch."""
            schedule = common.Transformed(*arrays[:width])
            payload_arrays = arrays[width : width + 4]
            query_inputs = arrays[width + 4]
            prepared = genome_object.prepare_pass(
                state, schedule, *payload_arrays, arrays[width + 5]
            )
            return genome_object.forward_prepared(
                state, schedule, prepared, query_inputs
            )

        return _run_one(
            runner,
            _executable_cache,
            (
                "prepared",
                _Identity(genome_object),
                _Identity(state),
                _Identity(common.Transformed),
            ),
            traced,
            (*fields, *payload.args(), inputs, samples),
        )

    return run
