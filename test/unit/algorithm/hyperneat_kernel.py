"""Run ``HyperNEAT`` kernels under an ``evoxc`` trace: host numpy in, host numpy out.

Why this module exists
----------------------
``src/tensorneat/algorithm/hyperneat/`` is the one algorithm subtree with no
behavioural test, and every one of its kernels (``_weights_from_query``,
``_assemble_substrate``, ``_query_substrate``, ``_expand_schedule``,
``transform``, ``forward``, ...) is a *method that expects symbolic tensors*.
A suite therefore cannot call one directly, any more than it can call a genome
kernel.  This module is the single place that knows the recipe, so a test reads
``run_method(hyper, "_weights_from_query", query)`` and nothing else.

The recipe -- three rules, all of them load-bearing
--------------------------------------------------
1. **A method is executed through a closure.**  ``evoxc`` refuses to let two
   ``@evoxc.defn`` functions call each other, so the traced graph is
   ``lambda *leaves: method(state_or_nothing, *rebuilt)`` and the ordinary method
   is the definition boundary itself.
2. **Only numpy arrays cross the boundary as data.**  ``support.runner.run_defn``
   turns a numpy leaf into a ``TensorSpec`` and bakes every other leaf into the
   trace as a static value -- which is exactly what a Python ``int`` (``samples``,
   ``pop_size``) or a ``None`` (an unused optional argument) has to be.
3. **The ``State`` is a static host object, never a graph input.**  It is captured
   by the closure.  ``HyperNEAT``'s kernels read *no entry* from it (the gene
   payload is always an explicit argument; only ``NEAT.setup`` reads
   ``state.randkey``), and a ``State`` handed to ``evoxc.build`` as an argument
   would be flattened into the population tensors it carries.

A namedtuple/pytree argument -- a ``Transformed`` schedule, a ``SubstrateSchedule``,
a ``GenomeTensors`` payload -- is **flattened by this module** into its leaves and
rebuilt *inside* the trace.  Passing such a container as one argument would let the
runner bake it in as a static value whose fields are plain numpy arrays, so every
``transformed.in_conn`` inside the kernel would be an ndarray instead of a symbolic
tensor.  Nested namedtuples (a ``SubstrateSchedule`` holds a ``Transformed``) are
handled recursively.

What a call returns
-------------------
Host numpy, always: ``support.runner.run_defn`` unwraps ``evoxc.Tensor`` leaves and
rebuilds namedtuples, so a returned ``Transformed`` / ``SubstrateSchedule`` /
``GenomeTensors`` comes back as the same namedtuple type holding numpy arrays.

Scope of the default geometry
-----------------------------
:func:`make_hyper` builds the smallest substrate that still exercises every block of
:class:`~tensorneat.algorithm.hyperneat.substrate.FullSubstrate`:

======================  ===========================================
``input_coors``         ``((0, 0), (1, 0))`` -- 2 neurons, the last is the bias
``hidden_coors``        ``((0, 1),)`` -- 1 hidden neuron, self-loop included
``output_coors``        ``((0, 2),)`` -- 1 output neuron
``P`` (population)      4 CPPNs
``Q`` (connections)     ``2*1 + 1*1 + 1*1 = 4`` -- the query count
``N`` (substrate nodes) ``2 + 1 + 1 = 4``
``C`` (CPPN inputs)     ``2 * coordinate_dim = 4`` -- the query width
======================  ===========================================

so the CPPN genome must be ``num_inputs=4`` / ``num_outputs=1`` and every payload
tensor has a leading axis of ``P = 4``.

No test body lives here and nothing here is collected by pytest (the module name has
no ``test_`` prefix); the suites import it through ``from .hyperneat_kernel import
make_hyper, run_method``.
"""

from __future__ import annotations

import inspect
from typing import Any

import evoxc

from support.runner import DEFAULT_BACKEND, run_defn, to_numpy

from tensorneat.algorithm.hyperneat import HyperNEAT, HyperNEATFeedForward, FullSubstrate
from tensorneat.algorithm.neat import NEAT
from tensorneat.common import ACT, AGG, State
from tensorneat.genome import DefaultGenome

__all__ = [
    "ACTIVATE_TIME",
    "CONNS_CNT",
    "HIDDEN_COORS",
    "INPUT_COORS",
    "NODES_CNT",
    "OUTPUT_COORS",
    "POP_SIZE",
    "QUERY_WIDTH",
    "SPECIES_SIZE",
    "SURVIVAL_THRESHOLD",
    "cppn_payload",
    "make_cppn_genome",
    "make_hyper",
    "make_neat",
    "make_state",
    "make_substrate",
    "run_method",
]

#: Input coordinates of the default substrate: two neurons, the **last** one is the bias
#: (``HyperNEAT.forward`` appends the column of ones there), so the substrate has one real
#: problem input and ``HyperNEAT.num_inputs == 1``.
INPUT_COORS = ((0.0, 0.0), (1.0, 0.0))

#: Hidden coordinates of the default substrate: one neuron, hence one self-loop connection.
HIDDEN_COORS = ((0.0, 1.0),)

#: Output coordinates of the default substrate: one neuron.
OUTPUT_COORS = ((0.0, 2.0),)

#: CPPN population size ``P`` of the default harness.  Small on purpose: every assertion in
#: the suites is readable by hand and one trace costs ~0.05 s at this size.
POP_SIZE = 4

#: Species-slot capacity ``S`` of the default CPPN algorithm.
SPECIES_SIZE = 2

#: Parent survival fraction of the default CPPN algorithm (``NEAT``'s own default).
SURVIVAL_THRESHOLD = 0.1

#: Recurrent activation ticks of the substrate executor.  ``HyperNEAT``'s default is 10; a
#: single-hidden-neuron substrate saturates long before that, and every tick is one extra
#: loop iteration in the traced graph, so the harness pins 3.
ACTIVATE_TIME = 3

#: CPPN input width ``C = 2 * coordinate_dim`` of the default substrate -- the number of
#: columns of ``substrate.query_coors``, and therefore ``neat.num_inputs``.
QUERY_WIDTH = 4

#: Connection (and query) count ``Q`` of the default substrate: ``2*1 + 1*1 + 1*1``.
CONNS_CNT = 4

#: Node count ``N`` of the default substrate: 2 inputs + 1 output + 1 hidden.
NODES_CNT = 4


# --------------------------------------------------------------------------- #
# builders (host-only, no graph is traced)
# --------------------------------------------------------------------------- #


def make_substrate(
    input_coors: Any = INPUT_COORS,
    hidden_coors: Any = HIDDEN_COORS,
    output_coors: Any = OUTPUT_COORS,
) -> FullSubstrate:
    """Build the default (or a caller-supplied) ``FullSubstrate`` geometry.

    Args:
        input_coors: ``(num_inputs, cd)`` coordinates, bias neuron last.
        hidden_coors: ``(num_hidden, cd)`` coordinates; must be non-empty.
        output_coors: ``(num_outputs, cd)`` coordinates.

    Returns:
        A host-only :class:`~tensorneat.algorithm.hyperneat.substrate.FullSubstrate`;
        constructing it builds no graph and creates no tensor.

    Note:
        The default is the module-level geometry, so
        ``make_substrate().conns_cnt == CONNS_CNT`` and
        ``make_substrate().query_coors.shape == (CONNS_CNT, QUERY_WIDTH)``.
        ``connection_type`` is ``"recurrent"``, which is what :class:`HyperNEAT` requires;
        a feedforward substrate (``DefaultSubstrate``/``MLPSubstrate``) is the caller's job.
    """
    return FullSubstrate(
        input_coors=input_coors,
        hidden_coors=hidden_coors,
        output_coors=output_coors,
    )


def make_cppn_genome(
    num_inputs: int = QUERY_WIDTH,
    num_outputs: int = 1,
    **overrides: Any,
) -> DefaultGenome:
    """Build the CPPN genome: a plain feedforward ``DefaultGenome``.

    Args:
        num_inputs: CPPN input width -- must equal ``substrate.query_coors.shape[1]``
            (``QUERY_WIDTH`` for the default substrate), *not* the substrate's input count.
        num_outputs: CPPN output width; ``HyperNEAT`` rejects anything but ``1`` because
            the single output is the connection weight.
        **overrides: Any other ``DefaultGenome`` keyword (``max_nodes``, ``mutation``, ...).

    Returns:
        A fresh :class:`~tensorneat.genome.DefaultGenome`; its operators are built per
        instance, so two calls never share mutable gene tables.
    """
    config: dict[str, Any] = {
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "output_transform": ACT.tanh,
    }
    config.update(overrides)
    return DefaultGenome(**config)


def make_neat(
    genome: Any = None,
    pop_size: int = POP_SIZE,
    species_size: int = SPECIES_SIZE,
    survival_threshold: float = SURVIVAL_THRESHOLD,
    **overrides: Any,
) -> NEAT:
    """Build the CPPN evolution algorithm around a genome (a fresh one by default).

    Args:
        genome: CPPN genome; ``None`` builds :func:`make_cppn_genome`.
        pop_size: CPPN population size ``P``; it is also the batch size of every decoded
            payload, so it must match the payload a test hands to ``run_method``.
        species_size: Species-slot capacity ``S``.
        survival_threshold: Parent survival fraction.
        **overrides: Any other ``NEAT`` keyword (``compatibility_threshold``, ...).

    Returns:
        A :class:`~tensorneat.algorithm.neat.NEAT`; constructing it builds no graph, so a
        test that only checks configuration pays nothing for it.
    """
    config: dict[str, Any] = {
        "genome": make_cppn_genome() if genome is None else genome,
        "pop_size": pop_size,
        "species_size": species_size,
        "survival_threshold": survival_threshold,
    }
    config.update(overrides)
    return NEAT(**config)


def make_hyper(
    substrate: Any = None,
    neat: Any = None,
    *,
    weight_threshold: float = 0.3,
    max_weight: float = 5.0,
    aggregation: Any = AGG.sum,
    activation: Any = ACT.tanh,
    activate_time: int = ACTIVATE_TIME,
    output_transform: Any = ACT.sigmoid,
    hyper_class: Any = HyperNEAT,
) -> HyperNEAT:
    """Build the shared small ``HyperNEAT`` over :func:`make_substrate` and :func:`make_neat`.

    Args:
        substrate: Host geometry; ``None`` builds :func:`make_substrate`.  A feedforward
            substrate is required when ``hyper_class is HyperNEATFeedForward``.
        neat: CPPN algorithm; ``None`` builds :func:`make_neat`.
        weight_threshold: Dead-zone half-width of the CPPN-to-weight map, ``0 <= x < 1``.
        max_weight: Saturating weight magnitude, ``> 0``.
        aggregation: Fixed substrate-node aggregation ``(z, valid_mask) -> z``.
        activation: Fixed substrate-node activation.  ``ACT.tanh`` rather than the
            constructor's ``ACT.sigmoid`` so a decoded substrate node does not saturate at
            1 within :data:`ACTIVATE_TIME` ticks.
        activate_time: Recurrent ticks of the substrate executor.
        output_transform: Output activation of the *substrate* network.
        hyper_class: ``HyperNEAT`` (recurrent, the default) or ``HyperNEATFeedForward``.

    Returns:
        The algorithm.  ``out.weight_threshold`` / ``out.max_weight`` are the values the
        traced map will use, and ``out.hyper_genome`` is the dense executor genome sized
        from the substrate (``max_nodes == NODES_CNT``, ``max_conns == CONNS_CNT``).

    Note:
        Every constructor keyword is filtered through ``hyper_class``'s signature, because
        the feedforward variant does not accept ``activate_time``; passing one constructor
        a keyword the other lacks would otherwise be a ``TypeError`` at the call site.
        ``__init__`` is pure host Python (validation + genome construction), so no graph is
        built here and ``setup`` remains the only workload.
    """
    config: dict[str, Any] = {
        "substrate": make_substrate() if substrate is None else substrate,
        "neat": make_neat() if neat is None else neat,
        "weight_threshold": weight_threshold,
        "max_weight": max_weight,
        "aggregation": aggregation,
        "activation": activation,
        "activate_time": activate_time,
        "output_transform": output_transform,
    }
    accepted = set(inspect.signature(hyper_class).parameters) - {"self"}
    return hyper_class(**{key: value for key, value in config.items() if key in accepted})


def make_state(hyper: HyperNEAT, seed: int = 0) -> State:
    """Run ``hyper.setup`` on a freshly seeded ``randkey`` and return the resulting state.

    Args:
        hyper: The algorithm from :func:`make_hyper`.
        seed: Seed handed to ``evoxc.random.key``.  ``setup`` splits it once, so a
            ``(seed, config, shape)`` triple reproduces the population exactly.

    Returns:
        The host :class:`~tensorneat.common.State` one generation carries:
        ``pop`` (a ``GenomeTensors`` of ``evoxc.Tensor`` leaves), ``generation`` (0-d
        float32 ``0.0``), ``species``, ``pop_size`` and the advanced ``randkey``.

    Raises:
        Any ``TensorNEATError`` / ``evoxc`` error ``setup`` raises: a broken configuration
        must surface here rather than in the kernel under test.

    Note:
        This is a plain function, not a fixture: a pytest fixture defined outside
        ``conftest.py`` is not auto-registered, and the suites in this directory prefer an
        explicit call.  ``setup`` costs one small ``evoxc.build`` (~0.02 s at ``P = 4``).
    """
    return hyper.setup(State(randkey=evoxc.random.key(int(seed))))


def cppn_payload(state: State) -> Any:
    """Return ``state.pop`` as host numpy -- the CPPN payload a decode is traced on.

    Args:
        state: A state returned by :func:`make_state` (or any state carrying ``pop``).

    Returns:
        The ``GenomeTensors`` namedtuple (``node_keys``/``node_attrs``/``conn_keys``/
        ``conn_attrs``) of **numpy** arrays, batch-first at ``P = state.pop_size``.  Pass it
        straight to :func:`run_method`; the harness flattens and rebuilds it.

    Raises:
        tensorneat.common.StateKeyError: If ``state`` has no ``pop`` entry (``setup`` was
            never run).
    """
    return to_numpy(state.pop)


# --------------------------------------------------------------------------- #
# the traced-call harness
# --------------------------------------------------------------------------- #

#: Signature parameter name that marks a method taking the host state first.
_STATE_PARAM = "state"


def run_method(
    obj: Any,
    method_name: str,
    *args: Any,
    state: Any = None,
    backend: str = DEFAULT_BACKEND,
    **options: Any,
) -> Any:
    """Trace ``obj.<method_name>`` once and run it, returning host numpy.

    Usage::

        weights = run_method(hyper, "_weights_from_query", query_res)       # no state
        payload = run_method(hyper, "_assemble_substrate", weights)         # no state
        schedule = run_method(hyper, "transform", pop, state=state)         # state first
        out = run_method(hyper, "forward", schedule, inputs, state=state)

    Args:
        obj: The object owning the method (normally the ``HyperNEAT`` from
            :func:`make_hyper`).
        method_name: Name of a *bound* method or plain function attribute of ``obj``.
        *args: The method's arguments after ``self`` (and after ``state`` when one is
            given), in order.  Each argument is one of:

            * a ``numpy.ndarray`` -- crosses the graph boundary as a tensor input;
            * a Python scalar (``int``/``float``/``bool``/``None``/``str``) -- a static
              value baked into the trace, which is how an ``samples``/``pop_size``
              argument or an unused ``None`` optional is passed;
            * a ``tuple``/namedtuple (recursively) of the two above -- its leaves are
              passed positionally and the container is rebuilt inside the trace, which
              is what a ``Transformed``/``SubstrateSchedule``/``GenomeTensors`` needs
              (see the module docstring).

            A ``list`` or ``dict`` argument is *not* flattened here; it reaches the
            runner as one value, which traverses it leaf by leaf.
        state: Host :class:`~tensorneat.common.State` to prepend as the method's first
            argument, or ``None`` for a method that takes no state
            (``_weights_from_query``, ``_assemble_substrate``, ``_expand_schedule``,
            ``_static_bounds``).  The state is captured by the traced closure as a static
            host object and is never a graph input.
        backend: ``evoxc`` backend name; ``"numpy"`` (the default) is the reference
            interpreter and the only backend the suites use outside ``test/backends``.
        **options: Extra ``evoxc.build`` options (e.g. ``device``).

    Returns:
        Host numpy: ``evoxc.Tensor`` leaves unwrapped and namedtuples rebuilt at every
        nesting level, so a returned ``Transformed``/``SubstrateSchedule`` is the same
        namedtuple type holding arrays.  The per-call *shape* is the method's own.

    Raises:
        TypeError: If ``method_name`` is not a callable attribute of ``obj``, or if the
            method takes a ``state`` first argument while ``state`` is ``None`` (a call
            that would otherwise silently shift every argument by one).
        evoxc.TraceError, evoxc.ShapeError, evoxc.DTypeError: Propagated unchanged.

    Note:
        Nothing is cached here: every call traces and builds a fresh executable
        (~0.05 s at the default sizes, which keeps a test that needs two different static
        configurations honest).  Callers that repeat one graph with only data changing
        should still go through this function -- ``evoxc.build`` is fast at these sizes --
        rather than hold an ``Executable`` themselves, which would bypass
        ``support.runner``.
    """
    method = getattr(obj, method_name)
    if not callable(method):
        raise TypeError(
            f"{type(obj).__name__}.{method_name} is not callable, got {method!r}"
        )
    parameters = list(inspect.signature(method).parameters)
    if state is None and parameters and parameters[0] == _STATE_PARAM:
        raise TypeError(
            f"{type(obj).__name__}.{method_name} takes '{_STATE_PARAM}' as its first "
            "argument; pass it with state=... so the remaining arguments are not shifted"
        )

    specs = tuple(_spec(argument) for argument in args)
    leaves = [leaf for argument in args for leaf in _leaves(argument)]

    def kernel(*values):
        rebuilt = []
        cursor = 0
        for spec in specs:
            argument, cursor = _rebuild(spec, values, cursor)
            rebuilt.append(argument)
        if state is None:
            return method(*rebuilt)
        return method(state, *rebuilt)

    return run_defn(kernel, *leaves, backend=backend, **options)


def _spec(value: Any) -> tuple:
    """Classify one call argument: a tree of specs for a tuple, one leaf otherwise."""
    if isinstance(value, tuple):
        return ("tuple", type(value), tuple(_spec(item) for item in value))
    return ("leaf",)


def _leaves(value: Any) -> list:
    """Flatten one call argument into the positional leaves the trace receives."""
    if isinstance(value, tuple):
        return [leaf for item in value for leaf in _leaves(item)]
    return [value]


def _rebuild(spec: tuple, values: tuple, cursor: int) -> tuple:
    """Rebuild one argument from ``values[cursor:]``, returning ``(argument, next_cursor)``."""
    if spec[0] == "leaf":
        return values[cursor], cursor + 1
    _, container, children = spec
    items = []
    for child in children:
        item, cursor = _rebuild(child, values, cursor)
        items.append(item)
    # ``tuple(*items)`` would read a namedtuple's first field as the tuple's length.
    return (tuple(items) if container is tuple else container(*items)), cursor
