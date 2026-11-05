"""Unit tests for ``HyperNEATFeedForward``: the feedforward substrate variant.

What this module proves
-----------------------
``HyperNEATFeedForward`` shares the whole CPPN-to-substrate decode with ``HyperNEAT``
and differs in exactly three declared places -- ``_genome_cls`` is a ``DefaultGenome``,
``_required_connection_type`` is ``"feedforward"`` and ``_genome_extra_kwargs()`` is
empty -- plus an ``__init__`` that drops ``activate_time``.  So every assertion here is
about one of two things: the *delta* is exactly that, or the *shared* decode really is
shared (same gene payload, same phenotype, same population-axis independence).

Coverage, in test order: the three subclass hooks and the symmetric base values they
override; the constructor signature (same documented defaults for every other argument,
no ``activate_time``, ``TypeError`` if one is passed); the executor genome (a
``DefaultGenome`` whose ``max_depth`` is the substrate's longest input-to-node path,
against :func:`_hand_bounds`, while the recurrent base passes no ``max_depth`` and keeps
its own default); ``_required_connection_type`` enforcement and the base class's ``None``
requirement, which accepts both geometries; the same CPPN payload decoded into the same
four gene tables with no dropped edge; the same phenotype on a two-layer substrate, where
the recurrent single tick and the level sweep are one arithmetic; the trailing-bias
contract, including the bias-first answer it rules out; the prepared-pass split
(``forward`` == ``prepare_forward`` + ``forward_prepared``) on a deeper substrate;
``activate_time`` being unobservable by a feedforward forward; ``transform``'s
population-axis independence for the feedforward executor; and one ``Pipeline`` step
that runs the variant's whole generation graph.

Oracles (never the implementation)
----------------------------------
* :func:`_two_layer_forward` -- a float32 transcription of the documented substrate
  forward for a topology whose every edge is ``input -> output``: the ones column
  appended **last**, one ``float32`` product per connection, the left-to-right
  accumulation the aggregation fold performs, then ``ACT.sigmoid``.  It reads the
  weights out of the decoded ``conn_attrs``, so it covers decode and execution at once.
* :func:`_hand_bounds` -- ``max_in_degree``/``max_depth`` re-derived from
  ``substrate.conn_index`` with a Python relaxation loop, not from ``node_depth``.
* :func:`_bias_first_answer` -- the same map with the ones column **prepended**, which
  is what the "last column is the bias" contract must rule out.
* Literal hand-stated values (``layers [3, 2, 1]`` -> in-degree ``3``, depth ``2``) and
  the substrate's own ``nodes``/``conn_index`` tables.

No network, no GPU, ``backend="numpy"``; every population is ``P = 4`` CPPNs and every
substrate is a tiny ``MLPSubstrate`` grid.
"""

from __future__ import annotations

import inspect
from typing import Any, NamedTuple

import numpy as np
import pytest

from tensorneat import Pipeline, problem
from tensorneat.algorithm.hyperneat import SubstrateSchedule
from tensorneat.algorithm.hyperneat.hyperneat import HyperNEAT
from tensorneat.algorithm.hyperneat.hyperneat_feedforward import HyperNEATFeedForward
from tensorneat.algorithm.hyperneat.substrate import MLPSubstrate
from tensorneat.common import ACT, AGG
from tensorneat.common.errors import TensorNEATError
from tensorneat.common.tools import to_numpy
from tensorneat.genome import DefaultGenome, RecurrentGenome

from .hyperneat_kernel import (
    POP_SIZE,
    cppn_payload,
    make_hyper,
    make_neat,
    make_state,
    make_substrate,
    run_method,
)

#: Layer sizes of the deep geometry: 3 inputs (2 problem inputs + bias) -> 2 hidden
#: -> 1 output.  Its widest in-degree is the hidden layer's ``3`` and its longest
#: input-to-node path is ``2`` (input -> hidden -> output).
DEEP_LAYERS = (3, 2, 1)

#: Layer sizes of the two-layer geometry: 2 inputs (1 problem input + bias) -> 1
#: output.  No hidden node, so the recurrent tick and the level sweep must be one
#: and the same arithmetic.
EDGE_LAYERS = (2, 1)

#: Layer sizes of the bias-contract geometry: 3 inputs (2 problem inputs + bias)
#: -> 1 output, the smallest substrate in which "which column carries the bias"
#: is unambiguous.
BIAS_LAYERS = (3, 1)

#: Coordinate box every substrate of this module is laid out in.
BOX = (-1.0, 1.0, -1.0, 1.0)

#: Recurrent ticks of the deep recurrent variant.  Deliberately not the default 10,
#: so ``_genome_extra_kwargs()`` cannot match by accident.
RECURRENT_TICKS = 3

#: ``(P, substrate.num_inputs - 1)`` problem inputs of the deep geometry.
DEEP_INPUTS = np.array(
    [[0.1, -0.4], [0.3, 0.6], [-1.2, 0.5], [0.0, 0.0]], dtype=np.float32
)

#: ``(P, 1)`` problem inputs of the two-layer geometry.
EDGE_INPUTS = np.array([[0.2], [-0.7], [1.5], [0.0]], dtype=np.float32)

#: Hand-painted weights of the bias-contract geometry, one per connection row
#: ``(input 0 -> output, input 1 -> output, bias -> output)``.  Three distinct
#: magnitudes, so a permuted column shows up immediately.
BIAS_WEIGHTS = np.array([0.5, -1.5, 2.0], dtype=np.float32)

#: ``(P, 2)`` problem inputs of the bias-contract geometry.
BIAS_INPUTS = np.array(
    [[0.1, 0.2], [0.4, -0.3], [-0.8, 0.6], [0.0, 1.0]], dtype=np.float32
)


# --------------------------------------------------------------------------- #
# oracles
# --------------------------------------------------------------------------- #


def _mlp(layers: tuple[int, ...]) -> MLPSubstrate:
    """Build a fresh host-only ``MLPSubstrate`` over :data:`BOX`.

    Args:
        layers: Per-layer neuron counts; ``layers[0]`` counts the bias neuron.

    Returns:
        The host geometry, with no graph and no tensor.
    """
    return MLPSubstrate(layers=list(layers), coor_range=BOX)


def _hand_bounds(substrate: Any) -> tuple[int, int]:
    """Derive ``(max_in_degree, max_depth)`` from ``conn_index`` alone.

    ``max_in_degree`` is the largest per-destination in-degree, floored at
    ``num_inputs`` (the genome constructor validates the window against its dense
    initial network) and at ``1``.  ``max_depth`` is the longest path from the input
    keys, floored at ``1``, computed by a plain relaxation loop over the edge list --
    the rule restated without :func:`tensorneat.common.host_graph.node_depth`.

    Args:
        substrate: A host substrate whose ``nodes``/``conn_index``/``num_inputs`` are
            read.

    Returns:
        ``(K, D)`` as Python ints.

    Raises:
        AssertionError: If some node is not reachable from the inputs, i.e. the
            substrate is not the DAG this oracle assumes.
    """
    conn_index = np.asarray(substrate.conn_index)
    nodes_cnt = int(substrate.nodes_cnt)
    num_inputs = int(substrate.num_inputs)
    in_degree = np.bincount(conn_index[:, 1], minlength=nodes_cnt)
    max_in_degree = max(int(in_degree.max()), num_inputs, 1)
    depth = {key: 0 for key in range(num_inputs)}
    for _ in range(nodes_cnt):
        for src, dst in conn_index:
            if int(src) in depth:
                candidate = depth[int(src)] + 1
                if depth.get(int(dst), -1) < candidate:
                    depth[int(dst)] = candidate
    assert len(depth) == nodes_cnt, (
        "every node must be reachable from the inputs for this oracle: reached "
        f"{sorted(depth)} of {nodes_cnt} nodes, got "
        f"{sorted(int(key) for key in substrate.nodes)}"
    )
    return max_in_degree, max(max(depth.values()), 1)


def _sigmoid_f32(z: np.ndarray) -> np.ndarray:
    """The documented output map ``1 / (1 + exp(-z))`` in float32 steps."""
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-z.astype(np.float32)))).astype(
        np.float32
    )


def _two_layer_forward(
    substrate: Any, weights: np.ndarray, inputs: np.ndarray
) -> np.ndarray:
    """Hand evaluation of a substrate whose every edge is ``input -> output``.

    The documented forward: append the ones column **last** (the trailing substrate
    input *is* the bias neuron), multiply each connection's source value by its weight,
    accumulate the window's slots left to right in connection-row order (the order the
    aggregation fold uses), then apply the output transform ``ACT.sigmoid``.  No node
    activation is involved because the only non-input nodes are output nodes.

    Args:
        substrate: The host geometry.
        weights: ``(P, Q)`` float32 decoded weights, ``Q = substrate.conns_cnt``.
        inputs: ``(P, substrate.num_inputs - 1)`` float32 problem inputs.

    Returns:
        ``(P, 1)`` float32 substrate outputs.

    Raises:
        AssertionError: If the substrate has a hidden node, i.e. the premise of this
            oracle is violated and the comparison would be meaningless.
    """
    conn_index = np.asarray(substrate.conn_index)
    num_inputs = int(substrate.num_inputs)
    output_key = int(substrate.nodes[num_inputs])
    assert np.all(conn_index[:, 1] == output_key), (
        "the two-layer oracle only covers substrates whose every connection ends at "
        f"the single output key {output_key}, got destinations {conn_index[:, 1].tolist()}"
    )
    assert conn_index.shape[0] == weights.shape[1], (
        "one weight per connection row: expected "
        f"{conn_index.shape[0]} weights, got {weights.shape[1]}"
    )
    biased = np.concatenate(
        [inputs.astype(np.float32), np.ones((inputs.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    acc = np.zeros(inputs.shape[0], dtype=np.float32)
    for row in range(conn_index.shape[0]):
        acc = (acc + weights[:, row] * biased[:, int(conn_index[row, 0])]).astype(
            np.float32
        )
    return _sigmoid_f32(acc)[:, None]


def _bias_first_answer(
    substrate: Any, weights: np.ndarray, inputs: np.ndarray
) -> np.ndarray:
    """The same map with the ones column **prepended** instead of appended.

    The wrong answer a bias-first implementation would produce; the bias-contract test
    asserts the traced forward differs from it, so the check discriminates.

    Args:
        substrate: The host geometry (every edge ``input -> output``).
        weights: ``(P, Q)`` float32 weights, in connection-row order.
        inputs: ``(P, substrate.num_inputs - 1)`` float32 problem inputs.

    Returns:
        ``(P, 1)`` float32 outputs of the permuted map.
    """
    conn_index = np.asarray(substrate.conn_index)
    shifted = np.concatenate(
        [np.ones((inputs.shape[0], 1), dtype=np.float32), inputs.astype(np.float32)],
        axis=1,
    )
    acc = np.zeros(inputs.shape[0], dtype=np.float32)
    for row in range(conn_index.shape[0]):
        acc = (acc + weights[:, row] * shifted[:, int(conn_index[row, 0])]).astype(
            np.float32
        )
    return _sigmoid_f32(acc)[:, None]


class _PreparedForward:
    """One graph that runs ``prepare_forward`` and then ``forward_prepared``.

    Why this adapter exists: ``prepare_forward`` returns a host-side record
    (``_PreparedPass`` holding ``_StepStatics``), which an ``evoxc`` graph cannot
    return, so the two documented halves are only observable *composed inside one
    trace*.  The adapter is that composition and nothing else -- ``run_method`` still
    owns the trace/flatten/rebuild recipe, so this file contains no evoxc plumbing.
    """

    def __init__(self, hyper: Any):
        """Store the algorithm whose two public halves are composed."""
        self.hyper = hyper

    def forward_via_prepared(
        self, state: Any, transformed: Any, samples: int, inputs: Any
    ) -> Any:
        """Return ``forward_prepared`` of the record ``prepare_forward`` derived."""
        prepared = self.hyper.prepare_forward(state, transformed, None, samples)
        return self.hyper.forward_prepared(state, transformed, prepared, inputs)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


class _VariantCase(NamedTuple):
    """Both variants over one substrate, plus the shared CPPN payload and schedules."""

    substrate: Any
    feedforward: Any
    recurrent: Any
    state: Any
    payload: Any
    schedule_ff: Any
    schedule_base: Any


def _build_case(layers: tuple[int, ...], activate_time: int) -> _VariantCase:
    """Construct both variants over one substrate and decode one shared payload.

    Both ``transform`` calls receive the **same** CPPN payload array, so the payload
    comparison is about the decode and nothing else.  The state is the feedforward
    instance's: ``HyperNEAT``'s kernels read no ``State`` entry (only ``setup`` reads
    ``randkey``), so a decode that did would show up as a wrong gene table, not a crash.

    Args:
        layers: Layer sizes of the ``MLPSubstrate`` to build.
        activate_time: Recurrent ticks of the base ``HyperNEAT``.

    Returns:
        Both algorithms, one state, one payload and the two decoded schedules.
    """
    substrate = _mlp(layers)
    feedforward = make_hyper(substrate=substrate, hyper_class=HyperNEATFeedForward)
    recurrent = make_hyper(
        substrate=substrate, hyper_class=HyperNEAT, activate_time=activate_time
    )
    state = make_state(feedforward, seed=0)
    payload = cppn_payload(state)
    return _VariantCase(
        substrate=substrate,
        feedforward=feedforward,
        recurrent=recurrent,
        state=state,
        payload=payload,
        schedule_ff=run_method(feedforward, "transform", payload, state=state),
        schedule_base=run_method(recurrent, "transform", payload, state=state),
    )


@pytest.fixture(scope="module")
def deep_case() -> _VariantCase:
    """Both variants over ``MLPSubstrate([3, 2, 1])``, the deep geometry."""
    return _build_case(DEEP_LAYERS, RECURRENT_TICKS)


@pytest.fixture(scope="module")
def edge_case() -> _VariantCase:
    """Both variants over ``MLPSubstrate([2, 1])``, the no-hidden-node geometry."""
    return _build_case(EDGE_LAYERS, 1)


@pytest.fixture(scope="module")
def bias_case() -> _VariantCase:
    """Both variants over ``MLPSubstrate([3, 1])``, the bias-contract geometry."""
    return _build_case(BIAS_LAYERS, 1)


@pytest.fixture(scope="module")
def tiny_pipeline() -> Pipeline:
    """A one-generation feedforward ``HyperNEAT`` pipeline (`pop_size=4`, `XOR`).

    ``fitness_target=1.0`` is unreachable (XOR fitness is a negated mse), so the
    pipeline never stops early and :meth:`Pipeline.step` is the only caller.
    """
    return Pipeline(
        algorithm=make_hyper(
            substrate=_mlp(BIAS_LAYERS), hyper_class=HyperNEATFeedForward
        ),
        problem=problem.XOR(),
        seed=0,
        generation_limit=1,
        fitness_target=1.0,
    )


# --------------------------------------------------------------------------- #
# 1. the variant delta
# --------------------------------------------------------------------------- #


def test_variant_hooks_select_a_default_genome(deep_case: _VariantCase) -> None:
    """The three subclass hooks are exactly the documented feedforward delta.

    Oracle: the class attributes themselves, plus the base class's values -- which must
    be the *other* choice on all three hooks.  The deep case is built with
    ``activate_time=3``, so the base's extra kwargs cannot match ``{}`` by accident.

    Assertion: ``_genome_cls`` is ``DefaultGenome``, ``_required_connection_type`` is
    ``"feedforward"`` and ``_genome_extra_kwargs()`` is ``{}``, against
    ``RecurrentGenome`` / ``None`` / ``{"activate_time": 3}`` on the base.
    """
    assert HyperNEATFeedForward._genome_cls is DefaultGenome, (
        "the feedforward variant must execute its substrate with DefaultGenome, got "
        f"{HyperNEATFeedForward._genome_cls!r}"
    )
    assert HyperNEATFeedForward._required_connection_type == "feedforward", (
        "the feedforward variant must require a feedforward substrate, got "
        f"{HyperNEATFeedForward._required_connection_type!r}"
    )
    assert deep_case.feedforward._genome_extra_kwargs() == {}, (
        "a feedforward executor has no recurrent horizon, so its genome kwargs must be "
        f"empty, got {deep_case.feedforward._genome_extra_kwargs()!r}"
    )

    assert HyperNEAT._genome_cls is RecurrentGenome, (
        f"the base class must default to RecurrentGenome, got {HyperNEAT._genome_cls!r}"
    )
    assert HyperNEAT._required_connection_type is None, (
        "the base class must accept either connection type, so its requirement must be "
        f"None, got {HyperNEAT._required_connection_type!r}"
    )
    assert deep_case.recurrent._genome_extra_kwargs() == {
        "activate_time": RECURRENT_TICKS
    }, (
        "the recurrent variant must forward its own activate_time, got "
        f"{deep_case.recurrent._genome_extra_kwargs()!r}"
    )


def test_constructor_signature_drops_activate_time_only() -> None:
    """``__init__`` mirrors ``HyperNEAT.__init__`` minus ``activate_time``.

    Oracle: the two signatures side by side and the documented defaults.

    Assertion: the parameter sets differ by exactly ``{"activate_time"}``; every other
    documented argument keeps the same default on both; and passing ``activate_time`` to
    the feedforward constructor is a ``TypeError`` rather than a silently ignored kwarg.
    """
    feedforward = inspect.signature(HyperNEATFeedForward.__init__).parameters
    recurrent = inspect.signature(HyperNEAT.__init__).parameters

    assert "activate_time" not in feedforward, (
        "HyperNEATFeedForward.__init__ must not accept activate_time, got parameters "
        f"{list(feedforward)}"
    )
    assert "activate_time" in recurrent, (
        "HyperNEAT.__init__ must keep activate_time, got parameters "
        f"{list(recurrent)}"
    )
    assert set(recurrent) - set(feedforward) == {"activate_time"}, (
        "the feedforward signature must differ from the base one by activate_time "
        f"only, got extra {sorted(set(recurrent) - set(feedforward))} and "
        f"missing {sorted(set(feedforward) - set(recurrent))}"
    )

    documented = {
        "weight_threshold": 0.3,
        "max_weight": 5.0,
        "aggregation": AGG.sum,
        "activation": ACT.sigmoid,
        "output_transform": ACT.sigmoid,
    }
    for name, default in documented.items():
        assert name in feedforward, (
            f"{name} must stay a keyword argument of the feedforward constructor, got "
            f"parameters {list(feedforward)}"
        )
        got = feedforward[name].default
        base = recurrent[name].default
        if callable(default):
            assert got is default, (
                f"{name} must default to the documented callable {default!r}, got {got!r}"
            )
            assert base is got, (
                f"{name} must keep the base class's default on the subclass too: base "
                f"{base!r}, subclass {got!r}"
            )
        else:
            assert got == default, (
                f"{name} must default to the documented value {default!r}, got {got!r}"
            )
            assert base == got, (
                f"{name} must keep the base class's default on the subclass too: base "
                f"{base!r}, subclass {got!r}"
            )

    for required in ("substrate", "neat"):
        assert feedforward[required].default is inspect.Parameter.empty, (
            f"{required} must stay required (no default), got "
            f"{feedforward[required].default!r}"
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'activate_time'"):
        HyperNEATFeedForward(
            substrate=_mlp(EDGE_LAYERS), neat=make_neat(), activate_time=3
        )


def test_executor_genome_is_a_default_genome_with_the_substrate_depth(
    deep_case: _VariantCase,
) -> None:
    """The executor is a dense ``DefaultGenome`` sized and deepened from the substrate.

    Oracle: :func:`_hand_bounds` -- in-degree and longest input-to-node path derived
    from ``conn_index`` alone -- restated as the literal ``(3, 2)`` for
    ``MLPSubstrate([3, 2, 1])``.

    Assertion: the genome is exactly a ``DefaultGenome``, dense (``max_nodes`` /
    ``max_conns`` are the substrate counts), with ``max_in_degree`` / ``max_depth``
    equal to the hand bounds and no ``activate_time`` at all; the derived depth is also
    the deepest level the decoded schedule visits, so the bound is neither gratuitous
    nor truncating.  Over the same substrate the recurrent base passes **no**
    ``max_depth``: its genome keeps ``RecurrentGenome``'s own default, which is not the
    substrate depth.
    """
    substrate = deep_case.substrate
    degree, depth = _hand_bounds(substrate)
    assert (degree, depth) == (3, 2), (
        "the hand derivation for MLPSubstrate([3, 2, 1]) must be in-degree 3 (the "
        f"hidden layer's three sources) and depth 2 (input -> hidden -> output), got "
        f"{(degree, depth)}"
    )

    genome = deep_case.feedforward._make_hyper_genome()
    assert type(genome) is DefaultGenome and not isinstance(genome, RecurrentGenome), (
        "the executor must be a DefaultGenome (never a RecurrentGenome), got "
        f"{type(genome).__name__}"
    )
    assert genome.max_nodes == substrate.nodes_cnt, (
        f"max_nodes must be the substrate's {substrate.nodes_cnt} nodes, got "
        f"{genome.max_nodes}"
    )
    assert genome.max_conns == substrate.conns_cnt, (
        f"max_conns must be the substrate's {substrate.conns_cnt} connections, got "
        f"{genome.max_conns}"
    )
    assert deep_case.feedforward._static_bounds() == {
        "max_in_degree": degree,
        "max_depth": depth,
    }, (
        "the feedforward static bounds must be the hand-derived in-degree and depth, "
        f"got {deep_case.feedforward._static_bounds()!r}"
    )
    assert genome.max_in_degree == degree, (
        f"genome.max_in_degree must be the hand-derived {degree}, got "
        f"{genome.max_in_degree}"
    )
    assert genome.max_depth == depth, (
        f"genome.max_depth must be the hand-derived substrate depth {depth}, got "
        f"{genome.max_depth}"
    )
    assert not hasattr(genome, "activate_time"), (
        "a feedforward executor genome has no recurrent horizon, so it must not carry "
        "an activate_time attribute"
    )
    deepest = int(np.asarray(deep_case.schedule_ff.schedule.node_level).max())
    assert deepest == depth, (
        "the decoded schedule's deepest node level must equal the derived max_depth, "
        f"got level {deepest} against max_depth {depth}"
    )

    recurrent_genome = deep_case.recurrent.hyper_genome
    assert type(recurrent_genome) is RecurrentGenome, (
        f"the base variant must execute with RecurrentGenome, got "
        f"{type(recurrent_genome).__name__}"
    )
    assert set(deep_case.recurrent._static_bounds()) == {"max_in_degree"}, (
        "the recurrent base passes no max_depth (its genome ticks activate_time "
        "instead), so its static bounds must hold max_in_degree only, got "
        f"{deep_case.recurrent._static_bounds()!r}"
    )
    assert recurrent_genome.max_depth != depth, (
        "the recurrent genome keeps its own max_depth default rather than the substrate "
        f"depth: both are {depth}, so the two variants would be indistinguishable"
    )
    assert recurrent_genome.max_depth == 16, (
        f"RecurrentGenome's documented max_depth default is 16, got "
        f"{recurrent_genome.max_depth}"
    )
    assert recurrent_genome.activate_time == RECURRENT_TICKS, (
        f"the recurrent genome must receive the case's {RECURRENT_TICKS} ticks, got "
        f"{recurrent_genome.activate_time}"
    )


def test_feedforward_variant_rejects_a_recurrent_substrate(
    edge_case: _VariantCase,
) -> None:
    """A non-feedforward substrate is rejected, and the base class stays neutral.

    Oracle: the documented message of ``HyperNEAT._validate`` and the substrates' own
    ``connection_type``.

    Assertion: the recurrent ``FullSubstrate`` raises ``TensorNEATError`` naming the
    class, ``'feedforward'`` and the substrate's ``'recurrent'`` type, while the base
    class's ``None`` requirement constructs over both geometries.
    """
    recurrent_substrate = make_substrate()
    assert recurrent_substrate.connection_type == "recurrent", (
        "the harness's default FullSubstrate must be the recurrent geometry, got "
        f"{recurrent_substrate.connection_type!r}"
    )
    with pytest.raises(TensorNEATError) as excinfo:
        HyperNEATFeedForward(substrate=recurrent_substrate, neat=make_neat())
    message = str(excinfo.value)
    assert "HyperNEATFeedForward" in message, (
        f"the rejection must name the variant, got {message!r}"
    )
    assert "connection_type='feedforward'" in message, (
        f"the rejection must name the required feedforward type, got {message!r}"
    )
    assert "'recurrent'" in message, (
        f"the rejection must report the substrate's actual type, got {message!r}"
    )

    assert HyperNEAT._required_connection_type is None, (
        "the base class must not require a connection type, got "
        f"{HyperNEAT._required_connection_type!r}"
    )
    base_recurrent = HyperNEAT(substrate=recurrent_substrate, neat=make_neat())
    base_feedforward = HyperNEAT(substrate=edge_case.substrate, neat=make_neat())
    assert base_recurrent._required_connection_type is None
    assert base_feedforward._required_connection_type is None, (
        "the base class must construct over a feedforward substrate too, got "
        f"{base_feedforward._required_connection_type!r}"
    )


# --------------------------------------------------------------------------- #
# 2. same decode, same phenotype
# --------------------------------------------------------------------------- #


def test_both_variants_decode_one_payload_into_the_same_gene_tables(
    deep_case: _VariantCase,
) -> None:
    """The shared decode yields identical gene tables from either variant.

    Oracle: the base ``HyperNEAT`` decode of the *same* CPPN payload.

    Assertion: all four payload arrays are equal array-for-array, and neither schedule
    dropped a connection (``edge_overflow == 0`` on every individual) -- a decode that
    painted the weights differently, or a genome whose static window is too narrow,
    shows up here.
    """
    feedforward = deep_case.schedule_ff
    base = deep_case.schedule_base
    assert isinstance(feedforward, SubstrateSchedule)
    assert isinstance(base, SubstrateSchedule)
    for name in ("node_keys", "node_attrs", "conn_keys", "conn_attrs"):
        np.testing.assert_array_equal(
            getattr(feedforward, name),
            getattr(base, name),
            err_msg=(
                f"both variants must decode the same CPPN payload into the same {name}"
            ),
        )

    for label, schedule in (("feedforward", feedforward), ("recurrent", base)):
        overflow = np.asarray(schedule.schedule.edge_overflow)
        assert overflow.shape == (POP_SIZE,), (
            f"the {label} schedule must report one overflow count per individual, got "
            f"shape {overflow.shape}"
        )
        assert not np.any(overflow), (
            f"the {label} schedule dropped edges on a valid dense substrate: "
            f"{overflow.ravel().tolist()}"
        )
        assert not np.any(np.asarray(schedule.overflow)), (
            f"the {label} schedule's overflow alias must be all-False, got "
            f"{np.asarray(schedule.overflow).ravel().tolist()}"
        )


def test_forward_matches_the_recurrent_single_tick_on_a_two_layer_substrate(
    edge_case: _VariantCase,
) -> None:
    """With no hidden node, both variants and the hand computation agree exactly.

    Oracle: :func:`_two_layer_forward` over the decoded ``conn_attrs`` weights -- bias
    column appended last, one float32 product per connection, left-to-right
    accumulation, ``ACT.sigmoid``.

    Assertion: ``forward`` from both variants (the recurrent one at ``activate_time=1``,
    one tick) equals the oracle and each other with ``rtol=0, atol=0``: on a one-layer
    phenotype the level sweep and the single tick are the same arithmetic, so a
    difference is a real reassociation, not float noise.
    """
    assert EDGE_INPUTS.shape == (POP_SIZE, edge_case.substrate.num_inputs - 1), (
        "the inputs must carry one row per CPPN and one column per real substrate "
        f"input: expected {(POP_SIZE, edge_case.substrate.num_inputs - 1)}, got "
        f"{EDGE_INPUTS.shape}"
    )
    weights = np.asarray(edge_case.schedule_ff.conn_attrs)[..., 0]
    expected = _two_layer_forward(edge_case.substrate, weights, EDGE_INPUTS)

    feedforward_out = run_method(
        edge_case.feedforward,
        "forward",
        edge_case.schedule_ff,
        EDGE_INPUTS,
        state=edge_case.state,
    )
    recurrent_out = run_method(
        edge_case.recurrent,
        "forward",
        edge_case.schedule_base,
        EDGE_INPUTS,
        state=edge_case.state,
    )

    assert feedforward_out.shape == (POP_SIZE, 1), (
        f"forward must return one output per row, got {feedforward_out.shape}"
    )
    assert feedforward_out.dtype == np.float32, (
        f"forward must return float32, got {feedforward_out.dtype}"
    )
    np.testing.assert_allclose(
        feedforward_out,
        expected,
        rtol=0,
        atol=0,
        err_msg=(
            "the feedforward substrate forward must be the documented two-layer "
            "computation on the decoded weights, bit for bit"
        ),
    )
    np.testing.assert_allclose(
        recurrent_out,
        expected,
        rtol=0,
        atol=0,
        err_msg=(
            "the recurrent base at activate_time=1 must compute the same single tick "
            "as the feedforward level sweep on a no-hidden-node substrate"
        ),
    )
    np.testing.assert_array_equal(
        feedforward_out,
        recurrent_out,
        err_msg=(
            "both variants must agree bit for bit on a depth-1 substrate; if they do "
            "not, one of them reorders the window sum or masks a slot differently"
        ),
    )


def test_forward_appends_the_bias_as_the_last_input_column(
    bias_case: _VariantCase,
) -> None:
    """The phenotype input width is ``num_inputs - 1`` and the ones column is last.

    Oracle: :func:`_two_layer_forward` (bias appended) versus :func:`_bias_first_answer`
    (bias prepended), on a hand-painted weight row ``[0.5, -1.5, 2.0]`` over
    ``MLPSubstrate([3, 1])``.

    Assertion: the traced forward equals the appended-bias map exactly and differs from
    the prepended-bias map, so the check tells the two column orders apart instead of
    restating whichever one the implementation uses.
    """
    substrate = bias_case.substrate
    assert bias_case.feedforward.num_inputs == substrate.num_inputs - 1 == 2, (
        "HyperNEAT reports num_inputs = substrate.num_inputs - 1 (the trailing bias "
        f"removed): expected 2, got {bias_case.feedforward.num_inputs}"
    )
    conn_index = np.asarray(substrate.conn_index)
    assert conn_index.shape == (BIAS_WEIGHTS.size, 2), (
        "the hand weight row must cover every connection exactly once: expected "
        f"{BIAS_WEIGHTS.size} rows, got {conn_index.shape[0]}"
    )
    assert list(conn_index[:, 0]) == [0, 1, 2], (
        "MLPSubstrate([3, 1]) must wire inputs 0, 1 and the bias 2 to the output in "
        f"that row order, got sources {conn_index[:, 0].tolist()}"
    )

    weights = np.tile(BIAS_WEIGHTS.reshape(1, -1), (POP_SIZE, 1))
    hand_attrs = weights[:, :, None].astype(np.float32)
    schedule = bias_case.schedule_ff
    hand_schedule = SubstrateSchedule(
        schedule=schedule.schedule,
        node_keys=schedule.node_keys,
        node_attrs=schedule.node_attrs,
        conn_keys=schedule.conn_keys,
        conn_attrs=hand_attrs,
    )
    out = run_method(
        bias_case.feedforward,
        "forward",
        hand_schedule,
        BIAS_INPUTS,
        state=bias_case.state,
    )

    appended = _two_layer_forward(substrate, weights, BIAS_INPUTS)
    prepended = _bias_first_answer(substrate, weights, BIAS_INPUTS)
    np.testing.assert_allclose(
        out,
        appended,
        rtol=0,
        atol=0,
        err_msg=(
            "the ones column must be appended after the real inputs, so the answer is "
            "sigmoid(w0*x0 + w1*x1 + w2*1) on the hand-painted weights"
        ),
    )
    assert not np.array_equal(out, prepended), (
        "the bias-first answer must differ from the traced forward, otherwise this "
        "test cannot distinguish which column carries the bias: "
        f"traced {out.ravel().tolist()} vs bias-first {prepended.ravel().tolist()}"
    )


def test_forward_equals_the_prepared_split_on_a_deep_substrate(
    deep_case: _VariantCase,
) -> None:
    """``forward`` is exactly ``prepare_forward`` + ``forward_prepared``.

    Oracle: the one-shot ``forward`` of the same variant on the same schedule.

    Assertion: the composed halves (through :class:`_PreparedForward`, the only way the
    host-side prepared record stays inside one graph) reproduce the one-shot output bit
    for bit on ``MLPSubstrate([3, 2, 1])``, and that output is finite with the
    documented ``(P, num_outputs)`` shape.
    """
    out = run_method(
        deep_case.feedforward,
        "forward",
        deep_case.schedule_ff,
        DEEP_INPUTS,
        state=deep_case.state,
    )
    split = run_method(
        _PreparedForward(deep_case.feedforward),
        "forward_via_prepared",
        deep_case.schedule_ff,
        1,
        DEEP_INPUTS,
        state=deep_case.state,
    )

    assert out.shape == (POP_SIZE, deep_case.substrate.num_outputs), (
        "forward must return (P, substrate.num_outputs): expected "
        f"{(POP_SIZE, deep_case.substrate.num_outputs)}, got {out.shape}"
    )
    assert np.isfinite(out).all(), (
        f"a valid dense substrate must produce finite outputs, got {out.ravel().tolist()}"
    )
    np.testing.assert_array_equal(
        split,
        out,
        err_msg=(
            "forward must equal prepare_forward + forward_prepared bit for bit; a "
            "difference means the hoisted operands disagree with the ones a one-shot "
            "forward derives"
        ),
    )


def test_forward_does_not_depend_on_activate_time(
    deep_case: _VariantCase,
) -> None:
    """A feedforward ``forward`` cannot observe a recurrent horizon.

    Oracle: the same forward on the same schedule before and after writing the
    inherited ``activate_time`` attribute, plus the genome's own attribute surface.

    Assertion: the executor genome has no ``activate_time`` and
    ``_genome_extra_kwargs()`` is empty (so no graph can read one), and a forward on a
    fixed payload is bit-identical after ``activate_time`` is set to a wildly different
    value -- the value has no consumer on this variant.
    """
    fresh = make_hyper(
        substrate=deep_case.substrate, hyper_class=HyperNEATFeedForward
    )
    assert not hasattr(fresh.hyper_genome, "activate_time"), (
        "a DefaultGenome executor must not carry activate_time, so no kernel can read "
        "a recurrent horizon"
    )
    assert fresh._genome_extra_kwargs() == {}, (
        f"no genome kwargs may carry a horizon, got {fresh._genome_extra_kwargs()!r}"
    )

    before = run_method(
        fresh, "forward", deep_case.schedule_ff, DEEP_INPUTS, state=deep_case.state
    )
    fresh.activate_time = 99
    after = run_method(
        fresh, "forward", deep_case.schedule_ff, DEEP_INPUTS, state=deep_case.state
    )
    np.testing.assert_array_equal(
        after,
        before,
        err_msg=(
            "the feedforward forward must not change when the inherited activate_time "
            "attribute changes; a difference means some kernel reads it"
        ),
    )


# --------------------------------------------------------------------------- #
# 3. population-axis independence of the feedforward decode
# --------------------------------------------------------------------------- #


def test_transform_decodes_one_individual_exactly_as_its_population_row(
    deep_case: _VariantCase,
) -> None:
    """One CPPN decoded alone is its row of the ``P = 4`` batch (the ``show`` contract).

    Oracle: the full-population decode of the same variant on the same payload, and the
    substrate's own ``nodes``/``conn_index`` tables.

    Assertion: every payload field of the one-individual decode equals row ``2`` of the
    batch decode array-for-array, and the decoded keys are the substrate tables
    broadcast over the population axis -- so ``Pipeline.show`` can decode a single
    individual without changing its phenotype.
    """
    row = 2
    payload = deep_case.payload
    single = type(payload)(*(field[row : row + 1] for field in payload))
    assert single.node_keys.shape[0] == 1, (
        f"the sliced payload must be one individual, got {single.node_keys.shape[0]}"
    )
    schedule = run_method(
        deep_case.feedforward, "transform", single, state=deep_case.state
    )

    for name in ("node_keys", "node_attrs", "conn_keys", "conn_attrs"):
        np.testing.assert_array_equal(
            getattr(schedule, name)[0],
            getattr(deep_case.schedule_ff, name)[row],
            err_msg=(
                f"decoding individual {row} alone must reproduce its {name} row of the "
                "population decode; the query axis must not leak between individuals"
            ),
        )

    substrate = deep_case.substrate
    np.testing.assert_array_equal(
        schedule.node_keys[0],
        np.asarray(substrate.nodes),
        err_msg=(
            "the decoded node keys must be the substrate's node table, one row per "
            "individual"
        ),
    )
    np.testing.assert_array_equal(
        schedule.conn_keys[0],
        np.asarray(substrate.conn_index),
        err_msg=(
            "the decoded connection keys must be the substrate's conn_index table, one "
            "row per individual"
        ),
    )


# --------------------------------------------------------------------------- #
# 4. the whole generation graph builds and runs
# --------------------------------------------------------------------------- #


def test_one_generation_step_populates_the_species_table(
    tiny_pipeline: Pipeline,
) -> None:
    """One ``Pipeline.step`` runs the feedforward generation graph end to end.

    Oracle: ``XOR``'s fitness definition (``-mse`` over four samples against
    ``[[0], [1], [1], [0]]``, so every value is finite and ``<= 0``), the configured
    population size, and the species bookkeeping invariants (a live key per allocated
    slot, one member per genome, every genome mapped to a live slot).

    Assertion: a one-generation step returns a finite ``(4,)`` fitness vector, advances
    the generation counter once, and leaves a species table whose live slot holds all
    four individuals.  This is the cheap proof that the variant's whole
    ``transform``/``evaluate``/``tell`` graph builds and runs; the multi-generation
    runs of the variant live in the examples and the integration suite.
    """
    state = tiny_pipeline.setup()
    state, previous_pop, fitness = tiny_pipeline.step(state)

    assert fitness.shape == (POP_SIZE,), (
        f"one step must report one fitness per CPPN, got {fitness.shape}"
    )
    assert fitness.dtype == np.float32, (
        f"fitness must be float32, got {fitness.dtype}"
    )
    assert np.isfinite(fitness).all(), (
        f"a valid substrate evaluation must be finite, got {fitness.tolist()}"
    )
    assert np.all(fitness <= 0.0), (
        "XOR fitness is a negated mse, so every value must be <= 0, got "
        f"{fitness.tolist()}"
    )
    assert int(to_numpy(state.generation)) == 1, (
        "one step must advance the generation counter from 0 to 1, got "
        f"{to_numpy(state.generation)!r}"
    )
    assert previous_pop.node_keys.shape[0] == POP_SIZE, (
        "the evaluated CPPN population must be the configured 4 individuals, got "
        f"{previous_pop.node_keys.shape[0]}"
    )

    species = state.species
    keys = np.asarray(to_numpy(species.species_keys))
    members = np.asarray(to_numpy(species.member_count))
    idx2species = np.asarray(to_numpy(species.idx2species))
    live = np.isfinite(keys)
    assert live.any(), (
        f"tell must leave at least one live species, got species_keys {keys.tolist()}"
    )
    assert int(members[live].sum()) == POP_SIZE, (
        "every genome must be counted by exactly one live species: expected "
        f"{POP_SIZE} members, got {members[live].sum()!r} over keys {keys.tolist()}"
    )
    assert np.all(np.isfinite(idx2species)), (
        f"every genome must be assigned to a species, got idx2species {idx2species}"
    )
    assert live[idx2species.astype(int)].all(), (
        "every genome must point at a live species slot, got idx2species "
        f"{idx2species.tolist()} against keys {keys.tolist()}"
    )
    assert float(to_numpy(species.next_species_key)) >= 1.0, (
        "at least one species key must have been allocated, got next_species_key "
        f"{to_numpy(species.next_species_key)!r}"
    )
