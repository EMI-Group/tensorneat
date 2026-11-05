"""End-to-end ``Pipeline`` runs of both HyperNEAT variants.

What this module proves
-----------------------
``src/tensorneat/algorithm/hyperneat/`` had no whole-``Pipeline`` test anywhere in the
tree: the two shipped scripts (``examples/func_fit/xor_hyperneat.py`` and
``xor_hyperneat_feedforward.py``) assert nothing, so nothing checked that a real
``Pipeline`` -- ``setup`` -> ``step`` -> ``tell`` -> ``transform`` -> ``evaluate`` ->
``show`` -- survives the recurrent and the feedforward decode at the same time.
``./test_substrate.py`` pins the host geometry and ``./test_hyperneat_weights.py`` the
weight map; this module is the only one that runs the assembled generation graph.

Every oracle here is independent of the code under test:

* ``_assert_finite_fitness`` restates ``FuncFit.evaluate``'s documented output -- a
  ``(P,)`` ``float32`` vector of ``finalize_fitness(-loss)``, therefore finite and
  non-positive for the non-negative ``mse`` loss of ``x[0] ** 2`` on ``[-1, 1]``.
* the "population advanced" test compares the *four-array payload* of generation 1 and
  generation 2 field by field and absorbs nothing: at least one individual must differ in
  at least one array, otherwise ``tell`` handed the parents back unchanged.
* the determinism test re-derives the documented ``(seed, config, shape)`` contract by
  running a second ``setup()`` + two steps on the *same* pipeline and comparing every
  array; ``test/unit/CONTEXT.md``'s reproducibility rule is exactly this statement.
* the ``network_dict`` test re-derives the expected node count from the payload itself
  (``node_keys != I_INF``) and cross-checks the structural links the dict claims --
  connection endpoints inside the node set, ``topo_order`` covering every node, the
  output rows inside ``useful_nodes``.
* the feedforward sizing test restates the `MLPSubstrate(layers=[2, 2, 1])` geometry as a
  closed form (``nodes = sum(layers)``, ``conns = 2*2 + 2*1``) and compares it with both
  the executor genome's statics and the rows of the decoded ``SubstrateSchedule``.
* the recorded-best test recomputes ``max`` over ``fitness_history`` -- the quantity
  ``_analyse`` logs from the fitness vector -- and compares it with ``best_fitness``, the
  quantity it selects from the same vector.

Two gotchas that are load-bearing here
-------------------------------------
1. **``np.array_equal`` is unusable without ``equal_nan=True``.** ``node_attrs`` /
   ``conn_attrs`` are ``NaN``-padded, so one array compared with itself reports "not
   equal" (a false alarm on determinism) and, in the per-individual "did it change?" walk,
   every padded individual reports "changed" (a false positive). Every comparison below
   passes ``equal_nan=True``.
2. **``evoxc.Tensor`` has no ``__array__``**, so a payload is read through
   ``support.runner.to_numpy``; ``tensorneat.common.tools.to_numpy`` is the one-*leaf*
   accessor and stacks a payload container instead of rebuilding it.

No network, no GPU, ``backend="numpy"``, ``pop_size = 4`` and a two-generation run, so the
whole module costs a few seconds and one graph build per ``Pipeline`` (the build dominates).
"""
from __future__ import annotations

import numpy as np
from support.runner import to_numpy
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.algorithm.hyperneat import (
    FullSubstrate,
    HyperNEAT,
    HyperNEATFeedForward,
    MLPSubstrate,
)
from tensorneat.common import ACT, I_INF
from .hyperneat_kernel import (
    ACTIVATE_TIME,
    HIDDEN_COORS,
    INPUT_COORS,
    OUTPUT_COORS,
    POP_SIZE,
    QUERY_WIDTH,
    SPECIES_SIZE,
    SURVIVAL_THRESHOLD,
    run_method,
)

#: Seed of every ``Pipeline`` here -- the first third of the documented
#: ``(seed, config, shape)`` reproducibility triple.
SEED = 42
#: Generations per run.  Two is the smallest number that exercises ``tell`` twice: the
#: first step speciates the initial population, the second re-speciates a mutated one.
GENERATION_LIMIT = 2
#: Rows of the ``CustomFuncFit`` dataset (and therefore ``show``'s report lines).
NUM_SAMPLES = 8
#: Layer sizes of the feedforward substrate, bias included in ``layers[0]``: one real
#: input + bias -> 2 hidden -> 1 output, matching ``HyperNEAT.num_inputs == 1``.
MLP_LAYERS = (2, 2, 1)
#: Coordinate box of the feedforward substrate.
MLP_COOR_RANGE = (-1.0, 1.0, -1.0, 1.0)
#: Field names of a gene payload, in ``GenomeTensors`` order.
PAYLOAD_FIELDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")


def target(x):
    """The problem to fit: ``x[0] ** 2``, non-negative on the whole sampling box."""
    return x[0] ** 2


# --------------------------------------------------------------------------- #
# builders -- the shipped wiring of ``examples/func_fit/xor_hyperneat*.py``, shrunk
# --------------------------------------------------------------------------- #
def _cppn():
    """The CPPN algorithm: one network per substrate-connection query.

    ``num_inputs`` is the *query* width (both endpoints' coordinates, ``2 * 2 = 4``), not
    the substrate's input count, and the single output is that connection's weight.
    """
    return algorithm.NEAT(
        pop_size=POP_SIZE,
        species_size=SPECIES_SIZE,
        survival_threshold=SURVIVAL_THRESHOLD,
        genome=genome.DefaultGenome(
            num_inputs=QUERY_WIDTH,
            num_outputs=1,
            output_transform=ACT.tanh,
        ),
    )


def _recurrent_hyper():
    """``HyperNEAT`` over the harness's recurrent ``FullSubstrate`` (2 in / 1 hidden / 1 out).

    The substrate is recurrent (its single hidden neuron carries a self-loop), so the
    phenotype is tick-unrolled ``activate_time`` times.
    """
    return HyperNEAT(
        substrate=FullSubstrate(
            input_coors=INPUT_COORS,
            hidden_coors=HIDDEN_COORS,
            output_coors=OUTPUT_COORS,
        ),
        neat=_cppn(),
        activation=ACT.tanh,
        activate_time=ACTIVATE_TIME,
        output_transform=ACT.sigmoid,
    )


def _feedforward_hyper():
    """``HyperNEATFeedForward`` over ``MLPSubstrate(layers=[2, 2, 1])`` -- no tick loop."""
    return HyperNEATFeedForward(
        substrate=MLPSubstrate(layers=list(MLP_LAYERS), coor_range=MLP_COOR_RANGE),
        neat=_cppn(),
        activation=ACT.tanh,
        output_transform=ACT.sigmoid,
    )


def _pipeline(hyper, generation_limit=GENERATION_LIMIT, fitness_target=1.0):
    """Wrap one algorithm in the shipped ``Pipeline`` wiring.

    ``fitness_target=1.0`` is unreachable for this problem by construction: the fitness is
    ``-loss <= 0``, so ``auto_run`` always executes its whole ``generation_limit``.
    ``is_detailed=False`` keeps the per-generation log out of the captured output.
    """
    return Pipeline(
        algorithm=hyper,
        problem=problem.CustomFuncFit(
            target,
            low_bounds=[-1.0],
            upper_bounds=[1.0],
            method="sample",
            num_samples=NUM_SAMPLES,
            seed=0,
        ),
        seed=SEED,
        generation_limit=generation_limit,
        fitness_target=fitness_target,
        is_detailed=False,
    )


# --------------------------------------------------------------------------- #
# oracles (hand-written, never read off the object under test)
# --------------------------------------------------------------------------- #
def _payload(state):
    """``state.pop`` as host numpy -- the four-array batch-first gene payload."""
    return to_numpy(state.pop)


def _assert_same_payload(left, right, message):
    """Assert two payloads agree array-for-array, NaN padding included."""
    for name in PAYLOAD_FIELDS:
        expected = getattr(left, name)
        actual = getattr(right, name)
        assert expected.shape == actual.shape, (
            f"{message}: {name} shape {actual.shape} != {expected.shape}"
        )
        assert np.array_equal(expected, actual, equal_nan=True), (
            f"{message}: {name} differs between the two payloads"
        )


def _assert_finite_fitness(fitness, label):
    """Assert ``FuncFit.evaluate``'s documented output: a finite, non-positive ``(P,)``."""
    assert fitness.shape == (POP_SIZE,), (
        f"{label}: fitness must be one value per individual, got {fitness.shape}"
    )
    assert fitness.dtype == np.float32, (
        f"{label}: finalize_fitness returns float32, got {fitness.dtype}"
    )
    assert np.all(np.isfinite(fitness)), (
        f"{label}: fitness must be finite (NaN fitness maps to -inf and means the "
        f"phenotype produced a NaN prediction), got {fitness}"
    )
    assert np.all(fitness <= 0.0), (
        f"{label}: fitness is the negated non-negative mse loss, so it must never be "
        f"positive, got {fitness}"
    )


# --------------------------------------------------------------------------- #
# 1. recurrent HyperNEAT: two generations, population advanced, deterministic
# --------------------------------------------------------------------------- #
def test_recurrent_two_generations_advance_the_population():
    """``setup`` is host-only, two steps give finite fitness, and ``tell`` moves the genes."""
    pipeline = _pipeline(_recurrent_hyper())
    hyper = pipeline.algorithm

    state = pipeline.setup()
    # Construction and setup are host-only: no graph exists until the first step.
    assert pipeline.builds == 0, "setup() must not build an executable"
    assert pipeline.shape_signature is None, "setup() must not pin a shape signature"
    # The phenotype has ONE real input: FullSubstrate's last input coordinate is the bias.
    assert hyper.num_inputs == 1
    assert hyper.num_outputs == 1
    assert pipeline.problem.input_shape == (1,)
    assert pipeline.problem.output_shape == (1,)
    # The CPPN is queried with both endpoints' coordinates, so its input width is the
    # query width -- not the substrate's input count.
    assert hyper.neat.genome.num_inputs == QUERY_WIDTH
    assert hyper.substrate.query_coors.shape[1] == QUERY_WIDTH
    setup_pop = _payload(state)
    assert setup_pop.node_keys.shape[0] == POP_SIZE, (
        "the payload is batch-first: population on axis 0"
    )
    assert int(to_numpy(state.generation)) == 0

    state1, evaluated1, fitness1 = pipeline.step(state)
    assert pipeline.builds == 1, "one executable per shape signature"
    _assert_finite_fitness(fitness1, "generation 1")
    # ``step`` returns the population that produced this fitness -- the setup population.
    _assert_same_payload(
        evaluated1, setup_pop, "step() must return the population it evaluated"
    )

    state2, evaluated2, fitness2 = pipeline.step(state1)
    assert pipeline.builds == 1, "a second generation of the same shape must reuse it"
    _assert_finite_fitness(fitness2, "generation 2")
    _assert_same_payload(
        evaluated2, _payload(state1), "step() must return the population it evaluated"
    )
    assert int(to_numpy(state2.generation)) == GENERATION_LIMIT

    # The population advanced: at least one individual differs in at least one array
    # (NaN padding included, so an untouched individual cannot look changed).
    before = _payload(state1)
    after = _payload(state2)
    moved = [
        index
        for index in range(POP_SIZE)
        if any(
            not np.array_equal(
                getattr(before, name)[index],
                getattr(after, name)[index],
                equal_nan=True,
            )
            for name in PAYLOAD_FIELDS
        )
    ]
    assert moved, (
        "tell() returned the parents unchanged: no individual of generation 2 differs "
        "from generation 1 in any of the four payload arrays"
    )

    # Determinism: a second setup() of the SAME pipeline reproduces both generations.
    replay = pipeline.setup()
    replay1, _, replay_fitness1 = pipeline.step(replay)
    replay2, _, replay_fitness2 = pipeline.step(replay1)
    assert pipeline.builds == 1, "the rebuilt graph is still one shape signature"
    _assert_same_payload(
        _payload(replay2),
        after,
        "the same (seed, config, shape) must reproduce the population bit for bit",
    )
    _assert_same_payload(_payload(replay1), before, "generation 1 must reproduce too")
    assert np.array_equal(replay_fitness1, fitness1), (
        "the same seed must reproduce generation 1's fitness exactly"
    )
    assert np.array_equal(replay_fitness2, fitness2), (
        "the same seed must reproduce generation 2's fitness exactly"
    )


# --------------------------------------------------------------------------- #
# 2. the best CPPN has a usable host ``network_dict`` and a printable report
# --------------------------------------------------------------------------- #
def test_best_genome_network_dict_and_report(capsys):
    """The recorded best is a one-individual batch with a consistent host graph."""
    pipeline = _pipeline(_recurrent_hyper())
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)

    assert best is not None, "two generations of a finite-fitness run must record a best"
    assert pipeline.best_genome is best, "auto_run returns the recorded best genome"
    cppn = pipeline._genome()
    # ``_genome`` falls back to the inner ``NEAT``'s genome: the payload belongs to the
    # CPPN, never to the fixed substrate.
    assert cppn is pipeline.algorithm.neat.genome
    assert cppn is not pipeline.algorithm, (
        "a wrapper algorithm carries no genome of its own"
    )
    assert best.node_keys.shape == (1, cppn.max_nodes), (
        "best_genome is state.pop at P = 1: a one-individual batch"
    )
    assert best.conn_keys.shape[0] == 1

    network = cppn.network_dict(state, best)
    nodes = network["nodes"]
    conns = network["conns"]
    assert nodes, "the CPPN must have at least its input and output nodes"
    assert conns, "a two-generation CPPN has at least one connection"
    live = int(np.count_nonzero(np.asarray(best.node_keys)[0] != I_INF))
    assert len(nodes) == live, (
        "network_dict must cover exactly the payload's live rows: "
        f"{len(nodes)} dicts for {live} keys"
    )
    assert len(nodes) >= cppn.num_inputs + cppn.num_outputs
    assert set(range(cppn.num_inputs)) <= set(nodes), "every input key is present"

    node_keys = set(nodes)
    for edge, entry in conns.items():
        assert set(edge) <= node_keys, (
            f"connection {edge} references a node outside the node set {sorted(node_keys)}"
        )
        # The dict entry and the tuple key must name the same endpoints.
        assert (entry["in"], entry["out"]) == edge
    assert set(network["topo_order"]) == node_keys, (
        "topological_sort must order every node exactly once"
    )
    assert set(network["useful_nodes"]) <= node_keys
    assert set(cppn.get_output_idx()) <= set(network["useful_nodes"]), (
        "an output node always reaches an output"
    )

    # ``show`` is the documented reporting tail: one line per dataset row plus the loss.
    pipeline.show(state, best)
    report = capsys.readouterr().out
    assert report.strip(), "pipeline.show printed nothing"
    assert report.count("input:") == NUM_SAMPLES, (
        f"show must print one line per sample ({NUM_SAMPLES}), got:\n{report}"
    )
    assert "loss:" in report


# --------------------------------------------------------------------------- #
# 3. feedforward HyperNEAT: two generations, finite fitness, repeatable
# --------------------------------------------------------------------------- #
def test_feedforward_two_generations_are_finite_and_repeatable():
    """The MLP-substrate variant runs through the same pipeline cycle, bit-reproducibly."""
    pipeline = _pipeline(_feedforward_hyper())
    hyper = pipeline.algorithm
    assert hyper.substrate.connection_type == "feedforward"
    assert hyper.num_inputs == 1

    state = pipeline.setup()
    state1, _, fitness1 = pipeline.step(state)
    _assert_finite_fitness(fitness1, "feedforward generation 1")
    state2, _, fitness2 = pipeline.step(state1)
    _assert_finite_fitness(fitness2, "feedforward generation 2")
    assert int(to_numpy(state2.generation)) == GENERATION_LIMIT
    # The best individual is at least as good as the worst (maximised fitness).
    assert np.max(fitness2) >= np.min(fitness2)
    assert np.max(fitness1) >= np.min(fitness1)

    replay = pipeline.setup()
    replay1, _, replay_fitness1 = pipeline.step(replay)
    replay2, _, replay_fitness2 = pipeline.step(replay1)
    _assert_same_payload(
        _payload(replay2),
        _payload(state2),
        "the same (seed, config, shape) must reproduce the feedforward population",
    )
    assert np.array_equal(replay_fitness1, fitness1)
    assert np.array_equal(replay_fitness2, fitness2)


# --------------------------------------------------------------------------- #
# 4. the feedforward executor is sized by the MLP substrate, and the best is the max
# --------------------------------------------------------------------------- #
def test_feedforward_phenotype_is_sized_by_the_mlp_substrate():
    """The decoded ``SubstrateSchedule`` is dense at exactly the substrate's geometry."""
    pipeline = _pipeline(_feedforward_hyper())
    hyper = pipeline.algorithm
    substrate = hyper.substrate
    state = pipeline.setup()

    # Closed form of MLPSubstrate(layers=[2, 2, 1]): one node per layer entry, one
    # connection per adjacent pair of entries.
    expected_nodes = sum(MLP_LAYERS)
    expected_conns = MLP_LAYERS[0] * MLP_LAYERS[1] + MLP_LAYERS[1] * MLP_LAYERS[2]
    assert substrate.nodes_cnt == expected_nodes
    assert substrate.conns_cnt == expected_conns
    assert hyper.hyper_genome.max_nodes == expected_nodes
    assert hyper.hyper_genome.max_conns == expected_conns

    schedule = run_method(hyper, "transform", _payload(state), state=state)
    assert schedule.node_keys.shape == (POP_SIZE, expected_nodes)
    assert schedule.conn_keys.shape == (POP_SIZE, expected_conns, 2)
    assert schedule.conn_attrs.shape == (POP_SIZE, expected_conns, 1)
    # A hypercube decode is dense by construction: no absent row and nothing dropped.
    assert np.all(np.asarray(schedule.node_keys) != I_INF)
    assert np.all(np.asarray(schedule.conn_keys)[..., 0] != I_INF)
    assert np.all(np.asarray(schedule.schedule.edge_overflow) == 0)

    state, best = pipeline.auto_run(state)
    assert best is not None
    recorded = max(value for _, value in pipeline.fitness_history)
    assert pipeline.best_fitness == recorded, (
        "best_fitness must be the largest per-generation max recorded in fitness_history"
    )
    assert pipeline.best_fitness <= 0.0
    # The recorded best is a CPPN individual -- the evolved payload -- so it is sized by
    # the CPPN genome, never by the fixed substrate executor.
    cppn = pipeline._genome()
    assert cppn is hyper.neat.genome
    assert best.node_keys.shape == (1, cppn.max_nodes)
