"""Parity fixtures: the frozen legacy library, the rewrite, and one genome description.

Everything in this file exists to make one sentence true for every module of
``./``: *the two libraries are fed the same host-built genome and the same host
data, so only the kernels differ.*

## Lazy imports (why nothing is imported at module scope)

``jax`` and ``tensorneat_legacy`` are optional (the ``bench`` extra), and
``import tensorneat`` currently raises ``NotImplementedError`` because ``src/`` is
a skeleton.  A module-scope import of either would turn ``pytest test/parity`` —
and the repo-wide run — into a collection error instead of a named skip.  So
:func:`legacy` calls ``pytest.importorskip`` **inside the fixture body** and
:func:`new` goes through :func:`support.lazy.require`, which converts only the
pending-implementation ``NotImplementedError`` into a skip.  No other suite may
import ``jax`` at module scope; this directory is the one place allowed to
(``../CONTEXT.md``).

The legacy library additionally needs its payload as ``jax`` arrays rather than
numpy arrays: under JAX 0.12 its forward loops with ``lax.fori_loop`` and indexes
``nodes[i, 0]`` with the traced loop index, so a plain numpy ``nodes`` raises
``TracerArrayConversionError``.  :func:`legacy_transformed` therefore wraps the
converted payload with ``jnp.asarray`` — verified against the real legacy code.

## One host-side description, two genomes

:func:`genome_pair` is the only place a parity genome is built.  Its arguments
are the *host-side description* (sizes, the ACT/AGG option **names**, the
transforms); it resolves those names against each library's own registry
(``tensorneat.common.ACT`` vs ``tensorneat_legacy.common.ACT``) and returns a
:class:`GenomePair`.  Three rules are enforced there and documented on the
fixture:

* both sides get ``init_hidden_layers`` from
  :func:`convert.legacy_init_hidden_layers`, so the legacy ``output_idx`` is the
  rewrite's output rows;
* the rewrite gets ``max_in_degree = convert.max_in_degree_for(max_nodes)``, so
  its incoming window is at least as wide as the legacy ``max_nodes`` column;
* ACT/AGG are single-option lists, so option ids agree (both are positions in the
  configured list) and no unselected option can poison the trace.

## Schedule and forward helpers

:func:`new_schedule` derives the rewrite's ``Transformed`` through the public
``genome.transform`` (never by naming a field of ``derive_schedule``) and asserts
``edge_overflow == 0`` before any test can compare values: a parity payload whose
schedule silently dropped an edge must fail, not drift.  :func:`new_forward` and
:func:`legacy_forward` evaluate the same payload on one input batch;
:func:`legacy_transformed` is exposed separately because ``test_distance`` /
``test_fitness`` / ``test_grad`` need the legacy schedule itself.

The ``Transformed`` namedtuple is passed **field by field** and rebuilt inside the
traced closure, exactly as in ``../unit/genome/conftest.py``: the shared runner
classifies only top-level numpy leaves as data, so a namedtuple argument would be
baked into the trace as a static value whose fields are plain numpy arrays.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import pytest

#: This directory, so ``import convert`` works no matter how pytest was invoked.
_PARITY_DIR = Path(__file__).resolve().parent
if str(_PARITY_DIR) not in sys.path:
    sys.path.insert(0, str(_PARITY_DIR))

import convert as conversion  # noqa: E402 - needs the sys.path guard above

#: Parity layout shared by the payload builders of ``support.payloads`` and the
#: genome pair: 2 inputs, 1 output, 8 node rows, 16 connection rows.
#: ``16 >= convert.legacy_initial_edge_count(8, 2, 1) == 15``, which is what the
#: legacy genome constructor requires for its dense initial network.
DEFAULT_NUM_INPUTS = 2
DEFAULT_NUM_OUTPUTS = 1
DEFAULT_MAX_NODES = 8
DEFAULT_MAX_CONNS = 16

#: The option names every parity genome configures (single-option lists on both
#: sides).  Both registries have the same canonical order, so the *names* are the
#: portable description and the ids they resolve to are positions in one-element
#: lists — identical on both sides by construction.
DEFAULT_ACTIVATION = "sigmoid"
DEFAULT_AGGREGATION = "sum"

#: Transforms usable by both sides.  Operator-only Python callables so one lambda
#: is a valid evoxc-graph transform *and* a valid ``jax`` transform: ``*`` and ``+``
#: dispatch to either tensor type, while ``numpy``/``jnp`` functions would not.
TRANSFORMS: dict[str, Callable[[Any], Any]] = {
    "identity": lambda x: x,
    "affine": lambda x: x * 2.0 + 1.0,
}


class GenomePair(NamedTuple):
    """The same genome, configured in both libraries, plus its two states.

    Attributes:
        num_inputs: Input width ``I`` (keys/rows ``0 .. I-1`` on both sides).
        num_outputs: Output width ``O`` (the last ``O`` rows/keys on both sides).
        max_nodes: Static node capacity ``N``.
        max_conns: Static connection capacity ``E``.
        max_in_degree: The rewrite's window width ``K``, equal to ``max_nodes``.
        new_genome: ``tensorneat.genome.DefaultGenome`` instance.
        new_state: ``new_genome.setup(tensorneat.common.State())``.
        legacy_genome: ``tensorneat_legacy.genome.DefaultGenome`` instance.
        legacy_state: ``legacy_genome.setup(tensorneat_legacy.common.State())``.
    """

    num_inputs: int
    num_outputs: int
    max_nodes: int
    max_conns: int
    max_in_degree: int
    new_genome: Any
    new_state: Any
    legacy_genome: Any
    legacy_state: Any


# --------------------------------------------------------------------------- inputs
@pytest.fixture(scope="module")
def legacy():
    """The frozen pre-rewrite JAX library, imported as ``tensorneat_legacy``.

    Returns:
        The ``tensorneat_legacy`` package: ``legacy.genome.DefaultGenome`` and the
        gene/operator classes, ``legacy.common.State``/``ACT``/``AGG`` and
        ``legacy.problem.XOR``/``XOR3d``.

    Notes:
        Both imports are skipped-not-errored: ``jax`` when the ``bench`` extra is
        absent, ``tensorneat_legacy`` when ``benchmarks/legacy_jax`` is missing
        (``../conftest.py`` puts it on ``sys.path`` when it exists).  The import
        happens **here**, never at module scope, so a numpy-only environment
        reports a skip instead of a collection error.
    """
    pytest.importorskip("jax", reason="parity needs the `bench` extra (jax)")
    return pytest.importorskip(
        "tensorneat_legacy",
        reason="parity needs benchmarks/legacy_jax on sys.path (see ../conftest.py)",
    )


@pytest.fixture(scope="module")
def new():
    """The rewrite under test: the ``tensorneat`` package.

    Returns:
        The imported package, so a test reads ``new.genome.DefaultGenome``,
        ``new.common.State``/``ACT``/``AGG``/``Transformed`` and
        ``new.problem.XOR``/``XOR3d``.

    Notes:
        Deferred through :func:`support.lazy.require`: while ``src/`` is a
        skeleton ``import tensorneat`` raises ``NotImplementedError`` from
        ``common/state.py``, and that — and only that — becomes a skip naming the
        missing slice.  An ``ImportError`` still surfaces.
    """
    from support.lazy import require

    return require("tensorneat")


@pytest.fixture(scope="module")
def payloads():
    """The hand-built gene payloads of ``../../support/payloads.py``.

    Returns:
        The ``support.payloads`` module: ``new_payload``, ``set_node``,
        ``set_conn``, ``Payload``, ``chain_payload``, ``skip_payload``,
        ``isolated_payload`` and ``I_INF``.

    Notes:
        Every parity topology is written by hand — never drawn — so the legacy
        side can be built from the very same gene rows.
    """
    from support.lazy import require

    return require("support.payloads")


@pytest.fixture(scope="module")
def runner():
    """The one sanctioned bridge to ``evoxc`` (``../../support/runner.py``).

    Returns:
        The module holding ``run_defn``, ``build_defn``, ``run_graph`` and
        ``to_numpy``.  Nothing in this directory calls ``evoxc`` directly.
    """
    from support.lazy import require

    return require("support.runner")


@pytest.fixture(scope="module")
def convert():
    """The conversion helpers of ``./convert.py`` (the column-layout spec).

    Returns:
        The module: ``to_legacy``, ``from_legacy``, ``max_in_degree_for``,
        ``legacy_init_hidden_layers``, ``legacy_initial_edge_count``, ``I_INF``,
        ``KEY_MAX`` and the legacy key widths.
    """
    return conversion


# --------------------------------------------------------------------------- genomes
@pytest.fixture(scope="module")
def genome_pair(new, legacy, convert):
    """Build the *same* genome in both libraries from one host-side description.

    Returns:
        ``build(num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16, *,
        activation="sigmoid", aggregation="sum", input_transform=None,
        output_transform=None) -> GenomePair``.

    Notes:
        Three conversions happen here, and each one is a parity rule:

        * ``init_hidden_layers = convert.legacy_init_hidden_layers(...)`` on both
          sides.  The legacy ``output_idx`` comes from its layer list, the
          rewrite's from its row/key invariant; the dense hidden layer that fills
          the gap is what makes "the output node" mean the same key on both sides.
        * ``max_in_degree = convert.max_in_degree_for(max_nodes)``: the rewrite's
          incoming window is ``max_nodes`` slots wide, like the legacy dense
          column, so ``Transformed.edge_overflow`` is ``0`` for every payload.
        * ``max_depth = max_nodes``: the unroll bound must not truncate a parity
          network — a tail left ``NaN`` by the depth bound would look like a value
          mismatch.
        * ACT/AGG are resolved **by name** from each library's own registry and
          configured as **one-element lists**, so an option id is ``0`` on both
          sides and no non-total option is evaluated into an unselected branch.
          A transform is looked up in :data:`TRANSFORMS` (operator-only callables
          valid for both libraries) or passed in explicitly.

    Raises:
        ValueError: If the requested ``max_conns`` cannot hold the legacy dense
            initial network (``convert.legacy_initial_edge_count``), or if the
            transform name is unknown.
    """
    transforms = TRANSFORMS

    def resolve_transform(transform: Any) -> Any:
        if transform is None or callable(transform):
            return transform
        try:
            return transforms[transform]
        except KeyError:  # a configuration mistake in the test, not a data issue
            raise ValueError(
                f"unknown transform {transform!r}; expected one of "
                f"{sorted(transforms)} or a callable"
            ) from None

    def build(
        num_inputs: int = DEFAULT_NUM_INPUTS,
        num_outputs: int = DEFAULT_NUM_OUTPUTS,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_conns: int = DEFAULT_MAX_CONNS,
        *,
        activation: str = DEFAULT_ACTIVATION,
        aggregation: str = DEFAULT_AGGREGATION,
        input_transform: Any = None,
        output_transform: Any = None,
    ) -> GenomePair:
        hidden = convert.legacy_init_hidden_layers(max_nodes, num_inputs, num_outputs)
        initial_edges = convert.legacy_initial_edge_count(
            max_nodes, num_inputs, num_outputs
        )
        if max_conns < initial_edges:
            raise ValueError(
                f"max_conns={max_conns} cannot hold the legacy dense initial "
                f"network of {initial_edges} edges for this layout"
            )
        new_input_transform = resolve_transform(input_transform)
        new_output_transform = resolve_transform(output_transform)

        new_node = new.genome.DefaultNode(
            aggregation_options=[getattr(new.common.AGG, aggregation)],
            activation_options=[getattr(new.common.ACT, activation)],
        )
        new_genome = new.genome.DefaultGenome(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_nodes=max_nodes,
            max_conns=max_conns,
            node_gene=new_node,
            conn_gene=new.genome.DefaultConn(),
            mutation=new.genome.DefaultMutation(),
            crossover=new.genome.DefaultCrossover(),
            distance=new.genome.DefaultDistance(),
            input_transform=new_input_transform,
            output_transform=new_output_transform,
            init_hidden_layers=hidden,
            max_in_degree=convert.max_in_degree_for(max_nodes),
            max_depth=max_nodes,
        )

        legacy_genome = legacy.genome.DefaultGenome(
            num_inputs,
            num_outputs,
            max_nodes,
            max_conns,
            legacy.genome.DefaultNode(
                aggregation_options=getattr(legacy.common.AGG, aggregation),
                activation_options=getattr(legacy.common.ACT, activation),
            ),
            legacy.genome.DefaultConn(),
            legacy.genome.DefaultMutation(),
            legacy.genome.DefaultCrossover(),
            legacy.genome.DefaultDistance(),
            output_transform=new_output_transform,
            input_transform=new_input_transform,
            init_hidden_layers=hidden,
        )

        return GenomePair(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_nodes=max_nodes,
            max_conns=max_conns,
            max_in_degree=convert.max_in_degree_for(max_nodes),
            new_genome=new_genome,
            new_state=new_genome.setup(new.common.State()),
            legacy_genome=legacy_genome,
            legacy_state=legacy_genome.setup(legacy.common.State()),
        )

    return build


# --------------------------------------------------------------------------- kernels
@pytest.fixture(scope="module")
def new_schedule(new, runner):
    """Derive the rewrite's ``Transformed`` for a payload, refusing a lossy schedule.

    Returns:
        ``run(pair, payload) -> Transformed`` of host numpy arrays, produced by the
        public ``genome.transform`` (so no test names a ``derive_schedule`` field).

    Notes:
        The ``edge_overflow == 0`` assertion lives here, in the one place the
        schedule is produced, because it is a precondition of every comparison in
        this directory: the legacy forward aggregates over a full ``max_nodes``
        column and truncates nothing, so a rewrite schedule that dropped an edge
        for lack of a window slot would compare two *different* networks.  The fix
        for such a payload is a larger ``max_in_degree`` (see
        ``convert.max_in_degree_for``), never a looser tolerance.
    """

    def run(pair: GenomePair, payload: Any):
        transformed = runner.run_defn(
            lambda node_keys, conn_keys: pair.new_genome.transform(
                pair.new_state, node_keys, conn_keys
            ),
            payload.node_keys,
            payload.conn_keys,
        )
        overflow = np.asarray(transformed.edge_overflow)
        assert np.all(overflow == 0), (
            "the rewrite schedule dropped "
            f"{overflow.tolist()} edge(s): the parity payload needs a wider "
            f"max_in_degree than {pair.max_in_degree} (currently max_nodes="
            f"{pair.max_nodes}); a lossy schedule must not be compared"
        )
        return transformed

    return run


@pytest.fixture(scope="module")
def new_forward(new, runner, new_schedule):
    """Run the rewrite's ``forward`` for one payload on one input batch.

    Returns:
        ``run(pair, payload, inputs) -> (P, num_outputs)`` numpy output.

    Notes:
        ``inputs`` is ``(P, num_inputs)`` — the rewrite is batch-first, population
        on axis 0 — so a multi-sample dataset is tiled on the host (see
        :func:`tile_payload`) instead of being evaluated sample by sample.
    """

    def run(pair: GenomePair, payload: Any, inputs: Any):
        transformed = new_schedule(pair, payload)
        width = len(new.common.Transformed._fields)
        return runner.run_defn(
            lambda *arrays: pair.new_genome.forward(
                pair.new_state,
                new.common.Transformed(*arrays[:width]),
                *arrays[width:],
            ),
            *tuple(transformed),
            *payload.args(),
            inputs,
        )

    return run


@pytest.fixture(scope="module")
def legacy_transformed(legacy, convert):
    """Run the legacy ``transform`` for one payload.

    Returns:
        ``run(pair, payload) -> transformed`` — the legacy ``(seqs, nodes, conns,
        u_conns)`` tuple, i.e. the topological order plus the payload and the dense
        ``(max_nodes, max_nodes)`` connection-index matrix.

    Notes:
        The converted payload is wrapped with ``jnp.asarray``: the legacy forward
        loops with ``lax.fori_loop`` and indexes ``nodes[i, 0]`` with the traced
        loop counter, which a numpy ``nodes`` cannot serve under JAX 0.12
        (``TracerArrayConversionError``).  Verified against the real legacy code.
    """
    jnp = pytest.importorskip("jax.numpy", reason="parity needs the `bench` extra (jax)")

    def run(pair: GenomePair, payload: Any):
        nodes, conns = convert.to_legacy(payload)
        return pair.legacy_genome.transform(
            pair.legacy_state, jnp.asarray(nodes), jnp.asarray(conns)
        )

    return run


@pytest.fixture(scope="module")
def legacy_forward(legacy, legacy_transformed):
    """Run the legacy ``forward`` for one payload on one input batch (one sample).

    Returns:
        ``run(pair, payload, inputs) -> (num_outputs,)`` numpy output.

    Notes:
        ``inputs`` is a single ``(num_inputs,)`` row: the legacy genome evaluates
        one network at a time and the pipeline vmaps over samples, so the parity
        tests compare one sample at a time (or vmap explicitly for the batch case)
        rather than relying on the legacy code accepting a rank-2 batch.
    """
    jnp = pytest.importorskip("jax.numpy", reason="parity needs the `bench` extra (jax)")

    def run(pair: GenomePair, payload: Any, inputs: Any):
        transformed = legacy_transformed(pair, payload)
        return np.asarray(
            pair.legacy_genome.forward(
                pair.legacy_state, transformed, jnp.asarray(inputs)
            )
        )

    return run


@pytest.fixture(scope="module")
def tile_payload(payloads):
    """Repeat every individual of a payload ``repeat`` times along the population axis.

    Returns:
        ``tile(payload, repeat) -> Payload`` with ``P * repeat`` individuals, in
        **individual-major** order: row ``p * repeat + s`` is individual ``p``'s
        sample ``s``.

    Notes:
        This is the layout ``problem/func_fit.FuncFit.evaluate`` documents for its
        batched call (``row p * S + s`` is sample ``s`` of individual ``p``), so a
        problem that tiles its dataset over the population finds a matching
        network on every row.  ``np.repeat`` is exactly that order.
    """

    def tile(payload: Any, repeat: int) -> Any:
        count = int(repeat)
        if count <= 0:
            raise ValueError(f"repeat must be positive, got {repeat}")
        return payloads.Payload(
            np.repeat(payload.node_keys, count, axis=0),
            np.repeat(payload.node_attrs, count, axis=0),
            np.repeat(payload.conn_keys, count, axis=0),
            np.repeat(payload.conn_attrs, count, axis=0),
        )

    return tile


@pytest.fixture(scope="module")
def new_distance(runner, new):
    """Run the rewrite's ``execute_distance`` on a batch of parent pairs.

    Returns:
        ``run(pair, left, right) -> (P,)`` numpy distances, where ``left``/``right``
        are payloads of the *same* shape (individual ``p`` of ``left`` is paired
        with individual ``p`` of ``right``).

    Notes:
        ``execute_distance`` takes two ``GenomeTensors``, so the eight arrays are
        passed flat and the two namedtuples are rebuilt inside the traced closure —
        the runner hands only top-level numpy leaves to ``evoxc.build`` as data.
    """

    def run(pair: GenomePair, left: Any, right: Any):
        width = len(new.genome.GenomeTensors._fields)
        return runner.run_defn(
            lambda *arrays: pair.new_genome.execute_distance(
                pair.new_state,
                new.genome.GenomeTensors(*arrays[:width]),
                new.genome.GenomeTensors(*arrays[width:]),
            ),
            *left.args(),
            *right.args(),
        )

    return run


@pytest.fixture(scope="module")
def legacy_distance(legacy, convert):
    """Run the legacy ``execute_distance`` on **one** parent pair.

    Returns:
        ``run(pair, left, right) -> float`` where both payloads hold exactly one
        individual (the legacy operator has no batch axis).

    Notes:
        The legacy operator is called with the ``(nodes, conns)`` matrices of each
        parent and returns a ``0-d`` array; the default weights are
        ``compatibility_disjoint=1.0`` / ``compatibility_weight=0.4`` on both
        sides, so no configuration has to be threaded through the pair.
    """
    jnp = pytest.importorskip("jax.numpy", reason="parity needs the `bench` extra (jax)")

    def run(pair: GenomePair, left: Any, right: Any) -> float:
        nodes1, conns1 = convert.to_legacy(left)
        nodes2, conns2 = convert.to_legacy(right)
        distance = pair.legacy_genome.execute_distance(
            pair.legacy_state,
            jnp.asarray(nodes1),
            jnp.asarray(conns1),
            jnp.asarray(nodes2),
            jnp.asarray(conns2),
        )
        return float(np.asarray(distance))

    return run


@pytest.fixture(scope="module")
def legacy_fitness(legacy, legacy_transformed):
    """Score one payload with the legacy ``problem.XOR`` / ``problem.XOR3d``.

    Returns:
        ``run(pair, payload, problem_name="XOR", error_method="mse") -> float``,
        one fitness value (the legacy ``FuncFit.evaluate`` reduces over every
        sample, so it returns a scalar per individual rather than a batch).

    Notes:
        ``act_func`` is the genome's own ``forward`` — the legacy problem calls
        ``vmap(act_func, in_axes=(None, None, 0))`` over ``self.inputs`` — and
        ``params`` is the single ``transformed`` tuple, so no sample tiling is
        needed on this side.  The ``randkey`` is unused by function fitting.
    """
    import jax

    def run(pair: GenomePair, payload: Any, *, problem_name: str = "XOR", error_method: str = "mse") -> float:
        problem = getattr(legacy.problem, problem_name)(error_method=error_method)
        transformed = legacy_transformed(pair, payload)
        fitness = problem.evaluate(
            pair.legacy_state,
            jax.random.PRNGKey(0),
            pair.legacy_genome.forward,
            transformed,
        )
        return float(np.asarray(fitness))

    return run


@pytest.fixture(scope="module")
def new_fitness(new, runner, new_schedule):
    """Score one payload with the rewrite's ``problem.XOR`` / ``problem.XOR3d``.

    Returns:
        ``run(pair, payload, problem_name="XOR", error_method="mse") -> (P,)``
        numpy fitness, one entry per individual of ``payload``.

    Notes:
        ``FuncFit.evaluate`` reads ``P`` from ``static_batch_size(transformed)``,
        tiles its baked dataset over that population axis and calls ``act_fn``
        **once** on the ``(P * S, I)`` batch (row ``p * S + s`` is sample ``s`` of
        individual ``p``).  One ``forward`` call cannot serve that batch — the
        genome's ``forward`` is itself batch-first over ``P`` individuals — so
        ``act_fn`` here reshapes the batch to ``(P, S, I)`` and calls ``forward``
        once per sample, stacking the ``(P, O)`` results back to ``(P * S, O)``.
        That is a static Python loop over ``S`` (no ``vmap``, as evoxc requires),
        and it keeps the population axis meaning "one network per row", so
        nothing about the problem layer is worked around.  The fixture therefore
        does **not** tile the payload; it scores the payload as given and returns
        one fitness per individual.  The problem is deterministic, so the key is
        a fixed ``evoxc.random`` key and is never advanced.
    """
    import evoxc

    enp = evoxc.numpy
    width = len(new.common.Transformed._fields)

    def act_fn_for(pair: GenomePair, schedule: Any, arrays: tuple):
        node_keys, node_attrs, conn_keys, conn_attrs = arrays

        def act_fn(batch):
            pop_size = int(schedule.node_level.shape[0])
            rows = int(batch.shape[0])
            samples = rows // pop_size
            grid = enp.reshape(batch, (pop_size, samples, int(batch.shape[1])))
            outputs = [
                pair.new_genome.forward(
                    pair.new_state,
                    schedule,
                    node_keys,
                    node_attrs,
                    conn_keys,
                    conn_attrs,
                    grid[:, sample, :],
                )
                for sample in range(samples)
            ]
            stacked = enp.stack(outputs, axis=1)
            return enp.reshape(stacked, (rows, int(outputs[0].shape[-1])))

        return act_fn

    def run(pair: GenomePair, payload: Any, *, problem_name: str = "XOR", error_method: str = "mse"):
        problem = getattr(new.problem, problem_name)(error_method=error_method)
        transformed = new_schedule(pair, payload)
        key = evoxc.random.key(0)

        def graph(*arrays):
            schedule = new.common.Transformed(*arrays[:width])
            act_fn = act_fn_for(pair, schedule, arrays[width:])
            return problem.evaluate(pair.new_state, key, act_fn, schedule)

        return np.asarray(runner.run_defn(graph, *tuple(transformed), *payload.args()))

    return run


@pytest.fixture(scope="module")
def new_grad(runner, new):
    """Differentiate the rewrite's ``forward`` through ``genome.grad``.

    Returns:
        ``run(pair, payload, transformed, inputs, loss_fn) -> (loss, (node_grad,
        conn_grad))`` of host numpy arrays, the graph being executed through
        :func:`support.runner.run_graph` (``lower`` → ``compile`` → ``load``).

    Notes:
        ``genome.grad`` is a graph *factory*: the ``evoxc.Graph`` it returns takes
        the five tensor arguments ``(node_keys, node_attrs, conn_keys, conn_attrs,
        inputs)`` in that order — only the two attribute tensors are
        differentiated, but the int32 keys and the inputs are ordinary graph
        inputs, while the schedule is specialised as a static constant of the
        factory call.  ``evoxc.grad`` returns gradients only, so the loss value is
        read from the *forward* pass of the same payload and the same ``loss_fn``
        (one extra executable) and returned alongside them, which is what makes
        ``test_grad``'s loss assertion a forward-tolerance comparison.
        ``loss_fn`` must therefore be built from evoxc ops: it is traced, not
        called on numpy data.
    """
    width = len(new.common.Transformed._fields)

    def run(pair: GenomePair, payload: Any, transformed: Any, inputs: Any, loss_fn: Callable):
        graph = pair.new_genome.grad(
            pair.new_state,
            transformed,
            *payload.args(),
            inputs,
            loss_fn,
        )
        node_grad, conn_grad = runner.run_graph(graph, *payload.args(), inputs)

        def loss_graph(*arrays):
            schedule = new.common.Transformed(*arrays[:width])
            node_keys, node_attrs, conn_keys, conn_attrs, batch = arrays[width:]
            return loss_fn(
                pair.new_genome.forward(
                    pair.new_state,
                    schedule,
                    node_keys,
                    node_attrs,
                    conn_keys,
                    conn_attrs,
                    batch,
                )
            )

        loss = runner.run_defn(loss_graph, *tuple(transformed), *payload.args(), inputs)
        return loss, (node_grad, conn_grad)

    return run


@pytest.fixture(scope="module")
def legacy_grad(legacy, convert):
    """Differentiate the legacy ``forward`` through the legacy ``genome.grad``.

    Returns:
        ``run(pair, payload, inputs, loss_fn) -> (loss, (grad_nodes, grad_conns))``
        as the legacy library returns it (``jax`` arrays; the caller unwraps with
        ``np.asarray``).

    Notes:
        The legacy ``grad`` vmaps ``forward`` over a rank-2 ``inputs`` batch and
        differentiates **one shared** ``(N, L)``/``(E, L)`` gene matrix, while the
        rewrite differentiates a ``(P, ...)`` payload.  With the payload tiled
        ``S`` times, the legacy total gradient is the axis-0 **sum** of the
        rewrite's per-individual gradients — which is why ``test_grad`` compares
        ``np.sum(node_grad, axis=0)`` against ``grad_nodes``.
    """
    jnp = pytest.importorskip("jax.numpy", reason="parity needs the `bench` extra (jax)")

    def run(pair: GenomePair, payload: Any, inputs: Any, loss_fn: Callable):
        nodes, conns = convert.to_legacy(payload)
        return pair.legacy_genome.grad(
            pair.legacy_state,
            jnp.asarray(nodes),
            jnp.asarray(conns),
            jnp.asarray(inputs),
            loss_fn,
        )

    return run
