"""Regression tests for the operands ``DefaultGenome.prepare_pass`` hoists.

What this module proves
-----------------------
``prepare_pass`` derives the loop-invariant operands of one forward pass once, and
two of them are the subject here:

* ``_PreparedPass.level_masks`` — the sweep's per-level update mask, one
  ``node_level == level`` comparison per unrolled level, compared **once per pass**
  in the entry block instead of once per query (a rollout asks one payload
  ``max_step`` times, so the comparison used to be paid ``max_step * max_depth``
  times). It must be bit-identical to deriving ``equal(node_level, level)`` per level
  on the same pass, and the sweep must really read it — a mask tuple that is rotated
  or cleared must change the output, otherwise the hoist would be a silent no-op
  that the equality alone could not tell from a working one;
* the prepared rank-0 scalars (``_StepStatics.scalars`` /
  ``NodeOperands.scalars``) — ``{0.0: zero, 1.0: one}``, materialised once per pass
  and handed to every option callable that **declares a ``scalars`` parameter**
  (``gene/node/base.py::_takes_scalars``). Every shipped ``act``/``agg`` option
  declares it today, so the literal side of the comparison is the test-local
  :func:`_literal_sigmoid`, which keeps the old one-argument call shape: the two
  paths must agree bit for bit; and the option that declares ``scalars`` must drop
  the per-level ``constant`` ops, that drop being the proof the prepared tensor —
  not a fresh literal — is what the activation used.

Oracles used here (never the kernel itself)
-------------------------------------------
* **numpy** — ``schedule.node_level == level`` computed host-side on the schedule's
  own ``int32`` array for the mask table, and the plain ``forward`` for the values;
* the **other path** — the literal-mask sweep and the literal-scalar option, both
  exact (``np.array_equal(..., equal_nan=True)``): a tolerance would hide a level or
  a literal that one path read differently;
* the **traced IR** — ``..ir_ops.op_names``, whose ``constant`` count is what tells
  a prepared scalar from a freshly materialised literal.

Gotchas this module encodes
---------------------------
* The schedule is passed **field by field** and rebuilt inside the traced closure
  (``./conftest.py``); a ``Transformed`` handed to the runner as one argument is
  baked into the trace with numpy fields.
* ``_PreparedPass`` is a ``NamedTuple``, which is what makes the tamper tests
  possible: ``prepared._replace(level_masks=...)`` is a host-side operation on a
  trace-local record, so a test can hand ``forward_prepared`` a mask tuple the real
  ``prepare_pass`` would never produce.
* ``max_depth=8`` on a payload whose deepest row is level 6, so levels 7 and 8 carry
  an all-``False`` mask on purpose — a case a sweep must survive unchanged (it is
  what makes the depth limit work) and a case the union invariant below must account
  for.
* Backend is ``"numpy"`` throughout, through the shared runner only: no device, no
  optional extra, no ``jax``/``torch``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..ir_ops import op_names

#: The three input rows (one all-zero row so a mishandled zero cannot pass).
INPUTS = np.array([[0.3, -0.2], [-1.5, 0.75], [0.0, 0.0]], dtype=np.float32)
#: Population of the shared payload.
POP_SIZE = 3
#: Static shapes of the payload and the genome below.
MAX_NODES = 8
MAX_CONNS = 16
MAX_DEPTH = 8
#: The level of ``chain_payload``'s output row (row ``7``), hand-derived: inputs at
#: level 0, rows ``2..7`` one level each.
OUTPUT_LEVEL = 6


@pytest.fixture(scope="module")
def swept(common, genome, payloads, run_transform):
    """The shared chain payload, its genome, an accelerator state and its schedule.

    ``payloads.chain_payload(3, 8, 16)`` wires ``0, 1 -> 2 -> ... -> 7`` on three
    identical individuals, so rows ``2..7`` sit at levels ``1..6`` and levels ``7``
    and ``8`` are empty. The state is the accelerator formulation, which is the one
    that selects the sweep (:func:`~tensorneat.genome.default._static_sweep`) at
    ``samples == 1`` — without it ``prepare_pass`` returns an empty ``level_masks``
    and every assertion below would be vacuous.

    Returns:
        ``(payload, genome_object, state, schedule)``.
    """
    payload = payloads.chain_payload(POP_SIZE, MAX_NODES, MAX_CONNS)
    genome_object = genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        max_in_degree=4,
        max_depth=MAX_DEPTH,
    )
    state = genome_object.setup(common.State(kernel_mode="accelerator"))
    return payload, genome_object, state, run_transform(genome_object, state, payload)


def _pass_closure(common, genome_object, state):
    """The traced half both mask tests share: prepare one pass from the traced arrays.

    Args:
        common: The ``tensorneat.common`` package (``Transformed``).
        genome_object: The genome whose statics match the payload.
        state: The trace-time host state (closed over, never passed as data).

    Returns:
        ``closure(*arrays) -> _PreparedPass`` where ``*arrays`` is the schedule's
        fields followed by the four payload arrays and the static ``samples``.
    """
    width = len(common.Transformed._fields)

    def closure(*arrays):
        """Rebuild the schedule and prepare its pass inside the active trace."""
        schedule = common.Transformed(*arrays[:width])
        return genome_object.prepare_pass(
            state, schedule, *arrays[width : width + 4], arrays[width + 4]
        )

    return closure


# --------------------------------------------------------------------------- #
# level_masks: bit-identical to the per-level comparison
# --------------------------------------------------------------------------- #
def test_level_masks_equal_the_per_level_comparison(common, runner, swept):
    """Invariant: ``prepared.level_masks`` is ``equal(node_level, level)`` for every level.

    Oracle: numpy, on the schedule's own ``int32`` ``node_level`` — plus the traced
    comparison built in the same graph, so the two spellings run on one operand.

    Assertion: one ``(P, N) bool`` mask per ``max_depth`` level; each is array-equal
    to both the in-graph ``equal(node_level, level)`` and the host numpy comparison;
    and the masks partition the reachable rows — their union is exactly "present
    with a level in ``1 .. max_depth``" and their sum is ``1`` there (no row is
    visited twice). Levels ``OUTPUT_LEVEL + 1 .. max_depth`` are all-``False``, the
    case the depth limit relies on.
    """
    import evoxc
    from evoxc import numpy as enp

    payload, genome_object, state, schedule = swept
    prepare = _pass_closure(common, genome_object, state)
    width = len(common.Transformed._fields)
    levels = tuple(range(1, MAX_DEPTH + 1))

    def traced(*arrays):
        """Prepare one pass and return its masks beside the comparisons."""
        prepared = prepare(*arrays)
        derived = tuple(
            enp.equal(prepared.node_level, enp.full((), level, dtype=evoxc.int32))
            for level in levels
        )
        return prepared.node_level, prepared.level_masks, derived

    args = (*schedule, *payload.args(), 1)
    node_level, masks, derived = runner.run_defn(traced, *args)

    assert len(masks) == MAX_DEPTH, (
        f"the sweep must prepare one mask per level, got {len(masks)}"
    )
    assert node_level.shape == (POP_SIZE, MAX_NODES)
    for level, mask, oracle in zip(levels, masks, derived):
        assert mask.shape == node_level.shape, f"level {level}: wrong mask shape"
        assert mask.dtype == np.bool_, f"level {level}: mask dtype {mask.dtype}"
        np.testing.assert_array_equal(mask, oracle, err_msg=f"level {level}")
        np.testing.assert_array_equal(
            mask, np.equal(schedule.node_level, np.int32(level)), err_msg=f"level {level}"
        )

    stacked = np.stack(masks)
    visited = stacked.sum(axis=0)
    reachable = np.logical_and(
        np.not_equal(schedule.node_level, np.iinfo(np.int32).max),
        np.logical_and(schedule.node_level >= 1, schedule.node_level <= MAX_DEPTH),
    )
    np.testing.assert_array_equal(visited > 0, reachable, err_msg="mask union")
    assert int(visited.max()) == 1, "a row was visited by more than one level"
    for level in range(OUTPUT_LEVEL + 1, MAX_DEPTH + 1):
        assert not masks[level - 1].any(), (
            f"level {level} is past the chain's depth {OUTPUT_LEVEL} and its mask "
            "must be all-False"
        )
    assert masks[0].any(), "level 1 must update the chain's first hidden row"


def test_the_sweep_reads_the_prepared_masks(common, runner, swept):
    """Invariant: the sweep consumes ``prepared.level_masks``; a tampered tuple changes it.

    Oracle: the plain ``forward`` of the same payload and schedule (exact), which
    reads the untouched masks. This is the half an equality test cannot prove: if
    ``_level_sweep`` re-derived ``equal(node_level, level)`` itself, every mask
    assertion above would still pass while the hoist did nothing.

    Three runs in one graph: the real pass, the same pass with its masks **rotated**
    by one level (level ``l`` reads level ``l + 1``'s mask, so the chain's first
    hidden row is never computed and every later row reads an uncomputed source),
    and the same pass with every mask **cleared**. The rotated and cleared read-outs
    must be all ``NaN`` while the real one is finite and equal to ``forward``.
    """
    import evoxc
    from evoxc import numpy as enp

    payload, genome_object, state, schedule = swept
    width = len(common.Transformed._fields)

    def traced(*arrays):
        """Run the real pass, then the rotated-mask and cleared-mask variants."""
        schedule_value = common.Transformed(*arrays[:width])
        payload_arrays = arrays[width : width + 4]
        prepared = genome_object.prepare_pass(
            state, schedule_value, *payload_arrays, arrays[width + 4]
        )
        masks = prepared.level_masks
        rotated = prepared._replace(level_masks=tuple(masks[1:]) + tuple(masks[:1]))
        cleared = prepared._replace(
            level_masks=tuple(
                enp.zeros(tuple(mask.shape), dtype=evoxc.bool_) for mask in masks
            )
        )
        inputs = arrays[width + 5]
        real = genome_object.forward_prepared(
            state, schedule_value, prepared, inputs
        )
        turned = genome_object.forward_prepared(
            state, schedule_value, rotated, inputs
        )
        blank = genome_object.forward_prepared(
            state, schedule_value, cleared, inputs
        )
        plain = genome_object.forward(state, schedule_value, *payload_arrays, inputs)
        return real, turned, blank, plain

    args = (*schedule, *payload.args(), 1, INPUTS)
    real, turned, blank, plain = runner.run_defn(traced, *args)

    assert np.isfinite(real).all(), (
        f"the real pass must compute the chain, got {real.ravel()}"
    )
    np.testing.assert_array_equal(real, plain)
    assert np.isnan(turned).all(), (
        "rotating the prepared masks by one level left a finite read-out, so the "
        f"sweep did not read them positionally: {turned.ravel()}"
    )
    assert np.isnan(blank).all(), (
        "clearing every prepared mask left a finite read-out, so the sweep did not "
        f"read them at all: {blank.ravel()}"
    )


# --------------------------------------------------------------------------- #
# the prepared scalars: offered to the options that declare them
# --------------------------------------------------------------------------- #
def _prepared_sigmoid(z, scalars=None):
    """``act.sigmoid`` spelled so that it *can* reuse the pass's prepared ``1.0``.

    Declares the ``scalars`` parameter :func:`~tensorneat.genome.gene.node.base._takes_scalars`
    looks for, and takes the rank-0 ``1.0`` from the mapping when the pass prepared
    one. Without a mapping it materialises its own literal — the documented fallback
    a kernel must keep, because a pass whose bundle lacks the value must still trace.

    Args:
        z: ``(P, N) float32`` pre-activation node values.
        scalars: ``{0.0: zero, 1.0: one}`` of the executing pass, or ``None``.

    Returns:
        ``(P, N) float32`` logistic sigmoid, exactly ``1 / (1 + exp(-z))``.
    """
    import evoxc
    from evoxc import numpy as enp

    one = enp.full((), 1.0, dtype=evoxc.float32) if scalars is None else scalars[1.0]
    return evoxc.divide(one, evoxc.add(one, evoxc.exp(evoxc.negate(z))))


def _literal_sigmoid(z):
    """``act.sigmoid`` in the call shape an option that declares no ``scalars`` keeps.

    The same expression as :func:`_prepared_sigmoid`, but with **no** ``scalars``
    parameter, so :func:`~tensorneat.genome.gene.node.base._takes_scalars` answers
    ``False`` and the node gene hands it the exact old one-argument call: each
    ``1.0`` is its own fresh ``enp.full`` literal, and ``evoxc`` merges no
    constants, so the unrolled sweep pays one ``constant`` op per literal per
    level. This is the non-scalars stand-in the comparison needs — every shipped
    :mod:`~tensorneat.common.functions.act` / :mod:`~tensorneat.common.functions.agg`
    option declares ``scalars`` today, so none of them can play that role.

    Args:
        z: ``(P, N) float32`` pre-activation node values.

    Returns:
        ``(P, N) float32`` logistic sigmoid, exactly ``1 / (1 + exp(-z))``.
    """
    import evoxc
    from evoxc import numpy as enp

    one = enp.full((), 1.0, dtype=evoxc.float32)
    return evoxc.divide(one, evoxc.add(one, evoxc.exp(evoxc.negate(z))))


def _swept_genome(genome, activation):
    """The shared chain genome with ``activation`` as its only activation option."""
    node_gene = genome.DefaultNode(activation_options=[activation])
    return genome.DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        max_in_degree=4,
        max_depth=MAX_DEPTH,
        node_gene=node_gene,
    )


def test_takes_scalars_is_the_signature_rule():
    """Invariant: ``_takes_scalars`` answers exactly "does ``fn`` declare ``scalars``".

    Oracle: :func:`inspect.signature` itself, applied to the shipped kernels and to
    kernels that do not declare the parameter. The rule is a *signature* read
    (host-side, cached per gene), never a name list, so a kernel of the
    ``common.functions`` tree that declares ``scalars`` flips the flag with no
    genome-side change — and one that does not keeps its exact old call shape.

    The mapping is live, not a future hook: ``act.sigmoid`` and the six ``agg``
    kernels named below declare ``scalars=None``, so a shipped forward pass really
    consumes the prepared tensors, while a kernel without the parameter
    (:func:`_literal_sigmoid`, a bare lambda) is handed its own literal and the old
    call shape.
    """
    import inspect

    from tensorneat.common.functions import AGG, ACT
    from tensorneat.genome.gene.node.base import _takes_scalars

    shipped = (ACT.sigmoid, AGG.sum, AGG.max, AGG.min, AGG.product, AGG.maxabs, AGG.mean)
    for fn in (*shipped, _literal_sigmoid, _prepared_sigmoid):
        assert _takes_scalars(fn) is ("scalars" in inspect.signature(fn).parameters), (
            f"{getattr(fn, '__name__', fn)}: the flag disagrees with its signature"
        )
    assert all(_takes_scalars(fn) is True for fn in shipped), (
        "every shipped act/agg option declares `scalars`, so the prepared mapping "
        "is what a shipped forward pass consumes"
    )
    assert _takes_scalars(_literal_sigmoid) is False
    assert _takes_scalars(_prepared_sigmoid) is True
    assert _takes_scalars(lambda z: z) is False


def test_the_prepared_scalar_path_is_bit_identical_to_the_literal_path(
    common, genome, runner, payloads, run_transform, run_forward
):
    """Invariant: an option reusing the pass's ``1.0`` returns the literal option's values.

    Oracle: :func:`_literal_sigmoid` — the same math spelled by an option that
    declares **no** ``scalars`` parameter, so it keeps its own ``enp.full``
    literal and the exact old call shape — on the same payload and schedule
    (exact, ``NaN``-aware). Both genomes are the accelerator formulation at
    ``samples == 1``, so both run the unrolled sweep.

    The shipped ``act.sigmoid`` cannot serve as the literal side any more: it now
    declares ``scalars=None`` (as do all six ``agg`` kernels), so the node gene
    hands *both* genomes the prepared tensor and the two traced graphs are op for
    op identical — a "fewer constants" assertion against it compares a graph with
    itself. ``_literal_sigmoid`` keeps the flag off, which is what makes the
    constant count discriminate.

    Assertion: the two read-outs are array-equal; the read-out is finite and non-zero
    (so the equality is not a vacuous ``NaN`` pattern); the scalars-aware genome's
    ``prepare_pass``+``forward_prepared`` graph traces **no** ``constant`` op beyond
    the ones its ``prepare_pass`` prelude already holds while the literal genome's
    full graph traces more than its own prelude — i.e. the level body of the
    prepared route is constant-free and the literal route is not —; and in total the
    scalars-aware genome holds **fewer** ``constant`` ops. Those counts are what
    proves the activation consumed ``_PreparedScalars`` instead of a fresh
    ``enp.full``.
    """
    import evoxc

    from tensorneat.genome.gene.node.base import _takes_scalars

    assert _takes_scalars(_literal_sigmoid) is False, (
        "the literal stand-in must declare no `scalars` parameter, otherwise the "
        "scalars-aware path is compared against itself"
    )
    assert _takes_scalars(_prepared_sigmoid) is True

    payload = payloads.chain_payload(POP_SIZE, MAX_NODES, MAX_CONNS)
    width = len(common.Transformed._fields)
    counts, preludes, values = {}, {}, {}
    for label, activation in (
        ("literal", _literal_sigmoid),
        ("prepared", _prepared_sigmoid),
    ):
        genome_object = _swept_genome(genome, activation)
        state = genome_object.setup(common.State(kernel_mode="accelerator"))
        schedule = run_transform(genome_object, state, payload)
        values[label] = run_forward(genome_object, state, schedule, payload, INPUTS)
        args = (*schedule, *payload.args(), 1, INPUTS)
        # ``samples`` is a static (a plain int), so it is traced as a value while
        # the schedule/payload/input leaves become tensor specs.
        specs = [
            runner.spec_like(array) if isinstance(array, np.ndarray) else array
            for array in args
        ]

        def forward_route(*arrays, genome_object=genome_object, state=state):
            """One ``prepare_pass`` + ``forward_prepared``, traced for its op names."""
            schedule_value = common.Transformed(*arrays[:width])
            prepared = genome_object.prepare_pass(
                state, schedule_value, *arrays[width : width + 4], arrays[width + 4]
            )
            return genome_object.forward_prepared(
                state, schedule_value, prepared, arrays[width + 5]
            )

        def prelude_route(*arrays, genome_object=genome_object, state=state):
            """``prepare_pass`` alone — the loop-invariant prelude of one pass.

            The prepared record is a ``NamedTuple`` that also holds the host-side
            ``_StepStatics``, which a graph cannot return, so the traced outputs
            are its per-level mask tensors: the prelude's own op histogram is what
            this route is counted for, never the returned record.
            """
            schedule_value = common.Transformed(*arrays[:width])
            prepared = genome_object.prepare_pass(
                state, schedule_value, *arrays[width : width + 4], arrays[width + 4]
            )
            return prepared.level_masks

        counts[label] = op_names(evoxc.trace(forward_route, *specs))
        preludes[label] = op_names(evoxc.trace(prelude_route, *specs[:-1]))

    literal, prepared = values["literal"], values["prepared"]
    assert np.isfinite(literal).all(), f"the chain must read out, got {literal.ravel()}"
    assert np.abs(literal).max() > 0.0, "every read-out is zero"
    np.testing.assert_array_equal(prepared, literal)

    prepared_constants = counts["prepared"].get("constant", 0)
    literal_constants = counts["literal"].get("constant", 0)
    assert prepared_constants == preludes["prepared"].get("constant", 0), (
        "the prepared route's level body traced "
        f"{prepared_constants - preludes['prepared'].get('constant', 0)} `constant` "
        f"op(s) beyond its {preludes['prepared'].get('constant', 0)}-constant "
        "prepare_pass prelude: it did not reuse the prepared scalar"
    )
    assert literal_constants > preludes["literal"].get("constant", 0), (
        "the literal route's level body must pay its own literal: its full graph "
        f"traced {literal_constants} `constant` op(s) against its "
        f"{preludes['literal'].get('constant', 0)}-constant prelude"
    )
    assert prepared_constants < literal_constants, (
        "the scalars-aware activation traced "
        f"{prepared_constants} constant op(s) against the literal option's "
        f"{literal_constants}: it did not reuse the prepared scalar"
    )


def test_an_option_without_a_mapping_materialises_its_own_literal(runner):
    """Invariant: a missing mapping key falls back to the option's own literal, exactly.

    Oracle: the same kernel called with the pass's ``1.0`` — both in one graph, on one
    operand. This is the contract a future ``common.functions`` kernel must keep: a
    pass whose bundle lacks a value still traces, and the value is unchanged.

    Assertion: the two branches are array-equal on a grid that includes a negative, a
    zero and a large magnitude (so a mishandled sign or saturation would show), and
    the mapped branch traced no extra ``constant`` beyond the caller's own.
    """
    import evoxc

    z = np.array([[-8.0, -1.0], [0.0, 1.0], [8.0, 30.0]], dtype=np.float32)

    def traced(z_value, one):
        """Sigmoid with the prepared ``1.0`` beside the no-mapping fallback."""
        return _prepared_sigmoid(z_value, {1.0: one}), _prepared_sigmoid(z_value, None)

    mapped, fallback = runner.run_defn(
        traced, z, np.float32(1.0)
    )
    assert np.isfinite(mapped).all() and np.isfinite(fallback).all()
    np.testing.assert_array_equal(mapped, fallback)
