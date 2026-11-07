"""``DefaultGenome``: depth-adaptive, level-synchronous feedforward evaluation.

The forward pass seeds the input rows once, then walks the depth levels — in one
``evoxc.while_loop`` whose trip count is derived from the schedule's ``node_level``, or as a
statically unrolled ``level = 1 .. max_depth`` sweep. Which spelling a graph is traced with is
:func:`._static_sweep`'s decision: the sweep on a non-tiled pass whose backend lowers it as one
capture-eligible body with no host involvement (iree), the loop everywhere else — on IREE a
``stablehlo.while`` carrying dispatches is lowered to a host ``cf`` loop with one submission per
trip, and the ``xla`` leg is unreliable with either spelling: its compiler captures the sweep's
single basic block as a HIP graph and SIGSEGVs inside ``libhsa-runtime64`` after one generation,
while the loop's nested ``while`` is not capture-eligible and merely crashes less often (1 of 3
trials survives at 21.45 ms/generation). An evolved network is usually 2-4 levels deep, so the
``max_depth`` ceiling would otherwise spend most of a counted pass on rows that cannot update.
Both spellings run the shared window reduce of
:class:`.base.BaseGenome` (:meth:`._node_step`), so the level step exists once.

A region op is safe *here* because this forward is never differentiated: ``evoxc.grad``
cannot differentiate ``while_loop``, and ``BaseGenome.grad`` deliberately builds its own dense
graph in ``grad.py`` instead of differentiating this one.
Everything is batch-first (population on axis 0); the shared window reduce lives in
:class:`.base.BaseGenome` and must not be re-implemented here.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import evoxc
import evoxc.numpy as enp
import numpy as np  # host-only: layout rotation and the ``forward_func`` read-out
import sympy as sp  # host-only: code generation

from ..common import (
    State,
    TensorNEATError,
    find_useful_nodes,
    kernel_mode,
    topological_sort,
)
from ..common.functions import ACT, AGG
from ..common.schedule_accelerator import is_accelerator
from ..common.tensorization import Transformed
from .base import BaseGenome
from .gene import BaseConn, BaseNode, DefaultConn, DefaultNode
from .operations import (
    BaseCrossover,
    BaseDistance,
    BaseMutation,
    DefaultCrossover,
    DefaultDistance,
    DefaultMutation,
)
from .operations.mutation.default import (
    DEFAULT_GENOME_CONN_ADD,
    DEFAULT_GENOME_CONN_DELETE,
)
from .utils import GenomeTensors
from .window import SeedPad, _i32, _iinf, by_samples, seed_pad, seed_values


class _PreparedPass(NamedTuple):
    """Loop-invariant operands of one ``DefaultGenome`` forward pass.

    :meth:`DefaultGenome.prepare_pass` derives it once per payload; each
    :meth:`DefaultGenome.forward_prepared` call then runs only the per-query half, so a rollout
    loop pays the pass prelude once instead of once per timestep.

    ``readout`` is the :class:`.step._ReadOut` of the schedule (the output-slot gather index,
    its liveness mask and the ``NaN`` fill), ``one`` the rank-0 ``int32`` the level loop counts
    with, and ``seed_pad`` the :class:`.window.SeedPad` plan of the value seed's ``NaN`` block
    (no block at all when the network has no non-input row). All three depend only on the
    schedule and the static shapes, so no query pays their derivation or their constants.

    ``sweep`` selects the level spelling (see :func:`._static_sweep`): ``True`` is the
    statically unrolled sweep of a non-tiled pass whose backend lowers it as one
    capture-eligible body, which reads no depth bound and no counter, so ``limit`` and ``one``
    are ``None`` there; every other pass (the host leg, any tiled batch, and any ``xla`` pass)
    keeps both and walks them in one ``evoxc.while_loop``.
    """

    node_attrs: evoxc.SymbolicTensor
    statics: Any
    limit: evoxc.SymbolicTensor | None
    node_level: evoxc.SymbolicTensor
    samples: int
    pop_size: int
    max_nodes: int
    readout: Any
    one: evoxc.SymbolicTensor | None
    seed_pad: SeedPad
    sweep: bool


#: Backends that must not carry the statically unrolled sweep. The ``xla`` leg's compiler
#: captures the sweep's rollout body — one capture-eligible basic block — as a HIP graph and
#: replays it through ``hipGraphLaunch``, which SIGSEGVs inside ``libhsa-runtime64`` after one
#: completed generation (2 of 2 trials; the launch budget shrinks with the captured body's
#: command count and a smaller ``max_depth`` does not help). The loop spelling's nested ``while``
#: is not capture-eligible, which makes it *less* crash-prone rather than safe: 21.45
#: ms/generation and 5 generations in 1 of 3 trials, exit 139 after 4 generations in the other
#: two. Only ``XLA_FLAGS=--xla_gpu_enable_command_buffer=`` makes that leg reliable, so the loop
#: is selected here as the least-bad spelling, never as a fix.
_SWEEP_UNSAFE_BACKENDS = ("xla",)


def _static_sweep(state: Any, samples: int) -> bool:
    """Whether this pass takes the statically unrolled level sweep instead of the loop.

    Three host-side statics decide it, all read off the trace-time ``State`` and never as
    traced values, so a graph contains exactly one of the two spellings:

    * ``kernel_mode`` must be ``"accelerator"``: the dispatch-bound numpy interpreter reads the
      trade the other way round (extra unrolled device work is not free there), so it keeps the
      loop.
    * ``samples`` must be ``1``. A rollout queries one payload ``max_step`` times per
      generation, which is where the unrolling pays; a tiled dataset is one query whose batch
      is ``P * samples`` rows, and every unrolled level would cost that whole batch — 16.0 ->
      51.0 ms/generation at ``P * samples = 250 000`` (func-fit, iree+rocm) — while the loop
      runs only the batch's real depth once.
    * the backend must not be ``xla`` (see :data:`_SWEEP_UNSAFE_BACKENDS`). On iree the sweep
      is the faster spelling: 60.48 against 145.47 ms/generation for ``cartpole_p100``
      (``max_step=200``, pop 100) at the shipped ``max_depth=16``. A hybrid static prefix plus a
      data-dependent tail is rejected, not shipped: 4 static levels plus the tail loop measures
      73.05 ms, *worse* than the pure sweep's 60.48, because the nested tail loop re-stages its
      bound on every enclosing iteration. On ``xla`` the sweep is a hard crash — its captured
      HIP graph SIGSEGVs after one completed generation (2 of 2 trials, at any ``max_depth``) —
      and the loop, whose nested ``while`` is not capture-eligible, is only the *least-bad*
      spelling there: 21.45 ms/generation and 5 generations in 1 of 3 trials, with the other two
      crashing after 4. ``XLA_FLAGS=--xla_gpu_enable_command_buffer=`` is the only reliable fix,
      so no working ``xla`` leg is claimed.

    A state that carries no ``backend`` entry — a bare ``State()`` in a unit test — is not the
    ``xla`` leg and therefore keeps the sweep, as does every backend other than ``xla``.

    Args:
        state: Host configuration container (``kernel_mode`` and, when the pipeline registered
            it, ``backend``).
        samples: Static tiling factor of the evaluated batch.

    Returns:
        bool: ``True`` for the unrolled sweep, ``False`` for the data-dependent loop.
    """
    if not is_accelerator(kernel_mode(state)):
        return False
    if samples != 1:
        return False
    backend = getattr(state, "backend", None)
    if isinstance(backend, str) and backend.strip().lower() in _SWEEP_UNSAFE_BACKENDS:
        return False
    return True


class DefaultGenome(BaseGenome):
    """Feedforward NEAT genome, behaviourally equivalent to NEAT-python.

    ``network_type = "feedforward"``: the schedule carries ``node_level`` (longest-path depth,
    inputs at level 0) and the forward pass evaluates one level per loop iteration, so a node is
    computed only after all of its real sources already have a value.
    """

    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes: int = 50,
        max_conns: int = 100,
        node_gene: BaseNode | None = None,
        conn_gene: BaseConn | None = None,
        mutation: BaseMutation | None = None,
        crossover: BaseCrossover | None = None,
        distance: BaseDistance | None = None,
        output_transform: Callable | None = None,
        input_transform: Callable | None = None,
        init_hidden_layers: Sequence[int] = (),
        max_in_degree: int | None = None,
        max_depth: int = 16,
        key_span: int | None = None,
    ):
        """Configure a feedforward genome with NEAT-python default genes and operators.

        Gene/operator arguments default to ``None`` and are then built **per instance**
        (``DefaultNode()``, ``DefaultConn()``, ``DefaultMutation()``, ``DefaultCrossover()``,
        ``DefaultDistance()``), never shared through a mutable default argument: a genome owns
        its gene objects, and two genomes that shared one would silently mutate each other's
        attribute tables.  A ``None`` ``mutation`` builds a fresh ``DefaultMutation`` with the
        genome's shipped connection-gate defaults (``conn_add=0.4``, ``conn_delete=0.05``); an
        explicitly constructed operator overrides them.

        See :class:`.base.BaseGenome` for the dense initial network
        (``num_inputs -> init_hidden_layers -> num_outputs``), the row/key invariant and the
        transform probes.

        Args:
            num_inputs: Input node count, also the network input width.
            num_outputs: Output node count, also the network output width.
            max_nodes: Static node capacity ``N``.
            max_conns: Static connection capacity ``E``.
            node_gene: Node gene; ``None`` builds a fresh ``DefaultNode()``.
            conn_gene: Connection gene; ``None`` builds a fresh ``DefaultConn()``.
            mutation: Mutation operator; ``None`` builds a fresh ``DefaultMutation`` with the
                genome's shipped connection-gate defaults (``conn_add=0.4``,
                ``conn_delete=0.05``); an explicit operator overrides them.
            crossover: Crossover operator; ``None`` builds a fresh ``DefaultCrossover()``.
            distance: Distance operator; ``None`` builds a fresh ``DefaultDistance()``.
            output_transform: Optional elementwise map of the ``(P, num_outputs)`` read-out.
            input_transform: Optional elementwise map of ``inputs`` before it is written into
                the input rows.
            init_hidden_layers: Width of each dense hidden layer; empty means one dense
                input-to-output layer.
            max_in_degree: Static window width ``K``; ``None`` resolves to
                ``max(initial in-degree, 4)``.
            max_depth: Static depth unroll count ``D``; a node deeper than ``D`` is never
                visited and keeps its ``NaN`` value. That truncation is silent: it never sets
                ``Transformed.overflow``, which is only the ``edge_overflow > 0`` alias of the
                connection-count overflow (a node with more than ``max_in_degree`` incoming
                edges).
            key_span: Static key bound ``S`` of the schedule's dense key table; ``None``
                resolves to ``max(max_nodes, num_inputs + num_outputs) + max_conns``. A caller
                that allocates structural keys beyond that default (the mutation operators take
                caller-allocated keys) must raise it here.

        Raises:
            TensorNEATError: If ``max_nodes``/``max_conns`` cannot hold the initial network, if
                ``init_hidden_layers`` needs an in-degree above ``max_in_degree``, if a
                gene/operator is not the expected type, or if a transform probe fails.
            ValueError: If a size argument is not a positive int.
        """
        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_nodes=max_nodes,
            max_conns=max_conns,
            node_gene=DefaultNode() if node_gene is None else node_gene,
            conn_gene=DefaultConn() if conn_gene is None else conn_gene,
            mutation=(
                DefaultMutation(
                    conn_add=DEFAULT_GENOME_CONN_ADD,
                    conn_delete=DEFAULT_GENOME_CONN_DELETE,
                )
                if mutation is None
                else mutation
            ),
            crossover=DefaultCrossover() if crossover is None else crossover,
            distance=DefaultDistance() if distance is None else distance,
            output_transform=output_transform,
            input_transform=input_transform,
            init_hidden_layers=init_hidden_layers,
            max_in_degree=max_in_degree,
            max_depth=max_depth,
            key_span=key_span,
        )

    def forward(
        self,
        state: State,
        transformed: Transformed,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        inputs: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Evaluate the population level by level, in one loop graph.

        This is :meth:`prepare_pass` followed by :meth:`forward_prepared`: a caller that
        queries one payload many times (a rollout loop) should prepare once and call
        :meth:`forward_prepared` per query.

        The exact algebra, with ``P`` the population, ``rows = P * samples`` the evaluated row
        count (a problem tiles its dataset or rollout over the population axis, so row
        ``p * samples + j`` is individual ``p``), ``N`` ``max_nodes``, ``K`` the window width
        and ``D`` ``max_depth``:

        1. ``values = full((rows, N), NaN)`` with ``input_transform(inputs)`` written into the
           input rows (``0..num_inputs-1``) — only those rows hold a number. The seed is
           assembled as ``concatenate([seed, NaN (rows, N - num_inputs)], axis=1)``: one op
           that leaves every other row at the ``NaN`` padding value, instead of a
           ``(rows, N)`` mask. The ``NaN`` block itself is a loop-invariant operand, so
           :meth:`prepare_pass` derives it once and a query pays only the ``concatenate``
           (and nothing at all when ``N == num_inputs``).
        2. ``hit_conn_attrs = self._hit_conn_attrs(transformed, conn_attrs, samples)``
           **once**: edge attributes are constant during a forward pass, so the per-slot gather
           is hoisted out of the loop, derived at the schedule's ``P`` rows and expanded to the
           evaluated ones (the schedule's numbers are sample-invariant). The attribute rows of
           absent nodes are rewritten to ``0`` in the same one-off step
           (``self._padding_safe_attrs``): the node gene casts its option-index columns to
           ``int32``, and those rows are never read back into ``values``.
        3. ``limit = self._depth_limit(transformed)`` — the deepest reachable level of this
           batch, clamped to ``[1, max_depth]`` — then
           ``evoxc.while_loop(cond, body, (1, values))`` walks ``level = 1 .. limit`` with
           ``values = self._node_step(state, transformed, node_attrs, hit_conn_attrs, values,
           update_mask=(node_level == level))``, ``node_level`` expanded once outside the loop.
           The sweep spelling instead unrolls the same step over
           ``level = 1 .. max_depth`` (:meth:`._level_sweep`) and derives neither ``limit``
           nor the counter.
           The two further terms the pre-rewrite formulation ANDed in are both implied:
           ``node_exists`` because the schedule stores ``I_INF`` in ``node_level`` on every
           absent (and every unreachable) row, and the static ``reduce_sum(in_valid, axis=-1)
           > 0`` because a row with no incoming slot has no *live* slot either. The dynamic
           "at least one *live* slot" term is ANDed in inside ``_node_step``, so a node whose
           sources are all NaN keeps NaN and its descendants lose their only live source: the
           legacy NaN poisoning, reproduced exactly.
        4. ``return self._output_values(transformed, values, self.output_transform,
           prepared.readout)`` — the rows ``transformed.output_slot`` names (here
           ``max_nodes-num_outputs..max_nodes-1``), then ``output_transform``. The read-out's
           flat gather index, its liveness mask and its ``NaN`` fill come from the prepared
           pass (:meth:`_output_readout`), so a query pays three ops — ``reshape``, ``gather``,
           ``select`` — instead of re-deriving the eleven-op read-out per timestep.

        Cost is ``O(depth x N x K)`` elementwise work: proportional to the *real* depth and the
        real in-degree, never ``max_nodes**2`` per node and never ``max_depth`` levels of typed
        graph. The bound is the population maximum because the loop counter is one scalar, so a
        batch containing one deep individual still pays for it. A node whose level exceeds
        ``max_depth`` is never visited and keeps the ``NaN`` seed (the documented depth limit; a
        rebuild with a larger ``max_depth`` is the fix, and nothing reports it — only a
        connection that missed its incoming window sets ``Transformed.overflow``, the
        ``edge_overflow > 0`` alias of the per-individual dropped-edge count).

        Args:
            state: Host configuration container.
            transformed: Execution schedule of the same population (``node_level``,
                ``node_exists (P, N)``, ``in_valid``/``in_src``/``in_conn (P, N, K)``).
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs: ``(P, N, NA) float32`` node attributes, NaN padded.
            conn_keys: ``(P, E, CK) int32`` connection keys, ``I_INF`` padded.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, NaN padded.
            inputs: ``(P * samples, num_inputs) float32`` network inputs, tiled
                individual-major over the population axis.

        Returns:
            ``(P * samples, num_outputs) float32`` outputs after ``output_transform``.

        Raises:
            TensorNEATError: If the batch is not a positive multiple of ``P`` (see
                :meth:`_sample_count`) — there is then no sample axis to expand the schedule
                with, and the two would silently disagree about their rows.
        """
        pop_size = int(transformed.node_level.shape[0])
        samples = self._sample_count(inputs, pop_size)
        prepared = self.prepare_pass(
            state, transformed, node_keys, node_attrs, conn_keys, conn_attrs, samples
        )
        return self.forward_prepared(state, transformed, prepared, inputs)

    def prepare_pass(
        self,
        state: State,
        transformed: Transformed,
        node_keys: evoxc.SymbolicTensor,
        node_attrs: evoxc.SymbolicTensor,
        conn_keys: evoxc.SymbolicTensor,
        conn_attrs: evoxc.SymbolicTensor,
        samples: int = 1,
    ) -> _PreparedPass:
        """Derive the loop-invariant operands of one forward pass.

        Everything :meth:`forward` computes once — the padding-safe node attributes, the
        ``_StepStatics`` of the value window (with its node operand bundle forced here, so the
        level loop never traces it), the depth bound and the level loop's rank-0 counter (both
        loop-leg only: the sweep reads neither), the expanded ``node_level``, the
        read-out of the schedule and the ``NaN`` block of the value seed — so
        :meth:`forward_prepared` can run once per query on the same payload.

        Args:
            state: Host configuration container.
            transformed: Execution schedule of the same population.
            node_keys: ``(P, N) int32`` node keys, ``I_INF`` padded.
            node_attrs: ``(P, N, NA) float32`` node attributes, NaN padded.
            conn_keys: ``(P, E, CK) int32`` connection keys; accepted for symmetry with
                :meth:`forward` and unused, like there.
            conn_attrs: ``(P, E, CA) float32`` connection attributes, NaN padded.
            samples: Static tiling factor ``S`` of the evaluated ``P * samples`` rows.

        Returns:
            ``_PreparedPass`` for :meth:`forward_prepared`.

        Raises:
            TensorNEATError: If ``samples`` is not a positive int.
        """
        if (
            isinstance(samples, bool)
            or not isinstance(samples, (int, np.integer))
            or int(samples) < 1
        ):
            raise TensorNEATError(
                f"samples must be a positive int, got {samples!r}"
            )
        samples = int(samples)
        pop_size, max_nodes = (int(size) for size in transformed.node_level.shape)
        # Absent rows read as ``0`` instead of ``NaN`` before the level loop: their option
        # index columns are cast to ``int32`` by the node gene, and a ``NaN``-to-int cast is a
        # warning (and a garbage id) even though those rows never reach ``values``.
        prepared_attrs = self._padding_safe_attrs(node_attrs, node_keys)
        # The statics are derived at the schedule's ``P`` rows and expanded to the evaluated
        # ``P * samples`` rows: row ``p * samples + j`` reads the copy of individual ``p``.
        statics = self._hit_conn_attrs(transformed, conn_attrs, samples)
        # Force the node operand bundle here so it is traced in this block instead of inside
        # the loop body, where the region would re-evaluate it on every iteration.
        statics.node_operands(self, prepared_attrs)
        # The value seed's ``NaN`` block: planned here (one broadcast of a rank-0 constant in
        # the host formulation) so a query pays only the splice of the input rows into it --
        # and no op at all when the network has no non-input row.
        pad = max_nodes - self.num_inputs
        # The level spelling is one host-side read of the pass's statics (_static_sweep): the
        # unrolled sweep needs neither the depth bound (a ``reduce_max`` dispatch only the loop
        # consumes) nor the loop's counter, but only a non-tiled pass on a backend that lowers
        # it as one capture-eligible body. ``samples == 1`` is one discriminator: a rollout
        # queries one payload ``max_step`` times per generation, which is where the unrolling
        # pays, while a tiled dataset is a single query and the unrolled form would multiply
        # its ``P * samples`` rows by ``max_depth`` -- 16.0 -> 51.0 ms/generation at
        # ``P * samples = 250 000`` rows (func-fit) on iree+rocm, so a tiled batch keeps the
        # ``while_loop`` verbatim -- do not "simplify" this gate away. The backend is the
        # other: ``xla`` captures the sweep's basic block as a HIP graph and SIGSEGVs after one
        # generation, while the loop's nested ``while`` is not capture-eligible, so that backend
        # asks for the loop even at ``samples == 1`` -- as the least-bad spelling, not as a fix
        # (it still crashes in 2 of 3 trials).
        sweep = _static_sweep(state, samples)
        return _PreparedPass(
            node_attrs=prepared_attrs,
            statics=statics,
            # One level per loop iteration, bounded by the *real* depth of this batch instead
            # of the static ``max_depth`` ceiling: it is a loop bound derived from the schedule
            # alone, not data.
            limit=None if sweep else self._depth_limit(transformed),
            node_level=by_samples(transformed.node_level, samples, pop_size),
            samples=samples,
            pop_size=pop_size,
            max_nodes=max_nodes,
            # The read-out is loop-invariant for the whole generation, so its gather index, its
            # liveness mask and its ``NaN`` fill are derived once per pass: a query then costs
            # ``reshape`` + ``gather`` + ``select``.
            readout=self._output_readout(transformed, samples),
            one=None if sweep else enp.full((), 1, dtype=evoxc.int32),
            seed_pad=seed_pad(
                kernel_mode(state), self.num_inputs, max_nodes, pop_size * samples
            ),
            sweep=sweep,
        )

    def forward_prepared(
        self,
        state: State,
        transformed: Transformed,
        prepared: _PreparedPass,
        inputs: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Run one forward pass from ``prepared``, the per-query half of :meth:`forward`.

        Emits only the input seed, the level sweep and the read-out; every loop-invariant
        operand comes from ``prepared`` — including the seed's ``NaN`` block, the loop's rank-0
        counter (loop leg only) and the read-out's gather index/mask/fill — so repeated queries
        on one payload pay the prelude once. ``prepared.sweep`` picks the spelling: the
        data-dependent ``evoxc.while_loop`` or the statically unrolled sweep
        (:func:`._static_sweep`).

        Args:
            state: Host configuration container.
            transformed: Execution schedule the pass was prepared with.
            prepared: Result of :meth:`prepare_pass` for this payload and ``samples``.
            inputs: ``(P * samples, num_inputs) float32`` network inputs, tiled
                individual-major over the population axis.

        Returns:
            ``(P * samples, num_outputs) float32`` outputs after ``output_transform``.

        Raises:
            TensorNEATError: If ``inputs`` does not carry ``pop_size * samples`` rows, so the
                requested batch disagrees with the prepared schedule.
        """
        rows = int(inputs.shape[0])
        expected = prepared.pop_size * prepared.samples
        if rows != expected:
            raise TensorNEATError(
                f"forward_prepared was asked for {rows} row(s) but prepare_pass derived the "
                f"pass for pop_size={prepared.pop_size} x samples={prepared.samples} = "
                f"{expected} row(s); prepare the pass with the samples of this batch"
            )
        seed = inputs if self.input_transform is None else self.input_transform(inputs)
        # The ``NaN`` block of the seed is a prepared plan (see :meth:`prepare_pass`): the
        # query pays only the splice of its input rows, and none at all when the network has no
        # non-input row.
        values = seed_values(seed, prepared.seed_pad)
        sweep = self._level_sweep if prepared.sweep else self._level_loop
        values = sweep(state, transformed, prepared, values)
        return self._output_values(
            transformed, values, self.output_transform, prepared.readout
        )

    def _level_loop(
        self,
        state: State,
        transformed: Transformed,
        prepared: _PreparedPass,
        values: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Loop spelling of the level sweep: one ``evoxc.while_loop`` over ``level = 1 .. limit``.

        The trip count is data-dependent (``prepared.limit`` is a rank-0 device tensor). On the
        numpy interpreter that is free. On a compiled device it is not: IREE does not run a
        dispatch-bearing ``stablehlo.while`` as one device command buffer — it hands the loop to
        the host as a ``cf`` loop, one ``stream.cmd.execute`` submission per trip, so a
        device-tensor predicate costs a blocking ``hal.fence.await`` plus a ``hal.buffer.load``
        per trip (~157 us measured). Nested inside an enclosing loop it is worse still, because
        the inner bound is re-staged (``queue.alloca`` + ``queue.copy`` + fence wait + load) on
        every enclosing iteration even when the inner loop then runs zero trips (~230 us per
        query end to end). That is why a non-tiled accelerator pass prefers
        :meth:`_level_sweep`. On the ``xla`` leg the sweep's captured HIP graph SIGSEGVs after
        one generation (2 of 2 trials) and this loop's nested ``while`` is not
        capture-eligible, so the loop is the least-bad spelling there — 21.45 ms/generation, and
        it still SIGSEGVs in 2 of 3 trials, the third surviving 5 generations (see
        :func:`._static_sweep`).

        Args:
            state: Host configuration container.
            transformed: Execution schedule the pass was prepared with.
            prepared: Result of :meth:`prepare_pass` with ``sweep=False``.
            values: ``(rows, max_nodes) float32`` seeded node values, ``NaN`` off the input rows.

        Returns:
            ``(rows, max_nodes) float32`` values after ``1 .. limit`` level steps.
        """
        one = prepared.one

        def cond(carry) -> evoxc.SymbolicTensor:
            """Keep iterating while the next unvisited level is still within the depth."""
            level, _values = carry
            return evoxc.less_equal(level, prepared.limit)

        def body(carry) -> tuple[evoxc.SymbolicTensor, evoxc.SymbolicTensor]:
            """Advance one level: the plain window step of :meth:`._node_step`.

            The ``node_exists`` term is not ANDed into the mask: the schedule sets
            ``node_level`` to ``I_INF`` on every absent (or unreachable) row, so
            ``node_level == level`` is already False there.
            """
            level, values = carry
            updated = self._node_step(
                state,
                transformed,
                prepared.node_attrs,
                prepared.statics,
                values,
                evoxc.equal(prepared.node_level, level),
            )
            return evoxc.add(level, one), updated

        _level, values = evoxc.while_loop(cond, body, (one, values))
        return values

    def _level_sweep(
        self,
        state: State,
        transformed: Transformed,
        prepared: _PreparedPass,
        values: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Accelerator spelling of the level sweep: ``level = 1 .. max_depth``, unrolled.

        One :meth:`._node_step` per level, traced at Python trace time, so the graph holds
        ``max_depth`` copies of the level step and *no* region op at all. On IREE that is what
        keeps the pass on the device: a dispatch-bearing ``stablehlo.while`` is lowered to a
        host ``cf`` loop with one submission per trip, and a nested region re-stages its bound
        on every enclosing iteration, whereas this body is a single capture-eligible basic
        block with no host involvement — 60.48 against 145.47 ms/generation for
        ``cartpole_p100`` (max_step 200, ``max_depth=16``) on iree+rocm. A hybrid static prefix
        plus a data-dependent tail is rejected, not shipped: 73.05 ms at 4 static levels is
        *worse* than the pure sweep's 60.48, because the tail loop is nested inside the rollout
        body and pays the bound re-staging per enclosing iteration. ``kernel_mode`` gates the
        sweep because the dispatch-bound numpy interpreter reads the trade the other way round,
        and ``samples`` because a tiled pass is ``P * samples`` rows (each unrolled level would
        cost that many rows: 16.0 -> 51.0 ms/generation at ``P * samples = 250 000``, iree+rocm)
        queried once per generation, where the loop runs the batch's real depth. On ``xla`` this
        same capture-eligible body is replayed as a HIP graph by the ROCm plugin and SIGSEGVs
        inside ``libhsa-runtime64`` after one completed generation (2 of 2 trials, at any
        ``max_depth``), so that backend selects the loop as the least-bad spelling: 21.45
        ms/generation and 5 generations in 1 of 3 trials, the other two crashing after 4 (all
        four facts in :func:`._static_sweep`).

        Same values as :meth:`_level_loop`, bit for bit:

        * level ``0`` (the input rows) is never updated by either spelling — both start at
          ``1``;
        * absent, unreachable and cyclic rows carry ``I_INF`` in ``node_level``, which matches
          no level of ``1 .. max_depth``, so they are never written either way;
        * a batch whose real depth is ``d < max_depth`` sees an all-``False`` update mask on
          levels ``d + 1 .. max_depth``, so the ``select(update, z, values)`` inside
          :meth:`._node_step` returns ``values`` unchanged — including the NaN poisoning of a
          node with no live source, because ``update`` is ``node_level == level`` ANDed with
          ``any_slot(live)``, and both spellings recompute it per level;
        * a row deeper than ``max_depth`` is never visited by either spelling and keeps its
          ``NaN`` seed (the documented depth limit; ``max_depth`` is a genome static already in
          the pipeline's executable-cache signature, so raising it is a rebuild, not a silent
          change).

        Args:
            state: Host configuration container.
            transformed: Execution schedule the pass was prepared with.
            prepared: Result of :meth:`prepare_pass` with ``sweep=True``.
            values: ``(rows, max_nodes) float32`` seeded node values, ``NaN`` off the input rows.

        Returns:
            ``(rows, max_nodes) float32`` values after ``max_depth`` level steps.
        """
        for level in range(1, self.max_depth + 1):
            values = self._node_step(
                state,
                transformed,
                prepared.node_attrs,
                prepared.statics,
                values,
                evoxc.equal(prepared.node_level, _i32(level)),
            )
        return values

    def _depth_limit(self, transformed: Transformed) -> evoxc.SymbolicTensor:
        """Deepest level this batch can reach, as a rank-0 loop bound in ``[1, max_depth]``.

        ``node_level`` is ``I_INF`` on every row that is absent, unreachable from an input, or
        on a cycle, so those rows are masked to ``0`` before the reduction: they are never
        visited, exactly as in the fully unrolled pass (their value stays ``NaN``). The
        population maximum is used because the loop counter is one scalar shared by the whole
        batch; clamping to at least ``1`` keeps the ordinary one-level network unchanged and
        clamping to ``max_depth`` preserves the documented static-unroll limit.

        Args:
            transformed: Execution schedule of the current population.

        Returns:
            Rank-0 ``int32`` tensor: ``min(max_depth, max(1, max(node_level)))``.
        """
        reachable = evoxc.logical_and(
            transformed.node_exists,
            evoxc.not_equal(transformed.node_level, _iinf(evoxc.int32)),
        )
        levels = evoxc.select(
            reachable, transformed.node_level, enp.zeros((), dtype=evoxc.int32)
        )
        deepest = evoxc.reduce_max(levels)  # rank-0: max over the population and the rows
        return evoxc.minimum(
            evoxc.maximum(deepest, enp.full((), 1, dtype=evoxc.int32)),
            enp.full((), self.max_depth, dtype=evoxc.int32),
        )

    def network_dict(
        self, state: State, genome: GenomeTensors, whether_re_cound_idx: bool = True
    ) -> dict[str, Any]:
        """Convert one individual to host dicts plus feedforward graph utilities.

        Extends :meth:`.base.BaseGenome.network_dict` with three host-only entries computed on
        the renumbered key sets — :func:`~tensorneat.common.host_graph.topological_sort` and
        :func:`~tensorneat.common.host_graph.find_useful_nodes` run in Python and never enter a
        graph.

        Args:
            state: Host configuration container.
            genome: Payload of one individual.
            whether_re_cound_idx: Renumber hidden keys to be contiguous above
                ``max(input/output keys)`` before converting.

        Returns:
            ``{"nodes", "conns", "topo_order", "topo_layers", "useful_nodes"}``: the key dicts
            of the base method plus the topological order, the layer grouping and the set of
            nodes that actually reach an output (nodes outside it may be NaN and are skipped by
            code generation).

        Raises:
            tensorneat.common.errors.ConfigError: If ``genome`` holds more than one individual.
            tensorneat.common.errors.GraphError: If the decoded topology has a cycle, which a
                feedforward genome cannot execute.
        """
        network = super().network_dict(state, genome, whether_re_cound_idx)
        node_keys = set(network["nodes"])
        conns = set(network["conns"])
        topo_order, topo_layers = topological_sort(node_keys, conns)
        network["topo_order"] = topo_order
        network["topo_layers"] = topo_layers
        network["useful_nodes"] = find_useful_nodes(
            node_keys, conns, set(self.get_output_idx())
        )
        return network

    def sympy_func(
        self,
        state: State,
        network: dict,
        sympy_input_transform: Any = None,
        sympy_output_transform: Any = None,
        backend: str = "numpy",
    ) -> tuple:
        """Build sympy expressions of a network and a NumPy evaluator (host-only).

        The legacy 8-tuple contract with ``..common.sympy_tools``: symbols are named by role
        (``i<k>`` raw input, ``norm<k>`` transformed input, ``h<k>`` hidden, ``o<k>`` output),
        nodes are emitted in ``topo_order`` and every node expression is substituted into its
        successors, so ``output_exprs`` are closed forms.  ``backend`` is ``"numpy"`` only in
        this rewrite (there is no JAX); ``AGG``/``ACT`` contribute their sympy modules to the
        ``lambdify`` call.  A missing ``sympy_*_transform`` while the genome has the
        corresponding transform emits a warning, as before.

        Args:
            state: Host configuration container.
            network: Output of :meth:`network_dict` (needs ``topo_order``/``useful_nodes``).
            sympy_input_transform: Sympy counterpart of ``input_transform``; a list with one
                entry per input, or a single callable applied to each.
            sympy_output_transform: Sympy counterpart of ``output_transform``.
            backend: Codegen backend, ``"numpy"`` only.

        Returns:
            ``(symbols, args_symbols, input_symbols, nodes_exprs, output_exprs, forward_func,
            topo_order, useful_nodes)`` — ``forward_func`` maps a sequence of inputs to a NumPy
            array of outputs with the gene values already substituted.

        Raises:
            TensorNEATError: If ``backend`` is not ``"numpy"`` — a compiled/JAX codegen backend
                no longer exists, and silently generating NumPy source for it would misreport
                what the caller got.
        """
        if backend != "numpy":
            raise TensorNEATError(
                "sympy_func backend must be 'numpy' (the only codegen backend of this "
                f"rewrite), got {backend!r}"
            )
        if sympy_input_transform is None:
            if self.input_transform is not None:
                warnings.warn(
                    "genome.input_transform is not None but sympy_input_transform is None: "
                    "the generated formula omits the input transform"
                )
            sympy_input_transform = lambda values: values  # noqa: E731 - identity elementwise
        if not isinstance(sympy_input_transform, list):
            sympy_input_transform = [sympy_input_transform] * self.num_inputs
        if sympy_output_transform is None and self.output_transform is not None:
            warnings.warn(
                "genome.output_transform is not None but sympy_output_transform is None: "
                "the generated formula omits the output transform"
            )
        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()
        order = network["topo_order"]
        hidden_idx = [
            key
            for key in network["nodes"]
            if key not in input_idx and key not in output_idx
        ]
        symbols = {}
        for key in network["nodes"]:
            if key in input_idx:
                # Two symbols per input: the raw ``i<k>`` and the transformed ``norm<k>``.
                offset = key - min(input_idx)
                symbols[-key - 1] = sp.Symbol(f"i{offset}")
                symbols[key] = sp.Symbol(f"norm{offset}")
            elif key in output_idx:
                symbols[key] = sp.Symbol(f"o{key - min(output_idx)}")
            else:
                symbols[key] = sp.Symbol(f"h{key - min(hidden_idx)}")
        nodes_exprs = {}
        args_symbols = {}
        for key in order:
            if key in input_idx:
                offset = key - min(input_idx)
                nodes_exprs[symbols[-key - 1]] = symbols[-key - 1]
                nodes_exprs[symbols[key]] = sympy_input_transform[offset](
                    symbols[-key - 1]
                )
                continue
            node_inputs = []
            for endpoints, conn_dict in network["conns"].items():
                if endpoints[1] != key:
                    continue
                message, conn_symbols = self.conn_gene.sympy_func(
                    state, conn_dict, symbols[endpoints[0]]
                )
                args_symbols.update(conn_symbols)
                node_inputs.append(message)
            expression, node_symbols = self.node_gene.sympy_func(
                state,
                network["nodes"][key],
                node_inputs,
                is_output_node=key in output_idx,
            )
            args_symbols.update(node_symbols)
            if key in output_idx and sympy_output_transform is not None:
                expression = sympy_output_transform(expression)
            nodes_exprs[symbols[key]] = expression
        input_symbols = [symbols[-key - 1] for key in input_idx]
        reduced_exprs = dict(nodes_exprs)
        for key in order:
            reduced_exprs[symbols[key]] = reduced_exprs[symbols[key]].subs(reduced_exprs)
        output_exprs = [reduced_exprs[symbols[key]] for key in output_idx]
        modules = ["numpy", AGG.sympy_module("numpy"), ACT.sympy_module("numpy")]
        evaluators = [
            sp.lambdify(input_symbols + list(args_symbols), expression, modules=modules)
            for expression in output_exprs
        ]

        def forward_func(values: Sequence[float]) -> np.ndarray:
            """Evaluate the symbolic network on ``num_inputs`` raw input values.

            Args:
                values: One value per input node, in input order (raw ``i<k>``, not
                    ``norm<k>`` — apply ``sympy_input_transform`` yourself if the genome has
                    one).

            Returns:
                ``(num_outputs,) float64`` numpy array with the gene values substituted.
            """
            return np.asarray(
                [
                    evaluator(*values, *args_symbols.values())
                    for evaluator in evaluators
                ],
                dtype=np.float64,
            )

        return (
            symbols,
            args_symbols,
            input_symbols,
            nodes_exprs,
            output_exprs,
            forward_func,
            order,
            network["useful_nodes"],
        )

    def visualize(
        self,
        network: dict,
        rotate: float = 0,
        reverse_node_order: bool = False,
        size: tuple = (300, 300, 300),
        color: tuple = ("yellow", "white", "blue"),
        with_labels: bool = False,
        edgecolors: str = "k",
        arrowstyle: str = "->",
        arrowsize: int = 3,
        edge_color: tuple = (0.3, 0.3, 0.3),
        save_path: str = "network.svg",
        save_dpi: int = 800,
        **kwargs: Any,
    ) -> None:
        """Draw a network as a layered directed graph (host-only).

        ``networkx`` and ``matplotlib`` are imported **inside** the body, so
        ``import tensorneat`` never pulls them in.  Nodes are placed with the multipartite
        layout of ``topo_layers``, reordered per layer to reduce edge crossings, then rotated by
        ``rotate`` degrees; input nodes use ``size[0]``/``color[0]``, hidden nodes the middle
        entry and output nodes the last.

        Args:
            network: Output of :meth:`network_dict` (needs ``topo_order``/``topo_layers``).
            rotate: Rotation of the layout in degrees.
            reverse_node_order: Reverse the node order within every layer.
            size: Node sizes ``(input, hidden, output)`` in drawing units.
            color: Node colours ``(input, hidden, output)``.
            with_labels: Draw the node keys.
            edgecolors: Marker edge colour.
            arrowstyle: Matplotlib arrow style of the edges.
            arrowsize: Arrow size in drawing units.
            edge_color: Edge colour.
            save_path: Output path of the figure.
            save_dpi: Resolution of the saved figure.
            **kwargs: Extra keyword arguments forwarded to ``networkx.draw``.

        Returns:
            ``None``; the figure is written to ``save_path`` and the matplotlib figure closed.
        """
        import networkx as nx
        from matplotlib import pyplot as plt

        conns = list(network["conns"])
        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()
        topo_order, topo_layers = network["topo_order"], network["topo_layers"]
        node2layer = {
            node: layer for layer, nodes in enumerate(topo_layers) for node in nodes
        }
        # Reorder the nodes of every layer by the position of their predecessors, so the
        # multipartite layout draws fewer crossing edges.
        subset_key = {}
        for layer, nodes in enumerate(topo_layers):
            if layer == 0 or len(nodes) == 1:
                subset_key[layer] = list(nodes)
                continue
            anchors = []
            for node in nodes:
                position = 0
                for row, predecessor in enumerate(topo_layers[layer - 1]):
                    if (predecessor, node) in conns:
                        position += row
                anchors.append(position)
            subset_key[layer] = [node for _, node in sorted(zip(anchors, nodes))]
        if reverse_node_order:
            subset_key = {layer: nodes[::-1] for layer, nodes in subset_key.items()}
        if not isinstance(size, tuple):
            size = (size, size, size)
        if not isinstance(color, tuple):
            color = (color, color, color)
        graph = nx.DiGraph()
        for node in topo_order:
            if node in input_idx:
                marker = (size[0], color[0])
            elif node in output_idx:
                marker = (size[2], color[2])
            else:
                marker = (size[1], color[1])
            graph.add_node(
                node, subset=node2layer[node], size=marker[0], color=marker[1]
            )
        graph.add_edges_from(conns)
        pos = nx.multipartite_layout(graph, subset_key=subset_key)
        angle = np.deg2rad(rotate)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        rotated_pos = {
            node: (
                cos_angle * x - sin_angle * y,
                sin_angle * x + cos_angle * y,
            )
            for node, (x, y) in pos.items()
        }
        nx.draw(
            graph,
            pos=rotated_pos,
            node_size=[data["size"] for data in graph.nodes.values()],
            node_color=[data["color"] for data in graph.nodes.values()],
            with_labels=with_labels,
            edgecolors=edgecolors,
            arrowstyle=arrowstyle,
            arrowsize=arrowsize,
            edge_color=edge_color,
            **kwargs,
        )
        plt.savefig(save_path, dpi=save_dpi)
        plt.close()
