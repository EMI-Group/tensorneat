"""Distance rows over one generation's population digest.

Speciation measures *every* species centre against the *same* population, twice
per generation, so the genome's two-phase distance API serves it directly:
:meth:`~tensorneat.genome.base.BaseGenome.prepare_distance` sorts the population
once per generation into a ``DistanceDigest``, and
:meth:`~tensorneat.genome.base.BaseGenome.measure_distance` turns one centre
genome into one ``(P,)`` distance row without re-sorting the population.

Shape contract
--------------
Every call this module makes to ``genome.measure_distance`` carries one centre
spelled by ``state.distance_centre``: a host string read at trace time
(``_validation.distance_centre_mode``), never a traced value.

* ``"tiled"`` -- **the default when the entry is absent**, i.e. what a bare
  ``State`` gets: the centre row repeated to ``pop_size`` identical rows (one
  rank-raising ``evoxc.broadcast`` per field, built by ``_kernels.centre_payload``).
  That is the population-shaped payload the genome's row-wise merge route takes,
  so no call can select its dense single-centre route -- whose per-call scatters
  a GPU lowers single-threaded, a measured 2x device-time regression.
* ``"single"``: a one-individual payload (leading axis ``1``, one
  ``evoxc.reshape`` per field), which selects that dense route.  It is the
  cheaper spelling on the numpy interpreter and the dearer one on a GPU, so the
  pipeline picks the mode from the device it runs on.

Both spellings carry the same centre row and return the same ``(P,)`` floats.
The ``group * pop_size`` batch the genome also accepts belongs to a
``group_digest``-tiled digest, which :meth:`DistanceRows.matrix` builds on the
accelerator spelling only (see `Measured cost`).

:meth:`DistanceRows.__call__` is single-valued: it accepts a centre payload of
batch ``1`` or batch ``pop_size`` and rejects every other batch width; in
``"tiled"`` mode a batch-``1`` centre is tiled to ``pop_size`` rows first, so the
mode's contract holds for every caller, while in ``"single"`` mode both widths
are measured as they stand.
It **is** the production route: ``_kernels.speciate`` calls ``distance_fn(...)``
from its per-iteration ``compute_row`` bodies for the gated ``reserve`` and
``assign_members`` scans, and unconditionally for every ``assign_members``
iteration on the dense path (a small ``Pipeline`` run traced 4 ``__call__``
invocations on the gated path and 2 on the dense one, plus one :meth:`matrix`:
``evoxc.scan`` peels step 0 at the enclosing level and traces the body again, so
each scan body is traced twice).
:meth:`DistanceRows.row` is the same measurement spelled explicitly for one
centre, and :meth:`DistanceRows.matrix` is the explicit whole-table form: one
such call per centre row (or per group of ``k`` centre rows on the accelerator
spelling), run as a single fixed-length ``evoxc.scan`` over the table's rows on
both spellings, so an ``(S, P)`` table is never one ``S * P`` batch.  The rows
are identical either way, because the compatibility kernel is row-independent,
and measuring inside a region keeps the kernel's transient footprint at one
call's worth of centres instead of ``S`` of them, which the scan body then
releases again before the next iteration.  ``matrix`` is never selected by
``__call__``: a caller that wants a whole table asks for it by name, which is
what the dense path's scan 1 does.

Nothing here re-implements distance arithmetic: every number comes from the
genome's own kernel, so the rows are bit-identical to
:meth:`~tensorneat.genome.base.BaseGenome.execute_distance` on the same centre and
population.

Measured cost
-------------
Numpy reference backend, ``species_size=20``, one configuration per fresh
process, 3 warmup + 20 timed ``Pipeline.step`` after the build step, six
interleaved rounds, medians of the per-round medians, ``XOR``, seed 42,
``is_detailed=False``; ``reference`` is the pre-change tree's measurement
(one-individual only, i.e. what ``"single"`` replaces):

* ``xor_p500`` ms per generation: 159.43 tiled, 112.73 single, 115.81 reference.
* ``xor_p1000`` ms per generation: 272.84 tiled, 176.61 single, 178.58 reference.
* ``xor_p100`` ms per generation: 42.85 tiled, 41.09 single, 40.92 reference.
* peak RSS ``P = 500`` / ``P = 1000``: 218.7 / 348.5 MiB tiled, 196.4 / 302.5 single.
* executed ops per generation at ``P = 500`` / ``P = 1000`` / ``P = 100``:
  7 466 / 7 471 / 3 817 tiled against 8 066 / 8 071 / 3 877 single (and reference).

``"single"`` is within noise of the reference it replaced (0.986x / 0.999x /
1.004x), so the switch costs the interpreter nothing.
``"tiled"`` is the legacy merge shape and dispatches **fewer** ops (-7.4 %) yet
runs +39 % / +54 % / +4.3 % slower: at these populations the numpy interpreter
is charged by element volume rather than by dispatch, and the merge's
``(pop_size, 2 * max_conns)`` row-width intermediates are that volume (the
genome records 2.39 against 1.35 ms per compatibility call at ``P = 500``).
So on numpy a pipeline selects ``"single"``, and ``"tiled"`` exists for
GPU-class devices, where the dense route's per-call scatters are lowered
single-threaded (40 x 0.97 ms + 40 x 0.52 ms of a generation) and the merge is
the 2x cheaper device shape.
The table costs exactly one ``measure_distance`` call per centre row on the host
spelling -- 40 calls per generation at ``species_size = 20``, 20 here and 20 in
scan 2 -- and what the scan buys there is the *region* around those calls, not
fewer of them.

Pre-measuring *both* speciation scans on the host spelling needs a ``cond`` for
scan 2's data-dependent ``pop[seed]`` centre and costs ~50 % more peak RSS for no
timed gain.

Grouped ``k``-centre calls are the **accelerator spelling's** table leg, never
the host one's: the genome's ``group_digest`` / ``GroupedDigest`` route is
bit-identical to ``k`` plain ``P``-row calls and is a row-wise-merge shape, so it
competes with ``"tiled"`` rather than with ``"single"``, and
``grouped.group_size`` answers ``k = 1`` for every non-accelerator state, which is
why the numpy pipeline traces exactly the calls it traced before.
The grouped leg scans ``K // k`` **groups**: one iteration gathers the ``k``
centre rows of its group, merges them with ``grouped.centre_batch`` and makes one
``measure_distance`` call against the ``k``-times tiled digest, ``k = 2`` on the
shipped accelerator statics, and at most ``K % k`` trailing rows are measured
singly against the plain digest.
The same grouping did **not** ship for numpy: there the interpreter is charged by
element volume rather than by dispatch, and the ``k``-times wider
``(k * P, 2 * max_conns)`` merge intermediates cost more than the dispatches they
save (253.2 against 233.8 MiB peak RSS, wall time bimodal at process level), so
the host leg keeps ``k = 1`` and one centre per call.
The grouped leg's own device numbers are not part of this record: the
1 267 -> 1 130 kernel / 7.45 -> 6.83 ms device-time pair was measured when the
table was spelled as straight-line calls, a spelling this module no longer
traces, so the grouped form here carries the same values but has no committed
measurement on this tree.
Scan 2 is not grouped at all: its per-slot centre is ``pop[seed]`` on a creating
slot, and ``seed`` depends on the assignments made by the earlier slots, so a
group of consecutive centres cannot be formed before their rows are measured.

Peak memory
-----------
``evoxc``'s numpy interpreter keeps every SSA result of a block in that block's
environment until the block returns, and pops a nested region's environment per
iteration.  A kernel inside a ``scan`` body therefore pays one region's
transients at a time, while a kernel at the top level keeps them alive for the
rest of the generation -- which is what the whole-table scan is for, and why the
table is not measured at the top level on either spelling (only its ``K % k``
tail is, and that is less than one group).
The measured footprint is per mode (the table above): 196.4 / 302.5 MiB at
``P = 500`` / ``P = 1000`` in ``"single"`` and 218.7 / 348.5 MiB in ``"tiled"``,
the merge's wider rows being the difference; build follows the same split,
because a loop body is traced once while a static loop over ``S`` centres traces
``S`` copies of the kernel.
That build term is why the region stays even on a compiled backend: spelling the
same rows as straight-line calls removes the ``while`` and its per-iteration
scaffolding, and measured on ``xor_p500`` it tripled the static generation graph
(2 174 -> 7 393 ops, 296 -> 531 ``hal.executable.export``) and took
``iree-compile`` from ~5.8-6.0 s to 11.4-14.2 s per shape, a net loss that was
reverted.
Any future hoist out of the region has to report peak RSS and build next to its
wall time.
"""

from __future__ import annotations

from typing import Any

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError
from tensorneat.genome.operations.distance.grouped import centre_batch, for_state

from ._kernels import centre_payload, int_const
from ._validation import distance_centre_mode


class DistanceRows:
    """Compatibility distances of species centres against one digested population.

    One instance is built per generation (in :meth:`NEAT.tell`) and handed to
    :meth:`SpeciesController.speciate` as its ``distance_fn``.  It is a plain
    traced helper -- created and called while the generation graph is being
    traced.

    Args:
        genome (Any): The genome owning ``measure_distance`` and the digest's
            gene layout.
        digest (Any): ``DistanceDigest`` returned by
            ``genome.prepare_distance(state, population)``.
        population (Any): The payload the digest was prepared from.  Every call
            is checked against its static shape, so a payload from a different
            generation or population size is rejected instead of silently
            measuring against a stale digest.
        pop_size (int): Static population size ``P`` -- the digest's batch, and
            the row count each measured centre is paired with.

    Raises:
        TensorNEATError: If the digest's population axis is not ``pop_size``.
    """

    def __init__(
        self,
        genome: Any,
        digest: Any,
        population: Any,
        pop_size: int,
    ) -> None:
        self.genome = genome
        self.digest = digest
        self.population = population
        self.pop_size = int(pop_size)
        digest_pop_size = int(digest.node_count.shape[0])
        if digest_pop_size != self.pop_size:
            raise TensorNEATError(
                "DistanceRows: the distance digest holds "
                f"{digest_pop_size} genomes but pop_size is {self.pop_size}; "
                "prepare the digest from the current population"
            )
        if int(population[0].shape[0]) != self.pop_size:
            raise TensorNEATError(
                "DistanceRows: the population payload has batch "
                f"{int(population[0].shape[0])} but pop_size is {self.pop_size}"
            )

    def __call__(self, state: Any, centre: Any, pop: Any = None) -> Any:
        """Distance rows of ``centre`` against the digested population.

        Args:
            state (Any): Shared state (forwarded to the genome kernel).
            centre (Any): Centre payload of batch ``1`` (one individual, measured
                against the digest's population rows) or batch ``pop_size`` (a
                population-shaped centre, measured row-wise as it stands).
            pop (Any, optional): The population payload ``speciate`` measures
                against; its static shape must match the digested population's.

        Returns:
            Any: ``(P,)`` float32 distances.

        Raises:
            TensorNEATError: If ``pop`` does not have the digested population's
                static shape, or if the centre batch is neither ``1`` nor ``P``.

        Note:
            This is the callable ``_kernels.speciate`` invokes as
            ``distance_fn(state, centre, pop)``: the gated path reaches it from
            both scan bodies, the dense path from every ``assign_members``
            iteration.  Those call sites always hand over a batch-``1`` centre and
            the digested population, so the ``pop_size``-wide branch and both
            rejections in the body are unexercised from inside the package.
            A whole ``(S, P)`` species table is not a ``__call__`` shape either:
            ask for it by name with :meth:`matrix`, so a table can never be
            mistaken for a centre payload.
        """
        if pop is not None:
            self._check_population(pop)
        batch = int(centre[0].shape[0])
        if batch != self.pop_size and batch != 1:
            raise TensorNEATError(
                "DistanceRows: a distance row takes a centre payload of batch 1 "
                f"or pop_size ({self.pop_size}), got batch {batch}; call "
                "matrix() directly for a whole species table"
            )
        return self.row(state, centre)

    def _check_population(self, pop: Any) -> None:
        """Reject a population payload whose static shape the digest cannot serve.

        The digest is the *only* thing the distances are measured against, so the
        payload a caller hands to ``speciate`` is a witness that the two agree:
        a payload with different static shapes cannot be the one the digest was
        prepared from.

        Args:
            pop (Any): The payload ``speciate`` is measuring against.

        Raises:
            TensorNEATError: If a field's static shape differs from the digested
                population's.
        """
        for name, field, expected in zip(
            ("node_keys", "node_attrs", "conn_keys", "conn_attrs"),
            pop,
            self.population,
        ):
            if tuple(int(size) for size in field.shape) != tuple(
                int(size) for size in expected.shape
            ):
                raise TensorNEATError(
                    f"DistanceRows: the measured population's {name} has shape "
                    f"{tuple(int(size) for size in field.shape)} but the digest "
                    f"was prepared from {tuple(int(size) for size in expected.shape)}; "
                    "build the digest from the same payload the scans compare "
                    "against"
                )

    def row(self, state: Any, centre: Any) -> Any:
        """``(P,)`` distance row of one centre payload (batch ``1`` or ``P``).

        Args:
            state (Any): Shared state; its ``distance_centre`` entry selects the
                centre spelling (:func:`._validation.distance_centre_mode`).
            centre (Any): Centre payload of one individual (batch ``1``) or a
                population-shaped one (batch ``pop_size``).

        Returns:
            Any: ``(P,)`` float32 distances.

        Raises:
            TensorNEATError: If the centre batch is neither ``1`` nor ``P``.

        Note:
            A batch-``1`` centre is tiled to ``pop_size`` rows in ``"tiled"``
            mode (the default) and handed on unchanged in ``"single"`` mode,
            where the genome pairs it against the digest's ``pop_size`` rows
            through its dense single-centre route; a ``pop_size``-row centre is
            measured row-wise as it stands in both modes.
        """
        batch = int(centre[0].shape[0])
        if batch != self.pop_size and batch != 1:
            raise TensorNEATError(
                f"DistanceRows: a distance row takes a centre payload of batch 1 "
                f"or pop_size ({self.pop_size}), got batch {batch}"
            )
        if batch == 1 and distance_centre_mode(state) == "tiled":
            # The mode's contract is population-shaped, so a scale-by-one centre
            # is tiled here rather than measured through the genome's dense route.
            centre = centre_payload(
                centre,
                tuple(field[0] for field in centre),
                self.pop_size,
                True,
            )
        return self.genome.measure_distance(state, centre, self.digest)

    def matrix(self, state: Any, centres: Any) -> Any:
        """``(S, P)`` distance matrix of a whole species table (explicit form).

        Args:
            state (Any): Shared state; its ``distance_centre`` entry selects the
                centre spelling and its ``kernel_mode`` the grouping
                (:func:`~tensorneat.genome.operations.distance.grouped.for_state`).
            centres (Any): Centre payload of ``K`` rows -- the ``(S, ...)``
                centre arrays of a speciation phase.

        Returns:
            Any: ``(K, P)`` float32 distances, row ``s`` against the whole
            population.

        Note:
            This is the explicit whole-table form, never reached through
            :meth:`__call__`: a caller holding an ``(S, ...)`` centre table asks
            for it by name, so a table cannot be mistaken for a batch-``1`` centre
            when ``S == 1``.  The one in-package caller is
            ``_kernels.speciate``'s dense path, which measures scan 1's whole
            centre table this way before the reservation scan.

            Both spellings are one fixed-length ``evoxc.scan`` over the table's
            rows: the kernel never sees a ``K * P`` batch, and because a scan
            body's region environment is popped per iteration, only one call's
            transients are alive at a time.  Measuring the same rows at the top
            level of the generation graph would keep all ``K`` of them resident
            -- see `Peak memory` in the module docstring.

            The **grouped** leg is the one ``grouped.for_state`` selects for this
            state (``k > 1``, which only an accelerator state in ``"tiled"`` mode
            answers), and it scans ``K // k`` groups instead of ``K`` rows: one
            iteration gathers the ``k`` centre rows of its group (``k`` is a
            static Python ``int``, so the indices are ``slot * k + j``), merges
            them with ``grouped.centre_batch`` and makes **one**
            ``measure_distance`` call against the ``k``-times tiled digest,
            returning the flat ``(k * P,)`` answer.  The step value stays rank 1
            on purpose: ``evoxc.scan`` stacks ``y`` into ``(groups,) +
            y.shape``, and a ``(k, P)`` ``y`` would make that a rank-3 write the
            StableHLO v1 exporter rejects.  The stacked ``(K // k, k * P)`` table
            is ``reshape`` d into ``(K - K % k, P)`` afterwards, and at most
            ``K % k`` trailing rows are measured singly against the plain digest
            and concatenated -- ``grouped.group_size`` caps ``k`` at the slot
            count, so that tail is never the whole table.
        """
        count = int(centres[0].shape[0])
        tiled = distance_centre_mode(state) == "tiled"
        digest, group = for_state(state, self.digest, species_size=count)

        def body(carry: Any, slot: Any) -> Any:
            """One centre row of the table, measured inside the scan region."""
            rows = tuple(evoxc.gather(field, slot, axis=0) for field in centres)
            return carry, self.genome.measure_distance(
                state,
                centre_payload(self.population, rows, self.pop_size, tiled),
                self.digest,
            )

        if group == 1 or not tiled:
            _, stacked = evoxc.scan(
                body,
                enp.zeros((1,), dtype=evoxc.float32),
                enp.arange(0, count, dtype=evoxc.int32),
            )
            return stacked

        measured = count - count % group

        def group_body(carry: Any, slot: Any) -> Any:
            """``group`` centre rows of the table, measured in one call."""
            base = evoxc.multiply(slot, int_const(group))
            payloads = []
            for offset in range(group):
                index = evoxc.add(base, int_const(offset))
                payloads.append(
                    centre_payload(
                        self.population,
                        tuple(evoxc.gather(field, index, axis=0) for field in centres),
                        self.pop_size,
                        True,
                    )
                )
            return carry, self.genome.measure_distance(
                state, *centre_batch(payloads), digest
            )

        _, merged = evoxc.scan(
            group_body,
            enp.zeros((1,), dtype=evoxc.float32),
            enp.arange(0, measured // group, dtype=evoxc.int32),
        )
        table = enp.reshape(merged, (measured, self.pop_size))
        if measured == count:
            return table
        tail = [
            self.genome.measure_distance(
                state,
                centre_payload(
                    self.population,
                    tuple(evoxc.gather(field, slot, axis=0) for field in centres),
                    self.pop_size,
                    tiled,
                ),
                self.digest,
            )
            for slot in range(measured, count)
        ]
        return enp.concatenate(
            [table] + [enp.reshape(row, (1, self.pop_size)) for row in tail], 0
        )

