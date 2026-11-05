"""Distance rows over one generation's population digest.

Speciation measures *every* species centre against the *same* population, twice
per generation, so the genome's two-phase distance API serves it directly:
:meth:`~tensorneat.genome.base.BaseGenome.prepare_distance` sorts the population
once per generation into a ``DistanceDigest``, and
:meth:`~tensorneat.genome.base.BaseGenome.measure_distance` turns one centre
genome into one ``(P,)`` distance row without re-sorting the population.

Shape contract
--------------
Every call this module makes to ``genome.measure_distance`` carries a centre
payload whose batch matches the digest it is measured against: ``pop_size`` rows
for the plain digest (row ``p`` the centre paired with population member ``p``)
and ``k * pop_size`` rows for a digest tiled ``k`` times.  ``measure_distance``
rejects every other width, so the two batch forms are the only ones any call site
here may build.

:meth:`DistanceRows.__call__` is single-valued: it accepts a centre payload of
batch ``pop_size`` (measured as it stands) or batch ``1`` (tiled to
``pop_size`` rows and measured in one call) and rejects every other batch width.
It **is** the production route: ``_kernels.speciate`` calls ``distance_fn(...)``
from its per-iteration ``compute_row`` bodies for the gated ``reserve`` and
``assign_members`` scans, and unconditionally for every ``assign_members``
iteration on the dense path (a small ``Pipeline`` run traced 4 ``__call__``
invocations on the gated path and 2 on the dense one, plus one :meth:`matrix`:
``evoxc.scan`` peels step 0 at the enclosing level and traces the body again, so
each scan body is traced twice).
What no in-repo call site reaches through it is only the batch-``1`` tiling
branch and the batch-width rejection, because ``compute_row`` always wraps its
slot's centre with ``single_payload`` (``pop_size`` rows), and
:meth:`_check_population`'s payload rejection, because every call site passes the
four arrays the digest was prepared from.
:meth:`DistanceRows.row` is the same measurement spelled explicitly for one
centre, and :meth:`DistanceRows.matrix` is the explicit whole-table form:
``K // k`` fixed-length ``evoxc.scan`` iterations, each measuring ``k`` centre
rows in one ``genome.measure_distance`` call against a digest tiled ``k`` times
(:data:`GROUP_CHOICES` picks ``k`` statically from the row count), so a whole
``(K, P)`` table is ``K // k`` calls of ``k * pop_size`` rows rather than ``K``
calls of ``pop_size`` rows.
The rows are identical either way, because the compatibility kernel is strictly
row-independent (elementwise per-pair work plus per-row reductions), and the
grouped digest documents the pairing: result row ``j * pop_size + p`` is centre
row ``j * pop_size + p`` against individual ``p``.
``matrix`` is never selected by ``__call__``: a caller that wants a whole table
asks for it by name, which is what the dense path's scan 1 does, and only scan 1.
Scan 2 keeps its per-iteration call inside ``assign_members``: a slot that
*creates* a species measures a carry-dependent ``pop[seed]`` centre, so a
pre-measured second table would have to cover all ``S`` slots to save the kernel
on the live non-creating ones -- the reverted route measured below, which pays
wall time for it.

Nothing here re-implements distance arithmetic: every number comes from the
genome's own kernel, so the rows are bit-identical to
:meth:`~tensorneat.genome.base.BaseGenome.execute_distance` on the same centre and
population, and a grouped table is bit-identical to the per-centre table it
replaces (verified on 10-generation ``xor`` runs at pop 500 and pop 100, 100
arrays each).

Measured cost
-------------
Numpy reference backend, ``species_size = 20``, one configuration per fresh
process, 3 warmup + 20 timed ``Pipeline.step``, seed 42, interleaved rounds so
box drift cancels (medians of three round medians): the dense path's scan-1
table is 164.11 ms / 233.8 MiB / 7 592 dispatched ops at ``P = 500`` with one
centre per ``measure_distance`` call, and 155.23 ms / 253.2 MiB / 6 589 ops with
two centres per call; ``custom_pagie_p500`` moves 1 116.90 ms / ~1 260 MiB /
12 145 ops to 1 053.63 ms / ~1 265 MiB / 10 791 ops.
The distance share (total minus a run whose ``BaseDistance.measure`` returns
zeros) is 3 675 -> 2 755 ops at ``xor_p500`` and 3 641 -> 2 604 at
``custom_pagie_p500``.  One ``measure_distance`` call emits ~92 dispatched ops
regardless of its row width, so the ten calls grouping removes from scan 1 are
920 of them, and ``group_digest`` adds 24 ops once per generation.
``xor_p100`` is untouched: its gated path never calls :meth:`matrix`, so every
variant dispatches 3 699 ops there and the wall time is statistically identical
(43.83 / 43.54 / 44.31 ms against 43.55 / 45.94 / 47.45 ms over three
interleaved rounds).

The group size is two because the dispatch win is bought with element volume: at
``k = 4`` scan 1's call count halves again and ``xor_p500`` falls to 6 004 ops,
but the step reads 189.45 ms / 305.2 MiB (+15 % wall, +30 % peak RSS) and
``custom_pagie_p500`` 1 088.10 ms / 1 318.1 MiB.  The rule applied is the lowest
dispatched-op count that keeps numpy wall time within ~5 % of the ungrouped form
and peak RSS within ~10 % on both dense cases, and ``k = 2`` is the only group
size inside both budgets.  A centre count that ``k`` does not divide falls back
to one centre per call (:meth:`DistanceRows._single_table`), so nothing changes
for an odd table.

Pre-measuring scan 1's table alone is what ships; extending the grouping to scan
2 as well -- the reverted route, which measures a second table after the
reservation scan -- is rejected on wall time: 6 661 ops and 258.96 ms / 274.0 MiB
at ``xor_p500`` (+58 % against the shipped form), 968.79 ms / 1 299.2 MiB / 9 556
ops at ``custom_pagie_p500``.  Its second table covers all ``S`` centres to save
the in-loop kernel on the live non-creating slots, while only a slot that
*creates* a species measures a carry-dependent ``pop[seed]`` centre, so the
pre-measurement re-introduces a whole population-shaped row per slot against the
per-iteration call it removes; it is 84 ms *faster* on ``custom_pagie_p500``, but
the wall-time budget is decided on ``xor_p500``, where it fails outright.

Peak memory
-----------
``evoxc``'s numpy interpreter keeps every SSA result of a block in that block's
environment until the block returns, and pops a nested region's environment per
iteration.  A kernel inside a ``scan`` body therefore pays one region's
transients at a time, while a kernel at the top level keeps its ``(k * pop_size,
2 * max_conns, ...)`` intermediates alive for the rest of the generation.  The
scan is what keeps the grouped block affordable: the grouped scan-1 table takes
the step from 233.8 to 253.2 MiB at ``P = 500`` (about 25 MiB over a
no-distance-kernel floor of ~228 MiB), whereas measuring the same grouped rows at
the top level of the generation graph costs ~226 MiB at that population.  Build
follows the same split, because a loop body is traced once and a static loop over
``S`` centres traces ``S`` copies of the kernel.
Any future hoist out of the region has to report peak RSS and build next to its
wall time.
"""

from __future__ import annotations

from typing import Any

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError

#: Candidate group sizes for one whole-table measurement, largest first.
#:
#: A ``k``-centre block asks the genome for ``k * pop_size`` distances in one
#: ``measure_distance`` call against a digest tiled ``k`` times (see
#: ``tensorneat.genome.operations.distance.group_digest``), which is where the
#: dispatch win of the dense path comes from: two centres per call halves the
#: number of compatibility kernels scan 1 runs.  The group must divide the
#: centre-table row count exactly, so a species size that is not a multiple of
#: two falls back to one centre per call.  Static, so the choice is made at trace
#: time and costs no op.
#:
#: Two rather than four: a ``k * pop_size``-row kernel costs more per row than a
#: ``pop_size``-row one on the numpy interpreter, and the wider block also grows
#: the peak live set.  Measured on ``xor_p500`` (one process per cell, 3 warmup
#: + 20 timed ``Pipeline.step``, interleaved rounds): one centre per call
#: 164.11 ms / 233.8 MiB / 7 592 dispatched ops, ``k = 2`` 155.23 ms / 253.2 MiB
#: / 6 589 ops, ``k = 4`` 189.45 ms / 305.2 MiB / 6 004 ops, i.e. four centres
#: per call is past the ~10 % peak-RSS and ~5 % wall-time budget this route is
#: allowed to spend for its dispatch win.
GROUP_CHOICES = (2, 1)


def _group_size(count: int) -> int:
    """Largest :data:`GROUP_CHOICES` entry that divides ``count`` exactly."""
    for size in GROUP_CHOICES:
        if count % size == 0:
            return size
    return 1


def int_const(value: int) -> Any:
    """0-d ``int32`` graph constant (local to this module's group indexing)."""
    return enp.full((), value, dtype=evoxc.int32)


def iota_i32(length: int) -> Any:
    """``(length,)`` ``int32`` index vector ``[0, 1, ...]``."""
    return enp.arange(length, dtype=evoxc.int32)


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
        pop_size (int): Static population size ``P`` (the digest's batch and the
            row count of every centre payload this class builds).

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
            centre (Any): Centre payload of batch ``pop_size`` (a
                population-shaped centre, measured row-wise as it stands) or
                batch ``1`` (one centre, tiled to ``pop_size`` rows).
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
            iteration.  Those call sites always hand over a population-shaped
            centre and the digested population, so the batch-``1`` tiling branch
            and both rejections in the body are unexercised from inside the
            package.
            A whole ``(S, P)`` species table is not a ``__call__`` shape either:
            ask for it by name with :meth:`matrix`, so a table can never be
            mistaken for a population-shaped centre.
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
            state (Any): Shared state.
            centre (Any): Centre payload; a batch of ``1`` is tiled to
                ``pop_size`` identical rows, a batch of ``pop_size`` is already
                population-shaped and measured as it stands.

        Returns:
            Any: ``(P,)`` float32 distances.

        Raises:
            TensorNEATError: If the centre batch is neither ``1`` nor ``P``.
        """
        batch = int(centre[0].shape[0])
        if batch == self.pop_size:
            prepared = centre
        elif batch == 1:
            prepared = self.population_shaped(centre)
        else:
            raise TensorNEATError(
                f"DistanceRows: a distance row takes a centre payload of batch 1 "
                f"or pop_size ({self.pop_size}), got batch {batch}"
            )
        return self.genome.measure_distance(state, prepared, self.digest)

    def population_shaped(self, centre: Any) -> Any:
        """Tile a one-row centre payload to exactly ``pop_size`` identical rows.

        Args:
            centre (Any): Centre payload of batch ``1``.

        Returns:
            Any: ``(pop_size, *rest)`` payload in which every row repeats the
            centre row, which is the shape ``measure_distance`` pairs against the
            digest.

        Note:
            One rank-raising ``evoxc.broadcast`` per field: its target rank is one
            above a batch-``1`` field's, which is the form the emitter accepts, so
            the whole tiling is a single IR op instead of a reshape/broadcast/
            reshape triple.  A centre of any other batch width conflicts with the
            target shape and raises ``evoxc``'s ``ShapeError`` at trace time --
            a wider centre is never a legal ``measure_distance`` input either.
        """
        return type(centre)(
            *(
                evoxc.broadcast(field, (self.pop_size,) + tuple(field.shape)[1:])
                for field in centre
            )
        )

    def matrix(self, state: Any, centres: Any) -> Any:
        """``(S, P)`` distance matrix of a whole species table (explicit form).

        Args:
            state (Any): Shared state.
            centres (Any): Centre payload of ``K`` rows -- the ``(S, ...)``
                centre arrays of a speciation phase.

        Returns:
            Any: ``(K, P)`` float32 distances, row ``s`` against the whole
            population.

        Note:
            This is the explicit whole-table form, never reached through
            :meth:`__call__`: a caller holding an ``(S, ...)`` centre table asks
            for it by name, so a table cannot be mistaken for a population-shaped
            centre when ``S == P``.  The one in-package caller is
            ``_kernels.speciate``'s dense path, which measures scan 1's whole
            centre table this way before the reservation scan; scan 2 keeps its
            per-iteration call, because only a species-creating slot needs a
            carry-dependent centre.
            See :meth:`_table` for the measurement itself.
        """
        return self._table(state, centres)

    def _table(self, state: Any, centres: Any) -> Any:
        """Measure a whole centre table as ``(K, P)``, ``k`` centres per call.

        Args:
            state (Any): Shared state.
            centres (Any): Centre payload of ``K`` rows.

        Returns:
            Any: ``(K, P)`` float32 distances.

        Note:
            ``K // k`` iterations of one fixed-length ``evoxc.scan``, ``k``
            being :func:`_group_size` of ``K``.  Each iteration gathers the
            block's ``k`` centre rows, expands them to ``k * pop_size`` rows in
            **group-major** order (block row ``j`` repeated ``pop_size`` times,
            which is the pairing a grouped digest documents: result row
            ``j * pop_size + p`` is centre row ``j * pop_size + p`` against
            individual ``p``), makes one ``genome.measure_distance`` call
            against the digest tiled ``k`` times, and returns its ``(k, P)``
            view; the scan stacks those to ``(K // k, k, P)``, which reshapes
            to ``(K, P)`` in table order.
            For ``k == 1`` the plain digest and the one-centre-per-iteration
            spelling are kept (:meth:`_single_table`): tiling a digest once
            would be a no-op, and the scan's per-iteration region is what bounds
            the kernel's live set (see `Peak memory` in the module docstring).
            The grouped digest is built here, from the generation's plain
            digest, and belongs to this trace; nothing is cached on ``self``
            across calls.
        """
        count = int(centres[0].shape[0])
        group = _group_size(count)
        if group == 1:
            return self._single_table(state, centres)
        batch = int_const(group)
        spanning = iota_i32(group)
        # Function-local: importing ``tensorneat.genome`` at module scope would
        # make ``algorithm`` depend on the sibling node at package import time.
        from tensorneat.genome.operations.distance import group_digest

        grouped = group_digest(self.digest, group)

        def body(carry: Any, block_index: Any) -> Any:
            """One ``k``-centre block of the table, measured inside the scan."""
            index = evoxc.add(evoxc.multiply(block_index, batch), spanning)
            block = type(centres)(
                *(evoxc.gather(field, index, axis=0) for field in centres)
            )
            expanded = type(centres)(
                *(self._group_shaped(field, group) for field in block)
            )
            rows = self.genome.measure_distance(state, expanded, grouped)
            return carry, evoxc.reshape(rows, (group, self.pop_size))

        _, stacked = evoxc.scan(
            body,
            enp.zeros((1,), dtype=evoxc.float32),
            iota_i32(count // group),
        )
        return evoxc.reshape(stacked, (count, self.pop_size))

    def _group_shaped(self, block: Any, group: int) -> Any:
        """Expand a ``(k, ...)`` centre block to ``k * pop_size`` group-major rows.

        Args:
            block (Any): One split gene array of ``k`` centre rows, rank above 1.
            group (int): Static ``k`` of the block.

        Returns:
            Any: ``(group * pop_size, ...)`` array whose row ``j * pop_size + p``
            is block row ``j``.

        Note:
            One ``reshape`` + one rank-raising ``evoxc.broadcast`` + one
            ``reshape`` per field: broadcasting an axis of length ``1`` up to
            ``pop_size`` repeats the row, and the trailing flattening puts the
            copies in the order a grouped digest pairs.
        """
        rest = tuple(block.shape)[1:]
        widened = evoxc.reshape(block, (group, 1) + rest)
        return evoxc.reshape(
            evoxc.broadcast(widened, (group, self.pop_size) + rest),
            (group * self.pop_size,) + rest,
        )

    def _single_table(self, state: Any, centres: Any) -> Any:
        """``(K, P)`` table measured one centre per scan iteration (``k == 1``)."""
        count = int(centres[0].shape[0])

        def body(carry: Any, slot: Any) -> Any:
            """One centre row of the table, measured inside the scan region."""
            centre = type(centres)(
                *(
                    evoxc.gather(field, evoxc.reshape(slot, (1,)), axis=0)
                    for field in centres
                )
            )
            return carry, self.genome.measure_distance(
                state, self.population_shaped(centre), self.digest
            )

        _, stacked = evoxc.scan(
            body,
            enp.zeros((1,), dtype=evoxc.float32),
            iota_i32(count),
        )
        return stacked
