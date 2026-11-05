"""Distance rows over one generation's population digest.

Speciation measures *every* species centre against the *same* population, twice
per generation, so the genome's two-phase distance API serves it directly:
:meth:`~tensorneat.genome.base.BaseGenome.prepare_distance` sorts the population
once per generation into a ``DistanceDigest``, and
:meth:`~tensorneat.genome.base.BaseGenome.measure_distance` turns one centre
genome into one ``(P,)`` distance row without re-sorting the population.

Shape contract
--------------
Every call this module makes to ``genome.measure_distance`` carries a
**population-shaped** centre payload: exactly ``pop_size`` rows, row ``p`` being
the centre measured against population member ``p``, next to the digest of the
same batch width.  ``measure_distance`` accepts no other width besides a
broadcastable ``1``, so the population-shaped form is the one shape that works
for both a single centre and a whole species table without a second digest
tiling.

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
centre, and :meth:`DistanceRows.matrix` is the explicit whole-table form: one
such call per centre row, run as a single fixed-length ``evoxc.scan`` over the
``S`` centre rows, so an ``(S, P)`` table is ``S`` calls of ``pop_size`` rows
rather than one ``S * P`` batch.  The rows are identical either way, because the
compatibility kernel is row-independent, and one centre per call keeps the
kernel's transient footprint at one population instead of ``S`` of them, which
the scan body then releases again before the next iteration.  ``matrix`` is
never selected by ``__call__``: a caller that wants a whole table asks for it by
name, which is what the dense path's scan 1 does.

Nothing here re-implements distance arithmetic: every number comes from the
genome's own kernel, so the rows are bit-identical to
:meth:`~tensorneat.genome.base.BaseGenome.execute_distance` on the same centre and
population.

Measured cost
-------------
Numpy reference backend, ``species_size=20``, one configuration per fresh
process, ``benchmarks/backend_measure.py --backend numpy --generations 20
--warmup 3`` (its numbers are ``Pipeline.step``): one population-shaped call per
centre is 160.9-162.2 ms per generation and 233.6 MiB peak RSS at ``P = 500``
with a 0.33 s build, and 314.7 ms / 391.7 MiB / 0.40 s at ``P = 1000``.  The
table costs exactly one ``measure_distance`` call per centre row; what the scan
buys is the *region* around those calls, not fewer of them.

Measuring the same rows at the top level of the generation graph instead reads
199.8 ms / 454.4 MiB / 0.56 s at ``P = 500`` and 339.5 ms / 809.4 MiB / 0.65 s
at ``P = 1000``: slower and roughly twice the footprint, for the reason in
`Peak memory` below.

Pre-measuring *both* speciation scans needs a ``cond`` for scan 2's
data-dependent ``pop[seed]`` centre and costs ~50 % more peak RSS for no timed
gain.

Grouping several centres into one ``measure_distance`` call is rejected: at
``P = 500`` / ``species_size = 20``, one centre per call is 160.9-162.2 ms per
generation and 233.6 MiB, ``k = 2`` is 219.7-280.5 ms and 273.8 MiB, ``k = 4``
is 208.8-219.5 ms and 353.1 MiB, and ``k = 10`` / ``k = 20`` are ~260 ms at
593 / 753 MiB.  The ``(rows, 2 * max_conns)`` kernel intermediates grow with
``k`` and the compatibility kernel is memory-bound at that width, so the wider
call's per-element cost exceeds the interpreter dispatch it saves: dispatches
fall 8061 -> 4885 at ``k = 4`` against a 28 % wall-time increase, with
``argsort`` 25.5 -> 37.0 ms, ``gather`` 21.5 -> 36.1 ms, ``cast`` 8.0 -> 17.7 ms
and ``reshape`` 4.3 -> 8.9 ms.  ``_kernels.speciate`` therefore keeps one
population-shaped call per centre, and
``tensorneat.genome.operations.distance.group_digest`` -- the tested grouped
route -- stays unwired here: it is an option for a dispatch-bound backend, not
for the numpy default.

Peak memory
-----------
``evoxc``'s numpy interpreter keeps every SSA result of a block in that block's
environment until the block returns, and pops a nested region's environment per
iteration.  A kernel inside a ``scan`` body therefore pays one region's
transients at a time, while a kernel at the top level keeps its ``(pop_size,
2 * max_conns, ...)`` intermediates alive for the rest of the generation.  The
whole-table scan adds ~12 MiB over the floor of a generation with no distance
kernel at all at ``P = 500`` (228.2 -> 240.5 MiB) and ~24 MiB at ``P = 1000``
(368.2 -> 391.7 MiB), where the top-level form adds 226 / 441 MiB.  Build
follows the same split, because a loop body is traced once and a static loop
over ``S`` centres traces ``S`` copies of the kernel.
Any future hoist out of the region has to report peak RSS and build next to its
wall time.
"""

from __future__ import annotations

from typing import Any

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError


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
            centre table this way before the reservation scan.
            One population-shaped ``measure_distance`` call per centre row, run
            as one fixed-length ``evoxc.scan`` over the ``K`` centre rows: the
            kernel never sees a ``K * P`` batch, and because a scan body's region
            environment is popped per iteration, only one centre's transients are
            alive at a time.  Measuring the same rows at the top level of the
            generation graph would keep all ``K`` of them resident -- see
            `Peak memory` in the module docstring.
        """
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
            enp.arange(0, count, dtype=evoxc.int32),
        )
        return stacked
