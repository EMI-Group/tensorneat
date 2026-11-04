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
:meth:`DistanceRows.row` is the same measurement spelled explicitly for one
centre, and :meth:`DistanceRows.matrix` is the explicit whole-table form: it
concatenates one ``row`` call per centre row, so an ``(S, P)`` table is ``S``
calls of ``pop_size`` rows rather than one ``S * P`` batch.  The rows are
identical either way, because the compatibility kernel is row-independent, and a
per-centre call keeps the transient footprint of the kernel at one population
instead of ``S`` of them.  ``matrix`` is never selected by ``__call__``: a
caller that wants a whole table asks for it by name.

Nothing here re-implements distance arithmetic: every number comes from the
genome's own kernel, so the rows are bit-identical to
:meth:`~tensorneat.genome.base.BaseGenome.execute_distance` on the same centre and
population.

Measured cost
-------------
``speciate`` on the numpy reference backend, ``species_size=20``, one
configuration per fresh process, ``Pipeline.step`` medians of one warmup plus
twenty timed generations after three setup generations, interleaved A/B against
the pre-change tree on the same box (which drifts by tens of percent between
sessions, so read the paired ratios):

* ``P = 100`` (gated path, unchanged): 43.0 ms before, 42.9 ms now;
* ``P = 500`` (dense path): 266.6 ms before, 230.9 ms now (about 13 % faster,
  and the best of three paired rounds in each direction);
* ``P = 1000`` (dense path): 463.0 ms before, 451.5 ms now (parity).

The win is scan 1's: ``speciate`` measures that centre table with :meth:`matrix`
before the scan instead of one population-shaped call inside every iteration, so
the compatibility kernel leaves the ``while`` body (59 ops per iteration instead
of 228).  The *number* of executed ops is unchanged -- the same kernel runs once
per centre either way -- so the gain is the interpreter's per-iteration region
and carry overhead, not the arithmetic.

Two variants were measured and rejected.  Pre-measuring *both* scans needs a
``cond`` for scan 2's data-dependent ``pop[seed]`` centre and costs ~+50 % peak
RSS (832 vs 556 MiB at ``P = 500``, 1540 vs 1021 MiB at ``P = 1000``) for no
timed gain.  Grouping several centres into one ``measure_distance`` call (a
tiled digest and a ``block * pop_size``-row centre batch) is 12 % cheaper than
one call per centre in isolation at ``P = 500`` (66.6 vs 75.4 ms for twenty
rows), but *slower* end to end: the wider batches materialise ``block`` times
bigger temporaries, which costs more than the dispatch it saves (block of 4:
266 ms at ``P = 500`` against 231 ms for one centre per call, and 472 vs 383 ms
at ``P = 1000``).
"""

from __future__ import annotations

from typing import Any

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError


def _rows(payload: Any, start: int, count: int) -> Any:
    """Slice ``count`` consecutive rows out of every field of a payload.

    Args:
        payload (Any): Payload whose fields are batch-first ``(K, *rest)``.
        start (int): Static first row.
        count (int): Static number of rows.

    Returns:
        Any: Payload of the same type with ``(count, *rest)`` fields.
    """
    rank = len(payload[0].shape)
    slicer = (slice(start, start + count),) + (slice(None),) * (rank - 1)
    return type(payload)(*(field[slicer] for field in payload))


def repeat_leading(x: Any, repeats: int) -> Any:
    """Repeat every row of a batch-first tensor ``repeats`` times along axis 0.

    Args:
        x (Any): Traced tensor of shape ``(K, *rest)``.
        repeats (int): Static repeat count, the population size.

    Returns:
        Any: ``(K * repeats, *rest)`` with rows ``k * repeats .. k * repeats +
        repeats - 1`` all equal to ``x[k]`` -- so a ``(1, ...)`` input becomes a
        population-shaped payload whose every row is that one centre.
    """
    shape = tuple(int(size) for size in x.shape)
    spread = evoxc.reshape(x, (shape[0], 1) + shape[1:])
    return evoxc.reshape(
        enp.broadcast_to(spread, (shape[0], repeats) + shape[1:]),
        (shape[0] * repeats,) + shape[1:],
    )


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
            A whole ``(S, P)`` species table is not a ``__call__`` shape: ask
            for it by name with :meth:`matrix`, so a table can never be mistaken
            for a population-shaped centre.
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
        """Tile a centre payload to exactly ``pop_size`` identical rows.

        Args:
            centre (Any): Centre payload of ``K`` rows.

        Returns:
            Any: ``(pop_size, *rest)`` payload in which every row repeats the
            centre row, which is the shape ``measure_distance`` pairs against the
            digest.
        """
        return type(centre)(*(repeat_leading(field, self.pop_size) for field in centre))

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
            centre when ``S == P``.
            One ``row`` call per centre row, concatenated along axis ``0``: each
            call carries a population-shaped centre, so the kernel always sees
            the batch width the digest was prepared with.  Concatenating rows of
            single calls keeps every transient at one population instead of
            ``K`` of them; the static row loop is trace-time only and costs no
            graph op.
        """
        count = int(centres[0].shape[0])
        rows = [
            evoxc.reshape(self.row(state, _rows(centres, slot, 1)), (1, self.pop_size))
            for slot in range(count)
        ]
        if count == 1:
            return rows[0]
        return evoxc.concatenate(rows, axis=0)
