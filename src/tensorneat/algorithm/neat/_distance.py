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
``speciate`` on the numpy reference backend, ``species_size=20``, medians of
three runs of five after four warm generations, same box:

* ``P = 100``: 30.7-32.1 ms with the old ``K = 4`` centre grouping, 31.6-33.3 ms
  with one call per centre (parity, within the box's drift);
* ``P = 500``: 235.0-237.5 ms before, 229.5-240.0 ms now (parity);
* ``P = 1000``: 439.8-441.8 ms before, 387.3-391.4 ms now (about 12 % faster).

The per-centre call shape itself is what the population-shaped contract costs:
measured with the rows still pre-computed before each scan (``2 * species_size``
distance trace sites and scan 2's per-iteration ``create`` ``cond``, 11 111 graph
ops against 3 353), it was +18 % at ``P = 500``, because four-centre calls
amortise the interpreter's per-op overhead over four populations.  Measuring the
same rows inside the scan bodies instead -- one trace site per phase, 1 361 ops,
no ``cond`` -- wins it back and then some: the wide batches were also
materialising ``species_size * pop_size`` rows per digest leaf, which is the cost
that dominated at ``P = 1000``.  The grouping and the pre-measurement were the
expensive parts, not the per-centre call.
"""

from __future__ import annotations

from typing import Any

import evoxc
import evoxc.numpy as enp

from tensorneat.common.errors import TensorNEATError

#: Number of centre rows one :meth:`DistanceRows.matrix_blocks` call measures at
#: a time.  Four is the measured optimum of the numpy reference interpreter: the
#: compatibility kernel is dispatched once per *call*, so a wider batch amortises
#: the per-op overhead over four populations, while a still wider one starts
#: paying for array traffic that the reused digest cannot absorb (one centre per
#: call 11.1 us/pair at ``P = 500``, four per call 6.7, twenty per call 11.9).
CENTRES_PER_BLOCK = 4

#: ``DistanceDigest`` fields that are *not* batch-first and must never be tiled
#: with the population axis: the two static scaffolds carry no ``shape`` at all,
#: and ``tail_counts`` is ``(2, P)`` -- its batch axis is the *last* one and it is
#: only read by the batch-1 single-centre fast path, which a block never takes.
_UNTILED_DIGEST_FIELDS = frozenset({"node_layout", "conn_layout", "tail_counts"})


def _slice_rows(tensor: Any, start: int, count: int) -> Any:
    """Slice ``count`` consecutive rows off one batch-first tensor.

    Args:
        tensor (Any): Traced tensor of shape ``(K, *rest)``.
        start (int): Static first row.
        count (int): Static number of rows.

    Returns:
        Any: ``(count, *rest)`` view of ``tensor``.
    """
    rank = len(tensor.shape)
    slicer = (slice(start, start + count),) + (slice(None),) * (rank - 1)
    return tensor[slicer]


def _rows(payload: Any, start: int, count: int) -> Any:
    """Slice ``count`` consecutive rows out of every field of a payload.

    Args:
        payload (Any): Payload whose fields are batch-first ``(K, *rest)``.
        start (int): Static first row.
        count (int): Static number of rows.

    Returns:
        Any: Payload of the same type with ``(count, *rest)`` fields.
    """
    return type(payload)(*(_slice_rows(field, start, count) for field in payload))


def _pad_last(payload: Any, extra: int) -> Any:
    """Append ``extra`` copies of a payload's last row.

    Used to bring a short final block up to the memoised block width: the extra
    rows measure a repeated centre and their distances are dropped by the
    caller's final slice, so the padded block is value-identical to a shorter
    call.

    Args:
        payload (Any): Payload of ``(K, *rest)`` fields with ``K >= 1``.
        extra (int): Static number of rows to append.

    Returns:
        Any: Payload with ``(K + extra, *rest)`` fields, the last row repeated.
    """
    return type(payload)(
        *(
            evoxc.concatenate(
                [field]
                + [_slice_rows(field, int(field.shape[0]) - 1, 1) for _ in range(extra)],
                axis=0,
            )
            for field in payload
        )
    )


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
        self._tiled: dict[int, Any] = {}
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


    def tiled_digest(self, block: int) -> Any:
        """The population digest repeated ``block`` times along its batch axis.

        Args:
            block (int): Static number of centre rows a block measures.

        Returns:
            Any: ``DistanceDigest`` whose every batch-first leaf is
            ``block * pop_size`` rows long, memoised per ``block`` so both
            speciation phases of one generation share one tiling.

        Note:
            A *row-wise* measurement pairs centre row ``k * pop_size + p`` with
            digest row ``k * pop_size + p``, so repeating the population
            ``block`` times turns a block of ``block`` centre rows, each tiled to
            ``pop_size`` rows, into one ordinary row-wise call -- the pairing is
            ``(centre k, individual p)`` exactly as a single-centre call pairs it.
        """
        cached = self._tiled.get(block)
        if cached is None:
            cached = self._tile_digest(self.digest, block)
            self._tiled[block] = cached
        return cached

    def _tile_digest(self, digest: Any, block: int) -> Any:
        """Tile every batch-first leaf of ``digest`` ``block`` times along axis 0.

        Args:
            digest (Any): ``DistanceDigest`` of ``pop_size`` genomes.
            block (int): Static repeat count.

        Returns:
            Any: The same digest type with ``block * pop_size`` genomes per
            tiled leaf, every copy equal to the original population.

        Note:
            Only leaves whose leading axis *is* the population axis are tiled
            (``shape[0] == pop_size``); the static scaffolds and ``tail_counts``
            are left as they are -- see ``_UNTILED_DIGEST_FIELDS``.
        """
        replacements = {}
        for name, value in zip(digest._fields, digest):
            if name in _UNTILED_DIGEST_FIELDS:
                continue
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) < 1 or int(shape[0]) != self.pop_size:
                continue
            replacements[name] = evoxc.concatenate([value] * block, axis=0)
        return digest._replace(**replacements)

    def matrix_blocks(
        self, state: Any, centres: Any, block: int = CENTRES_PER_BLOCK
    ) -> Any:
        """``(K, P)`` distance matrix measured in blocks of ``block`` centres.

        Args:
            state (Any): Shared state.
            centres (Any): Centre payload of ``K`` rows -- the ``(S, ...)``
                centre arrays of one speciation phase.
            block (int, optional): Static centres per ``measure_distance`` call
                (default :data:`CENTRES_PER_BLOCK`).

        Returns:
            Any: ``(K, P)`` float32 distances, row ``k`` measured against the
            whole population.

        Note:
            One ``measure_distance`` call per static block of ``block`` centre
            rows, each centre tiled to ``pop_size`` rows and paired with
            :meth:`tiled_digest` at the same total batch width.  A short final
            block is padded with its own last centre (whose distances the final
            slice drops), so every call reuses the memoised block-wide digest.
            The compatibility kernel is row-independent -- elementwise per-pair
            ops plus per-row reductions over the gene axis -- so a
            ``(block * pop_size,)`` call returns exactly the floats ``block``
            population-shaped calls do.
        """
        count = int(centres[0].shape[0])
        groups = []
        for start in range(0, count, block):
            take = min(block, count - start)
            chunk = _rows(centres, start, take)
            if take < block:
                chunk = _pad_last(chunk, block - take)
            prepared = type(chunk)(
                *(repeat_leading(field, self.pop_size) for field in chunk)
            )
            flat = self.genome.measure_distance(
                state, prepared, self.tiled_digest(block)
            )
            groups.append(evoxc.reshape(flat, (block, self.pop_size)))
        matrix = groups[0] if len(groups) == 1 else evoxc.concatenate(groups, axis=0)
        if count % block == 0:
            return matrix
        return evoxc.slice(matrix, (0, 0), (count, self.pop_size))
