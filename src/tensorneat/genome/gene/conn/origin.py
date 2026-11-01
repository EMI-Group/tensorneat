"""Origin (NEAT-paper) connection gene: ``DefaultConn`` plus a historical marker.

The weight machinery is inherited unchanged; this family adds one ``int32`` key
column and changes the genetic-operator semantics (whole-gene crossover).
Batch-first: the marker is read from the key columns, so alignment stays an exact
integer comparison and never touches the float attributes.
"""

import evoxc

from ....common import State
from .default import DefaultConn


class OriginConn(DefaultConn):
    """Connection gene of the original NEAT paper (EMI-Group/tensorneat issue 11).

    Key columns are ``(input_index, output_index, historical_marker)``; the single
    float32 attribute column is the inherited ``weight``. The historical marker is
    the NEAT-paper gene-alignment id: connections created by the same historical
    split share a marker and are homologous during crossover/distance even when
    their endpoints were re-routed.

    Differences from :class:`.default.DefaultConn`: key column 2 carries the marker,
    :meth:`crossover` picks a whole gene with one draw, and :meth:`get_historical_marker`
    exposes the marker. Reference: https://github.com/EMI-Group/tensorneat/issues/11
    """

    fixed_attrs: tuple[str, ...] = ("input_index", "output_index", "historical_marker")

    def crossover(
        self,
        state: State,
        key: evoxc.SymbolicTensor,
        attrs1: evoxc.SymbolicTensor,
        attrs2: evoxc.SymbolicTensor,
    ) -> evoxc.SymbolicTensor:
        """Pick one parent's whole attribute vector for each gene pair.

        A ``(*shape, 1)`` broadcast of one normal draw per gene row selects between
        the two parents' attribute rows, so an offspring gene inherits its weight
        from exactly one parent (NEAT-paper gene-level inheritance).

        Args:
            state: Host configuration container.
            key: Rank-0 ``evoxc.random`` key for the whole batch.
            attrs1: Attribute rows of the first parent, ``(*shape, 1)`` float32 for
                this family.
            attrs2: Key-aligned attribute rows of the second parent, same shape.

        Returns:
            Child attribute rows, ``(*shape, attr_width)`` float32, each row copied
            verbatim from one parent.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "one scalar draw per gene row (broadcast over attr_width) selects the whole "
            "attribute vector from attrs1 or attrs2"
        )

    def get_historical_marker(
        self, state: State, key_row: evoxc.SymbolicTensor
    ) -> evoxc.SymbolicTensor:
        """Read the historical marker, key column 2 of this family.

        Args:
            state: Host configuration container.
            key_row: ``(*shape, 3)`` int32 key columns; column 2 is the marker and is
                ``I_INF`` for absent connection rows.

        Returns:
            ``(*shape,)`` int32 historical markers, compared exactly (never a float
            attribute), which is what makes crossover/distance alignment an integer
            match.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("return key column 2 -> (*shape,) int32 historical markers")

    def repr(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
        precision: int = 2,
    ) -> str:
        """Render one connection gene as ``in / out / historical_marker / weight``.

        Args:
            state: Host configuration container.
            key_row: ``(3,)`` int32 key columns of a single gene.
            attr_row: ``(1,)`` float32 attribute columns of the same gene.
            precision: Decimal places for the weight.

        Returns:
            A single-line description, e.g.
            ``OriginConn(in: 1, out: 4, historical_marker: 7, weight: 0.50)``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError("format the in/out keys, the marker and the rounded weight")

    def to_dict(
        self,
        state: State,
        key_row: evoxc.SymbolicTensor,
        attr_row: evoxc.SymbolicTensor,
    ) -> dict:
        """Convert one connection gene to the dict used by host code generation.

        Args:
            state: Host configuration container.
            key_row: ``(3,)`` int32 key columns of a single gene.
            attr_row: ``(1,)`` float32 attribute columns of the same gene.

        Returns:
            ``{"in": int, "out": int, "historical_marker": int, "weight": float}``.

        Raises:
            NotImplementedError: Skeleton stub.
        """
        raise NotImplementedError(
            "map the in/out keys, the marker key column and the weight column to a dict"
        )
