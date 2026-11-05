"""Algorithm contract for TensorNEAT on the EvoX Compiler.

``BaseAlgorithm`` is the protocol driven by ``tensorneat.pipeline.Pipeline``: one
whole generation (``ask`` -> ``transform`` -> ``forward``/evaluate -> ``tell``)
becomes a single evoxc graph built by the pipeline's compile site, so every
method below runs inside an active trace and must be written batch-first with
static shapes.  ``show_details`` is the only host-side member.
"""

from typing import Any

from tensorneat.common import State


class BaseAlgorithm:
    """Abstract evolutionary algorithm contract.

    Population is axis 0: every method consumes and returns pytrees whose
    leading axis has length :attr:`pop_size`.

    Notes:
        * Implementations must not branch on traced values and must not read host
          values (``to_numpy``/``.item()``/``int()``) inside
          ``ask``/``transform``/``forward``/``forward_prepared``/``tell``.
        * Shapes are static per construction: population size, ``max_nodes`` and
          ``max_conns`` are fixed at build time, so a shape change forces the
          pipeline to rebuild the evoxc graph.
        * Constructors take plain Python configuration only; nothing in a
          constructor may build an evoxc graph.
    """

    def setup(self, state: State | None = None) -> State:
        """Register this algorithm's tensors onto ``state``.

        Args:
            state: Threaded immutable state, already carrying the genome's,
                problem's and pipeline's keys -- in particular ``randkey``, the
                only random source of a run.  ``None`` means "start from an empty
                :class:`tensorneat.common.State`"; it must be spelled this way
                because a ``State()`` default argument would be evaluated while
                this module is being imported.

        Returns:
            A new :class:`tensorneat.common.State` carrying this algorithm's keys
            (population tensors, generation counter, algorithm sub-state).

        Raises:
            NotImplementedError: Always, in the abstract base class.
        """
        raise NotImplementedError

    def ask(self, state: State) -> Any:
        """Return the population that the problem should evaluate.

        Args:
            state: Current algorithm state.

        Returns:
            A batch-first population pytree.  For NEAT it is the four-array split
            gene payload (``genome.utils.GenomeTensors``): ``node_keys``
            ``(P, max_nodes) int32``, ``node_attrs`` ``(P, max_nodes, A) float32``,
            ``conn_keys`` ``(P, max_conns, key_width) int32`` and ``conn_attrs``
            ``(P, max_conns, W) float32``, where absent rows are ``I_INF`` in the
            key arrays and NaN in the attribute arrays; ``ask`` costs zero ops.

        Notes:
            ``Pipeline._generation_kernel`` calls ``algorithm.ask(current)``
            *after* registering the state's payload on ``current`` and hands the
            result to ``transform`` and to the problem's ``evaluate``, so an
            override is honoured rather than bypassed.  A shipped ``ask`` returns
            ``state.pop`` unchanged (zero ops) and ``tell`` still receives the
            state's population, so an ``ask`` that returns a different population
            must document that the evaluated phenotype is not the one reported or
            checkpointed: ``previous_pop`` is ``state.pop``, not the asked payload.
        """
        raise NotImplementedError

    def transform(self, state: State, pop: Any) -> Any:
        """Decode the whole population into one batch-first execution schedule.

        Args:
            state: Current algorithm state.
            pop: Population pytree returned by :meth:`ask`, individual on axis 0.

        Returns:
            A batch-first ``Transformed`` pytree describing how to execute every
            individual (topological order, incoming-edge table, per-node depth and
            aggregation masks, as defined by the genome); axis 0 is the individual.
        """
        raise NotImplementedError

    def forward(self, state: State, transformed: Any, inputs: Any) -> Any:
        """Run a decoded batch of networks.

        A caller that queries one payload many times should call
        :meth:`prepare_forward` once and :meth:`forward_prepared` per query,
        which is the same computation with the loop-invariant half hoisted.

        Args:
            state: Current algorithm state.
            transformed: Batch-first schedule returned by :meth:`transform`.
            inputs: ``(pop_size, *input_shape)`` input activations.

        Returns:
            ``(pop_size, *output_shape)`` outputs.
        """
        raise NotImplementedError

    def prepare_forward(
        self, state: State, transformed: Any, pop: Any = None, samples: int = 1
    ) -> Any:
        """Derive the loop-invariant operands of one forward pass.

        Whatever the genome and payload need computed once per pass, for reuse
        across many queries of the same batch shape; the returned record is
        opaque to the pipeline.

        Args:
            state: Current algorithm state.
            transformed: Batch-first schedule returned by :meth:`transform`.
            pop: Population pytree the schedule was derived from; ``None`` means
                the algorithm's own current population.
            samples: Static tiling factor of the ``pop_size * samples`` rows the
                pass will be queried with.

        Returns:
            An opaque prepared record for :meth:`forward_prepared`.
        """
        raise NotImplementedError

    def forward_prepared(
        self, state: State, transformed: Any, prepared: Any, inputs: Any
    ) -> Any:
        """Run one forward pass from a :meth:`prepare_forward` record.

        Args:
            state: Current algorithm state.
            transformed: The schedule the record was prepared with.
            prepared: Record returned by :meth:`prepare_forward`.
            inputs: ``(pop_size * samples, *input_shape)`` input activations.

        Returns:
            ``(pop_size * samples, *output_shape)`` outputs.
        """
        raise NotImplementedError

    def tell(self, state: State, fitness: Any) -> State:
        """Consume one generation of fitness and produce the next state.

        Args:
            state: Current algorithm state.
            fitness: ``(pop_size,)`` float32 fitness, maximised; NaN and infinite
                entries are treated as ``-inf`` so a broken topology never wins
                selection.

        Returns:
            A new state holding the next population, the advanced ``randkey``, the
            incremented ``generation`` and the updated species bookkeeping.
        """
        raise NotImplementedError

    def show_details(self, state: State, fitness: Any) -> None:
        """Print per-generation details; the only host-side method.

        Args:
            state: Current algorithm state.
            fitness: ``(pop_size,)`` fitness of the generation just evaluated.
        """
        raise NotImplementedError

    @property
    def num_inputs(self) -> int:
        """Number of network inputs, as a static Python int.

        Returns:
            int: Input arity checked by ``Pipeline`` against
            ``problem.input_shape[-1]``.
        """
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:
        """Number of network outputs, as a static Python int.

        Returns:
            int: Output arity checked by ``Pipeline`` against
            ``problem.output_shape[-1]``.
        """
        raise NotImplementedError

    @property
    def pop_size(self) -> int:
        """Population size, as a static Python int.

        Returns:
            int: Length of axis 0 of every population tensor.
        """
        raise NotImplementedError
