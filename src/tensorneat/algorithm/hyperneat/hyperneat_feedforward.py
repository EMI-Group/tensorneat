"""HyperNEAT with a feedforward substrate, decoded by a fixed ``DefaultGenome``.

This variant shares the entire CPPN-to-substrate decode of
:mod:`tensorneat.algorithm.hyperneat.hyperneat`, and differs in exactly three places: the substrate
executor is a :class:`~tensorneat.genome.DefaultGenome` (feedforward, no recurrent activation
horizon), the constructor does not accept ``activate_time``, and construction requires a substrate
whose ``connection_type`` is ``"feedforward"`` (``MLPSubstrate`` and friends).

Nothing is re-implemented here: the decode still fuses the query axis into the population axis (so no
``vmap`` is reintroduced) and the substrate schedule returned by ``hyper_genome.transform`` simply has
the feedforward shape -- a topological order plus per-node incoming-edge tables instead of the
recurrent activation loop -- and its layout stays genome-owned.  One decode path, one place to be
correct.
"""

from typing import Callable

from tensorneat.common import ACT, AGG
from tensorneat.genome import DefaultGenome

from ..neat import NEAT
from .hyperneat import HyperNEAT
from .substrate import BaseSubstrate


class HyperNEATFeedForward(HyperNEAT):
    """HyperNEAT whose substrate network is feedforward.

    Attributes:
        substrate (BaseSubstrate): Host-only geometry with ``connection_type == "feedforward"``.
        neat (NEAT): The CPPN algorithm, shared with the recurrent variant.
        hyper_genome (DefaultGenome): Dense fixed-topology feedforward executor with ``max_nodes ==
            substrate.nodes_cnt`` and ``max_conns == substrate.conns_cnt``.
        weight_threshold (float): Dead-zone half-width of the CPPN-to-weight map.
        max_weight (float): Saturating magnitude of the mapped weight.
        aggregation (Callable): Fixed substrate-node aggregation.
        activation (Callable): Fixed substrate-node activation.
        output_transform (Callable): Output activation of the substrate network.

    Note:
        ``setup``, ``ask``, ``transform``, ``forward``, ``tell``, ``show_details``, ``num_inputs``,
        ``num_outputs`` and ``pop_size`` are inherited unchanged -- including
        ``transform``'s decode algebra (static ``(P, Q, C)`` query grid, ``C = neat.num_inputs``,
        query axis fused into the
        population axis, weight map into ``[-max_weight, max_weight]``, gene payload assembled by the
        gene-level builders) and ``forward``'s trailing bias column.  Only the executor genome
        differs, so this file declares no kernels of its own.
    """

    #: Feedforward executor genome (the recurrent variant uses ``RecurrentGenome``).
    _genome_cls = DefaultGenome

    #: A non-feedforward substrate would make the topological schedule ill-defined.
    _required_connection_type = "feedforward"

    def __init__(
        self,
        substrate: BaseSubstrate,
        neat: NEAT,
        weight_threshold: float = 0.3,
        max_weight: float = 5.0,
        aggregation: Callable = AGG.sum,
        activation: Callable = ACT.sigmoid,
        output_transform: Callable = ACT.sigmoid,
    ):
        """Build the feedforward HyperNEAT decoder.

        The signature mirrors :meth:`HyperNEAT.__init__` without ``activate_time``: a feedforward
        substrate has no recurrent activation horizon, and the inherited default is never forwarded
        to the genome (see :meth:`_genome_extra_kwargs`).

        Args:
            substrate: Host-only substrate geometry with ``connection_type == "feedforward"``.
            neat: The CPPN algorithm; its ``num_inputs`` must equal
                ``substrate.query_coors.shape[1]`` and its ``num_outputs`` must be ``1``.
            weight_threshold: Dead-zone half-width, ``0 <= weight_threshold < 1``.
            max_weight: Saturating weight magnitude, ``> 0``.
            aggregation: Fixed substrate-node aggregation ``(z, valid_mask) -> z``.
            activation: Fixed substrate-node activation.
            output_transform: Output activation of the substrate network.

        Raises:
            TensorNEATError: For every condition listed on :meth:`HyperNEAT.__init__`, plus a
                substrate whose ``connection_type`` is not ``"feedforward"``.

        Note:
            Validation and attribute assignment live in the inherited ``__init__``, driven by
            ``_genome_cls``, ``_required_connection_type`` and ``_genome_extra_kwargs()``; this
            override exists only to drop ``activate_time`` from the public signature.
        """
        raise NotImplementedError

    def _genome_extra_kwargs(self) -> dict:
        """Return no extra genome kwargs: a feedforward genome has no recurrent horizon.

        Returns:
            dict: ``{}``, in contrast to the recurrent variant's ``{"activate_time":
            self.activate_time}``.
        """
        raise NotImplementedError


__all__ = ["HyperNEATFeedForward"]
