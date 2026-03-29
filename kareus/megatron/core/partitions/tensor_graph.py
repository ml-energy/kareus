"""Core data structures for the graph-based partition system.

This module defines the foundational types used by ForwardPartition and
BackwardPartition for automatic tensor routing via named channels and
auto-generated tensor IDs.

Hierarchy:
    Specs (ComputeOpSpec / CommunicationOpSpec)
      → TensorGraphBuilder.add_op()
        → Concrete ops (ComputeOp / CommunicationOp) with wired TensorPorts
          → TensorGraph (the built graph)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


class CommunicationType(Enum):
    """Types of collective communication inserted between partitions."""

    ALL_REDUCE = auto()  # 1 input, 1 output (TP)
    ALL_GATHER_KV = auto()  # 2 inputs (k, v), 2 outputs (CP)
    REDUCE_SCATTER_KV = auto()  # 2 inputs (grad_k, grad_v), 2 outputs (CP)


@dataclass
class Channel:
    """Named connection point on an operator.

    Maps a semantic name to a positional port index used by
    ``fuser_forward`` / ``fuser_backward``.

    Conventions:
        - ``Channel(0, "main")`` — primary hidden_states tensor
        - Additional channels carry side-channel tensors
          (key, value, bias, residual, rotary_pos_emb, etc.)
        - Channels **persist** in the TensorGraphBuilder registry
          until overwritten by a later operator.
    """

    port_idx: int
    name: str


@dataclass
class TensorPort:
    """A single input or output port with an auto-generated tensor ID.

    The ``tensor_id`` is assigned by :class:`TensorGraphBuilder` during
    graph construction and used at runtime to route tensors through
    ``TensorStore``.
    """

    port_idx: int
    tensor_id: Optional[str] = None  # e.g. "t_0", "t_1" — assigned by builder


@dataclass
class ComputeOp:
    """A computation node in the tensor graph.

    The ``operator`` may be a :class:`PartitionableOperator` (simple ops)
    or a ``BasicOperation`` (from a decomposed ``FusedOperation``).

    ``op_id`` is decoupled from ``operator.op_id`` because
    ``BasicOperation`` (from TransformerEngine) doesn't carry one.
    The :class:`TensorGraphBuilder` assigns sequential ``op_id`` values.
    """

    operator: Any  # PartitionableOperator or BasicOperation (FusibleOperation)
    input_ports: List[TensorPort] = field(default_factory=list)
    output_ports: List[TensorPort] = field(default_factory=list)
    op_id: int = -1

    def get_input_tensor_ids(self) -> List[str]:
        return [p.tensor_id for p in self.input_ports]

    def get_output_tensor_ids(self) -> List[str]:
        return [p.tensor_id for p in self.output_ports]


@dataclass
class CommunicationOp:
    """A communication node in the tensor graph.

    The ``operator`` is assigned later by ``transformer_block`` once the
    actual communication primitive (AllReduce, AllGatherKV, etc.) is known.

    Delegation methods (``event_record``, ``event_wait``, ``fuser_forward``,
    ``sync``) forward to ``self.operator`` so that :class:`PartitionBase`
    can call them directly on the ``CommunicationOp`` instance.
    """

    comm_type: CommunicationType
    input_ports: List[TensorPort] = field(default_factory=list)
    output_ports: List[TensorPort] = field(default_factory=list)
    operator: Optional[Any] = None  # Assigned later by transformer_block

    def get_input_tensor_ids(self) -> List[str]:
        return [p.tensor_id for p in self.input_ports]

    def get_output_tensor_ids(self) -> List[str]:
        return [p.tensor_id for p in self.output_ports]

    # -- Delegation to operator (used by PartitionBase) --

    def event_record(self, stream):
        self.operator.event_record(stream)

    def event_wait(self):
        self.operator.event_wait()

    def fuser_forward(self, *args, **kwargs):
        return self.operator.fuser_forward(*args, **kwargs)

    def sync(self, stream):
        self.operator.sync(stream)


class PartitionableOperator(ABC):
    """Base interface for operators that participate in partitioning.

    Subclasses declare their forward compute/comm graph via
    :meth:`get_forward_ops` and :meth:`get_backward_ops`.  Channel
    information is declared for the **forward** direction only; backward
    channels are auto-derived by ``ComputeOpSpec(is_backward=True)``.

    ``op_id`` lives on :class:`ComputeOp`, not on the operator.

    Default implementations return a single ``ComputeOpSpec`` pointing
    at ``self``.  Override only when the operator introduces
    communication ops (e.g. AllReduce in linear layers, AllGatherKV
    in attention with context parallelism).
    """

    def get_forward_ops(self) -> List[Union[ComputeOpSpec, CommunicationOpSpec]]:
        """Return specs for forward-pass ops in execution order."""
        return [ComputeOpSpec(operator=self)]

    def get_backward_ops(self) -> List[Union[ComputeOpSpec, CommunicationOpSpec]]:
        """Return specs for backward-pass ops in execution order."""
        return [ComputeOpSpec(operator=self, is_backward=True)]

    def get_input_channels(self) -> List[Channel]:
        """Channels consumed by this operator's first forward op."""
        return [Channel(0, "main")]

    def get_output_channels(self) -> List[Channel]:
        """Channels produced by this operator's last forward op."""
        return [Channel(0, "main")]


@dataclass
class ComputeOpSpec:
    """Specification for creating a :class:`ComputeOp`.

    When ``is_backward=True``, input/output channels are automatically
    derived from the forward channels with a ``grad_`` prefix:
      - backward input channels  = ``grad_`` of forward output channels
      - backward output channels = ``grad_`` of forward input channels

    For ``BasicOperation`` operators (from decomposed ``FusedOperation``),
    use ``input_channels`` / ``output_channels`` to provide channel info
    explicitly since ``BasicOperation`` doesn't implement
    ``get_input_channels`` / ``get_output_channels``.
    """

    operator: Any  # PartitionableOperator or BasicOperation
    is_backward: bool = False
    op_id: Optional[int] = None  # Propagated to ComputeOp.op_id
    input_channels: Optional[List[Channel]] = None  # Override for BasicOperations
    output_channels: Optional[List[Channel]] = None  # Override for BasicOperations

    def get_input_channels(self) -> List[Channel]:
        if self.is_backward:
            fwd_out = self.output_channels or self.operator.get_output_channels()
            return [Channel(i, f"grad_{ch.name}") for i, ch in enumerate(fwd_out)]
        return self.input_channels or self.operator.get_input_channels()

    def get_output_channels(self) -> List[Channel]:
        if self.is_backward:
            fwd_in = self.input_channels or self.operator.get_input_channels()
            return [Channel(i, f"grad_{ch.name}") for i, ch in enumerate(fwd_in)]
        return self.output_channels or self.operator.get_output_channels()


@dataclass
class CommunicationOpSpec:
    """Specification for creating a :class:`CommunicationOp`.

    Channels are declared directly — no forward/backward reversal.
    Communication ops read and write the same set of channels.
    """

    comm_type: CommunicationType
    channels: List[Channel] = field(default_factory=list)

    def get_input_channels(self) -> List[Channel]:
        return self.channels

    def get_output_channels(self) -> List[Channel]:
        return self.channels  # Comm ops read and write the same channels


class TensorGraphBuilder:
    """Builds a tensor dependency graph using named channel routing.

    The builder maintains a **channel registry** that maps channel names
    to tensor IDs.  The registry **persists** across operators (it is NOT
    cleared after each op), enabling non-adjacent connections.  For
    example, ``"value"`` written by QKVPost reaches CoreAttn through
    Rotary because Rotary doesn't overwrite the ``"value"`` channel.

    Usage::

        builder = TensorGraphBuilder()
        builder.add_initial_channels({"main": "ext_main"})
        for spec in operator.get_forward_ops():
            builder.add_op(spec)
        graph = builder.build()
    """

    def __init__(self) -> None:
        self._channel_registry: Dict[str, str] = {}
        self._ops: List[Union[ComputeOp, CommunicationOp]] = []
        self._tensor_counter: int = 0
        self._op_id_counter: int = 0

    def _new_tensor_id(self) -> str:
        tid = f"t_{self._tensor_counter}"
        self._tensor_counter += 1
        return tid

    def _next_op_id(self) -> int:
        oid = self._op_id_counter
        self._op_id_counter += 1
        return oid

    def add_initial_channels(self, channels: Dict[str, str]) -> None:
        """Seed the channel registry with pre-existing tensor IDs.

        Typically used for the very first layer to inject external tensors::

            builder.add_initial_channels({
                "main": "ext_main",
                "residual": "ext_residual",
            })
        """
        self._channel_registry.update(channels)

    def add_op(
        self, spec: Union[ComputeOpSpec, CommunicationOpSpec]
    ) -> Union[ComputeOp, CommunicationOp]:
        """Add an op spec to the graph and return the concrete op.

        1. Wire **input ports** from the channel registry (missing
           channels produce ``ext_{name}`` tensor IDs).
        2. Create new tensor IDs for **output ports** and update the
           channel registry.
        """
        input_channels = spec.get_input_channels()
        output_channels = spec.get_output_channels()

        # Wire input ports
        input_ports: List[TensorPort] = []
        for ch in input_channels:
            tensor_id = self._channel_registry.get(ch.name)
            if tensor_id is None:
                tensor_id = f"ext_{ch.name}"
                self._channel_registry[ch.name] = tensor_id
            input_ports.append(TensorPort(port_idx=ch.port_idx, tensor_id=tensor_id))

        # Create output ports and update registry
        output_ports: List[TensorPort] = []
        for ch in output_channels:
            tensor_id = self._new_tensor_id()
            self._channel_registry[ch.name] = tensor_id
            output_ports.append(TensorPort(port_idx=ch.port_idx, tensor_id=tensor_id))

        # Build concrete op
        if isinstance(spec, ComputeOpSpec):
            op_id = spec.op_id if spec.op_id is not None else self._next_op_id()
            op = ComputeOp(
                operator=spec.operator,
                input_ports=input_ports,
                output_ports=output_ports,
                op_id=op_id,
            )
        else:
            op = CommunicationOp(
                comm_type=spec.comm_type,
                input_ports=input_ports,
                output_ports=output_ports,
            )

        self._ops.append(op)
        return op

    def build(self) -> TensorGraph:
        """Return the built :class:`TensorGraph`."""
        return TensorGraph(
            ops=list(self._ops),
            channel_registry=dict(self._channel_registry),
        )

    def get_channel_registry(self) -> Dict[str, str]:
        """Return a snapshot of the current channel→tensor_id mapping."""
        return dict(self._channel_registry)


@dataclass
class TensorGraph:
    """The built tensor dependency graph.

    Contains the ordered list of concrete ops (compute + communication)
    and the final channel registry mapping channel names to tensor IDs.
    """

    ops: List[Union[ComputeOp, CommunicationOp]] = field(default_factory=list)
    channel_registry: Dict[str, str] = field(default_factory=dict)

    def get_compute_ops(self) -> List[ComputeOp]:
        return [op for op in self.ops if isinstance(op, ComputeOp)]

    def get_comm_ops(self) -> List[CommunicationOp]:
        return [op for op in self.ops if isinstance(op, CommunicationOp)]

    def get_output_channel(self, channel_name: str) -> Optional[str]:
        """Return the tensor_id for *channel_name*, or ``None``."""
        return self.channel_registry.get(channel_name)
