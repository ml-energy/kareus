"""PartitionExecutor: bridge between operator lists and ForwardPartition/BackwardPartition.

Builds a TensorGraph from the partition's operators and comm op, then creates
the appropriate partition for BO profiling tests.

Usage (forward)::

    executor = PartitionExecutor(
        operators=[bda, norm, linear_qkv, qkv_post, rotary, attn, linear_proj],
        comm_operator=allreduce_op,
        direction="forward",
        partition_key="fwd_attn",
        comm_type=CommunicationType.ALL_REDUCE,
        initial_channel_names=["main", "residual", "rotary_pos_emb"],
    )

Usage (backward)::

    executor = PartitionExecutor(
        operators=[bda, norm, linear_qkv, qkv_post, rotary, attn, linear_proj],
        comm_operator=allreduce_op,
        direction="backward",
        partition_key="bwd_attn",
        comm_type=CommunicationType.ALL_REDUCE,
        initial_channel_names=["grad_main"],
        fwd_initial_channel_names=["main", "residual", "rotary_pos_emb"],
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from kareus.megatron.core.partitions.tensor_graph import (
    Channel,
    CommunicationOp,
    CommunicationType,
    ComputeOp,
    ComputeOpSpec,
    PartitionableOperator,
    TensorGraphBuilder,
    TensorPort,
)
from kareus.megatron.core.partitions.forward_partition import ForwardPartition
from kareus.megatron.core.partitions.backward_partition import BackwardPartition
from kareus.megatron.core.partitions.context_manager import NanoBatchContext
from kareus.transformer_engine.pytorch.ops.linear import Linear


class PartitionableLinear(Linear, PartitionableOperator):
    """Linear with PartitionableOperator mixin for partition graph building.

    By default declares main→main channels. Adds a ``"bias"`` output channel
    when both ``return_bias`` and ``has_bias`` are true.
    """

    def get_output_channels(self) -> List[Channel]:
        channels = [Channel(0, "main")]
        if getattr(self, '_return_bias', False) and self.has_bias:
            channels.append(Channel(1, "bias"))
        return channels


# Default communication channels per (direction, comm_type).
DEFAULT_COMM_CHANNELS: Dict[Tuple[str, CommunicationType], List[Channel]] = {
    ("forward", CommunicationType.ALL_REDUCE): [Channel(0, "main")],
    ("forward", CommunicationType.ALL_GATHER_KV): [Channel(0, "key"), Channel(1, "value")],
    ("backward", CommunicationType.ALL_REDUCE): [Channel(0, "grad_main")],
    ("backward", CommunicationType.ALL_GATHER_KV): [Channel(0, "grad_key"), Channel(1, "grad_value")],
    ("backward", CommunicationType.REDUCE_SCATTER_KV): [Channel(0, "grad_key"), Channel(1, "grad_value")],
}


class PartitionExecutor:
    """Build and execute a single partition from a list of operators.

    For backward direction, also builds a forward-only setup partition (no comm)
    so that ``run_forward_setup`` can populate OperationContexts before repeated
    backward profiling runs.

    Args:
        operators: Compute operators for this partition, in forward order.
        comm_operator: Physical communication primitive (AllReduce, AllGatherKV, etc.).
        direction: ``"forward"`` or ``"backward"``.
        partition_key: Matches ScheduleItem/ScheduleItemCP field name.
        comm_type: Communication type for the partition's comm op.
        initial_channel_names: Channel names to seed the graph builder
            (e.g. ``["main", "residual"]``). Tensor IDs are generated
            as ``ext_{name}`` internally.
        comm_channels: Explicit channels for the communication op. If ``None``,
            uses ``DEFAULT_COMM_CHANNELS[(direction, comm_type)]``.
        fwd_initial_channel_names: For backward direction only — channel names
            for the forward setup graph. Required when ``direction="backward"``.
    """

    def __init__(
        self,
        operators: List[PartitionableOperator],
        comm_operator,
        direction: str,
        partition_key: str,
        comm_type: CommunicationType,
        initial_channel_names: Optional[List[str]] = None,
        comm_channels: Optional[List[Channel]] = None,
        fwd_initial_channel_names: Optional[List[str]] = None,
    ):
        self.operators = operators
        self.direction = direction
        self.partition_key = partition_key

        if comm_channels is None:
            comm_channels = DEFAULT_COMM_CHANNELS.get((direction, comm_type))
            if comm_channels is None:
                raise ValueError(
                    f"No default comm channels for ({direction}, {comm_type}). "
                    f"Pass comm_channels explicitly."
                )

        initial_channel_names = initial_channel_names or (
            ["main"] if direction == "forward"
            else ["grad_main"]
        )

        # Convert channel names to {name: "ext_{name}"} for the graph builder
        initial_channel_map = {name: f"ext_{name}" for name in initial_channel_names}

        if direction == "forward":
            self._build_forward(
                operators, comm_operator, comm_type,
                initial_channel_map, comm_channels,
            )
        elif direction == "backward":
            fwd_channel_map = None
            if fwd_initial_channel_names is not None:
                fwd_channel_map = {name: f"ext_{name}" for name in fwd_initial_channel_names}
            self._build_backward(
                operators, comm_operator, comm_type,
                initial_channel_map, comm_channels,
                fwd_channel_map,
            )
        else:
            raise ValueError(f"Invalid direction: {direction}")

        self._initial_channels = initial_channel_map

    @staticmethod
    def _get_compute_spec(op, op_id: int, direction: str) -> ComputeOpSpec:
        """Create a ComputeOpSpec for an operator."""
        spec = ComputeOpSpec(operator=op, is_backward=(direction == "backward"))
        spec.op_id = op_id
        return spec

    def _build_partition_graph(
        self,
        operators: list,
        direction: str,
        initial_channels: Dict[str, str],
        comm_type: CommunicationType,
        comm_channels: List[Channel],
        comm_operator,
    ):
        """Build a TensorGraph with compute ops."""
        builder = TensorGraphBuilder()
        builder.add_initial_channels(initial_channels)

        if direction == "forward":
            iter_order = range(len(operators))
        else:
            iter_order = range(len(operators) - 1, -1, -1)

        for i in iter_order:
            builder.add_op(self._get_compute_spec(operators[i], i, direction))

        graph = builder.build()

        # Build comm op separately — its inputs come from pre_ctx (previous
        # partition), not from this partition's compute outputs.
        comm_op = CommunicationOp(
            comm_type=comm_type,
            input_ports=[TensorPort(port_idx=ch.port_idx, tensor_id=f"comm_{ch.name}") for ch in comm_channels],
            output_ports=[TensorPort(port_idx=ch.port_idx, tensor_id=f"comm_{ch.name}") for ch in comm_channels],
            operator=comm_operator,
        )
        return graph.get_compute_ops(), comm_op

    def _build_forward_only_graph(
        self,
        operators: list,
        initial_channels: Dict[str, str],
    ):
        """Build a compute-only forward graph (no comm ops) for backward setup."""
        builder = TensorGraphBuilder()
        builder.add_initial_channels(initial_channels)

        for i, op in enumerate(operators):
            builder.add_op(self._get_compute_spec(op, i, "forward"))

        return builder.build().get_compute_ops()

    def _build_forward(
        self,
        operators, comm_operator, comm_type,
        initial_channels, comm_channels,
    ):
        comp_ops, comm_op = self._build_partition_graph(
            operators, "forward", initial_channels,
            comm_type, comm_channels, comm_operator,
        )
        self.partition = ForwardPartition(
            partition_id=0,
            partition_key=self.partition_key,
            comp_ops=comp_ops,
            comm_op=comm_op,
        )

    def _build_backward(
        self,
        operators, comm_operator, comm_type,
        initial_channels, comm_channels,
        fwd_initial_channels,
    ):
        # 1. Build backward partition (profiling_mode=True)
        comp_ops, comm_op = self._build_partition_graph(
            operators, "backward", initial_channels,
            comm_type, comm_channels, comm_operator,
        )
        self.partition = BackwardPartition(
            partition_id=0,
            partition_key=self.partition_key,
            comp_ops=comp_ops,
            comm_op=comm_op,
            profiling_mode=True,
        )

        # 2. Build forward setup partition (compute only, no comm)
        if fwd_initial_channels is None:
            raise ValueError(
                "fwd_initial_channel_names is required for backward direction."
            )

        self._fwd_initial_channels = fwd_initial_channels
        fwd_comp_ops = self._build_forward_only_graph(operators, fwd_initial_channels)
        self._fwd_setup_partition = ForwardPartition(
            partition_id=0,
            partition_key=self.partition_key,
            comp_ops=fwd_comp_ops,
            comm_op=None,
        )

    def _set_tensors(
        self,
        ctx: NanoBatchContext,
        pre_ctx: NanoBatchContext,
        compute_tensors: Dict[str, object],
        comm_tensors: List[object],
    ) -> None:
        """Populate ctx/pre_ctx tensor stores from channel-name dicts.

        Args:
            compute_tensors: Maps channel names (matching keys in
                ``initial_channels``) to tensors.
            comm_tensors: Tensors for the comm op's input ports, in port order.
        """
        for name, tensor in compute_tensors.items():
            tensor_id = self._initial_channels.get(name)
            if tensor_id is None:
                raise ValueError(f"Unknown channel name: {name}")
            ctx.tensor_store.set(tensor_id, tensor)
        for port, tensor in zip(self.partition.comm_op.input_ports, comm_tensors):
            pre_ctx.tensor_store.set(port.tensor_id, tensor)

    def setup_contexts(
        self,
        compute_tensors: Dict[str, object],
        comm_tensors: List[object],
    ) -> None:
        """Create ctx/pre_ctx once and populate with tensors.

        Call this once in ``__init__`` of each test. For backward tests,
        call ``run_forward_setup`` separately before this (it needs
        different forward tensors that only the test knows).
        """
        self.ctx = NanoBatchContext(batch_idx=0)
        self.pre_ctx = NanoBatchContext(batch_idx=1)
        self._compute_tensors = compute_tensors
        self._comm_tensors = comm_tensors
        self._set_tensors(self.ctx, self.pre_ctx, compute_tensors, comm_tensors)

        # Transfer op_contexts from forward setup (backward only)
        if hasattr(self, '_fwd_ctx'):
            self.ctx.op_contexts = self._fwd_ctx.op_contexts

    def run_forward_setup(
        self,
        fwd_compute_tensors: Dict[str, object],
    ) -> None:
        """Run forward once to populate OperationContexts for backward.

        Must be called once before ``setup_contexts`` in backward tests.

        Args:
            fwd_compute_tensors: Maps ``fwd_initial_channels`` channel names
                to tensors. E.g. ``{"main": hidden_states}``.
        """
        ctx = NanoBatchContext(batch_idx=0)
        pre_ctx = NanoBatchContext(batch_idx=1)
        for name, tensor in fwd_compute_tensors.items():
            tensor_id = self._fwd_initial_channels[name]
            ctx.tensor_store.set(tensor_id, tensor)
        self._fwd_setup_partition.is_grad_enabled = True
        self._fwd_setup_partition.execute(ctx=ctx, pre_ctx=pre_ctx)
        ctx.finalize_recomputed_contexts()
        self._fwd_ctx = ctx

    def execute(self, overlap_window, sm_configs):
        """Execute the partition with the given overlap config."""
        self.partition._schedule_config = (overlap_window, sm_configs)
        return self.partition.execute(ctx=self.ctx, pre_ctx=self.pre_ctx)
