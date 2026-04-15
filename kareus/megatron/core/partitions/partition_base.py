from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from transformer_engine.pytorch.ops.op import OperationContext

    from .context_manager import NanoBatchContext
    from .tensor_graph import CommunicationOp, ComputeOp

OverlapWindow = Tuple[int, int]   # (comm_start, comm_end) — op_idx to launch/finish comm
ResourceShape = Tuple[int, int]   # (sm_num, block_size) — SM allocation for comm

# ---------------------------------------------------------------------------
# Overlap slot builder: maps operator class names to semantic groups.
# Memory-bound ops with the same group merge into one slot (first occurrence).
# Compute-bound ops each get their own slot.
# ---------------------------------------------------------------------------
_GROUP_BY_CLASS: Dict[str, str] = {
    "ResidualForkOp": "norm",
    "BiasDropoutAddOp": "norm",
    "PartitionableRMSNorm": "norm",
    "QKVPostProcessOp": "rope",
    "RotaryEmbeddingOp": "rope",
    "BiasSwigluOp": "activation",
    "PartitionableLinear": "linear",
    "TEColumnParallelLinearOp": "linear",
    "TERowParallelLinearOp": "linear",
    "TEDotProductAttentionOp": "attn",
}
_MEMORY_BOUND_GROUPS = frozenset({"norm", "rope", "activation"})


def _build_overlap_slots(comp_ops: List[ComputeOp]) -> Dict[str, int]:
    """Build a mapping from semantic slot names to op_idx positions.

    Memory-bound groups merge consecutive ops into one slot (keyed by
    the first occurrence).  Compute-bound ops each get their own slot.

    Returns:
        ``{"norm_0": 0, "linear_0": 2, ...}`` — slot name → op_idx.
    """
    slots: Dict[str, int] = {}
    group_counter: Dict[str, int] = {}
    prev_group: Optional[str] = None

    for op_idx, op in enumerate(comp_ops):
        cls_name = type(op.operator).__name__
        group = _GROUP_BY_CLASS.get(cls_name)
        if group is None:
            raise ValueError(
                f"Unknown operator class {cls_name!r} — add it to _GROUP_BY_CLASS"
            )

        if group in _MEMORY_BOUND_GROUPS:
            if group == prev_group:
                continue  # merge into the existing slot
            prev_group = group
        else:
            prev_group = None

        seq = group_counter.get(group, 0)
        group_counter[group] = seq + 1
        slots[f"{group}_{seq}"] = op_idx

    return slots


def _resolve_slot(slots: Dict[str, int], name: str) -> int:
    """Resolve a slot name to its op_idx, or ``-1`` for the ``"none"`` sentinel."""
    if name == "none":
        return -1
    if name not in slots:
        raise KeyError(
            f"Unknown overlap slot {name!r}. Available: {sorted(slots)}"
        )
    return slots[name]


@dataclass
class PartitionBase:
    """Base class for forward and backward partitions.

    Encapsulates the shared state and communication overlap logic used by
    both ``ForwardPartition`` and ``BackwardPartition``.

    Each partition contains:
    - comp_ops: compute ops for THIS nanobatch
    - comm_op: optional communication op for the OTHER nanobatch

    Tensor routing follows the ctx / pre_ctx pattern:
    - Compute ops read/write tensors via ``ctx.tensor_store``
    - Comm op reads/writes tensors via ``pre_ctx.tensor_store``
    """

    partition_id: int
    partition_key: str = ""
    nano_batch_idx: int = 0  # 0 or 1
    comp_ops: List[ComputeOp] = field(default_factory=list)
    comm_op: Optional[CommunicationOp] = None
    _schedule_config: Optional[Tuple[OverlapWindow, ResourceShape]] = None
    profiling_mode: bool = False

    # Captured from the caller *before* entering torch.autograd.Function,
    # where torch.is_grad_enabled() returns False.  Mirrors the pattern
    # used by TransformerEngine's _OperationFuserAutogradFunction.
    is_grad_enabled: bool = True

    def __post_init__(self) -> None:
        # Precompute semantic overlap slots so _setup_comm doesn't have to
        # rebuild them on every partition execution.
        self._overlap_slots: Dict[str, int] = _build_overlap_slots(self.comp_ops)

    def load_schedule(self, schedule) -> None:
        """Load overlap_window and resource_shape from the scheduler's current_schedule.

        Called by ``transformer_block`` before executing this partition.
        Maps ``partition_key`` to its config from *schedule*.
        """
        config = getattr(schedule, self.partition_key, None)
        if config is not None:
            self._schedule_config = (
                config.overlap_window,
                config.resource_shape,
            )

    def _get_comm_config(self) -> Tuple[Tuple[int, int], ResourceShape]:
        """Return ``((comm_start, comm_end), (sm_num, block_size))``.

        Resolves string slot names in the overlap window to integer op_idx
        via the cached ``_overlap_slots`` map.
        """
        if not self._schedule_config:
            return ((0, -1), (None, None))
        (comm_start, comm_end), resource_shape = self._schedule_config

        if isinstance(comm_start, str):
            comm_start = _resolve_slot(self._overlap_slots, comm_start)
            comm_end = _resolve_slot(self._overlap_slots, comm_end)
        return ((comm_start, comm_end), resource_shape)

    def _setup_comm(self) -> Tuple[int, int, Optional[int], Optional[int]]:
        """Parse comm config and emit the initial event_record if needed.

        Returns ``(comm_start, comm_end, sm_num, block_size)``.
        """
        current_stream = torch.cuda.current_stream()

        if self.comm_op is not None:
            (comm_start, comm_end), (sm_num, block_size) = self._get_comm_config()
        else:
            comm_start, comm_end = -1, -1
            sm_num, block_size = None, None

        if comm_end != -1:
            raise ValueError(
                f"comm_end must be -1 (sentinel 'none'); got {comm_end}. "
                "Partition execution does not use comm_end — set overlap_end to 'none'."
            )

        if comm_start == 0:
            self.comm_op.event_record(current_stream)

        return comm_start, comm_end, sm_num, block_size

    def _read_comm_inputs(
        self,
        pre_ctx: NanoBatchContext,
    ) -> Tuple[torch.Tensor, list]:
        """Read comm op inputs from the OTHER nanobatch's tensor store."""
        comm_input = pre_ctx.tensor_store.get(
            self.comm_op.input_ports[0].tensor_id
        )
        comm_extra_inputs: list = []
        if len(self.comm_op.input_ports) > 1:
            comm_extra_inputs = [tuple(
                pre_ctx.tensor_store.get(p.tensor_id)
                for p in self.comm_op.input_ports[1:]
            )]
        return comm_input, comm_extra_inputs

    def _launch_comm(
        self,
        pre_ctx: NanoBatchContext,
        sm_num: Optional[int],
        block_size: Optional[int],
        *,
        backward: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """Launch an overlapped communication operation.

        Waits on the recorded event, reads inputs from *pre_ctx*, then
        fires the async comm kernel.  Returns ``(comm_output, comm_extra_outputs)``.
        """
        self.comm_op.event_wait()
        comm_input, comm_extra_inputs = self._read_comm_inputs(pre_ctx)
        kwargs: dict = {"sm_num": sm_num, "block_size": block_size}
        if backward:
            kwargs["backward"] = True
        # ctx_arg: list = [None] if backward else [self._make_comm_ctx()]
        comm_output, comm_extra_outputs = self.comm_op.fuser_forward(
            [None], # ctx
            comm_input,
            basic_op_extra_inputs=comm_extra_inputs,
            basic_op_prev_ops=[None],
            basic_op_next_ops=[None],
            basic_op_kwargs=[kwargs],
        )
        return comm_output, comm_extra_outputs

    def _launch_comm_no_overlap(
        self,
        pre_ctx: NanoBatchContext,
        sm_num: Optional[int],
        block_size: Optional[int],
        *,
        backward: bool = False,
    ) -> Tuple[torch.Tensor, list]:
        """Execute comm without overlapping (comm_start == -1)."""
        current_stream = torch.cuda.current_stream()
        self.comm_op.event_record(current_stream)
        self.comm_op.event_wait()
        comm_input, comm_extra_inputs = self._read_comm_inputs(pre_ctx)
        kwargs: dict = {"sm_num": sm_num, "block_size": block_size}
        if backward:
            kwargs["backward"] = True
        # ctx_arg: list = [None] if backward else [self._make_comm_ctx()]
        comm_output, comm_extra_outputs = self.comm_op.fuser_forward(
            [None], # ctx
            comm_input,
            basic_op_extra_inputs=comm_extra_inputs,
            basic_op_prev_ops=[None],
            basic_op_next_ops=[None],
            basic_op_kwargs=[kwargs],
        )
        return comm_output, comm_extra_outputs

    def _sync_comm(
        self,
        pre_ctx: NanoBatchContext,
        comm_output: Optional[torch.Tensor],
        comm_extra_outputs: list,
    ) -> None:
        """Sync comm and write outputs to the OTHER nanobatch's tensor store."""
        current_stream = torch.cuda.current_stream()
        self.comm_op.sync(current_stream)

        pre_ctx.tensor_store.set(
            self.comm_op.output_ports[0].tensor_id, comm_output
        )
        if comm_extra_outputs:
            for port_idx, port in enumerate(self.comm_op.output_ports[1:]):
                extra_out = comm_extra_outputs[0][port_idx]
                if extra_out is not None:
                    pre_ctx.tensor_store.set(port.tensor_id, extra_out)

    @staticmethod
    def _make_comm_ctx() -> OperationContext:
        from transformer_engine.pytorch.ops.op import OperationContext
        return OperationContext()

    def execute(
        self,
        ctx: NanoBatchContext,
        pre_ctx: NanoBatchContext,
    ):
        raise NotImplementedError
