"""Fusible operation for all-gather with optional MSCCl backend."""

from __future__ import annotations
from typing import Optional, List

import torch

from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext

import kareus.msccl.msccl_comm as new_msccl_comm

K_AG: list[torch.Tensor | None] = [None, None]
V_AG: list[torch.Tensor | None] = [None, None]
K_TO_SAVE: list[torch.Tensor | None] = [None, None]
V_TO_SAVE: list[torch.Tensor | None] = [None, None]

class AllGatherKV():
    """All-gather tensor along outer dimension.

    Optionally uses MSCCl persistent buffers via kareus.msccl.msccl_comm.
    """

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        async_op: bool = True,
        backend: str = "nccl",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        nranks_per_node: int = 0,
        batch_idx: int = 0,
        tensor_size: Optional[list[int]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self.async_op: bool = async_op
        self.backend: str = backend
        self.nranks_per_node: int = nranks_per_node
        self.batch_idx: int = batch_idx

        # Cache rank/world_size for offset computations
        if rank:
            self.rank: Optional[int] = rank
        elif process_group:
            self.rank: Optional[int] = torch.distributed.get_rank(process_group)
        else:
            raise ValueError("rank or process_group must be provided")
        if world_size:
            self.world_size: Optional[int] = world_size
        elif process_group:
            self.world_size: Optional[int] = torch.distributed.get_world_size(process_group)
        else:
            raise ValueError("world_size or process_group must be provided")
        
        self.comm_stream: Optional[torch.cuda.Stream] = None
        self._work_handles: List[torch.distributed.Work] = []
        self.wait_event = torch.cuda.Event()

        global K_AG, V_AG
        if self.backend == "msccl":
            if K_AG[self.batch_idx] is None:
                K_AG[self.batch_idx] = torch.empty(
                    *tensor_size, dtype=dtype, device=device,
                )
            if V_AG[self.batch_idx] is None:
                V_AG[self.batch_idx] = torch.empty(
                    *tensor_size, dtype=dtype, device=device,
                )
            self.input_buffer_k = K_AG[self.batch_idx]
            self.input_buffer_v = V_AG[self.batch_idx]
            self.output_buffer_k = K_AG[self.batch_idx]
            self.output_buffer_v = V_AG[self.batch_idx]
            new_msccl_comm.msccl_AllGather_init(
                self.rank,
                self.world_size,
                self.input_buffer_k,
                self.process_group,
                self.nranks_per_node,
            )
            new_msccl_comm.msccl_AllGather_init(
                self.rank,
                self.world_size,
                self.input_buffer_v,
                self.process_group,
                self.nranks_per_node,
            )
            self.comm_stream = new_msccl_comm.AG_COMM_STREAM
    
    def set_stream(self, stream: torch.cuda.Stream):
        self.comm_stream = stream
    
    def event_record(self, stream: torch.cuda.Stream):
        if self.comm_stream is not None:
            self.wait_event.record(stream)
    
    def event_wait(self):
        if self.comm_stream is not None:
            self.comm_stream.wait_event(self.wait_event)

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        v: torch.Tensor,
        *,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
        sm_num: Optional[int] = None,
        block_size: Optional[int] = None,
        backward: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        global K_AG, V_AG

        k = input_
        if self.world_size == 1:
            return k, v

        if self.backend == "msccl":
            if sm_num is None or block_size is None:
                # Fallback to NCCL all_gather
                k_out, v_out = self._nccl_all_gather_kv(k, v)
                self.backend = "nccl"
                if not backward:
                    K_AG[self.batch_idx] = k_out
                    V_AG[self.batch_idx] = v_out
                    K_TO_SAVE[self.batch_idx] = k
                    V_TO_SAVE[self.batch_idx] = v
                return k_out, v_out
            else:
                k_out, v_out = self._msccl_all_gather_kv(k, v, sm_num, block_size)
                if not backward:
                    K_TO_SAVE[self.batch_idx] = k
                    V_TO_SAVE[self.batch_idx] = v
                return k_out, v_out
        else:
            k_out, v_out = self._nccl_all_gather_kv(k, v)
            return k_out, v_out
    
    def _msccl_all_gather_kv(self, k, v, sm_num, block_size):
        local_len = k.size(0)
        start = int(self.rank) * local_len
        end = start + local_len
        self.input_buffer_k[start:end].copy_(k, non_blocking=False)

        current_stream = torch.cuda.current_stream()
        self.event_record(current_stream)
        self.event_wait()

        self.input_buffer_v[start:end].copy_(v, non_blocking=False)

        new_msccl_comm.msccl_AllGather(sm_num, block_size)
        # new_msccl_comm.msccl_AllGather(sm_num, block_size)
        return self.output_buffer_k, self.output_buffer_v

    def _nccl_all_gather(self, x):
        out_shape = list(x.size())
        out_shape[0] *= self.world_size
        out = torch.empty(
            out_shape,
            dtype=x.dtype,
            device=x.device,
            memory_format=torch.contiguous_format,
        )
        x = x.contiguous()
        handle = torch.distributed.all_gather_into_tensor(
            out, x, group=self.process_group, async_op=self.async_op,
        )
        return out, handle

    def _nccl_all_gather_kv(self, k, v):
        k_out, k_handle = self._nccl_all_gather(k)
        v_out, v_handle = self._nccl_all_gather(v)
        self._work_handles.append(k_handle)
        self._work_handles.append(v_handle)
        return k_out, v_out

    def sync(self, current_stream: torch.cuda.Stream = None) -> None:
        if self.backend == "msccl":
            # new_msccl_comm.msccl_AllGather_sync()
            self.wait_event.record(self.comm_stream)
            current_stream.wait_event(self.wait_event)
        else:
            for handle in self._work_handles:
                handle.wait()
            self._work_handles = []
    
    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, any]],
    ) -> tuple[torch.Tensor, list[tuple[()]]]:
        """Override fuser_forward since we have extra inputs."""
        
        k = input_
        v, = basic_op_extra_inputs[0]
        kwargs = basic_op_kwargs[0].copy()
        kwargs['v'] = v

        k, v = self.op_forward(
            basic_op_ctxs[0],
            k,
            prev_op=basic_op_prev_ops[0],
            next_op=basic_op_next_ops[0],
            **kwargs,
        )

        return k, [(v,)]
