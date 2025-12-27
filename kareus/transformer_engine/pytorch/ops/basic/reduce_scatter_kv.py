"""Fusible operation for reduce-scatter with optional MSCCl backend."""

from __future__ import annotations
from typing import Optional, List

import torch

from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext

import kareus.msccl.msccl_comm as msccl_comm

K_RS: list[torch.Tensor | None] = [None, None]
V_RS: list[torch.Tensor | None] = [None, None]
K_RS_BUFFER: list[torch.Tensor | None] = [None, None]
V_RS_BUFFER: list[torch.Tensor | None] = [None, None]

class ReduceScatterKV():
    """Reduce-scatter tensor along outer dimension.

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
        self.batch_idx: int = batch_idx
        self.nranks_per_node: int = nranks_per_node

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
        self.msccl_op = None

        if self.backend == "msccl":
            global K_RS_BUFFER, V_RS
            if K_RS_BUFFER[self.batch_idx] is None:
                K_RS_BUFFER[self.batch_idx] = torch.randn(
                    *tensor_size, dtype=dtype, device=device,
                ).round(decimals=4)
            if V_RS_BUFFER[self.batch_idx] is None:
                V_RS_BUFFER[self.batch_idx] = torch.randn(
                    *tensor_size, dtype=dtype, device=device,
                ).round(decimals=4)
            self.input_buffer_k = K_RS_BUFFER[self.batch_idx]
            self.input_buffer_v = V_RS_BUFFER[self.batch_idx]
            self.output_buffer_k = K_RS_BUFFER[self.batch_idx]
            self.output_buffer_v = V_RS_BUFFER[self.batch_idx]
            self.msccl_op = msccl_comm.msccl_ReduceScatter_init(
                self.rank,
                self.world_size,
                self.input_buffer_k,
                self.input_buffer_v,
                self.process_group,
                self.nranks_per_node,
            )
            self.comm_stream = self.msccl_op.stream

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

        global K_RS, V_RS

        k = input_
        if backward:
            k = K_RS[self.batch_idx]
            v = V_RS[self.batch_idx]

        if self.world_size == 1:
            return k, v

        if self.backend == "msccl":
            if sm_num is None or block_size is None:
                # Fallback to NCCL reduce_scatter
                k_out, v_out = self._nccl_reduce_scatter_kv(k, v)
                chunk_len = k.size(0) // self.world_size
                start = int(self.rank) * chunk_len
                end = start + chunk_len
                return self.output_buffer_k[start:end], self.output_buffer_v[start:end]
                # # self.backend = "nccl"
                # return k_out, v_out
            else:
                k_out, v_out = self._msccl_reduce_scatter_kv(k, v, sm_num, block_size)
                return k_out, v_out
        else:
            k_out, v_out = self._nccl_reduce_scatter_kv(k, v)
            return k_out, v_out
    
    def _msccl_reduce_scatter_kv(self, k, v, sm_num, block_size):
        chunk_len = k.size(0) // self.world_size
        start = int(self.rank) * chunk_len
        end = start + chunk_len
        self.input_buffer_k.copy_(k, non_blocking=False)

        current_stream = torch.cuda.current_stream()
        self.event_record(current_stream)
        self.event_wait()

        self.input_buffer_v.copy_(v, non_blocking=False)

        k_out, v_out = self.msccl_op(sm_num, block_size)
        return self.output_buffer_k[start:end], self.output_buffer_v[start:end]

    def _nccl_reduce_scatter(self, x):
        chunk_len = x.size(0) // self.world_size
        out_shape = list(x.size())
        out_shape[0] = chunk_len
        out = torch.empty(
            out_shape,
            dtype=x.dtype,
            device=x.device,
            memory_format=torch.contiguous_format,
        )
        handle = torch.distributed.reduce_scatter_tensor(
            out, x, group=self.process_group, async_op=self.async_op,
        )
        return out, handle

    def _nccl_reduce_scatter_kv(self, k, v):
        k_out, k_handle = self._nccl_reduce_scatter(k)
        v_out, v_handle = self._nccl_reduce_scatter(v)
        self._work_handles.append(k_handle)
        self._work_handles.append(v_handle)
        return k_out, v_out

    def sync(self, current_stream: torch.cuda.Stream = None) -> None:
        if self.backend == "msccl":
            if len(self._work_handles) > 0:
                for handle in self._work_handles:
                    handle.wait()
                self._work_handles = []
            else:
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
