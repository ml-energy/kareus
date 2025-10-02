"""Fusible operation for all-gather with optional MSCCl backend."""

from __future__ import annotations
from typing import Optional, List

import torch

from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext

import kareus.msccl.msccl_comm as new_msccl_comm

K_AG = None
V_AG = None

class AllGatherKV(BasicOperation):
    """All-gather tensor along outer dimension.

    Optionally uses MSCCl persistent buffers via kareus.msccl.msccl_comm.
    """
    num_extra_inputs: int = 1
    num_extra_outputs: int = 1

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        async_op: bool = True,
        backend: str = "nccl",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        use_persistent_output: bool = False,
        input_buffer_k: Optional[torch.Tensor] = None,
        input_buffer_v: Optional[torch.Tensor] = None,
        nranks_per_node: int = 0,
    ) -> None:
        super().__init__()
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self.async_op: bool = async_op
        self.backend: str = backend
        self.use_persistent_output: bool = use_persistent_output
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

        if self.backend == "msccl":
            if self.use_persistent_output:
                assert input_buffer_k is not None, "input_buffer_k must be provided when use_persistent_output is True"
                assert input_buffer_v is not None, "input_buffer_v must be provided when use_persistent_output is True"
                self.output_buffer_k = input_buffer_k
                self.output_buffer_v = input_buffer_v
                self.input_buffer_k = input_buffer_k
                self.input_buffer_v = input_buffer_v
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
            else:
                raise NotImplementedError("use_persistent_output is not supported for msccl backend")
    
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
        if torch.distributed.get_world_size(self.process_group) == 1:
            return k, v
        
        if isinstance(k, QuantizedTensor):
            k = k.dequantize()
        k = k.contiguous()
        if isinstance(v, QuantizedTensor):
            v = v.dequantize()
        v = v.contiguous()

        if self.backend == "msccl":
            if sm_num is None or block_size is None:
                # Fallback to NCCL all_gather
                k_out, v_out = self._nccl_all_gather_kv(k, v)
                self.backend = "nccl"
                if backward:
                    K_AG = k_out
                    V_AG = v_out
                return k_out, v_out
            if self.use_persistent_output:
                k_out, v_out = self._msccl_all_gather_kv(k, v, sm_num, block_size)
                if backward:
                    K_AG = k_out
                    V_AG = v_out
                return k_out, v_out
            else:
                raise NotImplementedError("use_persistent_output is not supported for msccl backend")
        else:
            k_out, v_out = self._nccl_all_gather_kv(k, v)
            return k_out, v_out
    
    def _msccl_all_gather_kv(self, k, v, sm_num, block_size):
        local_len = k.size(0)
        start = int(self.rank) * local_len
        end = start + local_len
        self.input_buffer_k[start:end].copy_(k)
        self.input_buffer_v[start:end].copy_(v)
        new_msccl_comm.msccl_AllGather(sm_num, block_size)
        new_msccl_comm.msccl_AllGather(sm_num, block_size)
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

    def sync(self) -> None:
        if self.backend == "msccl":
            new_msccl_comm.msccl_AllGather_sync()
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
        


