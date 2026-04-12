"""Backward A-RS partition overlap test (CP+TP, REDUCE_SCATTER_KV).

Backward of attention (REDUCE_SCATTER_KV in the backward graph).
Operators (backward order): Attention
Communication: REDUCE_SCATTER_KV on grad_key/grad_value channels
"""

import os
import sys


import torch
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from kareus.megatron.core.extensions.ops import TEDotProductAttentionOp
from kareus.megatron.core.partitions.tensor_graph import Channel, CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.reduce_scatter_kv import ReduceScatterKV, K_RS, V_RS
from kareus.transformer_engine.pytorch.ops.basic.all_gather_kv import K_TO_SAVE, V_TO_SAVE, K_AG, V_AG
from megatron.core.transformer.enums import AttnMaskType

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from common import PartitionExecutor  # noqa: E402
from common import get_model_config  # noqa: E402


def init_distributed(rank, world_size, backend='nccl'):
    if world_size <= 1:
        return None
    if not dist.is_initialized():
        torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return dist.new_group(list(range(world_size)))


class PartitionTest:
    """Backward A-RS partition test (REDUCE_SCATTER_KV in attn+oproj backward)."""

    def __init__(self, args, rank=0, world_size=1):
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        assert world_size == args.context_parallel_size, (
            f"bwd_a_rs: world_size ({world_size}) must equal context_parallel_size ({args.context_parallel_size})"
        )
        self.context_parallel_size = world_size
        self.tensor_parallel_size = args.tensor_parallel_size

        self.cp_group = init_distributed(rank, world_size)

        self.batch_size = args.batch_size
        self.seq_length = args.seq_len
        self.local_seq_length = self.seq_length // self.context_parallel_size
        model = get_model_config(args.model_name)
        self.hidden_size = model.hidden_size
        self.num_attention_heads = model.num_attention_heads
        self.num_query_groups = model.num_query_groups
        self.head_dim = model.head_dim

        self.config = model.create_transformer_config(
            context_parallel_size=self.context_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
        )

        self.query, self.ag_key, self.ag_value = self._create_tensors()
        self.output_grad, self.grad_key, self.grad_value = self._create_grad_tensors()
        self._prepopulate_kv_globals()
        self.comp_ops, self.comm_op = self._create_operations()
        self.executor = self._create_executor()
        self.executor.run_forward_setup({"main": self.query, "key": self.ag_key, "value": self.ag_value})
        self.executor.setup_contexts(
            compute_tensors={"grad_main": self.output_grad},
            comm_tensors=[self.grad_key, self.grad_value],
        )

    def _create_tensors(self):
        nb = self.batch_size // 2
        sl = self.local_seq_length
        tp = self.tensor_parallel_size
        local_heads = self.num_attention_heads // tp
        local_qg = self.num_query_groups // tp

        query = torch.randn(sl, nb, local_heads, self.head_dim,
                             dtype=self.dtype, device=self.device, requires_grad=True)
        ag_key = torch.randn(sl, nb, local_qg, self.head_dim,
                              dtype=self.dtype, device=self.device, requires_grad=True)
        ag_value = torch.randn(sl, nb, local_qg, self.head_dim,
                                dtype=self.dtype, device=self.device, requires_grad=True)
        return query, ag_key, ag_value

    def _prepopulate_kv_globals(self):
        """Pre-populate AllGatherKV and ReduceScatterKV globals."""
        nb = self.batch_size // 2
        sl = self.local_seq_length
        tp = self.tensor_parallel_size
        local_qg = self.num_query_groups // tp
        K_TO_SAVE[0] = torch.randn(sl, nb, local_qg, self.head_dim,
                                    dtype=self.dtype, device=self.device)
        V_TO_SAVE[0] = torch.randn(sl, nb, local_qg, self.head_dim,
                                    dtype=self.dtype, device=self.device)
        K_AG[0] = torch.randn(self.seq_length, nb, local_qg, self.head_dim,
                               dtype=self.dtype, device=self.device)
        V_AG[0] = torch.randn(self.seq_length, nb, local_qg, self.head_dim,
                               dtype=self.dtype, device=self.device)
        # ReduceScatterKV backward reads from K_RS/V_RS (set by attn backward).
        # Pre-populate so comm can run before attn backward during profiling.
        K_RS[1] = torch.randn(self.seq_length, nb, local_qg, self.head_dim,
                               dtype=self.dtype, device=self.device)
        V_RS[1] = torch.randn(self.seq_length, nb, local_qg, self.head_dim,
                               dtype=self.dtype, device=self.device)

    def _create_grad_tensors(self):
        nb = self.batch_size // 2
        sl = self.local_seq_length
        tp = self.tensor_parallel_size
        proj_in = (self.head_dim * self.num_attention_heads) // tp
        local_qg = self.num_query_groups // tp
        output_grad = torch.randn(sl, nb, proj_in, dtype=self.dtype, device=self.device)
        gk = torch.randn(sl, nb, local_qg, self.head_dim, dtype=self.dtype, device=self.device)
        gv = torch.randn(sl, nb, local_qg, self.head_dim, dtype=self.dtype, device=self.device)
        return output_grad, gk, gv

    def _create_operations(self):
        tp = self.tensor_parallel_size
        local_qg = self.num_query_groups // tp

        attn = TEDotProductAttentionOp(
            config=self.config, layer_number=0,
            attn_mask_type=AttnMaskType.causal, attention_type="self",
            cp_comm_type="all_gather",
            profiling_mode=True, cp_size=self.context_parallel_size, rank=self.rank,
        )

        nb = self.batch_size // 2
        kv_size = [self.seq_length, nb, local_qg, self.head_dim]
        rs_comm = ReduceScatterKV(
            process_group=self.cp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            tensor_size=kv_size, device=self.device, dtype=self.dtype,
            batch_idx=1,
        )

        return [attn], rs_comm

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="backward",
            partition_key="bwd_a_rs",
            comm_type=CommunicationType.REDUCE_SCATTER_KV,
            initial_channel_names=["grad_main"],
            comm_channels=[Channel(0, "grad_key"), Channel(1, "grad_value")],
            fwd_initial_channel_names=["main", "key", "value"],
        )

    def test_config(self, overlap_window, sm_configs):
        self.executor.execute(overlap_window, sm_configs)


