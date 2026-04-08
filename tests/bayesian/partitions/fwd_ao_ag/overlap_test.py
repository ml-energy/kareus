"""Forward AO-AG partition overlap test (CP, ALL_GATHER_KV).

Operators: Attention → Linear(proj)
Communication: ALL_GATHER_KV on key/value channels (before attention)
"""

import os
import sys


import torch
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from kareus.megatron.core.extensions.ops import TEDotProductAttentionOp
from kareus.megatron.core.partitions.tensor_graph import Channel, CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.all_gather_kv import AllGatherKV, K_TO_SAVE, V_TO_SAVE, K_AG, V_AG
from megatron.core.transformer.enums import AttnMaskType

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from common import PartitionableLinear, PartitionExecutor  # noqa: E402
from common import get_model_config  # noqa: E402


def init_distributed(rank, world_size, backend='nccl'):
    if world_size <= 1:
        return None
    if not dist.is_initialized():
        torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return dist.new_group(list(range(world_size)))


class PartitionTest:
    """Forward AO-AG partition test (attention + oproj, CP allgather KV)."""

    def __init__(self, args, rank=0, world_size=1):
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
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
        self._prepopulate_kv_globals()
        self.comp_ops, self.comm_op = self._create_operations()
        self.executor = self._create_executor()
        self.executor.setup_contexts(
            compute_tensors={"main": self.query, "key": self.ag_key, "value": self.ag_value},
            comm_tensors=[self.ag_key, self.ag_value],
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
        """Pre-populate AllGatherKV globals needed by TEDotProductAttentionOp."""
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

    def _create_operations(self):
        tp = self.tensor_parallel_size

        attn = TEDotProductAttentionOp(
            config=self.config, layer_number=0,
            attn_mask_type=AttnMaskType.causal, attention_type="self",
            cp_comm_type="all_gather",
            profiling_mode=True, cp_size=self.context_parallel_size, rank=self.rank,
        )

        proj_in = (self.head_dim * self.num_attention_heads) // tp
        linear_proj = PartitionableLinear(
            in_features=proj_in, out_features=self.hidden_size,
            device=self.device, dtype=self.dtype, bias=self.config.add_bias_linear, return_bias=True,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )

        new_size = list(self.ag_value.size())
        new_size[0] = self.seq_length
        allgather = AllGatherKV(
            process_group=self.cp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            tensor_size=new_size, device=self.device, dtype=self.dtype,
            batch_idx=1,
        )

        return [attn, linear_proj], allgather

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="forward",
            partition_key="fwd_ao_ag",
            comm_type=CommunicationType.ALL_GATHER_KV,
            initial_channel_names=["main", "key", "value"],
            comm_channels=[Channel(0, "key"), Channel(1, "value")],
        )

    def test_config(self, overlap_window, sm_configs):
        self.executor.execute(overlap_window, sm_configs)
