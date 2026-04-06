"""Forward QKV-AG partition overlap test (CP, ALL_GATHER_KV).

Operators: BDA → RMSNorm → Linear(QKV) → QKVPost → Rotary
Communication: ALL_GATHER_KV on key/value channels (before attention)
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from kareus.megatron.core.extensions.ops import (
    BiasDropoutAddOp,
    PartitionableRMSNorm,
    QKVPostProcessOp,
    RotaryEmbeddingOp,
)
from kareus.megatron.core.partitions.tensor_graph import Channel, CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.all_gather_kv import AllGatherKV

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
    """Forward QKV-AG partition test (CP allgather KV)."""

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

        self.hidden_states, self.residual, self.rotary_pos_emb, self.ag_key, self.ag_value = (
            self._create_tensors()
        )
        self.comp_ops, self.comm_op = self._create_operations()
        self.executor = self._create_executor()
        self.executor.setup_contexts(
            compute_tensors={"main": self.hidden_states, "residual": self.residual,
                             "rotary_pos_emb": self.rotary_pos_emb},
            comm_tensors=[self.ag_key, self.ag_value],
        )

    def _create_tensors(self):
        nb = self.batch_size // 2
        sl = self.local_seq_length
        tp = self.tensor_parallel_size
        local_qg = self.num_query_groups // tp

        h = torch.randn(sl, nb, self.hidden_size, dtype=self.dtype, device=self.device, requires_grad=True)
        r = torch.randn(sl, nb, self.hidden_size, dtype=self.dtype, device=self.device, requires_grad=True)
        seq = torch.arange(sl, device=self.device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim))
        freqs = torch.outer(seq, inv_freq)
        rotary = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]

        ag_key = torch.randn(sl, nb, local_qg, self.head_dim, dtype=self.dtype, device=self.device, requires_grad=True)
        ag_value = torch.randn(sl, nb, local_qg, self.head_dim, dtype=self.dtype, device=self.device, requires_grad=True)
        return h, r, rotary, ag_key, ag_value

    def _create_operations(self):
        tp = self.tensor_parallel_size
        nb = self.batch_size // 2
        local_qg = self.num_query_groups // tp

        bda = BiasDropoutAddOp(has_bias=self.config.add_bias_linear, dropout_prob=self.config.hidden_dropout, training=True)
        norm = PartitionableRMSNorm(
            normalized_shape=self.hidden_size, eps=self.config.layernorm_epsilon,
            device=self.device, dtype=self.dtype,
        )
        qkv_size = (self.num_attention_heads * self.head_dim + 2 * self.num_query_groups * self.head_dim) // tp
        linear_qkv = PartitionableLinear(
            in_features=self.hidden_size, out_features=qkv_size,
            device=self.device, dtype=self.dtype, bias=self.config.add_bias_linear, return_bias=False,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )
        qkv_post = QKVPostProcessOp(
            num_query_groups_per_partition=local_qg,
            num_attention_heads_per_partition=self.num_attention_heads // tp,
            hidden_size_per_attention_head=self.head_dim,
            q_layernorm=None, k_layernorm=None, run_tests_fn=None, test_mode=False,
        )
        rotary = RotaryEmbeddingOp(config=self.config)

        new_size = list(self.ag_value.size())
        new_size[0] = self.seq_length
        allgather = AllGatherKV(
            process_group=self.cp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            tensor_size=new_size, device=self.device, dtype=self.dtype,
            batch_idx=1,
        )

        return [bda, norm, linear_qkv, qkv_post, rotary], allgather

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="forward",
            partition_key="fwd_qkv_ag",
            comm_type=CommunicationType.ALL_GATHER_KV,
            initial_channel_names=["main", "residual", "rotary_pos_emb"],
            comm_channels=[Channel(0, "key"), Channel(1, "value")],
        )

    def test_config(self, overlap_window, sm_configs):
        self.executor.execute(overlap_window, sm_configs)


if __name__ == "__main__":
    import argparse
    import random
    from torch.multiprocessing import spawn

    def _run(rank, ws, args, port):
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(ws), LOCAL_RANK=str(rank),
                          MASTER_ADDR="localhost", MASTER_PORT=str(port))
        test = PartitionTest(args, rank, ws)
        for ow in [(0, 5), (2, 5), (4, 5)]:
            for sm in range(1, 21):
                for _ in range(10):
                    test.test_config(ow, (sm, 1024))
        if dist.is_initialized():
            dist.destroy_process_group()

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, required=True)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, required=True)
    parser.add_argument("--context_parallel_size", "-cp", type=int, required=True)
    parser.add_argument("--batch_size", "-b", type=int, required=True)
    parser.add_argument("--seq_len", "-s", type=int, required=True)
    parser.add_argument("--model_name", "-m", type=str, required=True)
    args = parser.parse_args()
    spawn(_run, args=(args.world_size, args, random.randint(8000, 65535)), nprocs=args.world_size, join=True)
