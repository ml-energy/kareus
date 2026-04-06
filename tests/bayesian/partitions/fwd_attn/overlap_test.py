"""Forward attention partition overlap test (TP, ALL_REDUCE).

Operators: BDA → RMSNorm → Linear(QKV) → QKVPost → Rotary → Attention → Linear(proj)
Communication: ALL_REDUCE after proj output (main channel)
"""

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from kareus.megatron.core.extensions.ops import (
    BiasDropoutAddOp,
    PartitionableRMSNorm,
    QKVPostProcessOp,
    RotaryEmbeddingOp,
    TEDotProductAttentionOp,
)
from kareus.megatron.core.partitions.tensor_graph import CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
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
        print(f"Initialized distributed: rank={rank}, world_size={world_size}")
    ranks = list(range(world_size))
    tp_group = dist.new_group(ranks)
    print(f"Created tensor parallel group with ranks: {ranks}")
    return tp_group


class PartitionTest:
    """Forward attention partition test."""

    def __init__(self, args, rank=0, world_size=1):
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        self.tensor_parallel_size = world_size

        self.tp_group = init_distributed(rank, world_size)

        self.batch_size = args.batch_size
        self.seq_length = args.seq_len
        model = get_model_config(args.model_name)
        self.hidden_size = model.hidden_size
        self.num_attention_heads = model.num_attention_heads
        self.num_query_groups = model.num_query_groups
        self.head_dim = model.head_dim
        self.ffn_hidden_size = model.ffn_hidden_size

        self.config = model.create_transformer_config(
            context_parallel_size=1, tensor_parallel_size=world_size, dtype=self.dtype
        )

        self.hidden_states, self.residual, self.rotary_pos_emb, self.allreduce_inputs = (
            self.create_test_tensors()
        )
        self.comp_ops, self.comm_op = self.create_operations()
        self.executor = self.create_executor()
        self.executor.setup_contexts(
            compute_tensors={"main": self.hidden_states, "residual": self.residual,
                             "rotary_pos_emb": self.rotary_pos_emb},
            comm_tensors=[self.allreduce_inputs],
        )

    def create_test_tensors(self):
        nano_batch_size = self.batch_size // 2
        hidden_states = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        residual = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        seq = torch.arange(self.seq_length, device=self.device, dtype=torch.float32)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )
        freqs = torch.outer(seq, inv_freq)
        rotary_pos_emb = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]

        allreduce_inputs = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        return hidden_states, residual, rotary_pos_emb, allreduce_inputs

    def create_operations(self):
        tp = self.tensor_parallel_size
        nano_batch_size = self.batch_size // 2

        bda = BiasDropoutAddOp(has_bias=self.config.add_bias_linear, dropout_prob=self.config.hidden_dropout, training=True)
        norm = PartitionableRMSNorm(
            normalized_shape=self.hidden_size, eps=self.config.layernorm_epsilon,
            device=self.device, dtype=self.dtype,
        )
        qkv_size = (self.num_attention_heads * self.head_dim
                     + 2 * self.num_query_groups * self.head_dim) // tp
        linear_qkv = PartitionableLinear(
            in_features=self.hidden_size, out_features=qkv_size,
            device=self.device, dtype=self.dtype, bias=self.config.add_bias_linear, return_bias=False,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )
        qkv_post = QKVPostProcessOp(
            num_query_groups_per_partition=self.num_query_groups // tp,
            num_attention_heads_per_partition=self.num_attention_heads // tp,
            hidden_size_per_attention_head=self.head_dim,
            q_layernorm=None, k_layernorm=None, run_tests_fn=None, test_mode=False,
        )
        rotary = RotaryEmbeddingOp(config=self.config)
        attn = TEDotProductAttentionOp(
            config=self.config, layer_number=0,
            attn_mask_type=AttnMaskType.causal, attention_type="self",
            profiling_mode=True, cp_size=1, rank=self.rank,
        )
        proj_in = (self.head_dim * self.num_attention_heads) // tp
        linear_proj = PartitionableLinear(
            in_features=proj_in, out_features=self.hidden_size,
            device=self.device, dtype=self.dtype, bias=self.config.add_bias_linear, return_bias=True,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )

        allreduce = AllReduce(
            process_group=self.tp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            use_persistent_output=True, input_buffer=self.allreduce_inputs,
            tensor_size=[self.seq_length, nano_batch_size, self.hidden_size],
            device=self.device, dtype=self.dtype,
        )

        comp_ops = [bda, norm, linear_qkv, qkv_post, rotary, attn, linear_proj]
        return comp_ops, allreduce

    def create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="forward",
            partition_key="fwd_attn",
            comm_type=CommunicationType.ALL_REDUCE,
            initial_channel_names=["main", "residual", "rotary_pos_emb"],
        )

    def test_config(self, overlap_window, sm_configs):
        self.executor.execute(overlap_window, sm_configs)

    def run_overlap_test(self, frequency="default"):
        from zeus.monitor import ZeusMonitor

        monitor = None
        if self.rank == 0:
            monitor = ZeusMonitor(gpu_indices=list(range(self.world_size)))
            os.makedirs(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{frequency}", exist_ok=True)

        overlap_windows = [(-1, -1), (0, 8), (2, 8), (4, 8), (6, 8)]
        for ow in overlap_windows:
            for sm_num in range(3, 31, 3):
                for block_size in [512, 1024]:
                    sm_configs = (sm_num, block_size)
                    print(f"Overlap {ow} - SM: {sm_num}, Block: {block_size}")

                    # Warmup
                    torch.cuda.synchronize()
                    dist.barrier()
                    for _ in range(10):
                        self.test_config(ow, sm_configs)
                    torch.cuda.synchronize()
                    dist.barrier()

                    if self.rank == 0:
                        monitor.begin_window("step")
                    iterations = 100
                    for _ in range(iterations):
                        self.test_config(ow, sm_configs)
                    torch.cuda.synchronize()
                    dist.barrier()

                    if self.rank == 0:
                        result = monitor.end_window("step")
                        t = result.time / iterations
                        e = result.total_energy / iterations
                        print(f"  Time: {t*1000:.3f} ms, Energy: {e:.4f} J")


def overlap_test(rank, world_size, args, master_port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    test = PartitionTest(args, rank, world_size)
    try:
        test.run_overlap_test(getattr(args, 'frequency', 'default'))
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            os.system(f'pkill -P {os.getpid()}')
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    import random
    from torch.multiprocessing import spawn

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, required=True)
    parser.add_argument("--batch_size", "-b", type=int, required=True)
    parser.add_argument("--seq_len", "-s", type=int, required=True)
    parser.add_argument("--frequency", "-f", type=str, required=True)
    parser.add_argument("--model_name", "-m", type=str, required=True)
    args = parser.parse_args()

    print(f"fwd_attn overlap test: world_size={args.world_size}, bs={args.batch_size}, seq={args.seq_len}")
    spawn(overlap_test, args=(args.world_size, args, random.randint(8000, 65535)),
          nprocs=args.world_size, join=True)
