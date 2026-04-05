"""Forward MLP partition overlap test (TP, ALL_REDUCE).

Operators: BDA → RMSNorm → Linear(FC1) → BiasSwiglu → Linear(FC2)
Communication: ALL_REDUCE after FC2 output (main channel)
"""

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../fuser/'))

from common_config import FuserTestConfig
from kareus.megatron.core.extensions.ops import (
    BiasDropoutAddOp,
    BiasSwigluOp,
    PartitionableRMSNorm,
)
from kareus.megatron.core.partitions.tensor_graph import CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce

sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/'))
from partition_executor import PartitionableLinear, PartitionExecutor  # noqa: E402


def init_distributed(rank, world_size, backend='nccl'):
    if world_size <= 1:
        return None
    if not dist.is_initialized():
        torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    ranks = list(range(world_size))
    return dist.new_group(ranks)


class PartitionTest:
    """Forward MLP partition test."""

    def __init__(self, args, rank=0, world_size=1):
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        self.tensor_parallel_size = world_size

        self.tp_group = init_distributed(rank, world_size)

        self.batch_size = args.batch_size
        self.seq_length = args.seq_len // getattr(args, 'context_parallel_size', 1)
        self.hidden_size = FuserTestConfig.HIDDEN_SIZE
        self.ffn_hidden_size = FuserTestConfig.FFN_HIDDEN_SIZE

        self.config = FuserTestConfig.create_mlp_config(
            context_parallel_size=1,
            tensor_parallel_size=world_size,
            dtype=self.dtype,
        )

        self.hidden_states, self.residual, self.allreduce_inputs = self._create_tensors()
        self.comp_ops, self.comm_op = self._create_operations()
        self.executor = self._create_executor()
        self.executor.setup_contexts(
            compute_tensors={"main": self.hidden_states, "residual": self.residual},
            comm_tensors=[self.allreduce_inputs],
        )

    def _create_tensors(self):
        nb = self.batch_size // 2
        hidden_states = torch.randn(
            self.seq_length, nb, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        residual = torch.randn(
            self.seq_length, nb, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        allreduce_inputs = torch.randn(
            self.seq_length, nb, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        return hidden_states, residual, allreduce_inputs

    def _create_operations(self):
        tp = self.tensor_parallel_size
        nb = self.batch_size // 2

        bda = BiasDropoutAddOp(has_bias=False, dropout_prob=self.config.hidden_dropout, training=True)
        norm = PartitionableRMSNorm(
            normalized_shape=self.hidden_size, eps=self.config.layernorm_epsilon,
            device=self.device, dtype=self.dtype,
        )
        fc1_size = 2 * self.ffn_hidden_size // tp
        linear_fc1 = PartitionableLinear(
            in_features=self.hidden_size, out_features=fc1_size,
            device=self.device, dtype=self.dtype, bias=False, return_bias=True,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )
        swiglu = BiasSwigluOp(fp8_input_store=self.config.activation_func_fp8_input_store)
        fc2_size = self.ffn_hidden_size // tp
        linear_fc2 = PartitionableLinear(
            in_features=fc2_size, out_features=self.hidden_size,
            device=self.device, dtype=self.dtype, bias=False, return_bias=True,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )

        allreduce = AllReduce(
            process_group=self.tp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            use_persistent_output=True, input_buffer=self.allreduce_inputs,
            tensor_size=[self.seq_length, nb, self.hidden_size],
            device=self.device, dtype=self.dtype,
        )

        comp_ops = [bda, norm, linear_fc1, swiglu, linear_fc2]
        return comp_ops, allreduce

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="forward",
            partition_key="fwd_mlp",
            comm_type=CommunicationType.ALL_REDUCE,
            initial_channel_names=["main", "residual"],
        )

    def test_config(self, overlap_window, sm_configs):
        self.executor.execute(overlap_window, sm_configs)


def overlap_test(rank, world_size, args, master_port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    test = PartitionTest(args, rank, world_size)
    try:
        for ow in [(-1, -1), (0, 6), (2, 6), (4, 6)]:
            for sm in range(3, 31, 3):
                for bs in [512, 1024]:
                    print(f"Overlap {ow} - SM: {sm}, Block: {bs}")
                    for _ in range(10):
                        test.test_config(ow, (sm, bs))
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
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--context_parallel_size", "-cp", type=int, default=1)
    args = parser.parse_args()

    print(f"fwd_mlp overlap test: world_size={args.world_size}, bs={args.batch_size}, seq={args.seq_len}")
    spawn(overlap_test, args=(args.world_size, args, random.randint(8000, 65535)),
          nprocs=args.world_size, join=True)
