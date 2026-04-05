"""Backward MLP partition overlap test (TP, ALL_REDUCE).

Operators (backward order): Linear(FC2) → BiasSwiglu → Linear(FC1) → RMSNorm → BDA
Communication: ALL_REDUCE on grad_main channel
"""

import os
import sys
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
    return dist.new_group(list(range(world_size)))


class PartitionTest:
    """Backward MLP partition test."""

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
            context_parallel_size=1, tensor_parallel_size=world_size, dtype=self.dtype,
        )

        self.hidden_states, self.residual, self.allreduce_inputs = self._create_tensors()
        self.output_grad, self.allreduce_grad = self._create_grad_tensors()
        self.comp_ops, self.comm_op = self._create_operations()
        self.executor = self._create_executor()
        self.executor.run_forward_setup({"main": self.hidden_states, "residual": self.residual})
        self.executor.setup_contexts(
            compute_tensors={"grad_main": self.output_grad},
            comm_tensors=[self.allreduce_grad],
        )

    def _create_tensors(self):
        nb = self.batch_size // 2
        h = torch.randn(self.seq_length, nb, self.hidden_size, dtype=self.dtype, device=self.device, requires_grad=True)
        r = torch.randn(self.seq_length, nb, self.hidden_size, dtype=self.dtype, device=self.device, requires_grad=True)
        ar = torch.randn(self.seq_length, nb, self.hidden_size, dtype=self.dtype, device=self.device, requires_grad=True)
        return h, r, ar

    def _create_grad_tensors(self):
        nb = self.batch_size // 2
        og = torch.randn(self.seq_length, nb, self.hidden_size, dtype=self.dtype, device=self.device)
        ag = torch.randn(self.seq_length, nb, self.hidden_size, dtype=self.dtype, device=self.device)
        return og, ag

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

        return [bda, norm, linear_fc1, swiglu, linear_fc2], allreduce

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="backward",
            partition_key="bwd_mlp",
            comm_type=CommunicationType.ALL_REDUCE,
            initial_channel_names=["grad_main"],
            fwd_initial_channel_names=["main", "residual"],
        )

    def test_config(self, overlap_window, sm_configs):
        self.executor.execute(overlap_window, sm_configs)


if __name__ == "__main__":
    import argparse
    import random
    from torch.multiprocessing import spawn

    def _run(rank, world_size, args, master_port):
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size),
                          LOCAL_RANK=str(rank), MASTER_ADDR="localhost",
                          MASTER_PORT=str(master_port))
        test = PartitionTest(args, rank, world_size)
        try:
            for ow in [(-1, -1), (0, 6), (2, 6), (3, 6)]:
                for sm in range(3, 31, 3):
                    print(f"Overlap {ow} - SM: {sm}")
                    for _ in range(10):
                        test.test_config(ow, (sm, 1024))
        except Exception as e:
            traceback.print_exc()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--context_parallel_size", "-cp", type=int, default=1)
    args = parser.parse_args()
    spawn(_run, args=(args.world_size, args, random.randint(8000, 65535)), nprocs=args.world_size, join=True)
