"""Forward AO-AR partition overlap test (CP, ALL_REDUCE).

Operators: Attention → Linear(proj)
Communication: ALL_REDUCE on main channel (TP allreduce after proj)
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../fuser/'))

from common_config import FuserTestConfig
from kareus.megatron.core.extensions.ops import TEDotProductAttentionOp
from kareus.megatron.core.partitions.tensor_graph import CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from megatron.core.transformer.enums import AttnMaskType

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
    """Forward AO-AR partition test (attention + oproj, TP allreduce)."""

    def __init__(self, args, rank=0, world_size=1):
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        self.context_parallel_size = args.context_parallel_size
        self.tensor_parallel_size = args.tensor_parallel_size

        self.cp_group = init_distributed(rank, world_size)

        self.batch_size = args.batch_size
        self.seq_length = args.seq_len
        self.local_seq_length = self.seq_length // self.context_parallel_size
        self.hidden_size = FuserTestConfig.HIDDEN_SIZE
        self.num_attention_heads = FuserTestConfig.NUM_ATTENTION_HEADS
        self.num_query_groups = FuserTestConfig.NUM_QUERY_GROUPS
        self.head_dim = FuserTestConfig.HEAD_DIM

        self.config = FuserTestConfig.create_attention_config(
            context_parallel_size=self.context_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
        )

        self.query, self.key, self.value, self.allreduce_inputs = self._create_tensors()
        self.comp_ops, self.comm_op = self._create_operations()
        self.executor = self._create_executor()
        self.executor.setup_contexts(
            compute_tensors={"main": self.query, "key": self.key, "value": self.value},
            comm_tensors=[self.allreduce_inputs],
        )

    def _create_tensors(self):
        nb = self.batch_size // 2
        sl = self.local_seq_length
        tp = self.tensor_parallel_size
        local_heads = self.num_attention_heads // tp
        local_qg = self.num_query_groups // tp

        query = torch.randn(sl, nb, local_heads, self.head_dim,
                             dtype=self.dtype, device=self.device, requires_grad=True)
        key = torch.randn(sl, nb, local_qg, self.head_dim,
                           dtype=self.dtype, device=self.device, requires_grad=True)
        value = torch.randn(sl, nb, local_qg, self.head_dim,
                             dtype=self.dtype, device=self.device, requires_grad=True)
        allreduce_inputs = torch.randn(sl, nb, self.hidden_size,
                                        dtype=self.dtype, device=self.device, requires_grad=True)
        return query, key, value, allreduce_inputs

    def _create_operations(self):
        tp = self.tensor_parallel_size
        nb = self.batch_size // 2
        sl = self.local_seq_length

        attn = TEDotProductAttentionOp(
            config=self.config, layer_number=0,
            attn_mask_type=AttnMaskType.causal, attention_type="self",
        )

        proj_in = (self.head_dim * self.num_attention_heads) // tp
        linear_proj = PartitionableLinear(
            in_features=proj_in, out_features=self.hidden_size,
            device=self.device, dtype=self.dtype, bias=False, return_bias=True,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )

        allreduce = AllReduce(
            process_group=self.cp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            use_persistent_output=True, input_buffer=self.allreduce_inputs,
            tensor_size=[sl, nb, self.hidden_size],
            device=self.device, dtype=self.dtype,
        )

        return [attn, linear_proj], allreduce

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="forward",
            partition_key="fwd_ao_ar",
            comm_type=CommunicationType.ALL_REDUCE,
            initial_channel_names=["main", "key", "value"],
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
        for ow in [(0, 2)]:
            for sm in range(3, 31, 3):
                for _ in range(10):
                    test.test_config(ow, (sm, 1024))
        if dist.is_initialized():
            dist.destroy_process_group()

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--context_parallel_size", "-cp", type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    args = parser.parse_args()
    spawn(_run, args=(args.world_size, args, random.randint(8000, 65535)), nprocs=args.world_size, join=True)
