"""Backward A-AG partition overlap test (CP, ALL_GATHER_KV).

Backward of attention.
Operators: Attention → Linear(proj) (backward direction)
Communication: ALL_GATHER_KV on grad_key/grad_value channels
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
from kareus.megatron.core.partitions.tensor_graph import Channel, CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.all_gather_kv import AllGatherKV
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
    """Backward A-AG partition test (first ALL_GATHER_KV in attn+oproj backward)."""

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
        self.hidden_size = FuserTestConfig.HIDDEN_SIZE
        self.num_attention_heads = FuserTestConfig.NUM_ATTENTION_HEADS
        self.num_query_groups = FuserTestConfig.NUM_QUERY_GROUPS
        self.head_dim = FuserTestConfig.HEAD_DIM

        self.config = FuserTestConfig.create_attention_config(
            context_parallel_size=self.context_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
        )

        self.query, self.ag_key, self.ag_value = self._create_tensors()
        self.grad_key, self.grad_value = self._create_grad_tensors()
        self.comp_ops, self.comm_op = self._create_operations()
        self.executor = self._create_executor()
        self.executor.run_forward_setup({"main": self.query, "key": self.ag_key, "value": self.ag_value})
        self.executor.setup_contexts(
            compute_tensors={"grad_main": self.grad_key},
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

    def _create_grad_tensors(self):
        nb = self.batch_size // 2
        sl = self.local_seq_length
        tp = self.tensor_parallel_size
        local_qg = self.num_query_groups // tp
        gk = torch.randn(sl, nb, local_qg, self.head_dim, dtype=self.dtype, device=self.device)
        gv = torch.randn(sl, nb, local_qg, self.head_dim, dtype=self.dtype, device=self.device)
        return gk, gv

    def _create_operations(self):
        tp = self.tensor_parallel_size
        local_qg = self.num_query_groups // tp

        attn = TEDotProductAttentionOp(
            config=self.config, layer_number=0,
            attn_mask_type=AttnMaskType.causal, attention_type="self",
            cp_comm_type="all_gather",
        )
        attn.set_context_parallel_group(
            cp_size=self.context_parallel_size,
            rank=self.rank,
            cp_stream=torch.cuda.Stream(),
        )

        proj_in = (self.head_dim * self.num_attention_heads) // tp
        linear_proj = PartitionableLinear(
            in_features=proj_in, out_features=self.hidden_size,
            device=self.device, dtype=self.dtype, bias=False, return_bias=True,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )

        new_size = list(self.ag_value.size())
        new_size[0] = self.seq_length
        allgather = AllGatherKV(
            process_group=self.cp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            tensor_size=new_size, device=self.device, dtype=self.dtype,
        )

        return [attn, linear_proj], allgather

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="backward",
            partition_key="bwd_a_ag",
            comm_type=CommunicationType.ALL_GATHER_KV,
            initial_channel_names=["grad_main"],
            comm_channels=[Channel(0, "grad_key"), Channel(1, "grad_value")],
            fwd_initial_channel_names=["main", "key", "value"],
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
        for ow in [(0, 0)]:
            for sm in range(1, 21):
                for _ in range(5):
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
