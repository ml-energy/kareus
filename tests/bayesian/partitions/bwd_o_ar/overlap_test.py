"""Backward O-AR partition overlap test (TP, ALL_REDUCE).

Backward of oproj (ALL_REDUCE in the backward graph).
Operators: Attention → Linear(proj) (backward direction)
Communication: ALL_REDUCE on grad_main channel
world_size = tensor_parallel_size; context_parallel_size from args (profiling_mode).
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from kareus.megatron.core.extensions.ops import TEDotProductAttentionOp
from kareus.megatron.core.partitions.tensor_graph import Channel, CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.transformer_engine.pytorch.ops.basic.all_gather_kv import K_TO_SAVE, V_TO_SAVE, K_AG, V_AG
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
    """Backward O-AR partition test (ALL_REDUCE in attn+oproj backward)."""

    def __init__(self, args, rank=0, world_size=1):
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        self.tensor_parallel_size = world_size
        self.context_parallel_size = args.context_parallel_size

        self.tp_group = init_distributed(rank, world_size)

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

    def _create_grad_tensors(self):
        nb = self.batch_size // 2
        sl = self.local_seq_length
        tp = self.tensor_parallel_size
        local_qg = self.num_query_groups // tp
        output_grad = torch.randn(sl, nb, self.hidden_size, dtype=self.dtype, device=self.device)
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

        proj_in = (self.head_dim * self.num_attention_heads) // tp
        linear_proj = PartitionableLinear(
            in_features=proj_in, out_features=self.hidden_size,
            device=self.device, dtype=self.dtype, bias=self.config.add_bias_linear, return_bias=True,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )

        nb = self.batch_size // 2
        sl = self.local_seq_length
        allreduce_inputs = torch.randn(sl, nb, self.hidden_size, dtype=self.dtype, device=self.device, requires_grad=True)
        self.allreduce_inputs = allreduce_inputs
        allreduce = AllReduce(
            process_group=self.tp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            use_persistent_output=True, input_buffer=allreduce_inputs,
            tensor_size=[sl, nb, self.hidden_size],
            device=self.device, dtype=self.dtype,
        )

        return [attn, linear_proj], allreduce

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="backward",
            partition_key="bwd_o_ar",
            comm_type=CommunicationType.ALL_REDUCE,
            initial_channel_names=["grad_main"],
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
        for ow in [(0, 1)]:
            for sm in range(3, 31, 3):
                for _ in range(5):
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
