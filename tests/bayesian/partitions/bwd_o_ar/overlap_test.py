"""Backward O-AR partition overlap test (CP+TP, ALL_REDUCE).

Backward of oproj (ALL_REDUCE in the backward graph).
Operators (backward order): RMSNorm → BDA → Linear(proj)
Communication: ALL_REDUCE on grad_main channel
world_size = tensor_parallel_size; context_parallel_size from args (profiling_mode).
"""

import os
import sys


import torch

from kareus.megatron.core.extensions.ops import BiasDropoutAddOp, PartitionableRMSNorm
from kareus.megatron.core.partitions.tensor_graph import CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from common import PartitionableLinear, PartitionExecutor  # noqa: E402
from common import get_model_config, init_distributed  # noqa: E402


class PartitionTest:
    """Backward O-AR partition test (ALL_REDUCE in attn+oproj backward)."""

    def __init__(self, args, rank=0, world_size=1):
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        assert world_size == args.tensor_parallel_size, (
            f"bwd_o_ar: world_size ({world_size}) must equal tensor_parallel_size ({args.tensor_parallel_size})"
        )
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
        sl = self.local_seq_length
        tp = self.tensor_parallel_size
        proj_in = (self.head_dim * self.num_attention_heads) // tp

        hidden_states = torch.randn(sl, nb, proj_in,
                                    dtype=self.dtype, device=self.device, requires_grad=True)
        residual = torch.randn(sl, nb, self.hidden_size,
                               dtype=self.dtype, device=self.device, requires_grad=True)
        ar = torch.randn(sl, nb, self.hidden_size,
                         dtype=self.dtype, device=self.device, requires_grad=True)
        return hidden_states, residual, ar

    def _create_grad_tensors(self):
        nb = self.batch_size // 2
        sl = self.local_seq_length
        output_grad = torch.randn(sl, nb, self.hidden_size, dtype=self.dtype, device=self.device)
        allreduce_grad = torch.randn(sl, nb, self.hidden_size, dtype=self.dtype, device=self.device)
        return output_grad, allreduce_grad

    def _create_operations(self):
        tp = self.tensor_parallel_size
        nb = self.batch_size // 2
        sl = self.local_seq_length

        proj_in = (self.head_dim * self.num_attention_heads) // tp
        linear_proj = PartitionableLinear(
            in_features=proj_in, out_features=self.hidden_size,
            device=self.device, dtype=self.dtype, bias=self.config.add_bias_linear, return_bias=True,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )
        # Norm is from the next layer
        bda = BiasDropoutAddOp(has_bias=self.config.add_bias_linear, dropout_prob=self.config.hidden_dropout, training=True)
        norm = PartitionableRMSNorm(
            normalized_shape=self.hidden_size, eps=self.config.layernorm_epsilon,
            device=self.device, dtype=self.dtype,
        )

        allreduce = AllReduce(
            process_group=self.tp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            use_persistent_output=True, input_buffer=self.allreduce_inputs,
            tensor_size=[sl, nb, self.hidden_size],
            device=self.device, dtype=self.dtype,
        )

        return [linear_proj, bda, norm], allreduce

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="backward",
            partition_key="bwd_o_ar",
            comm_type=CommunicationType.ALL_REDUCE,
            initial_channel_names=["grad_main"],
            fwd_initial_channel_names=["main", "residual"],
        )

    def test_config(self, overlap_window, sm_configs):
        self.executor.execute(overlap_window, sm_configs)
