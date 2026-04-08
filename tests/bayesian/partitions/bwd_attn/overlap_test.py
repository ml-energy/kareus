"""Backward attention partition overlap test (TP, ALL_REDUCE).

Operators (backward order): RMSNorm → BDA → Linear(proj) → Attention → Rotary → QKVPost → Linear(QKV)
Communication: ALL_REDUCE on grad_main channel
"""

import os
import sys

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
    ranks = list(range(world_size))
    tp_group = dist.new_group(ranks)
    return tp_group


class PartitionTest:
    """Backward attention partition test."""

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

        self.config = model.create_transformer_config(
            context_parallel_size=1,
            tensor_parallel_size=world_size,
            dtype=self.dtype,
        )

        self.hidden_states, self.residual, self.rotary_pos_emb, self.allreduce_inputs = (
            self._create_tensors()
        )
        self.output_grad, self.allreduce_grad = self._create_grad_tensors()
        self.comp_ops, self.comm_op = self._create_operations()
        self.executor = self._create_executor()
        self.executor.run_forward_setup({"main": self.hidden_states, "residual": self.residual,
                                         "rotary_pos_emb": self.rotary_pos_emb})
        self.executor.setup_contexts(
            compute_tensors={"grad_main": self.output_grad},
            comm_tensors=[self.allreduce_grad],
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
        seq = torch.arange(self.seq_length, device=self.device, dtype=torch.float32)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )
        freqs = torch.outer(seq, inv_freq)
        rotary_pos_emb = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]

        allreduce_inputs = torch.randn(
            self.seq_length, nb, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        return hidden_states, residual, rotary_pos_emb, allreduce_inputs

    def _create_grad_tensors(self):
        nb = self.batch_size // 2
        output_grad = torch.randn(
            self.seq_length, nb, self.hidden_size,
            dtype=self.dtype, device=self.device,
        )
        allreduce_grad = torch.randn(
            self.seq_length, nb, self.hidden_size,
            dtype=self.dtype, device=self.device,
        )
        return output_grad, allreduce_grad

    def _create_operations(self):
        tp = self.tensor_parallel_size
        nb = self.batch_size // 2

        qkv_size = (self.num_attention_heads * self.head_dim
                     + 2 * self.num_query_groups * self.head_dim) // tp
        linear_qkv = PartitionableLinear(
            in_features=self.hidden_size, out_features=qkv_size,
            device=self.device, dtype=self.dtype, bias=self.config.add_bias_linear, return_bias=False,
            tensor_parallel_mode=None, tensor_parallel_group=None, tensor_parallel_size=None,
        )
        q_ln = PartitionableRMSNorm(
            self.head_dim, eps=self.config.layernorm_epsilon,
            device=self.device, dtype=self.dtype,
        ) if self.config.qk_layernorm else None
        k_ln = PartitionableRMSNorm(
            self.head_dim, eps=self.config.layernorm_epsilon,
            device=self.device, dtype=self.dtype,
        ) if self.config.qk_layernorm else None
        qkv_post = QKVPostProcessOp(
            num_query_groups_per_partition=self.num_query_groups // tp,
            num_attention_heads_per_partition=self.num_attention_heads // tp,
            hidden_size_per_attention_head=self.head_dim,
            q_layernorm=q_ln, k_layernorm=k_ln, run_tests_fn=None, test_mode=False,
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
            tensor_size=[self.seq_length, nb, self.hidden_size],
            device=self.device, dtype=self.dtype,
        )

        comp_ops = [linear_qkv, qkv_post, rotary, attn, linear_proj, bda, norm]
        return comp_ops, allreduce

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="backward",
            partition_key="bwd_attn",
            comm_type=CommunicationType.ALL_REDUCE,
            initial_channel_names=["grad_main"],
            fwd_initial_channel_names=["main", "residual", "rotary_pos_emb"],
        )

    def test_config(self, overlap_window, sm_configs):
        self.executor.execute(overlap_window, sm_configs)
