"""Backward QKV-RS partition overlap test (CP+TP, REDUCE_SCATTER_KV).

Operators (backward order): Rotary → QKVPostProcess → Linear(QKV)
Communication: REDUCE_SCATTER_KV on grad_key/grad_value channels.
"""

import os
import sys


import torch

from kareus.megatron.core.extensions.ops import (
    PartitionableRMSNorm,
    QKVPostProcessOp,
    RotaryEmbeddingOp,
)
from kareus.megatron.core.partitions.tensor_graph import Channel, CommunicationType
from kareus.transformer_engine.pytorch.ops.basic.reduce_scatter_kv import ReduceScatterKV, K_RS, V_RS

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from common import PartitionableLinear, PartitionExecutor  # noqa: E402
from common import get_model_config, init_distributed  # noqa: E402


class PartitionTest:
    """Backward QKV-RS partition test (CP reduce-scatter KV)."""

    def __init__(self, args, rank=0, world_size=1):
        self.device = torch.device('cuda')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        assert world_size == args.context_parallel_size, (
            f"bwd_qkv_rs: world_size ({world_size}) must equal context_parallel_size ({args.context_parallel_size})"
        )
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

        self.hidden_states, self.rotary_pos_emb = self._create_tensors()
        self.output_grad, self.grad_key, self.grad_value = self._create_grad_tensors()
        self._prepopulate_rs_globals()
        self.comp_ops, self.comm_op = self._create_operations()
        self.executor = self._create_executor()
        self.executor.run_forward_setup({"main": self.hidden_states,
                                         "rotary_pos_emb": self.rotary_pos_emb})
        self.executor.setup_contexts(
            compute_tensors={"grad_main": self.output_grad,
                             "grad_key": self.grad_key,
                             "grad_value": self.grad_value},
            comm_tensors=[self.grad_key, self.grad_value],
        )

    def _create_tensors(self):
        nb = self.batch_size // 2
        sl = self.local_seq_length
        h = torch.randn(sl, nb, self.hidden_size, dtype=self.dtype, device=self.device, requires_grad=True)
        seq = torch.arange(sl, device=self.device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim))
        freqs = torch.outer(seq, inv_freq)
        rotary = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
        return h, rotary

    def _create_grad_tensors(self):
        nb = self.batch_size // 2
        sl = self.local_seq_length
        tp = self.tensor_parallel_size
        local_heads = self.num_attention_heads // tp
        local_qg = self.num_query_groups // tp
        og = torch.randn(sl, nb, local_heads, self.head_dim, dtype=self.dtype, device=self.device)
        gk = torch.randn(sl, nb, local_qg, self.head_dim, dtype=self.dtype, device=self.device)
        gv = torch.randn(sl, nb, local_qg, self.head_dim, dtype=self.dtype, device=self.device)
        return og, gk, gv

    def _prepopulate_rs_globals(self):
        """Pre-populate ReduceScatterKV globals (normally set by attn backward)."""
        nb = self.batch_size // 2
        tp = self.tensor_parallel_size
        local_qg = self.num_query_groups // tp
        K_RS[1] = torch.randn(self.seq_length, nb, local_qg, self.head_dim,
                               dtype=self.dtype, device=self.device)
        V_RS[1] = torch.randn(self.seq_length, nb, local_qg, self.head_dim,
                               dtype=self.dtype, device=self.device)

    def _create_operations(self):
        tp = self.tensor_parallel_size
        nb = self.batch_size // 2
        local_qg = self.num_query_groups // tp

        qkv_size = (self.num_attention_heads * self.head_dim + 2 * self.num_query_groups * self.head_dim) // tp
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
            num_query_groups_per_partition=local_qg,
            num_attention_heads_per_partition=self.num_attention_heads // tp,
            hidden_size_per_attention_head=self.head_dim,
            q_layernorm=q_ln, k_layernorm=k_ln, run_tests_fn=None, test_mode=False,
        )
        rotary = RotaryEmbeddingOp(config=self.config)

        kv_size = [self.seq_length, nb, local_qg, self.head_dim]
        rs_comm = ReduceScatterKV(
            process_group=self.cp_group, async_op=True, backend="msccl",
            rank=self.rank, world_size=self.world_size,
            tensor_size=kv_size, device=self.device, dtype=self.dtype,
            batch_idx=1,
        )

        return [linear_qkv, qkv_post, rotary], rs_comm

    def _create_executor(self):
        return PartitionExecutor(
            operators=self.comp_ops,
            comm_operator=self.comm_op,
            direction="backward",
            partition_key="bwd_qkv_rs",
            comm_type=CommunicationType.REDUCE_SCATTER_KV,
            initial_channel_names=["grad_main", "grad_key", "grad_value"],
            comm_channels=[Channel(0, "grad_key"), Channel(1, "grad_value")],
            fwd_initial_channel_names=["main", "rotary_pos_emb"],
        )

    def test_config(self, overlap_window, sm_configs):
        self.executor.execute(overlap_window, sm_configs)
