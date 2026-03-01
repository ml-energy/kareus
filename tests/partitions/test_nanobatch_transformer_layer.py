#!/usr/bin/env python
"""Validate nanobatch overlap path for a full transformer layer with TP.

This script compares:
1) Nanobatch partition execution using:
   - graph build logic equivalent to TransformerBlock._build_partitions
   - TransformerBlockAutogradFunction
2) Direct sequential execution on a full batch:
   - Manual chaining of all individual operators from get_all_operators():
     ResidualFork -> LayerNorm -> QKV -> QKVPost -> Rotary -> CoreAttn -> Proj
     -> AR -> BDA -> ResidualFork -> PreMLPLayerNorm -> FC1 -> SwiGLU -> FC2
     -> AR -> BDA
   with explicit all-reduce after each row-parallel linear (proj, fc2)
   but before the corresponding BiasDropoutAdd.

Decoupled from the full Megatron training loop but requires Megatron
parallel state for tensor parallelism.

The transformer layer pattern follows gpt_layer_specs.py:
  InputLayerNorm -> QKV -> QKVPost -> Rotary -> CoreAttn -> Proj -> AR
  -> PreMLPLayerNorm -> FC1 -> SwiGLU -> FC2 -> AR

Expected usage:
    torchrun --nproc_per_node=2 tests/partitions/test_nanobatch_transformer_layer.py
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from megatron.core.parallel_state import (
    initialize_model_parallel,
    destroy_model_parallel,
    get_tensor_model_parallel_group,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator

from kareus.megatron.core.transformer.transformer_layer import TransformerLayer
from kareus.megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from kareus.megatron.core.partitions import (
    CommunicationType,
    ComputeOp,
    PartitionBuilder,
    SeedConfig,
    TensorGraphBuilder,
    TransformerBlockAutogradFunction,
)
from kareus.megatron.core.partitions.tensor_graph import (
    CommunicationOp,
    ComputeOpSpec,
)
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import (
    AllReduce as KareusAllReduce,
)


# ---------------------------------------------------------------------------
#  Autograd-compatible all-reduce for the reference path
# ---------------------------------------------------------------------------


class _AllReduceFunc(torch.autograd.Function):
    """Forward: sum across TP group. Backward: identity."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        out = input_.clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def allreduce_autograd(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return _AllReduceFunc.apply(x, group)


def create_rotary_pos_emb(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    rotary_base: float = 10000.0,
) -> torch.Tensor:
    """Create rotary position embeddings [seq_len, 1, 1, head_dim]."""
    seq = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (
        rotary_base ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim)
    )
    freqs = torch.outer(seq, inv_freq)
    rotary_pos_emb = torch.cat((freqs, freqs), dim=-1)
    return rotary_pos_emb[:, None, None, :]


# ---------------------------------------------------------------------------
#  Reference: sequential full-batch forward
# ---------------------------------------------------------------------------


def _reference_forward(
    layer: TransformerLayer,
    x: torch.Tensor,
    tp_group: dist.ProcessGroup,
    rotary_pos_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute a single transformer layer sequentially on the full batch.

    Manually calls every operation from ``get_all_operators()`` in order,
    inserting explicit all-reduce after each row-parallel linear (proj and
    fc2) but before the corresponding BiasDropoutAdd.  This is the
    mathematical ground truth that the partition system must match.

    Operation order (matching get_all_operators):
        attn_residual_fork -> input_layernorm ->
        linear_qkv -> qkv_postprocess -> rotary_embedding ->
        core_attention -> linear_proj ->
        [ALL-REDUCE] -> self_attn_bda ->
        mlp_residual_fork -> pre_mlp_layernorm ->
        linear_fc1 -> activation -> linear_fc2 ->
        [ALL-REDUCE] -> mlp_bda
    """
    # ---- Attention block ----
    # attn_residual_fork: fork x into (main → LN, residual → BDA)
    residual = x
    h = x

    # input_layernorm
    h = layer.input_layernorm(h)

    # self_attention.get_compute_ops(): linear_qkv, qkv_post, rotary, core_attn, linear_proj
    mixed_qkv, _ = layer.self_attention.linear_qkv(h)
    query, key, value = layer.self_attention.qkv_postprocess_op(mixed_qkv)
    if rotary_pos_emb is not None:
        query, key = layer.self_attention.rotary_embedding_op(query, key, rotary_pos_emb)
    core_attn_out = layer.self_attention.core_attention(
        query, key, value, None,
        attn_mask_type=layer.self_attention.attn_mask_type,
    )
    proj_out, proj_bias = layer.self_attention.linear_proj(core_attn_out, 0)

    # ALL-REDUCE (after self_attention, before bda)
    proj_out = allreduce_autograd(proj_out, tp_group)

    # self_attn_bda: residual + dropout(proj_out + bias)
    if proj_bias is not None:
        h = layer.self_attn_bda(proj_out, proj_bias, residual)
    else:
        h = layer.self_attn_bda(proj_out, residual)

    # ---- MLP block ----
    # mlp_residual_fork: fork h into (main → LN, residual → BDA)
    residual = h
    mlp_in = h

    # pre_mlp_layernorm
    mlp_in = layer.pre_mlp_layernorm(mlp_in)

    # mlp.get_compute_ops(): linear_fc1, activation_op, linear_fc2
    fc1_out, fc1_bias = layer.mlp.linear_fc1(mlp_in)
    if fc1_bias is not None:
        intermediate = layer.mlp.activation_op(fc1_out, fc1_bias)
    else:
        intermediate = layer.mlp.activation_op(fc1_out)
    fc2_out, fc2_bias = layer.mlp.linear_fc2(intermediate, 0)

    # ALL-REDUCE (after mlp, before bda)
    fc2_out = allreduce_autograd(fc2_out, tp_group)

    # mlp_bda: residual + dropout(fc2_out + bias)
    if fc2_bias is not None:
        output = layer.mlp_bda(fc2_out, fc2_bias, residual)
    else:
        output = layer.mlp_bda(fc2_out, residual)

    return output


# ---------------------------------------------------------------------------
#  Nanobatch block (same graph/autograd path as TransformerBlock)
# ---------------------------------------------------------------------------


class TransformerLayerNanoBatchBlock(torch.nn.Module):
    """Single-layer block using the same graph/autograd path as TransformerBlock.

    Builds the partition graph from the layer's operators, assigns AllReduce
    comm ops, and executes through TransformerBlockAutogradFunction.
    """

    def __init__(self, layer: TransformerLayer, process_group: dist.ProcessGroup):
        super().__init__()
        self.layer = layer
        self._process_group = process_group
        self._build_partitions()
        self._assign_allreduce_comm_ops()

    def _build_partitions(self) -> None:
        all_ops = list(self.layer.get_all_operators())

        fwd_builder = TensorGraphBuilder()
        fwd_builder.add_initial_channels({
            "main": "ext_main",
            "rotary_pos_emb": "ext_rotary_pos_emb",
        })

        fwd_op_id_map = {}
        for op in all_ops:
            for spec in op.get_forward_ops():
                concrete_op = fwd_builder.add_op(spec)
                if isinstance(concrete_op, ComputeOp):
                    fwd_op_id_map[id(spec.operator)] = concrete_op.op_id
        self.forward_tensor_graph = fwd_builder.build()

        bwd_builder = TensorGraphBuilder()
        bwd_builder.add_initial_channels({"grad_main": "ext_grad_main"})
        for op in reversed(all_ops):
            for spec in op.get_backward_ops():
                if isinstance(spec, ComputeOpSpec) and spec.op_id is None:
                    fwd_id = fwd_op_id_map.get(id(spec.operator))
                    if fwd_id is not None:
                        spec.op_id = fwd_id
                bwd_builder.add_op(spec)
        self.backward_tensor_graph = bwd_builder.build()

        builder = PartitionBuilder(
            forward_graph=self.forward_tensor_graph,
            backward_graph=self.backward_tensor_graph,
        )
        self.forward_partitions = builder.build_forward_partitions()
        self.backward_partitions = builder.build_backward_partitions()
        self.seed_config = SeedConfig()

    def _assign_allreduce_comm_ops(self) -> None:
        pg = self._process_group
        comm_ops = [
            KareusAllReduce(process_group=pg, async_op=True, backend="nccl")
            for _ in range(2)
        ]
        for partition in self.forward_partitions + self.backward_partitions:
            if (
                partition.comm_op is not None
                and partition.comm_op.comm_type == CommunicationType.ALL_REDUCE
            ):
                partition.comm_op.operator = comm_ops[partition.nano_batch_idx]

    def _get_all_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for op in self.layer.get_all_operators():
            params.extend(list(op.parameters()))
        return params

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(1)
        if batch_size % 2 != 0:
            raise ValueError(
                f"Batch size must be even for 2-way nanobatch split, got {batch_size}"
            )
        mid = batch_size // 2
        h1 = hidden_states[:, :mid, ...]
        h2 = hidden_states[:, mid:, ...]

        all_params = self._get_all_params()
        is_grad_enabled = torch.is_grad_enabled()
        h1_out, h2_out = TransformerBlockAutogradFunction.apply(
            h1,
            h2,
            rotary_pos_emb,
            None,  # attention_mask
            self.forward_partitions,
            self.backward_partitions,
            self.forward_tensor_graph,
            self.backward_tensor_graph,
            None,  # scheduler
            None,  # config
            self.seed_config,
            is_grad_enabled,
            *all_params,
        )
        return torch.cat([h1_out, h2_out], dim=1)


# ---------------------------------------------------------------------------
#  Partition formation verification
# ---------------------------------------------------------------------------


def _verify_partition_formation(
    nano_block: TransformerLayerNanoBatchBlock,
    rank: int,
) -> None:
    """Verify the partition formation matches expected transformer layer structure.

    Expected forward graph (16 ops = 14 Compute + 2 Communication):
        ResidualFork -> LN -> QKV -> QKVPost -> Rotary -> CoreAttn -> Proj
        -> AR
        -> BDA -> ResidualFork -> PreMLPLN -> FC1 -> SwiGLU -> FC2
        -> AR
        -> MLP_BDA

    Expected segments (split at each ALL_REDUCE):
        Seg0: [ResidualFork, LN, QKV, QKVPost, Rotary, CoreAttn, Proj], comm=AR
        Seg1: [BDA, ResidualFork, PreMLPLN, FC1, SwiGLU, FC2],         comm=AR
        Seg2: [MLP_BDA],                                                comm=None

    Expected partitions (6 = 3 segments x 2 nanobatches):
        P0 (fwd_seg0_nb0): nb=0, comp=Attention(7 ops), comm=None
        P1 (fwd_seg0_nb1): nb=1, comp=Attention(7 ops), comm=AR  (reduces NB0 proj output)
        P2 (fwd_seg1_nb0): nb=0, comp=BDA+MLP(6 ops),  comm=AR  (reduces NB1 proj output)
        P3 (fwd_seg1_nb1): nb=1, comp=BDA+MLP(6 ops),  comm=AR  (reduces NB0 fc2 output)
        P4 (fwd_seg2_nb0): nb=0, comp=MLP_BDA(1 op),   comm=AR  (reduces NB1 fc2 output)
        P5 (fwd_seg2_nb1): nb=1, comp=MLP_BDA(1 op),   comm=None
    """
    layer = nano_block.layer
    fwd_graph = nano_block.forward_tensor_graph
    fwd_parts = nano_block.forward_partitions

    # --- 1. Check forward graph ops ---
    ops = fwd_graph.ops
    compute_ops = [op for op in ops if isinstance(op, ComputeOp)]
    comm_ops = [op for op in ops if isinstance(op, CommunicationOp)]

    assert len(compute_ops) == 14, (
        f"Expected 14 ComputeOps, got {len(compute_ops)}"
    )
    assert len(comm_ops) == 2, (
        f"Expected 2 CommunicationOps, got {len(comm_ops)}"
    )
    assert len(ops) == 16, (
        f"Expected 16 total ops, got {len(ops)}"
    )

    for c in comm_ops:
        assert c.comm_type == CommunicationType.ALL_REDUCE, (
            f"Expected ALL_REDUCE, got {c.comm_type}"
        )

    # Verify operator identity for compute ops
    expected_operators = [
        layer.attn_residual_fork,
        layer.input_layernorm,
        layer.self_attention.linear_qkv,
        layer.self_attention.qkv_postprocess_op,
        layer.self_attention.rotary_embedding_op,
        layer.self_attention.core_attention,
        layer.self_attention.linear_proj,
        # --- AR boundary ---
        layer.self_attn_bda,
        layer.mlp_residual_fork,
        layer.pre_mlp_layernorm,
        layer.mlp.linear_fc1,
        layer.mlp.activation_op,
        layer.mlp.linear_fc2,
        # --- AR boundary ---
        layer.mlp_bda,
    ]
    for i, (cop, expected_op) in enumerate(zip(compute_ops, expected_operators)):
        assert cop.operator is expected_op, (
            f"compute_ops[{i}].operator should be {type(expected_op).__name__}, "
            f"got {type(cop.operator).__name__}"
        )

    # --- 2. Check segment splitting ---
    segments = PartitionBuilder._split_by_communications(ops)
    assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"

    seg0_comp, seg0_comm = segments[0]
    seg1_comp, seg1_comm = segments[1]
    seg2_comp, seg2_comm = segments[2]

    assert len(seg0_comp) == 7, f"Seg0 should have 7 compute ops, got {len(seg0_comp)}"
    assert seg0_comm is not None and seg0_comm.comm_type == CommunicationType.ALL_REDUCE, (
        "Seg0 comm should be ALL_REDUCE"
    )
    assert len(seg1_comp) == 6, f"Seg1 should have 6 compute ops, got {len(seg1_comp)}"
    assert seg1_comm is not None and seg1_comm.comm_type == CommunicationType.ALL_REDUCE, (
        "Seg1 comm should be ALL_REDUCE"
    )
    assert len(seg2_comp) == 1, f"Seg2 should have 1 compute op, got {len(seg2_comp)}"
    assert seg2_comm is None, f"Seg2 comm should be None, got {seg2_comm}"

    # Verify segment operator identity
    assert seg0_comp[0].operator is layer.attn_residual_fork, "Seg0[0] should be ResidualFork"
    assert seg0_comp[-1].operator is layer.self_attention.linear_proj, "Seg0[-1] should be Proj"
    assert seg1_comp[0].operator is layer.self_attn_bda, "Seg1[0] should be self_attn_bda"
    assert seg1_comp[-1].operator is layer.mlp.linear_fc2, "Seg1[-1] should be FC2"
    assert seg2_comp[0].operator is layer.mlp_bda, "Seg2[0] should be mlp_bda"

    # --- 3. Check partition formation ---
    assert len(fwd_parts) == 6, f"Expected 6 forward partitions, got {len(fwd_parts)}"

    p0, p1, p2, p3, p4, p5 = fwd_parts

    # P0: fwd_seg0_nb0, nb=0, comp=7 (Attention block), comm=None
    assert p0.partition_key == "fwd_seg0_nb0", f"P0 key: {p0.partition_key}"
    assert p0.nano_batch_idx == 0, f"P0 nb: {p0.nano_batch_idx}"
    assert len(p0.comp_ops) == 7, f"P0 should have 7 comp_ops, got {len(p0.comp_ops)}"
    assert p0.comp_ops[0].operator is layer.attn_residual_fork, "P0 first op should be ResidualFork"
    assert p0.comp_ops[-1].operator is layer.self_attention.linear_proj, "P0 last op should be Proj"
    assert p0.comm_op is None, f"P0 comm should be None, got {p0.comm_op}"

    # P1: fwd_seg0_nb1, nb=1, comp=7 (Attention block), comm=AR (reduces NB0 proj output)
    assert p1.partition_key == "fwd_seg0_nb1", f"P1 key: {p1.partition_key}"
    assert p1.nano_batch_idx == 1, f"P1 nb: {p1.nano_batch_idx}"
    assert len(p1.comp_ops) == 7, f"P1 should have 7 comp_ops, got {len(p1.comp_ops)}"
    assert p1.comp_ops[0].operator is layer.attn_residual_fork, "P1 first op should be ResidualFork"
    assert p1.comp_ops[-1].operator is layer.self_attention.linear_proj, "P1 last op should be Proj"
    assert p1.comm_op is not None, "P1 should have a comm_op"
    assert p1.comm_op.comm_type == CommunicationType.ALL_REDUCE, "P1 comm should be ALL_REDUCE"

    # P2: fwd_seg1_nb0, nb=0, comp=6 (BDA + MLP), comm=AR (reduces NB1 proj output)
    assert p2.partition_key == "fwd_seg1_nb0", f"P2 key: {p2.partition_key}"
    assert p2.nano_batch_idx == 0, f"P2 nb: {p2.nano_batch_idx}"
    assert len(p2.comp_ops) == 6, f"P2 should have 6 comp_ops, got {len(p2.comp_ops)}"
    assert p2.comp_ops[0].operator is layer.self_attn_bda, "P2 first op should be self_attn_bda"
    assert p2.comp_ops[-1].operator is layer.mlp.linear_fc2, "P2 last op should be FC2"
    assert p2.comm_op is not None, "P2 should have a comm_op"
    assert p2.comm_op.comm_type == CommunicationType.ALL_REDUCE, "P2 comm should be ALL_REDUCE"

    # P1 and P2 comm ops should share the same tensor IDs (both cloned from AR1)
    assert p1.comm_op.input_ports[0].tensor_id == p2.comm_op.input_ports[0].tensor_id, (
        f"P1 and P2 comm should read same tensor: "
        f"P1={p1.comm_op.input_ports[0].tensor_id}, P2={p2.comm_op.input_ports[0].tensor_id}"
    )
    assert p1.comm_op.output_ports[0].tensor_id == p2.comm_op.output_ports[0].tensor_id, (
        f"P1 and P2 comm should write same tensor: "
        f"P1={p1.comm_op.output_ports[0].tensor_id}, P2={p2.comm_op.output_ports[0].tensor_id}"
    )

    # P3: fwd_seg1_nb1, nb=1, comp=6 (BDA + MLP), comm=AR (reduces NB0 fc2 output)
    assert p3.partition_key == "fwd_seg1_nb1", f"P3 key: {p3.partition_key}"
    assert p3.nano_batch_idx == 1, f"P3 nb: {p3.nano_batch_idx}"
    assert len(p3.comp_ops) == 6, f"P3 should have 6 comp_ops, got {len(p3.comp_ops)}"
    assert p3.comp_ops[0].operator is layer.self_attn_bda, "P3 first op should be self_attn_bda"
    assert p3.comp_ops[-1].operator is layer.mlp.linear_fc2, "P3 last op should be FC2"
    assert p3.comm_op is not None, "P3 should have a comm_op"
    assert p3.comm_op.comm_type == CommunicationType.ALL_REDUCE, "P3 comm should be ALL_REDUCE"

    # P4: fwd_seg2_nb0, nb=0, comp=1 (mlp_bda), comm=AR (reduces NB1 fc2 output)
    assert p4.partition_key == "fwd_seg2_nb0", f"P4 key: {p4.partition_key}"
    assert p4.nano_batch_idx == 0, f"P4 nb: {p4.nano_batch_idx}"
    assert len(p4.comp_ops) == 1, f"P4 should have 1 comp_op, got {len(p4.comp_ops)}"
    assert p4.comp_ops[0].operator is layer.mlp_bda, "P4 comp should be mlp_bda"
    assert p4.comm_op is not None, "P4 should have a comm_op"
    assert p4.comm_op.comm_type == CommunicationType.ALL_REDUCE, "P4 comm should be ALL_REDUCE"

    # P3 and P4 comm ops should share the same tensor IDs (both cloned from AR2)
    assert p3.comm_op.input_ports[0].tensor_id == p4.comm_op.input_ports[0].tensor_id, (
        f"P3 and P4 comm should read same tensor: "
        f"P3={p3.comm_op.input_ports[0].tensor_id}, P4={p4.comm_op.input_ports[0].tensor_id}"
    )
    assert p3.comm_op.output_ports[0].tensor_id == p4.comm_op.output_ports[0].tensor_id, (
        f"P3 and P4 comm should write same tensor: "
        f"P3={p3.comm_op.output_ports[0].tensor_id}, P4={p4.comm_op.output_ports[0].tensor_id}"
    )

    # P1/P2 and P3/P4 comm ops should have DIFFERENT tensor IDs (different AR boundaries)
    assert p1.comm_op.input_ports[0].tensor_id != p3.comm_op.input_ports[0].tensor_id, (
        f"AR1 and AR2 should have different input tensors: "
        f"AR1={p1.comm_op.input_ports[0].tensor_id}, AR2={p3.comm_op.input_ports[0].tensor_id}"
    )

    # P5: fwd_seg2_nb1, nb=1, comp=1 (mlp_bda), comm=None
    assert p5.partition_key == "fwd_seg2_nb1", f"P5 key: {p5.partition_key}"
    assert p5.nano_batch_idx == 1, f"P5 nb: {p5.nano_batch_idx}"
    assert len(p5.comp_ops) == 1, f"P5 should have 1 comp_op, got {len(p5.comp_ops)}"
    assert p5.comp_ops[0].operator is layer.mlp_bda, "P5 comp should be mlp_bda"
    assert p5.comm_op is None, f"P5 comm should be None, got {p5.comm_op}"

    # --- 4. Check AllReduce operator assignment ---
    for p in [p1, p2, p3, p4]:
        assert p.comm_op.operator is not None, (
            f"{p.partition_key} comm_op.operator should be assigned"
        )

    # NB0 partitions (P2, P4) should share one operator; NB1 partitions (P1, P3) another
    assert p2.comm_op.operator is p4.comm_op.operator, (
        "P2 and P4 (both nb=0) should share the same AllReduce operator"
    )
    assert p1.comm_op.operator is p3.comm_op.operator, (
        "P1 and P3 (both nb=1) should share the same AllReduce operator"
    )
    assert p1.comm_op.operator is not p2.comm_op.operator, (
        "NB0 and NB1 AllReduce operators should be different instances"
    )

    # --- 5. Check tensor wiring consistency ---
    # The final output channel should exist
    final_tid = fwd_graph.get_output_channel("main")
    assert final_tid is not None, "Forward graph should have 'main' output channel"

    if rank == 0:
        print("[PARTITION CHECK] Forward graph: OK "
              f"(16 ops: 14 Compute + 2 Communication)")
        print("[PARTITION CHECK] Operator order: OK "
              "(ResidualFork→LN→QKV→QKVPost→Rotary→CoreAttn→Proj→AR"
              "→BDA→ResidualFork→PreMLPLN→FC1→SwiGLU→FC2→AR→MLP_BDA)")
        print("[PARTITION CHECK] Segments: OK "
              "(3 segments: [Attn(7),AR], [BDA+MLP(6),AR], [MLP_BDA(1),None])")
        print(f"[PARTITION CHECK] Partitions: OK (6 partitions)")
        for i, p in enumerate(fwd_parts):
            comm_str = (
                f"AR({p.comm_op.input_ports[0].tensor_id}"
                f"→{p.comm_op.output_ports[0].tensor_id})"
                if p.comm_op is not None else "None"
            )
            print(f"  P{i}: {p.partition_key} nb={p.nano_batch_idx} "
                  f"comp=[{len(p.comp_ops)} ops] comm={comm_str}")
        print("[PARTITION CHECK] AllReduce assignment: OK "
              "(nb0→op0, nb1→op1, distinct per nanobatch)")
        print(f"[PARTITION CHECK] Final output tensor: {final_tid}")


# ---------------------------------------------------------------------------
#  Execution behavior verification
# ---------------------------------------------------------------------------


def _verify_execution_behavior(
    nano_block: TransformerLayerNanoBatchBlock,
    ref_layer: TransformerLayer,
    x: torch.Tensor,
    tp_group: dist.ProcessGroup,
    rotary_pos_emb: Optional[torch.Tensor],
    rank: int,
) -> None:
    """Verify step-by-step partition execution matches expected behavior.

    Expected execution for 2 nanobatches (NB0, NB1) with 3 segments:
      Step 1 - P0: Attention(NB0)                        [no comm]
      Step 2 - P1: Attention(NB1) + AR(NB0 proj output)  [comm on NB0]
      Step 3 - P2: BDA+MLP(NB0)  + AR(NB1 proj output)  [comm on NB1]
      Step 4 - P3: BDA+MLP(NB1)  + AR(NB0 fc2 output)   [comm on NB0]
      Step 5 - P4: MLP_BDA(NB0)  + AR(NB1 fc2 output)   [comm on NB1]
      Step 6 - P5: MLP_BDA(NB1)                          [no comm]

    Computes ground-truth intermediates per nanobatch using ref_layer,
    then manually steps through partitions and compares tensor store values.
    """
    from kareus.megatron.core.partitions.context_manager import NanoBatchContext

    mid = x.size(1) // 2
    x_nb0 = x[:, :mid, :].detach().clone()
    x_nb1 = x[:, mid:, :].detach().clone()

    # --- Compute reference intermediates per nanobatch ---
    def _ref_fwd(h):
        """Run reference forward on a single nanobatch, returning key intermediates."""
        h = h.contiguous()
        residual_attn = h

        ln_out = ref_layer.input_layernorm(h)
        mixed_qkv, _ = ref_layer.self_attention.linear_qkv(ln_out)
        q, k, v = ref_layer.self_attention.qkv_postprocess_op(mixed_qkv)
        if rotary_pos_emb is not None:
            q, k = ref_layer.self_attention.rotary_embedding_op(
                q, k, rotary_pos_emb,
            )
        attn_out = ref_layer.self_attention.core_attention(
            q, k, v, None,
            attn_mask_type=ref_layer.self_attention.attn_mask_type,
        )
        proj_out, proj_bias = ref_layer.self_attention.linear_proj(attn_out, 0)

        ar_proj = proj_out.clone()
        dist.all_reduce(ar_proj, group=tp_group)

        if proj_bias is not None:
            bda_out = ref_layer.self_attn_bda(ar_proj, proj_bias, residual_attn)
        else:
            bda_out = ref_layer.self_attn_bda(ar_proj, residual_attn)

        residual_mlp = bda_out
        mlp_in = ref_layer.pre_mlp_layernorm(bda_out)
        fc1_out, fc1_bias = ref_layer.mlp.linear_fc1(mlp_in)
        if fc1_bias is not None:
            swiglu_out = ref_layer.mlp.activation_op(fc1_out, fc1_bias)
        else:
            swiglu_out = ref_layer.mlp.activation_op(fc1_out)
        fc2_out, fc2_bias = ref_layer.mlp.linear_fc2(swiglu_out, 0)

        ar_fc2 = fc2_out.clone()
        dist.all_reduce(ar_fc2, group=tp_group)

        if fc2_bias is not None:
            output = ref_layer.mlp_bda(ar_fc2, fc2_bias, residual_mlp)
        else:
            output = ref_layer.mlp_bda(ar_fc2, residual_mlp)

        return {
            "proj_out": proj_out, "ar_proj": ar_proj,
            "fc2_out": fc2_out, "ar_fc2": ar_fc2, "final": output,
        }

    with torch.no_grad():
        refs_nb0 = _ref_fwd(x_nb0)
        refs_nb1 = _ref_fwd(x_nb1)

    # --- Get tensor IDs from partition structure ---
    partitions = nano_block.forward_partitions
    proj_tid = partitions[0].comp_ops[-1].output_ports[0].tensor_id
    ar_proj_tid = partitions[1].comm_op.output_ports[0].tensor_id
    fc2_tid = partitions[2].comp_ops[-1].output_ports[0].tensor_id
    ar_fc2_tid = partitions[3].comm_op.output_ports[0].tensor_id
    final_tid = partitions[4].comp_ops[-1].output_ports[0].tensor_id

    if rank == 0:
        print(f"\n=== Execution Verification ===")
        print(f"  Tensor IDs: proj={proj_tid}, ar_proj={ar_proj_tid}, "
              f"fc2={fc2_tid}, ar_fc2={ar_fc2_tid}, final={final_tid}")

    # --- Create contexts and seed ---
    ctx_nb0 = NanoBatchContext(batch_idx=0)
    ctx_nb1 = NanoBatchContext(batch_idx=1)
    ctx_nb0.tensor_store.set("ext_main", x_nb0)
    ctx_nb1.tensor_store.set("ext_main", x_nb1)
    if rotary_pos_emb is not None:
        ctx_nb0.tensor_store.set("ext_rotary_pos_emb", rotary_pos_emb)
        ctx_nb1.tensor_store.set("ext_rotary_pos_emb", rotary_pos_emb)

    for p in partitions:
        p.is_grad_enabled = False

    checks_passed = True
    first_fail_step = None

    def _check(step, label, actual, expected, atol=1e-3):
        nonlocal checks_passed, first_fail_step
        diff = (actual - expected).abs().max().item()
        ok = diff < atol
        if not ok and first_fail_step is None:
            first_fail_step = step
            checks_passed = False
        if not ok:
            checks_passed = False
        if rank == 0:
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {label}: max_diff={diff:.6e}")

    # Step 1: P0 — Attention(NB0), no comm
    partitions[0].execute(ctx=ctx_nb0, pre_ctx=ctx_nb1)
    if rank == 0:
        print("Step 1 - P0: Attention(NB0), no comm")
    _check(1, "NB0 proj_out", ctx_nb0.tensor_store.get(proj_tid), refs_nb0["proj_out"])

    # Step 2: P1 — Attention(NB1) + AR(NB0)
    partitions[1].execute(ctx=ctx_nb1, pre_ctx=ctx_nb0)
    if rank == 0:
        print("Step 2 - P1: Attention(NB1) + AR(NB0)")
    _check(2, "NB1 proj_out", ctx_nb1.tensor_store.get(proj_tid), refs_nb1["proj_out"])
    _check(2, "NB0 ar_proj", ctx_nb0.tensor_store.get(ar_proj_tid), refs_nb0["ar_proj"])

    # Step 3: P2 — BDA+MLP(NB0) + AR(NB1)
    partitions[2].execute(ctx=ctx_nb0, pre_ctx=ctx_nb1)
    if rank == 0:
        print("Step 3 - P2: BDA+MLP(NB0) + AR(NB1)")
    _check(3, "NB0 fc2_out", ctx_nb0.tensor_store.get(fc2_tid), refs_nb0["fc2_out"])
    _check(3, "NB1 ar_proj", ctx_nb1.tensor_store.get(ar_proj_tid), refs_nb1["ar_proj"])

    # Step 4: P3 — BDA+MLP(NB1) + AR(NB0)
    partitions[3].execute(ctx=ctx_nb1, pre_ctx=ctx_nb0)
    if rank == 0:
        print("Step 4 - P3: BDA+MLP(NB1) + AR(NB0)")
    _check(4, "NB1 fc2_out", ctx_nb1.tensor_store.get(fc2_tid), refs_nb1["fc2_out"])
    _check(4, "NB0 ar_fc2", ctx_nb0.tensor_store.get(ar_fc2_tid), refs_nb0["ar_fc2"])

    # Step 5: P4 — MLP_BDA(NB0) + AR(NB1)
    partitions[4].execute(ctx=ctx_nb0, pre_ctx=ctx_nb1)
    if rank == 0:
        print("Step 5 - P4: MLP_BDA(NB0) + AR(NB1)")
    _check(5, "NB0 final", ctx_nb0.tensor_store.get(final_tid), refs_nb0["final"])
    _check(5, "NB1 ar_fc2", ctx_nb1.tensor_store.get(ar_fc2_tid), refs_nb1["ar_fc2"])

    # Step 6: P5 — MLP_BDA(NB1), no comm
    partitions[5].execute(ctx=ctx_nb1, pre_ctx=ctx_nb0)
    if rank == 0:
        print("Step 6 - P5: MLP_BDA(NB1), no comm")
    _check(6, "NB1 final", ctx_nb1.tensor_store.get(final_tid), refs_nb1["final"])

    if rank == 0:
        if checks_passed:
            print("[EXECUTION CHECK] All intermediate values match reference!")
        else:
            print(f"[EXECUTION CHECK] FAILED — first divergence at step {first_fail_step}")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _assert_close(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch: max_diff={_max_diff(a, b):.6e}, "
            f"a_shape={tuple(a.shape)}, b_shape={tuple(b.shape)}"
        )


def _init_distributed() -> Tuple[int, int, int]:
    """Lightweight distributed init using only torch.distributed."""
    if not dist.is_initialized():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _make_config(args, world_size: int) -> TransformerConfig:
    """Create a minimal TransformerConfig for the test."""
    os.environ.setdefault("NVTE_APPLY_QK_LAYER_SCALING", "0")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        kv_channels=args.hidden_size // args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        apply_query_key_layer_scaling=False,
        apply_rope_fusion=True,
        rotary_interleaved=False,
        flash_decode=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=world_size,
        context_parallel_size=1,
        sequence_parallel=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=True,
    )
    config.max_sequence_length = args.seq_len
    return config


def _build_layer(config: TransformerConfig, device) -> TransformerLayer:
    """Build a single TransformerLayer from the GPT TE spec."""
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    layer = TransformerLayer(
        config=config,
        submodules=layer_spec.submodules,
        layer_number=1,
    )
    return layer.to(device)


def _copy_params(src: TransformerLayer, dst: TransformerLayer) -> None:
    """Copy all parameters from src to dst."""
    with torch.no_grad():
        for p_src, p_dst in zip(src.parameters(), dst.parameters()):
            p_dst.copy_(p_src)


# ---------------------------------------------------------------------------
#  Test runner
# ---------------------------------------------------------------------------


def run_case(args: argparse.Namespace) -> None:
    rank, world_size, _ = _init_distributed()
    if args.batch_size % 2 != 0:
        raise ValueError("--batch-size must be even for two nanobatches")
    if world_size < 2:
        raise ValueError("Need at least 2 GPUs for tensor parallelism")

    # Megatron parallel state
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        create_gloo_process_groups=False,
    )

    # Microbatch calculator (TELinearOp reads get_micro_batch_size() at init)
    init_num_microbatches_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=args.batch_size,
        micro_batch_size=args.batch_size,
        data_parallel_size=1,
    )

    tp_group = get_tensor_model_parallel_group()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config = _make_config(args, world_size)

    # Build two identical layers (separate instances, shared weights)
    ref_layer = _build_layer(config, device)
    test_layer = _build_layer(config, device)
    _copy_params(ref_layer, test_layer)

    # Random input (same on all ranks since seeds are identical)
    torch.manual_seed(args.seed + 100)
    x = torch.randn(
        args.seq_len,
        args.batch_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )

    x_ref = x.detach().clone().requires_grad_(True)
    x_test = x.detach().clone().requires_grad_(True)

    head_dim = args.hidden_size // args.num_attention_heads
    rotary_pos_emb = create_rotary_pos_emb(
        seq_len=args.seq_len,
        head_dim=head_dim,
        device=device,
        dtype=torch.float32,
    )

    # ---- Reference: full-batch forward ----
    y_ref = _reference_forward(
        ref_layer, x_ref, tp_group, rotary_pos_emb=rotary_pos_emb,
    )
    loss_ref = y_ref.float().pow(2).mean()
    loss_ref.backward()

    # ---- Test: nanobatch partition execution ----
    nano_block = TransformerLayerNanoBatchBlock(test_layer, tp_group)

    # ---- Partition formation verification ----
    _verify_partition_formation(nano_block, rank)

    # ---- Execution behavior verification ----
    _verify_execution_behavior(
        nano_block, ref_layer, x, tp_group, rotary_pos_emb, rank,
    )

    y_test = nano_block(x_test, rotary_pos_emb=rotary_pos_emb)
    loss_test = y_test.float().pow(2).mean()
    loss_test.backward()

    atol = args.atol
    rtol = args.rtol

    # ---- Diagnostics ----
    mid = args.batch_size // 2
    if rank == 0:
        print(f"\n--- Forward output diagnostics ---")
        print(f"  Full output diff: {_max_diff(y_test, y_ref):.6e}")
        print(
            f"  NB0 diff ([:,:mid]): "
            f"{_max_diff(y_test[:, :mid, :], y_ref[:, :mid, :]):.6e}"
        )
        print(
            f"  NB1 diff ([:,mid:]): "
            f"{_max_diff(y_test[:, mid:, :], y_ref[:, mid:, :]):.6e}"
        )
        print(f"  y_ref  norm: {y_ref.float().norm():.4f}")
        print(f"  y_test norm: {y_test.float().norm():.4f}")

    # ---- Assertions ----
    _assert_close("forward_output", y_test, y_ref, atol=atol, rtol=rtol)
    _assert_close("input_grad", x_test.grad, x_ref.grad, atol=atol, rtol=rtol)

    # Parameter gradients
    ref_params = list(ref_layer.named_parameters())
    test_params = list(test_layer.named_parameters())
    for (name_ref, p_ref), (name_test, p_test) in zip(ref_params, test_params):
        if p_ref.grad is not None and p_test.grad is not None:
            _assert_close(
                f"param_grad({name_ref})",
                p_test.grad,
                p_ref.grad,
                atol=atol,
                rtol=rtol,
            )
        elif (p_ref.grad is None) != (p_test.grad is None):
            raise AssertionError(
                f"Gradient existence mismatch for {name_ref}: "
                f"ref has grad={p_ref.grad is not None}, "
                f"test has grad={p_test.grad is not None}"
            )

    dist.barrier()
    if rank == 0:
        print(
            f"\n[PASS] Nanobatch transformer layer matches sequential execution "
            f"(seq={args.seq_len}, batch={args.batch_size}, "
            f"hidden={args.hidden_size}, tp={world_size})"
        )


def _cleanup() -> None:
    try:
        destroy_model_parallel()
    except Exception:
        pass
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-query-groups", type=int, default=8)
    parser.add_argument("--ffn-hidden-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_case(args)
    finally:
        _cleanup()
