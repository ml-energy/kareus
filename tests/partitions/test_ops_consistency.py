#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Consistency test: compare Kareus operator implementations against
the TransformerEngine reference layer (te_transformer_layer.py).

Runs on a single GPU without distributed initialization.
Usage:
    python test_ops_consistency.py
    python test_ops_consistency.py --hidden_size 2048 --seq_len 512 --batch_size 2

For distributed comm-op tests (AllReduce, AllGatherKV, ReduceScatterKV):
    torchrun --nproc_per_node=2 test_ops_consistency.py --distributed
"""

import argparse
import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

import transformer_engine.pytorch as te
from transformer_engine.pytorch.ops.op import OperationContext

# Kareus ops
from kareus.megatron.core.extensions.ops.qkv_postprocess import QKVPostProcessOp
from kareus.megatron.core.extensions.ops.rotary_embedding import RotaryEmbeddingOp
from kareus.megatron.core.extensions.ops.bias_swiglu import BiasSwigluOp
from kareus.megatron.core.extensions.ops.residual_fork import ResidualForkOp
from kareus.megatron.core.extensions.ops.bias_dropout_add import BiasDropoutAddOp
from kareus.transformer_engine.pytorch.ops.basic.rmsnorm import RMSNorm as KareusRMSNorm
from kareus.transformer_engine.pytorch.ops import Linear as KareusLinear

# TE reference helpers
from te_transformer_layer import (
    qkv_post_process as ref_qkv_post_process,
    apply_rotary_embedding as ref_apply_rotary_embedding,
    swiglu as ref_swiglu,
    bias_swiglu as ref_bias_swiglu,
    create_rotary_pos_emb,
    TransformerLayerTE,
    ModelConfig,
)

# Megatron config (needed by RotaryEmbeddingOp and TEDotProductAttentionOp)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from kareus.megatron.core.extensions.ops.te_attention import TEDotProductAttentionOp

# Kareus comm ops
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce as KareusAllReduce
from kareus.transformer_engine.pytorch.ops.basic.all_gather_kv import AllGatherKV as KareusAllGatherKV
from kareus.transformer_engine.pytorch.ops.basic.reduce_scatter_kv import ReduceScatterKV as KareusReduceScatterKV


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def finalize_ctx(ctx: OperationContext):
    """Move to_save -> saved_tensors so op_backward can read them.

    The fuser framework normally does this; we do it manually in tests.
    """
    if ctx.to_save is not None:
        ctx.saved_tensors = ctx.to_save
        ctx.to_save = None


def allclose(a: torch.Tensor, b: torch.Tensor, name: str, atol: float = 1e-3, rtol: float = 1e-3):
    """Check two tensors are close and print diagnostics."""
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape mismatch {a.shape} vs {b.shape}")
        return False
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {name}: max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")
    return ok


def make_config(args) -> TransformerConfig:
    """Create a minimal TransformerConfig for single-GPU testing."""
    os.environ.setdefault("NVTE_APPLY_QK_LAYER_SCALING", "0")
    return TransformerConfig(
        num_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        kv_channels=args.head_dim,
        ffn_hidden_size=args.ffn_hidden_size,
        normalization="RMSNorm",
        layernorm_epsilon=args.layernorm_eps,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        apply_query_key_layer_scaling=False,
        apply_rope_fusion=True,
        rotary_interleaved=False,
        flash_decode=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
    )


# ---------------------------------------------------------------------------
# Individual Op Tests
# ---------------------------------------------------------------------------

def test_rmsnorm(args, device, dtype):
    """Compare Kareus RMSNorm (BasicOp) vs te.RMSNorm."""
    print("\n--- Test: RMSNorm ---")
    set_seed()

    te_norm = te.RMSNorm(args.hidden_size, eps=args.layernorm_eps, device=device, dtype=dtype)

    kareus_norm = KareusRMSNorm(
        normalized_shape=args.hidden_size,
        eps=args.layernorm_eps,
        device=device,
        dtype=dtype,
    )
    # Copy weights
    with torch.no_grad():
        kareus_norm.weight.copy_(te_norm.weight)

    x = torch.randn(args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype,
                     requires_grad=True)

    # --- Forward ---
    ref_out = te_norm(x)

    ctx = OperationContext()
    ctx.requires_grad = True
    kareus_out = kareus_norm.op_forward(ctx, x)

    ok = allclose(ref_out, kareus_out, "RMSNorm forward")

    # --- Backward ---
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out, retain_graph=True)
    ref_grad_x = x.grad
    # For kareus backward we need the saved tensors
    finalize_ctx(ctx)
    kareus_grad_x, kareus_grad_params = kareus_norm.op_backward(ctx, grad_out)

    if ref_grad_x is not None:
        ok &= allclose(ref_grad_x, kareus_grad_x, "RMSNorm backward grad_x")
    return ok


def test_qkv_postprocess(args, device, dtype):
    """Compare Kareus QKVPostProcessOp vs reference qkv_post_process.

    The Kareus op uses Megatron's grouped interleaved layout:
        [sq, b, ng, (np/ng)*hn + hn + hn]  (Q_group, K, V per group)
    while the reference uses a flat layout:
        [sq, b, q_all | k_all | v_all]
    We generate Q, K, V tensors independently and pack them into
    each format, then verify both produce the same Q, K, V.
    """
    print("\n--- Test: QKVPostProcess ---")
    set_seed()

    num_heads = args.num_attention_heads
    num_kv_heads = args.num_query_groups
    head_dim = args.head_dim
    heads_per_group = num_heads // num_kv_heads

    # Generate canonical Q, K, V
    q = torch.randn(args.seq_len, args.batch_size, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(args.seq_len, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(args.seq_len, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)

    # --- Build flat QKV for reference (q_all | k_all | v_all) ---
    flat_q = q.reshape(args.seq_len, args.batch_size, -1)                # [sq, b, np*hn]
    flat_k = k.reshape(args.seq_len, args.batch_size, -1)                # [sq, b, ng*hn]
    flat_v = v.reshape(args.seq_len, args.batch_size, -1)                # [sq, b, ng*hn]
    qkv_flat = torch.cat([flat_q, flat_k, flat_v], dim=-1)

    ref_q, ref_k, ref_v = ref_qkv_post_process(
        qkv_flat, num_heads, num_kv_heads, head_dim, args.seq_len, args.batch_size,
    )

    # --- Build grouped QKV for Kareus (per-group interleaved) ---
    # q reshaped to [sq, b, ng, heads_per_group, hn]
    q_grouped = q.view(args.seq_len, args.batch_size, num_kv_heads, heads_per_group, head_dim)
    # Per group: [q_group_flat, k_group, v_group]
    q_grouped_flat = q_grouped.reshape(args.seq_len, args.batch_size, num_kv_heads, heads_per_group * head_dim)
    qkv_grouped = torch.cat([q_grouped_flat, k, v], dim=-1)  # [sq, b, ng, (np/ng+2)*hn]
    qkv_megatron = qkv_grouped.reshape(args.seq_len, args.batch_size, -1)  # [sq, b, hp]

    op = QKVPostProcessOp(
        num_query_groups_per_partition=num_kv_heads,
        num_attention_heads_per_partition=num_heads,
        hidden_size_per_attention_head=head_dim,
    )
    ctx = OperationContext()
    kar_q, kar_k, kar_v = op.op_forward(ctx, qkv_megatron)

    ok = allclose(ref_q, kar_q, "QKVPost Q")
    ok &= allclose(ref_k, kar_k, "QKVPost K")
    ok &= allclose(ref_v, kar_v, "QKVPost V")

    # Also verify both match the canonical Q, K, V
    ok &= allclose(q, ref_q, "QKVPost Q vs canonical")
    ok &= allclose(k, ref_k, "QKVPost K vs canonical")
    ok &= allclose(v, ref_v, "QKVPost V vs canonical")
    return ok


def test_swiglu(args, device, dtype):
    """Compare Kareus BiasSwigluOp (no bias) vs reference swiglu."""
    print("\n--- Test: SwiGLU (no bias) ---")
    set_seed()

    x = torch.randn(args.seq_len, args.batch_size, 2 * args.ffn_hidden_size, device=device, dtype=dtype,
                     requires_grad=True)
    x_clone = x.detach().clone().requires_grad_(True)

    # Reference
    ref_out = ref_swiglu(x)

    # Kareus (no bias)
    op = BiasSwigluOp(fp8_input_store=False)
    # ctx = OperationContext()
    # kar_out = op.op_forward(ctx, x_clone, bias=None)
    kar_out = op(x_clone, None)

    ok = allclose(ref_out, kar_out, "SwiGLU forward")

    # Backward
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)

    # finalize_ctx(ctx)
    # kar_grad_x, (kar_grad_bias,) = op.op_backward(ctx, grad_out)
    kar_out.backward(grad_out)
    ok &= allclose(x.grad, x_clone.grad, "SwiGLU backward grad_x")
    return ok


def test_swiglu_with_bias(args, device, dtype):
    """Compare Kareus BiasSwigluOp (with bias) vs manual bias+swiglu."""
    print("\n--- Test: SwiGLU (with bias) ---")
    set_seed()

    x = torch.randn(args.seq_len, args.batch_size, 2 * args.ffn_hidden_size, device=device, dtype=dtype,
                     requires_grad=True)
    bias = torch.randn(2 * args.ffn_hidden_size, device=device, dtype=dtype, requires_grad=True)

    x_clone = x.detach().clone().requires_grad_(True)
    bias_clone = bias.detach().clone().requires_grad_(True)

    # Reference: bias + swiglu manually
    ref_out = ref_bias_swiglu(x, bias)

    # Kareus
    op = BiasSwigluOp(fp8_input_store=False)
    # ctx = OperationContext()
    # kar_out = op.op_forward(ctx, x_clone, bias=bias_clone)
    kar_out = op(x_clone, bias_clone)

    ok = allclose(ref_out, kar_out, "BiasSwiGLU forward")

    # Backward
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)

    # finalize_ctx(ctx)
    # kar_grad_x, (kar_grad_bias,) = op.op_backward(ctx, grad_out)
    kar_out.backward(grad_out)
    ok &= allclose(x.grad, x_clone.grad, "BiasSwiGLU backward grad_x")
    ok &= allclose(bias.grad, bias_clone.grad, "BiasSwiGLU backward grad_bias")
    return ok


def test_residual_fork(args, device, dtype):
    """Test ResidualForkOp: forward duplicates, backward accumulates."""
    print("\n--- Test: ResidualFork ---")
    set_seed()

    x = torch.randn(args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype)

    op = ResidualForkOp()
    ctx_list = [OperationContext()]
    main_out, extra_outs = op.fuser_forward(
        ctx_list, x,
        basic_op_extra_inputs=[()],
        basic_op_prev_ops=[None],
        basic_op_next_ops=[None],
        basic_op_kwargs=[{}],
    )
    residual_out = extra_outs[0][0]

    ok = allclose(x, main_out, "ResidualFork main")
    ok &= allclose(x, residual_out, "ResidualFork residual")

    # Backward: grad_main + grad_residual
    grad_main = torch.randn_like(x)
    grad_residual = torch.randn_like(x)
    grad_input, _, _ = op.fuser_backward(
        ctx_list, grad_main,
        basic_op_grad_extra_outputs=[(grad_residual,)],
    )
    expected = grad_main + grad_residual
    ok &= allclose(expected, grad_input, "ResidualFork backward")
    return ok


def test_bias_dropout_add(args, device, dtype):
    """Compare Kareus BiasDropoutAddOp (dropout=0) vs manual residual+x."""
    print("\n--- Test: BiasDropoutAdd (dropout=0, no bias) ---")
    set_seed()

    x = torch.randn(args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype,
                     requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)

    # Reference: residual + x (no bias, no dropout)
    ref_out = residual + x

    # Kareus
    op = BiasDropoutAddOp(dropout_prob=0.0, training=False)
    ctx = OperationContext()
    kar_out = op.op_forward(ctx, x, bias=None, residual=residual, dropout_prob=0.0, training=False)

    ok = allclose(ref_out, kar_out, "BDA forward")

    # Backward
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)

    finalize_ctx(ctx)
    kar_grad_x, (kar_grad_bias, kar_grad_residual) = op.op_backward(ctx, grad_out)
    ok &= allclose(x.grad, kar_grad_x, "BDA backward grad_x")
    ok &= allclose(residual.grad, kar_grad_residual, "BDA backward grad_residual")
    return ok


def test_rotary_embedding(args, config, device, dtype):
    """Compare Kareus RotaryEmbeddingOp vs reference apply_rotary_embedding."""
    print("\n--- Test: RotaryEmbedding ---")
    set_seed()

    num_heads = args.num_attention_heads
    num_kv_heads = args.num_query_groups
    head_dim = args.head_dim

    q = torch.randn(args.seq_len, args.batch_size, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(args.seq_len, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    rotary_pos_emb = create_rotary_pos_emb(args.seq_len, head_dim, device, dtype=torch.float32)

    # Reference (apex fused RoPE)
    ref_q, ref_k = ref_apply_rotary_embedding(q.clone(), k.clone(), rotary_pos_emb)

    # Kareus RotaryEmbeddingOp
    op = RotaryEmbeddingOp(config=config)
    ctx = OperationContext()
    kar_q, kar_k = op.op_forward(ctx, q.clone(), key=k.clone(), rotary_pos_emb=rotary_pos_emb)

    ok = allclose(ref_q, kar_q, "RoPE Q")
    ok &= allclose(ref_k, kar_k, "RoPE K")
    return ok


def test_attention(args, config, device, dtype):
    """Compare Kareus TEDotProductAttentionOp vs te.DotProductAttention."""
    print("\n--- Test: DotProductAttention ---")
    set_seed()

    num_heads = args.num_attention_heads
    num_kv_heads = args.num_query_groups
    head_dim = args.head_dim

    q = torch.randn(args.seq_len, args.batch_size, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(args.seq_len, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(args.seq_len, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Reference: te.DotProductAttention
    ref_attn = te.DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=head_dim,
        num_gqa_groups=num_kv_heads,
        attention_dropout=0.0,
        attn_mask_type="causal",
        qkv_format="sbhd",
    )
    ref_out = ref_attn(q.clone(), k.clone(), v.clone())

    # Kareus
    kar_attn = TEDotProductAttentionOp(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        attention_dropout=0.0,
    )
    kar_out = kar_attn.forward(
        q.clone(), k.clone(), v.clone(),
        attention_mask=None,
        attn_mask_type=AttnMaskType.causal,
    )

    ok = allclose(ref_out, kar_out, "Attention forward")
    return ok


def test_linear(args, device, dtype):
    """Compare Kareus BasicLinear op vs te.Linear."""
    print("\n--- Test: Linear ---")
    set_seed()

    from kareus.transformer_engine.pytorch.ops.basic import BasicLinear as KareusBasicLinear

    in_feat = args.hidden_size
    out_feat = args.ffn_hidden_size

    ref_linear = te.Linear(
        in_features=in_feat,
        out_features=out_feat,
        bias=False,
        device=device,
        params_dtype=dtype,
    )

    kar_linear = KareusBasicLinear(
        in_features=in_feat,
        out_features=out_feat,
        device=device,
        dtype=dtype,
    )

    # Copy weights
    with torch.no_grad():
        kar_linear.weight.copy_(ref_linear.weight)

    x = torch.randn(args.seq_len, args.batch_size, in_feat, device=device, dtype=dtype)

    ref_out = ref_linear(x)

    # Call through op_forward directly
    ctx = OperationContext()
    kar_out = kar_linear.op_forward(ctx, x)

    ok = allclose(ref_out, kar_out, "Linear forward")
    return ok


def test_linear_bias(args, device, dtype):
    """Compare Kareus BasicLinearBias op vs te.Linear (with bias).

    Tests both apply_bias (GEMM-fused) and return_bias modes,
    including forward output and backward gradients via fuser_forward/fuser_backward.
    """
    print("\n--- Test: LinearBias (apply_bias) ---")
    set_seed()

    from kareus.transformer_engine.pytorch.ops.basic import BasicLinearBias

    in_feat = args.hidden_size
    out_feat = args.ffn_hidden_size
    ok = True

    # ---------------------------------------------------------------
    # Mode 1: apply_bias=True (bias fused into GEMM)
    # ---------------------------------------------------------------
    ref_linear = te.Linear(
        in_features=in_feat,
        out_features=out_feat,
        bias=True,
        device=device,
        params_dtype=dtype,
    )

    kar_linear = BasicLinearBias(
        in_features=in_feat,
        out_features=out_feat,
        has_bias=True,
        apply_bias=True,
        return_bias=False,
        device=device,
        dtype=dtype,
    )

    # Copy weights and bias
    with torch.no_grad():
        kar_linear.weight.copy_(ref_linear.weight)
        kar_linear.bias.copy_(ref_linear.bias)

    x = torch.randn(args.seq_len, args.batch_size, in_feat, device=device, dtype=dtype,
                     requires_grad=True)
    x_clone = x.detach().clone().requires_grad_(True)

    # Reference forward
    ref_out = ref_linear(x)

    # Kareus forward via fuser_forward
    ctx = OperationContext()
    ctx.requires_grad = True
    kar_main, kar_extras = kar_linear.fuser_forward(
        [ctx], x_clone,
        basic_op_extra_inputs=[()],
        basic_op_prev_ops=[None],
        basic_op_next_ops=[None],
        basic_op_kwargs=[{}],
    )

    ok &= allclose(ref_out, kar_main, "LinearBias(apply) forward")

    # Check extra outputs: should be empty tuple (no extra output)
    assert kar_extras == [()], f"Expected empty extra outputs, got {kar_extras}"

    # Backward
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)

    finalize_ctx(ctx)
    kar_grad_input, kar_grad_params_list, _ = kar_linear.fuser_backward(
        [ctx], grad_out,
        basic_op_grad_extra_outputs=[()],
    )

    ok &= allclose(x.grad, kar_grad_input, "LinearBias(apply) backward grad_input")

    # grad_params = [grad_weight, grad_bias]
    kar_grad_weight, kar_grad_bias = kar_grad_params_list[0]
    ok &= allclose(ref_linear.weight.grad, kar_grad_weight, "LinearBias(apply) backward grad_weight")
    # apply_bias: grad_bias = grad_output (accumulated over seq & batch by autograd in ref)
    ref_bias_grad = ref_linear.bias.grad
    # Kareus grad_bias = grad_output (not reduced), so we reduce to match
    kar_bias_grad_reduced = kar_grad_bias.reshape(-1, out_feat).sum(dim=0)
    ok &= allclose(ref_bias_grad, kar_bias_grad_reduced, "LinearBias(apply) backward grad_bias")

    # ---------------------------------------------------------------
    # Mode 2: return_bias=True (bias returned as extra output)
    # ---------------------------------------------------------------
    print("\n--- Test: LinearBias (return_bias) ---")
    set_seed()

    ref_linear2 = te.Linear(
        in_features=in_feat,
        out_features=out_feat,
        bias=True,
        device=device,
        params_dtype=dtype,
    )

    kar_linear2 = BasicLinearBias(
        in_features=in_feat,
        out_features=out_feat,
        has_bias=True,
        apply_bias=False,
        return_bias=True,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        kar_linear2.weight.copy_(ref_linear2.weight)
        kar_linear2.bias.copy_(ref_linear2.bias)

    x2 = torch.randn(args.seq_len, args.batch_size, in_feat, device=device, dtype=dtype,
                      requires_grad=True)
    x2_clone = x2.detach().clone().requires_grad_(True)

    # Reference: linear without bias, then add bias manually
    ref_out2_no_bias = F.linear(x2, ref_linear2.weight)
    ref_out2 = ref_out2_no_bias + ref_linear2.bias

    # Kareus forward via fuser_forward (return_bias mode)
    ctx2 = OperationContext()
    ctx2.requires_grad = True
    kar_main2, kar_extras2 = kar_linear2.fuser_forward(
        [ctx2], x2_clone,
        basic_op_extra_inputs=[()],
        basic_op_prev_ops=[None],
        basic_op_next_ops=[None],
        basic_op_kwargs=[{}],
    )

    # Main output should be linear without bias
    ok &= allclose(ref_out2_no_bias, kar_main2, "LinearBias(return) forward main")
    # Extra output should be the bias
    assert len(kar_extras2) == 1 and len(kar_extras2[0]) == 1, \
        f"Expected 1 extra output, got {kar_extras2}"
    ok &= allclose(ref_linear2.bias, kar_extras2[0][0], "LinearBias(return) forward bias")

    # Backward with grad_bias from upstream
    grad_out2 = torch.randn_like(kar_main2)
    grad_bias_upstream = torch.randn(out_feat, device=device, dtype=dtype)

    # Reference backward: ref_out2 = F.linear(x2, W) + bias
    ref_out2.backward(grad_out2)

    finalize_ctx(ctx2)
    kar_grad_input2, kar_grad_params_list2, _ = kar_linear2.fuser_backward(
        [ctx2], grad_out2,
        basic_op_grad_extra_outputs=[(grad_bias_upstream,)],
    )

    ok &= allclose(x2.grad, kar_grad_input2, "LinearBias(return) backward grad_input")
    kar_grad_weight2, kar_grad_bias2 = kar_grad_params_list2[0]
    ok &= allclose(ref_linear2.weight.grad, kar_grad_weight2, "LinearBias(return) backward grad_weight")
    # return_bias: grad_bias comes from upstream (passed through)
    ok &= allclose(grad_bias_upstream, kar_grad_bias2, "LinearBias(return) backward grad_bias")

    # ---------------------------------------------------------------
    # Mode 3: has_bias=False (no bias parameter)
    # ---------------------------------------------------------------
    print("\n--- Test: LinearBias (no bias) ---")
    set_seed()

    ref_linear3 = te.Linear(
        in_features=in_feat,
        out_features=out_feat,
        bias=False,
        device=device,
        params_dtype=dtype,
    )

    kar_linear3 = BasicLinearBias(
        in_features=in_feat,
        out_features=out_feat,
        has_bias=False,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        kar_linear3.weight.copy_(ref_linear3.weight)

    x3 = torch.randn(args.seq_len, args.batch_size, in_feat, device=device, dtype=dtype,
                      requires_grad=True)
    x3_clone = x3.detach().clone().requires_grad_(True)

    ref_out3 = ref_linear3(x3)

    ctx3 = OperationContext()
    ctx3.requires_grad = True
    kar_main3, kar_extras3 = kar_linear3.fuser_forward(
        [ctx3], x3_clone,
        basic_op_extra_inputs=[()],
        basic_op_prev_ops=[None],
        basic_op_next_ops=[None],
        basic_op_kwargs=[{}],
    )

    ok &= allclose(ref_out3, kar_main3, "LinearBias(none) forward")
    assert kar_extras3 == [()], f"Expected empty extras, got {kar_extras3}"

    # Backward
    grad_out3 = torch.randn_like(ref_out3)
    ref_out3.backward(grad_out3)

    finalize_ctx(ctx3)
    kar_grad_input3, kar_grad_params_list3, _ = kar_linear3.fuser_backward(
        [ctx3], grad_out3,
        basic_op_grad_extra_outputs=[()],
    )

    ok &= allclose(x3.grad, kar_grad_input3, "LinearBias(none) backward grad_input")
    kar_grad_weight3, = kar_grad_params_list3[0]
    ok &= allclose(ref_linear3.weight.grad, kar_grad_weight3, "LinearBias(none) backward grad_weight")

    return ok


# ---------------------------------------------------------------------------
# Distributed Comm Op Tests
# ---------------------------------------------------------------------------

def init_distributed():
    """Initialize distributed process group for communication op tests.

    Must be launched via torchrun, e.g.:
        torchrun --nproc_per_node=2 test_ops_consistency.py --distributed
    """
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def test_all_reduce(args, device, dtype):
    """Compare Kareus AllReduce op vs torch.distributed.all_reduce (NCCL reference)."""
    print("\n--- Test: AllReduce ---")

    rank = torch.distributed.get_rank()
    pg = torch.distributed.distributed_c10d._get_default_group()

    # Each rank gets different data
    set_seed(42 + rank)
    x = torch.randn(args.seq_len, args.batch_size, args.hidden_size, device=device, dtype=dtype)

    # Reference: direct NCCL all-reduce (synchronous)
    x_ref = x.clone()
    torch.distributed.all_reduce(x_ref, group=pg)

    # Kareus AllReduce (async NCCL, then sync)
    op = KareusAllReduce(process_group=pg, async_op=True, backend="nccl")
    ctx = OperationContext()
    kareus_out = op.op_forward(ctx, x.clone())
    op.sync()

    ok = allclose(x_ref, kareus_out, "AllReduce forward")

    # Backward: identity pass-through
    grad = torch.randn_like(x)
    grad_out, _ = op.op_backward(ctx, grad.clone())
    ok &= allclose(grad, grad_out, "AllReduce backward")

    return ok


def test_all_gather_kv(args, device, dtype):
    """Compare Kareus AllGatherKV op vs torch.distributed.all_gather_into_tensor (NCCL reference)."""
    print("\n--- Test: AllGatherKV ---")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    pg = torch.distributed.distributed_c10d._get_default_group()

    num_kv_heads = args.num_query_groups
    head_dim = args.head_dim
    local_seq = args.seq_len

    # Each rank gets different local K, V
    set_seed(42 + rank)
    k_local = torch.randn(local_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_local = torch.randn(local_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Reference: direct NCCL all_gather_into_tensor
    k_ref = torch.empty(local_seq * world_size, args.batch_size, num_kv_heads, head_dim,
                         device=device, dtype=dtype)
    v_ref = torch.empty(local_seq * world_size, args.batch_size, num_kv_heads, head_dim,
                         device=device, dtype=dtype)
    torch.distributed.all_gather_into_tensor(k_ref, k_local.contiguous(), group=pg)
    torch.distributed.all_gather_into_tensor(v_ref, v_local.contiguous(), group=pg)

    # Kareus AllGatherKV (async NCCL, then sync)
    op = KareusAllGatherKV(process_group=pg, async_op=True, backend="nccl")
    ctx = OperationContext()
    k_kar, v_kar = op.op_forward(ctx, k_local.clone(), v_local.clone())
    op.sync()

    ok = allclose(k_ref, k_kar, "AllGatherKV K")
    ok &= allclose(v_ref, v_kar, "AllGatherKV V")

    return ok


def test_reduce_scatter_kv(args, device, dtype):
    """Compare Kareus ReduceScatterKV op vs torch.distributed.reduce_scatter_tensor (NCCL reference)."""
    print("\n--- Test: ReduceScatterKV ---")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    pg = torch.distributed.distributed_c10d._get_default_group()

    num_kv_heads = args.num_query_groups
    head_dim = args.head_dim
    # Full sequence divisible by world_size
    full_seq = args.seq_len * world_size

    # Each rank gets different full-size tensors
    set_seed(42 + rank)
    k_full = torch.randn(full_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_full = torch.randn(full_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Reference: direct NCCL reduce_scatter_tensor
    chunk_len = full_seq // world_size
    k_ref = torch.empty(chunk_len, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_ref = torch.empty(chunk_len, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    torch.distributed.reduce_scatter_tensor(k_ref, k_full.contiguous(), group=pg)
    torch.distributed.reduce_scatter_tensor(v_ref, v_full.contiguous(), group=pg)

    # Kareus ReduceScatterKV (async NCCL, then sync)
    op = KareusReduceScatterKV(process_group=pg, async_op=True, backend="nccl")
    ctx = OperationContext()
    k_kar, v_kar = op.op_forward(ctx, k_full.clone(), v_full.clone())
    op.sync()

    ok = allclose(k_ref, k_kar, "ReduceScatterKV K")
    ok &= allclose(v_ref, v_kar, "ReduceScatterKV V")

    return ok


# ---------------------------------------------------------------------------
# Distributed Comm Op Tests (msccl backend)
# ---------------------------------------------------------------------------

def test_all_reduce_msccl(args, device, dtype):
    """Compare Kareus AllReduce (msccl backend) vs torch.distributed.all_reduce.

    Follows the pattern from TransformerBlock.set_tensor_parallel_group().
    """
    print("\n--- Test: AllReduce (msccl) ---")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    pg = torch.distributed.distributed_c10d._get_default_group()
    current_stream = torch.cuda.current_stream()

    set_seed(42 + rank)
    tensor_size = [args.seq_len, args.batch_size, args.hidden_size]
    x = torch.randn(*tensor_size, device=device, dtype=dtype)

    # Reference: direct NCCL all-reduce
    x_ref = x.clone()
    torch.distributed.all_reduce(x_ref, group=pg)

    # Kareus AllReduce (msccl backend, kernel path)
    op = KareusAllReduce(
        process_group=pg,
        async_op=True,
        backend="msccl",
        rank=rank,
        world_size=world_size,
        tensor_size=tensor_size,
        device=device,
        dtype=dtype,
        batch_idx=0,
    )

    # msccl kernel operates on the persistent buffer in-place
    op.input_buffer.copy_(x)
    ctx = OperationContext()
    kareus_out = op.op_forward(
        ctx, op.input_buffer, sm_num=args.sm_num, block_size=args.block_size,
    )
    op.sync(current_stream)
    torch.cuda.synchronize()

    ok = allclose(x_ref, kareus_out, "AllReduce msccl forward")

    return ok


def test_all_gather_kv_msccl(args, device, dtype):
    """Compare Kareus AllGatherKV (msccl backend) vs torch.distributed.all_gather_into_tensor.

    Follows the pattern from TransformerBlock.set_context_parallel_group().
    """
    print("\n--- Test: AllGatherKV (msccl) ---")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    pg = torch.distributed.distributed_c10d._get_default_group()
    current_stream = torch.cuda.current_stream()

    num_kv_heads = args.num_query_groups
    head_dim = args.head_dim
    local_seq = args.seq_len
    full_seq = local_seq * world_size

    set_seed(42 + rank)
    k_local = torch.randn(local_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_local = torch.randn(local_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Reference: direct NCCL all_gather_into_tensor
    k_ref = torch.empty(full_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_ref = torch.empty(full_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    torch.distributed.all_gather_into_tensor(k_ref, k_local.contiguous(), group=pg)
    torch.distributed.all_gather_into_tensor(v_ref, v_local.contiguous(), group=pg)

    # Kareus AllGatherKV (msccl backend, kernel path)
    op = KareusAllGatherKV(
        process_group=pg,
        async_op=True,
        backend="msccl",
        rank=rank,
        world_size=world_size,
        tensor_size=[full_seq, args.batch_size, num_kv_heads, head_dim],
        device=device,
        dtype=dtype,
        batch_idx=0,
    )

    # msccl path copies local K/V into the buffer internally
    ctx = OperationContext()
    k_kar, v_kar = op.op_forward(
        ctx, k_local.clone(), v_local.clone(),
        sm_num=args.sm_num, block_size=args.block_size,
    )
    op.sync(current_stream)
    torch.cuda.synchronize()

    ok = allclose(k_ref, k_kar, "AllGatherKV msccl K")
    ok &= allclose(v_ref, v_kar, "AllGatherKV msccl V")

    return ok


def test_reduce_scatter_kv_msccl(args, device, dtype):
    """Compare Kareus ReduceScatterKV (msccl backend) vs torch.distributed.reduce_scatter_tensor.

    Follows the pattern from TransformerBlock.set_context_parallel_group().
    """
    print("\n--- Test: ReduceScatterKV (msccl) ---")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    pg = torch.distributed.distributed_c10d._get_default_group()
    current_stream = torch.cuda.current_stream()

    num_kv_heads = args.num_query_groups
    head_dim = args.head_dim
    full_seq = args.seq_len * world_size

    set_seed(42 + rank)
    k_full = torch.randn(full_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_full = torch.randn(full_seq, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Reference: direct NCCL reduce_scatter_tensor
    chunk_len = full_seq // world_size
    k_ref = torch.empty(chunk_len, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_ref = torch.empty(chunk_len, args.batch_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    torch.distributed.reduce_scatter_tensor(k_ref, k_full.contiguous(), group=pg)
    torch.distributed.reduce_scatter_tensor(v_ref, v_full.contiguous(), group=pg)

    # Kareus ReduceScatterKV (msccl backend, kernel path)
    op = KareusReduceScatterKV(
        process_group=pg,
        async_op=True,
        backend="msccl",
        rank=rank,
        world_size=world_size,
        tensor_size=[full_seq, args.batch_size, num_kv_heads, head_dim],
        device=device,
        dtype=dtype,
        batch_idx=0,
    )

    # msccl path copies full K/V into the buffer internally
    ctx = OperationContext()
    k_kar, v_kar = op.op_forward(
        ctx, k_full.clone(), v_full.clone(),
        sm_num=args.sm_num, block_size=args.block_size,
    )
    op.sync(current_stream)
    torch.cuda.synchronize()

    ok = allclose(k_ref, k_kar, "ReduceScatterKV msccl K")
    ok &= allclose(v_ref, v_kar, "ReduceScatterKV msccl V")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kareus ops vs TE reference consistency test")

    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--num_query_groups", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--ffn_hidden_size", type=int, default=8192)
    parser.add_argument("--layernorm_eps", type=float, default=1e-6)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--seq_len", "-s", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distributed", action="store_true",
                        help="Run distributed comm-op tests (requires torchrun)")
    parser.add_argument("--sm_num", type=int, default=24,
                        help="Number of SMs for msccl kernel launch")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="Thread block size for msccl kernel launch")

    args = parser.parse_args()
    args.head_dim = args.hidden_size // args.num_attention_heads

    # Initialize distributed before setting device when using torchrun
    if args.distributed:
        rank, world_size = init_distributed()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        if rank != 0:
            sys.stdout = open(os.devnull, "w")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("=" * 60)
    print("Kareus Ops vs TE Reference — Consistency Test")
    print("=" * 60)
    print(f"Device: {device}  Dtype: {dtype}")
    print(f"hidden_size={args.hidden_size}  heads={args.num_attention_heads}  "
          f"kv_heads={args.num_query_groups}  head_dim={args.head_dim}")
    print(f"ffn_hidden={args.ffn_hidden_size}  seq_len={args.seq_len}  batch={args.batch_size}")
    if args.distributed:
        print(f"distributed: rank={rank}  world_size={world_size}")
    print("=" * 60)

    set_seed(args.seed)
    config = make_config(args)

    results = {}

    # Individual op tests
    results["RMSNorm"] = test_rmsnorm(args, device, dtype)
    results["QKVPostProcess"] = test_qkv_postprocess(args, device, dtype)
    results["SwiGLU"] = test_swiglu(args, device, dtype)
    results["BiasSwiGLU"] = test_swiglu_with_bias(args, device, dtype)
    results["ResidualFork"] = test_residual_fork(args, device, dtype)
    results["BiasDropoutAdd"] = test_bias_dropout_add(args, device, dtype)
    results["RotaryEmbedding"] = test_rotary_embedding(args, config, device, dtype)
    results["Attention"] = test_attention(args, config, device, dtype)
    results["Linear"] = test_linear(args, device, dtype)
    results["LinearBias"] = test_linear_bias(args, device, dtype)

    # Distributed comm-op tests (require torchrun with >= 2 ranks)
    if args.distributed:
        results["AllReduce"] = test_all_reduce(args, device, dtype)
        results["AllGatherKV"] = test_all_gather_kv(args, device, dtype)
        results["ReduceScatterKV"] = test_reduce_scatter_kv(args, device, dtype)
        results["AllReduce_msccl"] = test_all_reduce_msccl(args, device, dtype)
        results["AllGatherKV_msccl"] = test_all_gather_kv_msccl(args, device, dtype)
        results["ReduceScatterKV_msccl"] = test_reduce_scatter_kv_msccl(args, device, dtype)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
