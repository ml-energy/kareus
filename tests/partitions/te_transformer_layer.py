#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test run a Transformer layer using original TransformerEngine interfaces.
"""

# module load singularity

# singularity exec --cleanenv --nv pytorch_26.01-py3.sif bash -lc '
# export CC=/usr/bin/gcc
# export CXX=/usr/bin/g++
# /sbin/ldconfig -n /.singularity.d/libs || true
# export TRITON_LIBCUDA_PATH=/.singularity.d/libs
# export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
# nsys profile -c cudaProfilerApi -f true -o test_transformer_layer --gpu-metrics-devices 0 python test_transformer_layer.py
# '

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_engine.pytorch as te
from apex.transformer.functional import fused_apply_rotary_pos_emb


# Insert a ~1ms delay on a GPU running at ~1.5 GHz
gpu_clock_hz = 1.98e9  # Adjust for your GPU (check with nvidia-smi -q -d CLOCK)
delay_seconds = 1e-3  # 1 millisecond
cycles = int(delay_seconds * gpu_clock_hz)


# ============================================================================
# Model Configuration (Llama 3.2 3B style, but can be overridden)
# ============================================================================
class ModelConfig:
    HIDDEN_SIZE = 4096
    NUM_ATTENTION_HEADS = 32
    HEAD_DIM = 128  # hidden_size / num_attention_heads
    NUM_QUERY_GROUPS = 8  # GQA groups (num_key_value_heads)
    FFN_HIDDEN_SIZE = 12288
    LAYERNORM_EPS = 1e-6


# ============================================================================
# Custom Operations (no external dependencies)
# ============================================================================

def qkv_post_process(
    qkv: torch.Tensor,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split and reshape QKV tensor into separate Q, K, V tensors.
    
    Input shape: [seq_len, batch_size, qkv_hidden_size]
    Output shapes:
        Q: [seq_len, batch_size, num_attention_heads, head_dim]
        K: [seq_len, batch_size, num_query_groups, head_dim]
        V: [seq_len, batch_size, num_query_groups, head_dim]
    """
    # Calculate sizes
    q_size = num_attention_heads * head_dim
    kv_size = num_query_groups * head_dim
    
    # Split QKV
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    
    # Reshape to [seq_len, batch_size, num_heads, head_dim]
    q = q.view(seq_len, batch_size, num_attention_heads, head_dim)
    k = k.view(seq_len, batch_size, num_query_groups, head_dim)
    v = v.view(seq_len, batch_size, num_query_groups, head_dim)
    
    return q, k, v


def apply_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to Q and K using fused CUDA kernel from apex.
    
    Args:
        q: [seq_len, batch_size, num_heads, head_dim]
        k: [seq_len, batch_size, num_kv_heads, head_dim]
        rotary_pos_emb: [seq_len, 1, 1, head_dim] - precomputed frequency embeddings (float dtype)
    
    Returns:
        Rotated q and k tensors with same shapes as input.
    """
    # Use fused RoPE kernel from apex
    # transpose_output_memory=False keeps the output in sbhd format
    q_embed = fused_apply_rotary_pos_emb(q, rotary_pos_emb, transpose_output_memory=False)
    k_embed = fused_apply_rotary_pos_emb(k, rotary_pos_emb, transpose_output_memory=False)
    
    return q_embed, k_embed


@torch.compile
def _swiglu_fwd(y: torch.Tensor) -> torch.Tensor:
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@torch.compile
def _bias_swiglu_fwd(y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    y = y + bias
    return _swiglu_fwd(y)


@torch.compile
def _swiglu_back(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )


@torch.compile
def _bias_swiglu_back(g: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    y = y + bias
    return _swiglu_back(g, y)


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return _swiglu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return _swiglu_back(grad_output, input)


class BiasSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return _bias_swiglu_fwd(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = _bias_swiglu_back(grad_output, input, bias)
        return tmp, tmp


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation with explicit backward formula."""
    return SwiGLUFunction.apply(x)


def bias_swiglu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Bias + SwiGLU activation with explicit backward formula."""
    return BiasSwiGLUFunction.apply(x, bias)


def create_rotary_pos_emb(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    rotary_base: float = 10000.0,
) -> torch.Tensor:
    """
    Create rotary position embeddings.
    
    Returns tensor of shape [seq_len, 1, 1, head_dim] containing the frequencies.
    """
    seq = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (
        rotary_base ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim)
    )
    freqs = torch.outer(seq, inv_freq)
    # Concatenate for full head_dim
    rotary_pos_emb = torch.cat((freqs, freqs), dim=-1)
    # Add batch and head dimensions: [seq_len, 1, 1, head_dim]
    return rotary_pos_emb[:, None, None, :]


# ============================================================================
# Transformer Layer using TE interfaces
# ============================================================================

class TransformerLayerTE(nn.Module):
    """
    A single Transformer layer using TransformerEngine primitives.
    
    Architecture:
        - Pre-attention RMSNorm
        - QKV Linear projection
        - QKV post-process (split/reshape)
        - Rotary position embedding
        - DotProductAttention
        - Output Linear projection
        - Residual add (no dropout)
        - Pre-MLP RMSNorm
        - FC1 Linear (gated, 2x ffn_hidden)
        - SwiGLU activation
        - FC2 Linear
        - Residual add (no dropout)
    """
    
    def __init__(
        self,
        hidden_size: int = ModelConfig.HIDDEN_SIZE,
        num_attention_heads: int = ModelConfig.NUM_ATTENTION_HEADS,
        num_query_groups: int = ModelConfig.NUM_QUERY_GROUPS,
        head_dim: int = ModelConfig.HEAD_DIM,
        ffn_hidden_size: int = ModelConfig.FFN_HIDDEN_SIZE,
        layernorm_eps: float = ModelConfig.LAYERNORM_EPS,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.head_dim = head_dim
        self.ffn_hidden_size = ffn_hidden_size
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype
        
        # QKV projection output size
        qkv_hidden_size = (
            num_attention_heads * head_dim +  # Q
            num_query_groups * head_dim +     # K
            num_query_groups * head_dim       # V
        )
        
        # Attention components
        self.attn_norm = te.RMSNorm(
            normalized_shape=hidden_size,
            eps=layernorm_eps,
            device=device,
            dtype=dtype,
        )
        
        self.qkv_proj = te.Linear(
            in_features=hidden_size,
            out_features=qkv_hidden_size,
            bias=False,
            device=device,
            params_dtype=dtype,
        )
        
        self.attn = te.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=head_dim,
            num_gqa_groups=num_query_groups,
            attention_dropout=0.0,  # No dropout
            attn_mask_type="causal",
            qkv_format="sbhd",  # [seq, batch, head, dim]
        )
        
        self.out_proj = te.Linear(
            in_features=num_attention_heads * head_dim,
            out_features=hidden_size,
            bias=False,
            device=device,
            params_dtype=dtype,
        )
        
        # MLP components
        self.mlp_norm = te.RMSNorm(
            normalized_shape=hidden_size,
            eps=layernorm_eps,
            device=device,
            dtype=dtype,
        )
        
        # FC1: gate and value together (2x ffn_hidden for SwiGLU)
        self.fc1 = te.Linear(
            in_features=hidden_size,
            out_features=2 * ffn_hidden_size,
            bias=False,
            device=device,
            params_dtype=dtype,
        )
        
        self.fc2 = te.Linear(
            in_features=ffn_hidden_size,
            out_features=hidden_size,
            bias=False,
            device=device,
            params_dtype=dtype,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer layer.
        
        Args:
            hidden_states: [seq_len, batch_size, hidden_size]
            rotary_pos_emb: [seq_len, 1, 1, head_dim]
        
        Returns:
            output: [seq_len, batch_size, hidden_size]
        """
        seq_len, batch_size, _ = hidden_states.shape
        residual = hidden_states
        
        # ======== Attention Block ========
        # Pre-attention norm
        hidden_states = self.attn_norm(hidden_states)
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)

        # torch.cuda._sleep(cycles) # insert a ~1ms delay
        
        # Split and reshape QKV
        q, k, v = qkv_post_process(
            qkv,
            num_attention_heads=self.num_attention_heads,
            num_query_groups=self.num_query_groups,
            head_dim=self.head_dim,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        
        # Apply rotary embeddings
        q, k = apply_rotary_embedding(q, k, rotary_pos_emb)
        
        # Attention
        attn_out = self.attn(q, k, v)

        # torch.cuda._sleep(cycles) # insert a ~1ms delay
        
        # Reshape attention output: [seq, batch, heads, dim] -> [seq, batch, hidden]
        attn_out = attn_out.view(seq_len, batch_size, -1)
        
        # Output projection
        attn_out = self.out_proj(attn_out)

        # torch.cuda._sleep(cycles) # insert a ~1ms delay
        
        # Residual add (no dropout)
        hidden_states = residual + attn_out
        
        # ======== MLP Block ========
        residual = hidden_states
        
        # Pre-MLP norm
        hidden_states = self.mlp_norm(hidden_states)
        
        # FC1 with SwiGLU
        hidden_states = self.fc1(hidden_states)
        hidden_states = swiglu(hidden_states)
        
        # FC2
        hidden_states = self.fc2(hidden_states)
        
        # Residual add (no dropout)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ============================================================================
# Test Runner
# ============================================================================

def run_test(args):
    """Run the transformer layer test."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print("=" * 60)
    print("Transformer Layer Test (TransformerEngine)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Num attention heads: {args.num_attention_heads}")
    print(f"Num query groups (GQA): {args.num_query_groups}")
    print(f"Head dim: {args.head_dim}")
    print(f"FFN hidden size: {args.ffn_hidden_size}")
    print("=" * 60)
    
    # Create model
    model = TransformerLayerTE(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        head_dim=args.head_dim,
        ffn_hidden_size=args.ffn_hidden_size,
        device=device,
        dtype=dtype,
    )
    model.eval()
    
    # Create input tensors
    hidden_states = torch.randn(
        args.seq_len, args.batch_size, args.hidden_size,
        dtype=dtype, device=device,
    )
    
    # Create rotary position embeddings
    rotary_pos_emb = create_rotary_pos_emb(
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        device=device,
        dtype=torch.float32,  # RoPE typically in float32
    )
    
    print(f"\nInput shape: {hidden_states.shape}")
    print(f"Rotary embedding shape: {rotary_pos_emb.shape}")
    
    # Warmup
    print("\nWarmup...")
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            output = model(hidden_states, rotary_pos_emb)
    torch.cuda.synchronize()
    
    # Run iterations
    print(f"\nRunning {args.num_iters} iterations...")
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        for i in range(args.num_iters):
            output = model(hidden_states, rotary_pos_emb)
            if i == 0:
                print(f"Output shape: {output.shape}")
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    
    print("\nTest completed successfully!")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Test Transformer Layer with TransformerEngine")
    
    # Model configuration
    parser.add_argument("--hidden_size", type=int, default=ModelConfig.HIDDEN_SIZE,
                        help="Hidden size (default: 8192)")
    parser.add_argument("--num_attention_heads", type=int, default=ModelConfig.NUM_ATTENTION_HEADS,
                        help="Number of attention heads (default: 64)")
    parser.add_argument("--num_query_groups", type=int, default=ModelConfig.NUM_QUERY_GROUPS,
                        help="Number of GQA groups (default: 8)")
    parser.add_argument("--head_dim", type=int, default=ModelConfig.HEAD_DIM,
                        help="Head dimension (default: 128)")
    parser.add_argument("--ffn_hidden_size", type=int, default=ModelConfig.FFN_HIDDEN_SIZE,
                        help="FFN hidden size (default: 28672)")
    
    # Test configuration
    parser.add_argument("--batch_size", "-b", type=int, default=8,
                        help="Batch size (default: 2)")
    parser.add_argument("--seq_len", "-s", type=int, default=4096,
                        help="Sequence length (default: 4096)")
    parser.add_argument("--warmup_iters", type=int, default=5,
                        help="Number of warmup iterations (default: 5)")
    parser.add_argument("--num_iters", type=int, default=100,
                        help="Number of test iterations (default: 10)")
    
    args = parser.parse_args()
    
    run_test(args)


if __name__ == "__main__":
    main()

