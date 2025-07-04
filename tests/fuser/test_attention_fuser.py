#!/usr/bin/env python3
"""
Test script for the attention fuser with the required operations:
- BDA (BiasDropoutAddOp)
- LayerNorm
- Linear_qkv (Linear transformation for queries, keys, values)
- post_process_qkv (QKVPostProcessOp)
- rotary embedding (RotaryEmbeddingOp)
- linear_proj (Linear projection for output)
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import sys
import os
import pytest
from typing import Optional, Tuple
import argparse
import random

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Import required operations
from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp
from transformer_engine.pytorch.ops.basic.layer_norm import LayerNorm
from kareus.transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.megatron.core.extensions.qkv_postprocess_op import create_qkv_postprocess_op
from kareus.megatron.core.extensions.rotary_embedding_op import create_rotary_embedding_op
from kareus.transformer_engine.pytorch.attention.dot_product_attention import DotProductAttentionOp
from kareus.megatron.core.extensions.te_linear import TEFusibleColumnParallelLinear, TEFusibleRowParallelLinear, TEFusibleLinear
from kareus.megatron.core.extensions.te_attention import TEFusibleDotProductAttention
from kareus.transformer_engine.pytorch.ops.linear import Linear
# Import attention fuser
from kareus.megatron.core.extensions.attention_fuser import AttentionFuser

# Import configuration
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def init_distributed(tensor_parallel_size: int = 1, backend: str = 'nccl'):
    """Initialize distributed processing for tensor parallelism.
    
    Parameters
    ----------
    tensor_parallel_size : int
        Size of tensor parallel group
    backend : str
        Distributed backend to use ('nccl', 'gloo', etc.)
        
    Returns
    -------
    torch.distributed.ProcessGroup or None
        Tensor parallel process group, or None if single process
    """
    if tensor_parallel_size <= 1:
        print("Single process mode - no distributed initialization needed")
        return None
        
    # Initialize the process group if not already initialized
    if not dist.is_initialized():
        # For testing, we'll use a single machine setup
        # In real scenarios, you'd have proper rank/world_size from environment
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', tensor_parallel_size))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set the device before initializing distributed
        torch.cuda.set_device(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        
        print(f"Initialized distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    # Create tensor parallel group
    if tensor_parallel_size > 1:
        # Create process groups for tensor parallelism
        ranks = list(range(tensor_parallel_size))
        tp_group = dist.new_group(ranks)
        print(f"Created tensor parallel group with ranks: {ranks}")
        return tp_group
    
    return None


class AttentionFuserTest:
    """Test suite for attention fuser operations."""

    def __init__(self, device='cuda', tensor_parallel_size: int = 1):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize distributed processing
        self.tp_group = init_distributed(tensor_parallel_size)
        
        # Test configuration
        self.batch_size = 2
        self.seq_length = 512
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.num_query_groups = 12  # For grouped query attention
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Create transformer config
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_query_groups=self.num_query_groups,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            qk_layernorm=False,
            apply_query_key_layer_scaling=False,
            rotary_interleaved=False,
            flash_decode=False,
            apply_rope_fusion=False,
            params_dtype=self.dtype,
        )

    def create_test_tensors(self):
        """Create test tensors for the attention operations."""
        # Input tensors
        hidden_states = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        # Bias and residual for BDA
        bias = torch.randn(
            self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        residual = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        # Rotary position embeddings
        rotary_pos_emb = torch.randn(
            self.batch_size, self.seq_length, self.num_attention_heads, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        
        # Attention mask (causal mask)
        # attention_mask = torch.tril(torch.ones(
        #     self.seq_length, self.seq_length,
        #     dtype=torch.bool, device=self.device
        # ))
        attention_mask = None

        allreduce_inputs = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        return hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs

    def create_operations(self):
        """Create all the required operations for the attention fuser."""
        
        # 1. BDA Operation (Bias Dropout Add)
        bda_op = BiasDropoutAddOp(
            dropout_prob=self.config.hidden_dropout,
            training=True
        )
        
        # 2. LayerNorm Operation
        layernorm_op = LayerNorm(
            normalized_shape=self.hidden_size,
            eps=self.config.layernorm_epsilon,
            device=self.device,
            dtype=self.dtype
        )
        
        # 3. Linear QKV Operation (transforms input to queries, keys, values)
        qkv_hidden_size = self.hidden_size * 3  # For Q, K, V combined
        # linear_qkv_op = TEFusibleColumnParallelLinear(
        #     self.hidden_size,
        #     qkv_hidden_size,
        #     config=self.config,
        #     init_method=self.config.output_layer_init_method,
        #     gather_output=False,
        #     bias=self.config.add_bias_linear,
        #     # input_is_parallel=True,
        #     skip_bias_add=False,
        #     is_expert=False,
        #     tp_comm_buffer_name='qkv',
        # )
        linear_qkv_op = Linear(
            in_features=self.hidden_size,
            out_features=qkv_hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=True,
            return_bias=False,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )
        # linear_qkv_op = TEFusibleLinear(
        #     input_size=self.hidden_size,
        #     output_size=qkv_hidden_size,
        #     parallel_mode="duplicated",
        #     config=self.config,
        #     init_method=self.config.output_layer_init_method,
        #     bias=True,
        #     skip_bias_add=False,
        #     is_expert=False,
        #     skip_weight_param_allocation=False,
        # )
        
        # 4. QKV Post-process Operation
        qkv_postprocess_op = create_qkv_postprocess_op(
            num_query_groups_per_partition=self.num_query_groups,
            num_attention_heads_per_partition=self.num_attention_heads,
            hidden_size_per_attention_head=self.head_dim,
            q_layernorm=None,
            k_layernorm=None,
        )
        
        # 5. Rotary Embedding Operation
        rotary_embedding_op = create_rotary_embedding_op(self.config)
        
        # 6. Dot Product Attention Operation
        attention_op = TEFusibleDotProductAttention(
            config=self.config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        
        # 7. Linear Projection Operation (output projection)
        # linear_proj_op = TEFusibleRowParallelLinear(
        #     self.hidden_size,
        #     self.hidden_size,
        #     config=self.config,
        #     init_method=self.config.output_layer_init_method,
        #     bias=self.config.add_bias_linear,
        #     input_is_parallel=True,
        #     skip_bias_add=True,
        #     is_expert=False,
        #     tp_comm_buffer_name='proj',
        # )
        linear_proj_op = Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=True,
            return_bias=True,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )
        # linear_proj_op = TEFusibleLinear(
        #     input_size=self.hidden_size,
        #     output_size=self.hidden_size,
        #     parallel_mode="duplicated",
        #     config=self.config,
        #     init_method=self.config.output_layer_init_method,
        #     bias=True,
        #     skip_bias_add=True,
        #     is_expert=False,
        #     skip_weight_param_allocation=False,
        # )
        
        # 8. AllReduce Communication Operation
        if self.tensor_parallel_size > 1:
            allreduce_comm_op = AllReduce(
                process_group=self.tp_group,
                async_op=True  # Use async mode as set in the modified AllReduce
            )
        
            return [
                bda_op,
                layernorm_op,
                linear_qkv_op,
                qkv_postprocess_op,
                rotary_embedding_op,
                attention_op,
                linear_proj_op,
                allreduce_comm_op
            ]

        else:
            return [
                bda_op,
                layernorm_op,
                linear_qkv_op,
                qkv_postprocess_op,
                rotary_embedding_op,
                attention_op,
                linear_proj_op,
            ]

    def test_individual_operations(self):
        """Test each operation individually to ensure they work correctly."""
        print("\n=== Testing Individual Operations ===")
        
        hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs = self.create_test_tensors()
        operations = self.create_operations()
        
        # Test BDA
        print("Testing BiasDropoutAddOp...")
        bda_op = operations[0]
        bda_output = bda_op(hidden_states, bias, residual)
        print(f"✓ BDA output shape: {bda_output.shape}")
        
        # Test LayerNorm
        print("Testing LayerNorm...")
        layernorm_op = operations[1]
        ln_output = layernorm_op(bda_output)
        print(f"✓ LayerNorm output shape: {ln_output.shape}")

        print(f"ln_output.dtype: {ln_output.dtype}")
        
        # Test Linear QKV
        print("Testing Linear QKV...")
        linear_qkv_op = operations[2]
        qkv_output, _ = linear_qkv_op(ln_output)
        print(f"✓ Linear QKV output shape: {qkv_output.shape}")

        print(f"qkv_output.dtype: {qkv_output.dtype}")
        
        # Test QKV Post-process
        print("Testing QKV Post-process...")
        qkv_postprocess_op = operations[3]
        q, k, v = qkv_postprocess_op(qkv_output)
        print(f"✓ QKV Post-process outputs - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
        
        # Test Rotary Embedding
        print("Testing Rotary Embedding...")
        rotary_embedding_op = operations[4]
        q_rope, k_rope = rotary_embedding_op(q, k, rotary_pos_emb)
        print(f"✓ Rotary Embedding outputs - Q: {q_rope.shape}, K: {k_rope.shape}")

        print(f"q_rope.dtype: {q_rope.dtype}")
        print(f"k_rope.dtype: {k_rope.dtype}")
        print(f"v.dtype: {v.dtype}")
        
        # Test Dot Product Attention
        print("Testing Dot Product Attention...")
        attention_op = operations[5]
        attn_output = attention_op(q_rope, k_rope, v, attention_mask, AttnMaskType.causal)
        print(f"✓ Attention output shape: {attn_output.shape}")
        
        # Test Linear Projection
        print("Testing Linear Projection...")
        linear_proj_op = operations[6]
        proj_output, _ = linear_proj_op(attn_output)
        print(f"✓ Linear Projection output shape: {proj_output.shape}")

        # Test AllReduce
        print("Testing AllReduce...")
        if self.tensor_parallel_size > 1:
            allreduce_comm_op = operations[7]
            allreduce_output = allreduce_comm_op(allreduce_inputs)
            if allreduce_comm_op.is_async_pending():
                print("  AllReduce operation is running asynchronously...")
                allreduce_comm_op.sync()
                print("  AllReduce operation synchronized successfully")
            print(f"✓ AllReduce output shape: {allreduce_output.shape}")
            
        # Verify AllReduce functionality
        if self.tensor_parallel_size > 1:
            print(f"  AllReduce performed with tensor parallel size: {self.tensor_parallel_size}")
        else:
            print("  AllReduce in single-process mode (no actual reduction)")
        
        # Test backward pass for individual operations
        print("\n--- Testing Individual Operations Backward Pass ---")
        
        # Create a loss from the final output
        loss = proj_output.sum()
        print(f"Loss value: {loss.item()}")
        
        # Clear any existing gradients
        for op in operations:
            for param in op.parameters():
                param.grad = None
        
        hidden_states.grad = None
        bias.grad = None
        residual.grad = None
        
        # Test backward pass
        print("Testing individual operations backward...")
        # try:
        loss.backward()
        print("✓ Individual operations backward pass successful")
        
        # Check gradients
        print("Checking individual operation gradients...")
        print(f"  Hidden states grad: {hidden_states.grad is not None}")
        print(f"  Bias grad: {bias.grad is not None}")
        print(f"  Residual grad: {residual.grad is not None}")
        
        # Check operation parameter gradients
        for i, op in enumerate(operations):
            param_count = sum(1 for _ in op.parameters())
            grad_count = sum(1 for param in op.parameters() if param.grad is not None)
            print(f"  Op {i} ({op.__class__.__name__}): {grad_count}/{param_count} params have gradients")
            
            # Check for finite gradients
            for name, param in op.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        print(f"    ✗ Non-finite gradient in op {i} param {name}")
                    else:
                        print(f"    ✓ Op {i} param {name}: gradient OK")
            
        # except Exception as e:
        #     print(f"✗ Individual operations backward failed: {str(e)}")
        #     import traceback
        #     traceback.print_exc()
        #     return False
        
        return True

    def test_attention_fuser(self):
        """Test the complete attention fuser with all operations."""
        print("\n=== Testing Attention Fuser ===")
        
        hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs = self.create_test_tensors()
        operations = self.create_operations()

        if self.tensor_parallel_size > 1:
            allreduce_comm_op = operations[7]
        else:
            allreduce_comm_op = None
        
        # Create attention fuser
        attention_fuser = AttentionFuser(
            ops=operations,
            allreduce_comm_op=allreduce_comm_op,
            fuse_ops=False
        )
        
        print("Attention fuser created successfully")
        print(f"Number of basic operations: {len(attention_fuser._basic_ops)}")
        print(f"Number of forward ops: {len(attention_fuser._forward_ops)}")
        print(f"Number of backward ops: {len(attention_fuser._backward_ops)}")
        
        # Test forward pass
        print("Testing fused forward pass...")
        # try:
        if self.tensor_parallel_size > 1:
            output, output_bias, output_residual, _ = attention_fuser(
                hidden_states=hidden_states,
                bias=bias,
                residual=residual,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                allreduce_inputs=allreduce_inputs,
                allreduce_overlap_window=(0, 1)
            )
        else:
            output, output_bias, output_residual, grad_allreduce_input = attention_fuser(
                hidden_states=hidden_states,
                bias=bias,
                residual=residual,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
            )
        
        print(f"✓ Fused forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output bias shape: {output_bias.shape}")
        print(f"  Output residual shape: {output_residual.shape}")
        
        # Test backward pass
        print("Testing fused backward pass...")
        loss = output.sum() + output_bias.sum() + output_residual.sum()

        if self.tensor_parallel_size > 1:
            loss += grad_allreduce_input.sum()

        loss.backward()
        
        print("✓ Fused backward pass successful")
        print(f"  Hidden states grad: {hidden_states.grad is not None}")
        print(f"  Bias grad: {bias.grad is not None}")
        print(f"  Residual grad: {residual.grad is not None}")

        if self.tensor_parallel_size > 1:
            print(f"  Grad allreduce input grad: {grad_allreduce_input.grad is not None}")
        
        return True
            
        # except Exception as e:
        #     print(f"✗ Fused attention failed: {str(e)}")
        #     import traceback
        #     traceback.print_exc()
        #     return False

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the fused operations."""
        print("\n=== Testing Gradient Flow ===")
        
        hidden_states, bias, residual, rotary_pos_emb, attention_mask = self.create_test_tensors()
        operations = self.create_operations()
        
        attention_fuser = AttentionFuser(
            ops=operations,
            fuse_ops=True
        )
        
        # Forward pass
        output, output_bias, output_residual = attention_fuser(
            hidden_states=hidden_states,
            bias=bias,
            residual=residual,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask
        )
        
        # Create a simple loss
        loss = output.mean() + output_bias.mean() + output_residual.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_checks = {
            "hidden_states": hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all(),
            "bias": bias.grad is not None and torch.isfinite(bias.grad).all(),
            "residual": residual.grad is not None and torch.isfinite(residual.grad).all(),
        }
        
        # Check operation parameter gradients
        for i, op in enumerate(operations):
            for name, param in op.named_parameters():
                if param.grad is not None:
                    finite_grad = torch.isfinite(param.grad).all()
                    grad_checks[f"op_{i}_{name}"] = finite_grad
                    if not finite_grad:
                        print(f"✗ Non-finite gradient in op {i} param {name}")
        
        all_grads_ok = all(grad_checks.values())
        if all_grads_ok:
            print("✓ All gradients are finite and properly computed")
        else:
            print("✗ Some gradients are missing or non-finite")
            for name, status in grad_checks.items():
                if not status:
                    print(f"  - {name}: {status}")
        
        return all_grads_ok

    def test_individual_vs_fuser_comparison(self):
        """Compare individual operations vs fuser with identical inputs."""
        print("\n=== Testing Individual vs Fuser Comparison ===")
        
        # Create identical test tensors for both tests
        set_random_seed(42)
        hidden_states1, bias1, residual1, rotary_pos_emb1, attention_mask1 = self.create_test_tensors()
        
        set_random_seed(42)
        hidden_states2, bias2, residual2, rotary_pos_emb2, attention_mask2 = self.create_test_tensors()
        
        # Create identical operations for both tests
        set_random_seed(42)
        operations1 = self.create_operations()
        
        set_random_seed(42)
        operations2 = self.create_operations()
        
        # Copy parameters to ensure they're identical
        for op1, op2 in zip(operations1, operations2):
            for (name1, param1), (name2, param2) in zip(op1.named_parameters(), op2.named_parameters()):
                param2.data.copy_(param1.data)
        
        print("=== Running Individual Operations ===")
        
        # Run individual operations
        x = hidden_states1
        
        # BDA
        x = operations1[0](x, bias1, residual1)
        print(f"Individual BDA output shape: {x.shape}")
        
        # LayerNorm
        x = operations1[1](x)
        print(f"Individual LayerNorm output shape: {x.shape}")
        
        # Linear QKV
        qkv_output, _ = operations1[2](x)
        print(f"Individual Linear QKV output shape: {qkv_output.shape}")
        
        # QKV Post-process
        q, k, v = operations1[3](qkv_output)
        print(f"Individual QKV Post-process - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
        
        # Rotary Embedding
        q_rope, k_rope = operations1[4](q, k, rotary_pos_emb1)
        print(f"Individual Rotary Embedding - Q: {q_rope.shape}, K: {k_rope.shape}")
        
        # Attention
        attn_output = operations1[5](q_rope, k_rope, v, attention_mask1, AttnMaskType.causal)
        print(f"Individual Attention output shape: {attn_output.shape}")
        
        # Linear Projection
        individual_output, individual_output_bias = operations1[6](attn_output)
        print(f"Individual Linear Projection output shape: {individual_output.shape}")
        
        # Individual backward
        individual_loss = individual_output.sum()
        print(f"Individual loss: {individual_loss.item()}")
        
        # Clear gradients
        for op in operations1:
            for param in op.parameters():
                param.grad = None
        hidden_states1.grad = None
        bias1.grad = None
        residual1.grad = None
        
        try:
            individual_loss.backward()
            print("✓ Individual backward successful")
            individual_backward_success = True
        except Exception as e:
            print(f"✗ Individual backward failed: {str(e)}")
            individual_backward_success = False
        
        print("\n=== Running Fuser Operations ===")
        
        # Run fuser
        attention_fuser = AttentionFuser(ops=operations2, fuse_ops=False)
        
        try:
            fuser_output, fuser_output_bias, fuser_output_residual = attention_fuser(
                hidden_states=hidden_states2,
                bias=bias2,
                residual=residual2,
                rotary_pos_emb=rotary_pos_emb2,
                attention_mask=attention_mask2
            )
            print(f"Fuser output shape: {fuser_output.shape}")
            print(f"Fuser output bias shape: {fuser_output_bias.shape}")
            print(f"Fuser output residual shape: {fuser_output_residual.shape}")
            
            fuser_loss = fuser_output.sum() + fuser_output_bias.sum() + fuser_output_residual.sum()
            print(f"Fuser loss: {fuser_loss.item()}")
            
            # Clear gradients
            for op in operations2:
                for param in op.parameters():
                    param.grad = None
            hidden_states2.grad = None
            bias2.grad = None
            residual2.grad = None
            
            try:
                fuser_loss.backward()
                print("✓ Fuser backward successful")
                fuser_backward_success = True
            except Exception as e:
                print(f"✗ Fuser backward failed: {str(e)}")
                print("This is where the cuBLAS error occurs!")
                import traceback
                traceback.print_exc()
                fuser_backward_success = False
                
        except Exception as e:
            print(f"✗ Fuser forward failed: {str(e)}")
            fuser_backward_success = False
        
        print("\n=== Comparison Results ===")
        print(f"Individual operations backward: {'✓ SUCCESS' if individual_backward_success else '✗ FAILED'}")
        print(f"Fuser operations backward: {'✓ SUCCESS' if fuser_backward_success else '✗ FAILED'}")
        
        if individual_backward_success and not fuser_backward_success:
            print("\n🔍 ANALYSIS: Individual operations work but fuser fails.")
            print("   This confirms the issue is in the fuser's gradient handling,")
            print("   specifically how it passes grad_extra_outputs to operations.")
        
        return individual_backward_success and fuser_backward_success

    def test_performance_comparison(self):
        """Compare performance of fused vs unfused operations."""
        print("\n=== Testing Performance Comparison ===")
        
        hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs = self.create_test_tensors()
        operations = self.create_operations()
        
        # Test unfused operations
        print("Testing unfused operations...")
        set_random_seed(42)
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_time.record()
        
        # Manual unfused forward pass
        x = operations[0](hidden_states, bias, residual)  # BDA
        x = operations[1](x)  # LayerNorm
        qkv, _ = operations[2](x)  # Linear QKV
        q, k, v = operations[3](qkv)  # QKV post-process
        q_rope, k_rope = operations[4](q, k, rotary_pos_emb)  # Rotary embedding
        attn_out = operations[5](q_rope, k_rope, v, attention_mask, AttnMaskType.causal)  # Attention
        final_out, _ = operations[6](attn_out)  # Linear projection
        allreduce_out = operations[7](allreduce_inputs)  # AllReduce
        if operations[7].is_async_pending():
            operations[7].sync()
        
        end_time.record()
        torch.cuda.synchronize()
        unfused_time = start_time.elapsed_time(end_time)
        
        print(f"Unfused operations time: {unfused_time:.3f} ms")
        
        # Test fused operations (if available)
        print("Testing fused operations...")
        set_random_seed(42)
        
        attention_fuser = AttentionFuser(ops=operations, fuse_ops=True)
        
        torch.cuda.synchronize()
        start_time.record()
        
        output, output_bias, output_residual = attention_fuser(
            hidden_states=hidden_states,
            bias=bias,
            residual=residual,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask
        )
        
        end_time.record()
        torch.cuda.synchronize()
        fused_time = start_time.elapsed_time(end_time)
        
        print(f"Fused operations time: {fused_time:.3f} ms")
        
        speedup = unfused_time / fused_time if fused_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        return True

    def run_all_tests(self):
        """Run all tests and return overall success status."""
        print("=" * 60)
        print("ATTENTION FUSER OPERATION TESTS")
        print("=" * 60)
        
        test_results = []
        
        # try:
        test_results.append(self.test_individual_operations())
        # except Exception as e:
        #     print(f"Individual operations test failed: {e}")
        #     test_results.append(False)
        
        # try:
        test_results.append(self.test_attention_fuser())
        # except Exception as e:
        #     print(f"Attention fuser test failed: {e}")
        #     test_results.append(False)
        
        # try:
        #     test_results.append(self.test_gradient_flow())
        # except Exception as e:
        #     print(f"Gradient flow test failed: {e}")
        #     test_results.append(False)
        
        # try:
        # test_results.append(self.test_individual_vs_fuser_comparison())
        # except Exception as e:
        #     print(f"Individual vs fuser comparison test failed: {e}")
        #     test_results.append(False)
        
        # try:
        #     test_results.append(self.test_performance_comparison())
        # except Exception as e:
        #     print(f"Performance comparison test failed: {e}")
        #     test_results.append(False)
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        test_names = [
            "Individual Operations",
            "Attention Fuser",
            # "Individual vs Fuser Comparison",
            # "Gradient Flow",
            # "Performance Comparison"
        ]
        
        for name, result in zip(test_names, test_results):
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{name}: {status}")
        
        overall_success = all(test_results)
        print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
        
        return overall_success


def run_process(rank, world_size, args, master_port):
    """Run the attention fuser tests in a distributed environment."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    # Create test instance and run tests
    test_runner = AttentionFuserTest(
        device=args.device, 
        tensor_parallel_size=args.tensor_parallel_size
    )
    success = test_runner.run_all_tests()

    # Clean up distributed if initialized
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed process group destroyed")


def main():
    """Main function to run the attention fuser tests."""
    parser = argparse.ArgumentParser(description='Attention Fuser Test with Distributed Support')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run tests on')
    parser.add_argument('--tensor-parallel-size', type=int, default=2, 
                        help='Tensor parallel size for distributed testing')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='Distributed backend (nccl, gloo)')
    
    args = parser.parse_args()
    
    print(f"Running tests with:")
    print(f"  Device: {args.device}")
    print(f"  Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"  Backend: {args.backend}")

    from torch.multiprocessing import spawn
    spawn(
        run_process,
        args=(
            args.tensor_parallel_size,
            args,
            random.randint(8000, 65535),
        ),
        nprocs=args.tensor_parallel_size,
        join=True,
    )


if __name__ == "__main__":
    exit(main())
