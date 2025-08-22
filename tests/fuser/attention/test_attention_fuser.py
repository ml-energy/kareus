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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Import required operations
from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp
from kareus.transformer_engine.pytorch.ops.basic.layer_norm import LayerNorm
from kareus.transformer_engine.pytorch.ops.basic.rmsnorm import RMSNorm
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
from kareus.megatron.core.extensions.partition_fuser import PartitionFuser

# Import configuration
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from cfuser.core.utils import nvtx_range

from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)


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
        self.dtype = torch.bfloat16
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize distributed processing
        self.tp_group = init_distributed(tensor_parallel_size)
        
        # Test configuration
        self.batch_size = 2
        self.seq_length = 4096
        self.hidden_size = 2048
        self.num_attention_heads = 32
        self.num_query_groups = 8  # For grouped query attention
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
            apply_rope_fusion=True,
            add_bias_linear=False,
            params_dtype=self.dtype,
        )

    def create_test_tensors(self):
        """Create test tensors for the attention operations."""
        # Input tensors
        hidden_states = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        # Bias and residual for BDA
        # bias = torch.randn(
        #     self.hidden_size,
        #     dtype=self.dtype, device=self.device, requires_grad=True
        # )
        bias = None
        residual = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        # Rotary position embeddings
        # rotary_pos_emb = torch.randn(
        #     self.batch_size, self.seq_length, self.num_attention_heads, self.head_dim,
        #     dtype=self.dtype, device=self.device
        # )
        seq = (
            torch.arange(self.seq_length, device=self.device, dtype=torch.float32)
            + 0
        )
        rotary_base = 10000
        inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )
        freqs = torch.outer(seq, inv_freq)
        rotary_pos_emb = torch.cat((freqs, freqs), dim=-1)
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]
        
        # Attention mask (causal mask)
        # attention_mask = torch.tril(torch.ones(
        #     self.seq_length, self.seq_length,
        #     dtype=torch.bool, device=self.device
        # ))
        attention_mask = None

        allreduce_inputs = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        return hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs

    def create_operations(self, rank: int, world_size: int):
        """Create all the required operations for the attention fuser."""
        
        # 1. BDA Operation (Bias Dropout Add)
        bda_op = BiasDropoutAddOp(
            dropout_prob=self.config.hidden_dropout,
            training=True
        )
        
        # 2. LayerNorm Operation
        layernorm_op = RMSNorm(
            normalized_shape=self.hidden_size,
            eps=self.config.layernorm_epsilon,
            device=self.device,
            dtype=self.dtype
        )
        
        # 3. Linear QKV Operation (transforms input to queries, keys, values)
        qkv_hidden_size = (
            self.num_attention_heads * self.head_dim +  # Query heads
            self.num_query_groups * self.head_dim +     # Key heads
            self.num_query_groups * self.head_dim       # Value heads
        )
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
            bias=False,
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
            bias=False,
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
                async_op=True,  # Use async mode as set in the modified AllReduce
                backend="msccl",
                rank=rank,
                world_size=world_size,
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

    def test_individual_operations(self, rank: int, world_size: int):
        """Test each operation individually to ensure they work correctly."""
        print("\n=== Testing Individual Operations ===")
        
        hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs = self.create_test_tensors()
        operations = self.create_operations(rank, world_size)
        
        # Helper function to check for infinite/NaN values
        def check_tensor_health(tensor, name):
            if torch.isnan(tensor).any():
                print(f"⚠️  NaN detected in {name}")
                return False
            if torch.isinf(tensor).any():
                print(f"⚠️  Inf detected in {name}")
                return False
            print(f"✓ {name} is healthy (min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f})")
            return True
        
        # Check initial tensors
        print("\n--- Checking Initial Tensors ---")
        check_tensor_health(hidden_states, "hidden_states")
        # check_tensor_health(bias, "bias")
        check_tensor_health(residual, "residual")
        check_tensor_health(rotary_pos_emb, "rotary_pos_emb")

        print(f"\nhidden_states.shape: {hidden_states.shape}")
        # print(f"bias.shape: {bias.shape}")
        print(f"residual.shape: {residual.shape}")
        print(f"rotary_pos_emb.shape: {rotary_pos_emb.shape}")
        print(f"allreduce_inputs.shape: {allreduce_inputs.shape}")
        
        # Test BDA
        print("\nTesting BiasDropoutAddOp...")
        bda_op = operations[0]
        with nvtx_range("BDA"):
            bda_output = bda_op(hidden_states, bias, residual)
        print(f"✓ BDA output shape: {bda_output.shape}")
        if not check_tensor_health(bda_output, "bda_output"):
            print("❌ BDA operation produced invalid values!")
            return False
        
        # Test LayerNorm
        print("\nTesting LayerNorm...")
        layernorm_op = operations[1]
        with nvtx_range("LayerNorm"):
            ln_output = layernorm_op(bda_output)
        print(f"✓ LayerNorm output shape: {ln_output.shape}")
        print(f"ln_output.dtype: {ln_output.dtype}")
        if not check_tensor_health(ln_output, "ln_output"):
            print("❌ LayerNorm operation produced invalid values!")
            return False
        
        # Test Linear QKV
        print("\nTesting Linear QKV...")
        linear_qkv_op = operations[2]
        with nvtx_range("Linear QKV"):
            qkv_output, _ = linear_qkv_op(ln_output)
        print(f"✓ Linear QKV output shape: {qkv_output.shape}")
        print(f"qkv_output.dtype: {qkv_output.dtype}")
        if not check_tensor_health(qkv_output, "qkv_output"):
            print("❌ Linear QKV operation produced invalid values!")
            return False
        
        # Test QKV Post-process
        print("\nTesting QKV Post-process...")
        qkv_postprocess_op = operations[3]
        with nvtx_range("QKV Post-process"):
            q, k, v = qkv_postprocess_op(qkv_output)
        print(f"✓ QKV Post-process outputs - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
        print(f"q.shape: {q.shape}")
        print(f"k.shape: {k.shape}")
        print(f"v.shape: {v.shape}")
        if not (check_tensor_health(q, "q") and check_tensor_health(k, "k") and check_tensor_health(v, "v")):
            print("❌ QKV Post-process operation produced invalid values!")
            return False
        
        # Test Rotary Embedding
        print("\nTesting Rotary Embedding...")
        rotary_embedding_op = operations[4]
        with nvtx_range("Rotary Embedding"):
            q_rope, k_rope = rotary_embedding_op(q, k, rotary_pos_emb)
        print(f"✓ Rotary Embedding outputs - Q: {q_rope.shape}, K: {k_rope.shape}")
        print(f"q_rope.dtype: {q_rope.dtype}")
        print(f"k_rope.dtype: {k_rope.dtype}")
        print(f"v.dtype: {v.dtype}")
        if not (check_tensor_health(q_rope, "q_rope") and check_tensor_health(k_rope, "k_rope")):
            print("❌ Rotary Embedding operation produced invalid values!")
            return False
        
        # Test Dot Product Attention
        print("\nTesting Dot Product Attention...")
        attention_op = operations[5]
        with nvtx_range("Attention"):
            attn_output = attention_op(q_rope, k_rope, v, attention_mask, AttnMaskType.causal)
        print(f"✓ Attention output shape: {attn_output.shape}")
        print(f"attn_output.shape: {attn_output.shape}")
        if not check_tensor_health(attn_output, "attn_output"):
            print("❌ Attention operation produced invalid values!")
            return False
        
        # Test Linear Projection
        print("\nTesting Linear Projection...")
        linear_proj_op = operations[6]
        with nvtx_range("Linear Projection"):
            proj_output, _ = linear_proj_op(attn_output)
        print(f"✓ Linear Projection output shape: {proj_output.shape}")
        if not check_tensor_health(proj_output, "proj_output"):
            print("❌ Linear Projection operation produced invalid values!")
            return False

        # Test AllReduce
        print("Testing AllReduce...")
        if self.tensor_parallel_size > 1:
            allreduce_comm_op = operations[7]
            with nvtx_range("AllReduce"):
                allreduce_output = allreduce_comm_op(allreduce_inputs, sm_num=4, block_size=1024)
                allreduce_comm_op.sync()
            print(f"✓ AllReduce output shape: {allreduce_output.shape}")
            
        # Verify AllReduce functionality
        if self.tensor_parallel_size > 1:
            print(f"  AllReduce performed with tensor parallel size: {self.tensor_parallel_size}")
        else:
            print("  AllReduce in single-process mode (no actual reduction)")
        
        # Test backward pass for individual operations
        print("\n--- Testing Individual Operations Backward Pass ---")
        
        # Create a loss from the final output (use float32 to avoid overflow)
        loss = proj_output.float().sum()
        print(f"Loss value: {loss.item()}")
        
        # Check if loss is finite
        if not torch.isfinite(loss):
            print("❌ Loss is not finite! Cannot proceed with backward pass.")
            return False
        
        # Clear any existing gradients
        for op in operations:
            for param in op.parameters():
                param.grad = None
        
        hidden_states.grad = None
        # bias.grad = None
        residual.grad = None
        
        # Test backward pass
        print("Testing individual operations backward...")
        # try:
        with nvtx_range("Individual loss.backward"):
            loss.backward()
        print("✓ Individual operations backward pass successful")
        
        # Check gradients
        print("Checking individual operation gradients...")
        print(f"  Hidden states grad: {hidden_states.grad is not None}")
        # print(f"  Bias grad: {bias.grad is not None}")
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

    def test_attention_fuser(self, rank: int, world_size: int):
        """Test the complete attention fuser with all operations."""
        print("\n=== Testing Attention Fuser ===")
        
        hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs = self.create_test_tensors()
        operations = self.create_operations(rank, world_size)

        if self.tensor_parallel_size > 1:
            allreduce_comm_op = operations[7]
        else:
            allreduce_comm_op = None
        
        # Create attention fuser
        attention_fuser = PartitionFuser(
            ops=operations[:7],
            allreduce_comm_op=allreduce_comm_op,
            fuse_ops=True
        )
        
        print("Attention fuser created successfully")
        print(f"Number of basic operations: {len(attention_fuser._basic_ops)}")
        print(f"forward ops: {attention_fuser._forward_ops}")
        print(f"backward ops: {attention_fuser._backward_ops}")

        # Test forward pass
        print("Testing fused forward pass...")
        # try:
        with nvtx_range("Attention Fuser"):
            if self.tensor_parallel_size > 1:
                output, output_bias, output_residual, allreduce_output = attention_fuser(
                    hidden_states=hidden_states,
                    bias=bias,
                    residual=residual,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_mask=attention_mask,
                    allreduce_input=allreduce_inputs,
                    allreduce_overlap_window=(0, 1),
                    allreduce_sm_configs=(4, 1024),
                    allreduce_overlap_window_backward=(0, 1),
                    allreduce_sm_configs_backward=(4, 1024),
                )
            else:
                output, output_bias, output_residual, _ = attention_fuser(
                    hidden_states=hidden_states,
                    bias=bias,
                    residual=residual,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_mask=attention_mask,
                )
        
        print(f"✓ Fused forward pass successful")
        print(f"  Output shape: {output.shape}")
        # print(f"  Output bias shape: {output_bias.shape}")
        print(f"  Output residual shape: {output_residual.shape}")
        
        # Test backward pass
        print("Testing fused backward pass...")
        # loss = output.float().sum() + output_residual.float().sum()

        with nvtx_range("Fuser loss.backward"):
            # loss.backward()
            output_grad = torch.randn_like(output)
            residual_grad = torch.randn_like(output_residual)
            allreduce_input_grad = torch.randn_like(allreduce_output)
            torch.autograd.backward(
                tensors=[output, output_residual, allreduce_output],
                grad_tensors=[output_grad, residual_grad, allreduce_input_grad],
                retain_graph=True,
            )
        
        print("✓ Fused backward pass successful")
        print(f"  Hidden states grad: {hidden_states.grad is not None}")
        # print(f"  Bias grad: {bias.grad is not None}")
        print(f"  Residual grad: {residual.grad is not None}")

        if self.tensor_parallel_size > 1:
            print(f"  Allreduce output: {allreduce_output.grad is not None}")
        
        return True
            
        # except Exception as e:
        #     print(f"✗ Fused attention failed: {str(e)}")
        #     import traceback
        #     traceback.print_exc()
        #     return False

    def run_all_tests(self, rank: int, world_size: int):
        """Run all tests and return overall success status."""
        print("=" * 60)
        print("ATTENTION FUSER OPERATION TESTS")
        print("=" * 60)
        
        test_results = []
        
        # try:
        test_results.append(self.test_individual_operations(rank, world_size))
        # except Exception as e:
        #     print(f"Individual operations test failed: {e}")
        #     test_results.append(False)
        
        # try:
        test_results.append(self.test_attention_fuser(rank, world_size))
        # except Exception as e:
        #     print(f"Attention fuser test failed: {e}")
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
    success = test_runner.run_all_tests(rank, world_size)

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
