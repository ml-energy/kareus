#!/usr/bin/env python3
"""
Test script for the MLP fuser with the required operations:
- BDA (BiasDropoutAddOp) - prev_self_attn_bda
- LayerNorm/RMSNorm - pre_mlp_layernorm
- Linear_fc1 (Linear transformation for intermediate)
- BiasSwigluOp (activation function with bias)
- Linear_fc2 (Linear transformation back to hidden)
- BDA (BiasDropoutAddOp) - post_mlp_bda (only for last layer)
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
from transformer_engine.pytorch.ops.basic.layer_norm import LayerNorm
from transformer_engine.pytorch.ops.basic.rmsnorm import RMSNorm
from kareus.transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.megatron.core.extensions.bias_swiglu_op import BiasSwigluOp
from kareus.megatron.core.extensions.te_linear import TEFusibleColumnParallelLinear, TEFusibleRowParallelLinear, TEFusibleLinear
from kareus.transformer_engine.pytorch.ops.linear import Linear

# Import MLP fuser
from kareus.megatron.core.extensions.partition_fuser import PartitionFuser

# Import configuration
from megatron.core.transformer.transformer_config import TransformerConfig
from cfuser.core.utils import nvtx_range


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


class MLPFuserTest:
    """Test suite for MLP fuser operations."""

    def __init__(self, device='cuda', tensor_parallel_size: int = 1):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16
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
        self.ffn_hidden_size = 8192  # 4x expansion typical for MLPs
        
        # Create transformer config
        self.config = TransformerConfig(
            num_layers=2,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_query_groups=self.num_query_groups,
            ffn_hidden_size=self.ffn_hidden_size,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.1,
            gated_linear_unit=True,  # Use SwiGLU
            activation_func=F.silu,
            bias_activation_fusion=True,
            add_bias_linear=True,
            params_dtype=self.dtype,
        )

    def create_test_tensors(self):
        """Create test tensors for the MLP operations."""
        # Input tensors
        hidden_states = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        # Residual connection
        residual = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        # Bias and residual for BDA
        bias = torch.randn(
            self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )

        # AllReduce inputs for tensor parallelism
        allreduce_inputs = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        return hidden_states, bias, residual, allreduce_inputs

    def create_operations(self, rank: int, world_size: int, is_last_layer: bool = False):
        """Create all the required operations for the MLP fuser."""
        
        operations = []
        
        # 1. BDA Operation (Bias Dropout Add) - prev_self_attn_bda
        prev_attn_bda_op = BiasDropoutAddOp(
            dropout_prob=self.config.hidden_dropout,
            training=True
        )
        operations.append(prev_attn_bda_op)
        
        # 2. LayerNorm Operation - pre_mlp_layernorm
        layernorm_op = LayerNorm(
            normalized_shape=self.hidden_size,
            eps=self.config.layernorm_epsilon,
            device=self.device,
            dtype=self.dtype
        )
        operations.append(layernorm_op)
        
        # 3. Linear FC1 Operation (input to intermediate with gating)
        # Since gated_linear_unit=True, output is 2 * ffn_hidden_size
        linear_fc1_op = Linear(
            in_features=self.hidden_size,
            out_features=2 * self.ffn_hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=True,
            return_bias=True,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )
        operations.append(linear_fc1_op)
        
        # 4. BiasSwigluOp (activation function with bias)
        bias_swiglu_op = BiasSwigluOp(
            fp8_input_store=self.config.activation_func_fp8_input_store
        )
        operations.append(bias_swiglu_op)
        
        # 5. Linear FC2 Operation (intermediate back to hidden)
        linear_fc2_op = Linear(
            in_features=self.ffn_hidden_size,
            out_features=self.hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=True,
            return_bias=True,  # Return bias for post_mlp_bda
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )
        operations.append(linear_fc2_op)
        
        # 6. Post MLP BDA Operation (only for last layer)
        if is_last_layer:
            post_mlp_bda_op = BiasDropoutAddOp(
                dropout_prob=self.config.hidden_dropout,
                training=True
            )
            operations.append(post_mlp_bda_op)
        
        # 7. AllReduce Communication Operation
        if self.tensor_parallel_size > 1:
            allreduce_comm_op = AllReduce(
                process_group=self.tp_group,
                async_op=True,
                backend="msccl",
                rank=rank,
                world_size=world_size,
            )
            return operations, allreduce_comm_op
        
        return operations, None

    def test_individual_operations(self, rank: int, world_size: int, is_last_layer: bool = False):
        """Test each operation individually to ensure they work correctly."""
        print(f"\n=== Testing Individual MLP Operations (is_last_layer={is_last_layer}) ===")
        
        hidden_states, bias, residual, allreduce_inputs = self.create_test_tensors()
        operations, allreduce_comm_op = self.create_operations(rank, world_size, is_last_layer)
        
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
        check_tensor_health(residual, "residual")
        check_tensor_health(bias, "bias")

        print(f"\nhidden_states.shape: {hidden_states.shape}")
        print(f"residual.shape: {residual.shape}")
        print(f"bias.shape: {bias.shape}")
        print(f"allreduce_inputs.shape: {allreduce_inputs.shape}")
        
        # Test prev_self_attn_bda
        print("\nTesting prev_self_attn BiasDropoutAddOp...")
        prev_bda_op = operations[0]
        with nvtx_range("Prev Attn BDA"):
            bda_output = prev_bda_op(hidden_states, bias, residual)
        print(f"✓ Prev Attn BDA output shape: {bda_output.shape}")
        if not check_tensor_health(bda_output, "prev_bda_output"):
            print("❌ Prev Attn BDA operation produced invalid values!")
            return False
        
        # Test LayerNorm
        print("\nTesting LayerNorm...")
        layernorm_op = operations[1]
        with nvtx_range("LayerNorm"):
            ln_output = layernorm_op(bda_output)
        print(f"✓ LayerNorm output shape: {ln_output.shape}")
        if not check_tensor_health(ln_output, "ln_output"):
            print("❌ LayerNorm operation produced invalid values!")
            return False
        
        # Test Linear FC1
        print("\nTesting Linear FC1...")
        linear_fc1_op = operations[2]
        with nvtx_range("Linear FC1"):
            fc1_output, fc1_bias = linear_fc1_op(ln_output)
        print(f"✓ Linear FC1 output shape: {fc1_output.shape}")
        if not check_tensor_health(fc1_output, "fc1_output"):
            print("❌ Linear FC1 operation produced invalid values!")
            return False
        
        # Test BiasSwigluOp
        print("\nTesting BiasSwigluOp...")
        bias_swiglu_op = operations[3]
        with nvtx_range("BiasSwigluOp"):
            swiglu_output = bias_swiglu_op(fc1_output, fc1_bias)
        print(f"✓ BiasSwigluOp output shape: {swiglu_output.shape}")
        if not check_tensor_health(swiglu_output, "swiglu_output"):
            print("❌ BiasSwigluOp operation produced invalid values!")
            return False
        
        # Test Linear FC2
        print("\nTesting Linear FC2...")
        linear_fc2_op = operations[4]
        with nvtx_range("Linear FC2"):
            fc2_output, fc2_output_bias = linear_fc2_op(swiglu_output)
        print(f"✓ Linear FC2 output shape: {fc2_output.shape}")
        print(f"✓ Linear FC2 bias shape: {fc2_output_bias.shape}")
        if not (check_tensor_health(fc2_output, "fc2_output") and check_tensor_health(fc2_output_bias, "fc2_output_bias")):
            print("❌ Linear FC2 operation produced invalid values!")
            return False
        
        # Test post_mlp_bda (only for last layer)
        if is_last_layer:
            print("\nTesting post_mlp BiasDropoutAddOp...")
            post_mlp_bda_op = operations[5]
            with nvtx_range("Post MLP BDA"):
                final_output = post_mlp_bda_op(fc2_output, fc2_output_bias, bda_output)  # Use bda_output as residual
            print(f"✓ Post MLP BDA output shape: {final_output.shape}")
            if not check_tensor_health(final_output, "final_output"):
                print("❌ Post MLP BDA operation produced invalid values!")
                return False
            mlp_final_output = final_output
        else:
            mlp_final_output = fc2_output

        # Test AllReduce
        if self.tensor_parallel_size > 1:
            print("\nTesting AllReduce...")
            with nvtx_range("AllReduce"):
                allreduce_output = allreduce_comm_op(allreduce_inputs, sm_num=4, block_size=1024)
                allreduce_comm_op.sync()
            print(f"✓ AllReduce output shape: {allreduce_output.shape}")
        
        # Test backward pass for individual operations
        print("\n--- Testing Individual Operations Backward Pass ---")
        
        # Create a loss from the final output
        loss = mlp_final_output.float().sum()
        if is_last_layer and len(operations) > 5:
            loss = loss + fc2_output_bias.float().sum()
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
        residual.grad = None
        bias.grad = None
        
        # Test backward pass
        print("Testing individual operations backward...")
        with nvtx_range("Individual loss.backward"):
            loss.backward()
        print("✓ Individual operations backward pass successful")
        
        # Check gradients
        print("Checking individual operation gradients...")
        print(f"  Hidden states grad: {hidden_states.grad is not None}")
        print(f"  Residual grad: {residual.grad is not None}")
        print(f"  Bias grad: {bias.grad is not None}")
        
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
        
        return True

    def test_mlp_fuser(self, rank: int, world_size: int, is_last_layer: bool = False):
        """Test the complete MLP fuser with all operations."""
        print(f"\n=== Testing MLP Fuser (is_last_layer={is_last_layer}) ===")
        
        hidden_states, bias, residual, allreduce_inputs = self.create_test_tensors()
        operations, allreduce_comm_op = self.create_operations(rank, world_size, is_last_layer)
        
        # Create MLP fuser
        mlp_fuser = PartitionFuser(
            ops=operations,
            allreduce_comm_op=allreduce_comm_op,
            fuse_ops=True,
        )
        
        print("MLP fuser created successfully")
        print(f"Number of basic operations: {len(mlp_fuser._basic_ops)}")
        print(f"Forward ops: {mlp_fuser._forward_ops}")
        print(f"Backward ops: {mlp_fuser._backward_ops}")

        # Test forward pass
        print("Testing fused forward pass...")
        with nvtx_range("MLP Fuser"):
            if self.tensor_parallel_size > 1:
                output, output_fc2_bias, out_residual, allreduce_output = mlp_fuser(
                    hidden_states=hidden_states,
                    bias=bias,
                    residual=residual,
                    allreduce_input=allreduce_inputs,
                    allreduce_overlap_window=(0, 1),
                    allreduce_sm_configs=(4, 1024),
                    allreduce_overlap_window_backward=(0, 1),
                    allreduce_sm_configs_backward=(4, 1024),
                )
        
        print(f"✓ Fused forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output FC2 bias shape: {output_fc2_bias.shape}")
        print(f"  Residual shape: {out_residual.shape}")
        
        # Test backward pass
        print("Testing fused backward pass...")
        loss = output.float().sum() + output_fc2_bias.float().sum() + out_residual.float().sum()

        with nvtx_range("Fuser loss.backward"):
            loss.backward()
        
        print("✓ Fused backward pass successful")
        print(f"  Hidden states grad: {hidden_states.grad is not None}")
        print(f"  Residual grad: {residual.grad is not None}")
        print(f"  Bias grad: {bias.grad is not None}")

        if self.tensor_parallel_size > 1:
            print(f"  Allreduce output grad: {allreduce_inputs.grad is not None}")
        
        return True

    def run_all_tests(self, rank: int, world_size: int):
        """Run all tests and return overall success status."""
        print("=" * 60)
        print("MLP FUSER OPERATION TESTS")
        print("=" * 60)
        
        test_results = []
        
        # Test for non-last layer
        test_results.append(self.test_individual_operations(rank, world_size, is_last_layer=False))
        test_results.append(self.test_mlp_fuser(rank, world_size, is_last_layer=False))
        
        # # Test for last layer
        # test_results.append(self.test_individual_operations(rank, world_size, is_last_layer=True))
        # test_results.append(self.test_mlp_fuser(rank, world_size, is_last_layer=True))
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        test_names = [
            "Individual Operations (Non-last Layer)",
            "MLP Fuser (Non-last Layer)",
            # "Individual Operations (Last Layer)",
            # "MLP Fuser (Last Layer)",
        ]
        
        for name, result in zip(test_names, test_results):
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{name}: {status}")
        
        overall_success = all(test_results)
        print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
        
        return overall_success


def run_process(rank, world_size, args, master_port):
    """Run the MLP fuser tests in a distributed environment."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    # Create test instance and run tests
    test_runner = MLPFuserTest(
        device=args.device, 
        tensor_parallel_size=args.tensor_parallel_size
    )
    success = test_runner.run_all_tests(rank, world_size)

    # Clean up distributed if initialized
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed process group destroyed")


def main():
    """Main function to run the MLP fuser tests."""
    parser = argparse.ArgumentParser(description='MLP Fuser Test with Distributed Support')
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

    if args.tensor_parallel_size > 1:
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
    else:
        # Single process test
        test_runner = MLPFuserTest(
            device=args.device, 
            tensor_parallel_size=args.tensor_parallel_size
        )
        success = test_runner.run_all_tests(0, 1)
        return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 