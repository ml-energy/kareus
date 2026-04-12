#!/usr/bin/env python3
"""
Test script to compare BiasDropoutAddOp with get_bias_dropout_add from Megatron.
This script tests both forward and backward passes for numerical accuracy.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append("/workspaces/Kareus")

# Try to import NVTX for profiling

from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_test_tensors(batch_size=4, seq_len=8192, hidden_size=32768, dtype=torch.float16, device='cuda'):
    """Create test tensors for comparison."""
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(hidden_size, dtype=dtype, device=device, requires_grad=True)
    residual = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device, requires_grad=True)
    return x, bias, residual


def test_forward_pass(x, bias, residual, dropout_prob=0.1, training=True, tolerance=1e-5):
    """Test forward pass comparison."""
    print(f"\n=== Forward Pass Test (dropout_prob={dropout_prob}, training={training}) ===")
    
    # Test with BiasDropoutAddOp
    set_random_seed(42)
    op_module = BiasDropoutAddOp(dropout_prob=dropout_prob, training=training)
    
    # Clone inputs for BiasDropoutAddOp
    x1 = x.clone().detach().requires_grad_(True)
    bias1 = bias.clone().detach().requires_grad_(True)
    residual1 = residual.clone().detach().requires_grad_(True)
    
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("op_forward_pass")
    output1 = op_module(x1, bias1, residual1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    # Test with get_bias_dropout_add
    set_random_seed(42)
    megatron_func = get_bias_dropout_add(training=training, fused=True)
    
    # Clone inputs for Megatron function
    x2 = x.clone().detach().requires_grad_(True)
    bias2 = bias.clone().detach().requires_grad_(True)
    residual2 = residual.clone().detach().requires_grad_(True)
    
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("megatron_forward_pass")
    output2 = megatron_func((x2, bias2), residual2, dropout_prob)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    # Compare outputs
    result = True
    if training and dropout_prob > 0:
        print("Note: Due to different random seeds in dropout, exact comparison may not be possible.")
        print("Comparing statistical properties instead...")
        
        # Compare means and standard deviations
        mean_diff = abs(output1.mean().item() - output2.mean().item())
        std_diff = abs(output1.std().item() - output2.std().item())
        
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Std difference: {std_diff:.6f}")
        
        # Check if they're statistically similar (loose tolerance for stochastic operations)
        if mean_diff < 0.1 and std_diff < 0.1:
            print("✓ Forward pass: Statistically similar outputs")
            result = True
        else:
            print("✗ Forward pass: Outputs differ significantly")
            result = False
    else:
        # For inference or no dropout, we can do exact comparison
        max_diff = torch.max(torch.abs(output1 - output2)).item()
        print(f"Max absolute difference: {max_diff:.8f}")
        
        if max_diff < tolerance:
            print("✓ Forward pass: Outputs match within tolerance")
            result = True
        else:
            print("✗ Forward pass: Outputs differ beyond tolerance")
            print(f"Expected output shape: {output2.shape}")
            print(f"Actual output shape: {output1.shape}")
            result = False
    return result


def test_deterministic_forward(x, bias, residual, tolerance=1e-6):
    """Test deterministic forward pass (no dropout)."""
    print(f"\n=== Deterministic Forward Pass Test ===")
    
    # Test without dropout
    set_random_seed(42)
    
    # BiasDropoutAddOp

    op_module = BiasDropoutAddOp(dropout_prob=0.0)
    op_module.eval()
    
    x1 = x.clone().detach()
    bias1 = bias.clone().detach()
    residual1 = residual.clone().detach()
    
    output1 = op_module(x1, bias1, residual1)
    
    # Manual calculation
    expected = residual + (x + bias)
    
    # Compare with manual calculation
    max_diff = torch.max(torch.abs(output1 - expected)).item()
    print(f"Max difference from manual calculation: {max_diff:.8f}")
    
    result = True
    if max_diff < tolerance:
        print("✓ Deterministic forward pass: Matches manual calculation")
        result = True
    else:
        print("✗ Deterministic forward pass: Differs from manual calculation")
        result = False
    return result


def test_backward_pass(x, bias, residual, dropout_prob=0.0, training=False, tolerance=1e-5):
    """Test backward pass comparison."""
    print(f"\n=== Backward Pass Test (dropout_prob={dropout_prob}, training={training}) ===")
    
    # Test BiasDropoutAddOp backward
    set_random_seed(42)
    
    x1 = x.clone().detach().requires_grad_(True)
    bias1 = bias.clone().detach().requires_grad_(True)
    residual1 = residual.clone().detach().requires_grad_(True)
    
    # Forward pass with our module
    op_module = BiasDropoutAddOp(dropout_prob=dropout_prob)
    if training:
        op_module.train()
    else:
        op_module.eval()
    
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("op_forward_for_backward")
    output1 = op_module(x1, bias1, residual1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    # Backward pass
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("op_backward_pass")
    grad_output = torch.randn_like(output1)
    output1.backward(grad_output, retain_graph=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    grad_x1 = x1.grad.clone() if x1.grad is not None else torch.zeros_like(x1)
    grad_bias1 = bias1.grad.clone() if bias1.grad is not None else torch.zeros_like(bias1)
    grad_residual1 = residual1.grad.clone() if residual1.grad is not None else torch.zeros_like(residual1)

    # Test with Megatron function
    set_random_seed(42)
    
    x2 = x.clone().detach().requires_grad_(True)
    bias2 = bias.clone().detach().requires_grad_(True)
    residual2 = residual.clone().detach().requires_grad_(True)
    
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("megatron_forward_for_backward")
    megatron_func = get_bias_dropout_add(training=training, fused=True)
    output2 = megatron_func((x2, bias2), residual2, dropout_prob)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("megatron_backward_pass")
    output2.backward(grad_output, retain_graph=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    grad_x2 = x2.grad.clone() if x2.grad is not None else torch.zeros_like(x2)
    grad_bias2 = bias2.grad.clone() if bias2.grad is not None else torch.zeros_like(bias2)
    grad_residual2 = residual2.grad.clone() if residual2.grad is not None else torch.zeros_like(residual2)
    
    # Compare gradients
    result = True
    if training and dropout_prob > 0:
        print("Note: Gradient comparison with dropout may vary due to different masks")
        
        # Compare gradient means for stochastic case
        x_grad_mean_diff = abs(grad_x1.mean().item() - grad_x2.mean().item())
        bias_grad_mean_diff = abs(grad_bias1.mean().item() - grad_bias2.mean().item())
        residual_grad_mean_diff = abs(grad_residual1.mean().item() - grad_residual2.mean().item())
        
        print(f"X gradient mean difference: {x_grad_mean_diff:.6f}")
        print(f"Bias gradient mean difference: {bias_grad_mean_diff:.6f}")
        print(f"Residual gradient mean difference: {residual_grad_mean_diff:.6f}")
        
        if x_grad_mean_diff < 0.1 and bias_grad_mean_diff < 0.1 and residual_grad_mean_diff < 0.1:
            print("✓ Backward pass: Gradients are statistically similar")
            result = True
        else:
            print("✗ Backward pass: Gradients differ significantly")
            result = False
    else:
        # Exact comparison for deterministic case
        x_grad_diff = torch.max(torch.abs(grad_x1 - grad_x2)).item()
        bias_grad_diff = torch.max(torch.abs(grad_bias1 - grad_bias2)).item()
        residual_grad_diff = torch.max(torch.abs(grad_residual1 - grad_residual2)).item()
        
        print(f"X gradient max difference: {x_grad_diff:.8f}")
        print(f"Bias gradient max difference: {bias_grad_diff:.8f}")
        print(f"Residual gradient max difference: {residual_grad_diff:.8f}")
        
        if x_grad_diff < tolerance and bias_grad_diff < tolerance and residual_grad_diff < tolerance:
            print("✓ Backward pass: All gradients match within tolerance")
            result = True
        else:
            print("✗ Backward pass: Some gradients differ beyond tolerance")
            result = False
    return result


def run_comprehensive_tests():
    """Run comprehensive comparison tests."""
    print("=" * 60)
    print("BiasDropoutAddOp vs get_bias_dropout_add Comparison Tests")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Using device: {device}")
    
    # Test configurations
    test_configs = [
        {'batch_size': 2, 'seq_len': 8, 'hidden_size': 16},
        {'batch_size': 4, 'seq_len': 32, 'hidden_size': 64},
        {'batch_size': 1, 'seq_len': 128, 'hidden_size': 256},
    ]
    
    dropout_probs = [0.0, 0.1, 0.5]
    training_modes = [True, False]
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        print(f"\n" + "="*50)
        print(f"Testing with config: {config}")
        print("="*50)
        
        # Create test tensors
        x, bias, residual = create_test_tensors(device=device, **config)
        
        # Test deterministic case first
        if not test_deterministic_forward(x, bias, residual):
            all_passed = False
        
        # Test various dropout probabilities and training modes
        for j, dropout_prob in enumerate(dropout_probs):
            for k, training in enumerate(training_modes):
                # Skip dropout=0 combinations we already tested
                if dropout_prob == 0.0 and training == False:
                    continue
                    
                if not test_forward_pass(x, bias, residual, dropout_prob, training):
                    all_passed = False
                
                # Only test backward for deterministic cases
                if dropout_prob == 0.0:
                    if not test_backward_pass(x, bias, residual, dropout_prob, training):
                        all_passed = False
    
    print(f"\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! BiasDropoutAddOp is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    run_comprehensive_tests()