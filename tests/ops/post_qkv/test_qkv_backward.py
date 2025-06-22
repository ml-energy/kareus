#!/usr/bin/env python3
"""
Test script to verify QKV post-processing backward pass against PyTorch autograd.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import sys
import os

sys.path.append("/workspaces/Kareus")

from kareus.megatron.core.extensions.qkv_postprocess_op import QKVPostProcessOp


def qkv_postprocess_reference(
    mixed_qkv: torch.Tensor,
    num_query_groups_per_partition: int,
    num_attention_heads_per_partition: int,
    hidden_size_per_attention_head: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of QKV post-processing using standard PyTorch operations."""
    
    # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
    new_tensor_shape = mixed_qkv.size()[:-1] + (
        num_query_groups_per_partition,
        (
            (num_attention_heads_per_partition // num_query_groups_per_partition + 2)
            * hidden_size_per_attention_head
        ),
    )
    mixed_qkv_reshaped = mixed_qkv.view(*new_tensor_shape)

    split_arg_list = [
        (
            num_attention_heads_per_partition
            // num_query_groups_per_partition
            * hidden_size_per_attention_head
        ),
        hidden_size_per_attention_head,
        hidden_size_per_attention_head,
    ]

    # [sq, b, ng, (np/ng + 2) * hn]
    # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
    (query, key, value) = torch.split(mixed_qkv_reshaped, split_arg_list, dim=3)

    # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
    query = query.reshape(query.size(0), query.size(1), -1, hidden_size_per_attention_head)

    return query, key, value


def test_qkv_backward():
    """Test the QKV post-processing backward pass against autograd."""
    
    print("Testing QKV Post-Processing Backward Pass")
    print("=" * 50)
    
    # Test parameters
    seq_len = 32
    batch_size = 4
    num_query_groups_per_partition = 8
    num_attention_heads_per_partition = 32
    hidden_size_per_attention_head = 64
    
    # Calculate total hidden size for mixed QKV
    hidden_size_total = num_query_groups_per_partition * (
        (num_attention_heads_per_partition // num_query_groups_per_partition + 2)
        * hidden_size_per_attention_head
    )
    
    print(f"Test Configuration:")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Query Groups per Partition: {num_query_groups_per_partition}")
    print(f"  Attention Heads per Partition: {num_attention_heads_per_partition}")
    print(f"  Hidden Size per Head: {hidden_size_per_attention_head}")
    print(f"  Total Hidden Size: {hidden_size_total}")
    print()
    
    # Create test input
    mixed_qkv = torch.randn(
        seq_len, batch_size, hidden_size_total, 
        dtype=torch.float32, requires_grad=True
    )
    
    print(f"Input shape: {mixed_qkv.shape}")
    
    # Test 1: Compare forward pass outputs
    print("\nTest 1: Forward Pass Comparison")
    print("-" * 30)
    
    # Reference forward pass
    query_ref, key_ref, value_ref = qkv_postprocess_reference(
        mixed_qkv, 
        num_query_groups_per_partition,
        num_attention_heads_per_partition, 
        hidden_size_per_attention_head
    )
    
    # Our implementation (without using the full operation, just the logic)
    op = QKVPostProcessOp(
        num_query_groups_per_partition=num_query_groups_per_partition,
        num_attention_heads_per_partition=num_attention_heads_per_partition,
        hidden_size_per_attention_head=hidden_size_per_attention_head,
    )
    
    # Mock context for forward pass
    class MockContext:
        def __init__(self):
            self.saved_tensors = None
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors
    
    ctx = MockContext()
    query_ours, key_ours, value_ours = op.op_forward(ctx, mixed_qkv)
    
    # Compare forward outputs
    query_diff = torch.max(torch.abs(query_ref - query_ours)).item()
    key_diff = torch.max(torch.abs(key_ref - key_ours)).item()
    value_diff = torch.max(torch.abs(value_ref - value_ours)).item()
    
    print(f"Query max difference: {query_diff:.2e}")
    print(f"Key max difference: {key_diff:.2e}")
    print(f"Value max difference: {value_diff:.2e}")
    
    forward_pass_ok = query_diff < 1e-6 and key_diff < 1e-6 and value_diff < 1e-6
    print(f"Forward pass test: {'PASS' if forward_pass_ok else 'FAIL'}")
    
    # Test 2: Compare backward pass
    print("\nTest 2: Backward Pass Comparison")
    print("-" * 30)
    
    # Create output gradients
    grad_query = torch.randn_like(query_ref)
    grad_key = torch.randn_like(key_ref)
    grad_value = torch.randn_like(value_ref)
    
    print(f"Grad query shape: {grad_query.shape}")
    print(f"Grad key shape: {grad_key.shape}")
    print(f"Grad value shape: {grad_value.shape}")
    
    # Method 1: Use autograd to compute reference gradients
    mixed_qkv_autograd = mixed_qkv.clone().detach().requires_grad_(True)
    query_auto, key_auto, value_auto = qkv_postprocess_reference(
        mixed_qkv_autograd,
        num_query_groups_per_partition,
        num_attention_heads_per_partition,
        hidden_size_per_attention_head
    )
    
    # Compute gradients using autograd
    loss = (query_auto * grad_query).sum() + (key_auto * grad_key).sum() + (value_auto * grad_value).sum()
    loss.backward()
    grad_mixed_qkv_autograd = mixed_qkv_autograd.grad
    
    # Method 2: Use our manual backward implementation
    grad_mixed_qkv_ours = op.op_backward(ctx, grad_query, grad_key, grad_value)
    
    # Compare backward outputs
    backward_diff = torch.max(torch.abs(grad_mixed_qkv_autograd - grad_mixed_qkv_ours)).item()
    backward_rel_diff = (backward_diff / torch.max(torch.abs(grad_mixed_qkv_autograd)).item())
    
    print(f"Backward max absolute difference: {backward_diff:.2e}")
    print(f"Backward max relative difference: {backward_rel_diff:.2e}")
    print(f"Autograd gradient shape: {grad_mixed_qkv_autograd.shape}")
    print(f"Our gradient shape: {grad_mixed_qkv_ours.shape}")
    
    backward_pass_ok = backward_diff < 1e-5
    print(f"Backward pass test: {'PASS' if backward_pass_ok else 'FAIL'}")
    
    # Test 3: Gradient check with finite differences
    print("\nTest 3: Finite Difference Gradient Check")
    print("-" * 40)
    
    eps = 1e-5
    grad_mixed_qkv_fd = torch.zeros_like(mixed_qkv)
    
    # Check a few random elements with finite differences
    num_checks = 10
    indices_to_check = [
        (torch.randint(0, seq_len, (1,)).item(),
         torch.randint(0, batch_size, (1,)).item(),
         torch.randint(0, hidden_size_total, (1,)).item())
        for _ in range(num_checks)
    ]
    
    max_fd_diff = 0.0
    for i, j, k in indices_to_check:
        # Forward pass with +eps
        mixed_qkv_plus = mixed_qkv.clone().detach()
        mixed_qkv_plus[i, j, k] += eps
        query_plus, key_plus, value_plus = qkv_postprocess_reference(
            mixed_qkv_plus,
            num_query_groups_per_partition,
            num_attention_heads_per_partition,
            hidden_size_per_attention_head
        )
        loss_plus = (query_plus * grad_query).sum() + (key_plus * grad_key).sum() + (value_plus * grad_value).sum()
        
        # Forward pass with -eps
        mixed_qkv_minus = mixed_qkv.clone().detach()
        mixed_qkv_minus[i, j, k] -= eps
        query_minus, key_minus, value_minus = qkv_postprocess_reference(
            mixed_qkv_minus,
            num_query_groups_per_partition,
            num_attention_heads_per_partition,
            hidden_size_per_attention_head
        )
        loss_minus = (query_minus * grad_query).sum() + (key_minus * grad_key).sum() + (value_minus * grad_value).sum()
        
        # Finite difference gradient
        fd_grad = (loss_plus - loss_minus) / (2 * eps)
        
        # Compare with our gradient
        our_grad = grad_mixed_qkv_ours[i, j, k].item()
        autograd_grad = grad_mixed_qkv_autograd[i, j, k].item()
        
        fd_diff_ours = abs(fd_grad.item() - our_grad)
        fd_diff_autograd = abs(fd_grad.item() - autograd_grad)
        
        max_fd_diff = max(max_fd_diff, fd_diff_ours)
        
        if i == indices_to_check[0][0] and j == indices_to_check[0][1] and k == indices_to_check[0][2]:
            print(f"Sample gradient check at ({i}, {j}, {k}):")
            print(f"  Finite difference: {fd_grad.item():.6f}")
            print(f"  Autograd:          {autograd_grad:.6f}")
            print(f"  Our implementation: {our_grad:.6f}")
            print(f"  FD vs Ours diff:   {fd_diff_ours:.2e}")
            print(f"  FD vs Autograd diff: {fd_diff_autograd:.2e}")
    
    finite_diff_ok = max_fd_diff < 1e-4
    print(f"Max finite difference error: {max_fd_diff:.2e}")
    print(f"Finite difference test: {'PASS' if finite_diff_ok else 'FAIL'}")
    
    # Summary
    print("\nTest Summary")
    print("=" * 20)
    all_tests_pass = forward_pass_ok and backward_pass_ok and finite_diff_ok
    print(f"Forward pass:       {'PASS' if forward_pass_ok else 'FAIL'}")
    print(f"Backward pass:      {'PASS' if backward_pass_ok else 'FAIL'}")
    print(f"Finite difference:  {'PASS' if finite_diff_ok else 'FAIL'}")
    print(f"Overall result:     {'PASS' if all_tests_pass else 'FAIL'}")
    
    return all_tests_pass


if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducible results
    success = test_qkv_backward()
    sys.exit(0 if success else 1) 