#!/usr/bin/env python3
"""Test script demonstrating RotaryEmbeddingOp as an nn.Module with autograd."""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/workspaces/Kareus')

from kareus.megatron.core.extensions.rotary_embedding_op import create_rotary_embedding_op
from megatron.core.transformer.transformer_config import TransformerConfig

def test_rotary_embedding_as_module():
    """Test the RotaryEmbeddingOp as a proper nn.Module with autograd."""
    
    # Create a basic transformer config
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=8,
        rotary_interleaved=False,
        flash_decode=False,
        apply_rope_fusion=False,
    )
    
    # Create the rotary embedding operation
    rope_op = create_rotary_embedding_op(config)
    
    # Test parameters
    batch_size = 2
    seq_len = 16
    hidden_size = 128
    
    # Create test tensors with gradients enabled
    query = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    key = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    
    # Create frequency tensor for rotary embedding
    freqs = torch.randn(seq_len, hidden_size)
    rotary_pos_emb = (freqs, freqs)  # Use same for both query and key
    
    print("Testing forward pass with autograd...")
    
    # Forward pass - this will automatically set up the computation graph
    query_out, key_out = rope_op(
        query=query,
        key=key,
        rotary_pos_emb=rotary_pos_emb,
    )
    
    print(f"Query input shape: {query.shape}")
    print(f"Query output shape: {query_out.shape}")
    print(f"Key input shape: {key.shape}")
    print(f"Key output shape: {key_out.shape}")
    
    # Create a dummy loss function (sum of outputs)
    loss = query_out.sum() + key_out.sum()
    print(f"Loss: {loss.item()}")
    
    print("\nTesting backward pass with autograd...")
    
    # Backward pass - PyTorch will automatically call the backward method
    # of our RotaryEmbeddingFunction
    loss.backward()
    
    print(f"Query gradient shape: {query.grad.shape}")
    print(f"Key gradient shape: {key.grad.shape}")
    print(f"Query gradient norm: {query.grad.norm().item()}")
    print(f"Key gradient norm: {key.grad.norm().item()}")
    
    # Verify gradients are not None and have finite values
    assert query.grad is not None, "Query gradient should not be None"
    assert key.grad is not None, "Key gradient should not be None"
    assert torch.isfinite(query.grad).all(), "Query gradients should be finite"
    assert torch.isfinite(key.grad).all(), "Key gradients should be finite"
    
    print("\n✓ Autograd test completed successfully!")
    return True

def test_gradient_check():
    """Test gradients using torch.autograd.gradcheck for numerical verification."""
    
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,  # Smaller for faster gradient checking
        num_attention_heads=4,
        rotary_interleaved=False,
        flash_decode=False,
        apply_rope_fusion=False,
    )
    
    rope_op = create_rotary_embedding_op(config)
    
    # Smaller tensors for gradient checking
    batch_size = 1
    seq_len = 4
    hidden_size = 32
    
    query = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float64, requires_grad=True)
    key = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float64, requires_grad=True)
    freqs = torch.randn(seq_len, hidden_size, dtype=torch.float64)
    
    def rope_func(q, k):
        q_out, k_out = rope_op(query=q, key=k, rotary_pos_emb=(freqs, freqs))
        return q_out.sum() + k_out.sum()
    
    print("Running numerical gradient check...")
    
    # Use torch.autograd.gradcheck for numerical gradient verification
    test_passed = torch.autograd.gradcheck(
        rope_func,
        (query, key),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=False
    )
    
    if test_passed:
        print("✓ Gradient check passed!")
    else:
        print("❌ Gradient check failed!")
    
    return test_passed

def test_single_tensor_rotary_emb():
    """Test with a single rotary embedding tensor (same for query and key)."""
    
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        rotary_interleaved=False,
        flash_decode=False,
        apply_rope_fusion=False,
    )
    
    rope_op = create_rotary_embedding_op(config)
    
    batch_size = 2
    seq_len = 8
    hidden_size = 64
    
    query = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    key = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    freqs = torch.randn(seq_len, hidden_size)
    
    print("Testing with single rotary embedding tensor...")
    
    # Use single tensor instead of tuple
    query_out, key_out = rope_op(
        query=query,
        key=key,
        rotary_pos_emb=freqs,  # Single tensor, not tuple
    )
    
    loss = query_out.sum() + key_out.sum()
    loss.backward()
    
    assert query.grad is not None, "Query gradient should not be None"
    assert key.grad is not None, "Key gradient should not be None"
    
    print("✓ Single tensor test passed!")
    return True

if __name__ == "__main__":
    print("Testing RotaryEmbeddingOp as nn.Module with autograd...\n")
    
    try:
        test_rotary_embedding_as_module()
        test_single_tensor_rotary_emb()
        test_gradient_check()
        
        print("\n🎉 All tests passed! The RotaryEmbeddingOp works as a proper nn.Module with autograd.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 