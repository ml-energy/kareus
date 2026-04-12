import torch
import pytest
import numpy as np
from torch.testing import assert_close
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from kareus.megatron.core.models.common.embedding.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    _apply_rotary_pos_emb_bshd_backward,
)

# Import the RotaryEmbeddingOp to test
from kareus.megatron.core.extensions.rotary_embedding_op import create_rotary_embedding_op
from megatron.core.transformer.transformer_config import TransformerConfig


class TestRoPEBackward:
    """Test suite for RoPE backward function implementation using RotaryEmbeddingOp as nn.Module."""
    
    @pytest.fixture
    def device(self):
        """Test on CUDA if available, otherwise CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def test_configs(self):
        """Different test configurations for comprehensive testing."""
        return [
            {
                "batch_size": 2,
                "seq_length": 8,
                "hidden_size": 64,
                "rotary_interleaved": False,
                "multi_latent_attention": False,
                "mscale": 1.0,
            },
            # {
            #     "batch_size": 1,
            #     "seq_length": 16,
            #     "hidden_size": 128,
            #     "rotary_interleaved": True,
            #     "multi_latent_attention": False,
            #     "mscale": 1.0,
            # },
            # {
            #     "batch_size": 4,
            #     "seq_length": 4,
            #     "hidden_size": 32,
            #     "rotary_interleaved": False,
            #     "multi_latent_attention": True,
            #     "mscale": 1.5,
            # },
        ]
    
    def create_test_tensors(self, config, device):
        """Create input tensors for testing."""
        batch_size = config["batch_size"]
        seq_length = config["seq_length"]
        hidden_size = config["hidden_size"]
        
        # Query and Key tensors
        query = torch.randn(batch_size, seq_length, hidden_size, device=device, dtype=torch.float32)
        key = torch.randn(batch_size, seq_length, hidden_size, device=device, dtype=torch.float32)
        query.requires_grad_(True)
        key.requires_grad_(True)
        
        # Frequency tensor (should match the rotary dimension)
        freqs = torch.randn(seq_length, hidden_size, device=device, dtype=torch.float32)
        
        return query, key, freqs
    
    def create_rope_op(self, config):
        """Create a RotaryEmbeddingOp with the given config."""
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=config["hidden_size"],
            num_attention_heads=8,  # Doesn't matter for this test
            rotary_interleaved=config["rotary_interleaved"],
            flash_decode=False,
            apply_rope_fusion=False,
        )
        return create_rotary_embedding_op(transformer_config)
    
    def reference_rope_forward(self, t, freqs, rotary_interleaved=False, 
                             multi_latent_attention=False, mscale=1.0):
        """Reference implementation using the original forward function."""
        from transformer_engine.pytorch.ops.op import OperationContext
        ctx = OperationContext()
        return _apply_rotary_pos_emb_bshd(
            ctx, t, freqs, rotary_interleaved, multi_latent_attention, mscale
        )
    
    def test_rope_op_autograd(self, test_configs, device):
        """Test that the RotaryEmbeddingOp works correctly with PyTorch autograd."""
        for config in test_configs:
            with torch.no_grad():
                torch.manual_seed(42)  # For reproducibility
            
            query, key, freqs = self.create_test_tensors(config, device)
            rope_op = self.create_rope_op(config)
            
            # Move to device if needed
            rope_op = rope_op.to(device)
            
            # Forward pass using RotaryEmbeddingOp as nn.Module
            # PyTorch will automatically handle the computation graph
            query_out, key_out = rope_op(
                query,
                key,
                freqs,
            )
            
            # Create a loss function (sum of outputs)
            loss = query_out.sum() + key_out.sum()
            
            # Backward pass - PyTorch autograd handles everything automatically
            loss.backward()
            
            # Check that gradients were computed
            assert query.grad is not None, f"Query gradient should not be None for config: {config}"
            assert key.grad is not None, f"Key gradient should not be None for config: {config}"
            assert torch.isfinite(query.grad).all(), f"Query gradients should be finite for config: {config}"
            assert torch.isfinite(key.grad).all(), f"Key gradients should be finite for config: {config}"
            
            # Check output shapes
            assert query_out.shape == query.shape, f"Query output shape mismatch for config: {config}"
            assert key_out.shape == key.shape, f"Key output shape mismatch for config: {config}"
    
    def test_rope_op_vs_reference(self, test_configs, device):
        """Test that RotaryEmbeddingOp produces results similar to reference implementation."""
        for config in test_configs:
            with torch.no_grad():
                torch.manual_seed(42)
            
            query, key, freqs = self.create_test_tensors(config, device)
            rope_op = self.create_rope_op(config)
            rope_op = rope_op.to(device)
            
            # Test with RotaryEmbeddingOp
            query_clone = query.clone().detach().requires_grad_(True)
            key_clone = key.clone().detach().requires_grad_(True)
            
            query_out_op, key_out_op = rope_op(
                query_clone,
                key_clone,
                freqs,
            )
            
            loss_op = query_out_op.sum() + key_out_op.sum()
            loss_op.backward()
            
            # Test with reference implementation (applied separately to query and key)
            query_ref = self.reference_rope_forward(
                query,
                freqs,
                config["rotary_interleaved"],
                config["multi_latent_attention"],
                config["mscale"]
            )
            key_ref = self.reference_rope_forward(
                key,
                freqs,
                config["rotary_interleaved"],
                config["multi_latent_attention"],
                config["mscale"]
            )
            
            loss_ref = query_ref.sum() + key_ref.sum()
            loss_ref.backward()
            
            # Compare outputs (with some tolerance for implementation differences)
            assert_close(
                query_out_op.detach(),
                query_ref.detach(),
                rtol=1e-4,
                atol=1e-5,
                msg=f"Query output mismatch for config: {config}"
            )
            
            assert_close(
                key_out_op.detach(),
                key_ref.detach(),
                rtol=1e-4,
                atol=1e-5,
                msg=f"Key output mismatch for config: {config}"
            )
            
            # Compare gradients
            assert_close(
                query_clone.grad,
                query.grad,
                rtol=1e-4,
                atol=1e-5,
                msg=f"Query gradient mismatch for config: {config}"
            )
            
            assert_close(
                key_clone.grad,
                key.grad,
                rtol=1e-4,
                atol=1e-5,
                msg=f"Key gradient mismatch for config: {config}"
            )
    
    def test_rope_op_gradient_check(self, device):
        """Test gradients using torch.autograd.gradcheck for numerical verification."""
        torch.manual_seed(42)
        
        # Smaller test for gradient checking
        config = {
            "batch_size": 1,
            "seq_length": 4,
            "hidden_size": 8,
            "rotary_interleaved": False,
            "multi_latent_attention": False,
            "mscale": 1.0,
        }
        
        query, key, freqs = self.create_test_tensors(config, device)
        rope_op = self.create_rope_op(config)
        rope_op = rope_op.to(device)
        
        # Use double precision for better numerical accuracy
        query = query.double()
        key = key.double()
        freqs = freqs.double()
        
        def rope_func(q, k):
            q_out, k_out = rope_op(query=q, key=k, rotary_pos_emb=freqs)
            return q_out.sum() + k_out.sum()
        
        # Use torch.autograd.gradcheck for numerical gradient verification
        torch.autograd.gradcheck(
            rope_func,
            (query, key),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True
        )
    
    def test_rope_op_single_tensor_embedding(self, device):
        """Test with single rotary embedding tensor (same for query and key)."""
        config = {
            "batch_size": 2,
            "seq_length": 8,
            "hidden_size": 32,
            "rotary_interleaved": False,
            "multi_latent_attention": False,
            "mscale": 1.0,
        }
        
        query, key, freqs = self.create_test_tensors(config, device)
        rope_op = self.create_rope_op(config)
        rope_op = rope_op.to(device)
        
        # Test with single tensor instead of tuple
        query_out, key_out = rope_op(
            query,
            key,
            freqs  # Single tensor, not tuple
        )
        
        loss = query_out.sum() + key_out.sum()
        loss.backward()
        
        assert query.grad is not None, "Query gradient should not be None"
        assert key.grad is not None, "Key gradient should not be None"
        assert torch.isfinite(query.grad).all(), "Query gradients should be finite"
        assert torch.isfinite(key.grad).all(), "Key gradients should be finite"
    
    def test_rope_op_edge_cases(self, device):
        """Test edge cases and boundary conditions."""
        config = {
            "batch_size": 1,
            "seq_length": 4,
            "hidden_size": 8,
            "rotary_interleaved": False,
            "multi_latent_attention": False,
            "mscale": 1.0,
        }
        
        rope_op = self.create_rope_op(config)
        rope_op = rope_op.to(device)
        
        # Test with zero input
        query_zero = torch.zeros(1, 4, 8, device=device, requires_grad=True)
        key_zero = torch.zeros(1, 4, 8, device=device, requires_grad=True)
        freqs = torch.randn(4, 8, device=device)
        
        query_out, key_out = rope_op(
            query_zero,
            key_zero,
            (freqs, freqs)
        )
        
        loss = query_out.sum() + key_out.sum()
        loss.backward()
        
        # Should not raise any errors
        assert query_zero.grad is not None
        assert key_zero.grad is not None
        
        # Test with very small values
        query_small = torch.full((1, 4, 8), 1e-8, device=device, requires_grad=True)
        key_small = torch.full((1, 4, 8), 1e-8, device=device, requires_grad=True)
        
        query_out_small, key_out_small = rope_op(
            query=query_small,
            key=key_small,
            rotary_pos_emb=(freqs, freqs)
        )
        
        loss_small = query_out_small.sum() + key_out_small.sum()
        loss_small.backward()
        
        assert query_small.grad is not None
        assert key_small.grad is not None
        assert torch.isfinite(query_small.grad).all()
        assert torch.isfinite(key_small.grad).all()
    
    def test_rope_op_different_dtypes(self, device):
        """Test with different floating point precisions."""
        config = {
            "batch_size": 2,
            "seq_length": 4,
            "hidden_size": 8,
            "rotary_interleaved": False,
            "multi_latent_attention": False,
            "mscale": 1.0,
        }
        
        rope_op = self.create_rope_op(config)
        rope_op = rope_op.to(device)
        
        for dtype in [torch.float32, torch.float64]:
            query = torch.randn(
                config["batch_size"], 
                config["seq_length"], 
                config["hidden_size"],
                device=device, 
                dtype=dtype,
                requires_grad=True
            )
            key = torch.randn(
                config["batch_size"], 
                config["seq_length"], 
                config["hidden_size"],
                device=device, 
                dtype=dtype,
                requires_grad=True
            )
            freqs = torch.randn(
                config["seq_length"], 
                config["hidden_size"],
                device=device, 
                dtype=dtype
            )
            
            query_out, key_out = rope_op(
                query,
                key,
                (freqs, freqs)
            )
            
            loss = query_out.sum() + key_out.sum()
            loss.backward()
            
            assert query.grad is not None
            assert key.grad is not None
            assert query.grad.dtype == dtype
            assert key.grad.dtype == dtype
            assert torch.isfinite(query.grad).all()
            assert torch.isfinite(key.grad).all()


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestRoPEBackward()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get test configs
    configs = [
        {
            "batch_size": 2,
            "seq_length": 8,
            "hidden_size": 64,
            "rotary_interleaved": False,
            "multi_latent_attention": False,
            "mscale": 1.0,
        },
    ]
    
    print(f"Running RotaryEmbeddingOp autograd tests on device: {device}")
    
    try:
        test_instance.test_rope_op_autograd(configs, device)
        print("✓ RotaryEmbeddingOp autograd test passed")
        
        test_instance.test_rope_op_vs_reference(configs, device)
        print("✓ RotaryEmbeddingOp vs reference implementation test passed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 