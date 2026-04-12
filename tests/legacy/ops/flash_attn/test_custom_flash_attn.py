#!/usr/bin/env python3

"""Test custom flash_attn implementation and save output for comparison."""
import sys
sys.path.append("/workspaces/Kareus")

import torch
import traceback
import dataclasses
from typing import Optional

@dataclasses.dataclass
class OperationContext:
    """State needed to apply an operation

    Saves state from forward pass for use in backward pass.

    """

    # Tensors that have been saved from forward function
    # Note: Available in the backward function, matching tensors from
    # to_save.
    saved_tensors: Optional[tuple[Optional[torch.Tensor], ...]] = None
    # Tensors to save for backward function
    # Note: Expected to be set in the forward function, either
    # directly or with save_for_backward.
    to_save: Optional[tuple[Optional[torch.Tensor], ...]] = None

    # Corresponding range in pipeline's list of saved tensors
    _saved_tensors_range: Optional[tuple[int, int]] = None

    # Whether backward pass is required
    requires_grad: bool = True

    def save_for_backward(self, *tensors: Optional[torch.Tensor]) -> None:
        """Register tensors to be saved for the backward function

        Expected to be called in the forward function.

        """
        self.to_save = tensors

class MockBackwardContext:
    """Mock context object that mimics the structure expected by FlashAttnFunc.backward"""
    def __init__(self, saved_tensors, dropout_p, softmax_scale, causal, window_size, 
                 softcap, alibi_slopes, deterministic):
        self.saved_tensors = saved_tensors
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.softcap = softcap
        self.alibi_slopes = alibi_slopes
        self.deterministic = deterministic

def test_custom_flash_attn():
    """Test custom flash_attn_func and save output."""
    
    print("🧪 Testing Custom Flash Attention Implementation")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    print(f"Device: {device}")
    print(f"Config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}")
    print(f"Data type: {dtype}")
    
    # Create test tensors with fixed seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) if torch.cuda.is_available() else None
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=False)
    
    # Ensure tensors are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    ctx = OperationContext()
    
    print(f"Input tensor shapes: {q.shape}")
    print(f"Input tensor device: {q.device}")
    print(f"Input tensor dtype: {q.dtype}")
    print(f"Manual backward testing enabled")
    
    try:
        # Import custom flash_attn_func and backward function
        print("\n📦 Importing custom flash_attn_func...")
        from kareus.flash_attn.flash_attn_interface import flash_attn_func, flash_attn_func_backward
        print("✅ Successfully imported custom flash_attn_func and flash_attn_func_backward")
        
        # Test different configurations
        test_configs = [
            {"causal": False, "name": "non_causal"},
            {"causal": True, "name": "causal"},
            {"causal": True, "dropout_p": 0.0, "name": "causal_no_dropout"},
            {"causal": False, "softmax_scale": 0.125, "name": "custom_scale"},
        ]
        
        results = {}
        
        for config in test_configs:
            config_name = config.pop("name")
            print(f"\n🔍 Testing configuration: {config_name}")
            print(f"   Parameters: {config}")
            
            try:
                # Forward pass - need to create a mock context to capture saved tensors
                class ForwardContext:
                    def __init__(self):
                        self.saved_tensors = None
                        self.dropout_p = None
                        self.softmax_scale = None
                        self.causal = None
                        self.window_size = None
                        self.softcap = None
                        self.alibi_slopes = None
                        self.deterministic = None
                    
                    def save_for_backward(self, *tensors):
                        self.saved_tensors = tensors
                
                # Call forward pass using the FlashAttnFunc class directly to capture context
                from kareus.flash_attn.flash_attn_interface import FlashAttnFunc
                
                forward_ctx = ForwardContext()
                
                # Set default parameters
                dropout_p = config.get('dropout_p', 0.0)
                softmax_scale = config.get('softmax_scale', None)
                causal = config.get('causal', False)
                window_size = config.get('window_size', (-1, -1))
                softcap = config.get('softcap', 0.0)
                alibi_slopes = config.get('alibi_slopes', None)
                deterministic = config.get('deterministic', False)
                return_softmax = config.get('return_softmax', False)
                
                output = FlashAttnFunc.forward(
                    forward_ctx,
                    q, k, v,
                    dropout_p,
                    softmax_scale,
                    causal,
                    window_size,
                    softcap,
                    alibi_slopes,
                    deterministic,
                    return_softmax
                )
                
                # Validate forward output
                assert output.shape == q.shape, f"Shape mismatch: {output.shape} vs {q.shape}"
                assert not torch.isnan(output).any(), "Output contains NaN"
                assert not torch.isinf(output).any(), "Output contains Inf"
                
                # Backward pass - manually call backward function
                print("   🔄 Testing backward pass...")
                
                # Create gradient output (simulating loss.backward())
                grad_output = torch.ones_like(output)
                
                # Create mock backward context with saved tensors and parameters
                backward_ctx = MockBackwardContext(
                    saved_tensors=forward_ctx.saved_tensors,
                    dropout_p=forward_ctx.dropout_p,
                    softmax_scale=forward_ctx.softmax_scale,
                    causal=forward_ctx.causal,
                    window_size=forward_ctx.window_size,
                    softcap=forward_ctx.softcap,
                    alibi_slopes=forward_ctx.alibi_slopes,
                    deterministic=forward_ctx.deterministic
                )
                
                # Call backward function manually
                dq, dk, dv = FlashAttnFunc.backward(backward_ctx, grad_output)[:3]
                
                # Validate gradients
                assert dq is not None, "dq is None"
                assert dk is not None, "dk is None"
                assert dv is not None, "dv is None"
                
                assert dq.shape == q.shape, f"dq shape mismatch: {dq.shape} vs {q.shape}"
                assert dk.shape == k.shape, f"dk shape mismatch: {dk.shape} vs {k.shape}"
                assert dv.shape == v.shape, f"dv shape mismatch: {dv.shape} vs {v.shape}"
                
                assert not torch.isnan(dq).any(), "dq contains NaN"
                assert not torch.isnan(dk).any(), "dk contains NaN" 
                assert not torch.isnan(dv).any(), "dv contains NaN"
                
                assert not torch.isinf(dq).any(), "dq contains Inf"
                assert not torch.isinf(dk).any(), "dk contains Inf"
                assert not torch.isinf(dv).any(), "dv contains Inf"
                
                # Store results
                results[config_name] = {
                    'output': output.detach().cpu(),
                    'q_grad': dq.detach().cpu(),
                    'k_grad': dk.detach().cpu(), 
                    'v_grad': dv.detach().cpu(),
                    'config': config.copy(),
                    'shape': output.shape,
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'grad_stats': {
                        'q_grad_norm': dq.norm().item(),
                        'k_grad_norm': dk.norm().item(),
                        'v_grad_norm': dv.norm().item(),
                        'q_grad_mean': dq.mean().item(),
                        'k_grad_mean': dk.mean().item(),
                        'v_grad_mean': dv.mean().item(),
                    }
                }
                
                print(f"   ✅ Forward Success! Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"   📊 Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
                print(f"   ✅ Backward Success! Gradient norms: q={dq.norm().item():.4f}, k={dk.norm().item():.4f}, v={dv.norm().item():.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                traceback.print_exc()
                results[config_name] = {'error': str(e)}
        
        # Save input tensors and results
        save_data = {
            'input_q': q.detach().cpu(),
            'input_k': k.detach().cpu(), 
            'input_v': v.detach().cpu(),
            'results': results,
            'test_config': {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'num_heads': num_heads,
                'head_dim': head_dim,
                'dtype': str(dtype),
                'device': str(device),
                'requires_grad': False,
                'manual_backward': True
            }
        }
        
        output_file = "custom_flash_attn_results.pt"
        torch.save(save_data, output_file)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print summary
        print(f"\n📋 Test Summary:")
        print(f"   Total configurations tested: {len(test_configs)}")
        successful = sum(1 for r in results.values() if 'error' not in r)
        print(f"   Successful: {successful}/{len(test_configs)}")
        
        if successful == len(test_configs):
            print("   🎉 All tests PASSED!")
            return True
        else:
            print("   ⚠️  Some tests FAILED!")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import custom flash_attn_func: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_custom_flash_attn()
    exit(0 if success else 1) 