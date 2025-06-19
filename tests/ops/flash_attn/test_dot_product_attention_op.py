#!/usr/bin/env python3

"""Test DotProductAttentionOp implementation and save results."""
import sys
sys.path.append("/workspaces/Kareus")

import torch
import traceback
import os

def test_dot_product_attention_op():
    """Test DotProductAttentionOp and save results."""
    
    print("🧪 Testing DotProductAttentionOp (BasicOperation Pattern)")
    print("=" * 70)
    
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
    
    # Create input tensors
    q = torch.randn(seq_len, batch_size, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(seq_len, batch_size, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(seq_len, batch_size, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    
    # Ensure tensors are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    print(f"Input tensor shapes: {q.shape}")
    print(f"Input tensor device: {q.device}")
    print(f"Input tensor dtype: {q.dtype}")
    
    try:
        # Import DotProductAttentionOp
        print("\n📦 Importing DotProductAttentionOp...")
        from kareus.transformer_engine.pytorch.attention.dot_product_attention import DotProductAttentionOp
        from transformer_engine.pytorch.ops.op import OperationContext
        print("✅ Successfully imported DotProductAttentionOp")
        
        # Test configurations (avoiding complex format conversions that require tex)
        test_configs = [
            {"attn_mask_type": "no_mask", "name": "no_mask"},
            {"attn_mask_type": "causal", "name": "causal"},
            {"attention_dropout": 0.0, "attn_mask_type": "causal", "name": "causal_no_dropout"},
            {"attn_mask_type": "no_mask", "softmax_scale": 0.125, "name": "custom_scale"},
        ]
        
        results = {}
        
        for config in test_configs:
            config_name = config.pop("name")
            print(f"\n🔍 Testing configuration: {config_name}")
            print(f"   Parameters: {config}")
            
            try:
                # Create fresh copies of input tensors
                q_test = q.clone().detach().requires_grad_(True)
                k_test = k.clone().detach().requires_grad_(True)
                v_test = v.clone().detach().requires_grad_(True)
                
                # Initialize DotProductAttentionOp
                attention_op = DotProductAttentionOp(
                    num_attention_heads=num_heads,
                    kv_channels=head_dim,
                    qkv_format="sbhd",
                    **config
                )
                attention_op.eval()  # Set to eval mode for consistent behavior
                
                # Create operation context
                ctx = OperationContext()
                
                # Forward pass using autograd-friendly call
                print("   🔄 Running forward pass...")
                output = attention_op(q_test, k_test, v_test)
                
                # Validate forward output
                # assert output.shape == q_test.shape, f"Shape mismatch: {output.shape} vs {q_test.shape}"
                assert not torch.isnan(output).any(), "Output contains NaN"
                assert not torch.isinf(output).any(), "Output contains Inf"
                
                print(f"   ✅ Forward Success! Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"   📊 Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
                
                # Backward pass using autograd
                print("   🔄 Running backward pass...")
                try:
                    loss = output.sum()
                    loss.backward()
                    
                    # Validate gradients
                    assert q_test.grad is not None, "q_test.grad is None"
                    assert k_test.grad is not None, "k_test.grad is None"
                    assert v_test.grad is not None, "v_test.grad is None"
                    
                    assert not torch.isnan(q_test.grad).any(), "q_test.grad contains NaN"
                    assert not torch.isnan(k_test.grad).any(), "k_test.grad contains NaN"
                    assert not torch.isnan(v_test.grad).any(), "v_test.grad contains NaN"
                    
                    assert not torch.isinf(q_test.grad).any(), "q_test.grad contains Inf"
                    assert not torch.isinf(k_test.grad).any(), "k_test.grad contains Inf"
                    assert not torch.isinf(v_test.grad).any(), "v_test.grad contains Inf"
                    
                    print(f"   ✅ Backward Success! Gradient norms: q={q_test.grad.norm().item():.4f}, k={k_test.grad.norm().item():.4f}, v={v_test.grad.norm().item():.4f}")
                    
                    # Save gradient statistics
                    backward_results = {
                        "backward_success": True,
                        "q_grad_norm": q_test.grad.norm().item(),
                        "k_grad_norm": k_test.grad.norm().item(),
                        "v_grad_norm": v_test.grad.norm().item(),
                    }
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    traceback.print_exc()
                    backward_results = {
                        "backward_success": False,
                        "backward_error": str(e)
                    }
                
                # Store results
                config_results = {
                    'forward_success': True,
                    'output_shape': list(output.shape),
                    'output_stats': {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                    },
                    **backward_results
                }
                
                # Store results
                results[config_name] = {
                    'output': output.detach().cpu(),
                    'q_grad': q_test.grad.detach().cpu(),
                    'k_grad': k_test.grad.detach().cpu(),
                    'v_grad': v_test.grad.detach().cpu(),
                    'config': config.copy(),
                    'stats': {
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'q_grad_norm': q_test.grad.norm().item(),
                        'k_grad_norm': k_test.grad.norm().item(),
                        'v_grad_norm': v_test.grad.norm().item(),
                    },
                    **config_results
                }
                
                print(f"   ✅ Configuration {config_name} completed successfully!")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                traceback.print_exc()
                results[config_name] = {'error': str(e)}
        
        # Save results
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
                'requires_grad': True,
                'implementation': 'DotProductAttentionOp'
            }
        }
        
        output_file = "dot_product_attention_op_results.pt"
        torch.save(save_data, output_file)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print summary
        print(f"\n📋 Test Summary:")
        print(f"   Total configurations tested: {len(test_configs)}")
        successful = sum(1 for r in results.values() if 'error' not in r)
        print(f"   Successful: {successful}/{len(test_configs)}")
        
        if successful == len(test_configs):
            print("   🎉 All DotProductAttentionOp tests PASSED!")
            return True
        else:
            print("   ⚠️  Some DotProductAttentionOp tests FAILED!")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import DotProductAttentionOp: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running DotProductAttentionOp test...")
    success = test_dot_product_attention_op()
    exit(0 if success else 1) 