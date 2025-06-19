#!/usr/bin/env python3

"""Test original flash_attn implementation and save output for comparison."""
import torch
import traceback

def test_original_flash_attn():
    """Test original flash_attn_func and save output."""
    
    print("🧪 Testing Original Flash Attention Implementation")
    print("=" * 60)
    
    # Test parameters (must match the custom test)
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    print(f"Device: {device}")
    print(f"Config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}")
    print(f"Data type: {dtype}")
    
    # Create test tensors with SAME FIXED SEED for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) if torch.cuda.is_available() else None
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    
    # Ensure tensors are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    print(f"Input tensor shapes: {q.shape}")
    print(f"Input tensor device: {q.device}")
    print(f"Input tensor dtype: {q.dtype}")
    print(f"Autograd backward testing enabled")
    
    try:
        # Import original flash_attn_func
        print("\n📦 Importing original flash_attn_func...")
        from flash_attn.flash_attn_interface import flash_attn_func
        print("✅ Successfully imported original flash_attn_func")
        
        # Test same configurations as custom test
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
                # Clear gradients
                if q.grad is not None:
                    q.grad.zero_()
                if k.grad is not None:
                    k.grad.zero_()
                if v.grad is not None:
                    v.grad.zero_()
                
                # Forward pass
                output = flash_attn_func(q, k, v, **config)
                
                # Validate forward output
                assert output.shape == q.shape, f"Shape mismatch: {output.shape} vs {q.shape}"
                assert not torch.isnan(output).any(), "Output contains NaN"
                assert not torch.isinf(output).any(), "Output contains Inf"
                
                # Backward pass using autograd
                print("   🔄 Testing backward pass...")
                # Create gradient output (same as manual test for consistency)
                grad_output = torch.ones_like(output)
                output.backward(grad_output)
                
                # Validate gradients
                assert q.grad is not None, "q.grad is None"
                assert k.grad is not None, "k.grad is None"
                assert v.grad is not None, "v.grad is None"
                
                assert not torch.isnan(q.grad).any(), "q.grad contains NaN"
                assert not torch.isnan(k.grad).any(), "k.grad contains NaN" 
                assert not torch.isnan(v.grad).any(), "v.grad contains NaN"
                
                assert not torch.isinf(q.grad).any(), "q.grad contains Inf"
                assert not torch.isinf(k.grad).any(), "k.grad contains Inf"
                assert not torch.isinf(v.grad).any(), "v.grad contains Inf"
                
                # Store results
                results[config_name] = {
                    'output': output.detach().cpu(),
                    'q_grad': q.grad.detach().cpu(),
                    'k_grad': k.grad.detach().cpu(), 
                    'v_grad': v.grad.detach().cpu(),
                    'config': config.copy(),
                    'shape': output.shape,
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'grad_stats': {
                        'q_grad_norm': q.grad.norm().item(),
                        'k_grad_norm': k.grad.norm().item(),
                        'v_grad_norm': v.grad.norm().item(),
                        'q_grad_mean': q.grad.mean().item(),
                        'k_grad_mean': k.grad.mean().item(),
                        'v_grad_mean': v.grad.mean().item(),
                    }
                }
                
                print(f"   ✅ Forward Success! Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"   📊 Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
                print(f"   ✅ Backward Success! Gradient norms: q={q.grad.norm().item():.4f}, k={k.grad.norm().item():.4f}, v={v.grad.norm().item():.4f}")
                
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
                'requires_grad': True,
                'manual_backward': False
            }
        }
        
        output_file = "original_flash_attn_results.pt"
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
        print(f"❌ Failed to import original flash_attn_func: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_original_flash_attn()
    exit(0 if success else 1) 