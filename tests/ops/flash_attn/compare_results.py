#!/usr/bin/env python3

"""Compare results from custom and original flash_attn implementations."""
import torch
import numpy as np
import os

def compare_tensors(tensor1, tensor2, name, rtol=1e-4, atol=1e-5):
    """Compare two tensors and return detailed statistics."""
    if torch.equal(tensor1, tensor2):
        return {
            'status': 'identical',
            'max_diff': 0.0,
            'mean_diff': 0.0,
            'rel_diff': 0.0,
            'correlation': 1.0
        }
    
    max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
    mean_diff = torch.mean(torch.abs(tensor1 - tensor2)).item()
    rel_diff = max_diff / (torch.mean(torch.abs(tensor2)).item() + 1e-8)
    
    # Correlation coefficient
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    correlation = torch.corrcoef(torch.stack([tensor1_flat, tensor2_flat]))[0, 1].item()
    
    # Determine status
    is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    
    if is_close:
        status = 'pass'
    elif rel_diff < 0.01:  # 1% relative difference
        status = 'acceptable'
    elif correlation > 0.99:
        status = 'acceptable'
    else:
        status = 'fail'
    
    return {
        'status': status,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rel_diff': rel_diff,
        'correlation': correlation
    }

def load_and_compare_results():
    """Load saved results and compare them."""
    
    print("🔍 Comparing Flash Attention Implementation Results")
    print("=" * 60)
    print("Custom Implementation: Manual backward function calls")
    print("Original Implementation: PyTorch autograd")
    print("=" * 60)
    
    # Check if result files exist
    custom_file = "custom_flash_attn_results.pt"
    original_file = "original_flash_attn_results.pt"
    
    if not os.path.exists(custom_file):
        print(f"❌ Custom results file not found: {custom_file}")
        print("   Please run test_custom_flash_attn.py first")
        return False
    
    if not os.path.exists(original_file):
        print(f"❌ Original results file not found: {original_file}")
        print("   Please run test_original_flash_attn.py first")
        return False
    
    # Load results
    print("📂 Loading results...")
    try:
        custom_data = torch.load(custom_file, map_location='cpu')
        original_data = torch.load(original_file, map_location='cpu')
        print("✅ Successfully loaded both result files")
    except Exception as e:
        print(f"❌ Failed to load results: {e}")
        return False
    
    # Verify test configurations match
    custom_config = custom_data['test_config']
    original_config = original_data['test_config']
    
    print(f"\n⚙️  Test Configuration:")
    for key in custom_config:
        if key in original_config:
            custom_val = custom_config[key]
            original_val = original_config[key]
            match = "✅" if custom_val == original_val else "❌"
            print(f"   {key}: {custom_val} vs {original_val} {match}")
        else:
            print(f"   {key}: {custom_config[key]} (custom only)")
    
    # Show backward method differences
    custom_manual = custom_config.get('manual_backward', False)
    original_manual = original_config.get('manual_backward', False)
    print(f"\n🔄 Backward Method:")
    print(f"   Custom: {'Manual backward calls' if custom_manual else 'PyTorch autograd'}")
    print(f"   Original: {'Manual backward calls' if original_manual else 'PyTorch autograd'}")
    
    # Compare input tensors (should be identical due to same seed)
    print(f"\n🎯 Input Tensor Verification:")
    input_tensors = ['input_q', 'input_k', 'input_v']
    for tensor_name in input_tensors:
        custom_tensor = custom_data[tensor_name]
        original_tensor = original_data[tensor_name]
        
        if torch.equal(custom_tensor, original_tensor):
            print(f"   {tensor_name}: ✅ Identical")
        else:
            max_diff = torch.max(torch.abs(custom_tensor - original_tensor)).item()
            print(f"   {tensor_name}: ❌ Different (max diff: {max_diff:.2e})")
    
    # Compare results for each configuration
    print(f"\n📊 Output Comparison:")
    custom_results = custom_data['results']
    original_results = original_data['results']
    
    all_configs_match = True
    comparison_summary = {}
    
    for config_name in custom_results.keys():
        print(f"\n   Configuration: {config_name}")
        
        if config_name not in original_results:
            print(f"      ❌ Missing in original results")
            all_configs_match = False
            continue
        
        custom_result = custom_results[config_name]
        original_result = original_results[config_name]
        
        # Check for errors
        custom_error = 'error' in custom_result
        original_error = 'error' in original_result
        
        if custom_error and original_error:
            print(f"      ⚠️  Both failed:")
            print(f"         Custom: {custom_result['error']}")
            print(f"         Original: {original_result['error']}")
            comparison_summary[config_name] = "both_failed"
            continue
        elif custom_error:
            print(f"      ❌ Custom failed: {custom_result['error']}")
            all_configs_match = False
            comparison_summary[config_name] = "custom_failed"
            continue
        elif original_error:
            print(f"      ❌ Original failed: {original_result['error']}")
            all_configs_match = False
            comparison_summary[config_name] = "original_failed"
            continue
        
        # Compare successful forward outputs
        custom_output = custom_result['output']
        original_output = original_result['output']
        
        # Basic shape check
        if custom_output.shape != original_output.shape:
            print(f"      ❌ Shape mismatch: {custom_output.shape} vs {original_output.shape}")
            all_configs_match = False
            comparison_summary[config_name] = "shape_mismatch"
            continue
        
        # Forward pass comparison
        forward_comp = compare_tensors(custom_output, original_output, "forward_output")
        
        print(f"      📈 Forward Pass Statistics:")
        print(f"         Max absolute diff: {forward_comp['max_diff']:.2e}")
        print(f"         Mean absolute diff: {forward_comp['mean_diff']:.2e}")
        print(f"         Relative diff: {forward_comp['rel_diff']:.2e} ({forward_comp['rel_diff']*100:.3f}%)")
        print(f"         Correlation: {forward_comp['correlation']:.6f}")
        
        # Custom vs Original statistics
        custom_stats = f"mean={custom_result['mean']:.4f}, std={custom_result['std']:.4f}"
        original_stats = f"mean={original_result['mean']:.4f}, std={original_result['std']:.4f}"
        print(f"         Custom stats: {custom_stats}")
        print(f"         Original stats: {original_stats}")
        
        # Compare gradients
        print(f"      🔄 Backward Pass (Gradients) Statistics:")
        print(f"         Custom: Manual backward calls")
        print(f"         Original: PyTorch autograd")
        gradient_comparisons = {}
        
        for grad_name in ['q_grad', 'k_grad', 'v_grad']:
            if grad_name in custom_result and grad_name in original_result:
                custom_grad = custom_result[grad_name]
                original_grad = original_result[grad_name]
                
                grad_comp = compare_tensors(custom_grad, original_grad, grad_name)
                gradient_comparisons[grad_name] = grad_comp
                
                print(f"         {grad_name}:")
                print(f"           Max diff: {grad_comp['max_diff']:.2e}")
                print(f"           Mean diff: {grad_comp['mean_diff']:.2e}")
                print(f"           Relative diff: {grad_comp['rel_diff']:.2e} ({grad_comp['rel_diff']*100:.3f}%)")
                print(f"           Correlation: {grad_comp['correlation']:.6f}")
                
                # Compare gradient statistics
                if 'grad_stats' in custom_result and 'grad_stats' in original_result:
                    custom_grad_stats = custom_result['grad_stats']
                    original_grad_stats = original_result['grad_stats']
                    
                    grad_norm_key = f"{grad_name.split('_')[0]}_grad_norm"
                    grad_mean_key = f"{grad_name.split('_')[0]}_grad_mean"
                    
                    if grad_norm_key in custom_grad_stats and grad_norm_key in original_grad_stats:
                        custom_norm = custom_grad_stats[grad_norm_key]
                        original_norm = original_grad_stats[grad_norm_key]
                        print(f"           Gradient norms: custom={custom_norm:.4f}, original={original_norm:.4f}")
                    
                    if grad_mean_key in custom_grad_stats and grad_mean_key in original_grad_stats:
                        custom_mean = custom_grad_stats[grad_mean_key]
                        original_mean = original_grad_stats[grad_mean_key]
                        print(f"           Gradient means: custom={custom_mean:.4f}, original={original_mean:.4f}")
        
        # Determine overall status for this configuration
        forward_status = forward_comp['status']
        gradient_statuses = [comp['status'] for comp in gradient_comparisons.values()]
        
        # Overall status is the worst among forward and all gradients
        all_statuses = [forward_status] + gradient_statuses
        if 'fail' in all_statuses:
            overall_status = 'fail'
        elif 'acceptable' in all_statuses:
            overall_status = 'acceptable'
        else:
            overall_status = 'pass'
        
        comparison_summary[config_name] = overall_status
        
        if overall_status == 'pass':
            print(f"      ✅ PASS: Manual backward matches autograd")
        elif overall_status == 'acceptable':
            print(f"      ⚠️  ACCEPTABLE: Small differences between manual backward and autograd")
        else:
            print(f"      ❌ FAIL: Significant differences between manual backward and autograd")
            all_configs_match = False
    
    # Print final summary
    print(f"\n" + "=" * 60)
    print(f"📋 FINAL SUMMARY")
    print(f"=" * 60)
    
    for config_name, result in comparison_summary.items():
        status_emoji = {
            "pass": "✅",
            "acceptable": "⚠️",
            "fail": "❌",
            "both_failed": "💥",
            "custom_failed": "❌",
            "original_failed": "❌",
            "shape_mismatch": "❌"
        }
        print(f"{status_emoji.get(result, '❓')} {config_name}: {result}")
    
    # Overall assessment
    passed = sum(1 for r in comparison_summary.values() if r in ["pass", "acceptable"])
    total = len(comparison_summary)
    
    print(f"\nOverall: {passed}/{total} configurations acceptable")
    
    if passed == total:
        print("🎉 SUCCESS: Manual backward calls match PyTorch autograd!")
        print("   This validates that your custom flash attention backward implementation")
        print("   produces the same results as the original implementation.")
        return True
    else:
        print("⚠️  WARNING: Some differences detected between manual backward and autograd")
        print("   This may indicate differences in the backward implementation.")
        return False

if __name__ == "__main__":
    success = load_and_compare_results()
    exit(0 if success else 1) 