#!/usr/bin/env python3

"""Compare results from custom and original flash_attn implementations."""
import torch
import numpy as np
import os

def load_and_compare_results():
    """Load saved results and compare them."""
    
    print("🔍 Comparing Flash Attention Implementation Results")
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
        custom_val = custom_config[key]
        original_val = original_config[key]
        match = "✅" if custom_val == original_val else "❌"
        print(f"   {key}: {custom_val} vs {original_val} {match}")
    
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
        
        # Compare successful outputs
        custom_output = custom_result['output']
        original_output = original_result['output']
        
        # Basic shape check
        if custom_output.shape != original_output.shape:
            print(f"      ❌ Shape mismatch: {custom_output.shape} vs {original_output.shape}")
            all_configs_match = False
            comparison_summary[config_name] = "shape_mismatch"
            continue
        
        # Statistical comparison
        max_diff = torch.max(torch.abs(custom_output - original_output)).item()
        mean_diff = torch.mean(torch.abs(custom_output - original_output)).item()
        rel_diff = max_diff / (torch.mean(torch.abs(original_output)).item() + 1e-8)
        
        # Correlation coefficient
        custom_flat = custom_output.flatten()
        original_flat = original_output.flatten()
        correlation = torch.corrcoef(torch.stack([custom_flat, original_flat]))[0, 1].item()
        
        print(f"      📈 Statistics:")
        print(f"         Max absolute diff: {max_diff:.2e}")
        print(f"         Mean absolute diff: {mean_diff:.2e}")
        print(f"         Relative diff: {rel_diff:.2e} ({rel_diff*100:.3f}%)")
        print(f"         Correlation: {correlation:.6f}")
        
        # Custom vs Original statistics
        custom_stats = f"mean={custom_result['mean']:.4f}, std={custom_result['std']:.4f}"
        original_stats = f"mean={original_result['mean']:.4f}, std={original_result['std']:.4f}"
        print(f"         Custom stats: {custom_stats}")
        print(f"         Original stats: {original_stats}")
        
        # Determine if results are acceptable
        # Allow for small numerical differences due to potential backend differences
        rtol, atol = 1e-4, 1e-5
        is_close = torch.allclose(custom_output, original_output, rtol=rtol, atol=atol)
        
        if is_close:
            print(f"      ✅ PASS: Outputs are numerically equivalent")
            comparison_summary[config_name] = "pass"
        elif rel_diff < 0.01:  # 1% relative difference
            print(f"      ⚠️  ACCEPTABLE: Small difference ({rel_diff*100:.3f}%)")
            comparison_summary[config_name] = "acceptable"
        elif correlation > 0.99:
            print(f"      ⚠️  ACCEPTABLE: High correlation ({correlation:.4f})")
            comparison_summary[config_name] = "acceptable"
        else:
            print(f"      ❌ FAIL: Significant difference")
            all_configs_match = False
            comparison_summary[config_name] = "fail"
    
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
        print("🎉 SUCCESS: Custom implementation matches original!")
        return True
    else:
        print("⚠️  WARNING: Some differences detected between implementations")
        return False

if __name__ == "__main__":
    success = load_and_compare_results()
    exit(0 if success else 1) 