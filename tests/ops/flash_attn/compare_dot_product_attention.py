#!/usr/bin/env python3

"""Compare DotProductAttentionOp vs DotProductAttention results."""
import sys
sys.path.append("/workspaces/Kareus")

import torch
import numpy as np
import os
import time
from typing import Dict, Any, Optional

def load_results(filename: str) -> Optional[Dict[str, Any]]:
    """Load test results from file."""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return None
    
    try:
        results = torch.load(filename, map_location='cpu')
        print(f"✅ Loaded {filename}")
        return results
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        return None

def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, name: str, tolerance: float = 1e-5) -> Dict[str, Any]:
    """Compare two tensors and return comparison metrics."""
    if tensor1.shape != tensor2.shape:
        return {
            'match': False,
            'error': f'Shape mismatch: {tensor1.shape} vs {tensor2.shape}'
        }
    
    # Convert to float32 for comparison if needed
    t1 = tensor1.float()
    t2 = tensor2.float()
    
    # Calculate differences
    diff = t1 - t2
    abs_diff = torch.abs(diff)
    rel_diff = abs_diff / (torch.abs(t1) + 1e-8)
    
    # Check if tensors match within tolerance
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # Check for NaN or Inf
    has_nan = torch.isnan(t1).any() or torch.isnan(t2).any()
    has_inf = torch.isinf(t1).any() or torch.isinf(t2).any()
    
    # Determine if tensors match
    matches = (max_abs_diff < tolerance and 
               max_rel_diff < tolerance and 
               not has_nan and not has_inf)
    
    return {
        'match': matches,
        'max_abs_diff': max_abs_diff,
        'max_rel_diff': max_rel_diff,
        'mean_abs_diff': mean_abs_diff,
        'mean_rel_diff': mean_rel_diff,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'tensor1_stats': {
            'min': t1.min().item(),
            'max': t1.max().item(),
            'mean': t1.mean().item(),
            'std': t1.std().item(),
        },
        'tensor2_stats': {
            'min': t2.min().item(),
            'max': t2.max().item(),
            'mean': t2.mean().item(),
            'std': t2.std().item(),
        }
    }

def compare_configurations(op_results: Dict[str, Any], mod_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare results for each configuration."""
    comparison = {}
    
    # Get common configurations
    op_configs = set(op_results.keys())
    mod_configs = set(mod_results.keys())
    common_configs = op_configs.intersection(mod_configs)
    
    print(f"📊 Comparing {len(common_configs)} common configurations")
    print(f"   Op-only configs: {op_configs - mod_configs}")
    print(f"   Mod-only configs: {mod_configs - op_configs}")
    
    for config_name in common_configs:
        print(f"\n🔍 Comparing configuration: {config_name}")
        
        op_config = op_results[config_name]
        mod_config = mod_results[config_name]
        
        # Check for errors
        if 'error' in op_config:
            print(f"   ❌ Op test failed: {op_config['error']}")
            comparison[config_name] = {'op_error': op_config['error']}
            continue
            
        if 'error' in mod_config:
            print(f"   ❌ Mod test failed: {mod_config['error']}")
            comparison[config_name] = {'mod_error': mod_config['error']}
            continue
        
        # Compare outputs and gradients
        output_comp = compare_tensors(op_config['output'], mod_config['output'], 'output')
        q_grad_comp = compare_tensors(op_config['q_grad'], mod_config['q_grad'], 'q_grad')
        k_grad_comp = compare_tensors(op_config['k_grad'], mod_config['k_grad'], 'k_grad')
        v_grad_comp = compare_tensors(op_config['v_grad'], mod_config['v_grad'], 'v_grad')
        
        # Compare statistics
        stats_comp = {
            'output_match': output_comp['match'],
            'q_grad_match': q_grad_comp['match'],
            'k_grad_match': k_grad_comp['match'],
            'v_grad_match': v_grad_comp['match'],
            'all_match': (output_comp['match'] and q_grad_comp['match'] and 
                         k_grad_comp['match'] and v_grad_comp['match']),
            'output_comparison': output_comp,
            'q_grad_comparison': q_grad_comp,
            'k_grad_comparison': k_grad_comp,
            'v_grad_comparison': v_grad_comp,
        }
        
        # Print results
        if stats_comp['all_match']:
            print(f"   ✅ All tensors match!")
        else:
            print(f"   ⚠️  Some tensors don't match:")
            if not output_comp['match']:
                print(f"      Output: max_abs_diff={output_comp['max_abs_diff']:.2e}")
            if not q_grad_comp['match']:
                print(f"      Q grad: max_abs_diff={q_grad_comp['max_abs_diff']:.2e}")
            if not k_grad_comp['match']:
                print(f"      K grad: max_abs_diff={k_grad_comp['max_abs_diff']:.2e}")
            if not v_grad_comp['match']:
                print(f"      V grad: max_abs_diff={v_grad_comp['max_abs_diff']:.2e}")
        
        comparison[config_name] = stats_comp
    
    return comparison

def print_summary(comparison: Dict[str, Any], op_config: Dict[str, Any], mod_config: Dict[str, Any]):
    """Print comparison summary."""
    print("\n" + "="*80)
    print("📋 COMPARISON SUMMARY")
    print("="*80)
    
    # Test configuration comparison
    print(f"\n🔧 Test Configuration:")
    print(f"   Op implementation: {op_config.get('implementation', 'Unknown')}")
    print(f"   Mod implementation: {mod_config.get('implementation', 'Unknown')}")
    print(f"   Batch size: {op_config.get('batch_size', 'Unknown')}")
    print(f"   Sequence length: {op_config.get('seq_len', 'Unknown')}")
    print(f"   Num heads: {op_config.get('num_heads', 'Unknown')}")
    print(f"   Head dim: {op_config.get('head_dim', 'Unknown')}")
    print(f"   Data type: {op_config.get('dtype', 'Unknown')}")
    
    # Results summary
    total_configs = len(comparison)
    successful_comparisons = sum(1 for comp in comparison.values() if comp.get('all_match', False))
    failed_comparisons = total_configs - successful_comparisons
    
    print(f"\n📊 Results Summary:")
    print(f"   Total configurations: {total_configs}")
    print(f"   Successful comparisons: {successful_comparisons}")
    print(f"   Failed comparisons: {failed_comparisons}")
    print(f"   Success rate: {successful_comparisons/total_configs*100:.1f}%")
    
    if successful_comparisons == total_configs:
        print("   🎉 ALL COMPARISONS PASSED!")
    else:
        print("   ⚠️  SOME COMPARISONS FAILED!")
        
        # List failed configurations
        failed_configs = [name for name, comp in comparison.items() 
                         if not comp.get('all_match', False)]
        if failed_configs:
            print(f"   Failed configs: {failed_configs}")
    
    # Detailed statistics for successful comparisons
    if successful_comparisons > 0:
        print(f"\n📈 Detailed Statistics (successful comparisons):")
        output_diffs = []
        q_grad_diffs = []
        k_grad_diffs = []
        v_grad_diffs = []
        
        for comp in comparison.values():
            if comp.get('all_match', False):
                output_diffs.append(comp['output_comparison']['max_abs_diff'])
                q_grad_diffs.append(comp['q_grad_comparison']['max_abs_diff'])
                k_grad_diffs.append(comp['k_grad_comparison']['max_abs_diff'])
                v_grad_diffs.append(comp['v_grad_comparison']['max_abs_diff'])
        
        if output_diffs:
            print(f"   Output max abs diff: mean={np.mean(output_diffs):.2e}, max={np.max(output_diffs):.2e}")
            print(f"   Q grad max abs diff: mean={np.mean(q_grad_diffs):.2e}, max={np.max(q_grad_diffs):.2e}")
            print(f"   K grad max abs diff: mean={np.mean(k_grad_diffs):.2e}, max={np.max(k_grad_diffs):.2e}")
            print(f"   V grad max abs diff: mean={np.mean(v_grad_diffs):.2e}, max={np.max(v_grad_diffs):.2e}")

def main():
    """Main comparison function."""
    print("🔍 Comparing DotProductAttentionOp vs DotProductAttention")
    print("=" * 80)
    
    # Load results
    op_results_file = "dot_product_attention_op_results.pt"
    mod_results_file = "dot_product_attention_mod_results.pt"
    
    op_data = load_results(op_results_file)
    mod_data = load_results(mod_results_file)
    
    if op_data is None or mod_data is None:
        print("❌ Cannot proceed with comparison - missing result files")
        print("   Please run both tests first:")
        print("   python test_dot_product_attention_op.py")
        print("   python test_dot_product_attention_mod.py")
        return False
    
    # Compare input tensors
    print("\n🔍 Comparing input tensors...")
    q_input_comp = compare_tensors(op_data['input_q'], mod_data['input_q'], 'input_q')
    k_input_comp = compare_tensors(op_data['input_k'], mod_data['input_k'], 'input_k')
    v_input_comp = compare_tensors(op_data['input_v'], mod_data['input_v'], 'input_v')
    
    if q_input_comp['match'] and k_input_comp['match'] and v_input_comp['match']:
        print("   ✅ Input tensors match")
    else:
        print("   ⚠️  Input tensors don't match - this may affect comparison validity")
    
    # Compare configurations
    comparison = compare_configurations(op_data['results'], mod_data['results'])
    
    # Print summary
    print_summary(comparison, op_data['test_config'], mod_data['test_config'])
    
    # Return success if all comparisons passed
    total_configs = len(comparison)
    successful_comparisons = sum(1 for comp in comparison.values() if comp.get('all_match', False))
    
    return successful_comparisons == total_configs

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 