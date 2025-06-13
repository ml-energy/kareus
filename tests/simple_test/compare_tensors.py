import torch
import os
import numpy as np
from pathlib import Path

def load_tensor(file_path):
    """Load a tensor from a .pt file"""
    try:
        tensor = torch.load(file_path, map_location='cpu')
        return tensor
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_tensors(tensor1, tensor2, name="tensors"):
    """Compare two tensors and return detailed statistics"""
    if tensor1 is None or tensor2 is None:
        print(f"Cannot compare {name}: one or both tensors are None")
        return
    
    print(f"\n=== Comparing {name} ===")
    
    # Basic shape and dtype comparison
    print(f"Tensor 1 shape: {tensor1.shape}")
    print(f"Tensor 2 shape: {tensor2.shape}")
    print(f"Tensor 1 dtype: {tensor1.dtype}")
    print(f"Tensor 2 dtype: {tensor2.dtype}")
    
    if tensor1.shape != tensor2.shape:
        print("⚠️  WARNING: Tensors have different shapes!")
        return
    
    if tensor1.dtype != tensor2.dtype:
        print("⚠️  WARNING: Tensors have different dtypes!")
    
    # Convert to same dtype for comparison if needed
    if tensor1.dtype != tensor2.dtype:
        tensor2 = tensor2.to(tensor1.dtype)
    
    # Calculate differences
    diff = tensor1 - tensor2
    abs_diff = torch.abs(diff)
    
    # Statistics
    print(f"Max absolute difference: {abs_diff.max().item():.10f}")
    print(f"Mean absolute difference: {abs_diff.mean().item():.10f}")
    print(f"Standard deviation of differences: {diff.std().item():.10f}")
    
    # Relative error (avoid division by zero)
    tensor1_abs = torch.abs(tensor1)
    non_zero_mask = tensor1_abs > 1e-15
    if non_zero_mask.any():
        rel_diff = abs_diff[non_zero_mask] / tensor1_abs[non_zero_mask]
        print(f"Max relative difference (non-zero elements): {rel_diff.max().item():.10f}")
        print(f"Mean relative difference (non-zero elements): {rel_diff.mean().item():.10f}")
    
    # Cosine similarity analysis
    print(f"\n--- Cosine Similarity Analysis ---")
    
    # Overall cosine similarity (flatten tensors)
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()
    
    # Compute overall cosine similarity
    cos_sim_overall = torch.nn.functional.cosine_similarity(
        tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1
    ).item()
    print(f"Overall cosine similarity: {cos_sim_overall:.10f}")
    
    # Cosine distance (1 - cosine similarity)
    cos_distance = 1 - cos_sim_overall
    print(f"Overall cosine distance: {cos_distance:.10f}")

    batch_size = tensor1.shape[1]
    seq_len = tensor1.shape[0]
    tensor1_2d = tensor1.view(seq_len * batch_size, -1)
    tensor2_2d = tensor2.view(seq_len * batch_size, -1)

    if batch_size > 1:
        cos_sims = torch.nn.functional.cosine_similarity(tensor1_2d, tensor2_2d, dim=1)
        print(f"Cosine similarity of each element: {cos_sims}")
        print(f"Mean cosine similarity: {cos_sims.mean().item()}")
        print(f"Std cosine similarity: {cos_sims.std().item()}")
        print(f"Min cosine similarity: {cos_sims.min().item()}")
        print(f"Max cosine similarity: {cos_sims.max().item()}")
    
    cos_equal = cos_sim_overall > 0.98
    print(f"Cosine similarity is equal: {cos_equal}")
    # Check if tensors are approximately equal
    rtol = 1e-5
    atol = 1e-8
    are_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    print(f"\nTensors are approximately equal (rtol={rtol}, atol={atol}): {are_close}")
    
    if not are_close:
        # Count number of different elements
        diff_elements = (~torch.isclose(tensor1, tensor2, rtol=rtol, atol=atol)).sum().item()
        total_elements = tensor1.numel()
        print(f"Number of different elements: {diff_elements}/{total_elements} ({100*diff_elements/total_elements:.2f}%)")
    
    # Check for exact equality
    exact_equal = torch.equal(tensor1, tensor2)
    print(f"Tensors are exactly equal: {exact_equal}")
    
    result = {
        'max_abs_diff': abs_diff.max().item(),
        'mean_abs_diff': abs_diff.mean().item(),
        'are_close': are_close,
        'exact_equal': exact_equal,
        'cosine_similarity': cos_sim_overall,
        'cosine_distance': cos_distance,
        'cosine_equal': cos_equal
    }

    result.update({
        'cosine_sim_mean': cos_sims.mean().item(),
        'cosine_sim_std': cos_sims.std().item(),
        'cosine_sim_min': cos_sims.min().item(),
        'cosine_sim_max': cos_sims.max().item()
    })
    
    return result

def main():
    """Main function to compare tensors in the two directories"""
    dir1 = Path("compare_results/te")
    dir2 = Path("compare_results/kareus")
    
    print("🔍 Tensor Comparison Tool")
    print("=" * 50)
    
    # Get list of .pt files in both directories
    files1 = sorted([f.name for f in dir1.glob("*.pt")])
    files2 = sorted([f.name for f in dir2.glob("*.pt")])
    
    print(f"Files in directory 1: {files1}")
    print(f"Files in directory 2: {files2}")
    
    # Find common files
    common_files = set(files1) & set(files2)
    only_in_dir1 = set(files1) - set(files2)
    only_in_dir2 = set(files2) - set(files1)
    
    if only_in_dir1:
        print(f"⚠️  Files only in directory 1: {list(only_in_dir1)}")
    if only_in_dir2:
        print(f"⚠️  Files only in directory 2: {list(only_in_dir2)}")
    
    if not common_files:
        print("❌ No common files found for comparison!")
        return
    
    print(f"📊 Comparing {len(common_files)} common files: {sorted(list(common_files))}")
    
    results = {}
    
    # Compare each common file
    for filename in sorted(common_files):
        file1_path = dir1 / filename
        file2_path = dir2 / filename
        
        print(f"\n📁 Loading {filename}...")
        tensor1 = load_tensor(file1_path)
        tensor2 = load_tensor(file2_path)
        
        if tensor1 is not None and tensor2 is not None:
            results[filename] = compare_tensors(tensor1, tensor2, filename)
        else:
            print(f"❌ Failed to load tensors for {filename}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    all_close = True
    all_exact = True
    all_cosine_equal = True
    for filename, result in results.items():
        status_close = "✅" if result['are_close'] else "❌"
        status_exact = "✅" if result['exact_equal'] else "❌"
        status_cosine = "✅" if result['cosine_equal'] else "❌"
        print(f"{filename}:")
        print(f"  Approximately equal: {status_close}")
        print(f"  Exactly equal: {status_exact}")
        print(f"  Cosine similarity equal: {status_cosine}")
        print(f"  Max absolute difference: {result['max_abs_diff']:.2e}")
        print(f"  Overall cosine similarity: {result['cosine_similarity']}")
        
        all_close &= result['are_close']
        all_exact &= result['exact_equal']
        all_cosine_equal &= result['cosine_equal']
    
    print(f"\n🎯 Overall result:")
    print(f"All tensors approximately equal: {'✅' if all_close else '❌'}")
    print(f"All tensors exactly equal: {'✅' if all_exact else '❌'}")
    print(f"All tensors cosine similarity equal: {'✅' if all_cosine_equal else '❌'}")
if __name__ == "__main__":
    main() 