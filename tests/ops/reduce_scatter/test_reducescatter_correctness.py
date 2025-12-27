#!/usr/bin/env python3
"""
Correctness test for ReduceScatterKV operation - verifies MSCCL results match PyTorch/NCCL.

Usage:
    torchrun --nproc_per_node=<N> test_reducescatter_correctness.py [options]

Example:
    torchrun --nproc_per_node=2 test_reducescatter_correctness.py --dtype bfloat16
    torchrun --nproc_per_node=4 test_reducescatter_correctness.py --dtype float16 --test-sizes 1024 4096 16384
"""
"""
torchrun --nproc_per_node=2 test_reducescatter_correctness.py \
        --dtype bfloat16 \
        --test-size 65536 \
        --sm-num 8 \
        --block-size 1024 \
        --rtol 1e-3 \
        --atol 1e-3
"""

import argparse
import sys
import os

import torch
import torch.distributed as dist

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from kareus.transformer_engine.pytorch.ops.basic.reduce_scatter_kv import ReduceScatterKV
from kareus.msccl import msccl_comm


def check_tensors_equal(tensor1, tensor2, rtol=1e-3, atol=1e-3, tensor_name="tensor"):
    """Check if two tensors are approximately equal."""
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        # Find locations of largest differences
        max_indices = torch.where(diff == torch.max(diff))
        
        # Get the first location where max occurs
        # max_indices is a tuple of tensors, one for each dimension
        idx_tuple = tuple(idx[0].item() for idx in max_indices)
        
        val1 = tensor1[idx_tuple].item()
        val2 = tensor2[idx_tuple].item()
        
        print(f"❌ MISMATCH in {tensor_name}:")
        print(f"   Max difference: {max_diff:.6e}")
        print(f"   Mean difference: {mean_diff:.6e}")
        print(f"   Location of max diff: {idx_tuple}")
        print(f"   Tensor1 value: {val1:.10f}")
        print(f"   Tensor2 value: {val2:.10f}")
        print(f"   Actual diff:   {abs(val1 - val2):.10f}")
        return False
    return True


def test_reducescatter_correctness(rank, world_size, device, dtype, test_size, args):
    """Test ReduceScatterKV correctness for a specific tensor size."""
    
    # Calculate tensor shape (full size before reduce-scatter)
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    total_elements_per_rank = test_size // bytes_per_elem
    
    # Create a 2D tensor for better memory access
    hidden_size = int(total_elements_per_rank ** 0.5)
    # MSCCL++ kernels in this repo operate in 4-byte (int32) units.
    # For fp16/bf16 that means the per-rank output payload must have an even number of elements.
    # Force an even hidden_size to guarantee seq_len_per_rank * hidden_size is even.
    if dtype in (torch.float16, torch.bfloat16) and (hidden_size % 2 == 1):
        hidden_size = max(2, hidden_size - 1)
    seq_len_per_rank = total_elements_per_rank // hidden_size
    effective_elements_per_rank = seq_len_per_rank * hidden_size
    effective_bytes_per_rank = effective_elements_per_rank * bytes_per_elem
    
    # Full tensor shape before reduce-scatter
    tensor_shape_full = [seq_len_per_rank * world_size, hidden_size]
    # Output shape after reduce-scatter (per rank)
    tensor_shape_per_rank = [seq_len_per_rank, hidden_size]
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Testing: Full shape {tensor_shape_full} → Per-rank shape {tensor_shape_per_rank}")
        print(
            f"         ({dtype}), Requested output size/rank: {test_size / (1024**2):.2f} MB "
            f"(effective: {effective_bytes_per_rank / (1024**2):.2f} MB)"
        )
        print(f"{'='*80}")
    
    # Create input tensors - use deterministic values for reproducibility
    torch.manual_seed(42 + rank)
    input_k = torch.randn(tensor_shape_full, dtype=dtype, device=device, requires_grad=False)
    input_v = torch.randn(tensor_shape_full, dtype=dtype, device=device, requires_grad=False)
    
    # Make copies for different backends
    input_k_ref = input_k.clone()
    input_v_ref = input_v.clone()
    
    # Create process group
    tp_group = dist.new_group(list(range(world_size)))
    
    # Test 1: PyTorch native ReduceScatter (reference)
    if rank == 0:
        print("Running PyTorch/NCCL ReduceScatter (reference)...")
    
    output_k_ref = torch.empty(tensor_shape_per_rank, dtype=dtype, device=device)
    output_v_ref = torch.empty(tensor_shape_per_rank, dtype=dtype, device=device)
    
    dist.reduce_scatter_tensor(output_k_ref, input_k_ref, group=tp_group)
    dist.reduce_scatter_tensor(output_v_ref, input_v_ref, group=tp_group)
    torch.cuda.synchronize()
    dist.barrier()
    
    if rank == 0:
        print("✓ PyTorch ReduceScatter completed")
    
    # Test 2: NCCL backend through our wrapper
    if rank == 0:
        print("Running NCCL backend through wrapper...")
    
    input_k_nccl = input_k.clone()
    input_v_nccl = input_v.clone()
    
    reducescatter_nccl = ReduceScatterKV(
        process_group=tp_group,
        async_op=args.async_op,
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    
    output_k_nccl, output_v_nccl = reducescatter_nccl.op_forward(
        None,  # ctx
        input_k_nccl,
        input_v_nccl,
    )
    if args.async_op:
        reducescatter_nccl.sync(torch.cuda.current_stream())
    torch.cuda.synchronize()
    dist.barrier()
    
    # Check NCCL results match PyTorch
    nccl_k_match = check_tensors_equal(
        output_k_nccl, output_k_ref, 
        rtol=args.rtol, atol=args.atol,
        tensor_name="NCCL K vs PyTorch K"
    )
    nccl_v_match = check_tensors_equal(
        output_v_nccl, output_v_ref, 
        rtol=args.rtol, atol=args.atol,
        tensor_name="NCCL V vs PyTorch V"
    )
    
    if rank == 0:
        if nccl_k_match and nccl_v_match:
            print("✓ NCCL backend matches PyTorch reference")
        else:
            print("✗ NCCL backend DOES NOT match PyTorch reference")
    
    # Test 3: MSCCL backend
    if rank == 0:
        print("Running MSCCL backend...")
    
    input_k_msccl = input_k.clone()
    input_v_msccl = input_v.clone()
    
    reducescatter_msccl = ReduceScatterKV(
        process_group=tp_group,
        async_op=True,
        backend="msccl",
        rank=rank,
        world_size=world_size,
        nranks_per_node=world_size,
        tensor_size=list(tensor_shape_full),
        device=device,
        dtype=dtype,
        batch_idx=0,
    )
    
    # Copy input to buffers
    reducescatter_msccl.input_buffer_k.copy_(input_k_msccl)
    reducescatter_msccl.input_buffer_v.copy_(input_v_msccl)
    
    # Run MSCCL ReduceScatter
    output_k_msccl, output_v_msccl = reducescatter_msccl.op_forward(
        None,  # ctx
        input_k_msccl,
        input_v_msccl,
        sm_num=args.sm_num,
        block_size=args.block_size,
    )
    reducescatter_msccl.sync(torch.cuda.current_stream())
    torch.cuda.synchronize()
    dist.barrier()
    
    # Check MSCCL results match PyTorch
    msccl_k_match = check_tensors_equal(
        output_k_msccl, output_k_ref,
        rtol=args.rtol, atol=args.atol,
        tensor_name="MSCCL K vs PyTorch K"
    )
    msccl_v_match = check_tensors_equal(
        output_v_msccl, output_v_ref,
        rtol=args.rtol, atol=args.atol,
        tensor_name="MSCCL V vs PyTorch V"
    )
    
    if rank == 0:
        if msccl_k_match and msccl_v_match:
            print("✓ MSCCL backend matches PyTorch reference")
        else:
            print("✗ MSCCL backend DOES NOT match PyTorch reference")
    
    # Check MSCCL matches NCCL
    msccl_nccl_k_match = check_tensors_equal(
        output_k_msccl, output_k_nccl,
        rtol=args.rtol, atol=args.atol,
        tensor_name="MSCCL K vs NCCL K"
    )
    msccl_nccl_v_match = check_tensors_equal(
        output_v_msccl, output_v_nccl,
        rtol=args.rtol, atol=args.atol,
        tensor_name="MSCCL V vs NCCL V"
    )
    
    if rank == 0:
        if msccl_nccl_k_match and msccl_nccl_v_match:
            print("✓ MSCCL matches NCCL backend")
        else:
            print("✗ MSCCL DOES NOT match NCCL backend")
    
    # Overall result
    all_match = (nccl_k_match and nccl_v_match and 
                 msccl_k_match and msccl_v_match and 
                 msccl_nccl_k_match and msccl_nccl_v_match)
    
    if rank == 0:
        print(f"{'='*80}")
        if all_match:
            print("✅ ALL TESTS PASSED - Results are correct!")
        else:
            print("❌ SOME TESTS FAILED - Results mismatch detected!")
        print(f"{'='*80}")
    
    # Cleanup to avoid memory leaks between tests
    del reducescatter_nccl
    del reducescatter_msccl
    del input_k_nccl, input_v_nccl
    del input_k_msccl, input_v_msccl
    del output_k_nccl, output_v_nccl
    del output_k_msccl, output_v_msccl
    torch.cuda.empty_cache()
    
    return all_match


def main():
    parser = argparse.ArgumentParser(description="Test ReduceScatterKV correctness")
    
    # Data type
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type to test (default: bfloat16)",
    )
    
    # Test configurations
    parser.add_argument(
        "--test-size",
        type=int,
        default=65536,  # bytes (64KB default per rank output)
        help="Test size per rank output in bytes (default: 64KB)",
    )
    
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for comparison (default: 1e-2)",
    )
    
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for comparison (default: 1e-2)",
    )
    
    # NCCL settings
    parser.add_argument(
        "--async-op",
        action="store_true",
        default=True,
        help="Use async reduce-scatter for NCCL (default: True)",
    )
    
    # MSCCL settings
    parser.add_argument(
        "--sm-num",
        type=int,
        default=8,
        help="Number of SMs for MSCCL (default: 8)",
    )
    
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="CUDA block size for MSCCL (default: 1024)",
    )
    
    args = parser.parse_args()
    
    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set CUDA device
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    if rank == 0:
        print("=" * 80)
        print("ReduceScatterKV Correctness Test")
        print("=" * 80)
        print(f"World size:       {world_size}")
        print(f"Data type:        {args.dtype}")
        print(f"Output size/rank: {args.test_size} bytes ({args.test_size / 1024:.1f} KB)")
        print(f"Input total size: {args.test_size * world_size} bytes ({args.test_size * world_size / 1024:.1f} KB)")
        print(f"Tolerance:        rtol={args.rtol}, atol={args.atol}")
        print(f"MSCCL config:     SM={args.sm_num}, BlockSize={args.block_size}")
        print("=" * 80)
    
    # Run single test
    try:
        all_passed = test_reducescatter_correctness(
            rank, world_size, device, dtype, args.test_size, args
        )
    except Exception as e:
        if rank == 0:
            print(f"\n❌ Exception occurred: {e}")
            import traceback
            traceback.print_exc()
        all_passed = False
    
    dist.barrier()
    
    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)
        if all_passed:
            print("✅ CORRECTNESS TEST PASSED!")
        else:
            print("❌ CORRECTNESS TEST FAILED - Please review the results above")
        print("=" * 80)
    
    # Cleanup MSCCL resources
    try:
        msccl_comm.msccl_cleanup()
    except Exception:
        pass
    
    # Cleanup distributed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

