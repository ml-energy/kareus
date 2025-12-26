#!/usr/bin/env python3
"""
Correctness test for AllReduce operation - verifies MSCCL results match PyTorch/NCCL.

Usage:
    torchrun --nproc_per_node=<N> test_allreduce_correctness.py [options]

Example:
    torchrun --nproc_per_node=2 test_allreduce_correctness.py --dtype bfloat16
    torchrun --nproc_per_node=4 test_allreduce_correctness.py --dtype float16 --test-sizes 1024 4096 16384
"""
"""
torchrun --nproc_per_node=2 test_allreduce_correctness.py \
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

from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.msccl import msccl_comm


def check_tensors_equal(tensor1, tensor2, rtol=1e-3, atol=1e-3, tensor_name="tensor"):
    """Check if two tensors are approximately equal."""
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        # Find locations of largest differences
        max_indices = torch.where(diff == torch.max(diff))
        
        print(f"❌ MISMATCH in {tensor_name}:")
        print(f"   Max difference: {max_diff:.6e}")
        print(f"   Mean difference: {mean_diff:.6e}")
        print(f"   Location of max diff: {[idx[0].item() for idx in max_indices]}")
        print(f"   MSCCL value: {tensor1.flatten()[max_indices[0][0]].item()}")
        print(f"   PyTorch value: {tensor2.flatten()[max_indices[0][0]].item()}")
        return False
    return True


def test_allreduce_correctness(rank, world_size, device, dtype, test_size, args):
    """Test AllReduce correctness for a specific tensor size."""
    
    # Calculate tensor shape
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    total_elements = test_size // bytes_per_elem
    
    # Create a 2D tensor for better memory access
    hidden_size = int(total_elements ** 0.5)
    seq_len = total_elements // hidden_size
    tensor_shape = [seq_len, hidden_size]
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Testing: {tensor_shape} ({dtype}), Size: {test_size / (1024**2):.2f} MB")
        print(f"{'='*80}")
    
    # Create input tensors - use deterministic values for reproducibility
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(tensor_shape, dtype=dtype, device=device, requires_grad=False)
    
    # Make a copy for PyTorch reference
    input_tensor_ref = input_tensor.clone()
    
    # Create process group
    tp_group = dist.new_group(list(range(world_size)))
    
    # Test 1: PyTorch native AllReduce (reference)
    if rank == 0:
        print("Running PyTorch/NCCL AllReduce (reference)...")
    
    dist.all_reduce(input_tensor_ref, op=dist.ReduceOp.SUM, group=tp_group)
    torch.cuda.synchronize()
    dist.barrier()
    
    if rank == 0:
        print("✓ PyTorch AllReduce completed")
    
    # Test 2: NCCL backend through our wrapper
    if rank == 0:
        print("Running NCCL backend through wrapper...")
    
    input_tensor_nccl = input_tensor.clone()
    allreduce_nccl = AllReduce(
        process_group=tp_group,
        async_op=args.async_op,
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    
    output_nccl = allreduce_nccl(input_tensor_nccl)
    if args.async_op:
        allreduce_nccl.sync()
    torch.cuda.synchronize()
    dist.barrier()
    
    # Check NCCL results match PyTorch
    nccl_match = check_tensors_equal(
        output_nccl, input_tensor_ref, 
        rtol=args.rtol, atol=args.atol,
        tensor_name="NCCL vs PyTorch"
    )
    
    if rank == 0:
        if nccl_match:
            print("✓ NCCL backend matches PyTorch reference")
        else:
            print("✗ NCCL backend DOES NOT match PyTorch reference")
    
    # Test 3: MSCCL backend
    if rank == 0:
        print("Running MSCCL backend...")
    
    input_tensor_msccl = input_tensor.clone()
    
    allreduce_msccl = AllReduce(
        process_group=tp_group,
        async_op=True,
        backend="msccl",
        rank=rank,
        world_size=world_size,
        tensor_size=list(tensor_shape),
        device=device,
        dtype=dtype,
        batch_idx=0,  # Use batch_idx 0 for single test
    )
    
    # Run MSCCL AllReduce
    allreduce_msccl.input_buffer.copy_(input_tensor_msccl)
    output_msccl = allreduce_msccl(
        allreduce_msccl.input_buffer, 
        sm_num=args.sm_num, 
        block_size=args.block_size
    )
    allreduce_msccl.sync(torch.cuda.current_stream())
    torch.cuda.synchronize()
    dist.barrier()
    
    # Check MSCCL results match PyTorch
    msccl_match = check_tensors_equal(
        output_msccl, input_tensor_ref,
        rtol=args.rtol, atol=args.atol,
        tensor_name="MSCCL vs PyTorch"
    )
    
    if rank == 0:
        if msccl_match:
            print("✓ MSCCL backend matches PyTorch reference")
        else:
            print("✗ MSCCL backend DOES NOT match PyTorch reference")
    
    # Check MSCCL matches NCCL
    msccl_nccl_match = check_tensors_equal(
        output_msccl, output_nccl,
        rtol=args.rtol, atol=args.atol,
        tensor_name="MSCCL vs NCCL"
    )
    
    if rank == 0:
        if msccl_nccl_match:
            print("✓ MSCCL matches NCCL backend")
        else:
            print("✗ MSCCL DOES NOT match NCCL backend")
    
    # Overall result
    all_match = nccl_match and msccl_match and msccl_nccl_match
    
    if rank == 0:
        print(f"{'='*80}")
        if all_match:
            print("✅ ALL TESTS PASSED - Results are correct!")
        else:
            print("❌ SOME TESTS FAILED - Results mismatch detected!")
        print(f"{'='*80}")
    
    # Cleanup to avoid memory leaks between tests
    del allreduce_nccl
    del allreduce_msccl
    del input_tensor_nccl
    del input_tensor_msccl
    del output_nccl
    del output_msccl
    torch.cuda.empty_cache()
    
    return all_match


def main():
    parser = argparse.ArgumentParser(description="Test AllReduce correctness")
    
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
        default=65536,  # bytes (64KB default)
        help="Test size in bytes (default: 64KB)",
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
        help="Use async all-reduce for NCCL (default: True)",
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
        print("AllReduce Correctness Test")
        print("=" * 80)
        print(f"World size:       {world_size}")
        print(f"Data type:        {args.dtype}")
        print(f"Test size:        {args.test_size} bytes ({args.test_size / 1024:.1f} KB)")
        print(f"Tolerance:        rtol={args.rtol}, atol={args.atol}")
        print(f"MSCCL config:     SM={args.sm_num}, BlockSize={args.block_size}")
        print("=" * 80)
    
    # Run single test
    try:
        all_passed = test_allreduce_correctness(
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

