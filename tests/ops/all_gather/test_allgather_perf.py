#!/usr/bin/env python3
"""
Performance test for msccl_AllGather with tensor shape [8, 4096, 8, 128].

Usage:
    torchrun --nproc_per_node=<N> test_allgather_perf.py [options]

Example:
    torchrun --nproc_per_node=8 test_allgather_perf.py
    torchrun --nproc_per_node=8 test_allgather_perf.py --warmup 10 --iters 50
"""

import argparse
import time

import torch
import torch.distributed as dist

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from kareus.msccl.msccl_comm import (
    msccl_AllGather_init,
    msccl_AllGather,
    msccl_AllGather_sync,
    msccl_cleanup,
)

# torchrun --nproc_per_node=2 test_allgather_perf.py

def main():
    parser = argparse.ArgumentParser(description="Test msccl_AllGather performance")
    parser.add_argument(
        "--nblocks",
        type=int,
        default=24,
        help="Number of CUDA blocks (default: 24)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="CUDA block size (default: 512)",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=3,
        help="Pipeline depth (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of test iterations (default: 20)",
    )
    parser.add_argument(
        "--nranks-per-node",
        type=int,
        default=0,
        help="Number of ranks per node (0 = auto-detect, default: 0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type (default: float16)",
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
    
    # Tensor shape: [8, 4096, 8, 128]
    tensor_shape = [8, 8192, 8, 128]
    num_elements = 8 * 8192 * 8 * 128
    
    if rank == 0:
        print("=" * 80)
        print("msccl_AllGather Performance Test")
        print("=" * 80)
        print(f"World size:       {world_size}")
        print(f"Tensor shape:     {tensor_shape}")
        print(f"Total elements:   {num_elements:,}")
        print(f"Data type:        {args.dtype}")
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        size_mb = num_elements * bytes_per_elem / (1024**2)
        print(f"Total data size:  {size_mb:.2f} MB")
        print(f"nblocks:          {args.nblocks}")
        print(f"block_size:       {args.block_size}")
        print(f"pipeline_depth:   {args.pipeline_depth}")
        print(f"Warmup iters:     {args.warmup}")
        print(f"Test iters:       {args.iters}")
        print(f"Ranks per node:   {args.nranks_per_node if args.nranks_per_node > 0 else 'auto'}")
        print("=" * 80)
        print()
    
    # Create input tensor
    input_tensor = torch.randn(tensor_shape, dtype=dtype, device=device)
    
    if rank == 0:
        print(f"Initializing msccl_AllGather...")
    
    # Initialize AllGather
    msccl_AllGather_init(
        rank=rank,
        world_size=world_size,
        input_tensor=input_tensor,
        group=None,
        nranks_per_node=args.nranks_per_node,
    )
    
    if rank == 0:
        print(f"Running warmup ({args.warmup} iterations)...")
    
    # Warmup iterations
    for i in range(args.warmup):
        msccl_AllGather(
            nblocks=args.nblocks,
            block_size=args.block_size,
            pipeline_depth=args.pipeline_depth
        )
        msccl_AllGather_sync()
    
    dist.barrier()
    
    if rank == 0:
        print(f"Running benchmark ({args.iters} iterations)...")
        print()
    
    # Benchmark iterations
    start_time = time.perf_counter()
    
    for i in range(args.iters):
        msccl_AllGather(
            nblocks=args.nblocks,
            block_size=args.block_size,
            pipeline_depth=args.pipeline_depth
        )
        # msccl_AllGather_sync()
        
        # if rank == 0 and (i + 1) % 10 == 0:
        #     print(f"  Completed {i + 1}/{args.iters} iterations...")
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_latency = total_time_ms / args.iters
    
    # Calculate bandwidth
    # input_tensor is already the total gathered size
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    bytes_transferred = num_elements * bytes_per_elem
    avg_bandwidth_gbps = (bytes_transferred / (1024**3)) / (avg_latency / 1000)
    
    if rank == 0:
        print()
        print("=" * 80)
        print("Results")
        print("=" * 80)
        print(f"Total time:       {total_time_ms:.3f} ms")
        print(f"Iterations:       {args.iters}")
        print(f"Average latency:  {avg_latency:.3f} ms")
        print(f"Bandwidth:        {avg_bandwidth_gbps:.2f} GB/s")
        print(f"Throughput:       {1000 / avg_latency:.2f} ops/sec")
        print("=" * 80)
    
    # Cleanup
    msccl_cleanup()
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

