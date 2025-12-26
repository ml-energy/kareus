#!/usr/bin/env python3
"""
Performance test for AllReduce operation with NCCL and MSCCL backends.

Usage:
    torchrun --nproc_per_node=<N> test_allreduce_perf.py [options]

Example:
    torchrun --nproc_per_node=2 test_allreduce_perf.py
    torchrun --nproc_per_node=4 test_allreduce_perf.py --warmup 20 --iters 100 --tensor-size-mb 16
    torchrun --nproc_per_node=2 test_allreduce_perf.py --backend msccl --sm-num 8 --block-size 1024
"""
"""
torchrun --nproc_per_node=2 test_allreduce_perf.py \
 --backend msccl \
    --dtype bfloat16 \
    --tensor-size-mb 32.0 \
    --sm-num 8 \
    --block-size 1024 \
    --warmup 10 \
    --iters 5000 \
    --measure-energy
"""

import argparse
import time
import sys
import os

import torch
import torch.distributed as dist

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce

# Import Zeus for energy measurement
try:
    from zeus.monitor import ZeusMonitor
    HAVE_ZEUS = True
except ImportError:
    HAVE_ZEUS = False
    print("Warning: Zeus not available. Energy measurements will be disabled.")


def format_bandwidth(bandwidth_gbps):
    """Format bandwidth for display."""
    return f"{bandwidth_gbps:.2f} GB/s"


def format_latency(latency_ms):
    """Format latency for display."""
    if latency_ms < 1.0:
        return f"{latency_ms * 1000:.2f} µs"
    else:
        return f"{latency_ms:.3f} ms"


def run_nccl_benchmark(args, rank, world_size, device, dtype):
    """Benchmark NCCL backend."""
    
    # Calculate tensor shape based on target size
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    total_elements = int((args.tensor_size_mb * 1024 * 1024) / bytes_per_elem)
    
    # Create a 2D tensor: [seq_len, hidden_size]
    # Try to make it roughly square for better memory access
    hidden_size = int(total_elements ** 0.5)
    seq_len = total_elements // hidden_size
    tensor_shape = [seq_len, hidden_size]
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("NCCL Backend Benchmark")
        print("=" * 80)
        print(f"Tensor shape:     {tensor_shape}")
        print(f"Total elements:   {total_elements:,}")
        print(f"Data type:        {args.dtype}")
        size_mb = total_elements * bytes_per_elem / (1024**2)
        print(f"Tensor size:      {size_mb:.2f} MB per rank")
        print(f"Async mode:       {args.async_op}")
        print(f"Warmup iters:     {args.warmup}")
        print(f"Test iters:       {args.iters}")
        print(f"Energy measure:   {args.measure_energy and HAVE_ZEUS}")
        print("=" * 80)
    
    # Create input tensor
    input_tensor = torch.randn(tensor_shape, dtype=dtype, device=device, requires_grad=False)
    
    # Create AllReduce operation
    tp_group = dist.new_group(list(range(world_size)))
    
    allreduce_op = AllReduce(
        process_group=tp_group,
        async_op=args.async_op,
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    
    if rank == 0:
        print("Running warmup...")
    
    # Warmup
    for _ in range(args.warmup):
        output = allreduce_op(input_tensor.clone())
        if args.async_op:
            allreduce_op.sync()
        torch.cuda.synchronize()
    
    dist.barrier()
    
    if rank == 0:
        print("Running benchmark...\n")
    
    # Initialize energy monitor
    monitor = None
    if rank == 0 and args.measure_energy and HAVE_ZEUS:
        monitor = ZeusMonitor(gpu_indices=list(range(world_size)))
    
    # Benchmark
    torch.cuda.synchronize()
    dist.barrier()
    
    if rank == 0 and monitor is not None:
        monitor.begin_window("benchmark")
    
    start_time = time.perf_counter()
    
    for _ in range(args.iters):
        output = allreduce_op(input_tensor.clone())
        if args.async_op:
            allreduce_op.sync()
    
    torch.cuda.synchronize()
    dist.barrier()
    
    end_time = time.perf_counter()
    
    if rank == 0 and monitor is not None:
        result = monitor.end_window("benchmark")
        total_energy_j = float(result.total_energy)
        avg_energy_j = total_energy_j / float(args.iters) / float(world_size)
    else:
        avg_energy_j = None
    
    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / args.iters
    
    # Calculate bandwidth
    # AllReduce transfers (world_size - 1) / world_size * 2 * data_size
    # Using ring algorithm approximation: 2 * (N-1) / N * data_size
    bytes_per_op = total_elements * bytes_per_elem * 2 * (world_size - 1) / world_size
    avg_bandwidth_gbps = (bytes_per_op / (1024**3)) / (avg_latency_ms / 1000)
    
    # Calculate algorithmic bandwidth (bus bandwidth)
    # For AllReduce: (N-1)/N * S where S is the size
    algo_bytes = total_elements * bytes_per_elem
    algo_bandwidth_gbps = (algo_bytes / (1024**3)) / (avg_latency_ms / 1000)
    
    if rank == 0:
        print("=" * 80)
        print("NCCL Results")
        print("=" * 80)
        print(f"Total time:          {total_time_ms:.3f} ms")
        print(f"Iterations:          {args.iters}")
        print(f"Average latency:     {format_latency(avg_latency_ms)}")
        print(f"Algorithm BW:        {format_bandwidth(algo_bandwidth_gbps)}")
        print(f"Bus bandwidth:       {format_bandwidth(avg_bandwidth_gbps)}")
        print(f"Throughput:          {1000 / avg_latency_ms:.2f} ops/sec")
        if avg_energy_j is not None:
            print(f"Avg energy/op:       {avg_energy_j:.6f} J")
            print(f"Avg power:           {avg_energy_j / (avg_latency_ms / 1000):.2f} W")
            print(f"Energy efficiency:   {(algo_bytes / (1024**3)) / avg_energy_j:.2f} GB/J")
        print("=" * 80)
    
    return avg_latency_ms, avg_bandwidth_gbps, avg_energy_j


def run_msccl_benchmark(args, rank, world_size, device, dtype):
    """Benchmark MSCCL backend."""
    
    # Calculate tensor shape based on target size
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    total_elements = int((args.tensor_size_mb * 1024 * 1024) / bytes_per_elem)
    
    # Create a 2D tensor
    hidden_size = int(total_elements ** 0.5)
    seq_len = total_elements // hidden_size
    tensor_shape = [seq_len, hidden_size]
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("MSCCL Backend Benchmark")
        print("=" * 80)
        print(f"Tensor shape:     {tensor_shape}")
        print(f"Total elements:   {total_elements:,}")
        print(f"Data type:        {args.dtype}")
        size_mb = total_elements * bytes_per_elem / (1024**2)
        print(f"Tensor size:      {size_mb:.2f} MB per rank")
        print(f"SM count:         {args.sm_num}")
        print(f"Block size:       {args.block_size}")
        print(f"Warmup iters:     {args.warmup}")
        print(f"Test iters:       {args.iters}")
        print(f"Energy measure:   {args.measure_energy and HAVE_ZEUS}")
        print("=" * 80)
    
    # Create input tensor
    input_tensor = torch.randn(tensor_shape, dtype=dtype, device=device, requires_grad=False)
    
    # Create AllReduce operation with MSCCL
    tp_group = dist.new_group(list(range(world_size)))
    
    allreduce_op = AllReduce(
        process_group=tp_group,
        async_op=True,
        backend="msccl",
        rank=rank,
        world_size=world_size,
        tensor_size=list(tensor_shape),
        device=device,
        dtype=dtype,
        batch_idx=0,
    )
    
    if rank == 0:
        print("Running warmup...")
    
    # Warmup
    for _ in range(args.warmup):
        # Copy input to buffer
        allreduce_op.input_buffer.copy_(input_tensor)
        output = allreduce_op(allreduce_op.input_buffer, sm_num=args.sm_num, block_size=args.block_size)
        allreduce_op.sync(torch.cuda.current_stream())
        torch.cuda.synchronize()
    
    dist.barrier()
    
    if rank == 0:
        print("Running benchmark...\n")
    
    # Initialize energy monitor
    monitor = None
    if rank == 0 and args.measure_energy and HAVE_ZEUS:
        monitor = ZeusMonitor(gpu_indices=list(range(world_size)))
    
    # Benchmark
    torch.cuda.synchronize()
    dist.barrier()
    
    if rank == 0 and monitor is not None:
        monitor.begin_window("benchmark")
    
    start_time = time.perf_counter()
    
    for _ in range(args.iters):
        allreduce_op.input_buffer.copy_(input_tensor)
        output = allreduce_op(allreduce_op.input_buffer, sm_num=args.sm_num, block_size=args.block_size)
        allreduce_op.sync(torch.cuda.current_stream())
    
    torch.cuda.synchronize()
    dist.barrier()
    
    end_time = time.perf_counter()
    
    if rank == 0 and monitor is not None:
        result = monitor.end_window("benchmark")
        total_energy_j = float(result.total_energy)
        avg_energy_j = total_energy_j / float(args.iters) / float(world_size)
    else:
        avg_energy_j = None
    
    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / args.iters
    
    # Calculate bandwidth
    bytes_per_op = total_elements * bytes_per_elem * 2 * (world_size - 1) / world_size
    avg_bandwidth_gbps = (bytes_per_op / (1024**3)) / (avg_latency_ms / 1000)
    
    # Algorithm bandwidth
    algo_bytes = total_elements * bytes_per_elem
    algo_bandwidth_gbps = (algo_bytes / (1024**3)) / (avg_latency_ms / 1000)
    
    if rank == 0:
        print("=" * 80)
        print("MSCCL Results")
        print("=" * 80)
        print(f"Total time:          {total_time_ms:.3f} ms")
        print(f"Iterations:          {args.iters}")
        print(f"Average latency:     {format_latency(avg_latency_ms)}")
        print(f"Algorithm BW:        {format_bandwidth(algo_bandwidth_gbps)}")
        print(f"Bus bandwidth:       {format_bandwidth(avg_bandwidth_gbps)}")
        print(f"Throughput:          {1000 / avg_latency_ms:.2f} ops/sec")
        if avg_energy_j is not None:
            print(f"Avg energy/op:       {avg_energy_j:.6f} J")
            print(f"Avg power:           {avg_energy_j / (avg_latency_ms / 1000):.2f} W")
            print(f"Energy efficiency:   {(algo_bytes / (1024**3)) / avg_energy_j:.2f} GB/J")
        print("=" * 80)
    
    return avg_latency_ms, avg_bandwidth_gbps, avg_energy_j


def run_msccl_sweep(args, rank, world_size, device, dtype):
    """Sweep over different SM counts and block sizes for MSCCL."""
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("MSCCL Parameter Sweep")
        print("=" * 80)
    
    results = []
    
    sm_range = range(args.sm_min, args.sm_max + 1, args.sm_step)
    block_sizes = args.block_sizes
    
    for sm_num in sm_range:
        for block_size in block_sizes:
            if rank == 0:
                print(f"\nTesting SM={sm_num}, BlockSize={block_size}")
            
            # Temporarily set args
            orig_sm = args.sm_num
            orig_bs = args.block_size
            orig_warmup = args.warmup
            orig_iters = args.iters
            
            args.sm_num = sm_num
            args.block_size = block_size
            args.warmup = 5  # Fewer warmup for sweep
            args.iters = 20  # Fewer iterations for sweep
            
            try:
                latency, bandwidth, energy = run_msccl_benchmark(args, rank, world_size, device, dtype)
                results.append({
                    'sm_num': sm_num,
                    'block_size': block_size,
                    'latency_ms': latency,
                    'bandwidth_gbps': bandwidth,
                    'energy_j': energy
                })
            except Exception as e:
                if rank == 0:
                    print(f"Failed with SM={sm_num}, BlockSize={block_size}: {e}")
            
            # Restore args
            args.sm_num = orig_sm
            args.block_size = orig_bs
            args.warmup = orig_warmup
            args.iters = orig_iters
            
            dist.barrier()
    
    if rank == 0 and results:
        print("\n" + "=" * 80)
        print("MSCCL Sweep Summary")
        print("=" * 80)
        if args.measure_energy and HAVE_ZEUS:
            print(f"{'SM Count':<10} {'Block Size':<12} {'Latency':<15} {'Bandwidth':<15} {'Energy':<15}")
        else:
            print(f"{'SM Count':<10} {'Block Size':<12} {'Latency':<15} {'Bandwidth':<15}")
        print("-" * 80)
        
        for result in results:
            line = (f"{result['sm_num']:<10} {result['block_size']:<12} "
                   f"{format_latency(result['latency_ms']):<15} "
                   f"{format_bandwidth(result['bandwidth_gbps']):<15}")
            if args.measure_energy and HAVE_ZEUS and result.get('energy_j') is not None:
                line += f"{result['energy_j']:.6f} J"
            print(line)
        
        # Find best configuration
        best = min(results, key=lambda x: x['latency_ms'])
        print("-" * 80)
        print(f"Best configuration: SM={best['sm_num']}, BlockSize={best['block_size']}")
        print(f"  Latency:   {format_latency(best['latency_ms'])}")
        print(f"  Bandwidth: {format_bandwidth(best['bandwidth_gbps'])}")
        if args.measure_energy and HAVE_ZEUS and best.get('energy_j') is not None:
            print(f"  Energy:    {best['energy_j']:.6f} J")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark AllReduce operation performance")
    
    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        default="both",
        choices=["nccl", "msccl", "both"],
        help="Backend to benchmark (default: both)",
    )
    
    # Tensor configuration
    parser.add_argument(
        "--tensor-size-mb",
        type=float,
        default=8.0,
        help="Tensor size in MB per rank (default: 8.0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type (default: float16)",
    )
    
    # Benchmark parameters
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of test iterations (default: 50)",
    )
    parser.add_argument(
        "--measure-energy",
        action="store_true",
        help="Measure energy consumption using Zeus monitor",
    )
    
    # NCCL-specific
    parser.add_argument(
        "--async-op",
        action="store_true",
        default=True,
        help="Use async all-reduce for NCCL (default: True)",
    )
    
    # MSCCL-specific
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
    
    # MSCCL sweep parameters
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep over different SM counts and block sizes for MSCCL",
    )
    parser.add_argument(
        "--sm-min",
        type=int,
        default=1,
        help="Minimum SM count for sweep (default: 1)",
    )
    parser.add_argument(
        "--sm-max",
        type=int,
        default=20,
        help="Maximum SM count for sweep (default: 20)",
    )
    parser.add_argument(
        "--sm-step",
        type=int,
        default=1,
        help="SM count step for sweep (default: 1)",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[512, 1024],
        help="Block sizes to test in sweep (default: 512 1024)",
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
        print("AllReduce Performance Benchmark")
        print("=" * 80)
        print(f"World size:       {world_size}")
        print(f"Backend:          {args.backend}")
        print("=" * 80)
    
    # Run benchmarks
    if args.sweep and (args.backend == "msccl" or args.backend == "both"):
        run_msccl_sweep(args, rank, world_size, device, dtype)
    else:
        if args.backend == "nccl" or args.backend == "both":
            run_nccl_benchmark(args, rank, world_size, device, dtype)
        
        if args.backend == "msccl" or args.backend == "both":
            run_msccl_benchmark(args, rank, world_size, device, dtype)
    
    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    
    if rank == 0:
        print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()

