import os
import random
import time
import argparse
import torch
import torch.distributed as dist
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from overlap_test_mlp import MLPFuserTest
from kareus.megatron.core.extensions.fusers.partition_fuser import PartitionFuser
from cuda import cudart


def init_env(rank, world_size, master_port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"


def _spawn_entry(rank, world_size, args, master_port):
    init_env(rank, world_size, master_port)

    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    tester = MLPFuserTest(args, rank, world_size)
    # Build tensors and ops once
    (
        hidden_states,
        bias,
        residual,
        allreduce_inputs,
    ) = tester.create_test_tensors()
    
    operations = tester.create_operations(allreduce_inputs)
    comp_ops = operations[:-1]
    allreduce_comm_op = operations[-1]
    
    mlp_fuser = PartitionFuser(
        ops=comp_ops,
        comm_op_bwd=allreduce_comm_op,
        fuse_ops=False,
    )

    # Create gradient tensors for backward pass
    nano_batch_size = tester.batch_size // 2
    output_grad = torch.randn(
        tester.seq_length, nano_batch_size, tester.hidden_size,
        dtype=tester.dtype, device=tester.device
    )
    residual_grad = torch.randn(
        tester.seq_length, nano_batch_size, tester.hidden_size,
        dtype=tester.dtype, device=tester.device
    )
    allreduce_input_grad = torch.randn(
        tester.seq_length, nano_batch_size, tester.hidden_size,
        dtype=tester.dtype, device=tester.device
    )

    overlap_window = (args.overlap_start, args.overlap_end)
    sm_configs = (args.sm_num, args.block_size)

    # Forward pass to get outputs (needed for backward)
    output, output_bias, output_residual, allreduce_output = mlp_fuser(
        hidden_states=hidden_states,
        bias=bias,
        residual=residual,
        comm_input=allreduce_inputs,
        comm_overlap_window_backward=overlap_window,
        comm_sm_configs_backward=sm_configs,
    )

    def run_backward():
        _ = torch.autograd.grad(
            outputs=[output, output_residual, allreduce_output],
            inputs=[hidden_states, residual, allreduce_inputs],
            grad_outputs=[output_grad, residual_grad, allreduce_input_grad],
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )

    # Warmup passes
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(10):
        run_backward()

    cudart.cudaProfilerStart()
    run_backward()
    cudart.cudaProfilerStop()

    torch.cuda.synchronize()
    dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=4)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--frequency", "-f", type=str, default="default")
    parser.add_argument("--overlap_start", type=int, default=0)
    parser.add_argument("--overlap_end", type=int, default=1)
    parser.add_argument("--sm_num", "-n", type=int, default=1)
    parser.add_argument("--block_size", "-t", type=int, default=1024)
    args = parser.parse_args()

    from torch.multiprocessing import spawn
    spawn(
        _spawn_entry,
        args=(args.world_size, args, 9003),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

