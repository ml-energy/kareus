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
    test_tensors = tester.create_test_tensors()
    operations = tester.create_operations(test_tensors[-1])

    comp_ops = operations[:-1]
    allreduce_comm_op = operations[-1]
    
    mlp_fuser = PartitionFuser(
        ops=comp_ops,
        allreduce_comm_op=allreduce_comm_op,
        fuse_ops=False,
    )

    # Warmup passes
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(10):
        mlp_fuser(
            hidden_states=test_tensors[0],
            bias=test_tensors[1],
            residual=test_tensors[2],
            allreduce_input=test_tensors[3],
            allreduce_overlap_window=(args.overlap_start, args.overlap_end),
            allreduce_sm_configs=(args.sm_num, args.block_size),
        )

    cudart.cudaProfilerStart()
    mlp_fuser(
        hidden_states=test_tensors[0],
        bias=test_tensors[1],
        residual=test_tensors[2],
        allreduce_input=test_tensors[3],
        allreduce_overlap_window=(args.overlap_start, args.overlap_end),
        allreduce_sm_configs=(args.sm_num, args.block_size),
    )
    cudart.cudaProfilerStop()

    torch.cuda.synchronize()
    dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--frequency", "-f", type=str, default="default")
    parser.add_argument("--overlap_start", type=int, default=0)
    parser.add_argument("--overlap_end", type=int, default=1)
    parser.add_argument("--sm_num", "-n", type=int, default=1)
    parser.add_argument("--block_size", "-t", type=int, default=1024)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None and len(visible.strip()) > 0:
        vis_list = [int(x) for x in visible.split(",") if x.strip() != ""]
        target_indices = vis_list
    else:
        raise ValueError("CUDA_VISIBLE_DEVICES is not set")

    from torch.multiprocessing import spawn
    spawn(
        _spawn_entry,
        args=(args.world_size, args, 9000 + target_indices[0]),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
