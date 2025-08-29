# make sure you have already run `pip install -e .`
import torch
import cfuser
import cfuser.msccl_comm
import torch.distributed as dist
from typing import List


def test_msccl_AlltoAll(rank, world_size, size=1024):
    # Create a random tensor
    tensor = torch.randn(size, dtype=torch.float16, device="cuda").contiguous()
    tensor_output = torch.randn(size, dtype=torch.float16, device="cuda").contiguous()

    stream = torch.cuda.Stream()

    # Perform the AlltoAll operation
    cfuser.msccl_comm.msccl_AlltoAll(
        input=tensor,
        output=tensor_output,
        # stream=stream,
        sm_num=world_size - 1,
        block_size=512,
        nranks=world_size,
        rank=rank,
    )


def test_msccl_AlltoAllv(
    rank,
    world_size,
    size=1024,
    ranks_mlp: List[int] = None,
    ranks_attn: List[int] = None,
    ranks_ulysses: List[int] = None,
    ranks_ring: List[int] = None,
    stream: torch.cuda.Stream = None,
):
    # Create a random tensor
    tensor = torch.randn((world_size, size), dtype=torch.float16, device="cuda").contiguous()
    # if rank in ranks_attn:
    #     tensor_output = torch.randn(size * len(ranks_mlp) // len(ranks_attn), dtype=torch.float16, device="cuda").contiguous()
    # else:
    #     tensor_output = torch.randn(1, dtype=torch.float16, device="cuda").contiguous()

    tensor_output = torch.randn(
        (len(ranks_mlp) // len(ranks_attn) * world_size, size), dtype=torch.float16, device="cuda"
    ).contiguous()

    cfuser.msccl_comm.msccl_AlltoAllv(
        input=tensor,
        output=tensor_output,
        current_rank=rank,
        ranks_mlp=ranks_mlp,
        ranks_attn=ranks_attn,
        ranks_ulysses=ranks_ulysses,
        ranks_ring=ranks_ring,
        # stream=stream,
        sm_num=world_size - 1,
        block_size=512,
        nranks=world_size,
        scatter_idx=2,
        gather_idx=1,
    )


def main(rank, world_size):
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12345", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    cfuser.msccl_comm.msccl_comm_init(rank, world_size)

    test_msccl_AlltoAll(rank, world_size)
    print(f"Rank {rank} test_msccl_AlltoAll could run")

    dist.barrier()

    ranks_mlp = list(range(world_size))
    ranks_attn = list(range(world_size // 2))
    ranks_ulysses = list(range(world_size // 2))
    ranks_ring = [rank]
    test_msccl_AlltoAllv(
        rank, world_size, ranks_mlp=ranks_mlp, ranks_attn=ranks_attn, ranks_ulysses=ranks_ulysses, ranks_ring=ranks_ring
    )
    print(f"Rank {rank} test_msccl_AlltoAllv could run")


if __name__ == "__main__":
    from torch.multiprocessing import spawn

    spawn(main, args=(4,), nprocs=4)
