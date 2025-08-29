import torch
import os
import torch.distributed as dist
from cfuser.core.long_ctx_attention.comm import custom_all_to_all_single_python


def test_all_to_all_single(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    if dist.get_rank() <= 1:
        group = dist.new_group([0, 1])
        seq_len = 2304
        hc = 24
        hdim = 128
        input = (
            torch.ones([2, hc, seq_len, hdim], device="cuda")
            if dist.get_rank() == 0
            else torch.tensor([], device="cuda")
        )
        input_split_sizes = [1, 1] if dist.get_rank() == 0 else [0, 0]
        output_split_sizes = [1, 0]
        print(
            f"rank {dist.get_rank()} input: {input.shape}, input_split_sizes: {input_split_sizes}, output_split_sizes: {output_split_sizes}"
        )

        output = torch.empty([1, hc, seq_len, hdim], device="cuda")
        custom_all_to_all_single_python(
            output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group
        )
        dist.barrier(group)
        print(f"rank {dist.get_rank()} output: {output.shape}")
    dist.destroy_process_group()


def mp():
    world_size = 4
    torch.multiprocessing.spawn(test_all_to_all_single, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"
    mp()
