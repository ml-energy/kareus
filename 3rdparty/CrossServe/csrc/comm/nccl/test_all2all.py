from cfuser.core.long_ctx_attention.pynccl_wrapper import NCCLLibrary, PyNcclCommunicator
import torch.distributed as dist
import torch


def main(rank, world_size):

    dist.init_process_group(backend="nccl", init_method="tcp://localhost:12345", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    nccl = PyNcclCommunicator(
        group=dist.group.WORLD,
        device=f"cuda:{rank}",
        library_path="/usr/lib/x86_64-linux-gnu/libnccl.so.2",
        custom_nccl_library_path="/workspaces/CrossServe/csrc/comm/build/libcustom_nccl_all2all.so",
    )

    bs = 1
    seq_len = 2048
    head_num = 24
    head_size = 128
    input_tensor = torch.randn(bs, seq_len // world_size, head_num, head_size, device=f"cuda:{rank}")

    input_tensor = (
        input_tensor.view(bs, seq_len // world_size, world_size, head_num // world_size, head_size)
        .transpose(0, 2)
        .contiguous()
    )

    # test all2all
    output_tensor = torch.empty_like(input_tensor, device=f"cuda:{rank}", dtype=input_tensor.dtype)
    output_tensor_1 = torch.empty_like(input_tensor, device=f"cuda:{rank}", dtype=input_tensor.dtype)
    nccl.all_to_all(input_tensor, output_tensor)
    dist.all_to_all_single(output_tensor_1, input_tensor)

    torch.cuda.synchronize()

    # test all2all single
    input_split_sizes = [2] * (world_size // 2) + [0] * (world_size // 2)
    output_split_sizes = [0] * world_size
    if rank < world_size // 2:
        output_split_sizes = [2] * world_size

    if rank < world_size // 2:
        output_tensor = torch.empty(
            world_size * 2, seq_len // world_size, bs, head_num // world_size, head_size, device=f"cuda:{rank}"
        ).contiguous()
    else:
        output_tensor = torch.empty(
            [0, seq_len // world_size, bs, head_num, head_size], device=f"cuda:{rank}", dtype=input_tensor.dtype
        ).contiguous()

    output_tensor_1 = torch.empty(output_tensor.shape, device=f"cuda:{rank}", dtype=input_tensor.dtype).contiguous()

    nccl.all_to_all_single(input_tensor, output_tensor, input_split_sizes, output_split_sizes)

    dist.all_to_all_single(
        output_tensor_1,
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=dist.group.WORLD,
    )
    torch.cuda.synchronize()

    assert torch.allclose(output_tensor, output_tensor_1)


if __name__ == "__main__":
    from torch.multiprocessing import spawn

    world_size = 4
    spawn(main, args=(world_size,), nprocs=world_size)
