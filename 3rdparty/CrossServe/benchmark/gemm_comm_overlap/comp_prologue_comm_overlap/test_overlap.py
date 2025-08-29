import os
import torch
import random
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)
from cfuser.core.distributed import (
    get_runtime_state,
    get_world_group,
    init_distributed_environment,
)
from cfuser.model_executor.models.transformers.transformer_flux import (
    multimodal_comp_prologue,
    unimodal_comp_prologue,
    multimodal_comp_epilogue,
    unimodal_comp_epilogue,
)
import cfuser.msccl_comm
from cfuser.core.long_ctx_attention.comm import all_to_all_4D
import torch.distributed as dist
from cfuser.core.utils import nvtx_range


def overlap_test(rank, world_size, args, master_port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    init_distributed_environment(rank=rank, local_rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    cfuser.msccl_comm.msccl_comm_init(rank, world_size)

    dtype = torch.bfloat16

    num_attention_heads = 24
    attention_head_dim = 128
    inner_dim = num_attention_heads * attention_head_dim

    from cfuser.core.distributed.parallel_state import initialize_model_parallel

    initialize_model_parallel(
        sequence_parallel_degree=args.parallel_degree,
        ulysses_degree=args.parallel_degree,
        ring_degree=1,
        backend="nccl",
    )

    block = (
        FluxTransformerBlock(
            dim=inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )
        .to(dtype)
        .to(f"cuda:{rank}")
    )

    seq_len = args.seq_len // args.parallel_degree

    text_seq_len = 256

    hidden_states = torch.randn(args.batch_size, seq_len, inner_dim).to(dtype).to(f"cuda:{rank}").contiguous()
    encoder_hidden_states = (
        torch.randn(args.batch_size, text_seq_len, inner_dim).to(dtype).to(f"cuda:{rank}").contiguous()
    )
    time_embd = torch.randn(args.batch_size, inner_dim).to(dtype).to(f"cuda:{rank}").contiguous()
    image_rotary_emb = (
        torch.randn(seq_len + text_seq_len, attention_head_dim).to(dtype).to(f"cuda:{rank}").contiguous(),
        torch.randn(seq_len + text_seq_len, attention_head_dim).to(dtype).to(f"cuda:{rank}").contiguous(),
    )
    fake_hidden_states = (
        torch.randn(args.batch_size, seq_len, num_attention_heads, attention_head_dim)
        .to(dtype)
        .to(f"cuda:{rank}")
        .contiguous()
    )
    fake_output = torch.empty_like(hidden_states).contiguous()

    ranks_mlp = list(range(world_size))
    ranks_attn = list(range(world_size))
    ranks_ulysses = list(range(world_size))
    ranks_ring = [rank]

    for i in range(args.warmup_steps):
        multimodal_comp_prologue(block, hidden_states, encoder_hidden_states, time_embd, image_rotary_emb)
        cfuser.msccl_comm.msccl_AlltoAllv(
            input=hidden_states,
            output=fake_output,
            current_rank=rank,
            ranks_mlp=ranks_mlp,
            ranks_attn=ranks_attn,
            ranks_ulysses=ranks_ulysses,
            ranks_ring=ranks_ring,
            sm_num=args.parallel_degree - 1,
            block_size=512,
            nranks=world_size,
            scatter_idx=2,
            gather_idx=1,
        )

    cfuser.msccl_comm.msccl_sync()
    torch.cuda.synchronize()

    dist.barrier()

    torch.cuda.cudart().cudaProfilerStart()

    current_stream = torch.cuda.current_stream()

    for i in range(args.repeat):
        with nvtx_range("multimodal_comp_prologue all2all by torch"):
            all_to_all_4D(fake_hidden_states, 2, 1, group=dist.group.WORLD, use_sync=True, async_op=False)
            all_to_all_4D(fake_hidden_states, 2, 1, group=dist.group.WORLD, use_sync=True, async_op=False)
            all_to_all_4D(fake_hidden_states, 2, 1, group=dist.group.WORLD, use_sync=True, async_op=False)
            with torch.cuda.stream(current_stream):
                multimodal_comp_prologue(block, hidden_states, encoder_hidden_states, time_embd, image_rotary_emb)

    torch.cuda.synchronize()
    dist.barrier()

    for i in range(args.repeat):
        with nvtx_range("multimodal_comp_prologue all2all sequential"):
            with nvtx_range("msccl_AlltoAllv"):
                cfuser.msccl_comm.msccl_AlltoAllv(
                    input=hidden_states,
                    output=fake_output,
                    current_rank=rank,
                    ranks_mlp=ranks_mlp,
                    ranks_attn=ranks_attn,
                    ranks_ulysses=ranks_ulysses,
                    ranks_ring=ranks_ring,
                    sm_num=args.parallel_degree - 1,
                    block_size=512,
                    nranks=world_size,
                    scatter_idx=2,
                    gather_idx=1,
                )
                cfuser.msccl_comm.msccl_AlltoAllv(
                    input=hidden_states,
                    output=fake_output,
                    current_rank=rank,
                    ranks_mlp=ranks_mlp,
                    ranks_attn=ranks_attn,
                    ranks_ulysses=ranks_ulysses,
                    ranks_ring=ranks_ring,
                    sm_num=args.parallel_degree - 1,
                    block_size=512,
                    nranks=world_size,
                    scatter_idx=2,
                    gather_idx=1,
                )
                cfuser.msccl_comm.msccl_AlltoAllv(
                    input=hidden_states,
                    output=fake_output,
                    current_rank=rank,
                    ranks_mlp=ranks_mlp,
                    ranks_attn=ranks_attn,
                    ranks_ulysses=ranks_ulysses,
                    ranks_ring=ranks_ring,
                    sm_num=args.parallel_degree - 1,
                    block_size=512,
                    nranks=world_size,
                    scatter_idx=2,
                    gather_idx=1,
                )

                cfuser.msccl_comm.msccl_sync()

            with nvtx_range("multimodal_comp_prologue"):
                multimodal_comp_prologue(block, hidden_states, encoder_hidden_states, time_embd, image_rotary_emb)

                current_stream.synchronize()

    torch.cuda.synchronize()
    dist.barrier()

    for i in range(args.repeat):
        with nvtx_range("multimodal_comp_prologue all2all concurrent"):
            with nvtx_range("msccl_AlltoAllv"):
                cfuser.msccl_comm.msccl_AlltoAllv(
                    input=hidden_states,
                    output=fake_output,
                    current_rank=rank,
                    ranks_mlp=ranks_mlp,
                    ranks_attn=ranks_attn,
                    ranks_ulysses=ranks_ulysses,
                    ranks_ring=ranks_ring,
                    sm_num=args.parallel_degree - 1,
                    block_size=512,
                    nranks=world_size,
                    scatter_idx=2,
                    gather_idx=1,
                )
                cfuser.msccl_comm.msccl_AlltoAllv(
                    input=hidden_states,
                    output=fake_output,
                    current_rank=rank,
                    ranks_mlp=ranks_mlp,
                    ranks_attn=ranks_attn,
                    ranks_ulysses=ranks_ulysses,
                    ranks_ring=ranks_ring,
                    sm_num=args.parallel_degree - 1,
                    block_size=512,
                    nranks=world_size,
                    scatter_idx=2,
                    gather_idx=1,
                )
                cfuser.msccl_comm.msccl_AlltoAllv(
                    input=hidden_states,
                    output=fake_output,
                    current_rank=rank,
                    ranks_mlp=ranks_mlp,
                    ranks_attn=ranks_attn,
                    ranks_ulysses=ranks_ulysses,
                    ranks_ring=ranks_ring,
                    sm_num=args.parallel_degree - 1,
                    block_size=512,
                    nranks=world_size,
                    scatter_idx=2,
                    gather_idx=1,
                )

            with nvtx_range("multimodal_comp_prologue"):
                multimodal_comp_prologue(block, hidden_states, encoder_hidden_states, time_embd, image_rotary_emb)

            cfuser.msccl_comm.msccl_sync()
            current_stream.synchronize()

    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.cudart().cudaProfilerStop()

    print(f"Rank {rank} done")


"""
export nsys_args="--force-overwrite true -w true --capture-range=cudaProfilerApi"
nsys profile ${nsys_args} -o log/benchmark/gemm_comm_overlap/comp_prologue_comm_overlap/batch_4_parallel_degree_4_seq_len_8192_warmup_steps_30_repeat_50 python benchmark/gemm_comm_overlap/comp_prologue_comm_overlap/test_overlap.py -b 4 -p 4 -s 8192 -w 30 -r 50

python benchmark/gemm_comm_overlap/comp_prologue_comm_overlap/test_overlap.py -b 4 -p 4 -s 8192 -w 30 -r 50
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--parallel_degree", "-p", type=int, default=1)
    parser.add_argument("--seq_len", "-s", type=int, default=1024)
    parser.add_argument("--warmup_steps", "-w", type=int, default=2)
    parser.add_argument("--repeat", "-r", type=int, default=3)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()

    from torch.multiprocessing import spawn

    nprocs = args.parallel_degree
    spawn(
        overlap_test,
        args=(
            nprocs,
            args,
            random.randint(8000, 65535),
        ),
        nprocs=nprocs,
    )
