import os
import json
import time
import random
import torch
import torch.distributed as dist
from cfuser.core.distributed import (
    get_runtime_state,
    get_world_group,
    init_distributed_environment,
)
from cfuser import cFuserFluxPipeline

from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from cfuser.core.utils.utils import nvtx_range

from cfuser.model_executor.models.transformers.transformer_flux import (
    multimodal_comp_prologue,
    unimodal_comp_prologue,
    multimodal_comp_epilogue,
    unimodal_comp_epilogue,
)


def comp_multimodal(
    block,
    hidden_states,
    encoder_hidden_states,
    time_embd,
    image_rotary_emb,
    seq_len,
    text_seq_len,
    inner_dim,
    dtype,
    rank,
    args,
):

    if dist.get_world_size() > 1:
        dist.barrier()

    with nvtx_range("multimodal_comp_prologue"):
        time_start = torch.cuda.Event(enable_timing=True)
        time_start.record()
        # time_s = time.perf_counter()
        for i in range(args.repeat):
            (
                query,
                key,
                value,
                joint_tensor_key,
                joint_tensor_value,
                head_dim,
                gate_msa,
                scale_mlp,
                shift_mlp,
                gate_mlp,
                c_gate_mlp,
                c_gate_msa,
                c_scale_mlp,
                c_shift_mlp,
            ) = multimodal_comp_prologue(block, hidden_states, encoder_hidden_states, time_embd, image_rotary_emb)
        time_end = torch.cuda.Event(enable_timing=True)
        time_end.record()
        # torch.cuda.synchronize(device=f"cuda:{rank}")
        time_end.synchronize()
        # time_e = time.perf_counter()
        # print(f"time: {time_e - time_s}")
        duration = time_start.elapsed_time(time_end)
        print(f"duration: {duration}ms")
        if dist.get_world_size() > 1:
            dist.barrier()
        multimodal_comp_prologue_duration = duration / 1000

    output = torch.randn(args.batch_size, seq_len + text_seq_len, inner_dim).to(dtype).to(f"cuda:{rank}")

    del query, key, value, joint_tensor_key, joint_tensor_value

    if dist.get_world_size() > 1:
        dist.barrier()

    with nvtx_range("multimodal_comp_epilogue"):
        time_begin = torch.cuda.Event(enable_timing=True)
        time_begin.record()
        for i in range(30):
            encoder_hidden_states, hidden_states = multimodal_comp_epilogue(
                block,
                output,
                hidden_states,
                encoder_hidden_states,
                head_dim,
                gate_msa,
                scale_mlp,
                shift_mlp,
                gate_mlp,
                c_gate_mlp,
                c_gate_msa,
                c_scale_mlp,
                c_shift_mlp,
                dtype=hidden_states.dtype,
            )
        time_end = torch.cuda.Event(enable_timing=True)
        time_end.record()
        torch.cuda.synchronize(device=f"cuda:{rank}")
        duration = time_begin.elapsed_time(time_end)
        if dist.get_world_size() > 1:
            dist.barrier()
        multimodal_comp_epilogue_duration = duration / 30 * args.repeat / 1000

    return multimodal_comp_prologue_duration, multimodal_comp_epilogue_duration


def comp_unimodal(
    block,
    hidden_states,
    time_embd,
    image_rotary_emb,
    seq_len,
    text_seq_len,
    num_attention_heads,
    attention_head_dim,
    dtype,
    rank,
    args,
):

    if dist.get_world_size() > 1:
        dist.barrier()
    with nvtx_range("unimodal_comp_prologue"):
        time_begin = torch.cuda.Event(enable_timing=True)
        time_begin.record()
        for i in range(args.repeat):
            (
                query,
                key,
                value,
                joint_tensor_key,
                joint_tensor_value,
                head_dim,
                batch_size,
                residual,
                gate,
                mlp_hidden_states,
            ) = unimodal_comp_prologue(
                block=block,
                hidden_states=hidden_states,
                time_embd=time_embd,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=None,
                index_req=0,
            )
        time_end = torch.cuda.Event(enable_timing=True)
        time_end.record()
        torch.cuda.synchronize(device=f"cuda:{rank}")
        duration = time_begin.elapsed_time(time_end)
        unimodal_comp_prologue_duration = duration / 1000

    output = (
        torch.randn(args.batch_size, seq_len + text_seq_len, num_attention_heads, attention_head_dim)
        .to(dtype)
        .to(f"cuda:{rank}")
    )

    if dist.get_world_size() > 1:
        dist.barrier()
    with nvtx_range("unimodal_comp_epilogue"):
        time_begin = torch.cuda.Event(enable_timing=True)
        time_begin.record()
        for i in range(args.repeat):
            hidden_states = unimodal_comp_epilogue(
                block,
                attn_output=output,
                residual=residual,
                gate=gate,
                mlp_hidden_states=mlp_hidden_states,
                head_dim=head_dim,
                dtype=residual.dtype,
            )
        time_end = torch.cuda.Event(enable_timing=True)
        time_end.record()
        torch.cuda.synchronize(device=f"cuda:{rank}")
        duration = time_begin.elapsed_time(time_end)
        unimodal_comp_epilogue_duration = duration / 1000
    if dist.get_world_size() > 1:
        dist.barrier()
    return unimodal_comp_prologue_duration, unimodal_comp_epilogue_duration


def test_non_attn(rank, world_size, args, master_port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"
    json_path = f"log/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json"
    if args.logging:
        # check if entry exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    if (
                        entry["bs"] == args.batch_size
                        and entry["seq_len"] == args.seq_len
                        and entry["hc"] == 24
                        and entry["hs"] == 128
                        and entry["ulysses_world_size"] == args.parallel_degree
                        and entry["ring_attn_world_size"] == 1
                    ):
                        if rank == 0:
                            print(
                                f"Entry already exists for bs {args.batch_size}, seq_len {args.seq_len}, hc 24, hs 128, ulysses_world_size {args.parallel_degree}, ring_attn_world_size 1"
                                + f"avg time for multimodal prologue: {entry['avg_time_multimodal_prologue']:.4f} s multimodal prologue time: {entry['time_multimodal_prologue']:.4f}s, avg time for unimodal prologue: {entry['avg_time_unimodal_prologue']:.4f} s unimodal prologue time: {entry['time_unimodal_prologue']:.4f}s, avg time for multimodal epilogue: {entry['avg_time_multimodal_epilogue']:.4f} s multimodal epilogue time: {entry['time_multimodal_epilogue']:.4f}s, avg time for unimodal epilogue: {entry['avg_time_unimodal_epilogue']:.4f} s unimodal epilogue time: {entry['time_unimodal_epilogue']:.4f}s"
                            )
                        return

    init_distributed_environment(rank=rank, local_rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
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

    single_block = (
        FluxSingleTransformerBlock(
            dim=inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )
        .to(dtype)
        .to(f"cuda:{rank}")
    )

    seq_len = args.seq_len // args.parallel_degree

    text_seq_len = 256

    hidden_states = torch.randn(args.batch_size, seq_len, inner_dim).to(dtype).to(f"cuda:{rank}")
    encoder_hidden_states = torch.randn(args.batch_size, text_seq_len, inner_dim).to(dtype).to(f"cuda:{rank}")
    time_embd = torch.randn(args.batch_size, inner_dim).to(dtype).to(f"cuda:{rank}")
    image_rotary_emb = (
        torch.randn(seq_len + text_seq_len, attention_head_dim).to(dtype).to(f"cuda:{rank}"),
        torch.randn(seq_len + text_seq_len, attention_head_dim).to(dtype).to(f"cuda:{rank}"),
    )

    for i in range(args.warmup_steps):
        multimodal_comp_prologue(block, hidden_states, encoder_hidden_states, time_embd, image_rotary_emb)
        hidden_states_ = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        unimodal_comp_prologue(
            single_block, hidden_states_, time_embd, image_rotary_emb, joint_attention_kwargs=None, index_req=0
        )
        del hidden_states_

    torch.cuda.empty_cache()

    torch.cuda.cudart().cudaProfilerStart()

    multimodal_comp_prologue_duration, multimodal_comp_epilogue_duration = comp_multimodal(
        block,
        hidden_states,
        encoder_hidden_states,
        time_embd,
        image_rotary_emb,
        seq_len,
        text_seq_len,
        inner_dim,
        dtype,
        rank,
        args,
    )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    unimodal_comp_prologue_duration, unimodal_comp_epilogue_duration = comp_unimodal(
        single_block,
        hidden_states,
        time_embd,
        image_rotary_emb,
        seq_len,
        text_seq_len,
        num_attention_heads,
        attention_head_dim,
        dtype,
        rank,
        args,
    )

    torch.cuda.cudart().cudaProfilerStop()

    if rank == 0:
        print(
            f"bs {args.batch_size} seqlen {args.seq_len} hc {num_attention_heads} hs {attention_head_dim} ulysses_world_size {args.parallel_degree} ring_attn_world_size {1} avg time for multimodal prologue: {multimodal_comp_prologue_duration / args.repeat:.4f} s multimodal prologue time: {multimodal_comp_prologue_duration:.4f}s, avg time for unimodal prologue: {unimodal_comp_prologue_duration / args.repeat:.4f} s unimodal prologue time: {unimodal_comp_prologue_duration:.4f}s, avg time for multimodal epilogue: {multimodal_comp_epilogue_duration / args.repeat:.4f} s multimodal epilogue time: {multimodal_comp_epilogue_duration:.4f}s, avg time for unimodal epilogue: {unimodal_comp_epilogue_duration / args.repeat:.4f} s unimodal epilogue time: {unimodal_comp_epilogue_duration:.4f}s"
        )
        if args.logging:

            json_path = f"log/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json"

            to_save = {
                "bs": args.batch_size,
                "seq_len": args.seq_len,
                "hc": num_attention_heads,
                "hs": attention_head_dim,
                "ulysses_world_size": args.parallel_degree,
                "ring_attn_world_size": 1,
                "num_iter": args.repeat,
                "avg_time_multimodal_prologue": multimodal_comp_prologue_duration / args.repeat,
                "time_multimodal_prologue": multimodal_comp_prologue_duration,
                "avg_time_unimodal_prologue": unimodal_comp_prologue_duration / args.repeat,
                "time_unimodal_prologue": unimodal_comp_prologue_duration,
                "avg_time_multimodal_epilogue": multimodal_comp_epilogue_duration / args.repeat,
                "time_multimodal_epilogue": multimodal_comp_epilogue_duration,
                "avg_time_unimodal_epilogue": unimodal_comp_epilogue_duration / args.repeat,
                "time_unimodal_epilogue": unimodal_comp_epilogue_duration,
            }

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            # Load existing data
            existing_data = []
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    existing_data = json.load(f)

            # Append new data
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
            existing_data.append(to_save)

            # Save updated data
            with open(json_path, "w") as f:
                json.dump(existing_data, f, indent=2)

    dist.destroy_process_group()


"""
# without nsys
python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py --batch_size 2 --seq_len 4096 --parallel_degree 1 --warmup_steps 80 --repeat 40
python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py --batch_size 2 --seq_len 4096 --parallel_degree 2 --warmup_steps 80 --repeat 40
python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py --batch_size 2 --seq_len 4096 --parallel_degree 4 --warmup_steps 80 --repeat 40

# with nsys
export nsys_args="--force-overwrite true -w true -s cpu --python-backtrace=cuda --cudabacktrace=all --capture-range=cudaProfilerApi"
export nsys_args="--force-overwrite true -w true --capture-range=cudaProfilerApi"
CUDA_VISIBLE_DEVICES=0 nsys profile ${nsys_args} -o log/benchmark/component_scaling_efficiency/non_attn_efficiency/batch_2_seq_len_4096_parallel_degree_1 python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py --batch_size 2 --seq_len 4096 --parallel_degree 1 --warmup_steps 80 --repeat 40
nsys profile ${nsys_args} -o log/benchmark/component_scaling_efficiency/non_attn_efficiency/batch_2_seq_len_4096_parallel_degree_2 python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py --batch_size 2 --seq_len 4096 --parallel_degree 2 --warmup_steps 80 --repeat 40
CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/benchmark/component_scaling_efficiency/non_attn_efficiency/batch_2_seq_len_4096_parallel_degree_4 python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py --batch_size 2 --seq_len 4096 --parallel_degree 4 --warmup_steps 80 --repeat 40
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--parallel_degree", "-p", type=int, default=1)
    parser.add_argument("--seq_len", "-s", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()

    from torch.multiprocessing import spawn

    nprocs = args.parallel_degree
    spawn(
        test_non_attn,
        args=(
            nprocs,
            args,
            random.randint(8000, 65535),
        ),
        nprocs=nprocs,
    )
