import os
import torch
import torch.distributed as dist
from cfuser.core.long_ctx_attention.ring.ring_flash_attn import (
    ring_flash_attn_func,
)


def benchmark(
    num_iter=100, forward_only=True, log=True, bs=1, seq_len=1024, ulysses_world_size=1, ring_attn_world_size=1
):
    json_path = "log/benchmark/component_scaling_efficiency/ring_scaling_efficiency/ring_scaling_efficiency.json"
    # check if log exists
    if log and os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = f.read()
        for entry in data:
            if (
                entry["bs"] == bs
                and entry["seq_len"] == seq_len
                and entry["hc"] == num_heads
                and entry["hs"] == head_dim
                and entry["ulysses_world_size"] == ulysses_world_size
                and entry["ring_attn_world_size"] == ring_attn_world_size
            ):
                if rank == 0:
                    print(
                        f"Entry already exists for bs {bs}, seq_len {seq_len}, hc {num_heads}, hs {head_dim}, ulysses_world_size {ulysses_world_size}, ring_attn_world_size {ring_attn_world_size}. "
                        + f"avg time per iter: {entry['avg_time']:.4f} s"
                    )
                return

    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == ring_attn_world_size
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    batch_size = bs
    deterministic = False
    # config of flux.dev 0.1
    seqlen = seq_len
    num_heads = 24 // ulysses_world_size
    num_kv_heads = 24 // ulysses_world_size
    head_dim = 128
    causal = False

    assert seqlen % (2 * world_size) == 0
    assert head_dim % 8 == 0

    q = torch.randn(
        batch_size,
        seqlen // world_size,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    ).contiguous()
    k = torch.randn(
        batch_size,
        seqlen // world_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    ).contiguous()
    v = torch.randn(
        batch_size,
        seqlen // world_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=False,
    ).contiguous()

    text_seq_len = 256

    joint_tensor_query = torch.randn(
        batch_size,
        text_seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    joint_tensor_key = torch.randn(
        batch_size,
        text_seq_len,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    joint_tensor_value = torch.randn(
        batch_size,
        text_seq_len,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )

    q = torch.cat([joint_tensor_query, q], dim=1)

    # warmup
    for _ in range(80):
        _ = ring_flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=False,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy="front",
        )

    if dist.get_world_size() >= 1:
        dist.barrier()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(num_iter):
        _ = ring_flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=False,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy="front",
        )
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if dist.get_world_size() >= 1:
        dist.barrier()

    if rank == 0:
        print(
            f"bs {bs} seqlen {seq_len} hc {num_heads} hs {head_dim} ulysses_world_size {ulysses_world_size} ring_attn_world_size {ring_attn_world_size} avg time per iter: {time / num_iter:.4f} s"
        )
        if log:
            import json

            json_path = (
                f"log/benchmark/component_scaling_efficiency/ring_scaling_efficiency/ring_scaling_efficiency.json"
            )

            to_save = {
                "bs": bs,
                "seq_len": seq_len,
                "hc": num_heads,
                "hs": head_dim,
                "ulysses_world_size": ulysses_world_size,
                "ring_attn_world_size": ring_attn_world_size,
                "num_iter": num_iter,
                "avg_time": time / num_iter,
                "time": time,
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
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 1037 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs 8 --seq_len 8192 --ulysses_world_size 1 --ring_attn_world_size 1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 1037 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs 2 --seq_len 4096 --ulysses_world_size 1 --ring_attn_world_size 2
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port 1037 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs 8 --seq_len 8192 --ulysses_world_size 2 --ring_attn_world_size 2
CUDA_VISIBLE_DEVICES=0,1,2,3  nsys profile --force-overwrite true -w true -s cpu -o debug_ring_attn_bs1_seq_len16384_world_size4 torchrun --nproc_per_node 4 --master_port 3097 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs 1 --seq_len 16384 --ulysses_world_size 1 --ring_attn_world_size 4
CUDA_VISIBLE_DEVICES=0,1,2,3  nsys profile --force-overwrite true -w true -s cpu -o debug_ring_attn_bs1_seq_len16384_world_size1 torchrun --nproc_per_node 1 --master_port 3097 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs 1 --seq_len 16384 --ulysses_world_size 1 --ring_attn_world_size 1
CUDA_VISIBLE_DEVICES=0,1 nsys profile --force-overwrite true -w true -s cpu -o debug_ring_attn_bs2_seq_len4096_world_size2 torchrun --nproc_per_node 2 --master_port 1037 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs 2 --seq_len 4096 --ulysses_world_size 1 --ring_attn_world_size 2
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--ulysses_world_size", type=int, default=1)
    parser.add_argument("--ring_attn_world_size", type=int, default=1)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()

    forward_only = True
    profile = False
    num_iter = 50 if forward_only else 10

    torch.cuda.empty_cache()
    benchmark(
        forward_only=forward_only,
        num_iter=num_iter,
        log=args.logging,
        bs=args.bs,
        seq_len=args.seq_len,
        ulysses_world_size=args.ulysses_world_size,
        ring_attn_world_size=args.ring_attn_world_size,
    )
