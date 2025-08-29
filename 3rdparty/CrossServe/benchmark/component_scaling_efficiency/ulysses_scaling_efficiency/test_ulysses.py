import os
import time
import torch
import torch.distributed as dist
from cfuser.core.long_ctx_attention.comm import all_to_all_4D
from cfuser.core.utils.zmq_utils import find_free_port
from torch.distributed.distributed_c10d import _get_default_group
from flash_attn import flash_attn_func
from cfuser.testing import assert_close, assert_close_with_threshold


def original_ulysses_attn(q, k, v, group=dist.group.WORLD):
    q_local_all = all_to_all_4D(q, scatter_idx=2, gather_idx=1, group=group)
    k_local_all = all_to_all_4D(k, scatter_idx=2, gather_idx=1, group=group)
    v_local_all = all_to_all_4D(v, scatter_idx=2, gather_idx=1, group=group)

    attn_output = flash_attn_func(
        q_local_all,
        k_local_all,
        v_local_all,
        causal=False,
    )

    ulysses_output = all_to_all_4D(attn_output, scatter_idx=1, gather_idx=2, group=group)

    return ulysses_output, attn_output, q_local_all, k_local_all, v_local_all


def test_ulysses_attn(
    rank, world_size, seq_len: int, batch_size: int, hc: int, hs: int, master_port: int = 1037, log: bool = False
):

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"
    json_path = "log/benchmark/component_scaling_efficiency/ulysses_scaling_efficiency/ulysses_scaling_efficiency.json"
    # Check if entry exists
    if log and os.path.exists(json_path):
        data = json.load(open(json_path, "r"))
        for entry in data:
            if (
                entry["bs"] == batch_size
                and entry["hc"] == hc
                and entry["hs"] == hs
                and entry["seq_len"] == seq_len
                and entry["ulysses_world_size"] == world_size
            ):
                if rank == 0:
                    print(
                        f"Entry already exists for bs {batch_size}, seq_len {seq_len}, world_size {world_size}."
                        + f"{repeat_times} iteration Time for ulysses attn: {entry['time']:.3f} s"
                    )

                return

    dist.init_process_group(rank=rank, world_size=world_size, backend="nccl", init_method="env://")

    repeat_times = 50

    # 0. prepare data
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % world_size}")
    Q = torch.randn(batch_size, seq_len, hc, hs, device=device, dtype=torch.bfloat16)
    K = torch.randn(batch_size, seq_len, hc, hs, device=device, dtype=torch.bfloat16)
    V = torch.randn(batch_size, seq_len, hc, hs, device=device, dtype=torch.bfloat16)

    dist.broadcast(Q, src=0)
    dist.broadcast(K, src=0)
    dist.broadcast(V, src=0)

    # warmup
    for _ in range(5):
        std_attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1, 2)
        dist.all_reduce(std_attn_output, op=dist.ReduceOp.AVG)

    q_local = Q[:, :, rank * (hc // world_size) : (rank + 1) * (hc // world_size), :].contiguous().clone()
    k_local = K[:, :, rank * (hc // world_size) : (rank + 1) * (hc // world_size), :].contiguous().clone()
    v_local = V[:, :, rank * (hc // world_size) : (rank + 1) * (hc // world_size), :].contiguous().clone()

    if rank == 0:
        print(f"bs {batch_size}, seq_len {seq_len}, world_size {world_size} running...")

    # 1. test original ulysses attn
    group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    # dist.barrier(group)
    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeat_times):
        flash_attn_func(q_local, k_local, v_local, causal=False)
        dist.barrier(group)
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if rank == 0:
        print(f"{time:.3f} sec")
        print(
            f"{repeat_times} iteration Time for ulysses attn for bs {batch_size}, seq_len {seq_len}, world_size {world_size}: {time:.3f} s"
        )
        if log:
            import json

            json_path = (
                f"log/benchmark/component_scaling_efficiency/ulysses_scaling_efficiency/ulysses_scaling_efficiency.json"
            )

            to_save = {
                "bs": batch_size,
                "seq_len": seq_len,
                "hc": hc,
                "hs": hs,
                "ulysses_world_size": world_size,
                "num_iter": repeat_times,
                "avg_time": time / repeat_times,
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
CUDA_VISIBLE_DEVICES=0 python benchmark/component_scaling_efficiency/ulysses_scaling_efficiency/test_ulysses.py --bs 8 --seq_len 1024 --world_size 1 --logging
CUDA_VISIBLE_DEVICES=2,3 python benchmark/component_scaling_efficiency/ulysses_scaling_efficiency/test_ulysses.py --bs 8 --seq_len 1024 --world_size 2 --logging
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()

    import torch.multiprocessing as mp

    # input params
    bs = args.bs
    seq_len = args.seq_len
    world_size = args.world_size
    log = args.logging

    # Flux Model params
    hc = 24
    hs = 128

    master_port = find_free_port()

    mp.spawn(
        test_ulysses_attn,
        args=(world_size, seq_len, bs, hc, hs, master_port, log),
        nprocs=world_size,
        join=True,
    )
