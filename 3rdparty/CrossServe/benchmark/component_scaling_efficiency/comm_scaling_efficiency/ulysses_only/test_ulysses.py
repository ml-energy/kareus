import os
import time
import torch
import torch.distributed as dist
from cfuser.core.long_ctx_attention.comm import all_to_all_4D
from cfuser.core.utils.zmq_utils import find_free_port
from torch.distributed.distributed_c10d import _get_default_group
from cfuser.core.long_ctx_attention.comm import uneven_all_to_all_4D, all_to_all_4D
from cfuser.core.utils.utils import nvtx_range


def test_ulysses_attn(
    rank,
    world_size,
    seq_len: int,
    batch_size: int,
    hc: int,
    hs: int,
    attn_world_size: int,
    mlp_world_size: int,
    repeat_times: int = 10,
    skip_a2a: bool = False,
    master_port: int = 1037,
    log: bool = False,
):

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"
    json_path = "log/benchmark/component_scaling_efficiency/comm_scaling_efficiency/comm_scaling_efficiency.json"
    # check if log exists
    if log and os.path.exists(json_path):
        data = json.load(open(json_path, "r"))
        for entry in data:
            if (
                entry["bs"] == batch_size
                and entry["seq_len"] == seq_len
                and entry["ulysses_world_size"] == world_size
                and entry["attn_world_size"] == attn_world_size
                and entry["mlp_world_size"] == mlp_world_size
                and skip_a2a
                or hasattr(entry, "a2a_time_scatter_idx_1_gather_idx_2")
            ):
                if rank == 0:
                    print(
                        f"Entry already exists for bs {batch_size}, seq_len {seq_len}, world_size {world_size}, attn_world_size {attn_world_size}, mlp_world_size {mlp_world_size}. "
                    )
                    print(
                        f"{repeat_times} iteration Time for uneven a2a for bs {batch_size}, "
                        + f"scatter_idx 2, gather_idx 1, seq_len {seq_len}, attn_world_size {attn_world_size}, mlp_world_size {mlp_world_size}: {entry['uneven_time_scatter_idx_2_gather_idx_1']:.3f} s"
                        + "\n"
                        + f"{repeat_times} iteration Time for uneven a2a for bs {batch_size}, "
                        + f"scatter_idx 1, gather_idx 2, seq_len {seq_len}, attn_world_size {attn_world_size}, mlp_world_size {mlp_world_size}: {entry['uneven_time_scatter_idx_1_gather_idx_2']:.3f} s"
                        + "\n"
                    )
                    if not skip_a2a:
                        print(
                            f"{repeat_times} iteration Time for a2a for bs {batch_size}, "
                            + f"scatter_idx 2, gather_idx 1, seq_len {seq_len}, attn_world_size {attn_world_size}, mlp_world_size {mlp_world_size}: {entry['a2a_time_scatter_idx_2_gather_idx_1']:.3f} s"
                            + "\n"
                            + f"{repeat_times} iteration Time for a2a for bs {batch_size}, "
                            + f"scatter_idx 1, gather_idx 2, seq_len {seq_len}, attn_world_size {attn_world_size}, mlp_world_size {mlp_world_size}: {entry['a2a_time_scatter_idx_1_gather_idx_2']:.3f} s"
                            + "\n"
                        )
                return
    dist.init_process_group(rank=rank, world_size=world_size, backend="nccl", init_method="env://")

    ranks_attn = list(range(attn_world_size))
    ranks_mlp = list(range(mlp_world_size))
    attn_group = dist.new_group(ranks=ranks_attn, backend="nccl")
    mlp_group = dist.new_group(ranks=ranks_mlp, backend="nccl")

    # 0. prepare data
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.manual_seed_all(42)

    q = torch.randn((batch_size, seq_len // world_size, hc, hs), device=device, dtype=torch.bfloat16).contiguous()
    k = torch.randn((batch_size, seq_len // world_size, hc, hs), device=device, dtype=torch.bfloat16).contiguous()
    v = torch.randn((batch_size, seq_len // world_size, hc, hs), device=device, dtype=torch.bfloat16).contiguous()
    empty_fake_attn_output = torch.Size(
        [batch_size, seq_len, hc // len(ranks_attn), hs], device=device, dtype=torch.bfloat16
    )

    # warmup
    for _ in range(50):
        a = torch.randn((4, 1024, 1024), device=f"cuda:{rank}") * torch.randn((4, 1024, 1024), device=f"cuda:{rank}")
        dist.all_reduce(a, op=dist.ReduceOp.AVG)
        # all to all kernels take much longer without the below warmup
        fake_attn_output_uneven = uneven_all_to_all_4D(
            q,
            ranks_send=ranks_mlp,
            ranks_recv=ranks_attn,
            group_send=mlp_group,
            group_recv=attn_group,
            scatter_idx=2,
            gather_idx=1,
            dtype=torch.bfloat16,
        )

        uneven_all_to_all_4D(
            fake_attn_output_uneven if rank in ranks_attn else empty_fake_attn_output,
            ranks_send=ranks_attn,
            ranks_recv=ranks_mlp,
            group_send=attn_group,
            group_recv=mlp_group,
            scatter_idx=1,
            gather_idx=2,
            dtype=torch.bfloat16,
        )

        fake_attn_output_a2a = all_to_all_4D(q, scatter_idx=2, gather_idx=1)
        all_to_all_4D(fake_attn_output_a2a, scatter_idx=1, gather_idx=2)

    if rank == 0:
        print(f"bs {batch_size}, seq_len {seq_len}, world_size {world_size} running...")

    dist.barrier()

    torch.cuda.cudart().cudaProfilerStart()

    # Test uneven all_to_all_4D
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeat_times):
        with nvtx_range("uneven_all_to_all_4D scatter_idx=2, gather_idx=1"):
            fake_attn_output_uneven = uneven_all_to_all_4D(
                q,
                ranks_send=ranks_mlp,
                ranks_recv=ranks_attn,
                group_send=mlp_group,
                group_recv=attn_group,
                scatter_idx=2,
                gather_idx=1,
                dtype=torch.bfloat16,
            )
            uneven_all_to_all_4D(
                k,
                ranks_send=ranks_mlp,
                ranks_recv=ranks_attn,
                group_send=mlp_group,
                group_recv=attn_group,
                scatter_idx=2,
                gather_idx=1,
                dtype=torch.bfloat16,
            )
            uneven_all_to_all_4D(
                v,
                ranks_send=ranks_mlp,
                ranks_recv=ranks_attn,
                group_send=mlp_group,
                group_recv=attn_group,
                scatter_idx=2,
                gather_idx=1,
                dtype=torch.bfloat16,
            )
    end.record()
    torch.cuda.synchronize()
    end.synchronize()
    uneven_time_scatter_idx_2_gather_idx_1 = begin.elapsed_time(end) / 1000.0

    dist.barrier()

    # Test uneven all_to_all_4D
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeat_times):
        with nvtx_range("uneven_all_to_all_4D scatter_idx=1, gather_idx=2"):
            uneven_all_to_all_4D(
                fake_attn_output_uneven if rank in ranks_attn else empty_fake_attn_output,
                ranks_send=ranks_attn,
                ranks_recv=ranks_mlp,
                group_send=attn_group,
                group_recv=mlp_group,
                scatter_idx=1,
                gather_idx=2,
                dtype=torch.bfloat16,
            )
    end.record()
    torch.cuda.synchronize()
    end.synchronize()
    uneven_time_scatter_idx_1_gather_idx_2 = begin.elapsed_time(end) / 1000.0

    dist.barrier()

    # Test all_to_all_4D
    if not skip_a2a:
        begin.record()
        for _ in range(repeat_times):
            with nvtx_range("all_to_all_4D scatter_idx=2, gather_idx=1"):
                fake_attn_output_a2a = all_to_all_4D(q, scatter_idx=2, gather_idx=1)
                all_to_all_4D(k, scatter_idx=2, gather_idx=1)
                all_to_all_4D(v, scatter_idx=2, gather_idx=1)
        end.record()
        torch.cuda.synchronize()
        end.synchronize()
        all_to_all_time_scatter_idx_2_gather_idx_1 = begin.elapsed_time(end) / 1000.0

        dist.barrier()

        begin.record()
        for _ in range(repeat_times):
            with nvtx_range("all_to_all_4D scatter_idx=1, gather_idx=2"):
                all_to_all_4D(fake_attn_output_a2a, scatter_idx=1, gather_idx=2)
        end.record()
        torch.cuda.synchronize()
        end.synchronize()
        all_to_all_time_scatter_idx_1_gather_idx_2 = begin.elapsed_time(end) / 1000.0

    torch.cuda.cudart().cudaProfilerStop()

    if rank == 0:
        print(
            f"{repeat_times} iteration Time for uneven a2a for bs {batch_size}, "
            + f"scatter_idx 2, gather_idx 1, seq_len {seq_len}, attn_world_size {attn_world_size}, mlp_world_size {mlp_world_size}: {uneven_time_scatter_idx_2_gather_idx_1:.3f} s"
            + "\n"
            + f"{repeat_times} iteration Time for uneven a2a for bs {batch_size}, "
            + f"scatter_idx 1, gather_idx 2, seq_len {seq_len}, attn_world_size {attn_world_size}, mlp_world_size {mlp_world_size}: {uneven_time_scatter_idx_1_gather_idx_2:.3f} s"
            + "\n"
        )
        if not skip_a2a:
            print(
                f"{repeat_times} iteration Time for a2a for bs {batch_size}, "
                + f"scatter_idx 2, gather_idx 1, seq_len {seq_len}, attn_world_size {attn_world_size}, mlp_world_size {mlp_world_size}: {all_to_all_time_scatter_idx_2_gather_idx_1:.3f} s"
                + "\n"
                + f"{repeat_times} iteration Time for a2a for bs {batch_size}, "
                + f"scatter_idx 1, gather_idx 2, seq_len {seq_len}, attn_world_size {attn_world_size}, mlp_world_size {mlp_world_size}: {all_to_all_time_scatter_idx_1_gather_idx_2:.3f} s"
                + "\n"
            )

        if log:
            import json

            json_path = (
                f"log/benchmark/component_scaling_efficiency/comm_scaling_efficiency/comm_scaling_efficiency.json"
            )

            to_save = {
                "bs": batch_size,
                "seq_len": seq_len,
                "hc": hc,
                "hs": hs,
                "ulysses_world_size": world_size,
                "num_iter": repeat_times,
                "avg_uneven_time_scatter_idx_2_gather_idx_1": uneven_time_scatter_idx_2_gather_idx_1 / repeat_times,
                "uneven_time_scatter_idx_2_gather_idx_1": uneven_time_scatter_idx_2_gather_idx_1,
                "avg_uneven_time_scatter_idx_1_gather_idx_2": uneven_time_scatter_idx_1_gather_idx_2 / repeat_times,
                "uneven_time_scatter_idx_1_gather_idx_2": uneven_time_scatter_idx_1_gather_idx_2,
                "mlp_world_size": mlp_world_size,
                "attn_world_size": attn_world_size,
            }
            if not skip_a2a:
                # fmt: off
                to_save["avg_a2a_time_scatter_idx_2_gather_idx_1"] = all_to_all_time_scatter_idx_2_gather_idx_1 / repeat_times
                to_save["a2a_time_scatter_idx_2_gather_idx_1"] = all_to_all_time_scatter_idx_2_gather_idx_1
                to_save["avg_a2a_time_scatter_idx_1_gather_idx_2"] = all_to_all_time_scatter_idx_1_gather_idx_2 / repeat_times
                # fmt: on
                to_save["a2a_time_scatter_idx_1_gather_idx_2"] = all_to_all_time_scatter_idx_1_gather_idx_2

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
CUDA_VISIBLE_DEVICES=0,1 python benchmark/component_scaling_efficiency/comm_scaling_efficiency/test_ulysses.py --bs 8 --seq_len 1024 --attn_world_size 1 --mlp_world_size 2 --logging
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark/component_scaling_efficiency/comm_scaling_efficiency/test_ulysses.py --bs 8 --seq_len 1024 --attn_world_size 4 --mlp_world_size 4 --repeat_times 50
CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile --force-overwrite true -o log/benchmark/component_scaling_efficiency/comm_scaling_efficiency/debug_ulysses_bs8_seq_len1024_world_size4_repeat_times50 python benchmark/component_scaling_efficiency/comm_scaling_efficiency/test_ulysses.py --bs 8 --seq_len 1024 --attn_world_size 4 --mlp_world_size 4 --repeat_times 50
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", "-b", type=int, default=1)
    parser.add_argument("--seq_len", "-s", type=int, default=1024)
    parser.add_argument("--attn_world_size", type=int, default=1)
    parser.add_argument("--mlp_world_size", type=int, default=1)
    parser.add_argument("--skip_a2a", type=eval, default=False)
    parser.add_argument("--repeat_times", type=int, default=50)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()

    import torch.multiprocessing as mp

    # input params
    bs = args.bs
    seq_len = args.seq_len
    world_size = max(args.attn_world_size, args.mlp_world_size)
    log = args.logging
    attn_world_size = args.attn_world_size
    mlp_world_size = args.mlp_world_size
    repeat_times = args.repeat_times
    skip_a2a = args.skip_a2a
    # Flux Model params
    hc = 24
    hs = 128

    master_port = find_free_port()

    mp.spawn(
        test_ulysses_attn,
        args=(
            world_size,
            seq_len,
            bs,
            hc,
            hs,
            attn_world_size,
            mlp_world_size,
            repeat_times,
            skip_a2a,
            master_port,
            log,
        ),
        nprocs=world_size,
        join=True,
    )
