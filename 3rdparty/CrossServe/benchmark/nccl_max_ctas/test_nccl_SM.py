import os
import torch
import torch.distributed as dist
from cfuser.core.long_ctx_attention.comm import all_to_all_4D
from cfuser.core.utils.zmq_utils import find_free_port
from flash_attn import flash_attn_func


class ATTN(torch.nn.Module):
    def __init__(self, bs, seq_len, hs, hc, rank):
        super().__init__()
        self.bs = bs
        self.seq_len = seq_len
        self.hs = hs
        self.hc = hc

        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()

        self.linear = torch.nn.Linear(hs * hc, hs * hc, device=torch.device(f"cuda:{rank}"), dtype=torch.bfloat16)

    def comp_qkv(self, q, k, v, group=dist.group.WORLD):
        for i in range(8):
            bs, shard_seq_len, hc, hs = q.shape
            q = self.linear(q.reshape(bs * shard_seq_len, hc * hs)).reshape(bs, shard_seq_len, hc, hs)
            k = self.linear(k.reshape(bs * shard_seq_len, hc * hs)).reshape(bs, shard_seq_len, hc, hs)
            v = self.linear(v.reshape(bs * shard_seq_len, hc * hs)).reshape(bs, shard_seq_len, hc, hs)
        return q, k, v

    def comm_qkv(self, q, k, v, group=dist.group.WORLD, async_op=False):
        q_local_all = all_to_all_4D(q, scatter_idx=2, gather_idx=1, group=group, async_op=async_op)
        k_local_all = all_to_all_4D(k, scatter_idx=2, gather_idx=1, group=group, async_op=async_op)
        v_local_all = all_to_all_4D(v, scatter_idx=2, gather_idx=1, group=group, async_op=async_op)
        return q_local_all, k_local_all, v_local_all

    def comp_attn(self, q, k, v, group=dist.group.WORLD):
        attn_output = flash_attn_func(
            q,
            k,
            v,
            causal=False,
        )
        return attn_output

    def comm_attn(self, attn_output, group=dist.group.WORLD, async_op=False):
        ulysses_output = all_to_all_4D(attn_output, scatter_idx=1, gather_idx=2, group=group, async_op=async_op)
        return ulysses_output

    def reshape_comm_tensor(self, tensor, world_size, bs, shard_seqlen, hc, hs, group=dist.group.WORLD):
        seqlen = shard_seqlen * world_size
        shard_hc = hc // world_size
        tensor = tensor.reshape(seqlen, bs, shard_hc, hs)
        tensor = tensor.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        return tensor

    def forward(self, q1, k1, v1, q2, k2, v2, group=dist.group.WORLD, async_op=False, overlap=False):

        if not overlap:
            self.stream2 = self.stream1
            assert async_op == False

        bs, shard_seqlen, hc, hs = q1.shape
        world_size = dist.get_world_size(group)

        event1 = torch.cuda.Event(enable_timing=False)
        event2 = torch.cuda.Event(enable_timing=False)

        with torch.cuda.stream(self.stream1):
            with torch.cuda.nvtx.range("comp_qkv 1"):
                q1, k1, v1 = self.comp_qkv(q1, k1, v1, group=group)
                event1.record()
            with torch.cuda.nvtx.range("comm_qkv 1"):
                q1_local_all, k1_local_all, v1_local_all = self.comm_qkv(q1, k1, v1, group=group, async_op=async_op)

        with torch.cuda.stream(self.stream2):
            self.stream2.wait_event(event1)
            with torch.cuda.nvtx.range("comp_qkv 2"):
                q2, k2, v2 = self.comp_qkv(q2, k2, v2, group=group)
                event2.record()

        with torch.cuda.stream(self.stream1):
            if async_op:
                q1_local_all, q1_local_all_handle = q1_local_all
                k1_local_all, k1_local_all_handle = k1_local_all
                v1_local_all, v1_local_all_handle = v1_local_all

                q1_local_all_handle.wait()
                k1_local_all_handle.wait()
                v1_local_all_handle.wait()

                q1_local_all = self.reshape_comm_tensor(q1_local_all, world_size, bs, shard_seqlen, hc, hs, group=group)
                k1_local_all = self.reshape_comm_tensor(k1_local_all, world_size, bs, shard_seqlen, hc, hs, group=group)
                v1_local_all = self.reshape_comm_tensor(v1_local_all, world_size, bs, shard_seqlen, hc, hs, group=group)

            self.stream1.wait_event(event2)
            with torch.cuda.nvtx.range("comp_attn 1"):
                attn1_output = self.comp_attn(q1_local_all, k1_local_all, v1_local_all, group=group)
            # ulysses1_output = self.comm_attn(attn1_output, group=group, async_op=async_op)

        with torch.cuda.stream(self.stream2):
            q2_local_all, k2_local_all, v2_local_all = self.comm_qkv(q2, k2, v2, group=group, async_op=async_op)
            if async_op:
                q2_local_all, q2_local_all_handle = q2_local_all
                k2_local_all, k2_local_all_handle = k2_local_all
                v2_local_all, v2_local_all_handle = v2_local_all

                q2_local_all_handle.wait()
                k2_local_all_handle.wait()
                v2_local_all_handle.wait()

                q2_local_all = self.reshape_comm_tensor(q2_local_all, world_size, bs, shard_seqlen, hc, hs, group=group)
                k2_local_all = self.reshape_comm_tensor(k2_local_all, world_size, bs, shard_seqlen, hc, hs, group=group)
                v2_local_all = self.reshape_comm_tensor(v2_local_all, world_size, bs, shard_seqlen, hc, hs, group=group)

            with torch.cuda.nvtx.range("comp_attn 2"):
                attn2_output = self.comp_attn(q2_local_all, k2_local_all, v2_local_all, group=group)
            # ulysses2_output = self.comm_attn(attn2_output, group=group, async_op=async_op)

        torch.cuda.current_stream().wait_stream(self.stream1)
        torch.cuda.current_stream().wait_stream(self.stream2)

        return attn1_output, attn2_output


def test_ulysses_attn(
    rank,
    world_size,
    seq_len: int,
    batch_size: int,
    hc: int,
    hs: int,
    master_port: int = 1037,
    log: bool = False,
    overlap: bool = True,
):

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

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
    for _ in range(10):
        std_attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1, 2)
        dist.all_reduce(std_attn_output, op=dist.ReduceOp.AVG)

    q_local = Q[:, rank * (seq_len // world_size) : (rank + 1) * (seq_len // world_size), :, :].contiguous().clone()
    k_local = K[:, rank * (seq_len // world_size) : (rank + 1) * (seq_len // world_size), :, :].contiguous().clone()
    v_local = V[:, rank * (seq_len // world_size) : (rank + 1) * (seq_len // world_size), :, :].contiguous().clone()

    q1, k1, v1 = q_local.clone(), k_local.clone(), v_local.clone()
    q2, k2, v2 = q_local.clone(), k_local.clone(), v_local.clone()

    if rank == 0:
        print(f"bs {batch_size}, seq_len {seq_len}, world_size {world_size} running...")

    ulysses_attn = ATTN(batch_size, seq_len, hs, hc, rank)

    dist.barrier()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeat_times):
        attn1_output, attn2_output = ulysses_attn(
            q1, k1, v1, q2, k2, v2, group=dist.group.WORLD, async_op=False, overlap=overlap
        )
        dist.barrier()
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if rank == 0:
        print(f"{time:.3f} sec")
        print(
            f"NCCL_MAX_CTAS={os.getenv('NCCL_MAX_CTAS')} NCCL_MIN_CTAS={os.getenv('NCCL_MIN_CTAS')} {repeat_times} iteration Time for ulysses attn for bs {batch_size}, seq_len {seq_len}, world_size {world_size}: {time:.3f} s"
        )
        if log:
            import json

            json_path = f"log/benchmark/nccl_max_ctas/nccl_max_ctas.json"

            to_save = {
                "bs": batch_size,
                "seq_len": seq_len,
                "hc": hc,
                "hs": hs,
                "world_size": world_size,
                "num_iter": repeat_times,
                "avg_time": time / repeat_times,
                "time": time,
                "NCCL_MAX_CTAS": os.getenv("NCCL_MAX_CTAS"),
                "NCCL_MIN_CTAS": os.getenv("NCCL_MIN_CTAS"),
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
CUDA_VISIBLE_DEVICES=0 python benchmark/nccl_max_ctas/test_nccl_SM.py --bs 8 --seq_len 1024 --world_size 1
NCCL_MAX_CTAS=20 NCCL_MIN_CTAS=20 CUDA_VISIBLE_DEVICES=2,3 python benchmark/nccl_max_ctas/test_nccl_SM.py --bs 8 --seq_len 1024 --world_size 2
NCCL_MAX_CTAS=1 NCCL_MIN_CTAS=1 CUDA_VISIBLE_DEVICES=0,1,2,3  nsys profile --force-overwrite true -w true -s cpu -o log/benchmark/nccl_max_ctas/nccl_max_ctas_1_1_bs8_seq_len4096_world_size4 python benchmark/nccl_max_ctas/test_nccl_SM.py --bs 8 --seq_len 4096 --world_size 4
NCCL_MAX_CTAS=1 NCCL_MIN_CTAS=1 CUDA_VISIBLE_DEVICES=0,1,2,3  nsys profile --force-overwrite true -w true -s cpu -o log/benchmark/nccl_max_ctas/nccl_max_ctas_1_1_bs8_seq_len4096_world_size4_no_overlap python benchmark/nccl_max_ctas/test_nccl_SM.py --bs 8 --seq_len 4096 --world_size 4 --no_overlap
CUDA_VISIBLE_DEVICES=0,1,2,3  nsys profile --force-overwrite true -w true -s cpu -o log/benchmark/nccl_max_ctas/default_nccl_max_ctas_1_1_bs8_seq_len4096_world_size4_no_overlap python benchmark/nccl_max_ctas/test_nccl_SM.py --bs 8 --seq_len 4096 --world_size 4
NCCL_MAX_CTAS=16 NCCL_MIN_CTAS=16 CUDA_VISIBLE_DEVICES=0,1,2,3  nsys profile --force-overwrite true -w true -s cpu -o log/benchmark/nccl_max_ctas/nccl_max_ctas_16_16_bs8_seq_len4096_world_size4 python benchmark/nccl_max_ctas/test_nccl_SM.py --bs 8 --seq_len 4096 --world_size 4
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--no_overlap", action="store_true", default=False)
    parser.add_argument("--logging", action="store_true", default=False)
    args = parser.parse_args()

    import torch.multiprocessing as mp

    # input params
    bs = args.bs
    seq_len = args.seq_len
    world_size = args.world_size
    overlap = not args.no_overlap
    log = args.logging

    # Flux Model params
    hc = 24
    hs = 128
    master_port = find_free_port()

    mp.spawn(
        test_ulysses_attn,
        args=(world_size, seq_len, bs, hc, hs, master_port, log, overlap),
        nprocs=world_size,
        join=True,
    )
