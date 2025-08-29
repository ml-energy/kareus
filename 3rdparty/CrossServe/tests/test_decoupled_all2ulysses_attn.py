import os
import time
import torch
import torch.distributed as dist
from cfuser.core.long_ctx_attention.comm import all_to_all_4D, uneven_all_to_all_4D, uneven_decoupled_all_to_all_4D
from cfuser.core.utils.zmq_utils import find_free_port
from torch.distributed.distributed_c10d import _get_default_group
from cfuser.core.utils.utils import nvtx_range
from cfuser.testing import assert_close

CUSTOM_BACKEND = "python"


def check_all2all_uneven(rank, world_size, shape, dtype, async_op):
    """
    Test all_to_all_single with uneven split sizes.
    """
    x = torch.rand(shape, dtype=dtype, device=f"cuda:{rank}")
    input_split_sizes = torch.zeros(world_size, dtype=torch.int32, device=f"cuda:{rank}")
    input_split_sizes[0 : world_size // 2] = x.shape[0] // (world_size // 2)
    output_split_sizes = torch.zeros_like(input_split_sizes, device=f"cuda:{rank}")

    dist.all_to_all_single(output_split_sizes, input_split_sizes, group=_get_default_group())

    output_shape = list(shape)
    output_shape[0] = sum(output_split_sizes)
    output = torch.zeros(output_shape, device=x.device, dtype=x.dtype)

    input_split_size_list = input_split_sizes.tolist()
    output_split_size_list = output_split_sizes.tolist()

    origin_hanle = dist.all_to_all_single(
        output,
        x,
        output_split_sizes=output_split_size_list,
        input_split_sizes=input_split_size_list,
        group=_get_default_group(),
        async_op=async_op,
    )

    if async_op:
        origin_hanle.wait()


def original_ulysses_attn(q, k, v, group=dist.group.WORLD):
    q_local_all = all_to_all_4D(q, scatter_idx=2, gather_idx=1, group=group)
    k_local_all = all_to_all_4D(k, scatter_idx=2, gather_idx=1, group=group)
    v_local_all = all_to_all_4D(v, scatter_idx=2, gather_idx=1, group=group)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        q_local_all.transpose(1, 2),
        k_local_all.transpose(1, 2),
        v_local_all.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)

    ulysses_output = all_to_all_4D(attn_output, scatter_idx=1, gather_idx=2, group=group)

    return ulysses_output, attn_output, q_local_all, k_local_all, v_local_all


def diasggregated_ulysses_attn(q, k, v, ranks_mlp, ranks_attn, group_mlp, group_attn):
    bs, shard_seqlen, hc, hs = q.shape
    a2a_dtype = q.dtype

    # uneven_all_to_all_4D is deprecated, please use uneven_decoupled_all_to_all_4D instead
    # q_local = uneven_all_to_all_4D(
    #     q,
    #     ranks_send=ranks_mlp,
    #     ranks_recv=ranks_attn,
    #     group_send=group_mlp,
    #     group_recv=group_attn,
    #     custom_backend=CUSTOM_BACKEND,
    #     dtype=a2a_dtype,
    # )
    # k_local = uneven_all_to_all_4D(
    #     k,
    #     ranks_send=ranks_mlp,
    #     ranks_recv=ranks_attn,
    #     group_send=group_mlp,
    #     group_recv=group_attn,
    #     custom_backend=CUSTOM_BACKEND,
    #     dtype=a2a_dtype,
    # )
    # v_local = uneven_all_to_all_4D(
    #     v,
    #     ranks_send=ranks_mlp,
    #     ranks_recv=ranks_attn,
    #     group_send=group_mlp,
    #     group_recv=group_attn,
    #     custom_backend=CUSTOM_BACKEND,
    #     dtype=a2a_dtype,
    # )
    q_local = uneven_decoupled_all_to_all_4D(
        q,
        ranks_mlp=ranks_mlp,
        ranks_attn=ranks_attn,
        ranks_ulysses=ranks_attn,
        ranks_ring=[ranks_attn[0]],
        group_mlp=group_mlp,
        dtype=a2a_dtype,
    )
    k_local = uneven_decoupled_all_to_all_4D(
        k,
        ranks_mlp=ranks_mlp,
        ranks_attn=ranks_attn,
        ranks_ulysses=ranks_attn,
        ranks_ring=[ranks_attn[0]],
        group_mlp=group_mlp,
        dtype=a2a_dtype,
    )
    v_local = uneven_decoupled_all_to_all_4D(
        v,
        ranks_mlp=ranks_mlp,
        ranks_attn=ranks_attn,
        ranks_ulysses=ranks_attn,
        ranks_ring=[ranks_attn[0]],
        group_mlp=group_mlp,
        dtype=a2a_dtype,
    )

    if q_local is not None:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_local.transpose(1, 2),
            k_local.transpose(1, 2),
            v_local.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
    else:
        attn_output = torch.Size([bs, shard_seqlen * len(ranks_mlp), hc // len(ranks_attn), hs])

    # ulysses_output = uneven_all_to_all_4D(
    #     attn_output,
    #     scatter_idx=1,
    #     gather_idx=2,
    #     ranks_send=ranks_attn,
    #     ranks_recv=ranks_mlp,
    #     group_send=group_attn,
    #     group_recv=group_mlp,
    #     custom_backend=CUSTOM_BACKEND,
    #     dtype=a2a_dtype,
    # )

    ulysses_output = uneven_decoupled_all_to_all_4D(
        attn_output,
        ranks_mlp=ranks_mlp,
        ranks_attn=ranks_attn,
        ranks_ulysses=ranks_attn,
        ranks_ring=[ranks_attn[0]],
        group_mlp=group_mlp,
        dtype=a2a_dtype,
        scatter_idx=1,
        gather_idx=2,
    )

    return ulysses_output, attn_output, q_local, k_local, v_local


def test_ulysses_attn(rank, world_size, seq_len: int, batch_size: int, hc: int, hs: int, master_port: int = 1037):
    """
    Compare the overhead of all2all and uneven_decoupled_all2all.
    For decoupled sp, we must use uneven_decoupled_all2all instead of traditional all2all.
    here we simulate the ulysses Attention Computation with 4D tensor to verify the correctness and performance of uneven_decoupled_all2all/even_all2all(original).
    """

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    dist.init_process_group(rank=rank, world_size=world_size, backend="nccl", init_method="env://")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    repeat_times = 20

    # 0. prepare data
    check_all2all_uneven(rank, world_size, (seq_len, batch_size, hc, hs), torch.float16, async_op=False)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % world_size}")
    Q = torch.randn(batch_size, seq_len, hc, hs, device=device)
    K = torch.randn(batch_size, seq_len, hc, hs, device=device)
    V = torch.randn(batch_size, seq_len, hc, hs, device=device)

    dist.broadcast(Q, src=0)
    dist.broadcast(K, src=0)
    dist.broadcast(V, src=0)

    # warmup
    for _ in range(50):
        std_attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1, 2)
        dist.all_reduce(std_attn_output, op=dist.ReduceOp.AVG)

    q_local = Q[:, rank * (seq_len // world_size) : (rank + 1) * (seq_len // world_size), :, :].contiguous().clone()
    k_local = K[:, rank * (seq_len // world_size) : (rank + 1) * (seq_len // world_size), :, :].contiguous().clone()
    v_local = V[:, rank * (seq_len // world_size) : (rank + 1) * (seq_len // world_size), :, :].contiguous().clone()

    torch.cuda.cudart().cudaProfilerStart()

    # 1. test original ulysses attn
    group = dist.new_group(ranks=list(range(world_size)), backend="nccl", use_local_synchronization=True)
    start_event.record()
    for _ in range(repeat_times):
        with nvtx_range("original_ulysses_attn"):
            ulysses_output_even, attn_output_even, q_even, k_even, v_even = original_ulysses_attn(
                q_local, k_local, v_local, group=group
            )
    # No need to put dist.barrier here (only blocks CPU); we want to measure CUDA kernel completetion.
    end_event.record()
    end_event.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1e3
    print(f"[rank {rank}] original ulysses attn time: {elapsed_time / repeat_times} s")

    assert_close(
        std_attn_output[:, seq_len // world_size * rank : seq_len // world_size * (rank + 1), :, :], ulysses_output_even
    )

    # 2. test decoupled ulysses attn
    ranks_mlp = list(range(world_size))
    ranks_attn = list(range(world_size))
    group_mlp = dist.new_group(ranks=ranks_mlp, backend="nccl")
    group_attn = dist.new_group(ranks=ranks_attn, backend="nccl")

    dist.barrier(group_mlp)
    start_event.record()
    for _ in range(repeat_times):
        with nvtx_range("diasggregated_ulysses_attn 1"):
            ulysses_output_uneven_1, attn_output_uneven_1, q_uneven_1, k_uneven_1, v_uneven_1 = (
                diasggregated_ulysses_attn(q_local, k_local, v_local, ranks_mlp, ranks_attn, group_mlp, group_attn)
            )
    end_event.record()
    end_event.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1e3
    print(f"[rank {rank}] decoupled ulysses attn time: {elapsed_time / repeat_times} s")

    if rank in ranks_attn:
        assert_close(
            Q[
                :,
                :,
                hc // len(ranks_attn) * ranks_attn.index(rank) : hc // len(ranks_attn) * (ranks_attn.index(rank) + 1),
                :,
            ],
            q_uneven_1,
        )
        assert_close(
            ulysses_output_uneven_1,
            std_attn_output[
                :,
                seq_len // world_size * ranks_attn.index(rank) : seq_len // world_size * (ranks_attn.index(rank) + 1),
                :,
                :,
            ],
        )

    # 3. test uneven decoupled ulysses attn
    ranks_attn = list(range(world_size // 2))
    # ranks_attn = [0]
    group_attn = dist.new_group(ranks=ranks_attn, backend="nccl")
    # print(f"rank in group: {torch.distributed.get_rank()} group_attn: {group_attn}")
    dist.barrier(group_mlp)
    start_event.record()
    for _ in range(repeat_times):
        with nvtx_range("diasggregated_ulysses_attn 2"):
            ulysses_output_uneven_2, attn_output_uneven_2, q_uneven_2, k_uneven_2, v_uneven_2 = (
                diasggregated_ulysses_attn(q_local, k_local, v_local, ranks_mlp, ranks_attn, group_mlp, group_attn)
            )
    end_event.record()
    end_event.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1e3
    print(f"[rank {rank}] uneven decoupled ulysses attn time: {elapsed_time / repeat_times} s")

    assert isinstance(ulysses_output_uneven_2, torch.Tensor)
    # print(f"[rank {rank}] shape of ulysses_output_uneven_2: {ulysses_output_uneven_2.shape}")

    assert_close(
        ulysses_output_uneven_2,
        std_attn_output[:, seq_len // world_size * rank : seq_len // world_size * (rank + 1), :, :],
    )

    if rank in ranks_attn:
        assert_close(
            Q[
                :,
                :,
                hc // len(ranks_attn) * ranks_attn.index(rank) : hc // len(ranks_attn) * (ranks_attn.index(rank) + 1),
                :,
            ],
            q_uneven_2,
        )
        assert_close(
            ulysses_output_uneven_2,
            std_attn_output[
                :,
                seq_len // world_size * ranks_attn.index(rank) : seq_len // world_size * (ranks_attn.index(rank) + 1),
                :,
                :,
            ],
        )

    torch.cuda.cudart().cudaProfilerStop()

    dist.destroy_process_group()


"""
export nsys_args="--force-overwrite true -w true --capture-range=cudaProfilerApi"
CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/tests/test_decoupled_all2ulysses_attn_bs1_u4_h24_s128 python tests/test_decoupled_all2ulysses_attn.py

python tests/test_decoupled_all2ulysses_attn.py
"""

if __name__ == "__main__":
    import torch.multiprocessing as mp

    # input params
    bs = 1
    seq_len = 4352 * 4
    world_size = 4

    # Flux Model params
    hc = 24
    hs = 128

    master_port = find_free_port()

    mp.spawn(
        test_ulysses_attn,
        args=(world_size, seq_len, bs, hc, hs, master_port),
        nprocs=world_size,
        join=True,
    )
