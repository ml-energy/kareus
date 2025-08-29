import os
import torch
import torch.distributed as dist
from cfuser.core.long_ctx_attention.comm import all_to_all_4D, uneven_decoupled_all_to_all_4D
from cfuser.core.utils.zmq_utils import find_free_port
from cfuser.core.utils.utils import nvtx_range
import cfuser.msccl_comm
from cfuser.testing import assert_close, assert_close_with_threshold
from cfuser.core.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from cfuser.core.distributed.globals import set_seq_parallel_pg, PROCESS_GROUP
from cfuser.logger import init_logger

logger = init_logger(__name__)

CUSTOM_BACKEND = "python"


def decoupled_comm(q, ranks_mlp, ranks_attn, ranks_ulysses, ranks_ring, group_mlp, group_ring):
    """
    decoupled communication.
    """

    # 1. all to all
    bs, shard_seqlen, hc, hs = q.shape
    q_local = uneven_decoupled_all_to_all_4D(
        q, ranks_mlp, ranks_attn, ranks_ulysses, ranks_ring, group_mlp, dtype=q.dtype
    )

    # 2. ring attn
    if dist.get_rank() in ranks_ring:
        attn_output = q_local
    else:
        attn_output = torch.Size([bs, shard_seqlen * len(ranks_mlp) // len(ranks_ring), hc // len(ranks_ulysses), hs])

    # 3. all to all
    out_all = uneven_decoupled_all_to_all_4D(
        attn_output,
        ranks_mlp,
        ranks_attn,
        ranks_ulysses,
        ranks_ring,
        group_mlp,
        scatter_idx=1,
        gather_idx=2,
        dtype=q.dtype,
    )

    return out_all


def test_decoupled_comm(rank, world_size, seq_len: int, batch_size: int, hc: int, hs: int, master_port: int = 1037):
    """
    Test decoupled communication.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"
    init_distributed_environment(world_size=world_size, rank=rank, backend="nccl")

    ring_attn_degree = 2
    ulysses_degree = world_size // ring_attn_degree
    initialize_model_parallel(ring_degree=ring_attn_degree, ulysses_degree=ulysses_degree)

    # mscclpp
    cfuser.msccl_comm.msccl_comm_init(rank, world_size)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    repeat_times = 20

    # 0. prepare data
    device = torch.device(f"cuda:{rank % world_size}")
    dtype = torch.bfloat16
    Q = torch.randn(batch_size, seq_len, hc, hs, device=device, dtype=dtype)
    dist.broadcast(Q, src=0)

    q_local = Q[:, rank * (seq_len // world_size) : (rank + 1) * (seq_len // world_size), :, :].contiguous().clone()

    # 1. randomly set up the env
    set_seq_parallel_pg(
        sp_ulysses_degree=ulysses_degree,
        sp_ring_degree=ring_attn_degree,
        rank=rank,
        ranks=list(range(world_size)),
        world_size=world_size,
        index_req=0,
    )
    PROCESS_GROUP.set_seq_parallel_pg(sp_name="non_attn_sp", ranks=list(range(world_size)), index_req=0)

    # 2. warmup
    for _ in range(30):
        dist.all_reduce(Q, group=PROCESS_GROUP.get_non_attn_pg(index_req=0), op=dist.ReduceOp.AVG)

    torch.cuda.cudart().cudaProfilerStart()

    def test(ulysses_degree, ring_attn_degree, attn_ranks):
        set_seq_parallel_pg(
            sp_ulysses_degree=ulysses_degree,
            sp_ring_degree=ring_attn_degree,
            rank=rank,
            ranks=attn_ranks,
            world_size=ulysses_degree * ring_attn_degree,
            index_req=0,
        )
        PROCESS_GROUP.set_seq_parallel_pg(sp_name="non_attn_sp", ranks=list(range(world_size)), index_req=0)
        non_attn_ranks = PROCESS_GROUP.get_non_attn_ranks(index_req=0)
        attn_ranks = PROCESS_GROUP.get_attn_ranks(index_req=0)
        ulysses_ranks = PROCESS_GROUP.get_ulysses_ranks(index_req=0)
        ring_ranks = PROCESS_GROUP.get_ring_ranks(index_req=0)
        group_mlp = PROCESS_GROUP.get_non_attn_pg(index_req=0)
        group_ring = PROCESS_GROUP.get_ring_pg(index_req=0)
        group_ulysses = PROCESS_GROUP.get_ulysses_pg(index_req=0)
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                f"-------testing mlp-{non_attn_ranks}-attn-{attn_ranks}-ulysses{len(ulysses_ranks)}-ring{len(ring_ranks)}-------"
            )
        dist.barrier()
        # print(
        #     f"[rank {rank}] ulysses_degree: {ulysses_degree}, ring_attn_degree: {ring_attn_degree}, attn_ranks: {attn_ranks}"
        # )
        # print(
        #     f"[rank {rank}] non_attn_ranks: {non_attn_ranks}, attn_ranks: {attn_ranks}, ulysses_ranks: {ulysses_ranks}, ring_ranks: {ring_ranks}"
        # )
        decoupled_comm_output = decoupled_comm(
            q_local, non_attn_ranks, attn_ranks, ulysses_ranks, ring_ranks, group_mlp, group_ring
        )
        dist.barrier()
        assert_close(decoupled_comm_output, q_local)
        if dist.get_rank() == 0:
            print(f"-------passed-------")
        dist.barrier()

    ulysses_degrees = [1, 2, 4, 8]
    ring_degrees = [1, 2, 4, 8]
    for ulysses_degree in ulysses_degrees:
        for ring_attn_degree in ring_degrees:
            if ulysses_degree * ring_attn_degree > world_size:
                continue
            attn_ranks_list = [
                list(range(i, i + ulysses_degree * ring_attn_degree))
                for i in range(world_size - ulysses_degree * ring_attn_degree + 1)
            ]
            for attn_ranks in attn_ranks_list:
                test(ulysses_degree, ring_attn_degree, attn_ranks)

    # test(2, 2, [0, 1, 2, 3])

    torch.cuda.cudart().cudaProfilerStop()


"""
export nsys_args="--force-overwrite true -w true --capture-range=cudaProfilerApi"
CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/tests/test_decoupled_all2all_bs4_u4_h24_s128 python tests/test_decoupled_all2all_operator.py
python tests/test_decoupled_all2all_operator.py 2>&1 | tee log.log
"""

if __name__ == "__main__":
    import torch.multiprocessing as mp

    # input params
    bs = 1
    world_size = 4
    seq_len = 4352 * world_size

    # Flux Model params
    hc = 24
    hs = 128

    master_port = find_free_port()

    mp.spawn(
        test_decoupled_comm,
        args=(world_size, seq_len, bs, hc, hs, master_port),
        nprocs=world_size,
        join=True,
    )
