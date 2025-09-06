import torch
import torch.distributed as dist
from typing import List

try:
    from .msccl_comm import (
        init_bootstrap,
        get_unique_id,
        init_communicator,
        msccl_alltoall,
        # msccl_alltoallv, # deprecated
        msccl_alltoallv_cached,
        init_NetAlltoAllv_wrapper,
        init_NetAllReduce_wrapper_cached,
        init_NetAllReduce_wrapper,  # Added new function name
        msccl_allreduce,
        msccl_allreduce_cached,
        init_NetAllReduce_wrapper_bf16,
        msccl_allreduce_cached_bf16,
        msccl_allreduce_bf16,
    )
except ImportError as e:
    print(f"Warning: Could not import msccl_comm: {e}")

COMM = None
COMM_STREAM = None
# Track which subgroup COMM was initialized for
_COMM_GROUP_RANK = None
_COMM_GROUP_SIZE = None


def msccl_comm_init(rank, world_size):
    global COMM
    global COMM_STREAM
    init_bootstrap(rank, world_size)
    if rank == 0:
        unique_id = get_unique_id()
        # print(f"Unique ID: {unique_id}")
        dist_list = [unique_id]
    else:
        dist_list = [None]

    dist.broadcast_object_list(dist_list, src=0, group=dist.group.WORLD)

    unique_id = dist_list[0]

    COMM = init_communicator(world_size, unique_id, rank)

    COMM_STREAM = torch.cuda.Stream()
    init_NetAlltoAllv_wrapper(COMM, rank, world_size, COMM_STREAM.cuda_stream)


def msccl_AllReduce_init(
        rank: int,
        world_size: int,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        group: dist.ProcessGroup = None,
    ):
    global COMM
    global COMM_STREAM
    global _COMM_GROUP_RANK
    global _COMM_GROUP_SIZE

    # Determine subgroup rank/size and the process group for broadcast
    if group is not None:
        sub_group = group
        sub_rank = dist.get_rank(group=sub_group)
        sub_world_size = dist.get_world_size(group=sub_group)
    else:
        sub_group = dist.group.WORLD
        sub_rank = rank
        sub_world_size = world_size

    need_reinit = (
        COMM is None or _COMM_GROUP_RANK != sub_rank or _COMM_GROUP_SIZE != sub_world_size
    )

    if need_reinit:
        init_bootstrap(sub_rank, sub_world_size)
        if sub_rank == 0:
            unique_id = get_unique_id()
            dist_list = [unique_id]
        else:
            dist_list = [None]

        # Get the first rank in the subgroup to use as source
        group_ranks = dist.get_process_group_ranks(sub_group)
        src_rank = group_ranks[0]
        dist.broadcast_object_list(dist_list, src=src_rank, group=sub_group)

        unique_id = dist_list[0]

        COMM = init_communicator(sub_world_size, unique_id, sub_rank)
        _COMM_GROUP_RANK = sub_rank
        _COMM_GROUP_SIZE = sub_world_size

    if COMM_STREAM is None:
        COMM_STREAM = torch.cuda.Stream()

    if input_tensor.dtype == torch.bfloat16:
        init_NetAllReduce_wrapper_bf16(COMM,
                                       input_tensor.data_ptr(),
                                       output_tensor.data_ptr(),
                                       input_tensor.numel(),
                                       sub_rank,
                                       sub_world_size,
                                       COMM_STREAM.cuda_stream)
    else:
        init_NetAllReduce_wrapper(COMM,
                                  input_tensor.data_ptr(),
                                  output_tensor.data_ptr(),
                                  input_tensor.numel(),
                                  sub_rank,
                                  sub_world_size,
                                  COMM_STREAM.cuda_stream)


def msccl_AllReduce_init_cached(
        rank: int, 
        world_size: int, 
    ):
    global COMM
    global COMM_STREAM
    init_bootstrap(rank, world_size)
    if rank == 0:
        unique_id = get_unique_id()
        # print(f"Unique ID: {unique_id}")
        dist_list = [unique_id]
    else:
        dist_list = [None]

    dist.broadcast_object_list(dist_list, src=0, group=dist.group.WORLD)

    unique_id = dist_list[0]

    COMM = init_communicator(world_size, unique_id, rank)

    COMM_STREAM = torch.cuda.Stream()
    init_NetAllReduce_wrapper_cached(COMM,
                                    rank, 
                                    world_size, 
                                    COMM_STREAM.cuda_stream)
    # return COMM_STREAM

def msccl_AllReduce_cached(
    input: torch.Tensor,
    output: torch.Tensor,
    sm_num: int,
    block_size: int,
    stream: torch.cuda.Stream = None,
):
    assert COMM is not None, "Communicator not initialized"
    # assert COMM_STREAM is not None, "Stream not initialized"

    if stream is None:
        stream = COMM_STREAM
    if input.dtype == torch.bfloat16:
        msccl_allreduce_cached_bf16(
            COMM,
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            output.numel(),
            stream.cuda_stream,
            sm_num,
            block_size,
        )
    else:
        msccl_allreduce_cached(
            COMM,
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            output.numel(),
            stream.cuda_stream,
            sm_num,
            block_size,
        )

def msccl_AllReduce(
    sm_num: int,
    block_size: int,
    stream: torch.cuda.Stream = None,
    bf16: bool = False,
):
    assert COMM is not None, "Communicator not initialized"
    # assert COMM_STREAM is not None, "Stream not initialized"

    if stream is None:
        stream = COMM_STREAM
    # Note: uses last initialized dtype; prefer cached path for correctness
    if bf16:
        msccl_allreduce_bf16(
            COMM,
            stream.cuda_stream,
            sm_num,
            block_size,
        )
    else:
        msccl_allreduce(
            COMM,
            stream.cuda_stream,
            sm_num,
            block_size,
        )


def msccl_AlltoAll(
    input: torch.Tensor,
    output: torch.Tensor,
    # stream: torch.cuda.Stream,
    sm_num: int,
    block_size: int,
    nranks: int,
    rank: int,
):
    assert COMM is not None, "Communicator not initialized"
    assert COMM_STREAM is not None, "Stream not initialized"
    # block_size = 512
    # sm_num = nranks - 1
    msccl_alltoall(
        COMM,
        input.data_ptr(),
        output.data_ptr(),
        input.numel() * input.element_size(),
        COMM_STREAM.cuda_stream,
        sm_num,
        block_size,
        nranks,
        rank,
    )


from cfuser.core.long_ctx_attention.utils import computeLengthsAndOffsets, compute_split_sizes


def msccl_AlltoAllv(
    input: torch.Tensor,
    output: torch.Tensor,
    current_rank: int,
    ranks_mlp: List[int],
    ranks_attn: List[int],
    ranks_ulysses: List[int],
    ranks_ring: List[int],
    # stream: torch.cuda.Stream,
    sm_num: int,
    block_size: int,
    nranks: int,
    scatter_idx: int,
    gather_idx: int,
):
    assert COMM is not None, "Communicator not initialized"
    assert COMM_STREAM is not None, "Stream not initialized"
    assert len(ranks_mlp) == nranks, "ranks_mlp must be of length nranks"

    input_split_size_list, output_split_size_list = compute_split_sizes(
        current_rank=current_rank,
        ranks_mlp=ranks_mlp,
        ranks_attn=ranks_attn,
        # ranks_ulysses,
        # ranks_ring,
        ulysses_world_size=len(ranks_ulysses),
        ring_world_size=len(ranks_ring),
        scatter_idx=scatter_idx,
        gather_idx=gather_idx,
    )
    input_lengths, input_offsets = computeLengthsAndOffsets(
        input_split_size_list,
        input,
    )
    # print(f"[rank {current_rank}] input_split_size_list: {input_split_size_list}")
    # print(f"[rank {current_rank}] input_lengths: {input_lengths}")
    # print(f"[rank {current_rank}] input_offsets: {input_offsets}")

    output_lengths_all = []
    output_offsets_all = []
    for rank in ranks_mlp:
        _, output_split_size_list = compute_split_sizes(
            current_rank=rank,
            ranks_mlp=ranks_mlp,
            ranks_attn=ranks_attn,
            # ranks_ulysses,
            # ranks_ring,
            ulysses_world_size=len(ranks_ulysses),
            ring_world_size=len(ranks_ring),
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
        )
        output_lengths, output_offsets = computeLengthsAndOffsets(output_split_size_list, output)
        output_lengths_all.append(output_lengths)
        output_offsets_all.append(output_offsets)

        if rank == current_rank:
            output_split_size_list_host = [int(x) for x in output_split_size_list]
            # print(f"[rank {current_rank}] output_split_size_list: {output_split_size_list_host}")
            # print(f"[rank {current_rank}] output_lengths: {output_lengths}")
            # print(f"[rank {current_rank}] output_offsets: {output_offsets}")

    msccl_alltoallv_cached(
        COMM,
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),  # there is no `* input.element_size()`
        output.numel(),  # there is no `* output.element_size()`
        COMM_STREAM.cuda_stream,
        sm_num,
        block_size,
        # nranks,
        # current_rank,
        input_lengths,
        input_offsets,
        output_lengths_all,
        output_offsets_all,
    )


def msccl_sync():
    COMM_STREAM.synchronize()
