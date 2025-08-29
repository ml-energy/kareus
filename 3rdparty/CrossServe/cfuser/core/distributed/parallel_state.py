# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py
# Copyright 2023 The vLLM team.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
from typing import List, Optional, Dict

import torch
import torch.distributed
import cfuser.envs as envs

from cfuser.logger import init_logger
from .group_coordinator import (
    GroupCoordinator,
    SequenceParallelGroupCoordinator,
)

from cfuser.core.distributed.globals import (
    PROCESS_GROUP,
    set_seq_parallel_pg,
)

env_info = envs.PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]

logger = init_logger(__name__)

_WORLD: Optional[GroupCoordinator] = None
_SP = PROCESS_GROUP.get_non_attn_pg(index_req=0)


# * QUERY
def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


# SP
def get_sp_group(index_req: int = 0) -> SequenceParallelGroupCoordinator:
    assert (
        PROCESS_GROUP.get_non_attn_pg(index_req=index_req) is not None
    ), "pipeline model parallel group is not initialized"
    return PROCESS_GROUP.get_non_attn_pg(index_req=index_req)


def get_sequence_parallel_world_size(index_req: int = 0):
    """Return world size for the sequence parallel group."""
    return get_sp_group(index_req).size()


def get_sequence_parallel_rank(index_req: int = 0):
    """Return my rank for the sequence parallel group."""
    return get_sp_group(index_req).rank()


def get_ulysses_parallel_world_size(index_req: int = 0):
    return PROCESS_GROUP.get_ulysses_pg(index_req).size()


def get_ulysses_parallel_rank(index_req: int = 0):
    return PROCESS_GROUP.get_ulysses_pg(index_req).rank()


def get_ring_parallel_world_size(index_req: int = 0):
    return PROCESS_GROUP.get_ring_pg(index_req).size()


def get_ring_parallel_rank(index_req: int = 0):
    return PROCESS_GROUP.get_ring_pg(index_req).rank()


def init_world_group(ranks: List[int], local_rank: int, backend: str) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
    )


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):

    # if we don't support nccl or torch , we should use gloo
    if not torch.distributed.is_nccl_available() and not torch.distributed.is_torch_cuda_available():
        backend = "gloo"

    logger.debug(
        "world_size=%d rank=%d local_rank=%d " "distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing " "distributed environment"
        )
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            # device_id=torch.device(f"cuda:{local_rank}") if local_rank != -1 else None,
        )
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        else:
            local_rank = rank

    torch.cuda.set_device(local_rank)

    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        assert (
            _WORLD.world_size == torch.distributed.get_world_size()
        ), "world group already initialized with a different world size"


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return _SP is not None


def initialize_model_parallel(
    sequence_parallel_degree: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        sequence_parallel_degree: number of GPUs used for sequence parallelism.
        ulysses_degree: number of GPUs used for ulysses sequence parallelism.
        ring_degree: number of GPUs used for ring sequence parallelism.
        backend: distributed backend of pytorch collective comm.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    PROCESS_GROUP.generate_all_process_groups(world_size)

    from cfuser.core.distributed.globals import MAX_CHANNELS_PROC_GROUP

    for index_req in range(MAX_CHANNELS_PROC_GROUP):
        set_seq_parallel_pg(
            sp_ulysses_degree=ulysses_degree,
            sp_ring_degree=ring_degree,
            rank=get_world_group().rank_in_group,
            ranks=list(range(world_size)),
            world_size=get_world_group().world_size,
            index_req=index_req,
        )

        PROCESS_GROUP.set_seq_parallel_pg(sp_name="non_attn_sp", ranks=list(range(world_size)), index_req=index_req)

    global _SP
    _SP = PROCESS_GROUP.get_non_attn_pg(index_req=0)
    assert _SP is not None, "PROCESS_GROUP.get_non_attn_pg is not initialized"


def set_runtime_config(
    ranks: List[int],
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    non_attn_sp_ranks: List[int] = [],
    index_req: int = 0,
):

    set_seq_parallel_pg(
        sp_ulysses_degree=ulysses_degree,
        sp_ring_degree=ring_degree,
        rank=get_world_group().rank_in_group,
        ranks=ranks,
        world_size=len(ranks),
        index_req=index_req,
    )

    if non_attn_sp_ranks:
        PROCESS_GROUP.set_seq_parallel_pg(sp_name="non_attn_sp", ranks=non_attn_sp_ranks, index_req=index_req)

    global _SP
    _SP = PROCESS_GROUP.get_non_attn_pg(index_req=0)
    assert _SP is not None, "PROCESS_GROUP.get_non_attn_pg is not initialized"


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    PROCESS_GROUP.destroy()


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
