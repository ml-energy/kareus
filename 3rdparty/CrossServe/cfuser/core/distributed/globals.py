from typing import List, Dict

import torch
import os

from .utils import generate_all_subsets, filter_continuous_sequences

from cfuser.logger import init_logger

logger = init_logger(__name__)


MAX_CHANNELS_PROC_GROUP = 20
ULYSSES_OFF = -100


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.all_process_groups: List[torch.distributed.ProcessGroup] = []
        self.all_process_groups_map: Dict[str, torch.distributed.ProcessGroup] = {}

        self.index_req = 0

        # NOTE As we disaggregate the mlp and attn, so here ULYSSES_PG and RING_PG are for Attn SP, NON_ATTN_SP_PG is for Non-Attn SP, just like DP for MLP

        self.ATTN_SP_PGs = [None] * MAX_CHANNELS_PROC_GROUP
        self.ULYSSES_PGs = [None] * MAX_CHANNELS_PROC_GROUP
        self.RING_PGs = [None] * MAX_CHANNELS_PROC_GROUP
        self.NON_ATTN_SP_PGs = [None] * MAX_CHANNELS_PROC_GROUP
        self.non_attn_ranks_list = [None] * MAX_CHANNELS_PROC_GROUP
        self.attn_ranks_list = [None] * MAX_CHANNELS_PROC_GROUP
        self.ring_ranks_list = [None] * MAX_CHANNELS_PROC_GROUP
        self.ulysses_ranks_list = [None] * MAX_CHANNELS_PROC_GROUP

        self.ATTN_SP_PG = None
        self.ULYSSES_PG = None
        self.RING_PG = None
        self.NON_ATTN_SP_PG = None
        self.attn_ranks = []
        self.non_attn_ranks = []

        self.generated = False

        self.custom_nccl_comm = None

    def get_ring_ranks(self, index_req: int = 0):
        assert index_req < len(self.ring_ranks_list), f"index_req {index_req} out of range"
        return self.ring_ranks_list[index_req]

    def get_ulysses_ranks(self, index_req: int = 0):
        assert index_req < len(self.ulysses_ranks_list), f"index_req {index_req} out of range"
        return self.ulysses_ranks_list[index_req]

    def get_attn_ranks(self, index_req: int = 0):
        assert index_req < len(self.attn_ranks_list), f"index_req {index_req} out of range"
        return self.attn_ranks_list[index_req]

    def get_non_attn_ranks(self, index_req: int = 0):
        assert index_req < len(self.non_attn_ranks_list), f"index_req {index_req} out of range"
        return self.non_attn_ranks_list[index_req]

    def get_non_attn_rank(self, index_req: int = 0):
        if self.get_non_attn_pg(index_req) == ULYSSES_OFF:
            return ULYSSES_OFF
        return self.get_non_attn_pg(index_req).rank()

    def get_non_attn_size(self, index_req: int = 0):
        if self.get_non_attn_pg(index_req) == ULYSSES_OFF:
            return len(self.get_non_attn_ranks(index_req))
        return self.get_non_attn_pg(index_req).size()

    def get_ulysses_rank(self, index_req: int = 0):
        if self.get_ulysses_pg(index_req) == ULYSSES_OFF:
            return ULYSSES_OFF
        return self.get_ulysses_pg(index_req).rank()

    def get_ring_rank(self, index_req: int = 0):
        if self.get_ring_pg(index_req) == ULYSSES_OFF:
            return ULYSSES_OFF
        return self.get_ring_pg(index_req).rank()

    def get_ulysses_size(self, index_req: int = 0):
        if self.get_ulysses_pg(index_req) == ULYSSES_OFF:
            return len(self.get_ulysses_ranks(index_req))
        return self.get_ulysses_pg(index_req).size()

    def get_ring_size(self, index_req: int = 0):
        if self.get_ring_pg(index_req) == ULYSSES_OFF:
            return len(self.get_ring_ranks(index_req))
        return self.get_ring_pg(index_req).size()

    def get_attn_pg(self, index_req: int = 0):
        assert index_req < len(self.ATTN_SP_PGs), f"index_req {index_req} out of range"
        return self.ATTN_SP_PGs[index_req]

    def get_non_attn_pg(self, index_req: int = 0):
        assert index_req < len(self.NON_ATTN_SP_PGs), f"index_req {index_req} out of range"
        return self.NON_ATTN_SP_PGs[index_req]

    def get_ulysses_pg(self, index_req: int = 0):
        assert index_req < len(self.ULYSSES_PGs), f"index_req {index_req} out of range"
        return self.ULYSSES_PGs[index_req]

    def get_ring_pg(self, index_req: int = 0):
        assert index_req < len(self.RING_PGs), f"index_req {index_req} out of range"
        return self.RING_PGs[index_req]

    def generate_all_process_groups(self, world_size):
        assert not self.generated, "all process groups are already generated"
        all_subsets = generate_all_subsets(world_size)
        # continuous_subsets = filter_continuous_sequences(all_subsets)
        continuous_subsets = all_subsets
        for subset in continuous_subsets:
            group = torch.distributed.new_group(subset, backend="nccl")
            self.all_process_groups.append(group)
            self.all_process_groups_map[str(subset)] = group
        try:
            # avoid circular import
            from cfuser.core.long_ctx_attention.pynccl_wrapper import PyNcclCommunicator
            from cfuser.core.distributed.parallel_state import get_world_group

            # Get the directory where globals.py is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            custom_nccl_path = os.path.join(current_dir, "libcustom_nccl_all2all.so")

            self.custom_nccl_comm = PyNcclCommunicator(
                self.all_process_groups_map[str(list(range(world_size)))],
                # device=torch.device(f"cuda:{get_world_group().rank}"),
                device=torch.device(f"cuda"),
                library_path="/usr/lib/x86_64-linux-gnu/libnccl.so.2",
                custom_nccl_library_path=custom_nccl_path,
            )
        except Exception as e:
            logger.info(f"Failed to create custom nccl communicator: {e}. You can use other backends.")
        self.generated = True

    def set_seq_parallel_pg(self, sp_name: str = "ulysses", ranks: List[int] = [], index_req: int = 0):
        assert self.generated, "all process groups are not generated"
        if sp_name == "ulysses":
            self.ULYSSES_PGs[index_req] = self.all_process_groups_map[str(ranks)]
            # logger.info(f"set ulysses pg {ranks}, {self.ULYSSES_PGs[index_req]} for index_req {index_req}")
            self.ulysses_ranks_list[index_req] = ranks
        elif sp_name == "ring":
            self.RING_PGs[index_req] = self.all_process_groups_map[str(ranks)]
            self.ring_ranks_list[index_req] = ranks
        elif sp_name == "non_attn_sp":
            self.NON_ATTN_SP_PGs[index_req] = self.all_process_groups_map[str(ranks)]
            self.non_attn_ranks_list[index_req] = ranks
        elif sp_name == "attn_sp":
            self.ATTN_SP_PGs[index_req] = self.all_process_groups_map[str(ranks)]
            self.attn_ranks_list[index_req] = ranks
        else:
            raise ValueError(f"Invalid sp_name: {sp_name}")

    def destroy(self):
        for pg in self.all_process_groups:
            torch.distributed.destroy_process_group(pg)


PROCESS_GROUP = ProcessGroupSingleton()


def set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, ranks, world_size, use_ulysses_low=True, index_req=0):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree
    (ulysses_degree, dp_degree)
    """
    sp_degree = sp_ring_degree * sp_ulysses_degree

    assert len(ranks) == world_size, f"len(ranks) {len(ranks)} != world_size {world_size}"
    assert world_size == sp_degree, f"attn world_size {world_size} != sp_degree {sp_degree}"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  # world_size // sp_ring_degree

    index_req = index_req

    PROCESS_GROUP.set_seq_parallel_pg(sp_name="attn_sp", ranks=ranks, index_req=index_req)

    ulysses_set = False
    ring_set = False
    if use_ulysses_low:
        for i in range(num_ulysses_pgs):
            ulysses_ranks_idx = list(
                range(
                    i * sp_ulysses_degree,
                    (i + 1) * sp_ulysses_degree,
                )
            )
            ulysses_ranks = [ranks[index_req] for index_req in ulysses_ranks_idx]
            if rank in ulysses_ranks:
                PROCESS_GROUP.set_seq_parallel_pg(sp_name="ulysses", ranks=ulysses_ranks, index_req=index_req)
                ulysses_set = True

            if not ulysses_set:
                PROCESS_GROUP.set_seq_parallel_pg(sp_name="ulysses", ranks=ulysses_ranks, index_req=index_req)

        for i in range(num_ring_pgs):
            ring_ranks_idx = list(range(i, sp_degree, num_ring_pgs))
            ring_ranks = [ranks[index_req] for index_req in ring_ranks_idx]
            if rank in ring_ranks:
                PROCESS_GROUP.set_seq_parallel_pg(sp_name="ring", ranks=ring_ranks, index_req=index_req)
                ring_set = True

            if not ring_set:
                PROCESS_GROUP.set_seq_parallel_pg(sp_name="ring", ranks=ring_ranks, index_req=index_req)
