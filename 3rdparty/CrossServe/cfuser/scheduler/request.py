from dataclasses import dataclass, field
from typing import List

from cfuser.config.config import InputConfig


@dataclass
class ScheduledRequest:
    req_ids: List[int]
    attn_ranks: List[int]
    non_attn_ranks: List[int]
    attn_ulysses_degree: int
    attn_ring_degree: int
    non_attn_sp_degree: int
    input_config: InputConfig = field(default_factory=InputConfig)
    estimated_time: float | None = None

    def __post_init__(self):
        # assert len(self.req_ids) == self.input_config.batch_size
        pass


@dataclass
class ScheduledRequests:
    """
    A list of scheduled requests, with the same non-attn ranks
    Because of the scaling efficiency of MLP and ATTN, we always assume that the non-attn parallel degree is always the same or larger than the attn parallel degree.
    """

    requests: List[ScheduledRequest] = field(default_factory=list)
    attn_ranks: List[int] = field(default_factory=list)
    non_attn_ranks: List[int] = field(default_factory=list)
    req_ids: List[int] = field(default_factory=list)
    estimated_time: float | None = None

    def __post_init__(self):
        for request in self.requests:
            self.attn_ranks.extend(request.attn_ranks)
            self.non_attn_ranks.extend(request.non_attn_ranks)
            self.req_ids.extend(request.req_ids)
        self.attn_ranks = list(set(sorted(self.attn_ranks)))
        self.non_attn_ranks = list(set(sorted(self.non_attn_ranks)))
        assert len(self.attn_ranks) <= len(self.non_attn_ranks)
        assert all(rank in self.non_attn_ranks for rank in self.attn_ranks)
