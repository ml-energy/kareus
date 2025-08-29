from abc import ABC
from typing import List, Union, Tuple
from copy import deepcopy
from cfuser.config.config import InputConfig
from .perf_model import PerfModel, CostEstimator
from .request import ScheduledRequest, ScheduledRequests
from dataclasses import dataclass

from cfuser.logger import init_logger

logger = init_logger(__name__)


class Scheduler(ABC):
    def __init__(self, ranks: List[int]):
        self.input_config_list: List[Tuple[int, InputConfig]] = []
        self.world_size = len(ranks)
        self.ranks = ranks

    def add_request(self, request_id: int, input_config: InputConfig):
        self.input_config_list.append((request_id, input_config))

    def add_ranks(self, ranks: List[int]):
        self.ranks += ranks
        self.ranks = list(set(self.ranks))
        self.ranks.sort()

    def remove_ranks(self, ranks: List[int]):
        self.ranks = [rank for rank in self.ranks if rank not in ranks]
        self.ranks.sort()

    def running_full(self):
        """
        Check if all workers are running full
        """
        return len(self.ranks) == 0

    def schedule(self):
        """
        according to current queue, we should decide which request should be sent to which worker
        """
        raise NotImplementedError

    def remaining_requests(self):
        return len(self.input_config_list)


class NaiveScheduler(Scheduler):
    """
    FCFS Scheduler without any perf model guidance
    """

    def __init__(self, ranks: List[int]):
        super().__init__(ranks)

    def schedule(self):
        # naive scheduler, just send the first request to the first worker
        if self.remaining_requests() == 0 or len(self.ranks) == 0:
            return []
        request_id, input_config = self.input_config_list.pop(0)
        ranks = self.ranks
        req = ScheduledRequest(
            req_ids=[request_id],
            input_config=input_config,
            attn_ranks=ranks,
            non_attn_ranks=ranks,
            # attn_ulysses_degree=len(ranks),
            # attn_ring_degree=1,
            attn_ulysses_degree=1,
            attn_ring_degree=len(ranks),
            non_attn_sp_degree=len(ranks),
        )

        return [ScheduledRequests(requests=[req], estimated_time=req.estimated_time)]


class ScalingEfficientScheduler(Scheduler):
    """
    According to the scaling efficiency, select the best parallel configuration for each request.
    FCFS scheduler is used here."""

    def __init__(self, ranks: List[int], perf_model: Union[PerfModel, str]):
        logger.warning(f"ScalingEfficientScheduler is deprecated, please use DecoupledScheduler instead")
        if isinstance(perf_model, str):
            perf_model = PerfModel(perf_model)
        elif isinstance(perf_model, PerfModel):
            perf_model = perf_model
        else:
            raise ValueError("perf_model must be a PerfModel or a path to a csv file")

        super().__init__(ranks)
        self.perf_model: PerfModel = perf_model

    def schedule(self):
        """
        Currently, we only consider the FCFS and don't consider overall tpt.
        TODO(Junze Ma): consider overall tpt in the future, improve the scheduler.
        """
        return self.fcfs_schedule()

    def fcfs_schedule(self):

        if self.remaining_requests() == 0 or len(self.ranks) == 0:
            return []

        ret_req_list = []

        while self.remaining_requests() > 0 and self.ranks:
            request_id, input_config = self.input_config_list.pop(0)
            batch_size = input_config.batch_size
            height = input_config.height
            width = input_config.width
            latency_threshold = input_config.latency_threshold

            # Find best configuration for this input_config
            best_config = self.perf_model.find_best_degrees(
                target_batch_size=batch_size,
                target_height=height,
                target_width=width,
                target_num_inference_steps=input_config.num_inference_steps,
                latency_threshold=latency_threshold,
            )

            if best_config is None:
                self.input_config_list.insert(0, (request_id, input_config))
                continue

            gpus_per_group = best_config.ulysses_degree * best_config.ring_degree

            remaining_ranks = deepcopy(self.ranks)
            if gpus_per_group > len(remaining_ranks):
                # TODO(Junze Ma): need to improve this part
                req = ScheduledRequest(
                    req_ids=[request_id],
                    input_config=input_config,
                    attn_ranks=remaining_ranks,
                    non_attn_ranks=remaining_ranks,
                    attn_ulysses_degree=len(remaining_ranks),
                    attn_ring_degree=1,
                    non_attn_sp_degree=len(remaining_ranks),
                    estimated_time=best_config.estimated_time,
                )
                self.remove_ranks(remaining_ranks)
            else:
                req = ScheduledRequest(
                    req_ids=[request_id],
                    input_config=input_config,
                    attn_ranks=remaining_ranks[:gpus_per_group],
                    non_attn_ranks=remaining_ranks[:gpus_per_group],
                    attn_ulysses_degree=best_config.ulysses_degree,
                    attn_ring_degree=best_config.ring_degree,
                    non_attn_sp_degree=gpus_per_group,
                    estimated_time=best_config.estimated_time,
                )
                self.remove_ranks(remaining_ranks[:gpus_per_group])

            ret_req_list.append(req)

        # logger.info(f"schedule {ret_req_list}")
        return [ScheduledRequests(requests=[ret_req]) for ret_req in ret_req_list]


class DecoupledScheduler(Scheduler):
    """
    Decoupled scaling efficient scheduler, which means each request's Attn and Non-Attn are dispatched to different workers.
    FCFS scheduler is used here.
    """

    def __init__(
        self,
        ranks: List[int],
        comm_scaling_efficiency_path: str = "log/benchmark/component_scaling_efficiency/comm_scaling_efficiency/comm_scaling_efficiency.json",
        ring_scaling_efficiency_path: str = "log/benchmark/component_scaling_efficiency/ring_scaling_efficiency/ring_scaling_efficiency.json",
        non_attn_scaling_efficiency_path: str = "log/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json",
        schedule_logic: str = "disaggregated_scaling_efficient",
    ):
        super().__init__(ranks)

        # NOTE: run benchmark/component_scaling_efficiency to get the cost_estimator json file
        self.cost_estimator = CostEstimator(
            num_gpus=len(ranks),
            ring_attn_scaling_efficiency_path=ring_scaling_efficiency_path,
            non_attn_scaling_efficiency_path=non_attn_scaling_efficiency_path,
        )
        self.schedule_logic = schedule_logic

    def schedule(self):
        """
        Currently it is a random scheduler, only test the runnable of disaggregated sp.
        TODO(Junze Ma): improve the scheduler, dynamic programming.
        """
        if self.schedule_logic == "disaggregated_scaling_efficient":
            return self.fcfs_schedule()
        elif self.schedule_logic == "scaling_efficient":
            return self.fcfs_scaling_schedule()
        elif self.schedule_logic == "naive":
            return self.fcfs_naive_schedule()
        else:
            raise ValueError(f"Invalid schedule_logic: {self.schedule_logic}")

    def dyn_packing_schedule(self):
        """
        TODO(@Jeff): dynamic programming to find the best packing strategy
        """
        raise NotImplementedError

    def fcfs_naive_schedule(self):
        # naive scheduler, just send the first request to all workers
        if self.remaining_requests() == 0 or len(self.ranks) == 0:
            return []
        request_id, input_config = self.input_config_list.pop(0)
        ranks = self.ranks
        est_time, u_degree, r_degree = self.cost_estimator.naive_cost(
            bs=input_config.batch_size,
            seq_len=input_config.height * input_config.width // 16 // 16,
            hc=24,  # flux
            hs=128,  # flux
            gpu_num=len(ranks),
        )
        req = ScheduledRequest(
            req_ids=[request_id],
            input_config=input_config,
            attn_ranks=ranks,
            non_attn_ranks=ranks,
            # attn_ulysses_degree=len(ranks),
            # attn_ring_degree=1,
            attn_ulysses_degree=u_degree,
            attn_ring_degree=r_degree,
            non_attn_sp_degree=len(ranks),
        )
        self.remove_ranks(ranks)

        return [ScheduledRequests(requests=[req], estimated_time=est_time * input_config.num_inference_steps)]

    def fcfs_scaling_schedule(self):
        if self.remaining_requests() == 0 or len(self.ranks) == 0:
            return []

        ret_req_list = []

        while self.remaining_requests() > 0 and self.ranks:
            request_id, input_config = self.input_config_list.pop(0)
            batch_size = input_config.batch_size
            height = input_config.height
            width = input_config.width
            latency_threshold = input_config.latency_threshold

            # find the cost-efficient configuration for this input_config
            min_cost, u_degree, r_degree = self.cost_estimator.scaling_efficiency(
                bs=batch_size,
                seq_len=height * width // 16 // 16,
                hc=24,  # flux
                hs=128,  # flux
                strategy="economy",
            )

            estimated_time = min_cost * input_config.num_inference_steps

            gpus_per_group = u_degree * r_degree
            remaining_ranks = deepcopy(self.ranks)
            if gpus_per_group > len(remaining_ranks):
                # self.input_config_list.insert(0, (request_id, input_config))
                req = ScheduledRequest(
                    req_ids=[request_id],
                    input_config=input_config,
                    attn_ranks=remaining_ranks,
                    non_attn_ranks=remaining_ranks,
                    attn_ulysses_degree=len(remaining_ranks),
                    attn_ring_degree=1,
                    non_attn_sp_degree=len(remaining_ranks),
                    estimated_time=estimated_time,
                )
                self.remove_ranks(remaining_ranks)
            else:
                req = ScheduledRequest(
                    req_ids=[request_id],
                    input_config=input_config,
                    attn_ranks=remaining_ranks[:gpus_per_group],
                    non_attn_ranks=remaining_ranks[:gpus_per_group],
                    attn_ulysses_degree=u_degree,
                    attn_ring_degree=r_degree,
                    non_attn_sp_degree=gpus_per_group,
                    estimated_time=estimated_time,
                )
                self.remove_ranks(remaining_ranks[:gpus_per_group])

            ret_req_list.append(req)

        out_req_lists = []
        for ret_req in ret_req_list:
            out_req_lists.append(ScheduledRequests(requests=[ret_req], estimated_time=ret_req.estimated_time))

        return out_req_lists

    def fcfs_schedule(self):
        if self.remaining_requests() == 0 or len(self.ranks) == 0:
            return []

        ret_req_list = []
        ret_req_lists = []

        attn_remaining_ranks = []
        non_attn_ranks = []

        # logger.info(f"self.ranks: {self.ranks}")
        while self.remaining_requests() > 0 and self.ranks:

            request_id, input_config = self.input_config_list.pop(0)

            estimate_cost, mlp_min_gpus, attn_u_degree, attn_r_degree = self.cost_estimator.disaggregated_scaling(
                bs=input_config.batch_size,
                seq_len=input_config.height * input_config.width // 16 // 16,
                hc=24,  # flux
                hs=128,  # flux
                strategy="economy",
            )

            # it is wrong here
            if attn_u_degree * attn_r_degree > len(self.ranks) and attn_u_degree * attn_r_degree > len(
                attn_remaining_ranks
            ):
                self.input_config_list.insert(0, (request_id, input_config))
                # TODO(Runyu): actually here we should try other remaining requests
                return []

            assert (
                mlp_min_gpus >= attn_u_degree * attn_r_degree
            ), f"mlp_min_gpus should be greater than or equal to attn_u_degree * attn_r_degree, input_config: {input_config}"

            if len(attn_remaining_ranks) >= attn_u_degree * attn_r_degree:
                attn_ranks = attn_remaining_ranks[: attn_u_degree * attn_r_degree]
                attn_remaining_ranks = attn_remaining_ranks[attn_u_degree * attn_r_degree :]
            else:
                non_attn_ranks = self.ranks[:mlp_min_gpus]
                attn_ranks = non_attn_ranks[: attn_u_degree * attn_r_degree]
                attn_remaining_ranks = non_attn_ranks[attn_u_degree * attn_r_degree :]

            req = ScheduledRequest(
                req_ids=[request_id],
                input_config=input_config,
                attn_ranks=attn_ranks,
                non_attn_ranks=non_attn_ranks,
                attn_ulysses_degree=attn_u_degree,
                attn_ring_degree=attn_r_degree,
                non_attn_sp_degree=mlp_min_gpus,
            )

            ret_req_list.append(req)

            if len(attn_remaining_ranks) == 0:
                self.remove_ranks(non_attn_ranks)
                ret_req_lists.append(ret_req_list)
                attn_remaining_ranks = []
                ret_req_list = []

        # if there are remaining requests, we should add them back to the input_config_list
        if attn_remaining_ranks:
            ret_req_lists.append(ret_req_list)
            self.remove_ranks(ret_req_list[0].non_attn_ranks)

        out_req_lists = []
        for ret_req_list in ret_req_lists:
            out_req_lists.append(
                ScheduledRequests(requests=ret_req_list, estimated_time=self.compute_estimated_time(ret_req_list))
            )

        return out_req_lists

    def compute_estimated_time(self, requests: List[ScheduledRequest]):
        """
        Compute the estimated time for a given list of requests.
        """

        hc = 24
        hs = 128
        num_layers = 19
        num_single_layers = 38

        max_attn_time = 0.0
        non_attn_time = 0.0
        steps = requests[0].input_config.num_inference_steps

        for request in requests:

            assert request.input_config.num_inference_steps == steps

            bs = request.input_config.batch_size
            seq_len = request.input_config.height * request.input_config.width // 16 // 16
            non_attn_sp_degree = request.non_attn_sp_degree
            attn_u_degree = request.attn_ulysses_degree
            attn_r_degree = request.attn_ring_degree

            single_attn_time = self.cost_estimator.ring_attn_dict.get(
                (bs, seq_len, hc, hs, attn_u_degree, attn_r_degree)
            )

            max_attn_time = max(max_attn_time, single_attn_time * (num_layers + num_single_layers))

            multimodal_prologue, unimodal_prologue, multimodal_epilogue, unimodal_epilogue = (
                self.cost_estimator.non_attn_dict.get((bs, seq_len, hc, hs, non_attn_sp_degree))
            )
            non_attn_time += (multimodal_prologue + multimodal_epilogue) * num_layers + (
                unimodal_epilogue + unimodal_prologue
            ) * num_single_layers

        estimated_time = (max_attn_time + non_attn_time) * steps

        return estimated_time


@dataclass
class DisaggregatedConfig:
    attn_u_degree: int
    attn_r_degree: int
    attn_total_degree: int


class DynamicDisaggregatedScalingEfficientScheduler(DecoupledScheduler):
    """
    Disaggregated scaling efficient scheduler, which means each request's Attn and Non-Attn are dispatched to different workers.
    DP scheduler is used here.
    """

    def __init__(
        self,
        ranks: List[int],
        ulysses_scaling_efficiency_path: str = None,
        ring_scaling_efficiency_path: str = None,
        non_attn_scaling_efficiency_path: str = None,
        schedule_logic: str = "disaggregated_scaling_efficient",
    ):
        super().__init__(ranks)
        self.input_config_list: List[Tuple[int, InputConfig, DisaggregatedConfig]] = []

        # assert ulysses_scaling_efficiency_path is not None, "ulysses_scaling_efficiency_path is required"
        # assert ring_scaling_efficiency_path is not None, "ring_scaling_efficiency_path is required"
        # assert non_attn_scaling_efficiency_path is not None, "non_attn_scaling_efficiency_path is required"

        # NOTE: run benchmark/component_scaling_efficiency to get the cost_estimator json file
        self.cost_estimator = CostEstimator(
            num_gpus=len(ranks),
            ring_attn_scaling_efficiency_path="log/benchmark/component_scaling_efficiency/ring_scaling_efficiency/ring_scaling_efficiency.json",
            non_attn_scaling_efficiency_path="log/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json",
        )
        self.schedule_logic = schedule_logic

    def add_request(self, request_id: int, input_config: InputConfig):
        _, mlp_min_gpus, attn_u_degree, attn_r_degree = self.cost_estimator.disaggregated_scaling(
            bs=input_config.batch_size,
            seq_len=input_config.height * input_config.width // 16 // 16,
            hc=24,  # flux
            hs=128,  # flux
        )
        assert (
            mlp_min_gpus == self.world_size
        ), f"mlp_min_gpus should be equal to self.world_size, input_config: {input_config}"

        print(
            f"request_id: {request_id}, attn_u_degree: {attn_u_degree}, attn_r_degree: {attn_r_degree}, attn_total_degree: {attn_u_degree * attn_r_degree}"
        )
        self.input_config_list.append(
            (request_id, input_config, DisaggregatedConfig(attn_u_degree, attn_r_degree, attn_u_degree * attn_r_degree))
        )

    def _best_choice(self, index_req: List[List[int]]):
        # Given a list of choices, return the best one
        # Use as many GPU as possible then try to FCFS
        num_gpus = [sum([self.input_config_list[i][2].attn_total_degree for i in b]) for b in idx]
        return idx[num_gpus.index(max(num_gpus))]

    def schedule(self):
        """
        Currently it is a random scheduler, only test the runnable of disaggregated sp.
        """
        if self.schedule_logic == "dynamic_disaggregated_scaling_efficient":
            return self.dyn_packing_schedule()
        else:
            raise ValueError(f"Invalid schedule_logic: {self.schedule_logic}")

    def dyn_packing_schedule(self):
        reqs = self.input_config_list
        capacity = len(self.ranks)

        # knapsack over number of gpus
        dp = [[[] for _ in range(capacity + 1)] for _ in range(len(reqs) + 1)]
        for i in range(len(reqs) + 1):
            for j in range(capacity + 1):
                if i == 0 or j == 0:
                    continue
                elif j >= reqs[i - 1][2].attn_total_degree:
                    dp[i][j] = self._best_choice(
                        [dp[i - 1][j], dp[i - 1][j - reqs[i - 1][2].attn_total_degree] + [i - 1]]
                    )
                else:
                    dp[i][j] = dp[i - 1][j]

        return dp[len(reqs)][capacity]
