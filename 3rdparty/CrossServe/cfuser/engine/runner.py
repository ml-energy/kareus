import os
from typing import Optional, Union, List

import numpy as np
import PIL.Image

import torch
from cfuser import cFuserFluxPipeline
from cfuser.config import EngineConfig, InputConfig
from cfuser.core.utils.zmq_utils import Dealer
from cfuser.core.distributed.parallel_state import set_runtime_config, init_distributed_environment
from cfuser.core.distributed.globals import PROCESS_GROUP
from cfuser.scheduler.request import ScheduledRequests
from .struct import RunnerOutput

from cfuser.logger import init_logger

logger = init_logger(__name__)


class CServeRunner:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        engine_config: EngineConfig,
        zmq_router_port: int = None,
        **kwargs,
    ):

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.rank = local_rank

        if not torch.distributed.is_initialized():
            init_distributed_environment(local_rank=self.rank, rank=self.rank, distributed_init_method="env://")

        # TODO(@lry89757): add support for other models, here we hardcode for Flux models, but we may support other models in the future.
        self.pipeline = cFuserFluxPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            engine_config=engine_config,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to(f"cuda")

        self.zmq_dealer: Optional[Dealer] = None
        if zmq_router_port is not None:
            self.zmq_dealer = Dealer(
                router_address="localhost",
                router_port=zmq_router_port,
                worker_id=f"Worker-{self.rank}",
            )

    def serve(self):
        assert self.zmq_dealer is not None, "zmq_dealer is not initialized"
        while True:
            request = self.zmq_dealer.recv_from_router()

            logger.info(f"rank {self.rank} received one request")

            if request.get("type") == "generate":
                reqs: ScheduledRequests = request.get("Requests")
                output = self.generate(reqs)

                if self.rank == reqs.non_attn_ranks[0]:
                    self.zmq_dealer.send_to_router(
                        {"status": "success", "output": RunnerOutput(images=output, req_ids=reqs.req_ids)}
                    )
            elif request.get("type") == "done":
                logger.info(f"rank {self.rank} finished")
                break
            else:
                self.zmq_dealer.send_to_router(
                    {
                        "status": "error",
                        "message": f"Invalid request type: {request.get('type')}",
                        "req_ids": reqs.req_ids,
                    }
                )

    def generate(self, reqs: ScheduledRequests) -> Union[List[PIL.Image.Image], np.ndarray]:

        self.change_runtime_config(reqs)

        input_configs = [req.input_config for req in reqs.requests]
        generators = [None] * len(input_configs)
        for i in range(len(input_configs)):
            if input_configs[i].seed is not None:
                generators[i] = torch.Generator(device="cuda").manual_seed(input_configs[i].seed)

        outputs = self.pipeline.inference_requests_batch(input_configs=input_configs, generators=generators)

        return outputs

    def change_runtime_config(self, reqs: ScheduledRequests):

        for i in range(len(reqs.requests)):
            # logger.info(f"rank {reqs.requests[i].attn_ranks} set runtime config for request {i}")
            logger.info(f"rank {self.rank} set runtime config for request {reqs.requests[i]}")
            set_runtime_config(
                ranks=reqs.requests[i].attn_ranks,
                ulysses_degree=reqs.requests[i].attn_ulysses_degree,
                ring_degree=reqs.requests[i].attn_ring_degree,
                non_attn_sp_ranks=reqs.requests[i].non_attn_ranks,
                index_req=i,
            )
        # torch.distributed.barrier(PROCESS_GROUP.get_non_attn_pg(index_req=0))
