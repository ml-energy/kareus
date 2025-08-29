import asyncio
from typing import List, Union, Dict, AsyncGenerator
from cfuser.core.utils.zmq_utils import (
    find_free_port,
)
from cfuser.engine.proc_launcher import launch_worker
from .struct import EngineOutput, RunnerOutput, ReqState
import multiprocessing as mp
from cfuser.config import InputConfig
from cfuser.scheduler import NaiveScheduler, ScalingEfficientScheduler, DecoupledScheduler
from cfuser.scheduler.request import ScheduledRequests
from cfuser.core.utils.zmq_utils import Router
from cfuser.config.args import ServerArgs
from cfuser.utils import get_gen_req_id
from cfuser.logger import init_logger


logger = init_logger(__name__)


def _set_evns_and_config():
    mp.set_start_method("spawn", force=True)


class _CServeEngine:

    def __init__(
        self,
        server_args: ServerArgs,
    ):
        self.engine_processes = []

        parallel_degree = server_args.nproc_per_node

        self.parallel_degree = parallel_degree
        self.zmq_router_port = find_free_port()

        self.master_addr = server_args.master_addr
        self.dist_master_port = find_free_port()

        _set_evns_and_config()

        for rank in range(parallel_degree):
            proc = mp.Process(
                target=launch_worker,
                args=(
                    rank,
                    parallel_degree,
                    self.zmq_router_port,
                    self.master_addr,
                    self.dist_master_port,
                    server_args.engine_config,
                ),
            )
            proc.start()
            self.engine_processes.append(proc)

        self.scheduler = DecoupledScheduler(
            ranks=list(range(parallel_degree)),
            schedule_logic=server_args.schedule_logic,
        )

        self.zmq_router = Router("localhost", self.zmq_router_port, parallel_degree)

        logger.info("Engine init done")

    async def init_zmq_router(self):
        await self.zmq_router.init()

    def add_requests(self, input_configs: Union[InputConfig, List[InputConfig]], request_ids: List[int] = None):

        if isinstance(input_configs, InputConfig):
            input_configs = [input_configs]

        if request_ids is None:
            if isinstance(input_configs, InputConfig):
                request_ids = [get_gen_req_id()]
            elif isinstance(input_configs, List):
                request_ids = [get_gen_req_id() for _ in range(len(input_configs))]
            elif input_configs is None:
                return
            else:
                raise ValueError("input_configs must be a single InputConfig or a list of InputConfig")
        elif input_configs is None:
            raise ValueError("input_configs cannot be None when specifying request_ids")

        assert len(request_ids) == len(input_configs), "request_ids and input_configs must have the same length"
        for request_id, config in zip(request_ids, input_configs):
            self.scheduler.add_request(request_id=request_id, input_config=config)

    def add_request(self, input_config: InputConfig, request_id: int = None):
        if request_id is None:
            request_id = get_gen_req_id()
        if isinstance(input_config, InputConfig):
            self.scheduler.add_request(request_id=request_id, input_config=input_config)
        else:
            raise ValueError("input_config must be a single InputConfig")

    async def handle_single_bunch_requests(self, sched_req: ScheduledRequests) -> RunnerOutput:

        input_packet = {
            "type": "generate",
            "Requests": sched_req,
        }

        await self.zmq_router.send_to_workers([f"Worker-{i}" for i in sched_req.non_attn_ranks], input_packet)

        worker_id, output = await self.zmq_router.recv_from_worker()
        if output.get("status") == "success":
            runner_output = output.get("output")
            self.scheduler.add_ranks(sched_req.non_attn_ranks)
        else:
            self.scheduler.add_ranks(sched_req.non_attn_ranks)
            raise Exception(f"Worker-{worker_id} failed to generate images")

        return runner_output

    async def step(self) -> AsyncGenerator[EngineOutput, None]:
        reqs = self.scheduler.schedule()

        if len(reqs) == 0:
            return

        # Create tasks for all requests concurrently
        pending = [asyncio.create_task(self.handle_single_bunch_requests(req)) for req in reqs]

        # Yield results as they complete
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    result = await task
                    yield result
                except Exception as e:
                    logger.error(f"Task failed with error: {e}")
                    continue

    async def engine_step(self):
        output_list = []
        async for output in self.step():
            if output is None:
                continue
            output_list.append(output)
        return output_list

    def __exit__(self, exc_type, exc_value, traceback):
        for process in self.engine_processes:
            process.terminate()


class CServeEngine(_CServeEngine):
    """
    Offline Sync Server version of `CServeRuntime`
    """

    def __init__(self, server_args: ServerArgs):
        super().__init__(server_args)
        asyncio.run(self.init_zmq_router())

    async def _run_engine(self, send_done_packet: bool = True):
        output_list = []
        tasks = []
        while self.scheduler.remaining_requests() > 0:
            task = asyncio.create_task(self.engine_step())
            tasks.append(task)
            await asyncio.sleep(0.3)

        for task in tasks:
            output_list.extend(await task)

        if send_done_packet:
            await self._send_done_packet()

        return output_list

    async def _send_done_packet(self):
        done_packet = {
            "type": "done",
        }
        await self.zmq_router.send_to_workers([f"Worker-{i}" for i in self.scheduler.ranks], done_packet)

    def generate(
        self, configs: Union[InputConfig, List[InputConfig]] = None, send_done_packet: bool = True
    ) -> List[EngineOutput]:
        self.add_requests(input_configs=configs)

        output_list = asyncio.run(self._run_engine(send_done_packet=send_done_packet))

        return output_list


class CServeRuntime:
    """
    Online Async Server version of `CServeEngine`
    """

    def __init__(self, server_args: ServerArgs):
        self.engine = _CServeEngine(server_args)
        self.req_states: Dict[int, ReqState] = {}  # request id -> request state
        self.request_outputs: Dict[int, EngineOutput] = {}  # request id -> request output
        self.log_requests = True

    async def init_zmq_router(self):
        await self.engine.init_zmq_router()

    async def engine_step(self):
        async for output in self.engine.step():
            if output is None:
                continue
            for request_id, image in zip(output.req_ids, output.images):
                self.req_states[request_id].output = EngineOutput(request_id, image)
                self.req_states[request_id].success = True
                self.req_states[request_id].event.set()

    async def generate(self, config: InputConfig, request_id: int = None):

        request_event = asyncio.Event()
        self.req_states[request_id] = ReqState(request_event, None, False)

        if self.log_requests:
            logger.info(f"Received request {request_id}: ")

        self.engine.add_request(input_config=config, request_id=request_id)

        while True:
            assert request_id in self.req_states, "request_id not found in req_states"

            if not self.engine.scheduler.running_full():
                asyncio.create_task(self.engine_step())

            try:
                await asyncio.wait_for(request_event.wait(), timeout=0.1)  # NOTE(@runyu): don't make this too big
            except asyncio.TimeoutError:
                # just for avoiding deadlock
                # https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/async_llm_engine.py#L153
                continue

            request_output = self.req_states[request_id].output
            return request_output
