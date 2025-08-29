import time
import csv
import numpy as np
import asyncio
from benchmark_serving import PoissonProcess, GammaProcess, Workload, ModelStatsResult, DEFAULT_WARMUP
from cfuser.scheduler.scheduler import NaiveScheduler, DecoupledScheduler
from cfuser.scheduler.request import ScheduledRequests
from cfuser.config import InputConfig
from typing import List, Union, Dict, AsyncGenerator
from dataclasses import asdict
from cfuser.config.args import ServerArgs
from cfuser.utils import get_gen_req_id
from cfuser.engine.struct import EngineOutput, ReqState, RunnerOutput
from cfuser.logger import init_logger

logger = init_logger(__name__)

SCALE_FACTOR = 3


class _FakeEngine:
    def __init__(self, schedule_logic: str = "naive", parallel_degree: int = 1):

        self.schedule_logic = schedule_logic

        LOG_DIR = "log_A100x4_80GB/benchmark/component_scaling_efficiency"
        # LOG_DIR = "log_4xA40_48GB/benchmark/component_scaling_efficiency"
        self.scheduler = DecoupledScheduler(
            ranks=list(range(parallel_degree)),
            ring_scaling_efficiency_path=f"{LOG_DIR}/ring_scaling_efficiency/ring_scaling_efficiency.json",
            non_attn_scaling_efficiency_path=f"{LOG_DIR}/non_attn_efficiency/non_attn_efficiency.json",
            schedule_logic=schedule_logic,
        )

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

        # logger.info(f"sched_req {sched_req.req_ids} estimated_time: {sched_req.estimated_time}")
        await asyncio.sleep(sched_req.estimated_time / SCALE_FACTOR)
        self.scheduler.add_ranks(sched_req.non_attn_ranks)

        return RunnerOutput(sched_req.req_ids, [np.zeros((1, 1, 1, 1)) for _ in range(len(sched_req.req_ids))])

    async def step(self) -> AsyncGenerator[EngineOutput, None]:
        # logger.info(f"ranks: {self.scheduler.ranks}")
        reqs = self.scheduler.schedule()
        if len(reqs) == 0:
            return

        for sched_req in reqs:
            logger.info(
                f"sched_req {sched_req.req_ids} for {sched_req.non_attn_ranks} estimated_time: {sched_req.estimated_time}"
            )

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


class FakeRuntime:
    """
    Online Async Server version of `CServeEngine`
    """

    def __init__(self, schedule_logic: str = "naive", parallel_degree: int = 1):
        self.engine = _FakeEngine(schedule_logic, parallel_degree)
        self.req_states: Dict[int, ReqState] = {}  # request id -> request state
        self.request_outputs: Dict[int, EngineOutput] = {}  # request id -> request output
        self.log_requests = False

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
            logger.info(f"Received request {request_id}: {config}")

        self.engine.add_request(input_config=config, request_id=request_id)

        while True:
            assert request_id in self.req_states, "request_id not found in req_states"

            if not self.engine.scheduler.running_full():
                asyncio.create_task(self.engine_step())

            try:
                await asyncio.wait_for(request_event.wait(), timeout=0.2)  # NOTE(@runyu): don't make this too big
            except asyncio.TimeoutError:
                # just for avoiding deadlock
                # https://github.com/vllm-project/vllm/blob/32b6816e556f69f1672085a6267e8516bcb8e622/vllm/engine/async_llm_engine.py#L153
                continue

            request_output = self.req_states[request_id].output
            return request_output


async def submit_workload(runtime: FakeRuntime, workload: Workload):
    loop = asyncio.get_event_loop()
    start_time = loop.time()

    tasks = []
    for i in range(len(workload.arrivals)):
        arrival_time = workload.arrivals[i]
        request = workload.requests[i]
        send_time = start_time + arrival_time

        async def scheduled_request(send_time, request, arrival_time):
            now = loop.time()
            if send_time > now:
                await asyncio.sleep(send_time - now)
            else:
                logger.info("Request is too early")

            logger.info(
                f"Sending req {request.req_idx} bs: {request.data['batch_size']} height: {request.data['height']} width: {request.data['width']} scheduled at {arrival_time * SCALE_FACTOR:.3f}s"
            )

            input_config = InputConfig(**request.data)
            request_output = await runtime.generate(input_config, request.req_idx)

            completion_time = loop.time()

            logger.info(
                f"Request {request.req_idx} bs: {input_config.batch_size} height: {input_config.height} width: {input_config.width} completed at {(completion_time - start_time) * SCALE_FACTOR:.3f}s"
            )

            request.submit_time = (send_time - start_time) * SCALE_FACTOR
            request.completion_time = (completion_time - start_time) * SCALE_FACTOR

            return request.completion_time

        task = asyncio.create_task(scheduled_request(send_time, request, arrival_time))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    return results


def get_workload(arrival_rate: float, duration: float, seed: int = 0) -> Workload:
    """Generate a synthetic workload with given parameters"""
    workload = PoissonProcess(arrival_rate).generate_workload("", start=0, duration=duration, seed=seed)
    return workload


def main(args):

    global SCALE_FACTOR
    SCALE_FACTOR = args.scaling_factor
    workload = get_workload(arrival_rate=args.arrival_rate, duration=args.duration, seed=args.seed)

    logger.info(f"workload: {workload}")
    workload = Workload([arrival_time / SCALE_FACTOR for arrival_time in workload.arrivals], workload.requests)
    workload.rate = workload.rate / SCALE_FACTOR

    # if args.shape_dist:
    #     try:
    #         with open(args.shape_dist, mode="r", encoding="utf-8") as csvfile:
    #             reader = csv.DictReader(csvfile)
    #             raw_shape_distribution = [
    #                 (int(row["width"]), int(row["height"]), float(row["percentage"])) for row in reader
    #             ]
    #             # recalculate the shape percentage
    #             total_percentage = sum(x[2] for x in raw_shape_distribution)
    #             shape_distribution = [((x[0], x[1]), x[2] / total_percentage) for x in raw_shape_distribution]
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"File '{args.shape_dist}' was not found.")
    #     except csv.Error as e:
    #         raise ValueError(f"Error reading the CSV file: {e}")
    # else:
    #     shape_distribution = [((args.width, args.height), 1.0)]

    # shape_distribution = [((512, 512), 0.4), ((1024, 1024), 0.5), ((2048, 2048), 0.1)]
    shape_distribution = [((1024, 1024), 1)]
    # shape_distribution = [((512, 512), 1)]
    batch_size_distribution = [(1, 0.1), (2, 0.5), (4, 0.4)]

    for req in workload.requests:
        shape_idx = np.random.choice(range(len(shape_distribution)), p=[x[1] for x in shape_distribution])
        shape = shape_distribution[shape_idx][0]

        input_config = InputConfig(
            batch_size=2,
            height=shape[0],
            width=shape[1],
            prompt="test prompt",
            num_inference_steps=30,
            output_type="latent",
        )

        req.data = asdict(input_config)

    runtime = FakeRuntime(schedule_logic=args.schedule_logic, parallel_degree=4)

    completion_times = asyncio.run(submit_workload(runtime, workload))

    workload.arrivals = [x * SCALE_FACTOR for x in workload.arrivals]

    stats = workload.compute_stats(completion_times, 3)

    workload.print_stats(stats)


"""
python benchmark/serving/serving_simulator.py --arrival_rate 0.3 --duration 30 --seed 0 --scaling_factor 10
python benchmark/serving/serving_simulator.py --arrival_rate 0.3 --duration 30 --seed 0 --scaling_factor 1
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arrival_rate", type=float, default=0.3)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schedule_logic", type=str, default="disaggregated_scaling_efficient")
    parser.add_argument("--scaling_factor", type=int, default=1)
    args = parser.parse_args()

    # time_start = time.time()
    main(args)
    # time_end = time.time()
    # logger.info(f"Time taken: {time_end - time_start:.3f}s")
