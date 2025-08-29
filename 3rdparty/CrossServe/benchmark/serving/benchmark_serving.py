"""Workload definition
Borrowed from https://github.com/alpa-projects/mms/blob/main/alpa_serve/simulator/workload.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
import dataclasses
from typing import Any, List, Optional
from cfuser.config import InputConfig
import csv
import numpy as np
from collections import defaultdict
from dataclasses import asdict
import argparse

import asyncio
import aiohttp

import torch
from pathlib import Path
import json

DEFAULT_WARMUP = 5
eps = 1e-6


def to_str_round(x: Any, decimal: int = 6):
    """Print a python object but round all floating point numbers."""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        tmp_str = ", ".join([to_str_round(y, decimal=decimal) for y in x])
        return "[" + tmp_str + "]"
    if isinstance(x, dict):
        return str({k: to_str_round(v, decimal=decimal) for k, v in x.items()})
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, (float, np.floating)):
        format_str = f"%.{decimal}f"
        return format_str % x
    if x is None:
        return str(x)
    raise ValueError("Invalid value: " + str(x))


@dataclasses.dataclass
class Request:
    """A single request."""

    model_name: str
    data: dict | None
    slo: Optional[float]
    req_idx: int
    submit_time: float | None = None  # This will be filled later
    completion_time: float | None = None  # This will be filled later


ModelStatsResult = namedtuple(
    "ModelStatsResult",
    (
        "name",
        "num_requests",
        "throughput",
        "latency_mean",
        "latency_std",
        "latency_p90",
        "latency_p99",
        "latency",
        "request_starts",
        "request_finishes",
        "request_rate",
        "shape_distribution",
    ),
)


class ArrivalProcess(ABC):
    @abstractmethod
    def rate(self):
        """Return the mean arrival rate."""
        raise NotImplementedError()

    @abstractmethod
    def cv(self):
        """Return the coefficient of variation of the gap between
        the requests."""
        raise NotImplementedError()

    @abstractmethod
    def generate_arrivals(self, start: float, duration: float, seed: int = 0):
        raise NotImplementedError()

    @abstractmethod
    def generate_workload(
        self,
        model_name: str,
        start: float,
        duration: float,
        slo: Optional[float] = None,
        seed: int = 0,
    ):
        """Generate a workload with the arrival process.

        Args:
            model_name (str): Name of the model.
            start (float): The start time of the workload.
            duration (float): The duration of the workload.
            slo (Optional[float]): The service level objective of each model.
            seed (int): The random seed.
        """
        raise NotImplementedError()

    def __str__(self):
        return f"{self.__class__.__name__}(" f"rate={self.rate()}, " f"cv={self.cv()})"

    def params(self):
        return self.rate(), self.cv()


class DeterministicProcess(ArrivalProcess):
    """Deterministic arrival process."""

    def __init__(self, arrival_rate: float):
        """Create a deterministic arrival process.

        Args:
            arrival_rate (float): The arrival rate of the process. The gap
                between the requests is 1 / arrival_rate seconds.
        """
        self.rate_ = arrival_rate

    def rate(self):
        return self.rate_

    def cv(self):
        return 0

    def generate_workload(
        self,
        model_name: str,
        start: float,
        duration: float,
        slo: Optional[float] = None,
        seed: int = 0,
    ):
        n_requests = int(duration * self.rate_)
        interval = 1 / self.rate_
        ticks = [start + i * interval for i in range(n_requests)]
        return Workload(ticks, [Request(model_name, None, slo, i, None, None) for i in range(n_requests)])


class GammaProcess(ArrivalProcess):
    """Gamma arrival process."""

    def __init__(self, arrival_rate: float, cv: float):
        """Initialize a gamma arrival process.

        Args:
            arrival_rate: mean arrival rate.
            cv: coefficient of variation. When cv == 1, the arrival process is
                Poisson process.
        """
        self.rate_ = arrival_rate
        self.cv_ = cv
        self.shape = 1 / (cv * cv)
        self.scale = cv * cv / arrival_rate

    def rate(self):
        return self.rate_

    def cv(self):
        return self.cv_

    def generate_arrivals(self, start: float, duration: float, seed: int = 0):
        np.random.seed(seed)

        batch_size = max(int(self.rate_ * duration * 1.2), 1)
        intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
        pt = 0

        ticks = []
        cur = start + intervals[0]
        end = start + duration
        while cur < end:
            ticks.append(cur)

            pt += 1
            if pt >= batch_size:
                intervals = np.random.gamma(self.shape, self.scale, size=batch_size)
                pt = 0

            cur += intervals[pt]

        return ticks

    def generate_workload(
        self,
        model_name: str,
        start: float,
        duration: float,
        slo: Optional[float] = None,
        seed: int = 0,
    ):
        ticks = self.generate_arrivals(start, duration, seed)
        return Workload(ticks, [Request(model_name, None, slo, i, None, None) for i in range(len(ticks))])


class PoissonProcess(GammaProcess):
    """Poisson arrival process."""

    def __init__(self, arrival_rate: float):
        """Initialize a Poisson arrival process.

        Args:
            arrival_rate: The mean arrival rate.
        """
        super().__init__(arrival_rate, 1)


class Workload:
    """A sorted list of requests."""

    def __init__(self, arrivals: List[float], requests: List[Request]):
        assert len(arrivals) == len(requests)

        self.arrivals = np.array(arrivals)
        self.requests = requests

        self.enable_simulator_cache = False
        self.cached_data = None

        if len(self.arrivals) > 1:
            intervals = self.arrivals[1:] - self.arrivals[:-1]
            self.rate = 1 / (np.mean(intervals) + eps)
            self.cv = np.std(intervals) * self.rate
        else:
            self.rate = 0
            self.cv = 0

    def split_round_robin(self, number: int):
        rets = []
        for i in range(number):
            rets.append(self[i::number])
        return rets

    def split_time_interval(self, interval: float):
        if len(self.arrivals) < 1:
            return []

        ws = []
        start_i = 0
        start_time = self.arrivals[start_i]
        for i in range(len(self.arrivals)):
            if self.arrivals[i] > start_time + interval:
                ws.append(self[start_i:i])
                start_i = i
                start_time = self.arrivals[i]

        ws.append(self[start_i:])
        return ws

    def compute_stats(
        self,
        completion_times: List[float],
        warmup: float,
    ):
        """Compute the statistics of serving results."""
        start = self.arrivals
        finish = np.asarray(completion_times)
        # Skip the first and last `warmup` seconds
        if len(self.arrivals) > 1:
            skip = int(warmup / (self.arrivals[-1] - self.arrivals[0]) * len(self.arrivals))
            if skip > 0:
                start = start[skip:-skip]
                finish = finish[skip:-skip]
                requests = self.requests[skip:-skip]
            else:
                requests = self.requests
        if not len(start):
            return ModelStatsResult("", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        # Compute stats
        name = requests[0].model_name

        # Compute stats
        throughput = len(finish) / (finish[-1] - start[0])
        latency = finish - start

        sorted_latency = np.sort(latency)
        latency_p90 = sorted_latency[int(0.90 * len(sorted_latency))]
        latency_p99 = sorted_latency[int(0.99 * len(sorted_latency))]

        shape_distribution = defaultdict(int)

        for req in requests:
            if req.data is not None:
                shape = (req.data["width"], req.data["height"])
                shape_distribution[shape] += 1

        return ModelStatsResult(
            name,
            len(start),
            throughput,
            np.mean(latency),
            np.std(latency),
            latency_p90,
            latency_p99,
            latency,
            start,
            finish,
            len(start) / (start[-1] - start[0]),
            [(k, v / len(start)) for k, v in shape_distribution.items()],
        )

    @staticmethod
    def print_stats(stats: ModelStatsResult):
        """Print the statistics of serving results."""
        print(f"model: {stats.name}, " f"#req: {stats.num_requests}, " f"rate: {stats.request_rate:.2f} q/s")
        print(f"throughput: {stats.throughput:.2f} q/s, ")
        print(
            f"latency mean: {stats.latency_mean:.2f} s, "
            f"std: {stats.latency_std:.2f} s, "
            f"p90: {stats.latency_p90:.2f} s"
        )

    @classmethod
    def empty(cls):
        return cls([], [])

    @classmethod
    def merge(cls, *args):
        if len(args) == 1:
            return args[0]

        number = sum(len(x) for x in args)

        merged_arrivals = np.concatenate(tuple(x.arrivals for x in args))
        merged_requests = sum((x.requests for x in args), [])

        sorted_indices = np.argsort(merged_arrivals)

        arrivals = [None] * number
        requests = [None] * number

        for i, j in enumerate(sorted_indices):
            arrivals[i] = merged_arrivals[j]
            requests[i] = merged_requests[j]
            requests[i].req_idx = i

        return cls(arrivals, requests)

    def __getitem__(self, key):
        if isinstance(key, slice):
            arrivals = self.arrivals.__getitem__(key)
            requests = self.requests.__getitem__(key)
            return Workload(arrivals, requests)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        return Workload.merge(self, other)

    def __len__(self):
        return len(self.arrivals)

    def __str__(self):
        return (
            f"Workload(len={len(self)}, "
            f"rate={self.rate:.2f}, "
            f"CV={self.cv:.2f}, "
            f"tstamps={to_str_round(self.arrivals[:20])} ...)"
        )


def read_column_from_csv(filepath: str, column: str):
    results = []
    try:
        with open(filepath, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(f"File does not contain a {column} column.")
            for row in reader:
                r = row.get(column)
                if r:
                    results.append(r)
        if not len(results):
            raise ValueError(f"No data found in the {column} column.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filepath}' was not found.")
    except csv.Error as e:
        raise ValueError(f"Error reading the CSV file: {e}")
    return results


def populate_requests(
    workload: Workload,
    dataset: str | None = None,
    shape_distribution: List[tuple[tuple[int, int], float]] = [((1024, 1024), 1)],
    num_inference_steps: int = 10,
    output_type: str = "latent",
    seed: int = 0,
    batch_size: int = 1,
):
    """Polpulate the Requests in the workload

    Args:
        workload: (Workload): the workload to populate
        dataset (str): the dataset path
        height (int): the request image height
        width (int): the request image width
        seed (int): The random seed.
    """
    np.random.seed(seed)

    if dataset is not None:
        captions = read_column_from_csv(dataset, "caption")
    else:
        captions = len(workload.requests) * ["hello world"]

    for req in workload.requests:
        prompt = np.random.choice(captions)
        shape_idx = np.random.choice(range(len(shape_distribution)), p=[x[1] for x in shape_distribution])
        shape = shape_distribution[shape_idx][0]

        input_config = InputConfig(
            batch_size=batch_size,
            height=shape[0],
            width=shape[1],
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
        )
        req.data = asdict(input_config)


async def test(
    arrival_time: float,
    start_time: float,
    semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
):
    async with semaphore:
        sent_time = loop.time() - start_time
        print(f"Sent request at {sent_time:.3f}s (Scheduled: {arrival_time:.3f}s)")


async def submit_request(
    session: aiohttp.ClientSession,
    url: str,
    req: Request,
    semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
):
    data = req.data

    async with semaphore:
        try:
            sent_time = loop.time()
            async with session.post(url, json=data) as response:
                completion_time = loop.time()
                status = response.status
                # relative times
                return status, sent_time, completion_time

        except Exception as e:
            print(f"Request f{req.req_idx} failed: {e}")
            return None, sent_time, sent_time


URL = "http://localhost:1037/v1/generate"


async def schedule_requests(
    loop: asyncio.AbstractEventLoop,
    session: aiohttp.ClientSession,
    workload: Workload,
    start_time: float,
    semaphore: asyncio.Semaphore,
):
    tasks = []
    for i in range(len(workload.arrivals)):
        arrival_time = workload.arrivals[i]
        request = workload.requests[i]
        send_time = start_time + arrival_time

        async def scheduled_request(send_time, req: Request, arrival_time):
            if req.data is None:
                raise ValueError(f"Request {req.req_idx} not populated")

            now = loop.time()
            if send_time > now:
                await asyncio.sleep(send_time - now)
            else:
                print("Request is too early")

            print(f"Sending req {req.req_idx} scheduled at {arrival_time:.3f}s")
            status, sent_time, completion_time = await submit_request(
                session,
                URL,
                req,
                semaphore,
                loop,
            )
            req.submit_time = sent_time - start_time
            req.completion_time = completion_time - start_time
            if status != None:
                print(
                    f"Completed request {req.data['width']}x{req.data['height']} "
                    f"at {req.completion_time:.3f}s send at {req.submit_time:.3f}s "
                    f"(Scheduled: {arrival_time:.3f}s): Status Code {status}"
                )
            return req.completion_time

        task = asyncio.create_task(scheduled_request(send_time, request, arrival_time))
        tasks.append(task)
    return tasks


async def submit_workload(workload: Workload):
    loop = asyncio.get_event_loop()
    MAX_CONCURRENT_REQUESTS = 20
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    start_time = loop.time()
    async with aiohttp.ClientSession() as session:
        tasks = await schedule_requests(loop, session, workload, start_time, semaphore)
        results = await asyncio.gather(*tasks)
        return results


def get_active_gpu_count():
    """Return the number of GPUs with total memory usage > 100MB across all processes and containers."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        memory_usage = [int(x) for x in result.stdout.strip().split("\n")]
        return sum(1 for mem in memory_usage if mem > 100)  # 100 MB threshold
    except (subprocess.SubprocessError, FileNotFoundError):
        # Fallback to torch if nvidia-smi is not available
        if not torch.cuda.is_available():
            return 0

        active_gpus = 0
        for i in range(torch.cuda.device_count()):
            memory_usage = torch.cuda.memory_reserved(i)
            if memory_usage > 100 * 1024 * 1024:  # 100MB in bytes
                active_gpus += 1
        return active_gpus


def generate_log_filename(args, gpu_name):
    """Generate the log filename based on the parameters."""
    # Clean up GPU name for filename
    gpu_name = gpu_name.replace(" ", "_").replace("/", "_").lower()

    # Get active GPU count
    k = get_active_gpu_count()
    if args.world_size:
        k = args.world_size

    # Get shape distribution identifier
    shape_dist_id = "default"
    if args.shape_dist:
        shape_dist_id = Path(args.shape_dist).stem

    # Construct filename
    filename = (
        f"{k}-{gpu_name}-{args.distribution}-{args.rate}-{args.cv}"
        f"-seed{args.seed}-{args.duration}s-{args.num_inference_steps}steps"
        f"-{args.output_type}-{shape_dist_id}.json"
    )
    return filename


def create_log_entry(args, stats, workload, shape_distribution, gpu_name):
    """Create the log entry dictionary."""
    from datetime import datetime

    # Convert numpy arrays to lists for JSON serialization
    latency_list = stats.latency.tolist() if hasattr(stats.latency, "tolist") else stats.latency
    request_starts_list = (
        stats.request_starts.tolist() if hasattr(stats.request_starts, "tolist") else stats.request_starts
    )
    request_finishes_list = (
        stats.request_finishes.tolist() if hasattr(stats.request_finishes, "tolist") else stats.request_finishes
    )

    # Create requests list
    requests_data = []
    for req in workload.requests:
        req_data = {
            "req_idx": req.req_idx,
            "submit_time": req.submit_time,
            "completion_time": req.completion_time,
            "data": req.data,
        }
        requests_data.append(req_data)

    # Create the log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_name": stats.name,
        "num_requests": stats.num_requests,
        "request_rate": stats.request_rate,
        "throughput": stats.throughput,
        "latency_mean": stats.latency_mean,
        "latency_std": stats.latency_std,
        "latency_p90": stats.latency_p90,
        "parameters": {
            "gpu_name": gpu_name,
            "active_gpus": get_active_gpu_count(),
            "distribution": args.distribution,
            "rate": args.rate,
            "cv": args.cv,
            "seed": args.seed,
            "duration": args.duration,
            "num_inference_steps": args.num_inference_steps,
            "output_type": args.output_type,
            "shape_distribution": shape_distribution,
        },
        "metrics": {
            "latency": latency_list,
            "request_starts": request_starts_list,
            "request_finishes": request_finishes_list,
        },
        "requests": requests_data,
    }
    return log_entry


def save_benchmark_log(args, stats, workload, shape_distribution):
    """Save the benchmark results to a JSON file."""
    # Get GPU information
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        gpu_name = gpu_names[0]
    else:
        gpu_name = "CPU"

    # Create log directory
    log_dir = Path("log/benchmark/serving")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename and create log entry
    filename = generate_log_filename(args, gpu_name)
    log_entry = create_log_entry(args, stats, workload, shape_distribution, gpu_name)

    # Save to file
    log_path = log_dir / filename
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    print(f"Benchmark results saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--distribution", choices=["poisson", "gamma"], default="poisson")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--rate", type=float, default=1)
    parser.add_argument("--cv", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", "-n", type=int, default=10)
    parser.add_argument("--output-type", "-t", choices=["latent", "pil"], default="latent")
    parser.add_argument("--shape-dist", "-s", help="shape distribution csv file path", type=str)
    parser.add_argument("--world-size", "-w", help="overwrite world size", type=int)
    args = parser.parse_args()

    # """
    # wget https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv
    # wget https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv
    # TODO: replace "video" with "image"
    # NOTE: This dataset has very long prompts, but CLIP can only handle sequences up to 77 tokens
    # """

    # w1 = PoissonProcess(5).generate_workload("", start=0, duration=10, seed=0)
    # populate_requests(w1)
    # completion_times = asyncio.run(submit_workload(w1))
    # print("W1 Completion times:", completion_times)

    if args.distribution == "poisson":
        workload = PoissonProcess(args.rate).generate_workload("", start=0, duration=args.duration, seed=args.seed)
    elif args.distribution == "gamma":
        workload = GammaProcess(args.rate, args.cv).generate_workload(
            "", start=0, duration=args.duration, seed=args.seed
        )
    else:
        raise ValueError(f"Invalid distribution: {args.distribution}")

    if args.shape_dist:
        try:
            with open(args.shape_dist, mode="r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                raw_shape_distribution = [
                    (int(row["width"]), int(row["height"]), float(row["percentage"])) for row in reader
                ]
                # recalculate the shape percentage
                total_percentage = sum(x[2] for x in raw_shape_distribution)
                shape_distribution = [((x[0], x[1]), x[2] / total_percentage) for x in raw_shape_distribution]
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{args.shape_dist}' was not found.")
        except csv.Error as e:
            raise ValueError(f"Error reading the CSV file: {e}")
    else:
        shape_distribution = [((args.width, args.height), 1.0)]

    print(f"dataset: {args.dataset}, seed: {args.seed}, duration: {args.duration}")
    print(f"Shape distribution: {shape_distribution}")
    print(f"distribution: {args.distribution}, rate: {args.rate}, cv: {args.cv}, numbers: {len(workload.requests)}")

    populate_requests(
        workload,
        args.dataset,
        shape_distribution,
        args.num_inference_steps,
        args.output_type,
        args.seed,
        args.batch_size,
    )

    completion_times = asyncio.run(submit_workload(workload))

    print(f"distribution: {args.distribution}, rate: {args.rate}, cv: {args.cv}, numbers: {len(workload.requests)}")
    stats = workload.compute_stats(completion_times, DEFAULT_WARMUP)
    workload.print_stats(stats)

    save_benchmark_log(args, stats, workload, shape_distribution)
