"""GPU frequency control and distributed hardware evaluation."""

from __future__ import annotations

import os
import time
import argparse
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import pynvml
from zeus.monitor import ZeusMonitor
from torch.multiprocessing import spawn

from .encoding import decode_vec


def get_visible_gpu_indices() -> List[int] | None:
    """Return GPU indices from CUDA_VISIBLE_DEVICES, or None if unset/empty."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None and len(visible.strip()) > 0:
        return [int(x) for x in visible.split(",") if x.strip()]
    return None


def _set_gpu_frequency(target_freq_mhz: int, device_indices: List[int] | None = None) -> None:
    """Attempt to set application clocks via NVML (best effort)."""
    pynvml.nvmlInit()
    all_indices = list(range(pynvml.nvmlDeviceGetCount()))
    indices = device_indices if device_indices is not None else all_indices
    for i in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_freq_mhz, target_freq_mhz)
    time.sleep(2)
    pynvml.nvmlShutdown()


def reset_gpu_clocks(device_indices: List[int] | None = None) -> None:
    """Reset GPU clocks to default (unlocked) state via NVML."""
    pynvml.nvmlInit()
    all_indices = list(range(pynvml.nvmlDeviceGetCount()))
    indices = device_indices if device_indices is not None else all_indices
    for i in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
    time.sleep(2)
    pynvml.nvmlShutdown()


def _dist_batch_eval_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    master_port: int,
    task_list,
    results_dict,
    partition_test_runner_cls,
    sleep_after_eval: float = 5.0,
):
    """
    Distributed worker that initializes tensors and partition once and evaluates
    a sequence of configurations provided via a shared task list.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    partition_test = partition_test_runner_cls(args, rank, world_size)

    monitor = ZeusMonitor(gpu_indices=list(range(world_size))) if rank == 0 else None

    num_tasks = len(task_list)
    for ti in range(num_tasks):
        if rank == 0:
            task = task_list[ti]
            print(f"Evaluating task {ti} of {num_tasks}: freq={task['freq_mhz']} | overlap={task['overlap_start']}-{task['overlap_end']} | sm={task['sm']} | block={task['block']}")
        else:
            task = None
        obj = [task]
        dist.broadcast_object_list(obj, src=0, group=partition_test.group)
        task = obj[0]

        freq_mhz = int(task["freq_mhz"])
        overlap_window = (int(task["overlap_start"]), int(task["overlap_end"]))
        sm_num = int(task["sm"])
        block_size = int(task["block"])
        idx = int(task["index"])

        if rank == 0:
            _set_gpu_frequency(freq_mhz, device_indices=get_visible_gpu_indices())

        # Warmup
        torch.cuda.synchronize()
        dist.barrier()
        for i in range(10):
            partition_test.test_config(overlap_window, (sm_num, block_size))
        torch.cuda.current_stream().synchronize()
        time_start = time.time()
        for i in range(8):
            partition_test.test_config(overlap_window, (sm_num, block_size))
        torch.cuda.synchronize()
        dist.barrier()
        time_end = time.time()
        duration = (time_end - time_start) / 8.0

        if rank == 0:
            iterations = int(max(1, round(5.0 / max(duration, 1e-6))))
            obj_list = [iterations]
            print(f"Duration: {duration * 1000} ms, Required iterations: {iterations}")
        else:
            obj_list = [None]
        dist.broadcast_object_list(obj_list, src=0, group=partition_test.group)
        iterations = obj_list[0]

        # Measure
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0 and monitor is not None:
            monitor.begin_window("step")
        for _ in range(iterations):
            partition_test.test_config(overlap_window, (sm_num, block_size))
        torch.cuda.synchronize()
        dist.barrier()
        if hasattr(partition_test, "clean"):
            print(f"Cleaning up partition test")
            partition_test.clean()

        if rank == 0 and monitor is not None:
            result = monitor.end_window("step")
            avg_time_s = float(result.time) / float(iterations)
            avg_energy_j = float(result.total_energy) / float(iterations) / float(world_size)

            results_dict[idx] = {
                "energy_j": float(avg_energy_j),
                "time_s": float(avg_time_s),
            }
        time.sleep(sleep_after_eval)

    if rank == 0:
        pid = os.getpid()
        print(f"Killing process group {pid}")
        os.system(f'pkill -P {pid}')
    if dist.is_initialized():
        dist.destroy_process_group()


def measure_batch_on_hardware(
    x_vec_list: List[np.ndarray],
    args: argparse.Namespace,
    partition_test,
    partition_test_runner_cls,
    sleep_after_eval: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Evaluate a batch of configurations in a single distributed run.

    Returns a list of (energy_j, time_s) aligned with x_vec_list.
    Record building and eval-log writing are handled by the caller via
    :func:`orchestration.log_batch_eval_results`.
    """
    master_port = partition_test.master_port

    manager = mp.Manager()
    task_list = manager.list()
    for i, x_vec in enumerate(x_vec_list):
        cfg = decode_vec(partition_test, x_vec)
        task_list.append({
            "index": int(i),
            "freq_mhz": int(cfg["freq"]),
            "overlap_start": int(cfg["overlap"][0]),
            "overlap_end": int(cfg["overlap"][1]),
            "sm": int(cfg["sm"]),
            "block": int(cfg["block"]),
        })

    results_dict = manager.dict()
    spawn(
        _dist_batch_eval_worker,
        args=(
            args.world_size,
            args,
            master_port,
            task_list,
            results_dict,
            partition_test_runner_cls,
            sleep_after_eval,
        ),
        nprocs=args.world_size,
        join=True,
    )

    results: List[Tuple[float, float]] = []
    for i in range(len(x_vec_list)):
        e_j = float(results_dict[i]["energy_j"])
        t_s = float(results_dict[i]["time_s"])
        results.append((e_j, t_s))
    return results
