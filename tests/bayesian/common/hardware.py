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
from zeus.profile import measure as zeus_measure
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
    measurement_duration: float = 5.0,
    cooldown_duration: float = 5.0,
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

    monitor = ZeusMonitor(gpu_indices=[rank])

    num_tasks = len(task_list)
    for ti in range(num_tasks):
        task = task_list[ti]
        if rank == 0:
            print(f"Evaluating task {ti} of {num_tasks}: freq={task['freq_mhz']} | overlap={task['overlap_start']}-{task['overlap_end']} | sm={task['sm']} | block={task['block']}")

        freq_mhz = int(task["freq_mhz"])
        overlap_window = (task["overlap_start"], task["overlap_end"])
        sm_num = int(task["sm"])
        block_size = int(task["block"])
        idx = int(task["index"])

        if rank == 0:
            _set_gpu_frequency(freq_mhz, device_indices=get_visible_gpu_indices())

        dist.barrier()

        def target_fn():
            partition_test.test_config(overlap_window, (sm_num, block_size))

        trial = zeus_measure(
            target_function=target_fn,
            zeus_monitor=monitor,
            measurement_duration=measurement_duration,
            cooldown_duration=cooldown_duration,
        )

        if rank == 0:
            results_dict[idx] = {
                "energy_j": float(trial.energy_per_iter) / float(world_size),
                "time_s": float(trial.time_per_iter),
            }

    del monitor
    if dist.is_initialized():
        dist.destroy_process_group()


def measure_batch_on_hardware(
    x_vec_list: List[np.ndarray],
    args: argparse.Namespace,
    partition_test,
    partition_test_runner_cls,
    measurement_duration: float = 5.0,
    cooldown_duration: float = 5.0,
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
            "overlap_start": cfg["overlap"][0],
            "overlap_end": cfg["overlap"][1],
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
            measurement_duration,
            cooldown_duration,
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
