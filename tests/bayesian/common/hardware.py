"""GPU frequency control, distributed hardware evaluation, and cache loading."""

from __future__ import annotations

import os
import time
import json
import random
import argparse
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.distributed as dist
import pynvml
from zeus.monitor import ZeusMonitor
from torch.multiprocessing import spawn

from .encoding import (
    encode_cfg,
    one_hot_encode,
    decode_vec,
    generate_all_configurations,
    get_unevaluated_configs,
)


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
    eval_log_path: str,
    partition_test_runner_cls,
    sleep_after_eval: float = 5.0,
):
    """
    Distributed worker that initializes tensors and fuser once and evaluates
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
        selection_flags = task.get("selection_flags") or {}
        predicted_values = task.get("predicted_values") or {}

        if rank == 0:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible is not None and len(visible.strip()) > 0:
                vis_list = [int(x) for x in visible.split(",") if x.strip() != ""]
                target_indices = vis_list
            else:
                target_indices = None
            _set_gpu_frequency(freq_mhz, device_indices=target_indices)

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

            record = {
                "freq": int(freq_mhz),
                "overlap_start": int(overlap_window[0]),
                "overlap_end": int(overlap_window[1]),
                "sm": int(sm_num),
                "block": int(block_size),
                "time_s": avg_time_s,
                "energy_j": avg_energy_j,
            }
            record.update({
                "selected_exploit_eff": bool(selection_flags.get("selected_exploit_eff", False)),
                "selected_exploit_real": bool(selection_flags.get("selected_exploit_real", False)),
                "selected_time": bool(selection_flags.get("selected_time", False)),
                "selected_explore": bool(selection_flags.get("selected_explore", False)),
            })
            record.update({
                "pred_time_s": predicted_values.get("time_s"),
                "pred_energy_eff_j": predicted_values.get("energy_eff_j"),
                "pred_energy_real_j": predicted_values.get("energy_real_j"),
            })

            with open(eval_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

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
    selection_flags_list: Optional[List[Optional[Dict[str, bool]]]] = None,
    predicted_values_list: Optional[List[Optional[Dict[str, float]]]] = None,
    sleep_after_eval: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Evaluate a batch of configurations in a single distributed run.

    Returns a list of (energy_j, time_s) aligned with x_vec_list.
    """
    eval_log_path = partition_test.eval_log_path
    master_port = partition_test.master_port

    manager = mp.Manager()
    task_list = manager.list()
    for i, x_vec in enumerate(x_vec_list):
        cfg = decode_vec(partition_test, x_vec)
        flags = (selection_flags_list[i] if selection_flags_list is not None else None) or {}
        preds = (predicted_values_list[i] if predicted_values_list is not None else None) or {}
        task_list.append({
            "index": int(i),
            "freq_mhz": int(cfg["freq"]),
            "overlap_start": int(cfg["overlap"][0]),
            "overlap_end": int(cfg["overlap"][1]),
            "sm": int(cfg["sm"]),
            "block": int(cfg["block"]),
            "selection_flags": flags,
            "predicted_values": preds,
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
            eval_log_path,
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


def try_load_initial_from_cache(
    args: argparse.Namespace,
    p2p_power_w: float,
    n_init: int,
    acq_batch: int,
    partition_test,
    partition_test_runner_cls,
):
    """
    Try to load cached configurations and measurements from eval_results.jsonl.

    Returns tuple:
      (use_cached_initial, X_train, X_train_encoded, init_time, init_eff_energy,
       init_avg_energy, all_records, skipped_batches)
    """
    eval_log_path = partition_test.eval_log_path

    init_time: List[float] = []
    init_eff_energy: List[float] = []
    init_avg_energy: List[float] = []
    all_records: List[Tuple[int, int, int, int, int, float, float, float]] = []

    if not os.path.exists(eval_log_path):
        return False, None, None, init_time, init_eff_energy, init_avg_energy, all_records, 0

    try:
        with open(eval_log_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        parsed = [json.loads(ln) for ln in lines]
        seen = set()
        unique: List[dict] = []
        for r in parsed:
            key = (int(r["freq"]), int(r["overlap_start"]), int(r["overlap_end"]), int(r["sm"]))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        if len(unique) == 0:
            return False, None, None, init_time, init_eff_energy, init_avg_energy, all_records, 0

        total_cached = len(unique)
        print(f"Found cached measurements for {total_cached} unique configs at {eval_log_path}")

        X_train_list: List[np.ndarray] = []
        for i in range(total_cached):
            r = unique[i]
            cfg = {
                "freq": int(r["freq"]),
                "sm": int(r["sm"]),
                "block": int(r["block"]),
                "overlap": (int(r["overlap_start"]), int(r["overlap_end"]))
            }
            vec = encode_cfg(partition_test, cfg)
            X_train_list.append(vec)
            t_s = float(r["time_s"])
            e_j = float(r["energy_j"])
            eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
            init_time.append(t_s)
            init_avg_energy.append(e_j)
            init_eff_energy.append(eff_e_j)
            all_records.append((
                cfg['freq'],
                cfg['overlap'][0],
                cfg['overlap'][1],
                cfg['sm'],
                cfg['block'],
                t_s,
                e_j,
                eff_e_j,
            ))
        # Top-up missing initial points
        missing = int(n_init - total_cached)
        if missing > 0:
            print(f"Cached initial points {total_cached} < n_init {n_init}; generating and evaluating {missing} additional configs...")
            all_configs = generate_all_configurations(partition_test)
            existing = np.array(X_train_list) if len(X_train_list) > 0 else np.empty((0, 3), dtype=np.int64)
            remaining = get_unevaluated_configs(all_configs, existing)
            if len(remaining) == 0:
                print("Warning: no remaining configurations available to top up initial set.")
            else:
                if missing > len(remaining):
                    missing = len(remaining)
                picked_indices = random.sample(range(len(remaining)), missing)
                vecs_to_eval: List[np.ndarray] = []
                cfgs_decoded: List[Dict[str, int]] = []
                for j, idx in enumerate(picked_indices):
                    vec = remaining[idx]
                    cfg_dec = decode_vec(partition_test, vec)
                    print(
                        f"  [topup {j+1}/{missing}] freq={cfg_dec['freq']} | sm={cfg_dec['sm']} | block={cfg_dec['block']} | overlap={cfg_dec['overlap']}"
                    )
                    vecs_to_eval.append(vec)
                    cfgs_decoded.append(cfg_dec)
                    X_train_list.append(vec)
                batch_results = measure_batch_on_hardware(
                    x_vec_list=vecs_to_eval,
                    args=args,
                    partition_test=partition_test,
                    partition_test_runner_cls=partition_test_runner_cls,
                )
                for k, (e_j, t_s) in enumerate(batch_results):
                    cfg_dec = cfgs_decoded[k]
                    eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
                    init_time.append(float(t_s))
                    init_eff_energy.append(float(eff_e_j))
                    init_avg_energy.append(float(e_j))
                    all_records.append((cfg_dec['freq'], cfg_dec['overlap'][0], cfg_dec['overlap'][1], cfg_dec['sm'], cfg_dec['block'], float(t_s), float(e_j), float(eff_e_j)))
        X_train = np.array(X_train_list)
        X_train_encoded = np.array([one_hot_encode(partition_test, x) for x in X_train])
        skipped_batches = 0
        if total_cached > n_init:
            extra = int(total_cached - n_init)
            skipped_batches = int(extra // max(1, int(acq_batch)))
            leftover = int(extra % max(1, int(acq_batch)))
            if skipped_batches > 0:
                print(
                    f"Cache covers {extra} evaluations beyond initial {n_init} -> skip {skipped_batches} batch(es) (acq_batch={acq_batch}), leftover={leftover}"
                )
        return True, X_train, X_train_encoded, init_time, init_eff_energy, init_avg_energy, all_records, skipped_batches
    except Exception as exc:
        print(f"Warning: failed to parse cached evals at {eval_log_path}: {exc}. Falling back to hardware evaluation.")
        return False, None, None, init_time, init_eff_energy, init_avg_energy, all_records, 0
