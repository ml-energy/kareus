#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common utilities shared by Bayesian optimization scripts (attention/mlp, forward/backward).

Includes:
- XGBoost training and prediction helpers
- Hypervolume and normalization helpers
- Candidate generation helpers
- NVML-based GPU application clock setter (best-effort)
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import pynvml
import xgboost as xgb
import os
import argparse
import torch.distributed as dist
import json
from zeus.monitor import ZeusMonitor
from torch.multiprocessing import spawn
import multiprocessing as mp
import random
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import matplotlib.pyplot as plt


FREQ_IDX = 0
SM_IDX = 1
# BLOCK_IDX = 2
OVERLAP_IDX = 2


def encode_cfg(partition_test, cfg: Dict[str, int]) -> np.ndarray:
    """
    Encode configuration to an index vector [freq_idx, sm_idx, overlap_idx].

    cfg keys:
      - freq: actual GPU core frequency (per partition_test.FREQ_VALUES)
      - sm: actual SM count (1..20)
      - block: CUDA block size (512 or 1024)
      - overlap: overlap window tuple from partition_test.OVERLAP_WINDOWS
    """
    freq_idx = partition_test.FREQ_VALUES.index(cfg["freq"])  # 0..len-1
    sm_idx = partition_test.SM_VALUES.index(cfg["sm"])  # 0..19
    overlap_idx = partition_test.OVERLAP_WINDOWS.index(cfg["overlap"])  # 0..len-1
    return np.array([freq_idx, sm_idx, overlap_idx], dtype=np.int64)


def one_hot_encode(partition_test, x: np.ndarray) -> np.ndarray:
    """
    One-hot encode categorical features (overlap) and keep freq/SM as numeric.

    x: [freq_idx, sm_idx, overlap_idx]
    """
    # Use actual MHz value for frequency instead of the categorical index
    freq_mhz = float(partition_test.FREQ_VALUES[int(x[FREQ_IDX])])
    numeric = np.array([freq_mhz, x[SM_IDX]], dtype=np.float32)
    overlap_one_hot = np.zeros(len(partition_test.OVERLAP_WINDOWS), dtype=np.float32)
    overlap_one_hot[int(x[OVERLAP_IDX])] = 1.0
    return np.concatenate([numeric, overlap_one_hot], axis=0)


def decode_vec(partition_test, x: np.ndarray) -> Dict[str, int]:
    """
    Decode an index vector [freq_idx, sm_idx, overlap_idx] back to a config dict.
    """
    freq_idx = int(np.clip(round(float(x[FREQ_IDX])), 0, len(partition_test.FREQ_VALUES) - 1))
    sm_idx = int(np.clip(round(float(x[SM_IDX])), 0, len(partition_test.SM_VALUES) - 1))
    overlap_idx = int(np.clip(round(float(x[OVERLAP_IDX])), 0, len(partition_test.OVERLAP_WINDOWS) - 1))
    return {
        "freq": partition_test.FREQ_VALUES[freq_idx],
        "sm": partition_test.SM_VALUES[sm_idx],
        "block": 1024,
        "overlap": partition_test.OVERLAP_WINDOWS[overlap_idx],
    }


# -----------------------------
# Real evaluation via distributed run
# -----------------------------

def _set_gpu_frequency(target_freq_mhz: int, device_indices: List[int] | None = None) -> None:
    """Attempt to set application clocks via NVML (best effort).

    If device_indices is provided, only set those NVML indices; otherwise set all
    NVML-visible devices.
    """
    pynvml.nvmlInit()
    all_indices = list(range(pynvml.nvmlDeviceGetCount()))
    indices = device_indices if device_indices is not None else all_indices
    for i in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_freq_mhz, target_freq_mhz)
        time.sleep(1)
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
):
    """
    Distributed worker that initializes tensors and fuser once and evaluates
    a sequence of configurations provided via a shared task list. Rank 0 reads
    tasks and broadcasts to all ranks; only rank 0 records results/logs.
    
    Args:
        partition_test_runner_cls: The PartitionTestRunner CLASS (not instance)
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    # Instantiate the test runner (initializes distributed environment)
    partition_test = partition_test_runner_cls(args, rank, world_size)
    
    monitor = ZeusMonitor(gpu_indices=list(range(world_size))) if rank == 0 else None

    # Iterate over all tasks prepared by the parent process
    num_tasks = len(task_list)
    for ti in range(num_tasks):
        # Rank 0 reads the task and broadcasts to other ranks
        if rank == 0:
            task = task_list[ti]
            print(f"Evaluating task {ti} of {num_tasks}: freq={task['freq_mhz']} | overlap={task['overlap_start']}-{task['overlap_end']} | sm={task['sm']} | block={task['block']}")
        else:
            task = None
        obj = [task]
        dist.broadcast_object_list(obj, src=0, group=partition_test.group)
        task = obj[0]

        freq_mhz = int(task["freq_mhz"])  # set frequency per candidate within worker
        overlap_window = (int(task["overlap_start"]), int(task["overlap_end"]))
        sm_num = int(task["sm"])
        block_size = int(task["block"])
        idx = int(task["index"])  # original index within this batch
        selection_flags = task.get("selection_flags") or {}
        predicted_values = task.get("predicted_values") or {}

        # Best-effort: set GPU frequency on rank 0 (applies to all NVML-visible devices)
        if rank == 0:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible is not None and len(visible.strip()) > 0:
                vis_list = [int(x) for x in visible.split(",") if x.strip() != ""]
                target_indices = vis_list
            else:
                target_indices = None
            _set_gpu_frequency(freq_mhz, device_indices=target_indices)

        # Warmup to determine iteration count for this configuration
        torch.cuda.synchronize()
        dist.barrier()
        for i in range(10):
            if i == 2:
                time_start = time.time()
            partition_test.test_config(overlap_window, (sm_num, block_size))
        torch.cuda.synchronize()
        dist.barrier()
        time_end = time.time()
        duration = (time_end - time_start) / 8.0

        if rank == 0:
            iterations = int(max(1, round(8.0 / max(duration, 1e-6))))
            obj_list = [iterations]
        else:
            obj_list = [None]
        dist.broadcast_object_list(obj_list, src=0, group=partition_test.group)
        iterations = obj_list[0]

        # Measure this configuration
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0 and monitor is not None:
            monitor.begin_window("step")
        for _ in range(iterations):
            partition_test.test_config(overlap_window, (sm_num, block_size))
        torch.cuda.synchronize()
        dist.barrier()

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

            # Persist per-config log
            with open(eval_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Store ordered results back to shared dict
            results_dict[idx] = {
                "energy_j": float(avg_energy_j),
                "time_s": float(avg_time_s),
            }
        time.sleep(10)

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
) -> List[Tuple[float, float]]:
    """
    Evaluate a batch of configurations in a single distributed run. Tensors and monitor
    are initialized once, then each configuration is measured in sequence.

    Args:
        partition_test: Config holder instance (used in parent process)
        partition_test_runner_cls: PartitionTestRunner CLASS to instantiate in workers

    Returns a list of (energy_j, time_s) aligned with x_vec_list.
    """
    eval_log_path = partition_test.eval_log_path
    master_port = partition_test.master_port

    manager = mp.Manager()
    # Prepare task list with all configs and metadata
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
        ),
        nprocs=args.world_size,
        join=True,
    )

    # Collect results in input order
    results: List[Tuple[float, float]] = []
    for i in range(len(x_vec_list)):
        e_j = float(results_dict[i]["energy_j"])  # type: ignore[index]
        t_s = float(results_dict[i]["time_s"])    # type: ignore[index]
        results.append((e_j, t_s))
    return results


# -----------------------------
# Cached initial load helper
# -----------------------------

def try_load_initial_from_cache(
    args: argparse.Namespace,
    p2p_power_w: float,
    n_init: int,
    acq_batch: int,
    partition_test,
    partition_test_runner_cls,
):
    """
    Try to load cached configurations and measurements from eval_results.jsonl if available.
    Behavior:
      - If cache contains >= n_init unique points, use ALL cached points to seed X_train.
      - If cache contains < n_init, top up by evaluating additional configs to reach n_init.

    Args:
        partition_test: Config holder instance (used in parent process)
        partition_test_runner_cls: PartitionTestRunner CLASS to instantiate in workers

    Returns tuple:
      (use_cached_initial, X_train, X_train_encoded, init_time, init_eff_energy, init_avg_energy, all_records, skipped_batches)

    Where skipped_batches indicates how many full acquisition batches (of size acq_batch)
    beyond the initial n_init are already covered by cache, so the BO loop can resume from
    the next iteration.
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
        # Deduplicate while preserving order
        seen = set()
        unique: List[dict] = []
        for r in parsed:
            # Key excludes block since it's fixed per kernel type (not a search parameter)
            key = (int(r["freq"]), int(r["overlap_start"]), int(r["overlap_end"]), int(r["sm"]))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        if len(unique) == 0:
            return False, None, None, init_time, init_eff_energy, init_avg_energy, all_records, 0

        total_cached = len(unique)
        print(f"Found cached measurements for {total_cached} unique configs at {eval_log_path}")

        # Use ALL cached configs to seed training data
        X_train_list: List[np.ndarray] = []
        for i in range(total_cached):
            r = unique[i]
            cfg = {
                "freq": int(r["freq"]),
                "sm": int(r["sm"]),
                "overlap": (int(r["overlap_start"]), int(r["overlap_end"]))
            }
            # Note: block size is stored in cache but not needed for encoding (it's fixed per kernel type)
            vec = encode_cfg(partition_test, cfg)
            X_train_list.append(vec)
            t_s = float(r["time_s"])
            e_j = float(r["energy_j"])
            eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
            init_time.append(t_s)
            init_avg_energy.append(e_j)
            init_eff_energy.append(eff_e_j)
            all_records.append((cfg['freq'], cfg['overlap'][0], cfg['overlap'][1], cfg['sm'], t_s, e_j, eff_e_j))
        # Top-up missing initial points by generating and evaluating additional configs
        missing = int(n_init - total_cached)
        if missing > 0:
            print(f"Cached initial points {total_cached} < n_init {n_init}; generating and evaluating {missing} additional configs...")
            all_configs = generate_all_configurations(partition_test)
            existing = np.array(X_train_list) if len(X_train_list) > 0 else np.empty((0, 3), dtype=np.int64)
            remaining: List[np.ndarray] = []
            for cfg_vec in all_configs:
                if existing.shape[0] == 0 or not is_config_in_dataset(cfg_vec, existing):
                    remaining.append(cfg_vec)
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
                # Evaluate picked configs in a single distributed spawn
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
                    all_records.append((cfg_dec['freq'], cfg_dec['overlap'][0], cfg_dec['overlap'][1], cfg_dec['sm'], float(t_s), float(e_j), float(eff_e_j)))
        X_train = np.array(X_train_list)
        X_train_encoded = np.array([one_hot_encode(partition_test, x) for x in X_train])
        # Determine how many full batches to skip given cached points beyond n_init
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


# -----------------------------
# Surrogate models and EHVI
# -----------------------------

def train_xgb_models(X_encoded: np.ndarray, y_energy: np.ndarray, y_time: np.ndarray):
    dtrain_energy = xgb.DMatrix(X_encoded, label=y_energy)
    dtrain_time = xgb.DMatrix(X_encoded, label=y_time)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "eta": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
    }
    # params = AUTOTVM_PARAMS
    energy_model = xgb.train(params, dtrain_energy, num_boost_round=100)
    time_model = xgb.train(params, dtrain_time, num_boost_round=100)
    return energy_model, time_model


def train_xgb_energy_only(X_encoded: np.ndarray, y_energy: np.ndarray):
    dtrain_energy = xgb.DMatrix(X_encoded, label=y_energy)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "eta": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
    }
    energy_model = xgb.train(params, dtrain_energy, num_boost_round=100)
    return energy_model


def train_xgb_ensemble(
    X_encoded: np.ndarray,
    y_energy: np.ndarray,
    y_time: np.ndarray,
    ensemble_size: int = 5,
    bootstrap_frac: float = 0.8,
    base_seed: int = 42,
):
    """
    Train an ensemble of XGBoost regressors via bootstrap resampling and different seeds
    to estimate predictive uncertainty.

    Returns a list of (energy_model, time_model) tuples.
    """
    n = X_encoded.shape[0]
    ensemble = []
    for i in range(int(max(1, ensemble_size))):
        rng = np.random.RandomState(base_seed + i)
        if bootstrap_frac >= 1.0:
            idx = np.arange(n)
        else:
            m = max(1, int(round(bootstrap_frac * n)))
            idx = rng.choice(n, size=m, replace=True)
        Xb = X_encoded[idx]
        yeb = y_energy[idx]
        ytb = y_time[idx]

        dtrain_energy = xgb.DMatrix(Xb, label=yeb)
        dtrain_time = xgb.DMatrix(Xb, label=ytb)
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "eta": 0.3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "seed": int(base_seed + i),
        }
        # params = {**AUTOTVM_PARAMS, "seed": int(base_seed + i)}
        energy_model = xgb.train(params, dtrain_energy, num_boost_round=100)
        time_model = xgb.train(params, dtrain_time, num_boost_round=100)
        ensemble.append((energy_model, time_model))
    return ensemble


def predict_ensemble_stats(
    ensemble_models: List[Tuple[xgb.Booster, xgb.Booster]],
    X_encoded: np.ndarray,
):
    """
    For each row in X_encoded, return mean and std for (energy, time) across ensemble.
    Returns:
      energy_mean, energy_std, time_mean, time_std  (each shape [N])
    """
    if len(ensemble_models) == 0:
        dtest = xgb.DMatrix(X_encoded)
        return (
            np.zeros(X_encoded.shape[0], dtype=np.float64),
            np.ones(X_encoded.shape[0], dtype=np.float64),
            np.zeros(X_encoded.shape[0], dtype=np.float64),
            np.ones(X_encoded.shape[0], dtype=np.float64),
        )

    energy_preds = []
    time_preds = []
    for em, tm in ensemble_models:
        dtest = xgb.DMatrix(X_encoded)
        energy_preds.append(em.predict(dtest))
        time_preds.append(tm.predict(dtest))
    energy_preds = np.vstack(energy_preds)  # [E, N]
    time_preds = np.vstack(time_preds)      # [E, N]
    energy_mean = np.mean(energy_preds, axis=0)
    energy_std = np.std(energy_preds, axis=0)
    time_mean = np.mean(time_preds, axis=0)
    time_std = np.std(time_preds, axis=0)
    return energy_mean, energy_std, time_mean, time_std


def predict_performance(models, X_encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    energy_model, time_model = models
    dtest = xgb.DMatrix(X_encoded)
    energy_pred = energy_model.predict(dtest)
    time_pred = time_model.predict(dtest)
    return energy_pred, time_pred


def calculate_dominated_hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    # Convert minimization (energy, time) to maximization by negation and compute HV via BoTorch.
    Y = -torch.tensor(points, dtype=torch.double)
    ref = -torch.tensor(ref_point, dtype=torch.double)
    hv = Hypervolume(ref_point=ref)
    hv_value = hv.compute(Y)
    return float(hv_value)


def normalize_objectives(data: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    ranges = max_vals - min_vals
    # Avoid division by zero
    ranges = np.where(ranges == 0, 1.0, ranges)
    return (data - min_vals) / ranges


def expected_hypervolume_improvement(
    candidate_vec: np.ndarray,
    pareto_front: np.ndarray,
    models,
    ref_point: np.ndarray,
    partition_test,
    normalization_bounds: tuple = None,
    current_hv_cached: float | None = None,
    pareto_front_norm_cached: np.ndarray | None = None,
    ref_point_norm_cached: np.ndarray | None = None,
) -> float:
    """
    Calculate expected hypervolume improvement with normalized objectives.
    
    Args:
        candidate_vec: Candidate configuration vector
        pareto_front: Current Pareto front points
        models: Trained surrogate models
        ref_point: Reference point for hypervolume calculation
        partition_test: Partition test runner with configuration values
        normalization_bounds: Tuple of (min_vals, max_vals) for normalization
    
    Returns:
        Expected hypervolume improvement
    """
    candidate_encoded = one_hot_encode(partition_test, candidate_vec).reshape(1, -1)
    e_pred, t_pred = predict_performance(models, candidate_encoded)
    predicted_point = np.array([[e_pred[0], t_pred[0]]], dtype=np.float64)
    
    # Apply normalization if bounds are provided
    if normalization_bounds is not None:
        min_vals, max_vals = normalization_bounds

        # Use cached normalized front/ref if provided
        if pareto_front_norm_cached is None or ref_point_norm_cached is None:
            pareto_front_norm = normalize_objectives(pareto_front, min_vals, max_vals)
            ref_point_norm = normalize_objectives(ref_point.reshape(1, -1), min_vals, max_vals).flatten()
        else:
            pareto_front_norm = pareto_front_norm_cached
            ref_point_norm = ref_point_norm_cached

        predicted_point_norm = normalize_objectives(predicted_point, min_vals, max_vals)

        current_hv = current_hv_cached if current_hv_cached is not None else calculate_dominated_hypervolume(pareto_front_norm, ref_point_norm)
        new_front_norm = np.vstack([pareto_front_norm, predicted_point_norm])
        new_hv = calculate_dominated_hypervolume(new_front_norm, ref_point_norm)
    else:
        current_hv = current_hv_cached if current_hv_cached is not None else calculate_dominated_hypervolume(pareto_front, ref_point)
        new_front = np.vstack([pareto_front, predicted_point])
        new_hv = calculate_dominated_hypervolume(new_front, ref_point)
    
    return float(max(0.0, new_hv - current_hv))


# -----------------------------
# Candidate generation
# -----------------------------

def generate_all_configurations(partition_test) -> List[np.ndarray]:
    configs: List[np.ndarray] = []
    for freq_idx in range(len(partition_test.FREQ_VALUES)):
        for sm_idx in range(len(partition_test.SM_VALUES)):
            for overlap_idx in range(len(partition_test.OVERLAP_WINDOWS)):
                configs.append(
                    np.array([freq_idx, sm_idx, overlap_idx], dtype=np.int64)
                )
    return configs


def is_config_in_dataset(config: np.ndarray, dataset: np.ndarray) -> bool:
    return any(np.array_equal(config, x) for x in dataset)


# -----------------------------
# Visualization helpers
# -----------------------------

def save_iteration_plots(
    ib: int,
    partition_test,
    args: argparse.Namespace,
    prev_energy_eff: np.ndarray,
    prev_energy_real: np.ndarray,
    prev_time: np.ndarray,
    new_time: List[float],
    new_eff_energy: List[float],
    new_real_energy: List[float],
    cat_exploit_eff: List[bool],
    cat_exploit_real: List[bool],
    cat_time: List[bool],
    cat_explore: List[bool],
) -> None:
    base_logs_dir = partition_test.logs_dir + "/figures"
    os.makedirs(base_logs_dir, exist_ok=True)

    # Compute previous Pareto fronts (measured before this iteration's new points)
    Y_eff_prev = np.column_stack((prev_energy_eff, prev_time)) if len(prev_time) > 0 else np.empty((0, 2))
    pareto_mask_eff_prev = is_non_dominated(-torch.tensor(Y_eff_prev, dtype=torch.double)).cpu().numpy().astype(bool) if Y_eff_prev.shape[0] > 0 else np.array([], dtype=bool)
    Y_real_prev = np.column_stack((prev_energy_real, prev_time)) if len(prev_time) > 0 else np.empty((0, 2))
    pareto_mask_real_prev = is_non_dominated(-torch.tensor(Y_real_prev, dtype=torch.double)).cpu().numpy().astype(bool) if Y_real_prev.shape[0] > 0 else np.array([], dtype=bool)

    new_time_arr = np.array(new_time, dtype=float)
    new_eff_arr = np.array(new_eff_energy, dtype=float)
    new_real_arr = np.array(new_real_energy, dtype=float)
    cat_eff = np.array(cat_exploit_eff, dtype=bool)
    cat_real = np.array(cat_exploit_real, dtype=bool)
    cat_time_arr = np.array(cat_time, dtype=bool)
    cat_explore_arr = np.array(cat_explore, dtype=bool)

    # Plot Effective-energy frontier
    plt.figure(figsize=(7, 5))
    # Previous measured points (effective): x=time, y=energy
    if Y_eff_prev.shape[0] > 0:
        plt.scatter(Y_eff_prev[:, 1], Y_eff_prev[:, 0], c="#888888", s=20, label="Measured prev")
        if np.any(pareto_mask_eff_prev):
            front = Y_eff_prev[pareto_mask_eff_prev]
            front_sorted = front[np.argsort(front[:, 1])]
            plt.plot(front_sorted[:, 1], front_sorted[:, 0], "-", c="#1f77b4", label="Pareto prev (eff)")
    # New measured points (categorized)
    if new_time_arr.size > 0:
        if np.any(cat_eff):
            plt.scatter(new_time_arr[cat_eff], new_eff_arr[cat_eff], marker="x", c="#d62728", s=50, label="Exploit eff")
        if np.any(cat_time_arr):
            plt.scatter(new_time_arr[cat_time_arr], new_eff_arr[cat_time_arr], marker="o", facecolors="none", edgecolors="#2ca02c", s=60, label="Time picks")
        if np.any(cat_explore_arr):
            plt.scatter(new_time_arr[cat_explore_arr], new_eff_arr[cat_explore_arr], marker="s", c="#9467bd", s=40, label="Explore")
    plt.xlabel("Time (s)")
    plt.ylabel("Effective energy (J)")
    plt.title(f"Iter {ib+1}: Time vs Effective energy (measured)")
    plt.legend(loc="best")
    eff_path = os.path.join(base_logs_dir, f"iter_{ib+1:02d}_effective.png")
    plt.tight_layout()
    plt.savefig(eff_path)
    plt.close()

    # Plot Real-energy frontier
    plt.figure(figsize=(7, 5))
    if Y_real_prev.shape[0] > 0:
        plt.scatter(Y_real_prev[:, 1], Y_real_prev[:, 0], c="#888888", s=20, label="Measured prev")
        if np.any(pareto_mask_real_prev):
            front = Y_real_prev[pareto_mask_real_prev]
            front_sorted = front[np.argsort(front[:, 1])]
            plt.plot(front_sorted[:, 1], front_sorted[:, 0], "-", c="#ff7f0e", label="Pareto prev (real)")
    if new_time_arr.size > 0:
        if np.any(cat_real):
            plt.scatter(new_time_arr[cat_real], new_real_arr[cat_real], marker="+", c="#1f77b4", s=60, label="Exploit real")
        if np.any(cat_time_arr):
            plt.scatter(new_time_arr[cat_time_arr], new_real_arr[cat_time_arr], marker="o", facecolors="none", edgecolors="#2ca02c", s=60, label="Time picks")
        if np.any(cat_explore_arr):
            plt.scatter(new_time_arr[cat_explore_arr], new_real_arr[cat_explore_arr], marker="s", c="#9467bd", s=40, label="Explore")
    plt.xlabel("Time (s)")
    plt.ylabel("Real energy (J)")
    plt.title(f"Iter {ib+1}: Time vs Real energy (measured)")
    plt.legend(loc="best")
    real_path = os.path.join(base_logs_dir, f"iter_{ib+1:02d}_real.png")
    plt.tight_layout()
    plt.savefig(real_path)
    plt.close()


# -----------------------------
# High-level orchestration helpers
# -----------------------------

def setup_initial_data(
    args: argparse.Namespace,
    partition_test,
    partition_test_runner_cls,
    p2p_power_w: float,
    all_configs: List[np.ndarray],
    n_init: int,
):
    """
    Load cached data or generate and evaluate initial configurations.
    
    Returns:
        tuple: (X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real, all_records, start_batch_idx)
    """
    use_cached_initial, X_train_cached, X_train_encoded_cached, init_time, init_eff_energy, init_avg_energy, all_records, skipped_batches = try_load_initial_from_cache(
        args=args,
        p2p_power_w=p2p_power_w,
        n_init=n_init,
        acq_batch=int(args.acq_batch),
        partition_test=partition_test,
        partition_test_runner_cls=partition_test_runner_cls,
    )
    
    if not use_cached_initial:
        init_indices = random.sample(range(len(all_configs)), n_init)
        X_train = np.array([all_configs[i] for i in init_indices])
        X_train_encoded = np.array([one_hot_encode(partition_test, x) for x in X_train])

        print(f"Total {len(all_configs)} configurations")
        print(f"Generated {X_train.shape[0]} initial configurations")
        print("Evaluating initial configurations on hardware...")

        start_time = time.time()
        cfgs_decoded: List[Dict[str, int]] = []
        for i in range(X_train.shape[0]):
            cfg = decode_vec(partition_test, X_train[i])
            print(
                f"  [{i+1}/{X_train.shape[0]}] freq={cfg['freq']} | sm={cfg['sm']} | overlap={cfg['overlap']}"
            )
            cfgs_decoded.append(cfg)
        
        # Evaluate all configs in a single batch
        batch_results = measure_batch_on_hardware(
            x_vec_list=list(X_train),
            args=args,
            partition_test=partition_test,
            partition_test_runner_cls=partition_test_runner_cls,
        )
        
        # Process results
        for i, (e_j, t_s) in enumerate(batch_results):
            cfg = cfgs_decoded[i]
            eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
            init_time.append(float(t_s))
            init_eff_energy.append(float(eff_e_j))
            init_avg_energy.append(float(e_j))
            all_records.append((cfg['freq'], cfg['overlap'][0], cfg['overlap'][1], cfg['sm'], cfg['block'], float(t_s), float(e_j), float(eff_e_j)))
            print(f"  [{i+1}/{X_train.shape[0]}] -> Energy={e_j:.4f} J, Time={t_s:.6f} s")
        
        init_eval_time = time.time() - start_time
        print(f"Initial evaluation completed in {init_eval_time:.2f} s")
    else:
        X_train = X_train_cached
        X_train_encoded = X_train_encoded_cached

    y_energy_eff = np.array(init_eff_energy, dtype=np.float64)
    y_time = np.array(init_time, dtype=np.float64)
    y_energy_real = np.array(init_avg_energy, dtype=np.float64)
    
    print(
        f"Initial ranges: Energy [{np.min(y_energy_eff):.4f}, {np.max(y_energy_eff):.4f}] J | "
        f"Time [{np.min(y_time):.6f}, {np.max(y_time):.6f}] s"
    )
    
    start_batch_idx = int(skipped_batches) if use_cached_initial else 0
    if start_batch_idx > 0:
        print(f"Resuming from batch {start_batch_idx+1} (skipped {start_batch_idx} full batch(es) from cache)")
    
    return X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real, all_records, start_batch_idx


def compute_normalization_bounds(
    args: argparse.Namespace,
    y_energy_eff: np.ndarray,
    y_energy_real: np.ndarray,
    y_time: np.ndarray,
):
    """
    Calculate normalization bounds for objectives if enabled.
    
    Returns:
        tuple: (normalization_bounds_eff, normalization_bounds_real)
    """
    normalization_bounds_eff = None
    normalization_bounds_real = None
    
    if args.normalize_objectives:
        min_vals_eff = np.array([np.min(y_energy_eff), np.min(y_time)])
        max_vals_eff = np.array([np.max(y_energy_eff), np.max(y_time)])
        normalization_bounds_eff = (min_vals_eff, max_vals_eff)
        
        min_vals_real = np.array([np.min(y_energy_real), np.min(y_time)])
        max_vals_real = np.array([np.max(y_energy_real), np.max(y_time)])
        normalization_bounds_real = (min_vals_real, max_vals_real)
        
        print(f"  Norm bounds (eff)  - Energy: [{min_vals_eff[0]:.4f}, {max_vals_eff[0]:.4f}], Time: [{min_vals_eff[1]:.6f}, {max_vals_eff[1]:.6f}]")
        print(f"  Norm bounds (real) - Energy: [{min_vals_real[0]:.4f}, {max_vals_real[0]:.4f}], Time: [{min_vals_real[1]:.6f}, {max_vals_real[1]:.6f}]")
    else:
        print("  Using raw objectives without normalization")
    
    return normalization_bounds_eff, normalization_bounds_real


def score_candidates_with_ehvi(
    candidates: np.ndarray,
    cand_encoded: np.ndarray,
    current_front_eff: np.ndarray,
    current_front_real: np.ndarray,
    models_eff,
    models_real,
    ref_point_eff: np.ndarray,
    ref_point_real: np.ndarray,
    partition_test,
    normalization_bounds_eff,
    normalization_bounds_real,
):
    """
    Score all candidates using EHVI for both effective and real energy objectives.
    
    Returns:
        tuple: (ehvi_eff_values, ehvi_real_values)
    """
    # Precompute HV caches
    current_hv_eff_cached = None
    pareto_front_eff_norm_cached = None
    ref_point_eff_norm_cached = None
    if normalization_bounds_eff is not None:
        min_vals_eff, max_vals_eff = normalization_bounds_eff
        pareto_front_eff_norm_cached = normalize_objectives(current_front_eff, min_vals_eff, max_vals_eff)
        ref_point_eff_norm_cached = normalize_objectives(ref_point_eff.reshape(1, -1), min_vals_eff, max_vals_eff).flatten()
        current_hv_eff_cached = calculate_dominated_hypervolume(pareto_front_eff_norm_cached, ref_point_eff_norm_cached)
    else:
        current_hv_eff_cached = calculate_dominated_hypervolume(current_front_eff, ref_point_eff)

    current_hv_real_cached = None
    pareto_front_real_norm_cached = None
    ref_point_real_norm_cached = None
    if normalization_bounds_real is not None:
        min_vals_real, max_vals_real = normalization_bounds_real
        pareto_front_real_norm_cached = normalize_objectives(current_front_real, min_vals_real, max_vals_real)
        ref_point_real_norm_cached = normalize_objectives(ref_point_real.reshape(1, -1), min_vals_real, max_vals_real).flatten()
        current_hv_real_cached = calculate_dominated_hypervolume(pareto_front_real_norm_cached, ref_point_real_norm_cached)
    else:
        current_hv_real_cached = calculate_dominated_hypervolume(current_front_real, ref_point_real)

    # Score via EHVI for both effective-energy and real-energy objectives
    ehvi_eff_values: List[float] = []
    ehvi_real_values: List[float] = []
    for vec in candidates:
        ehvi_eff = expected_hypervolume_improvement(
            vec,
            current_front_eff,
            models_eff,
            ref_point_eff,
            partition_test,
            normalization_bounds_eff,
            current_hv_cached=current_hv_eff_cached,
            pareto_front_norm_cached=pareto_front_eff_norm_cached,
            ref_point_norm_cached=ref_point_eff_norm_cached,
        )
        ehvi_real = expected_hypervolume_improvement(
            vec,
            current_front_real,
            models_real,
            ref_point_real,
            partition_test,
            normalization_bounds_real,
            current_hv_cached=current_hv_real_cached,
            pareto_front_norm_cached=pareto_front_real_norm_cached,
            ref_point_norm_cached=ref_point_real_norm_cached,
        )
        ehvi_eff_values.append(ehvi_eff)
        ehvi_real_values.append(ehvi_real)
    
    return np.array(ehvi_eff_values), np.array(ehvi_real_values)


def select_acquisition_batch(
    candidates: np.ndarray,
    cand_encoded: np.ndarray,
    ehvi_eff_values: np.ndarray,
    ehvi_real_values: np.ndarray,
    ensemble_models,
    models_eff,
    args: argparse.Namespace,
    partition_test,
):
    """
    Select candidates for next batch using exploit/explore/time strategy.
    
    Returns:
        tuple: (selected, final_idx, exploit_eff_idx, exploit_real_idx, time_idx, explore_idx)
    """
    # Compute uncertainty from ensemble predictions
    e_mean, e_std, t_mean, t_std = predict_ensemble_stats(ensemble_models, cand_encoded)

    # Single-model predictions for exploit-style time picks
    pred_energy_single_eff, pred_time_single = predict_performance(models_eff, cand_encoded)
    
    if args.uncertainty_metric == "sum":
        unc_score = e_std + t_std
    elif args.uncertainty_metric == "max":
        unc_score = np.maximum(e_std, t_std)
    elif args.uncertainty_metric == "energy_std":
        unc_score = e_std
    else:  # time_std
        unc_score = t_std

    # Split acquisition into exploit, time-focused, and explore
    k_total = int(args.acq_batch)
    k_time = int(round(args.time_fraction * k_total))
    k_remaining = max(0, k_total - k_time)
    k_explore = int(round(args.explore_fraction * k_remaining))
    k_exploit = max(0, k_remaining - k_explore)

    # Indices for exploit: split between effective and real energy objectives
    exploit_idx: List[int] = []
    exploit_eff_idx: List[int] = []
    exploit_real_idx: List[int] = []
    if k_exploit > 0:
        k_exploit_eff = k_exploit // 2
        k_exploit_real = k_exploit - k_exploit_eff

        # Top by EHVI (effective energy)
        top_eff = np.argsort(ehvi_eff_values)[-k_exploit_eff:][::-1].tolist() if k_exploit_eff > 0 else []
        picked = set()
        for idx in top_eff:
            if idx not in picked:
                exploit_idx.append(idx)
                exploit_eff_idx.append(idx)
                picked.add(idx)

        # Top by EHVI (real energy), excluding already picked
        if k_exploit_real > 0:
            top_real = np.argsort(ehvi_real_values)[-k_exploit_real:][::-1].tolist()
            for idx in top_real:
                if idx not in picked:
                    exploit_idx.append(idx)
                    exploit_real_idx.append(idx)
                    picked.add(idx)

        # Backfill to reach k_exploit using combined EHVI max, if needed
        if len(exploit_idx) < k_exploit:
            combined = np.maximum(ehvi_eff_values, ehvi_real_values)
            for idx in np.argsort(combined)[::-1].tolist():
                if idx not in picked:
                    exploit_idx.append(idx)
                    # Assign backfilled to eff/real based on which EHVI is larger
                    if ehvi_eff_values[idx] >= ehvi_real_values[idx]:
                        exploit_eff_idx.append(idx)
                    else:
                        exploit_real_idx.append(idx)
                    picked.add(idx)
                if len(exploit_idx) >= k_exploit:
                    break

    # Indices for time-focused picks (smallest predicted time), excluding exploit
    time_idx = []
    if k_time > 0:
        sorted_time = np.argsort(pred_time_single).tolist()  # ascending (smallest time first)
        picked_time_exclude = set(exploit_idx)
        for idx in sorted_time:
            if idx not in picked_time_exclude:
                time_idx.append(idx)
            if len(time_idx) >= k_time:
                break

    # Indices for explore (highest uncertainty), excluding exploit and time
    explore_idx = []
    if k_explore > 0:
        sorted_unc = np.argsort(unc_score)[::-1].tolist()
        picked = set(exploit_idx) | set(time_idx)
        for idx in sorted_unc:
            if idx not in picked:
                explore_idx.append(idx)
            if len(explore_idx) >= k_explore:
                break

    final_idx = exploit_idx + time_idx + explore_idx
    if len(final_idx) < k_total:
        # Backfill from remaining EHVI in case of shortages
        combined = np.maximum(ehvi_eff_values, ehvi_real_values)
        remaining = [i for i in np.argsort(combined)[::-1].tolist() if i not in set(final_idx)]
        final_idx.extend(remaining[: k_total - len(final_idx)])

    selected = candidates[final_idx]
    
    # Print selected candidates
    print("Selected candidates (exploit + time + explore):")
    for i, idx in enumerate(final_idx):
        vec = candidates[idx]
        cfg = decode_vec(partition_test, vec)
        tag = "exploit" if idx in exploit_idx else ("time" if idx in time_idx else "explore")
        print(
            f"  {i+1}: [{tag}] EHVI_eff={ehvi_eff_values[idx]:.6g} | EHVI_real={ehvi_real_values[idx]:.6g} | UNC={unc_score[idx]:.6g} | freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
        )
    
    return selected, final_idx, exploit_eff_idx, exploit_real_idx, time_idx, explore_idx


def update_datasets_with_results(
    X_train: np.ndarray,
    X_train_encoded: np.ndarray,
    y_energy_eff: np.ndarray,
    y_time: np.ndarray,
    y_energy_real: np.ndarray,
    selected: np.ndarray,
    batch_results: List[Tuple[float, float]],
    partition_test,
    p2p_power_w: float,
    all_records: List[Tuple],
):
    """
    Process batch evaluation results and update training datasets.
    
    Returns:
        tuple: (X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real, new_time, new_eff_energy, new_avg_energy)
    """
    new_time: List[float] = []
    new_eff_energy: List[float] = []
    new_avg_energy: List[float] = []
    
    for i, (e_j, t_s) in enumerate(batch_results):
        vec = selected[i]
        cfg = decode_vec(partition_test, vec)
        eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
        new_time.append(float(t_s))
        new_eff_energy.append(float(eff_e_j))
        new_avg_energy.append(float(e_j))
        all_records.append((cfg['freq'], cfg['overlap'][0], cfg['overlap'][1], cfg['sm'], cfg['block'], float(t_s), float(e_j), float(eff_e_j)))
        print(f"    -> Energy={e_j:.4f} J, Time={t_s:.6f} s (effective={eff_e_j:.4f} J)")

    # Update datasets
    X_train = np.vstack([X_train, selected])
    X_train_encoded = np.vstack([X_train_encoded, [one_hot_encode(partition_test, x) for x in selected]])
    y_energy_eff = np.append(y_energy_eff, np.array(new_eff_energy, dtype=np.float64))
    y_time = np.append(y_time, np.array(new_time, dtype=np.float64))
    y_energy_real = np.append(y_energy_real, np.array(new_avg_energy, dtype=np.float64))
    
    return X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real, new_time, new_eff_energy, new_avg_energy


def save_pareto_and_results(
    args: argparse.Namespace,
    partition_test,
    X_train: np.ndarray,
    y_energy_eff: np.ndarray,
    y_time: np.ndarray,
    y_energy_real: np.ndarray,
    all_records: List[Tuple],
):
    """
    Compute and save final Pareto fronts and all evaluated results.
    """
    print("\n===============================================")
    print("Final Energy-vs-Time Pareto Fronts")
    print("===============================================")
    
    # Effective-energy Pareto
    neg_Y_eff = -torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)
    pareto_mask_eff = is_non_dominated(neg_Y_eff)
    pareto_indices_eff = torch.where(pareto_mask_eff)[0].cpu().numpy().tolist()
    pareto_results_eff: List[Tuple[Dict[str, int], float, float]] = []
    for idx in pareto_indices_eff:
        cfg = decode_vec(partition_test, X_train[idx])
        e = float(y_energy_eff[idx])
        t = float(y_time[idx])
        pareto_results_eff.append((cfg, e, t))

    # Real-energy Pareto
    neg_Y_real = -torch.tensor(np.column_stack((y_energy_real, y_time)), dtype=torch.double)
    pareto_mask_real = is_non_dominated(neg_Y_real)
    pareto_indices_real = torch.where(pareto_mask_real)[0].cpu().numpy().tolist()
    pareto_results_real: List[Tuple[Dict[str, int], float, float]] = []
    for idx in pareto_indices_real:
        cfg = decode_vec(partition_test, X_train[idx])
        e = float(y_energy_real[idx])
        t = float(y_time[idx])
        pareto_results_real.append((cfg, e, t))

    print(f"Found {len(pareto_results_eff)} effective-energy Pareto points and {len(pareto_results_real)} real-energy Pareto points")
    print("\nEffective-energy Pareto sorted by Energy (ascending):")
    for i, (cfg, e, t) in enumerate(sorted(pareto_results_eff, key=lambda z: z[1])):
        print(
            f"{i+1}. EffEnergy={e:.4f} J | Time={t:.6f} s | freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
        )
    print("\nReal-energy Pareto sorted by Energy (ascending):")
    for i, (cfg, e, t) in enumerate(sorted(pareto_results_real, key=lambda z: z[1])):
        print(
            f"{i+1}. RealEnergy={e:.4f} J | Time={t:.6f} s | freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
        )

    # Save Pareto frontiers to logs directory
    logs_dir = f"logs/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/forward"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save effective-energy Pareto frontier
    csv_eff_path = os.path.join(logs_dir, "results_pareto_frontier_effective.csv")
    with open(csv_eff_path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        for idx in pareto_indices_eff:
            cfg = decode_vec(partition_test, X_train[idx])
            t = float(y_time[idx])
            e_avg = float(y_energy_real[idx]) if idx < len(y_energy_real) else ''
            e_eff = float(y_energy_eff[idx])
            f.write(
                f"{cfg['freq']},{cfg['overlap'][0]},{cfg['overlap'][1]},{cfg['sm']},{cfg['block']},{t},{e_avg},{e_eff}\n"
            )
    print(f"Saved effective-energy Pareto frontier to {csv_eff_path}")

    # Save real-energy Pareto frontier
    csv_real_path = os.path.join(logs_dir, "results_pareto_frontier_real.csv")
    with open(csv_real_path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        for idx in pareto_indices_real:
            cfg = decode_vec(partition_test, X_train[idx])
            t = float(y_time[idx])
            e_avg = float(y_energy_real[idx])
            e_eff = float(y_energy_eff[idx]) if idx < len(y_energy_eff) else ''
            f.write(
                f"{cfg['freq']},{cfg['overlap'][0]},{cfg['overlap'][1]},{cfg['sm']},{cfg['block']},{t},{e_avg},{e_eff}\n"
            )
    print(f"Saved real-energy Pareto frontier to {csv_real_path}")

    # Save all evaluated results
    csv_all_path = os.path.join(logs_dir, "results_all.csv")
    with open(csv_all_path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        for rec in all_records:
            f.write(f"{rec[0]},{rec[1]},{rec[2]},{rec[3]},{rec[4]},{rec[5]},{rec[6]},{rec[7]}\n")
    print(f"Saved all evaluated results to {csv_all_path}")
