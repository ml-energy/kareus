#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian optimization for attention fuser overlap-window and communication configs
using real hardware measurements (time and energy) per candidate.

This script reuses the AttentionFuserTest execution path to evaluate a single
configuration by spawning a distributed run and measuring via ZeusMonitor.

Search algorithm:
- Surrogate models: two XGBoost regressors (energy, time)
- Acquisition: Expected Hypervolume Improvement (deterministic proxy)
- Discrete search space: overlap window (categorical), number of SMs (ordinal),
  CUDA block size (categorical)

Note on GPU frequency:
- This script accepts a frequency argument used for bookkeeping. If desired and
  permitted, application clocks can be set via NVML (optional; best-effort).
"""

import os
import sys
import time
import math
import random
import argparse
import itertools
import json
import tempfile
import uuid
import numpy as np
import multiprocessing as mp
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Local imports: make current dir importable, then import the test harness
CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
FUSER_DIR = os.path.join(CUR_DIR, '..', '..', 'fuser')
if FUSER_DIR not in sys.path:
    sys.path.append(FUSER_DIR)

from overlap_test_qkv_ag import AttentionFuserTest  # noqa: E402
from common_config import FuserTestConfig  # noqa: E402

# Third-party utilities used by the evaluation path
from zeus.monitor import ZeusMonitor  # noqa: E402
from kareus.megatron.core.extensions.qkv_fuser2 import QKVPartitionFuser2 as PartitionFuser  # noqa: E402

try:
    import pynvml  # noqa: F401
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import xgboost as xgb
except ImportError as exc:
    raise RuntimeError("xgboost must be installed to run this script.") from exc

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

try:
    import matplotlib.pyplot as plt  # noqa: F401
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False


# -----------------------------
# Search space and encodings
# -----------------------------

# Align overlap windows with AttentionFuserTest.get_overlap_windows()
OVERLAP_WINDOWS: List[Tuple[int, int]] = [
    (-1, -1), (0, 5), (2, 5),
]

# Communication SM counts and CUDA block sizes to consider
SM_VALUES: List[int] = FuserTestConfig.get_comm_sm_values()
BLOCK_VALUES: List[int] = [512, 1024]

# Frequency values are determined at runtime from --gpu_type
FREQ_VALUES: List[int] = []

# Feature indices for encoded vectors [freq_idx, sm_idx, block_idx, overlap_idx]
FREQ_IDX = 0
SM_IDX = 1
BLOCK_IDX = 2
OVERLAP_IDX = 3


def encode_cfg(cfg: Dict[str, int]) -> np.ndarray:
    """
    Encode configuration to an index vector [sm_idx, block_idx, overlap_idx].

    cfg keys:
      - freq: actual GPU core frequency (per FREQ_VALUES)
      - sm: actual SM count (1..20)
      - block: CUDA block size (512 or 1024)
      - overlap: overlap window tuple from OVERLAP_WINDOWS
    """
    freq_idx = FREQ_VALUES.index(cfg["freq"])  # 0..len-1
    sm_idx = SM_VALUES.index(cfg["sm"])  # 0..19
    block_idx = BLOCK_VALUES.index(cfg["block"])  # 0..1
    overlap_idx = OVERLAP_WINDOWS.index(cfg["overlap"])  # 0..len-1
    return np.array([freq_idx, sm_idx, block_idx, overlap_idx], dtype=np.int64)


def one_hot_encode(x: np.ndarray) -> np.ndarray:
    """
    One-hot encode categorical features (block, overlap) and keep SM as numeric.

    x: [sm_idx, block_idx, overlap_idx]
    """
    # Use actual MHz value for frequency instead of the categorical index
    freq_mhz = float(FREQ_VALUES[int(x[FREQ_IDX])])
    numeric = np.array([freq_mhz, x[SM_IDX]], dtype=np.float32)
    # numeric = np.array([x[FREQ_IDX], x[SM_IDX]], dtype=np.float32)
    block_one_hot = np.zeros(len(BLOCK_VALUES), dtype=np.float32)
    block_one_hot[int(x[BLOCK_IDX])] = 1.0
    overlap_one_hot = np.zeros(len(OVERLAP_WINDOWS), dtype=np.float32)
    overlap_one_hot[int(x[OVERLAP_IDX])] = 1.0
    return np.concatenate([numeric, block_one_hot, overlap_one_hot], axis=0)


def decode_vec(x: np.ndarray) -> Dict[str, int]:
    """
    Decode an index vector [sm_idx, block_idx, overlap_idx] back to a config dict.
    """
    freq_idx = int(np.clip(round(float(x[FREQ_IDX])), 0, len(FREQ_VALUES) - 1))
    sm_idx = int(np.clip(round(float(x[SM_IDX])), 0, len(SM_VALUES) - 1))
    block_idx = int(np.clip(round(float(x[BLOCK_IDX])), 0, len(BLOCK_VALUES) - 1))
    overlap_idx = int(np.clip(round(float(x[OVERLAP_IDX])), 0, len(OVERLAP_WINDOWS) - 1))
    return {
        "freq": FREQ_VALUES[freq_idx],
        "sm": SM_VALUES[sm_idx],
        "block": BLOCK_VALUES[block_idx],
        "overlap": OVERLAP_WINDOWS[overlap_idx],
    }


# -----------------------------
# Real evaluation via distributed run
# -----------------------------

def _set_gpu_frequency(target_freq_mhz: int, device_indices: List[int] | None = None) -> None:
    """Attempt to set application clocks via NVML (best effort).

    If device_indices is provided, only set those NVML indices; otherwise set all
    NVML-visible devices.
    """
    if not _NVML_AVAILABLE or target_freq_mhz <= 0:
        return
    pynvml.nvmlInit()
    all_indices = list(range(pynvml.nvmlDeviceGetCount()))
    indices = device_indices if device_indices is not None else all_indices
    for i in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_freq_mhz, target_freq_mhz)
        time.sleep(1)
    pynvml.nvmlShutdown()


def _dist_eval_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    freq_mhz: int,
    overlap_window: Tuple[int, int],
    sm_num: int,
    block_size: int,
    master_port: int,
    shared_results: dict,
    eval_log_path: str,
    selection_flags: Optional[Dict[str, bool]] = None,
    predicted_values: Optional[Dict[str, float]] = None,
) -> None:
    """
    Distributed worker: run a single configuration and measure time/energy.
    Only rank 0 writes results into shared_results dictionary and file.
    """
    # try:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    # Initialize the test runner and build ops
    test = AttentionFuserTest(args, rank=rank, world_size=world_size)

    hidden_states, bias, residual, rotary_pos_emb, attention_mask, allgather_key, allgather_value = (
        test.create_test_tensors()
    )
    operations = test.create_operations(allgather_value)
    comp_ops = operations[:-1]
    comm_op = operations[-1]

    attention_fuser = PartitionFuser(
        ops=comp_ops,
        comm_op_fwd=comm_op,
        fuse_ops=False,
        profile=True,
    )

    # Warmup to determine iteration count
    torch.cuda.synchronize()
    dist.barrier()
    for i in range(10):
        if i == 2:
            time_start = time.time()
        _ = attention_fuser(
            hidden_states=hidden_states,
            bias=bias,
            residual=residual,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            comm_key=allgather_key,
            comm_value=allgather_value,
            comm_overlap_window=overlap_window,
            comm_sm_configs=(sm_num, block_size),
        )
    torch.cuda.synchronize()
    dist.barrier()
    time_end = time.time()
    duration = (time_end - time_start) / 8.0

    if rank == 0:
        iterations = int(max(1, round(8.0 / max(duration, 1e-6))))
        obj_list = [iterations]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0, group=test.cp_group)
    iterations = obj_list[0]

    # Measure in a single window with ZeusMonitor on rank 0
    monitor = ZeusMonitor(gpu_indices=list(range(world_size))) if rank == 0 else None

    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        monitor.begin_window("step")
    for _ in range(iterations):
        _ = attention_fuser(
            hidden_states=hidden_states,
            bias=bias,
            residual=residual,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            comm_key=allgather_key,
            comm_value=allgather_value,
            comm_overlap_window=overlap_window,
            comm_sm_configs=(sm_num, block_size),
        )
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
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
        # BO selection category flags (default False if not provided)
        flags = selection_flags or {}
        record.update({
            "selected_exploit_eff": bool(flags.get("selected_exploit_eff", False)),
            "selected_exploit_real": bool(flags.get("selected_exploit_real", False)),
            "selected_time": bool(flags.get("selected_time", False)),
            "selected_explore": bool(flags.get("selected_explore", False)),
        })
        preds = predicted_values or {}
        # Optional model predictions captured at selection time
        record.update({
            "pred_time_s": preds.get("time_s"),
            "pred_energy_eff_j": preds.get("energy_eff_j"),
            "pred_energy_real_j": preds.get("energy_real_j"),
        })
        
        # Store results in shared dictionary
        shared_results["energy_j"] = avg_energy_j
        shared_results["time_s"] = avg_time_s
        
        # Still write to file for logging
        with open(eval_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    if rank == 0:
        pid = os.getpid()
        print(f"Killing process group {pid}")
        os.system(f'pkill -P {pid}')
    if dist.is_initialized():
        dist.destroy_process_group()


def measure_on_hardware(
    x_vec: np.ndarray,
    args: argparse.Namespace,
    selection_flags: Optional[Dict[str, bool]] = None,
    predicted_values: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Evaluate a candidate configuration on real hardware by spawning a
    distributed run and returning (energy_j, time_s).
    """
    cfg = decode_vec(x_vec)
    overlap_window = cfg["overlap"]
    sm_num = cfg["sm"]
    block_size = cfg["block"]
    freq_mhz = int(cfg["freq"])  # set frequency per candidate

    # Best-effort: set GPU frequency per candidate (parent process) respecting CUDA_VISIBLE_DEVICES
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None and len(visible.strip()) > 0:
        vis_list = [int(x) for x in visible.split(",") if x.strip() != ""]
        target_indices = vis_list
    else:
        target_indices = None  # all NVML-visible
    _set_gpu_frequency(freq_mhz, device_indices=target_indices)

    # Persistent evaluation log file per run
    logs_dir = f"logs/{args.model_name}/cp{args.context_parallel_size}-tp{args.tensor_parallel_size}-bs{args.batch_size}-seq{args.seq_len}"
    os.makedirs(logs_dir, exist_ok=True)
    eval_log_path = os.path.join(logs_dir, "eval_results.jsonl")

    master_port = 9002
    manager = mp.Manager()
    shared_results = manager.dict()

    spawn(
        _dist_eval_worker,
        args=(
            args.world_size,
            args,
            freq_mhz,
            overlap_window,
            sm_num,
            block_size,
            master_port,
            shared_results,
            eval_log_path,
            selection_flags,
            predicted_values,
        ),
        nprocs=args.world_size,
        join=True,
    )

    energy_j = float(shared_results["energy_j"])
    time_s = float(shared_results["time_s"])
    return energy_j, time_s


def measure_from_profile_results(
    x_vec: np.ndarray,
    args: argparse.Namespace,
    logs_base_dir: str = "/workspaces/Kareus/tests/fuser/attention/logs"
) -> Tuple[float, float]:
    """
    Read energy and time measurements from pre-existing profile results in CSV files
    instead of running on hardware.
    """
    cfg = decode_vec(x_vec)
    overlap_start, overlap_end = cfg["overlap"]
    sm_num = cfg["sm"]
    block_size = cfg["block"]
    freq_mhz = int(cfg["freq"])
    
    logs_dir = f"logs/{args.model_name}/cp{args.context_parallel_size}-tp{args.tensor_parallel_size}-bs{args.batch_size}-seq{args.seq_len}"
    csv_file_path = os.path.join(logs_base_dir, logs_dir, str(freq_mhz), "energy_results.csv")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Profile results not found at: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)
    
    matching_rows = df[
        (df['overlap_start'] == overlap_start) &
        (df['overlap_end'] == overlap_end) &
        (df['comm_sm_number'] == sm_num) &
        (df['comm_block_size'] == block_size)
    ]
    
    if matching_rows.empty:
        print(
            f"Configuration not found in {csv_file_path}: "
            f"overlap=({overlap_start},{overlap_end}), sm={sm_num}, block={block_size}"
        )
        return 300.0, 1.0
    
    row = matching_rows.iloc[0]
    
    time_s = float(row["0:time (s)"])
    energy_j = float(row["0:total energy (J)"]) / args.world_size
    
    return energy_j, time_s


# -----------------------------
# Cached initial load helper
# -----------------------------

def try_load_initial_from_cache(
    args: argparse.Namespace,
    p2p_power_w: float,
    n_init: int,
    acq_batch: int,
):
    """
    Try to load cached configurations and measurements from eval_results.jsonl if available.
    Behavior:
      - If cache contains >= n_init unique points, use ALL cached points to seed X_train.
      - If cache contains < n_init, top up by evaluating additional configs to reach n_init.

    Returns tuple:
      (use_cached_initial, X_train, X_train_encoded, init_time, init_eff_energy, init_avg_energy, all_records, skipped_batches)

    Where skipped_batches indicates how many full acquisition batches (of size acq_batch)
    beyond the initial n_init are already covered by cache, so the BO loop can resume from
    the next iteration.
    """
    logs_dir = f"logs/{args.model_name}/cp{args.context_parallel_size}-tp{args.tensor_parallel_size}-bs{args.batch_size}-seq{args.seq_len}"
    eval_log_path = os.path.join(logs_dir, "eval_results.jsonl")

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
            key = (int(r["freq"]), int(r["overlap_start"]), int(r["overlap_end"]), int(r["sm"]), int(r["block"]))
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
                "block": int(r["block"]),
                "overlap": (int(r["overlap_start"]), int(r["overlap_end"]))
            }
            vec = encode_cfg(cfg)
            X_train_list.append(vec)
            t_s = float(r["time_s"])
            e_j = float(r["energy_j"])
            eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
            init_time.append(t_s)
            init_avg_energy.append(e_j)
            init_eff_energy.append(eff_e_j)
            all_records.append((cfg['freq'], cfg['overlap'][0], cfg['overlap'][1], cfg['sm'], cfg['block'], t_s, e_j, eff_e_j))
        # Top-up missing initial points by generating and evaluating additional configs
        missing = int(n_init - total_cached)
        if missing > 0:
            print(f"Cached initial points {total_cached} < n_init {n_init}; generating and evaluating {missing} additional configs...")
            all_configs = generate_all_configurations()
            existing = np.array(X_train_list) if len(X_train_list) > 0 else np.empty((0, 4), dtype=np.int64)
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
                for j, idx in enumerate(picked_indices):
                    vec = remaining[idx]
                    cfg_dec = decode_vec(vec)
                    print(
                        f"  [topup {j+1}/{missing}] freq={cfg_dec['freq']} | sm={cfg_dec['sm']} | block={cfg_dec['block']} | overlap={cfg_dec['overlap']}"
                    )
                    e_j, t_s = measure_on_hardware(vec, args)
                    eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
                    init_time.append(float(t_s))
                    init_eff_energy.append(float(eff_e_j))
                    init_avg_energy.append(float(e_j))
                    all_records.append((cfg_dec['freq'], cfg_dec['overlap'][0], cfg_dec['overlap'][1], cfg_dec['sm'], cfg_dec['block'], float(t_s), float(e_j), float(eff_e_j)))
                    X_train_list.append(vec)
        X_train = np.array(X_train_list)
        X_train_encoded = np.array([one_hot_encode(x) for x in X_train])
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

AUTOTVM_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 10,
    "eta": 0.1,                # learning rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "tree_method": "hist",     # fast on CPU; switch to 'gpu_hist' if you really want GPU
}

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
        normalization_bounds: Tuple of (min_vals, max_vals) for normalization
    
    Returns:
        Expected hypervolume improvement
    """
    candidate_encoded = one_hot_encode(candidate_vec).reshape(1, -1)
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

def generate_all_configurations() -> List[np.ndarray]:
    configs: List[np.ndarray] = []
    for freq_idx in range(len(FREQ_VALUES)):
        for sm_idx in range(len(SM_VALUES)):
            for block_idx in range(len(BLOCK_VALUES)):
                for overlap_idx in range(len(OVERLAP_WINDOWS)):
                    configs.append(
                        np.array([freq_idx, sm_idx, block_idx, overlap_idx], dtype=np.int64)
                    )
    return configs


def is_config_in_dataset(config: np.ndarray, dataset: np.ndarray) -> bool:
    return any(np.array_equal(config, x) for x in dataset)


# -----------------------------
# Visualization helpers
# -----------------------------

def _save_iteration_plots(
    ib: int,
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
    if not _MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available; skipping iteration plots.")
        return

    # Prepare directories (align with forward path under /forward/figures)
    base_logs_dir = f"logs/{args.model_name}/cp{args.context_parallel_size}-tp{args.tensor_parallel_size}-bs{args.batch_size}-seq{args.seq_len}/forward/figures"
    os.makedirs(base_logs_dir, exist_ok=True)

    # Compute previous Pareto fronts (measured before this iteration's new points)
    Y_eff_prev = np.column_stack((prev_energy_eff, prev_time)) if len(prev_time) > 0 else np.empty((0, 2))
    pareto_mask_eff_prev = is_non_dominated(-torch.tensor(Y_eff_prev, dtype=torch.double)).cpu().numpy().astype(bool) if Y_eff_prev.shape[0] > 0 else np.array([], dtype=bool)
    Y_real_prev = np.column_stack((prev_energy_real, prev_time)) if len(prev_time) > 0 else np.empty((0, 2))
    pareto_mask_real_prev = is_non_dominated(-torch.tensor(Y_real_prev, dtype=torch.double)).cpu().numpy().astype(bool) if Y_real_prev.shape[0] > 0 else np.array([], dtype=bool)

    # Convert new lists to arrays
    new_time_arr = np.array(new_time, dtype=float)
    new_eff_arr = np.array(new_eff_energy, dtype=float)
    new_real_arr = np.array(new_real_energy, dtype=float)
    cat_eff = np.array(cat_exploit_eff, dtype=bool)
    cat_real = np.array(cat_exploit_real, dtype=bool)
    cat_time_arr = np.array(cat_time, dtype=bool)
    cat_explore_arr = np.array(cat_explore, dtype=bool)

    # Plot Effective-energy frontier
    try:
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
    except Exception as _exc:
        print(f"Warning: failed to save effective plot for iter {ib+1}: {_exc}")

    # Plot Real-energy frontier
    try:
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
    except Exception as _exc:
        print(f"Warning: failed to save real plot for iter {ib+1}: {_exc}")


# -----------------------------
# Main optimization loop
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default=FuserTestConfig.MODEL_NAME)
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--context_parallel_size", "-cp", type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--gpu_type", type=str, choices=["A40", "A100"], default=FuserTestConfig.GPU_TYPE)

    parser.add_argument("--n_init", type=int, default=FuserTestConfig.BO_DEFAULT_N_INIT)
    parser.add_argument("--batches", type=int, default=FuserTestConfig.BO_DEFAULT_BATCHES)
    parser.add_argument("--acq_batch", type=int, default=FuserTestConfig.BO_DEFAULT_ACQ_BATCH, help="New evaluations per batch")
    parser.add_argument("--use_effective_energy", action="store_true",
                        help="Use effective energy instead of real energy for GBT training (Pareto frontier still uses effective energy)")
    parser.add_argument("--normalize_objectives", action="store_true",
                        help="Normalize energy and time objectives to [0,1] range for balanced hypervolume calculation (default: True)")

    # Exploration / uncertainty options
    parser.add_argument("--explore_fraction", type=float, default=0.25,
                        help="Fraction of each acquisition batch reserved for uncertainty-driven exploration (0..1)")
    parser.add_argument("--ensemble_size", type=int, default=5,
                        help="Size of the XGBoost ensemble used to estimate predictive uncertainty")
    parser.add_argument("--bootstrap_frac", type=float, default=0.8,
                        help="Bootstrap fraction for training each ensemble member")
    parser.add_argument("--uncertainty_metric", type=str, choices=["sum", "max", "energy_std", "time_std"], default="sum",
                        help="How to combine energy/time predictive std into a single uncertainty score")
    parser.add_argument("--time_fraction", type=float, default=0.25,
                        help="Fraction of each acquisition batch reserved for time-optimal candidates (0..1)")

    args = parser.parse_args()

    print("===============================================")
    print("Bayesian Optimization for Attention Fuser (real measurements)")
    print("===============================================")
    print(f"World size: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"GPU Type: {args.gpu_type}")
    print(f"Initial points: {args.n_init}, Batches: {args.batches}, Per-batch evals: {args.acq_batch}")
    print(f"Energy type for GBT training: {'Effective' if args.use_effective_energy else 'Real'}")
    print(f"Objective normalization: {'Enabled' if args.normalize_objectives else 'Disabled'}")
    print(f"Acquisition fractions: explore={args.explore_fraction}, time={args.time_fraction}")

    # Configure frequency values based on GPU type
    global FREQ_VALUES
    if args.gpu_type == "A40":
        FREQ_VALUES = list(map(int, np.arange(1740, 900 - 60, -60)))
        # FREQ_VALUES = [1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000]
    else:  # A100
        FREQ_VALUES = list(map(int, np.arange(1410, 900 - 30, -30)))
    print(f"Frequency search set has {len(FREQ_VALUES)} values (min={min(FREQ_VALUES)}, max={max(FREQ_VALUES)})")

    # p2p power per GPU type (W)
    p2p_power_w = FuserTestConfig.get_p2p_power(args.gpu_type)

    # Build initial design by random sampling from the discrete space
    all_configs = generate_all_configurations()
    total_configs = len(all_configs)
    n_init = min(args.n_init, total_configs)
    
    use_cached_initial, X_train_cached, X_train_encoded_cached, init_time, init_eff_energy, init_avg_energy, all_records, skipped_batches = try_load_initial_from_cache(
        args=args,
        p2p_power_w=p2p_power_w,
        n_init=n_init,
        acq_batch=int(args.acq_batch),
    )
    
    if not use_cached_initial:
        init_indices = random.sample(range(total_configs), n_init)
        X_train = np.array([all_configs[i] for i in init_indices])
        X_train_encoded = np.array([one_hot_encode(x) for x in X_train])

        print(f"Total {len(FREQ_VALUES)} frequency values, {len(SM_VALUES)} SMs, {len(BLOCK_VALUES)} block sizes, {len(OVERLAP_WINDOWS)} overlap values")
        print(f"Total {len(all_configs)} configurations")
        print(f"Generated {X_train.shape[0]} initial configurations")
        print("Evaluating initial configurations on hardware...")

        start_time = time.time()
        for i in range(X_train.shape[0]):
            cfg = decode_vec(X_train[i])
            print(
                f"  [{i+1}/{X_train.shape[0]}] freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
            )
            e_j, t_s = measure_on_hardware(X_train[i], args)
            # e_j, t_s = measure_from_profile_results(X_train[i], args)
            eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
            init_time.append(float(t_s))
            init_eff_energy.append(float(eff_e_j))
            init_avg_energy.append(float(e_j))
            all_records.append((cfg['freq'], cfg['overlap'][0], cfg['overlap'][1], cfg['sm'], cfg['block'], float(t_s), float(e_j), float(eff_e_j)))
            print(f"    -> Energy={e_j:.4f} J, Time={t_s:.6f} s")
        start_time_marker = start_time
    else:
        X_train = X_train_cached  # type: ignore[assignment]
        X_train_encoded = X_train_encoded_cached  # type: ignore[assignment]
        # To keep later timing print meaningful
        start_time_marker = time.time()

    y_energy_eff = np.array(init_eff_energy, dtype=np.float64)  # effective energy
    y_time = np.array(init_time, dtype=np.float64)
    y_energy_real = np.array(init_avg_energy, dtype=np.float64)  # real energy
    
    # Select energy type for training based on parameter
    y_energy_for_training = y_energy_eff if args.use_effective_energy else y_energy_real
    print(f"Using {'effective' if args.use_effective_energy else 'real'} energy for GBT training")
    
    # Pareto frontier always uses effective energy
    Y_torch = torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)

    init_eval_time = time.time() - start_time_marker
    print(f"Initial evaluation completed in {init_eval_time:.2f} s")
    print(
        f"Initial ranges: Energy [{np.min(y_energy_eff):.4f}, {np.max(y_energy_eff):.4f}] J | "
        f"Time [{np.min(y_time):.6f}, {np.max(y_time):.6f}] s"
    )

    # Reference points: a bit worse than worst observed so far (separate for eff and real energy)
    ref_point_eff = np.array([np.max(y_energy_eff) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)
    ref_point_real = np.array([np.max(y_energy_real) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)

    print("\n===============================================")
    print(f"Starting optimization loop ({args.batches} batches, {args.acq_batch} evals/batch)")
    print("===============================================")

    # If using cache, skip fully-covered batches beyond initial n_init
    start_batch_idx = int(skipped_batches) if use_cached_initial else 0
    if start_batch_idx > 0:
        print(f"Resuming from batch {start_batch_idx+1} (skipped {start_batch_idx} full batch(es) from cache)")
    total_start = time.time()
    for ib in range(int(start_batch_idx), int(args.batches)):
        print(f"\n[Batch {ib+1}/{args.batches}] Training surrogate models on {len(X_train)} points...")
        # Train separate models for effective and real energy; share time model
        energy_model_eff, time_model = train_xgb_models(X_train_encoded, y_energy_eff, y_time)
        energy_model_real = train_xgb_energy_only(X_train_encoded, y_energy_real)
        models_eff = (energy_model_eff, time_model)
        models_real = (energy_model_real, time_model)
        # Train ensemble for uncertainty estimates
        ensemble_models = train_xgb_ensemble(
            X_train_encoded,
            y_energy_for_training,
            y_time,
            ensemble_size=args.ensemble_size,
            bootstrap_frac=args.bootstrap_frac,
        )

        current_front_eff = np.column_stack((y_energy_eff, y_time))
        current_front_real = np.column_stack((y_energy_real, y_time))

        # Generate candidate pool (all remaining configs)
        candidates = []
        for cfg_vec in all_configs:
            if not is_config_in_dataset(cfg_vec, X_train):
                candidates.append(cfg_vec)
        if len(candidates) == 0:
            print("No new candidates available. Stopping early.")
            break
        candidates = np.array(candidates)
        # Encode candidates once for vectorized predictions
        cand_encoded = np.array([one_hot_encode(x) for x in candidates])

        # Calculate normalization bounds from current data to balance energy and time scales
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
        
        # Precompute current HV once per batch (optionally using normalization)
        # Precompute HV caches for both fronts (eff and real)
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
                normalization_bounds_real,
                current_hv_cached=current_hv_real_cached,
                pareto_front_norm_cached=pareto_front_real_norm_cached,
                ref_point_norm_cached=ref_point_real_norm_cached,
            )
            ehvi_eff_values.append(ehvi_eff)
            ehvi_real_values.append(ehvi_real)
        ehvi_eff_values = np.array(ehvi_eff_values)
        ehvi_real_values = np.array(ehvi_real_values)

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

        print("Selected candidates (exploit + explore):")
        for i, idx in enumerate(final_idx):
            vec = candidates[idx]
            cfg = decode_vec(vec)
            tag = "exploit" if idx in exploit_idx else ("time" if idx in time_idx else "explore")
            print(
                f"  {i+1}: [{tag}] EHVI_eff={ehvi_eff_values[idx]:.6g} | EHVI_real={ehvi_real_values[idx]:.6g} | UNC={unc_score[idx]:.6g} | freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
            )

        # Evaluate selected candidates on hardware
        print("Evaluating selected candidates on hardware...")
        new_time: List[float] = []
        new_eff_energy: List[float] = []
        new_avg_energy: List[float] = []  # real avg energy
        for i, vec in enumerate(selected):
            cfg = decode_vec(vec)
            print(f"  [{i+1}/{len(selected)}] freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}")
            sel_idx = final_idx[i]
            flags = {
                "selected_exploit_eff": bool(sel_idx in exploit_eff_idx),
                "selected_exploit_real": bool(sel_idx in exploit_real_idx),
                "selected_time": bool(sel_idx in time_idx),
                "selected_explore": bool(sel_idx in explore_idx),
            }
            # Predictions for logging
            cand_enc = one_hot_encode(vec).reshape(1, -1)
            pred_eff_e, pred_time = predict_performance(models_eff, cand_enc)
            pred_real_e, _ = predict_performance(models_real, cand_enc)
            preds = {
                "time_s": float(pred_time[0]),
                "energy_eff_j": float(pred_eff_e[0]),
                "energy_real_j": float(pred_real_e[0]),
            }
            e_j, t_s = measure_on_hardware(vec, args, selection_flags=flags, predicted_values=preds)
            # e_j, t_s = measure_from_profile_results(vec, args)
            eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
            new_time.append(float(t_s))
            new_eff_energy.append(float(eff_e_j))
            new_avg_energy.append(float(e_j))
            all_records.append((cfg['freq'], cfg['overlap'][0], cfg['overlap'][1], cfg['sm'], cfg['block'], float(t_s), float(e_j), float(eff_e_j)))
            print(f"    -> Energy={e_j:.4f} J, Time={t_s:.6f} s (effective={eff_e_j:.4f} J)")

        # Update datasets
        X_train = np.vstack([X_train, selected])
        X_train_encoded = np.vstack([X_train_encoded, [one_hot_encode(x) for x in selected]])
        y_energy_eff = np.append(y_energy_eff, np.array(new_eff_energy, dtype=np.float64))
        y_time = np.append(y_time, np.array(new_time, dtype=np.float64))
        y_energy_real = np.append(y_energy_real, np.array(new_avg_energy, dtype=np.float64))
        
        # Update training energy array based on parameter
        y_energy_for_training = y_energy_eff if args.use_effective_energy else y_energy_real
        
        # Pareto frontier always uses effective energy
        Y_torch = torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)

        # Update reference point (always use effective energy for Pareto)
        ref_point = np.array([np.max(y_energy_eff) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)

        # Pareto count
        neg_Y = -Y_torch
        pareto_mask = is_non_dominated(neg_Y)
        pareto_count = int(torch.sum(pareto_mask).item())

        print(f"  Total evaluations so far: {X_train.shape[0]}")
        print(f"  Current Pareto points count: {pareto_count}")
        print(
            f"  Best observed -> Energy: {np.min(y_energy_eff):.4f} J | Time: {np.min(y_time):.6f} s"
        )

        # Save iteration visualization AFTER evaluation with measured values
        _save_iteration_plots(
            ib=ib,
            args=args,
            prev_energy_eff=y_energy_eff[:-len(new_eff_energy)] if len(new_eff_energy) > 0 else y_energy_eff,
            prev_energy_real=y_energy_real[:-len(new_avg_energy)] if len(new_avg_energy) > 0 else y_energy_real,
            prev_time=y_time[:-len(new_time)] if len(new_time) > 0 else y_time,
            new_time=new_time,
            new_eff_energy=new_eff_energy,
            new_real_energy=new_avg_energy,
            cat_exploit_eff=[(i in exploit_eff_idx) for i in final_idx],
            cat_exploit_real=[(i in exploit_real_idx) for i in final_idx],
            cat_time=[(i in time_idx) for i in final_idx],
            cat_explore=[(i in explore_idx) for i in final_idx],
        )

    total_time = time.time() - total_start
    print(f"\nOptimization completed in {total_time:.2f} s")

    # Final Pareto fronts (effective energy and real energy)
    print("\n===============================================")
    print("Final Energy-vs-Time Pareto Fronts")
    print("===============================================")
    # Effective-energy Pareto
    neg_Y_eff = -torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)
    pareto_mask_eff = is_non_dominated(neg_Y_eff)
    pareto_indices_eff = torch.where(pareto_mask_eff)[0].cpu().numpy().tolist()
    pareto_results_eff: List[Tuple[Dict[str, int], float, float]] = []
    for idx in pareto_indices_eff:
        cfg = decode_vec(X_train[idx])
        e = float(y_energy_eff[idx])
        t = float(y_time[idx])
        pareto_results_eff.append((cfg, e, t))

    # Real-energy Pareto
    neg_Y_real = -torch.tensor(np.column_stack((y_energy_real, y_time)), dtype=torch.double)
    pareto_mask_real = is_non_dominated(neg_Y_real)
    pareto_indices_real = torch.where(pareto_mask_real)[0].cpu().numpy().tolist()
    pareto_results_real: List[Tuple[Dict[str, int], float, float]] = []
    for idx in pareto_indices_real:
        cfg = decode_vec(X_train[idx])
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

    # Save Pareto frontier to logs directory
    logs_dir = f"logs/{args.model_name}/cp{args.context_parallel_size}-tp{args.tensor_parallel_size}-bs{args.batch_size}-seq{args.seq_len}/forward"
    os.makedirs(logs_dir, exist_ok=True)
    # Save effective-energy Pareto frontier
    csv_eff_path = os.path.join(logs_dir, "results_pareto_frontier_effective.csv")
    with open(csv_eff_path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        for idx in pareto_indices_eff:
            cfg = decode_vec(X_train[idx])
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
            cfg = decode_vec(X_train[idx])
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


if __name__ == "__main__":
    main()


