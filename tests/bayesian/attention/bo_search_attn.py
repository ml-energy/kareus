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
from typing import Dict, List, Tuple
import pandas as pd

# Local imports: make current dir importable, then import the test harness
CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from overlap_test_attn import AttentionFuserTest  # noqa: E402

# Third-party utilities used by the evaluation path
from zeus.monitor import ZeusMonitor  # noqa: E402
from kareus.megatron.core.extensions.partition_fuser import PartitionFuser  # noqa: E402

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


# -----------------------------
# Search space and encodings
# -----------------------------

# Align overlap windows with AttentionFuserTest.get_overlap_windows()
OVERLAP_WINDOWS: List[Tuple[int, int]] = [
    (-1, -1),
    (0, 1), (2, 3), (4, 5), (6, 6), (7, 8),
    (0, 3), (2, 5), (4, 6), (6, 8),
    (0, 5), (2, 6), (4, 8),
    (0, 6), (2, 8),
    (0, 8),
]

# Communication SM counts and CUDA block sizes to consider
SM_VALUES: List[int] = list(range(1, 21))
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
    numeric = np.array([x[FREQ_IDX], x[SM_IDX]], dtype=np.float32)
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
    pynvml.nvmlShutdown()


def _dist_eval_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    freq_mhz: int,
    overlap_window: Tuple[int, int],
    sm_num: int,
    block_size: int,
    eval_log_path: str,
    master_port: int,
    shared_results: dict,
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

    hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs = (
        test.create_test_tensors()
    )
    operations = test.create_operations(allreduce_inputs)
    comp_ops = operations[:7]
    allreduce_comm_op = operations[7]

    attention_fuser = PartitionFuser(
        ops=comp_ops,
        allreduce_comm_op=allreduce_comm_op,
        fuse_ops=False,
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
            allreduce_input=allreduce_inputs,
            allreduce_overlap_window=overlap_window,
            allreduce_sm_configs=(sm_num, block_size),
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
    dist.broadcast_object_list(obj_list, src=0, group=test.tp_group)
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
            allreduce_input=allreduce_inputs,
            allreduce_overlap_window=overlap_window,
            allreduce_sm_configs=(sm_num, block_size),
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
    logs_dir = f"logs/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}"
    os.makedirs(logs_dir, exist_ok=True)
    eval_log_path = os.path.join(logs_dir, "eval_results.jsonl")

    master_port = random.randint(12000, 65000)
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
            eval_log_path,
            master_port,
            shared_results,
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
    
    logs_dir = f"tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}"
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
    energy_model = xgb.train(params, dtrain_energy, num_boost_round=100)
    time_model = xgb.train(params, dtrain_time, num_boost_round=100)
    return energy_model, time_model


def predict_performance(models, X_encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    energy_model, time_model = models
    dtest = xgb.DMatrix(X_encoded)
    energy_pred = energy_model.predict(dtest)
    time_pred = time_model.predict(dtest)
    return energy_pred, time_pred


def calculate_dominated_hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    neg_points = -torch.tensor(points, dtype=torch.double)
    neg_ref = -torch.tensor(ref_point, dtype=torch.double)
    pareto_mask = is_non_dominated(neg_points)
    pareto_points = neg_points[pareto_mask]
    if len(pareto_points) == 0:
        return 0.0
    sorted_indices = torch.argsort(pareto_points[:, 0])
    sorted_points = pareto_points[sorted_indices]
    hv = 0.0
    prev_x = neg_ref[0]
    for point in sorted_points:
        width = point[0] - prev_x
        height = point[1] - neg_ref[1]
        hv += width * height
        prev_x = point[0]
    return float(hv.item())


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
        pareto_front_norm = normalize_objectives(pareto_front, min_vals, max_vals)
        predicted_point_norm = normalize_objectives(predicted_point, min_vals, max_vals)
        ref_point_norm = normalize_objectives(ref_point.reshape(1, -1), min_vals, max_vals).flatten()
        
        current_hv = calculate_dominated_hypervolume(pareto_front_norm, ref_point_norm)
        new_front_norm = np.vstack([pareto_front_norm, predicted_point_norm])
        new_hv = calculate_dominated_hypervolume(new_front_norm, ref_point_norm)
    else:
        current_hv = calculate_dominated_hypervolume(pareto_front, ref_point)
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
# Main optimization loop
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--gpu_type", type=str, choices=["A40", "A100"], default="A40")

    parser.add_argument("--n_init", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--acq_batch", type=int, default=4, help="New evaluations per batch")
    parser.add_argument("--use_effective_energy", action="store_true", default=True,
                        help="Use effective energy instead of real energy for GBT training (Pareto frontier still uses effective energy)")
    parser.add_argument("--normalize_objectives", action="store_true", default=True,
                        help="Normalize energy and time objectives to [0,1] range for balanced hypervolume calculation (default: True)")

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

    # Configure frequency values based on GPU type
    global FREQ_VALUES
    if args.gpu_type == "A40":
        FREQ_VALUES = list(map(int, np.arange(1740, 900 - 15, -15)))
        # FREQ_VALUES = [1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000]
    else:  # A100
        FREQ_VALUES = list(map(int, np.arange(1410, 900 - 15, -15)))
    print(f"Frequency search set has {len(FREQ_VALUES)} values (min={min(FREQ_VALUES)}, max={max(FREQ_VALUES)})")

    # p2p power per GPU type (W)
    if args.gpu_type == "A40":
        p2p_power_w = 90.0
    else:
        p2p_power_w = 70.0

    # Build initial design by random sampling from the discrete space
    all_configs = generate_all_configurations()
    total_configs = len(all_configs)
    n_init = min(args.n_init, total_configs)
    init_indices = random.sample(range(total_configs), n_init)
    X_train = np.array([all_configs[i] for i in init_indices])
    X_train_encoded = np.array([one_hot_encode(x) for x in X_train])

    print(f"Total {len(FREQ_VALUES)} frequency values, {len(SM_VALUES)} SMs, {len(BLOCK_VALUES)} block sizes, {len(OVERLAP_WINDOWS)} overlap values")
    print(f"Total {len(all_configs)} configurations")
    print(f"Generated {X_train.shape[0]} initial configurations")
    print("Evaluating initial configurations on hardware...")

    start_time = time.time()
    init_time: List[float] = []
    init_eff_energy: List[float] = []
    init_avg_energy: List[float] = []  # real average energy over TP ranks
    all_records: List[Tuple[int, int, int, int, int, float, float, float]] = []
    for i in range(X_train.shape[0]):
        cfg = decode_vec(X_train[i])
        print(
            f"  [{i+1}/{X_train.shape[0]}] freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
        )
        e_j, t_s = measure_on_hardware(X_train[i], args)
        # e_j, t_s = measure_from_profile_results(X_train[i], args)
        # Adjust energy for Pareto optimization: effective_energy = real_energy - p2p_power * time
        eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
        init_time.append(float(t_s))
        init_eff_energy.append(float(eff_e_j))
        init_avg_energy.append(float(e_j))
        all_records.append((cfg['freq'], cfg['overlap'][0], cfg['overlap'][1], cfg['sm'], cfg['block'], float(t_s), float(e_j), float(eff_e_j)))
        print(f"    -> Energy={e_j:.4f} J, Time={t_s:.6f} s")

    y_energy_eff = np.array(init_eff_energy, dtype=np.float64)  # effective energy
    y_time = np.array(init_time, dtype=np.float64)
    y_energy_real = np.array(init_avg_energy, dtype=np.float64)  # real energy
    
    # Select energy type for training based on parameter
    y_energy_for_training = y_energy_eff if args.use_effective_energy else y_energy_real
    print(f"Using {'effective' if args.use_effective_energy else 'real'} energy for GBT training")
    
    # Pareto frontier always uses effective energy
    Y_torch = torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)

    init_eval_time = time.time() - start_time
    print(f"Initial evaluation completed in {init_eval_time:.2f} s")
    print(
        f"Initial ranges: Energy [{np.min(y_energy_eff):.4f}, {np.max(y_energy_eff):.4f}] J | "
        f"Time [{np.min(y_time):.6f}, {np.max(y_time):.6f}] s"
    )

    # Reference point: a bit worse than worst observed so far (always use effective energy for Pareto)
    ref_point = np.array([np.max(y_energy_eff) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)

    print("\n===============================================")
    print(f"Starting optimization loop ({args.batches} batches, {args.acq_batch} evals/batch)")
    print("===============================================")

    total_start = time.time()
    for ib in range(int(args.batches)):
        print(f"\n[Batch {ib+1}/{args.batches}] Training surrogate models on {len(X_train)} points...")
        models = train_xgb_models(X_train_encoded, y_energy_for_training, y_time)

        current_front = np.column_stack((y_energy_eff, y_time))

        # Generate candidate pool (all remaining configs)
        candidates = []
        for cfg_vec in all_configs:
            if not is_config_in_dataset(cfg_vec, X_train):
                candidates.append(cfg_vec)
        if len(candidates) == 0:
            print("No new candidates available. Stopping early.")
            break
        candidates = np.array(candidates)

        # Calculate normalization bounds from current data to balance energy and time scales
        normalization_bounds = None
        if args.normalize_objectives:
            min_vals = np.array([np.min(y_energy_eff), np.min(y_time)])
            max_vals = np.array([np.max(y_energy_eff), np.max(y_time)])
            normalization_bounds = (min_vals, max_vals)
            print(f"  Normalization bounds - Energy: [{min_vals[0]:.4f}, {max_vals[0]:.4f}], Time: [{min_vals[1]:.6f}, {max_vals[1]:.6f}]")
        else:
            print("  Using raw objectives without normalization")
        
        # Score via EHVI and select top-K
        ehvi_values: List[float] = []
        for vec in candidates:
            ehvi = expected_hypervolume_improvement(vec, current_front, models, ref_point, normalization_bounds)
            ehvi_values.append(ehvi)
        top_idx = np.argsort(ehvi_values)[-args.acq_batch:]
        selected = candidates[top_idx]

        print("Selected candidates:")
        for i, vec in enumerate(selected):
            cfg = decode_vec(vec)
            print(f"  {i+1}: freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}")

        # Evaluate selected candidates on hardware
        print("Evaluating selected candidates on hardware...")
        new_time: List[float] = []
        new_eff_energy: List[float] = []
        new_avg_energy: List[float] = []  # real avg energy
        for i, vec in enumerate(selected):
            cfg = decode_vec(vec)
            print(f"  [{i+1}/{len(selected)}] freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}")
            e_j, t_s = measure_on_hardware(vec, args)
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

    total_time = time.time() - total_start
    print(f"\nOptimization completed in {total_time:.2f} s")

    # Final Pareto front
    print("\n===============================================")
    print("Final Energy-vs-Time Pareto Front")
    print("===============================================")
    neg_Y = -torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)
    pareto_mask = is_non_dominated(neg_Y)
    pareto_indices = torch.where(pareto_mask)[0].cpu().numpy().tolist()

    pareto_results: List[Tuple[Dict[str, int], float, float]] = []
    for idx in pareto_indices:
        cfg = decode_vec(X_train[idx])
        e = float(y_energy_eff[idx])
        t = float(y_time[idx])
        pareto_results.append((cfg, e, t))

    print(f"Found {len(pareto_results)} Pareto-optimal points")
    print("\nPareto front sorted by Energy (ascending):")
    for i, (cfg, e, t) in enumerate(sorted(pareto_results, key=lambda z: z[1])):
        print(
            f"{i+1}. Energy={e:.4f} J | Time={t:.6f} s | freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
        )
    print("\nPareto front sorted by Time (ascending):")
    for i, (cfg, e, t) in enumerate(sorted(pareto_results, key=lambda z: z[2])):
        print(
            f"{i+1}. Time={t:.6f} s | Energy={e:.4f} J | freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
        )

    # Save Pareto frontier to logs directory
    logs_dir = f"logs/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}"
    os.makedirs(logs_dir, exist_ok=True)
    csv_path = os.path.join(logs_dir, "results_pareto_frontier.csv")
    with open(csv_path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        # Note: y_energy_eff holds effective energy values used for Pareto ranking.
        for idx in pareto_indices:
            cfg = decode_vec(X_train[idx])
            e_eff = float(y_energy_eff[idx])
            t = float(y_time[idx])
            e_avg = float(y_energy_real[idx]) if idx < len(y_energy_real) else ''
            f.write(
                f"{cfg['freq']},{cfg['overlap'][0]},{cfg['overlap'][1]},{cfg['sm']},{cfg['block']},{t},{e_avg},{e_eff}\n"
            )
    print(f"Saved Pareto frontier to {csv_path}")

    # Save all evaluated results
    csv_all_path = os.path.join(logs_dir, "results_all.csv")
    with open(csv_all_path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        for rec in all_records:
            f.write(f"{rec[0]},{rec[1]},{rec[2]},{rec[3]},{rec[4]},{rec[5]},{rec[6]},{rec[7]}\n")
    print(f"Saved all evaluated results to {csv_all_path}")


if __name__ == "__main__":
    main()


