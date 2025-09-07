#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian optimization for MLP fuser (backward) overlap-window and communication configs
using real hardware measurements (time and energy) per candidate.

This script mirrors the attention backward optimizer but targets the MLP fuser.
"""

import os
import sys
import time
import random
import argparse
import json
import numpy as np
import multiprocessing as mp
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from typing import Dict, List, Tuple
import pandas as pd

# Local imports: make current dir importable
CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from overlap_test_mlp_backward import MLPFuserTest  # noqa: E402

from zeus.monitor import ZeusMonitor  # noqa: E402
from kareus.megatron.core.extensions.partition_fuser_profile import PartitionFuser  # noqa: E402

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


# -----------------------------
# Search space and encodings
# -----------------------------

OVERLAP_WINDOWS: List[Tuple[int, int]] = [
    (-1, -1),
    (0, 1), (2, 2), (3, 4), (5, 6),
    (0, 2), (2, 4), (3, 6),
    (0, 4), (2, 6),
    (0, 6),
]

SM_VALUES: List[int] = list(range(1, 21))
BLOCK_VALUES: List[int] = [512, 1024]

FREQ_VALUES: List[int] = []

FREQ_IDX = 0
SM_IDX = 1
BLOCK_IDX = 2
OVERLAP_IDX = 3


def encode_cfg(cfg: Dict[str, int]) -> np.ndarray:
    freq_idx = FREQ_VALUES.index(cfg["freq"])  # 0..len-1
    sm_idx = SM_VALUES.index(cfg["sm"])  # 0..19
    block_idx = BLOCK_VALUES.index(cfg["block"])  # 0..1
    overlap_idx = OVERLAP_WINDOWS.index(cfg["overlap"])  # 0..len-1
    return np.array([freq_idx, sm_idx, block_idx, overlap_idx], dtype=np.int64)


def one_hot_encode(x: np.ndarray) -> np.ndarray:
    freq_mhz = float(FREQ_VALUES[int(x[FREQ_IDX])])
    numeric = np.array([freq_mhz, x[SM_IDX]], dtype=np.float32)
    block_one_hot = np.zeros(len(BLOCK_VALUES), dtype=np.float32)
    block_one_hot[int(x[BLOCK_IDX])] = 1.0
    overlap_one_hot = np.zeros(len(OVERLAP_WINDOWS), dtype=np.float32)
    overlap_one_hot[int(x[OVERLAP_IDX])] = 1.0
    return np.concatenate([numeric, block_one_hot, overlap_one_hot], axis=0)


def decode_vec(x: np.ndarray) -> Dict[str, int]:
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
# Real evaluation via distributed run (backward)
# -----------------------------

def _set_gpu_frequency(target_freq_mhz: int, device_indices: List[int] | None = None) -> None:
    if not _NVML_AVAILABLE or target_freq_mhz <= 0:
        return
    pynvml.nvmlInit()
    all_indices = list(range(pynvml.nvmlDeviceGetCount()))
    indices = device_indices if device_indices is not None else all_indices
    for i in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, target_freq_mhz, target_freq_mhz)
    pynvml.nvmlShutdown()
    time.sleep(1)


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
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    test = MLPFuserTest(args, rank=rank, world_size=world_size)

    hidden_states, bias, residual, allreduce_inputs = test.create_test_tensors()
    operations = test.create_operations(allreduce_inputs)
    comp_ops = operations[:-1]
    allreduce_comm_op = operations[-1]

    mlp_fuser = PartitionFuser(
        ops=comp_ops,
        allreduce_comm_op=allreduce_comm_op,
        fuse_ops=False,
    )

    # Prepare gradient tensors for backward pass
    nano_batch_size = test.batch_size // 2
    output_grad = torch.randn(
        test.seq_length, nano_batch_size, test.hidden_size,
        dtype=test.dtype, device=test.device
    )
    residual_grad = torch.randn(
        test.seq_length, nano_batch_size, test.hidden_size,
        dtype=test.dtype, device=test.device
    )
    allreduce_input_grad = torch.randn(
        test.seq_length, nano_batch_size, test.hidden_size,
        dtype=test.dtype, device=test.device
    )

    torch.cuda.synchronize()
    dist.barrier()
    for i in range(10):
        if i == 0:
            output, output_bias, output_residual, allreduce_output = mlp_fuser(
                hidden_states=hidden_states,
                bias=bias,
                residual=residual,
                allreduce_input=allreduce_inputs,
                allreduce_overlap_window=(-1, -1),
                allreduce_sm_configs=(20, 1024),
                allreduce_overlap_window_backward=overlap_window,
                allreduce_sm_configs_backward=(sm_num, block_size),
            )
        if i == 2:
            time_start = time.time()
        torch.autograd.backward(
            tensors=[output, output_residual, allreduce_output],
            grad_tensors=[output_grad, residual_grad, allreduce_input_grad],
            retain_graph=True,
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

    monitor = ZeusMonitor(gpu_indices=list(range(world_size))) if rank == 0 else None

    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        monitor.begin_window("step")
    for _ in range(iterations):
        torch.autograd.backward(
            tensors=[output, output_residual, allreduce_output],
            grad_tensors=[output_grad, residual_grad, allreduce_input_grad],
            retain_graph=True,
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

        shared_results["energy_j"] = avg_energy_j
        shared_results["time_s"] = avg_time_s

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
    cfg = decode_vec(x_vec)
    overlap_window = cfg["overlap"]
    sm_num = cfg["sm"]
    block_size = cfg["block"]
    freq_mhz = int(cfg["freq"])  # set frequency per candidate

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None and len(visible.strip()) > 0:
        vis_list = [int(x) for x in visible.split(",") if x.strip() != ""]
        target_indices = vis_list
    else:
        target_indices = None
    _set_gpu_frequency(freq_mhz, device_indices=target_indices)

    logs_dir = f"logs/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/backward"
    os.makedirs(logs_dir, exist_ok=True)
    eval_log_path = os.path.join(logs_dir, "eval_results.jsonl")

    master_port = 9011
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
    cfg = decode_vec(x_vec)
    overlap_start, overlap_end = cfg["overlap"]
    sm_num = cfg["sm"]
    block_size = cfg["block"]
    freq_mhz = int(cfg["freq"])

    logs_dir = f"tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}"
    csv_file_path = os.path.join(logs_base_dir, logs_dir, str(freq_mhz), "mlp_backward_energy_results.csv")

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


def train_xgb_ensemble(
    X_encoded: np.ndarray,
    y_energy: np.ndarray,
    y_time: np.ndarray,
    ensemble_size: int = 5,
    bootstrap_frac: float = 0.8,
    base_seed: int = 42,
):
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
        energy_model = xgb.train(params, dtrain_energy, num_boost_round=100)
        time_model = xgb.train(params, dtrain_time, num_boost_round=100)
        ensemble.append((energy_model, time_model))
    return ensemble

def predict_performance(models, X_encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    energy_model, time_model = models
    dtest = xgb.DMatrix(X_encoded)
    energy_pred = energy_model.predict(dtest)
    time_pred = time_model.predict(dtest)
    return energy_pred, time_pred


def predict_ensemble_stats(
    ensemble_models: List[Tuple[xgb.Booster, xgb.Booster]],
    X_encoded: np.ndarray,
):
    if len(ensemble_models) == 0:
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
    energy_preds = np.vstack(energy_preds)
    time_preds = np.vstack(time_preds)
    return (
        np.mean(energy_preds, axis=0),
        np.std(energy_preds, axis=0),
        np.mean(time_preds, axis=0),
        np.std(time_preds, axis=0),
    )


def calculate_dominated_hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    Y = -torch.tensor(points, dtype=torch.double)
    ref = -torch.tensor(ref_point, dtype=torch.double)
    hv = Hypervolume(ref_point=ref)
    hv_value = hv.compute(Y)
    return float(hv_value)


def normalize_objectives(data: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    ranges = max_vals - min_vals
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
    candidate_encoded = one_hot_encode(candidate_vec).reshape(1, -1)
    e_pred, t_pred = predict_performance(models, candidate_encoded)
    predicted_point = np.array([[e_pred[0], t_pred[0]]], dtype=np.float64)

    if normalization_bounds is not None:
        min_vals, max_vals = normalization_bounds
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
# Cached initial load helper
# -----------------------------

def try_load_initial_from_cache(
    args: argparse.Namespace,
    p2p_power_w: float,
    n_init: int,
):
    logs_dir = f"logs/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/backward"
    eval_log_path = os.path.join(logs_dir, "eval_results.jsonl")

    init_time: List[float] = []
    init_eff_energy: List[float] = []
    init_avg_energy: List[float] = []
    all_records: List[Tuple[int, int, int, int, int, float, float, float]] = []

    if not os.path.exists(eval_log_path):
        return False, None, None, init_time, init_eff_energy, init_avg_energy, all_records

    try:
        with open(eval_log_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        parsed = [json.loads(ln) for ln in lines]
        seen = set()
        unique: List[dict] = []
        for r in parsed:
            key = (int(r["freq"]), int(r["overlap_start"]), int(r["overlap_end"]), int(r["sm"]), int(r["block"]))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        if len(unique) == 0:
            return False, None, None, init_time, init_eff_energy, init_avg_energy, all_records

        k = min(n_init, len(unique))
        print(f"Found cached measurements for {len(unique)} configs; using {k} for initial points from {eval_log_path}")
        X_train_list: List[np.ndarray] = []
        for i in range(k):
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
        X_train = np.array(X_train_list)
        X_train_encoded = np.array([one_hot_encode(x) for x in X_train])
        return True, X_train, X_train_encoded, init_time, init_eff_energy, init_avg_energy, all_records
    except Exception as exc:
        print(f"Warning: failed to parse cached evals at {eval_log_path}: {exc}. Falling back to hardware evaluation.")
        return False, None, None, init_time, init_eff_energy, init_avg_energy, all_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--gpu_type", type=str, choices=["A40", "A100"], default="A100")

    parser.add_argument("--n_init", type=int, default=128)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--acq_batch", type=int, default=32, help="New evaluations per batch")
    parser.add_argument("--use_effective_energy", action="store_true",
                        help="Use effective energy instead of real energy for GBT training (Pareto frontier still uses effective energy)")
    parser.add_argument("--normalize_objectives", action="store_true",
                        help="Normalize energy and time objectives to [0,1] range for balanced hypervolume calculation (default: True)")

    parser.add_argument("--explore_fraction", type=float, default=0.25)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--bootstrap_frac", type=float, default=0.8)
    parser.add_argument("--uncertainty_metric", type=str, choices=["sum", "max", "energy_std", "time_std"], default="sum")
    parser.add_argument("--time_fraction", type=float, default=0.2)

    args = parser.parse_args()

    print("===============================================")
    print("Bayesian Optimization for MLP Fuser (backward; real measurements)")
    print("===============================================")

    global FREQ_VALUES
    if args.gpu_type == "A40":
        FREQ_VALUES = list(map(int, np.arange(1740, 1000 - 30, -30)))
    else:
        FREQ_VALUES = list(map(int, np.arange(1410, 960 - 15, -15)))
    print(f"Frequency search set has {len(FREQ_VALUES)} values (min={min(FREQ_VALUES)}, max={max(FREQ_VALUES)})")

    if args.gpu_type == "A40":
        p2p_power_w = 90.0
    else:
        p2p_power_w = 70.0

    all_configs = generate_all_configurations()
    total_configs = len(all_configs)
    n_init = min(args.n_init, total_configs)

    use_cached_initial, X_train_cached, X_train_encoded_cached, init_time, init_eff_energy, init_avg_energy, all_records = try_load_initial_from_cache(
        args=args,
        p2p_power_w=p2p_power_w,
        n_init=n_init,
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
        start_time_marker = time.time()

    y_energy_eff = np.array(init_eff_energy, dtype=np.float64)
    y_time = np.array(init_time, dtype=np.float64)
    y_energy_real = np.array(init_avg_energy, dtype=np.float64)

    y_energy_for_training = y_energy_eff if args.use_effective_energy else y_energy_real
    print(f"Using {'effective' if args.use_effective_energy else 'real'} energy for GBT training")

    Y_torch = torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)

    init_eval_time = time.time() - start_time_marker
    print(f"Initial evaluation completed in {init_eval_time:.2f} s")
    print(
        f"Initial ranges: Energy [{np.min(y_energy_eff):.4f}, {np.max(y_energy_eff):.4f}] J | "
        f"Time [{np.min(y_time):.6f}, {np.max(y_time):.6f}] s"
    )

    ref_point_eff = np.array([np.max(y_energy_eff) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)
    ref_point_real = np.array([np.max(y_energy_real) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)

    print("\n===============================================")
    print(f"Starting optimization loop ({args.batches} batches, {args.acq_batch} evals/batch)")
    print("===============================================")

    total_start = time.time()
    for ib in range(int(args.batches)):
        print(f"\n[Batch {ib+1}/{args.batches}] Training surrogate models on {len(X_train)} points...")
        energy_model_eff, time_model = train_xgb_models(X_train_encoded, y_energy_eff, y_time)
        energy_model_real, _ = train_xgb_models(X_train_encoded, y_energy_real, y_time)
        models_eff = (energy_model_eff, time_model)
        models_real = (energy_model_real, time_model)
        ensemble_models = train_xgb_ensemble(
            X_train_encoded,
            y_energy_eff if args.use_effective_energy else y_energy_real,
            y_time,
            ensemble_size=args.ensemble_size,
            bootstrap_frac=args.bootstrap_frac,
        )

        current_front_eff = np.column_stack((y_energy_eff, y_time))
        current_front_real = np.column_stack((y_energy_real, y_time))

        candidates = []
        for cfg_vec in all_configs:
            if not is_config_in_dataset(cfg_vec, X_train):
                candidates.append(cfg_vec)
        if len(candidates) == 0:
            print("No new candidates available. Stopping early.")
            break
        candidates = np.array(candidates)
        cand_encoded = np.array([one_hot_encode(x) for x in candidates])

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

        e_mean, e_std, t_mean, t_std = predict_ensemble_stats(ensemble_models, cand_encoded)

        pred_energy_single_eff, pred_time_single = predict_performance(models_eff, cand_encoded)

        if args.uncertainty_metric == "sum":
            unc_score = e_std + t_std
        elif args.uncertainty_metric == "max":
            unc_score = np.maximum(e_std, t_std)
        elif args.uncertainty_metric == "energy_std":
            unc_score = e_std
        else:
            unc_score = t_std

        k_total = int(args.acq_batch)
        k_time = int(round(args.time_fraction * k_total))
        k_remaining = max(0, k_total - k_time)
        k_explore = int(round(args.explore_fraction * k_remaining))
        k_exploit = max(0, k_remaining - k_explore)

        exploit_idx: List[int] = []
        if k_exploit > 0:
            k_exploit_eff = k_exploit // 2
            k_exploit_real = k_exploit - k_exploit_eff
            top_eff = np.argsort(ehvi_eff_values)[-k_exploit_eff:][::-1].tolist() if k_exploit_eff > 0 else []
            picked = set()
            for idx in top_eff:
                if idx not in picked:
                    exploit_idx.append(idx)
                    picked.add(idx)
            if k_exploit_real > 0:
                top_real = np.argsort(ehvi_real_values)[-k_exploit_real:][::-1].tolist()
                for idx in top_real:
                    if idx not in picked:
                        exploit_idx.append(idx)
                        picked.add(idx)
            if len(exploit_idx) < k_exploit:
                combined = np.maximum(ehvi_eff_values, ehvi_real_values)
                for idx in np.argsort(combined)[::-1].tolist():
                    if idx not in picked:
                        exploit_idx.append(idx)
                        picked.add(idx)
                    if len(exploit_idx) >= k_exploit:
                        break

        time_idx = []
        if k_time > 0:
            sorted_time = np.argsort(pred_time_single).tolist()
            picked_time_exclude = set(exploit_idx)
            for idx in sorted_time:
                if idx not in picked_time_exclude:
                    time_idx.append(idx)
                if len(time_idx) >= k_time:
                    break

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
            combined = np.maximum(ehvi_eff_values, ehvi_real_values)
            remaining = [i for i in np.argsort(combined)[::-1].tolist() if i not in set(final_idx)]
            final_idx.extend(remaining[: k_total - len(final_idx)])

        selected = candidates[final_idx]

        print("Selected candidates (exploit + time + explore):")
        for i, idx in enumerate(final_idx):
            vec = candidates[idx]
            cfg = decode_vec(vec)
            tag = "exploit" if idx in exploit_idx else ("time" if idx in time_idx else "explore")
            print(
                f"  {i+1}: [{tag}] EHVI_eff={ehvi_eff_values[idx]:.6g} | EHVI_real={ehvi_real_values[idx]:.6g} | UNC={unc_score[idx]:.6g} | freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
            )

        print("Evaluating selected candidates on hardware...")
        new_time: List[float] = []
        new_eff_energy: List[float] = []
        new_avg_energy: List[float] = []
        for i, vec in enumerate(selected):
            cfg = decode_vec(vec)
            print(f"  [{i+1}/{len(selected)}] freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}")
            e_j, t_s = measure_on_hardware(vec, args)
            eff_e_j = float(e_j) - float(p2p_power_w) * float(t_s)
            new_time.append(float(t_s))
            new_eff_energy.append(float(eff_e_j))
            new_avg_energy.append(float(e_j))
            all_records.append((cfg['freq'], cfg['overlap'][0], cfg['overlap'][1], cfg['sm'], cfg['block'], float(t_s), float(e_j), float(eff_e_j)))
            print(f"    -> Energy={e_j:.4f} J, Time={t_s:.6f} s (effective={eff_e_j:.4f} J)")

        X_train = np.vstack([X_train, selected])
        X_train_encoded = np.vstack([X_train_encoded, [one_hot_encode(x) for x in selected]])
        y_energy_eff = np.append(y_energy_eff, np.array(new_eff_energy, dtype=np.float64))
        y_time = np.append(y_time, np.array(new_time, dtype=np.float64))
        y_energy_real = np.append(y_energy_real, np.array(new_avg_energy, dtype=np.float64))

        Y_torch = torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)

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

    print("\n===============================================")
    print("Final Energy-vs-Time Pareto Fronts (MLP Backward)")
    print("===============================================")
    neg_Y_eff = -torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)
    pareto_mask_eff = is_non_dominated(neg_Y_eff)
    pareto_indices_eff = torch.where(pareto_mask_eff)[0].cpu().numpy().tolist()
    pareto_results_eff: List[Tuple[Dict[str, int], float, float]] = []
    for idx in pareto_indices_eff:
        cfg = decode_vec(X_train[idx])
        e = float(y_energy_eff[idx])
        t = float(y_time[idx])
        pareto_results_eff.append((cfg, e, t))

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

    logs_dir = f"logs/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/backward"
    os.makedirs(logs_dir, exist_ok=True)
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

    csv_all_path = os.path.join(logs_dir, "results_all.csv")
    with open(csv_all_path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        for rec in all_records:
            f.write(f"{rec[0]},{rec[1]},{rec[2]},{rec[3]},{rec[4]},{rec[5]},{rec[6]},{rec[7]}\n")
    print(f"Saved all evaluated results to {csv_all_path}")


if __name__ == "__main__":
    main()


