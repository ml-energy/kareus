"""SearchSpace, BOSearchConfig, PartitionTestConfig, build_argparser, run_bo_search."""

from __future__ import annotations

import os
import sys
import csv
import time
import dataclasses
import argparse
from typing import List, Tuple, Callable

import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

# Ensure fuser test config is importable
_FUSER_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'fuser')
if _FUSER_DIR not in sys.path:
    sys.path.append(_FUSER_DIR)
from common_config import FuserTestConfig  # noqa: E402

from .encoding import BLOCK_SIZE, one_hot_encode, generate_all_configurations, get_unevaluated_configs
from .surrogates import (
    train_xgb_models,
    train_xgb_energy_only,
    train_xgb_ensemble,
    predict_performance,
)
from .orchestration import (
    setup_initial_data,
    compute_normalization_bounds,
    score_candidates_with_ehvi,
    select_acquisition_batch,
    update_datasets_with_results,
    save_pareto_and_results,
    save_iteration_plots,
)
from .hardware import measure_batch_on_hardware


@dataclasses.dataclass
class SearchSpace:
    """Per-partition search space constants."""
    overlap_windows: List[Tuple[int, int]]
    sm_values: List[int]
    n_init: int = 96
    batches: int = 4
    acq_batch: int = 32
    master_port: int = 9002
    explore_fraction: float = 0.2
    time_fraction: float = 0.2


@dataclasses.dataclass
class BOSearchConfig:
    """Per-file output/logging configuration."""
    banner: str                           # e.g. "Attention Fuser (forward)"
    logs_dir_fn: Callable                 # (args) -> str
    eval_log_filename: str = "eval_results.jsonl"
    world_size_default: str = "tp"        # "tp" or "cp"
    timing_csv: str = "fwd"              # "fwd" (3 cols) or "bwd" (4 cols, +eval_time_s)


class PartitionTestConfig:
    """
    Unified lightweight configuration holder for bo_utils compatibility.
    Used only in parent process — does NOT initialize distributed environment.
    """
    def __init__(self, args: argparse.Namespace, search_space: SearchSpace,
                 bo_config: BOSearchConfig, freq_values: List[int]):
        self.args = args
        logs_dir = bo_config.logs_dir_fn(args)
        os.makedirs(logs_dir, exist_ok=True)
        self.eval_log_path = os.path.join(logs_dir, bo_config.eval_log_filename)
        self.logs_dir = logs_dir
        self.master_port = search_space.master_port

        self.FREQ_VALUES = freq_values
        self.SM_VALUES = search_space.sm_values
        self.OVERLAP_WINDOWS = search_space.overlap_windows
        self.BLOCK_SIZE = BLOCK_SIZE


def build_argparser(search_space: SearchSpace, bo_config: BOSearchConfig) -> argparse.ArgumentParser:
    """Build the full argparse parser with per-partition defaults."""
    parser = argparse.ArgumentParser()

    ws_default = (FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE
                  if bo_config.world_size_default == "tp"
                  else FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)

    parser.add_argument("--model_name", "-m", type=str, default=FuserTestConfig.MODEL_NAME)
    parser.add_argument("--world_size", "-w", type=int, default=ws_default)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int,
                        default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--context_parallel_size", "-cp", type=int,
                        default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--gpu_type", type=str, choices=["A40", "A100"],
                        default=FuserTestConfig.GPU_TYPE)

    parser.add_argument("--n_init", type=int, default=search_space.n_init)
    parser.add_argument("--batches", type=int, default=search_space.batches)
    parser.add_argument("--acq_batch", type=int, default=search_space.acq_batch,
                        help="New evaluations per batch")
    parser.add_argument("--use_effective_energy", action="store_true",
                        help="Use effective energy instead of real energy for GBT training")
    parser.add_argument("--normalize_objectives", action="store_true",
                        help="Normalize energy and time objectives to [0,1] range")

    parser.add_argument("--explore_fraction", type=float, default=search_space.explore_fraction,
                        help="Fraction of each acquisition batch reserved for exploration (0..1)")
    parser.add_argument("--ensemble_size", type=int, default=5,
                        help="Size of the XGBoost ensemble for predictive uncertainty")
    parser.add_argument("--bootstrap_frac", type=float, default=0.8,
                        help="Bootstrap fraction for training each ensemble member")
    parser.add_argument("--uncertainty_metric", type=str,
                        choices=["sum", "max", "energy_std", "time_std"], default="sum",
                        help="How to combine energy/time std into uncertainty score")
    parser.add_argument("--time_fraction", type=float, default=search_space.time_fraction,
                        help="Fraction of each acquisition batch reserved for time-optimal candidates (0..1)")

    return parser


def run_bo_search(
    search_space: SearchSpace,
    bo_config: BOSearchConfig,
    runner_cls: type,
) -> None:
    """
    Unified Bayesian optimization loop.

    Each bo_search_*.py defines its own SearchSpace, BOSearchConfig, and
    PartitionTestRunner, then calls this function from __main__.
    """
    args = build_argparser(search_space, bo_config).parse_args()

    # Attach search-space values to args so runners can read them
    args.sm_values = search_space.sm_values
    args.overlap_windows = search_space.overlap_windows

    print("===============================================")
    print(f"Bayesian Optimization for {bo_config.banner} (real measurements)")
    print("===============================================")
    print(f"World size: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"GPU Type: {args.gpu_type}")
    print(f"Initial points: {args.n_init}, Batches: {args.batches}, Per-batch evals: {args.acq_batch}")
    print(f"Energy type for GBT training: {'Effective' if args.use_effective_energy else 'Real'}")
    print(f"Objective normalization: {'Enabled' if args.normalize_objectives else 'Disabled'}")
    print(f"Acquisition fractions: explore={args.explore_fraction}, time={args.time_fraction}")

    # Compute frequency values based on GPU type
    if args.gpu_type == "A40":
        freq_values = list(map(int, np.arange(1740, 900 - 60, -60)))
    else:  # A100
        freq_values = list(map(int, np.arange(1410, 900 - 30, -30)))
    args.freq_values = freq_values
    print(f"Frequency search set has {len(freq_values)} values (min={min(freq_values)}, max={max(freq_values)})")

    p2p_power_w = FuserTestConfig.get_p2p_power(args.gpu_type)

    partition_test = PartitionTestConfig(args, search_space, bo_config, freq_values)
    all_configs = generate_all_configurations(partition_test)
    total_configs = len(all_configs)
    n_init = min(args.n_init, total_configs)

    print(f"Total {len(freq_values)} frequency values, {len(search_space.sm_values)} SMs, "
          f"{len(search_space.overlap_windows)} overlap values")

    # Setup initial data (load from cache or evaluate fresh)
    initial_start = time.time()
    X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real, all_records, start_batch_idx = setup_initial_data(
        args=args,
        partition_test=partition_test,
        partition_test_runner_cls=runner_cls,
        p2p_power_w=p2p_power_w,
        all_configs=all_configs,
        n_init=n_init,
    )
    initial_time_s = time.time() - initial_start
    print(f"Initial evaluation completed in {initial_time_s:.2f} s")

    y_energy_for_training = y_energy_eff if args.use_effective_energy else y_energy_real
    print(f"Using {'effective' if args.use_effective_energy else 'real'} energy for GBT training")

    ref_point_eff = np.array([np.max(y_energy_eff) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)
    ref_point_real = np.array([np.max(y_energy_real) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)

    print("\n===============================================")
    print(f"Starting optimization loop ({args.batches} batches, {args.acq_batch} evals/batch)")
    print("===============================================")

    # Timing CSV setup
    csv_name = f"bo_timing_{bo_config.timing_csv}.csv"
    timing_csv_path = os.path.join(partition_test.logs_dir, csv_name)
    if not os.path.exists(timing_csv_path):
        with open(timing_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["batch_idx", "train_time_s", "select_time_s"]
            if bo_config.timing_csv == "bwd":
                header.append("eval_time_s")
            writer.writerow(header)

    total_start = time.time()
    for ib in range(int(start_batch_idx), int(args.batches)):
        print(f"\n[Batch {ib+1}/{args.batches}] Training surrogate models on {len(X_train)} points...")
        train_start = time.time()
        energy_model_eff, time_model = train_xgb_models(X_train_encoded, y_energy_eff, y_time)
        energy_model_real = train_xgb_energy_only(X_train_encoded, y_energy_real)
        models_eff = (energy_model_eff, time_model)
        models_real = (energy_model_real, time_model)

        ensemble_models = train_xgb_ensemble(
            X_train_encoded,
            y_energy_for_training,
            y_time,
            ensemble_size=args.ensemble_size,
            bootstrap_frac=args.bootstrap_frac,
        )
        train_time_s = time.time() - train_start
        select_start = time.time()

        candidates = get_unevaluated_configs(all_configs, X_train)
        if len(candidates) == 0:
            print("No new candidates available. Stopping early.")
            break
        candidates = np.array(candidates)
        cand_encoded = np.array([one_hot_encode(partition_test, x) for x in candidates])

        normalization_bounds_eff, normalization_bounds_real = compute_normalization_bounds(
            args, y_energy_eff, y_energy_real, y_time
        )

        current_front_eff = np.column_stack((y_energy_eff, y_time))
        current_front_real = np.column_stack((y_energy_real, y_time))
        ehvi_eff_values, ehvi_real_values = score_candidates_with_ehvi(
            candidates, cand_encoded, current_front_eff, current_front_real,
            models_eff, models_real, ref_point_eff, ref_point_real,
            partition_test, normalization_bounds_eff, normalization_bounds_real,
        )

        selected, final_idx, exploit_eff_idx, exploit_real_idx, time_idx, explore_idx = select_acquisition_batch(
            candidates, cand_encoded, ehvi_eff_values, ehvi_real_values,
            ensemble_models, models_eff, args, partition_test,
        )
        select_time_s = time.time() - select_start

        # Evaluate selected candidates on hardware
        print(f"Evaluating selected candidates on hardware ({bo_config.banner})...")
        sel_flags_list: List[dict] = []
        sel_preds_list: List[dict] = []
        for i, vec in enumerate(selected):
            sel_idx = final_idx[i]
            flags = {
                "selected_exploit_eff": bool(sel_idx in exploit_eff_idx),
                "selected_exploit_real": bool(sel_idx in exploit_real_idx),
                "selected_time": bool(sel_idx in time_idx),
                "selected_explore": bool(sel_idx in explore_idx),
            }
            cand_enc = one_hot_encode(partition_test, vec).reshape(1, -1)
            pred_eff_e, pred_time = predict_performance(models_eff, cand_enc)
            pred_real_e, _ = predict_performance(models_real, cand_enc)
            preds = {
                "time_s": float(pred_time[0]),
                "energy_eff_j": float(pred_eff_e[0]),
                "energy_real_j": float(pred_real_e[0]),
            }
            sel_flags_list.append(flags)
            sel_preds_list.append(preds)

        eval_start = time.time()
        batch_results = measure_batch_on_hardware(
            x_vec_list=list(selected),
            args=args,
            partition_test=partition_test,
            partition_test_runner_cls=runner_cls,
            selection_flags_list=sel_flags_list,
            predicted_values_list=sel_preds_list,
        )
        eval_time_s = time.time() - eval_start

        # Write timing CSV row
        with open(timing_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [ib + 1, f"{train_time_s:.6f}", f"{select_time_s:.6f}"]
            if bo_config.timing_csv == "bwd":
                row.append(f"{eval_time_s:.6f}")
            writer.writerow(row)

        X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real, new_time, new_eff_energy, new_avg_energy = update_datasets_with_results(
            X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real,
            selected, batch_results, partition_test, p2p_power_w, all_records,
        )

        y_energy_for_training = y_energy_eff if args.use_effective_energy else y_energy_real

        ref_point_eff = np.array([np.max(y_energy_eff) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)
        ref_point_real = np.array([np.max(y_energy_real) * 1.1, np.max(y_time) * 1.1], dtype=np.float64)

        Y_current = torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)
        neg_Y = -Y_current
        pareto_mask = is_non_dominated(neg_Y)
        pareto_count = int(torch.sum(pareto_mask).item())

        print(f"  Total evaluations so far: {X_train.shape[0]}")
        print(f"  Current Pareto points count: {pareto_count}")
        print(
            f"  Best observed -> Energy: {np.min(y_energy_eff):.4f} J | Time: {np.min(y_time):.6f} s"
        )

        save_iteration_plots(
            ib=ib,
            partition_test=partition_test,
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

    save_pareto_and_results(
        args, partition_test, X_train, y_energy_eff, y_time, y_energy_real, all_records
    )
