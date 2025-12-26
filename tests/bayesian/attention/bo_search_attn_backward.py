#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bayesian optimization for attention fuser backward overlap-window and communication configs
using real hardware measurements (time and energy) per candidate.

Mirrors the forward-path optimizer but evaluates the backward pass by spawning
distributed runs and measuring via ZeusMonitor.
"""

import os
import sys
import time
import random
import argparse
import csv
import numpy as np
import torch
from typing import List, Dict, Tuple

CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
FUSER_DIR = os.path.join(CUR_DIR, '..', '..', 'fuser')
if FUSER_DIR not in sys.path:
    sys.path.append(FUSER_DIR)
BO_UTILS_DIR = os.path.join(CUR_DIR, '..')
if BO_UTILS_DIR not in sys.path:
    sys.path.append(BO_UTILS_DIR)

from overlap_test_attn import AttentionFuserTest  # noqa: E402
from common_config import FuserTestConfig  # noqa: E402
from kareus.megatron.core.extensions.fusers.partition_fuser import PartitionFuser  # noqa: E402

from bo_utils import (  # noqa: E402
    one_hot_encode,
    decode_vec,
    train_xgb_models,
    train_xgb_energy_only,
    train_xgb_ensemble,
    predict_performance,
    predict_ensemble_stats,
    generate_all_configurations,
    is_config_in_dataset,
    calculate_dominated_hypervolume,
    normalize_objectives,
    expected_hypervolume_improvement,
    setup_initial_data,
    compute_normalization_bounds,
    score_candidates_with_ehvi,
    select_acquisition_batch,
    update_datasets_with_results,
    save_pareto_and_results,
    measure_batch_on_hardware,
)

from botorch.utils.multi_objective.pareto import is_non_dominated


# -----------------------------
# Search space
# -----------------------------

# Editable
OVERLAP_WINDOWS = [
    (0, 8), (2, 8), (3, 8), (5, 8), # (7, 8),
]
SM_VALUES = list(range(3, 31, 3))

# Frequency values are determined at runtime from --gpu_type
FREQ_VALUES = []

BO_DEFAULT_N_INIT = 96
BO_DEFAULT_BATCHES = 4
BO_DEFAULT_ACQ_BATCH = 32

MASTER_PORT = 9003


# -----------------------------
# Backward configuration and runner
# -----------------------------

class PartitionTestConfig:
    """
    Lightweight configuration holder for bo_utils compatibility (backward version).
    Used only in parent process.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        logs_dir = f"logs/{args.model_name}/cp{args.context_parallel_size}-tp{args.tensor_parallel_size}-bs{args.batch_size}-seq{args.seq_len}/backward"
        os.makedirs(logs_dir, exist_ok=True)
        self.eval_log_path = os.path.join(logs_dir, "eval_results_bwd.jsonl")
        self.logs_dir = logs_dir
        self.master_port = MASTER_PORT

        self.FREQ_VALUES = FREQ_VALUES
        self.SM_VALUES = SM_VALUES
        self.OVERLAP_WINDOWS = OVERLAP_WINDOWS
        self.BLOCK_SIZE = 1024  # Fixed


class PartitionTestRunner:
    """
    Initializes tensors and a fuser capable of backward evaluation.
    test_config(overlap_window, sm_configs) performs one backward step.
    """
    def __init__(self, args: argparse.Namespace, rank: int, world_size: int) -> None:
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.test = AttentionFuserTest(args, rank=rank, world_size=world_size)
        (
            self.hidden_states,
            self.bias,
            self.residual,
            self.rotary_pos_emb,
            self.attention_mask,
            self.allreduce_inputs,
        ) = self.test.create_test_tensors()

        operations = self.test.create_operations(self.allreduce_inputs)
        comp_ops = operations[:-1]
        allreduce_comm_op = operations[-1]

        self.attention_fuser = PartitionFuser(
            ops=comp_ops,
            comm_op_bwd=allreduce_comm_op,
            fuse_ops=False,
        )

        nano_batch_size = self.test.batch_size // 2
        self.output_grad = torch.randn(
            self.test.seq_length, nano_batch_size, self.test.hidden_size,
            dtype=self.test.dtype, device=self.test.device
        )
        self.residual_grad = torch.randn(
            self.test.seq_length, nano_batch_size, self.test.hidden_size,
            dtype=self.test.dtype, device=self.test.device
        )
        self.allreduce_input_grad = torch.randn(
            self.test.seq_length, nano_batch_size, self.test.hidden_size,
            dtype=self.test.dtype, device=self.test.device
        )

        self.output = None
        self.output_residual = None
        self.allreduce_output = None

        self.group = self.test.tp_group
        self.FREQ_VALUES = FREQ_VALUES
        self.SM_VALUES = SM_VALUES
        self.OVERLAP_WINDOWS = OVERLAP_WINDOWS

    @property
    def tp_group(self):
        return self.test.tp_group

    def test_config(self, overlap_window, sm_configs):
        if self.output is None:
            self.output, self.output_bias, self.output_residual, self.allreduce_output = self.attention_fuser(
                hidden_states=self.hidden_states,
                bias=self.bias,
                residual=self.residual,
                rotary_pos_emb=self.rotary_pos_emb,
                attention_mask=self.attention_mask,
                comm_input=self.allreduce_inputs,
                comm_overlap_window_backward=overlap_window,
                comm_sm_configs_backward=sm_configs,
            )
        # torch.autograd.backward(
        #     tensors=[self.output, self.output_residual, self.allreduce_output],
        #     grad_tensors=[self.output_grad, self.residual_grad, self.allreduce_input_grad],
        #     retain_graph=True,
        # )
        _ = torch.autograd.grad(
            outputs=[self.output, self.output_residual, self.allreduce_output],
            inputs=[self.hidden_states, self.residual, self.allreduce_inputs],
            grad_outputs=[self.output_grad, self.residual_grad, self.allreduce_input_grad],
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )

    def clean(self):
        self.output = None
        self.output_bias = None
        self.output_residual = None
        self.allreduce_output = None


# -----------------------------
# Main optimization loop
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default=FuserTestConfig.MODEL_NAME)
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--context_parallel_size", "-cp", type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--gpu_type", type=str, choices=["A40", "A100"], default=FuserTestConfig.GPU_TYPE)

    parser.add_argument("--n_init", type=int, default=BO_DEFAULT_N_INIT)
    parser.add_argument("--batches", type=int, default=BO_DEFAULT_BATCHES)
    parser.add_argument("--acq_batch", type=int, default=BO_DEFAULT_ACQ_BATCH, help="New evaluations per batch")
    parser.add_argument("--use_effective_energy", action="store_true",
                        help="Use effective energy instead of real energy for GBT training (Pareto frontier still uses effective energy)")
    parser.add_argument("--normalize_objectives", action="store_true",
                        help="Normalize energy and time objectives to [0,1] range for balanced hypervolume calculation")

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
    print("Bayesian Optimization for Attention Fuser (backward; real measurements)")
    print("===============================================")
    print(f"World size: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"GPU Type: {args.gpu_type}")
    print(f"Initial points: {args.n_init}, Batches: {args.batches}, Per-batch evals: {args.acq_batch}")
    print(f"Energy type for GBT training: {'Effective' if args.use_effective_energy else 'Real'}")
    print(f"Objective normalization: {'Enabled' if args.normalize_objectives else 'Disabled'}")
    print(f"Acquisition fractions: explore={args.explore_fraction}, time={args.time_fraction}")

    global FREQ_VALUES
    if args.gpu_type == "A40":
        FREQ_VALUES = list(map(int, np.arange(1740, 900 - 60, -60)))
    else:
        FREQ_VALUES = list(map(int, np.arange(1410, 900 - 30, -30)))
    print(f"Frequency search set has {len(FREQ_VALUES)} values (min={min(FREQ_VALUES)}, max={max(FREQ_VALUES)})")

    p2p_power_w = FuserTestConfig.get_p2p_power(args.gpu_type)

    partition_test = PartitionTestConfig(args)
    all_configs = generate_all_configurations(partition_test)
    total_configs = len(all_configs)
    n_init = min(args.n_init, total_configs)

    print(f"Total {len(FREQ_VALUES)} frequency values, {len(SM_VALUES)} SMs, {len(OVERLAP_WINDOWS)} overlap values")
    initial_start = time.time()

    # Setup initial data (load from cache or evaluate fresh) using backward runner
    X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real, all_records, start_batch_idx = setup_initial_data(
        args=args,
        partition_test=partition_test,
        partition_test_runner_cls=PartitionTestRunner,
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

    # CSV logging for per-batch timings
    timing_csv_path = os.path.join(partition_test.logs_dir, "bo_timing_bwd.csv")
    if not os.path.exists(timing_csv_path):
        with open(timing_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_idx", "train_time_s", "select_time_s", "eval_time_s"])

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

        candidates = []
        for cfg_vec in all_configs:
            if not is_config_in_dataset(cfg_vec, X_train):
                candidates.append(cfg_vec)
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
        eval_start = time.time()
        
        print("Evaluating selected candidates on hardware (backward)...")
        sel_flags_list: List[Dict[str, bool]] = []
        sel_preds_list: List[Dict[str, float]] = []
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

        batch_results = measure_batch_on_hardware(
            x_vec_list=list(selected),
            args=args,
            partition_test=partition_test,
            partition_test_runner_cls=PartitionTestRunner,
            selection_flags_list=sel_flags_list,
            predicted_values_list=sel_preds_list,
        )

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

        eval_time_s = time.time() - eval_start
        with open(timing_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ib + 1, f"{train_time_s:.6f}", f"{select_time_s:.6f}", f"{eval_time_s:.6f}"])

        print(f"  Total evaluations so far: {X_train.shape[0]}")
        print(f"  Current Pareto points count: {pareto_count}")
        print(
            f"  Best observed -> Energy: {np.min(y_energy_eff):.4f} J | Time: {np.min(y_time):.6f} s"
        )

        from bo_utils import save_iteration_plots
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


if __name__ == "__main__":
    main()

