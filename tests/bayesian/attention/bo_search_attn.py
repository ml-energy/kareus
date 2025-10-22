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
import random
import argparse
import json
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
from kareus.megatron.core.extensions.partition_fuser import PartitionFuser  # noqa: E402

# Import utility functions from bo_utils
from bo_utils import (  # noqa: E402
    encode_cfg,
    one_hot_encode,
    decode_vec,
    measure_batch_on_hardware,
    try_load_initial_from_cache,
    train_xgb_models,
    train_xgb_energy_only,
    train_xgb_ensemble,
    predict_ensemble_stats,
    predict_performance,
    calculate_dominated_hypervolume,
    normalize_objectives,
    expected_hypervolume_improvement,
    generate_all_configurations,
    is_config_in_dataset,
    save_iteration_plots,
)

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume


# -----------------------------
# Search space and encodings
# -----------------------------

# Editable
OVERLAP_WINDOWS = [
    (0, 8), (2, 8), (4, 8), (6, 8), # (7, 8),
]
SM_VALUES = list(range(1, 21))

# Frequency values are determined at runtime from --gpu_type
FREQ_VALUES = []

BO_DEFAULT_N_INIT = 96
BO_DEFAULT_BATCHES = 8
BO_DEFAULT_ACQ_BATCH = 32

MASTER_PORT = 9002

# -----------------------------
# Real evaluation via distributed run
# -----------------------------

class PartitionTestConfig:
    """
    Lightweight configuration holder for bo_utils compatibility.
    Does NOT initialize distributed environment - used in parent process.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        logs_dir = f"logs/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/forward"
        os.makedirs(logs_dir, exist_ok=True)
        self.eval_log_path = os.path.join(logs_dir, "eval_results.jsonl")
        self.logs_dir = logs_dir
        self.master_port = MASTER_PORT
        
        self.FREQ_VALUES = FREQ_VALUES
        self.SM_VALUES = SM_VALUES
        self.OVERLAP_WINDOWS = OVERLAP_WINDOWS
        self.BLOCK_SIZE = 1024  # Fixed for attention


class PartitionTestRunner:
    """
    Wraps AttentionFuserTest and PartitionFuser setup and exposes test_config(overlap_window, sm_configs).
    Should ONLY be instantiated after spawn in worker processes.
    """
    def __init__(self, args: argparse.Namespace, rank: int, world_size: int) -> None:
        self.args = args
        self.rank = rank
        self.world_size = world_size

        # Initialize distributed test
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
            comm_op_fwd=allreduce_comm_op,
            fuse_ops=False,
        )

        self.group = self.test.tp_group
        self.FREQ_VALUES = FREQ_VALUES
        self.SM_VALUES = SM_VALUES
        self.OVERLAP_WINDOWS = OVERLAP_WINDOWS

    @property
    def tp_group(self):
        return self.test.tp_group

    def test_config(self, overlap_window, sm_configs):
        return self.attention_fuser(
            hidden_states=self.hidden_states,
            bias=self.bias,
            residual=self.residual,
            rotary_pos_emb=self.rotary_pos_emb,
            attention_mask=self.attention_mask,
            comm_input=self.allreduce_inputs,
            comm_overlap_window=overlap_window,
            comm_sm_configs=sm_configs,
        )


# -----------------------------
# Main optimization loop
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--gpu_type", type=str, choices=["A40", "A100"], default=FuserTestConfig.GPU_TYPE)

    parser.add_argument("--n_init", type=int, default=BO_DEFAULT_N_INIT)
    parser.add_argument("--batches", type=int, default=BO_DEFAULT_BATCHES)
    parser.add_argument("--acq_batch", type=int, default=BO_DEFAULT_ACQ_BATCH, help="New evaluations per batch")
    parser.add_argument("--use_effective_energy", action="store_true",
                        help="Use effective energy instead of real energy for GBT training (Pareto frontier still uses effective energy)")
    parser.add_argument("--normalize_objectives", action="store_true",
                        help="Normalize energy and time objectives to [0,1] range for balanced hypervolume calculation (default: True)")

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
    else:  # A100
        FREQ_VALUES = list(map(int, np.arange(1410, 900 - 30, -30)))
    print(f"Frequency search set has {len(FREQ_VALUES)} values (min={min(FREQ_VALUES)}, max={max(FREQ_VALUES)})")

    # p2p power per GPU type (W)
    p2p_power_w = FuserTestConfig.get_p2p_power(args.gpu_type)

    partition_test = PartitionTestConfig(args)
    all_configs = generate_all_configurations(partition_test)
    total_configs = len(all_configs)
    n_init = min(args.n_init, total_configs)
    
    use_cached_initial, X_train_cached, X_train_encoded_cached, init_time, init_eff_energy, init_avg_energy, all_records, skipped_batches = try_load_initial_from_cache(
        args=args,
        p2p_power_w=p2p_power_w,
        n_init=n_init,
        acq_batch=int(args.acq_batch),
        partition_test=partition_test,
        partition_test_runner_cls=PartitionTestRunner,
    )
    
    if not use_cached_initial:
        init_indices = random.sample(range(total_configs), n_init)
        X_train = np.array([all_configs[i] for i in init_indices])
        X_train_encoded = np.array([one_hot_encode(partition_test, x) for x in X_train])

        print(f"Total {len(FREQ_VALUES)} frequency values, {len(SM_VALUES)} SMs, {len(OVERLAP_WINDOWS)} overlap values")
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
            partition_test_runner_cls=PartitionTestRunner,
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
        start_time_marker = start_time
    else:
        X_train = X_train_cached
        X_train_encoded = X_train_encoded_cached 
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
        cand_encoded = np.array([one_hot_encode(partition_test, x) for x in candidates])

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
            cfg = decode_vec(partition_test, vec)
            tag = "exploit" if idx in exploit_idx else ("time" if idx in time_idx else "explore")
            print(
                f"  {i+1}: [{tag}] EHVI_eff={ehvi_eff_values[idx]:.6g} | EHVI_real={ehvi_real_values[idx]:.6g} | UNC={unc_score[idx]:.6g} | freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
            )

        # Evaluate selected candidates with a single distributed spawn for the whole batch
        print("Evaluating selected candidates on hardware (single spawn per batch)...")
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

    # Save Pareto frontier to logs directory
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


if __name__ == "__main__":
    main()


