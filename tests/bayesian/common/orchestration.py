"""High-level orchestration: initial setup, scoring, selection, dataset updates, saving, visualization."""

from __future__ import annotations

import os
import time
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated

from .encoding import one_hot_encode, decode_vec
from .surrogates import (
    predict_ensemble_stats,
    predict_performance,
    calculate_dominated_hypervolume,
    normalize_objectives,
    expected_hypervolume_improvement,
)
from .hardware import measure_batch_on_hardware, try_load_initial_from_cache


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

        batch_results = measure_batch_on_hardware(
            x_vec_list=list(X_train),
            args=args,
            partition_test=partition_test,
            partition_test_runner_cls=partition_test_runner_cls,
        )

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
    """Calculate normalization bounds for objectives if enabled."""
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
    """Score all candidates using EHVI for both effective and real energy objectives."""
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

    ehvi_eff_values: List[float] = []
    ehvi_real_values: List[float] = []
    for vec in candidates:
        ehvi_eff = expected_hypervolume_improvement(
            vec, current_front_eff, models_eff, ref_point_eff, partition_test,
            normalization_bounds_eff,
            current_hv_cached=current_hv_eff_cached,
            pareto_front_norm_cached=pareto_front_eff_norm_cached,
            ref_point_norm_cached=ref_point_eff_norm_cached,
        )
        ehvi_real = expected_hypervolume_improvement(
            vec, current_front_real, models_real, ref_point_real, partition_test,
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
    """Select candidates for next batch using exploit/explore/time strategy."""
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
    exploit_eff_idx: List[int] = []
    exploit_real_idx: List[int] = []
    if k_exploit > 0:
        k_exploit_eff = int(round(0.4 * k_exploit))
        k_exploit_real = int(k_exploit) - k_exploit_eff

        top_eff = np.argsort(ehvi_eff_values)[-k_exploit_eff:][::-1].tolist() if k_exploit_eff > 0 else []
        picked = set()
        for idx in top_eff:
            if idx not in picked:
                exploit_idx.append(idx)
                exploit_eff_idx.append(idx)
                picked.add(idx)

        if k_exploit_real > 0:
            top_real = np.argsort(ehvi_real_values)[-k_exploit_real:][::-1].tolist()
            for idx in top_real:
                if idx not in picked:
                    exploit_idx.append(idx)
                    exploit_real_idx.append(idx)
                    picked.add(idx)

        if len(exploit_idx) < k_exploit:
            combined = np.maximum(ehvi_eff_values, ehvi_real_values)
            for idx in np.argsort(combined)[::-1].tolist():
                if idx not in picked:
                    exploit_idx.append(idx)
                    if ehvi_eff_values[idx] >= ehvi_real_values[idx]:
                        exploit_eff_idx.append(idx)
                    else:
                        exploit_real_idx.append(idx)
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
    """Process batch evaluation results and update training datasets."""
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
    """Compute and save final Pareto fronts and all evaluated results."""
    print("\n===============================================")
    print("Final Energy-vs-Time Pareto Fronts")
    print("===============================================")

    neg_Y_eff = -torch.tensor(np.column_stack((y_energy_eff, y_time)), dtype=torch.double)
    pareto_mask_eff = is_non_dominated(neg_Y_eff)
    pareto_indices_eff = torch.where(pareto_mask_eff)[0].cpu().numpy().tolist()
    pareto_results_eff: List[Tuple[Dict[str, int], float, float]] = []
    for idx in pareto_indices_eff:
        cfg = decode_vec(partition_test, X_train[idx])
        e = float(y_energy_eff[idx])
        t = float(y_time[idx])
        pareto_results_eff.append((cfg, e, t))

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

    logs_dir = partition_test.logs_dir
    os.makedirs(logs_dir, exist_ok=True)

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

    csv_all_path = os.path.join(logs_dir, "results_all.csv")
    with open(csv_all_path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        for rec in all_records:
            f.write(f"{rec[0]},{rec[1]},{rec[2]},{rec[3]},{rec[4]},{rec[5]},{rec[6]},{rec[7]}\n")
    print(f"Saved all evaluated results to {csv_all_path}")


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
    if Y_eff_prev.shape[0] > 0:
        plt.scatter(Y_eff_prev[:, 1], Y_eff_prev[:, 0], c="#888888", s=20, label="Measured prev")
        if np.any(pareto_mask_eff_prev):
            front = Y_eff_prev[pareto_mask_eff_prev]
            front_sorted = front[np.argsort(front[:, 1])]
            plt.plot(front_sorted[:, 1], front_sorted[:, 0], "-", c="#1f77b4", label="Pareto prev (eff)")
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
