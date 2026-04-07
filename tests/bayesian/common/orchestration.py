"""High-level orchestration: initial setup, scoring, selection, dataset updates, saving, visualization."""

from __future__ import annotations

import os
import json
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated

from .encoding import (
    encode_cfg,
    one_hot_encode,
    decode_vec,
    get_unevaluated_configs,
)
from .surrogates import (
    predict_ensemble_stats,
    predict_performance,
    expected_hypervolume_improvement,
    HVContext,
)
from .hardware import measure_batch_on_hardware


def _process_measurement(
    cfg: Dict, e_j: float, t_s: float, p2p_power_w: float,
) -> Tuple[float, float, float, Tuple[int, int, int, int, int, float, float, float]]:
    """Compute effective energy and build an 8-field record tuple from one measurement."""
    eff_e_j = e_j - p2p_power_w * t_s
    record = (
        cfg["freq"], cfg["overlap"][0], cfg["overlap"][1],
        cfg["sm"], cfg["block"],
        t_s, e_j, eff_e_j,
    )
    return t_s, eff_e_j, e_j, record


def _accumulate_batch_measurements(
    vecs: List[np.ndarray],
    batch_results: List[Tuple[float, float]],
    partition_test,
    p2p_power_w: float,
    times: List[float],
    eff_energies: List[float],
    avg_energies: List[float],
    records: List[Tuple],
) -> None:
    """Decode each vec, compute effective energy, and append to the accumulator lists in-place."""
    for i, (e_j, t_s) in enumerate(batch_results):
        cfg = decode_vec(partition_test, vecs[i])
        t, eff_e, avg_e, rec = _process_measurement(cfg, float(e_j), float(t_s), p2p_power_w)
        times.append(t)
        eff_energies.append(eff_e)
        avg_energies.append(avg_e)
        records.append(rec)


def _build_eval_record(
    cfg: Dict,
    avg_time_s: float,
    avg_energy_j: float,
    selection_flags: Optional[Dict[str, bool]] = None,
    predicted_values: Optional[Dict[str, float]] = None,
) -> Dict:
    """Build a single JSONL-compatible evaluation record from a decoded config."""
    record = {
        "freq": int(cfg["freq"]),
        "overlap_start": int(cfg["overlap"][0]),
        "overlap_end": int(cfg["overlap"][1]),
        "sm": int(cfg["sm"]),
        "block": int(cfg["block"]),
        "time_s": avg_time_s,
        "energy_j": avg_energy_j,
    }
    if selection_flags is not None:
        record.update({
            "selected_dynamic": bool(selection_flags.get("selected_dynamic", False)),
            "selected_real": bool(selection_flags.get("selected_real", False)),
            "selected_time": bool(selection_flags.get("selected_time", False)),
            "selected_uncertainty": bool(selection_flags.get("selected_uncertainty", False)),
        })
    if predicted_values is not None:
        record.update({
            "pred_time_s": predicted_values.get("time_s"),
            "pred_energy_eff_j": predicted_values.get("energy_eff_j"),
            "pred_energy_real_j": predicted_values.get("energy_real_j"),
        })
    return record


def _log_eval_records(records: List[Dict], eval_log_path: str) -> None:
    """Append evaluation records to a JSONL log file."""
    with open(eval_log_path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def log_batch_eval_results(
    x_vec_list: List[np.ndarray],
    batch_results: List[Tuple[float, float]],
    eval_log_path: str,
    partition_test,
    selection_flags_list: Optional[List[Dict[str, bool]]] = None,
    predicted_values_list: Optional[List[Dict[str, float]]] = None,
) -> None:
    """Decode configs, build records, and append them to the eval JSONL log."""
    records: List[Dict] = []
    for i, (e_j, t_s) in enumerate(batch_results):
        cfg = decode_vec(partition_test, x_vec_list[i])
        flags = selection_flags_list[i] if selection_flags_list is not None else None
        preds = predicted_values_list[i] if predicted_values_list is not None else None
        records.append(_build_eval_record(cfg, t_s, e_j, flags, preds))
    _log_eval_records(records, eval_log_path)


def _load_eval_log(path: str) -> List[dict]:
    """Read a JSONL eval log and return deduplicated records.

    Deduplicates by (freq, overlap_start, overlap_end, sm), keeping the
    first occurrence of each configuration.  Returns ``[]`` when the file
    is missing, empty, or unparseable.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        parsed = [json.loads(ln) for ln in lines]

        seen: set = set()
        unique: List[dict] = []
        for r in parsed:
            key = (int(r["freq"]), int(r["overlap_start"]), int(r["overlap_end"]), int(r["sm"]))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
    except Exception as exc:
        print(f"Warning: failed to parse eval log at {path}: {exc}")
        return []


def _records_to_measurements(
    records: List[dict],
    partition_test,
    p2p_power_w: float,
) -> Tuple[
    List[np.ndarray],
    List[float],
    List[float],
    List[float],
    List[Tuple[int, int, int, int, int, float, float, float]],
]:
    """Convert raw JSONL records into encoded vectors and measurement lists.

    Returns:
        (X_list, times, eff_energies, avg_energies, record_tuples)
    """
    X_list: List[np.ndarray] = []
    times: List[float] = []
    eff_energies: List[float] = []
    avg_energies: List[float] = []
    record_tuples: List[Tuple[int, int, int, int, int, float, float, float]] = []

    for r in records:
        cfg = {
            "freq": int(r["freq"]),
            "sm": int(r["sm"]),
            "block": int(r["block"]),
            "overlap": (int(r["overlap_start"]), int(r["overlap_end"])),
        }
        X_list.append(encode_cfg(partition_test, cfg))
        t_s, eff_e, avg_e, rec = _process_measurement(
            cfg, float(r["energy_j"]), float(r["time_s"]), p2p_power_w,
        )
        times.append(t_s)
        eff_energies.append(eff_e)
        avg_energies.append(avg_e)
        record_tuples.append(rec)

    return X_list, times, eff_energies, avg_energies, record_tuples


def _compute_skip_batches(n_cached: int, n_init: int, acq_batch: int) -> int:
    """Return the number of full acquisition batches the cache covers beyond *n_init*."""
    if n_cached <= n_init:
        return 0
    extra = n_cached - n_init
    return extra // max(1, acq_batch)


def build_selection_metadata(
    selected: np.ndarray,
    final_idx: List[int],
    dynamic_idx: List[int],
    real_idx: List[int],
    time_idx: List[int],
    explore_idx: List[int],
    models_eff,
    models_real,
    partition_test,
) -> Tuple[List[Dict[str, bool]], List[Dict[str, float]]]:
    """Build selection flags and surrogate predictions for each selected candidate."""
    sel_flags_list: List[Dict[str, bool]] = []
    sel_preds_list: List[Dict[str, float]] = []
    for i, vec in enumerate(selected):
        sel_idx = final_idx[i]
        flags = {
            "selected_dynamic": bool(sel_idx in dynamic_idx),
            "selected_real": bool(sel_idx in real_idx),
            "selected_time": bool(sel_idx in time_idx),
            "selected_uncertainty": bool(sel_idx in explore_idx),
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
    return sel_flags_list, sel_preds_list


def setup_initial_data(
    args: argparse.Namespace,
    partition_test,
    partition_test_runner_cls,
    p2p_power_w: float,
    all_configs: List[np.ndarray],
    n_init: int,
):
    """Bootstrap the BO training set from cached + fresh hardware evaluation.

    The pipeline has four phases that run unconditionally:

    1. **Load** -- read previously evaluated configs from the JSONL log.
    2. **Convert** -- turn cached records into encoded vectors and measurements.
    3. **Evaluate deficit** -- if fewer than *n_init* points exist, randomly
       sample the shortfall from *all_configs* and measure on hardware.
    4. **Build arrays** -- assemble final numpy training arrays and compute
       the acquisition-batch resume offset.

    Returns:
        (X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real,
         all_records, start_batch_idx)
        where *start_batch_idx* is >0 when the cache already covers one or
        more full acquisition batches.
    """
    acq_batch = int(args.acq_batch)

    # Phase 1 + 2: load cached measurements and convert to measurement lists
    cached_records = _load_eval_log(partition_test.eval_log_path)
    n_cached = len(cached_records)

    if n_cached > 0:
        print(f"Found {n_cached} cached measurements at {partition_test.eval_log_path}")
        X_list, times, eff_energies, avg_energies, records = _records_to_measurements(
            cached_records, partition_test, p2p_power_w,
        )
    else:
        X_list, times, eff_energies, avg_energies, records = [], [], [], [], []

    # Phase 3: evaluate the deficit (same path whether cache had 0 or partial data)
    n_needed = max(0, n_init - len(X_list))
    if n_needed > 0:
        existing = np.array(X_list) if X_list else np.empty((0, 3), dtype=np.int64)
        remaining = get_unevaluated_configs(all_configs, existing)
        n_to_eval = min(n_needed, len(remaining))

        if n_to_eval == 0:
            print("Warning: no remaining configurations available to fill initial set.")
        else:
            sample_indices = random.sample(range(len(remaining)), n_to_eval)
            new_vecs = [remaining[i] for i in sample_indices]

            print(f"Total {len(all_configs)} configurations")
            print(f"Evaluating {n_to_eval} configurations on hardware (cached={n_cached}, target={n_init})...")
            for j, vec in enumerate(new_vecs):
                cfg = decode_vec(partition_test, vec)
                print(
                    f"  [{j+1}/{n_to_eval}] freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
                )

            batch_results = measure_batch_on_hardware(
                x_vec_list=new_vecs,
                args=args,
                partition_test=partition_test,
                partition_test_runner_cls=partition_test_runner_cls,
            )
            log_batch_eval_results(
                new_vecs, batch_results,
                partition_test.eval_log_path, partition_test,
            )

            _accumulate_batch_measurements(
                new_vecs, batch_results, partition_test, p2p_power_w,
                times, eff_energies, avg_energies, records,
            )
            for j, (e_j, t_s) in enumerate(batch_results):
                print(f"  [{j+1}/{n_to_eval}] -> Energy={e_j:.4f} J, Time={t_s:.6f} s")

            X_list.extend(new_vecs)

    # Phase 4: build training arrays and compute resume offset
    X_train = np.array(X_list)
    X_train_encoded = np.array([one_hot_encode(partition_test, x) for x in X_train])

    y_energy_eff = np.array(eff_energies, dtype=np.float64)
    y_time = np.array(times, dtype=np.float64)
    y_energy_real = np.array(avg_energies, dtype=np.float64)

    print(
        f"Initial ranges: Energy [{np.min(y_energy_eff):.4f}, {np.max(y_energy_eff):.4f}] J | "
        f"Time [{np.min(y_time):.6f}, {np.max(y_time):.6f}] s"
    )

    start_batch_idx = _compute_skip_batches(n_cached, n_init, acq_batch)
    if start_batch_idx > 0:
        print(f"Resuming from batch {start_batch_idx+1} (skipped {start_batch_idx} full batch(es) from cache)")

    return X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real, records, start_batch_idx


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
    cand_encoded: np.ndarray,
    models_eff,
    models_real,
    hv_ctx_eff: HVContext,
    hv_ctx_real: HVContext,
):
    """Rank candidates by Expected Hypervolume Improvement (EHVI).

    Each ``HVContext`` already holds the (optionally normalised) Pareto front,
    reference point, and precomputed baseline hypervolume for one energy
    variant.  This function simply scores every encoded candidate against
    both contexts.

    Returns:
        (ehvi_eff, ehvi_real): two arrays of the same length as
        *cand_encoded*.  Higher values indicate greater predicted
        Pareto-front expansion.
    """
    ehvi_eff_values = np.array([
        expected_hypervolume_improvement(enc.reshape(1, -1), models_eff, hv_ctx_eff)
        for enc in cand_encoded
    ])
    ehvi_real_values = np.array([
        expected_hypervolume_improvement(enc.reshape(1, -1), models_real, hv_ctx_real)
        for enc in cand_encoded
    ])
    return ehvi_eff_values, ehvi_real_values


def _pick_top_k(scores: np.ndarray, k: int, exclude: set,
                descending: bool = True) -> List[int]:
    """Return up to *k* indices with the best *scores*, skipping *exclude*."""
    if k <= 0:
        return []
    order = np.argsort(scores)[::-1] if descending else np.argsort(scores)
    picked: List[int] = []
    for idx in order.tolist():
        if idx not in exclude:
            picked.append(idx)
            if len(picked) >= k:
                break
    return picked


def _compute_uncertainty(args: argparse.Namespace,
                         e_std: np.ndarray, t_std: np.ndarray) -> np.ndarray:
    """Combine ensemble energy/time stds into a single uncertainty score."""
    metric = args.uncertainty_metric
    if metric == "sum":
        return e_std + t_std
    if metric == "max":
        return np.maximum(e_std, t_std)
    if metric == "energy_std":
        return e_std
    return t_std


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
    """Assemble the next acquisition batch via a four-way budget split.

    The *acq_batch* budget is divided into four groups selected in order:
      1. **real**: candidates with the highest real-energy EHVI.
      2. **dynamic**: candidates with the highest effective/dynamic-energy EHVI.
      3. **time**: candidates with the lowest predicted latency.
      4. **uncertainty**: candidates with the highest ensemble uncertainty
         (remainder after the first three groups).

    Fractions are controlled by ``args.real_fraction``,
    ``args.dynamic_fraction``, and ``args.time_fraction``; any shortfall
    is back-filled from the combined EHVI ranking.

    Returns:
        (selected, final_idx, dynamic_idx, real_idx, time_idx, explore_idx)
        giving the chosen vectors, their indices into *candidates*, and
        per-category index lists for logging.
    """
    e_mean, e_std, t_mean, t_std = predict_ensemble_stats(ensemble_models, cand_encoded)
    _, pred_time_single = predict_performance(models_eff, cand_encoded)
    unc_score = _compute_uncertainty(args, e_std, t_std)

    k_total = int(args.acq_batch)
    k_real = int(round(args.real_fraction * k_total))
    k_dynamic = int(round(args.dynamic_fraction * k_total))
    k_time = int(round(args.time_fraction * k_total))
    k_uncertainty = max(0, k_total - k_real - k_dynamic - k_time)

    exclude: set = set()
    real_idx = _pick_top_k(ehvi_real_values, k_real, exclude)
    exclude.update(real_idx)
    dynamic_idx = _pick_top_k(ehvi_eff_values, k_dynamic, exclude)
    exclude.update(dynamic_idx)
    time_idx = _pick_top_k(pred_time_single, k_time, exclude, descending=False)
    exclude.update(time_idx)
    explore_idx = _pick_top_k(unc_score, k_uncertainty, exclude)
    exclude.update(explore_idx)

    final_idx = real_idx + dynamic_idx + time_idx + explore_idx
    if len(final_idx) < k_total:
        combined = np.maximum(ehvi_eff_values, ehvi_real_values)
        final_idx.extend(_pick_top_k(combined, k_total - len(final_idx), exclude))

    selected = candidates[final_idx]

    real_set = set(real_idx)
    dynamic_set = set(dynamic_idx)
    time_set = set(time_idx)
    print("Selected candidates (real + dynamic + time + uncertainty):")
    for i, idx in enumerate(final_idx):
        cfg = decode_vec(partition_test, candidates[idx])
        if idx in real_set:
            tag = "real"
        elif idx in dynamic_set:
            tag = "dynamic"
        elif idx in time_set:
            tag = "time"
        else:
            tag = "uncertainty"
        print(
            f"  {i+1}: [{tag}] EHVI_eff={ehvi_eff_values[idx]:.6g} | EHVI_real={ehvi_real_values[idx]:.6g} | UNC={unc_score[idx]:.6g} | freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
        )

    return selected, final_idx, dynamic_idx, real_idx, time_idx, explore_idx


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

    _accumulate_batch_measurements(
        list(selected), batch_results, partition_test, p2p_power_w,
        new_time, new_eff_energy, new_avg_energy, all_records,
    )
    for e_j, t_s in batch_results:
        eff_e_j = e_j - p2p_power_w * t_s
        print(f"    -> Energy={e_j:.4f} J, Time={t_s:.6f} s (effective={eff_e_j:.4f} J)")

    X_train = np.vstack([X_train, selected])
    X_train_encoded = np.vstack([X_train_encoded, [one_hot_encode(partition_test, x) for x in selected]])
    y_energy_eff = np.append(y_energy_eff, np.array(new_eff_energy, dtype=np.float64))
    y_time = np.append(y_time, np.array(new_time, dtype=np.float64))
    y_energy_real = np.append(y_energy_real, np.array(new_avg_energy, dtype=np.float64))

    return X_train, X_train_encoded, y_energy_eff, y_time, y_energy_real, new_time, new_eff_energy, new_avg_energy


def _extract_pareto(
    y_energy: np.ndarray,
    y_time: np.ndarray,
    X_train: np.ndarray,
    partition_test,
) -> Tuple[List[int], List[Tuple[Dict, float, float]]]:
    """Return Pareto-optimal indices and decoded (cfg, energy, time) triples."""
    neg_Y = -torch.tensor(np.column_stack((y_energy, y_time)), dtype=torch.double)
    indices = torch.where(is_non_dominated(neg_Y))[0].cpu().numpy().tolist()
    results = [
        (decode_vec(partition_test, X_train[i]), float(y_energy[i]), float(y_time[i]))
        for i in indices
    ]
    return indices, results


def _print_pareto(label: str, energy_tag: str, results: List[Tuple[Dict, float, float]]) -> None:
    """Print a Pareto front sorted by energy ascending."""
    print(f"\n{label} Pareto sorted by Energy (ascending):")
    for i, (cfg, e, t) in enumerate(sorted(results, key=lambda z: z[1])):
        print(
            f"{i+1}. {energy_tag}={e:.4f} J | Time={t:.6f} s | "
            f"freq={cfg['freq']} | sm={cfg['sm']} | block={cfg['block']} | overlap={cfg['overlap']}"
        )


def _write_pareto_csv(
    path: str,
    pareto_indices: List[int],
    X_train: np.ndarray,
    y_time: np.ndarray,
    y_energy_real: np.ndarray,
    y_energy_eff: np.ndarray,
    partition_test,
) -> None:
    """Write a Pareto frontier CSV with both energy columns."""
    with open(path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        for idx in pareto_indices:
            cfg = decode_vec(partition_test, X_train[idx])
            f.write(
                f"{cfg['freq']},{cfg['overlap'][0]},{cfg['overlap'][1]},"
                f"{cfg['sm']},{cfg['block']},"
                f"{float(y_time[idx])},{float(y_energy_real[idx])},{float(y_energy_eff[idx])}\n"
            )


def save_pareto_and_results(
    args: argparse.Namespace,
    partition_test,
    X_train: np.ndarray,
    y_energy_eff: np.ndarray,
    y_time: np.ndarray,
    y_energy_real: np.ndarray,
    all_records: List[Tuple],
):
    """Compute final Pareto fronts for both energy variants and save all results to CSV."""
    print("\n===============================================")
    print("Final Energy-vs-Time Pareto Fronts")
    print("===============================================")

    idx_eff, results_eff = _extract_pareto(y_energy_eff, y_time, X_train, partition_test)
    idx_real, results_real = _extract_pareto(y_energy_real, y_time, X_train, partition_test)

    print(f"Found {len(results_eff)} effective-energy Pareto points and {len(results_real)} real-energy Pareto points")
    _print_pareto("Effective-energy", "EffEnergy", results_eff)
    _print_pareto("Real-energy", "RealEnergy", results_real)

    logs_dir = partition_test.logs_dir
    os.makedirs(logs_dir, exist_ok=True)

    csv_eff_path = os.path.join(logs_dir, "results_pareto_frontier_effective.csv")
    _write_pareto_csv(csv_eff_path, idx_eff, X_train, y_time, y_energy_real, y_energy_eff, partition_test)
    print(f"Saved effective-energy Pareto frontier to {csv_eff_path}")

    csv_real_path = os.path.join(logs_dir, "results_pareto_frontier_real.csv")
    _write_pareto_csv(csv_real_path, idx_real, X_train, y_time, y_energy_real, y_energy_eff, partition_test)
    print(f"Saved real-energy Pareto frontier to {csv_real_path}")

    csv_all_path = os.path.join(logs_dir, "results_all.csv")
    with open(csv_all_path, "w") as f:
        f.write("frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,effect_energy_J\n")
        for rec in all_records:
            f.write(f"{rec[0]},{rec[1]},{rec[2]},{rec[3]},{rec[4]},{rec[5]},{rec[6]},{rec[7]}\n")
    print(f"Saved all evaluated results to {csv_all_path}")


def _plot_pareto_frontier(
    Y_prev: np.ndarray,
    pareto_mask: np.ndarray,
    new_time: np.ndarray,
    new_energy: np.ndarray,
    cat_eff: np.ndarray,
    cat_real: np.ndarray,
    cat_time: np.ndarray,
    cat_explore: np.ndarray,
    pareto_color: str,
    pareto_label: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    """Draw a single time-vs-energy Pareto plot with category-coded new points."""
    plt.figure(figsize=(7, 5))
    if Y_prev.shape[0] > 0:
        plt.scatter(Y_prev[:, 1], Y_prev[:, 0], c="#888888", s=20, label="Measured prev")
        if np.any(pareto_mask):
            front = Y_prev[pareto_mask]
            front_sorted = front[np.argsort(front[:, 1])]
            plt.plot(front_sorted[:, 1], front_sorted[:, 0], "-", c=pareto_color, label=pareto_label)
    if new_time.size > 0:
        if np.any(cat_eff):
            plt.scatter(new_time[cat_eff], new_energy[cat_eff], marker="x", c="#d62728", s=50, label="Dynamic")
        if np.any(cat_real):
            plt.scatter(new_time[cat_real], new_energy[cat_real], marker="+", c="#1f77b4", s=60, label="Real")
        if np.any(cat_time):
            plt.scatter(new_time[cat_time], new_energy[cat_time], marker="o", facecolors="none", edgecolors="#2ca02c", s=60, label="Time")
        if np.any(cat_explore):
            plt.scatter(new_time[cat_explore], new_energy[cat_explore], marker="s", c="#9467bd", s=40, label="Uncertainty")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Return a boolean mask of non-dominated rows (minimising both objectives)."""
    if Y.shape[0] == 0:
        return np.array([], dtype=bool)
    return is_non_dominated(-torch.tensor(Y, dtype=torch.double)).cpu().numpy().astype(bool)


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
    cat_dynamic: List[bool],
    cat_real: List[bool],
    cat_time: List[bool],
    cat_uncertainty: List[bool],
) -> None:
    """Save per-iteration Pareto frontier plots for effective and real energy."""
    base_logs_dir = partition_test.logs_dir + "/figures"
    os.makedirs(base_logs_dir, exist_ok=True)

    Y_eff_prev = np.column_stack((prev_energy_eff, prev_time)) if len(prev_time) > 0 else np.empty((0, 2))
    Y_real_prev = np.column_stack((prev_energy_real, prev_time)) if len(prev_time) > 0 else np.empty((0, 2))

    new_time_arr = np.array(new_time, dtype=float)
    cat_eff = np.array(cat_dynamic, dtype=bool)
    cat_real_arr = np.array(cat_real, dtype=bool)
    cat_time_arr = np.array(cat_time, dtype=bool)
    cat_explore_arr = np.array(cat_uncertainty, dtype=bool)

    _plot_pareto_frontier(
        Y_prev=Y_eff_prev, pareto_mask=pareto_mask(Y_eff_prev),
        new_time=new_time_arr, new_energy=np.array(new_eff_energy, dtype=float),
        cat_eff=cat_eff, cat_real=cat_real_arr, cat_time=cat_time_arr, cat_explore=cat_explore_arr,
        pareto_color="#1f77b4", pareto_label="Pareto prev (eff)",
        ylabel="Effective energy (J)",
        title=f"Iter {ib+1}: Time vs Effective energy (measured)",
        save_path=os.path.join(base_logs_dir, f"iter_{ib+1:02d}_effective.png"),
    )
    _plot_pareto_frontier(
        Y_prev=Y_real_prev, pareto_mask=pareto_mask(Y_real_prev),
        new_time=new_time_arr, new_energy=np.array(new_real_energy, dtype=float),
        cat_eff=cat_eff, cat_real=cat_real_arr, cat_time=cat_time_arr, cat_explore=cat_explore_arr,
        pareto_color="#ff7f0e", pareto_label="Pareto prev (real)",
        ylabel="Real energy (J)",
        title=f"Iter {ib+1}: Time vs Real energy (measured)",
        save_path=os.path.join(base_logs_dir, f"iter_{ib+1:02d}_real.png"),
    )
