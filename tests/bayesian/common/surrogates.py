"""XGBoost surrogate models, ensemble uncertainty, hypervolume, and EHVI."""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch
import xgboost as xgb
from botorch.utils.multi_objective.hypervolume import Hypervolume

from .encoding import one_hot_encode

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "eta": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
}
NUM_BOOST_ROUNDS = 100
ENSEMBLE_SIZE = 5
BOOTSTRAP_FRAC = 0.8
ENSEMBLE_BASE_SEED = 42


def train_xgb_models(X_encoded: np.ndarray, y_energy: np.ndarray, y_time: np.ndarray):
    dtrain_energy = xgb.DMatrix(X_encoded, label=y_energy)
    dtrain_time = xgb.DMatrix(X_encoded, label=y_time)
    energy_model = xgb.train(XGB_PARAMS, dtrain_energy, num_boost_round=NUM_BOOST_ROUNDS)
    time_model = xgb.train(XGB_PARAMS, dtrain_time, num_boost_round=NUM_BOOST_ROUNDS)
    return energy_model, time_model


def train_xgb_energy_only(X_encoded: np.ndarray, y_energy: np.ndarray):
    dtrain_energy = xgb.DMatrix(X_encoded, label=y_energy)
    energy_model = xgb.train(XGB_PARAMS, dtrain_energy, num_boost_round=NUM_BOOST_ROUNDS)
    return energy_model


def train_xgb_ensemble(
    X_encoded: np.ndarray,
    y_energy: np.ndarray,
    y_time: np.ndarray,
    ensemble_size: int = ENSEMBLE_SIZE,
    bootstrap_frac: float = BOOTSTRAP_FRAC,
    base_seed: int = ENSEMBLE_BASE_SEED,
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
        params = {**XGB_PARAMS, "seed": int(base_seed + i)}
        energy_model = xgb.train(params, dtrain_energy, num_boost_round=NUM_BOOST_ROUNDS)
        time_model = xgb.train(params, dtrain_time, num_boost_round=NUM_BOOST_ROUNDS)
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
        return (
            np.zeros(X_encoded.shape[0], dtype=np.float64),
            np.ones(X_encoded.shape[0], dtype=np.float64),
            np.zeros(X_encoded.shape[0], dtype=np.float64),
            np.ones(X_encoded.shape[0], dtype=np.float64),
        )

    energy_preds = []
    time_preds = []
    dtest = xgb.DMatrix(X_encoded)
    for em, tm in ensemble_models:
        energy_preds.append(em.predict(dtest))
        time_preds.append(tm.predict(dtest))
    energy_preds = np.vstack(energy_preds)
    time_preds = np.vstack(time_preds)
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
    """
    candidate_encoded = one_hot_encode(partition_test, candidate_vec).reshape(1, -1)
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
